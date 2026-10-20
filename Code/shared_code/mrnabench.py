"""mRNABench: full transcripts, their regions, and the gate that decides whether to use them.

Research plan Priorities 3 and 4 (Gantt 42). The stability benchmark BioLangFusion uses is
coding sequence only, so it cannot say whether the untranslated regions carry anything.
mRNABench publishes mature transcripts with per-nucleotide tracks, which is what a
full-transcript test needs, and the plan is explicit that the datasets must pass a gate
before being treated as one:

    Verify sequences include 5'/3' UTRs, CDS coordinates, labels, species and non-leaking
    train/test groups. If transcript regions are absent, do not treat it as a full-transcript
    test.

``gate`` is that check, run on the data as published rather than on its description.

The tracks. Each row carries ``sequence`` and two binary tracks of the same length: ``cds``
marks the first nucleotide of every codon, and ``splice`` marks the 3' end of every exon. So
the coding region is not given as coordinates and is recovered here: the first and last
marked positions bound it, and the marks three apart are what proves the frame. Everything
in this module derives from that one convention, and ``cds_bounds`` raises when a row does
not obey it rather than returning a region that is quietly out of frame.

Labels are kept apart, as the plan insists: half-life, mean ribosome load and translation
efficiency are different measurements, each with its own dataset, column and split, and
none of them is relabelled as another.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from . import matrix_cache
from .embeddings import sequences_fingerprint
from .encoders import ENCODERS, protein_input
from .sequences import dna_to_rna, rna_to_dna, translate_cds

HF = "https://huggingface.co/datasets/morrislab"


@dataclass(frozen=True)
class MRNADataset:
    """One mRNABench dataset: where it comes from, what it measures, how it is split.

    ``targets`` are the dataset's own target columns, one prediction task each. ``split``
    is the splitting rule the benchmark uses: "homology" where gene groups exist, "random"
    for the synthetic libraries, whose rows are not genes at all.
    """

    key: str
    label: str
    file: str
    species: str
    targets: tuple
    split: str
    measures: str
    source: str


# The catalog entries the research plan names, with the files the benchmark publishes.
# Taken from mrna_bench 1.3.2 (datasets/*.py and datasets/dataset_catalog.py), so the target
# columns and split rules are the benchmark's own rather than a reading of its paper.
DATASETS = {
    d.key: d
    for d in [
        MRNADataset("rnahl-human", "RNA half-life, human (Saluki)",
                    "rnahl-saluki-human/resolve/main/rnahl-saluki-human.parquet", "human",
                    ("target",), "homology",
                    "transcript half-life, first principal component over 39 human datasets",
                    "Agarwal and Kelley 2022, Genome Biology 23:245; redistributed by mRNABench"),
        MRNADataset("rnahl-mouse", "RNA half-life, mouse (Saluki)",
                    "rnahl-saluki-mouse/resolve/main/rnahl-saluki-mouse.parquet", "mouse",
                    ("target",), "homology",
                    "transcript half-life, first principal component over 27 mouse datasets",
                    "Agarwal and Kelley 2022; redistributed by mRNABench"),
        MRNADataset("translation-efficiency-human", "Translation efficiency, human",
                    "translation-efficiency-human/resolve/main/te_human.parquet", "human",
                    ("target",), "homology",
                    "translation efficiency from ribosome profiling, one transcript per gene",
                    "mRNABench, from the RiboNN release (Nat Biotechnol 2025)"),
        MRNADataset("translation-efficiency-mouse", "Translation efficiency, mouse",
                    "translation-efficiency-mouse/resolve/main/te_mouse.parquet", "mouse",
                    ("target",), "homology",
                    "translation efficiency from ribosome profiling, one transcript per gene",
                    "mRNABench, from the RiboNN release"),
        MRNADataset("mrl-sugimoto", "Mean ribosome load, isoform resolved (Sugimoto)",
                    "mrl-sugimoto/resolve/main/mrl-sugimoto.parquet", "human",
                    ("target",), "homology",
                    "mean ribosome load per transcript isoform, NOT half-life",
                    "Sugimoto et al. 2022; redistributed by mRNABench"),
        MRNADataset("mrl-hl-lbkwk", "MRL and half-life, synthetic mRNA (LBKWK)",
                    "mrl-hl-lbkwk/resolve/main/mrl-hl-lbkwk.parquet", "synthetic",
                    ("target_in_cell_half_life", "target_ribosome_load"), "random",
                    "two separate targets on synthetic mRNAs: in-cell half-life and ribosome load",
                    "Leppek et al. 2022; redistributed by mRNABench"),
    ]
}

# The half-life datasets of Priority 3 and the translation-efficiency datasets of Priority 4,
# named apart so a notebook cannot quietly pool them.
HALF_LIFE = ("rnahl-human", "rnahl-mouse")
TRANSLATION_EFFICIENCY = ("translation-efficiency-human", "translation-efficiency-mouse")
RIBOSOME_LOAD = ("mrl-sugimoto",)
SYNTHETIC_PAIR = ("mrl-hl-lbkwk",)

# Where downloaded parquet files land. Already ignored by Git, like the other caches.
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / ".data-cache" / "mrnabench"


def download(key, cache_dir=DEFAULT_CACHE):
    """Fetch one dataset's parquet file once and return its local path."""
    spec = DATASETS[key]
    path = Path(cache_dir) / f"{key}.parquet"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{HF}/{spec.file}"
    partial = path.with_suffix(".parquet.partial")
    with urlopen(url) as response, open(partial, "wb") as handle:  # noqa: S310 (fixed host)
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    partial.replace(path)
    return path


def load(key, cache_dir=DEFAULT_CACHE):
    """Return one dataset as published: sequence, tracks, gene, chromosome and targets."""
    frame = pd.read_parquet(download(key, cache_dir))
    return frame.assign(sequence=frame["sequence"].astype(str).str.strip().str.upper())


def cds_bounds(cds_track):
    """Return (start, end) of the coding region from the benchmark's codon-start track.

    The track marks the first nucleotide of each codon, so the marks are three apart and the
    region runs from the first mark to three past the last. A row whose marks are not three
    apart does not describe a reading frame, and raising here is the point: the whole
    comparison between regions depends on the coding region being the coding region.
    """
    marks = np.flatnonzero(np.asarray(cds_track))
    if marks.size == 0:
        raise ValueError("no coding region is marked in this row's cds track")
    if marks.size > 1 and not np.all(np.diff(marks) == 3):
        raise ValueError("the cds track's marks are not one per codon, so the frame is unclear")
    return int(marks[0]), int(marks[-1] + 3)


def regions(sequence, cds_track):
    """Split one transcript into its 5' untranslated region, coding sequence and 3' region."""
    start, end = cds_bounds(cds_track)
    end = min(end, len(sequence))
    return {"utr5": sequence[:start], "cds": sequence[start:end], "utr3": sequence[end:]}


def region_frame(frame):
    """Add utr5, cds, utr3 and their lengths to every row that carries a usable frame.

    The published ``cds`` column is the codon-start track; in the returned frame that name
    holds the coding sequence itself, because every later step reads regions as strings and
    two columns of one name would be ambiguous. The track stays available in the frame this
    was called on.

    Rows whose track does not describe a reading frame are kept with empty regions and a
    reason, so the gate can count them instead of a silent drop deciding the dataset's size.
    """
    rows = []
    for sequence, track in zip(frame["sequence"], frame["cds"]):
        try:
            parts = regions(sequence, track)
            rows.append({**parts, "frame_problem": ""})
        except ValueError as problem:
            rows.append({"utr5": "", "cds": "", "utr3": "", "frame_problem": str(problem)})
    found = pd.DataFrame(rows, index=frame.index)
    for name in ("utr5", "cds", "utr3"):
        found[f"{name}_len"] = found[name].str.len()
    return pd.concat([frame.drop(columns=["cds"]), found], axis=1)


def gate(frame, spec):
    """Answer the plan's feasibility questions about one dataset, from the data itself.

    Every entry is a measured number rather than a claim: whether the regions are present,
    whether the coding sequences are in frame, whether the labels are there, whether the
    rows can be grouped without leaking, and whether the same sequence appears twice. The
    verdict is the plan's own rule: a dataset without untranslated regions is not a
    full-transcript test, whatever else is true of it.
    """
    full = region_frame(frame)
    usable = full["frame_problem"] == ""
    targets = {t: float(frame[t].notna().mean()) for t in spec.targets if t in frame}
    has_genes = "gene" in frame
    genes = frame["gene"] if has_genes else pd.Series(dtype=object)
    answer = {
        "dataset": spec.key,
        "rows": int(len(frame)),
        "species": spec.species,
        "measures": spec.measures,
        "targets present": ", ".join(f"{k} {v:.0%}" for k, v in targets.items()) or "none",
        "has cds track": "cds" in frame,
        "has splice track": "splice" in frame,
        "rows with a coding region": float(usable.mean()),
        "rows with a 5' region": float((full.loc[usable, "utr5_len"] > 0).mean()),
        "rows with a 3' region": float((full.loc[usable, "utr3_len"] > 0).mean()),
        "coding sequence in frame": float((full.loc[usable, "cds_len"] % 3 == 0).mean()),
        "median transcript nt": int(frame["sequence"].str.len().median()),
        "longest transcript nt": int(frame["sequence"].str.len().max()),
        "median cds nt": int(full.loc[usable, "cds_len"].median()) if usable.any() else 0,
        "distinct genes": int(genes.nunique()) if has_genes else 0,
        "largest gene group": int(genes.value_counts().max()) if has_genes and len(genes) else 0,
        "repeated sequences": float(frame["sequence"].duplicated().mean()),
        "split rule": spec.split,
    }
    answer["verdict"], answer["caveats"] = _verdict(answer)
    return answer


# Below this many rows, a grouped split leaves a test set too small to resolve the
# differences this project compares; mRNABench's own paper flags the same dataset.
MINIMUM_ROWS = 1_000


def _verdict(answer):
    """The plan's gate, applied: what this dataset may be used for, and what to watch.

    Returns a one-line verdict and the caveats behind it. The verdict answers only the
    plan's question, whether this is a full-transcript test; the caveats carry everything
    that would otherwise have to be remembered when the numbers are read.
    """
    caveats = []
    if not answer["has cds track"]:
        return "no coding-region track: not a region comparison", "the file has no cds column"
    if answer["rows with a 5' region"] < 0.5 or answer["rows with a 3' region"] < 0.5:
        verdict = "regions missing: a sequence-level task, not a full-transcript test"
    elif answer["split rule"] == "homology" and answer["distinct genes"] < 100:
        verdict = "too few gene groups for a non-leaking split"
    else:
        verdict = "passes: full transcript, regions and grouped splits all present"

    missing_frame = 1 - answer["rows with a coding region"]
    if missing_frame > 0.001:
        caveats.append(f"{missing_frame:.1%} of rows mark no coding region, so a region "
                       "comparison must exclude them or score them as their own class")
    no_three_prime = 1 - answer["rows with a 3' region"]
    if no_three_prime > 0.001:
        caveats.append(f"{no_three_prime:.1%} of coding rows have no 3' region")
    if answer["rows"] < MINIMUM_ROWS:
        caveats.append(f"{answer['rows']:,} rows is too few to resolve small differences")
    if answer["split rule"] != "homology":
        caveats.append("the library is synthetic, so rows are split at random and a gene "
                       "grouping does not apply")
    if answer["repeated sequences"] > 0.001:
        caveats.append(f"{answer['repeated sequences']:.1%} of sequences repeat")
    return verdict, "; ".join(caveats)


REGION_NAMES = ("utr5", "cds", "utr3")


def region_features(regions, parts):
    """Counted features for a set of transcript regions: the composition floor an encoder must clear.

    Each named region contributes its GC fraction and its log length. The coding sequence
    additionally contributes the full composition vector of ``sequences.composition_features``,
    which counts codons **in frame**, and that is the reason this is not simply applied to the
    concatenated transcript: codon counting from the first nucleotide of a 5' region is out of
    frame and measures nothing.

    The sets are therefore nested. The features of "full transcript" contain those of "CDS
    only" plus the two untranslated regions, so comparing them asks exactly the plan's
    question, whether the regions outside the coding sequence add anything, rather than
    comparing two differently broken featurizations.
    """
    rows = []
    for _, row in regions.iterrows():
        found = []
        for name in REGION_NAMES:
            if name in parts:
                text = row[name]
                found += [gc_of(text), np.log1p(len(text))]
        if "cds" in parts and len(row["cds"]) >= 3:
            found += composition_of(row["cds"])
        rows.append(found)
    width = max(len(r) for r in rows)
    return np.asarray([r + [0.0] * (width - len(r)) for r in rows], dtype=float)


def gc_of(text):
    """GC fraction of a region, zero for an empty one rather than undefined."""
    from .sequences import gc_content

    return gc_content(rna_to_dna(text)) if text else 0.0


def composition_of(cds):
    """The project's composition vector for one coding sequence, counted in frame."""
    from .sequences import composition_features

    trimmed = cds[: len(cds) - (len(cds) % 3)]
    return list(composition_features(rna_to_dna(trimmed)))


def homology_map(species, cache_dir=DEFAULT_CACHE):
    """Return the benchmark's gene-to-homology-group table when it has been downloaded.

    mRNABench groups paralogues from the NCBI gene table so that related genes cannot sit on
    both sides of a split. That map is a separate download; when it is absent this returns
    None and the caller falls back to grouping by gene, which is strictly weaker and must be
    recorded as a deviation rather than passed off as the benchmark's own split.
    """
    path = Path(cache_dir) / "homology_maps" / f"{species}_homology_map.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    return dict(zip(table["gene_name"], table["gene_group"]))


def grouping(frame, species, cache_dir=DEFAULT_CACHE):
    """Return (group label per row, what the grouping is) for a non-leaking split.

    A gene with no homology group gets a group of its own, which is what the benchmark does
    too: an ungrouped gene is its own group rather than a free row.
    """
    if "gene" not in frame:
        return np.arange(len(frame)), "row (the library is synthetic, with no genes)"
    genes = frame["gene"].astype(str).to_numpy()
    mapping = homology_map(species, cache_dir)
    if mapping is None:
        return genes, "gene (the homology map is not downloaded; weaker than the benchmark's)"
    groups = np.array([str(mapping.get(g, f"__{g}")) for g in genes])
    return groups, "homology group (the benchmark's own paralogue grouping)"


def grouped_split(groups, seed, ratios=(0.7, 0.15, 0.15)):
    """Split rows into train, validation and test with whole groups kept together.

    The benchmark's own ratio, and its practice of assigning groups rather than rows. Groups
    are shuffled with the seed and filled into train, then validation, then test, so a run
    repeated with the same seed gives the same split and a different seed gives a different
    one; the benchmark reports the spread over nine such seeds.
    """
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("the three split ratios must sum to one")
    groups = np.asarray(groups)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    total = len(groups)
    wanted = [ratios[0] * total, (ratios[0] + ratios[1]) * total]
    picked, filled = [[], [], []], 0
    for group in unique:
        bucket = 0 if filled < wanted[0] else 1 if filled < wanted[1] else 2
        picked[bucket].append(members[group])
        filled += len(members[group])
    return tuple(np.sort(np.concatenate(p)) if p else np.array([], dtype=int) for p in picked)


def probe(X, y, split, alphas=(1e-3, 1e-2, 1e-1, 1, 10)):
    """Fit the benchmark's probe and score it the benchmark's way.

    mRNABench fits ``RidgeCV`` over exactly these five penalties on the training split and
    reports Pearson r on held-out rows; Spearman and mean squared error come back too
    because the project reports Spearman everywhere else and the two should be seen
    together. The probe is fit on frozen embeddings, which is the comparison the plan asks
    for: representations, not fine-tuned models.
    """
    from scipy.stats import pearsonr, spearmanr
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    train, validation, test = split
    model = Pipeline([("scale", StandardScaler()),
                      ("model", RidgeCV(alphas=np.asarray(alphas)))])
    model.fit(X[train], y[train])
    scores = {}
    for name, rows in (("val", validation), ("test", test)):
        if len(rows) == 0:
            continue
        predicted = model.predict(X[rows])
        scores[f"{name}_r"] = float(pearsonr(predicted, y[rows]).statistic)
        scores[f"{name}_spearman"] = float(spearmanr(predicted, y[rows]).statistic)
        scores[f"{name}_mse"] = float(np.mean((predicted - y[rows]) ** 2))
    scores["alpha"] = float(model.named_steps["model"].alpha_)
    return scores


def probe_over_seeds(X, y, groups, seeds, ratios=(0.7, 0.15, 0.15)):
    """Run the probe on one split per seed and return every seed's scores as a frame.

    The benchmark reports the mean and standard error over its nine seeds rather than one
    split's number, because a single grouped split of a few thousand genes moves by more
    than the differences being compared.
    """
    rows = []
    for seed in seeds:
        split = grouped_split(groups, seed, ratios)
        rows.append({"seed": seed, **probe(X, y, split),
                     "train": len(split[0]), "test": len(split[2])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Embedding whole transcripts with encoders whose context is shorter than one
# ---------------------------------------------------------------------------

def encoder_text(encoder, nucleotides, protein=None):
    """The string one encoder reads for a transcript region or its translation."""
    if encoder.modality == "protein":
        return protein_input(encoder, protein if protein is not None
                             else translate_cds(nucleotides))
    return rna_to_dna(nucleotides) if encoder.alphabet == "dna" else dna_to_rna(nucleotides)


def chunk(text, size):
    """Cut a string into consecutive windows of at most size characters."""
    if size < 1:
        raise ValueError("a window must hold at least one character")
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


def pooled_chunked(text, tokenizer, model, device, window):
    """Mean-pool one sequence, embedding it in windows when it is longer than the context.

    The windows are consecutive and non-overlapping, their token states are concatenated,
    and the mean is taken over the whole concatenation, which is mRNABench's rule for a
    model whose context is shorter than a transcript. A window cannot see outside itself,
    so this is an approximation and is recorded as a deviation wherever it is used.
    """
    import torch

    states = []
    for piece in chunk(text, window):
        encoded = tokenizer(piece, return_tensors="pt", truncation=True,
                            max_length=model.config.max_position_embeddings)
        with torch.no_grad():
            out = model(**{k: v.to(device) for k, v in encoded.items()})
        states.append(out[0][0].float().cpu().numpy())
    return np.concatenate(states, axis=0).mean(axis=0)


def embed_with_cache(encoder_key, texts, cache_path, device="cpu", window=None,
                     progress_every=200, progress=print):
    """Pool every text with one Stage 1 encoder, in windows, reusing a matching cache.

    ``window`` is the number of characters per window; it defaults to what the encoder's own
    token limit allows, counting the way its tokenizer counts. The cache records the encoder,
    the window and a fingerprint of the inputs, so a matrix embedded under a different window
    is never mistaken for this one.
    """
    from .encoders import free_memory, load_encoder

    encoder = ENCODERS[encoder_key]
    texts = list(texts)
    if not texts:
        raise ValueError("Embedding requires at least one input sequence.")
    window = window or default_window(encoder)
    expected = {
        "cache_version": 1,
        "encoder": encoder.key,
        "checkpoint": encoder.checkpoint,
        "revision": encoder.revision or "",
        "window": int(window),
        "n_sequences": len(texts),
        "fingerprint": sequences_fingerprint(texts),
    }

    def compute():
        tokenizer, model = load_encoder(encoder, device)
        try:
            vectors = []
            for i, text in enumerate(texts, start=1):
                vectors.append(pooled_chunked(text, tokenizer, model, device, window))
                if progress and progress_every and i % progress_every == 0:
                    progress(f"    {encoder.key}: {i}/{len(texts)}")
            return np.vstack(vectors)
        finally:
            del model, tokenizer
            free_memory()

    return matrix_cache.load_or_compute(cache_path, expected, compute)


# How many nucleotides (or residues) of input one token of each encoder covers, for the
# window arithmetic. Protein encoders read one residue per token; the k-mer DNA encoders six
# nucleotides; the codon encoders three; the rest one.
NUCLEOTIDES_PER_TOKEN = {"ntv2_100m": 6, "nt500m": 6, "mrnafm": 3, "calm": 3}


def default_window(encoder):
    """The longest input, in characters, that stays inside an encoder's token limit.

    Two tokens are left for the start and end markers. DNABERT-2's byte-pair vocabulary has
    no fixed rate, so it gets the conservative one character per token rather than a guess
    that would overrun its context on a GC-poor transcript.
    """
    per_token = 1 if encoder.modality == "protein" else NUCLEOTIDES_PER_TOKEN.get(encoder.key, 1)
    return (encoder.max_len - 2) * per_token
