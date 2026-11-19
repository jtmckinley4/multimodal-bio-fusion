"""Build pair tables from published RNA-protein interaction files.

The partner tasks need a table of pairs with their sequences, their entity groups and an
honest record of where each label came from (``pair_tasks.PairFrame``). The published
benchmarks do not arrive in that shape, and the way they are reshaped decides what a later
result can claim, so the decisions are here rather than inside a notebook.

**The files this reads.** The Dobbs lab RPIMotif repository publishes four sets as three-line
records: a header, the protein sequence, the RNA sequence.

* ``Positive_Fasta_RPI2241.txt`` and ``Negative_Fasta_RPI2241.txt``, 2,241 records each.
  The positives are RNA-protein chain pairs from one structural complex. The negatives were
  built by pairing a protein chain from one complex with an RNA chain from another.
* ``NPInter_UniqSeq.txt``, 11,281 curated interactions, headers naming a NONCODE RNA, a
  UniProt protein and a species.
* ``NRBP_pairs.txt``, 971 records pairing proteins with no recorded RNA binding against RNAs
  drawn from elsewhere.

**Negatives are the decision, not a detail.** Both negative files are constructed, not
measured: nobody tested those pairs. They are kept, and labelled ``artificial_sampled`` so
that no later table can present them as evidence of non-binding, but they are also weak in a
specific way. A negative built by drawing two entities at random is predictable from how often
each entity appears, which is the shortcut the degree baseline in ``pair_tasks`` exists to
expose. ``sample_negatives`` builds a stronger alternative: pairs that are absent from the
positive set, drawn so each entity keeps the same number of negatives as positives. A model
cannot then score by learning which entities are popular, because popularity is matched
across the two classes.

**Entity identity and groups.** An identifier names an entity; a group is what a split may not
separate. Here a group is an exact sequence, so the same protein appearing under two PDB chain
identifiers cannot land on both sides of a split. That is the floor, not the ceiling: proteins
of 90% identity will still be separated, and clustering by similarity before the real runs is
recorded as a known gap rather than silently assumed away.

**One alphabet.** RPI2241 writes RNA with U, NPInter with T. Both are mapped to U, and the
substitution is recorded in the dataset's metadata.

**Malformed records are dropped, not repaired.** A few dozen records across these files carry a
character that is not a nucleotide inside an otherwise ordinary RNA sequence, such as an ``F``
near the end of a tRNA. Guessing what was meant would put an invented sequence into a training
set, so those records are dropped and counted in the dataset's metadata.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import pair_tasks
from .pair_tasks import PairFrame

RNA_LETTERS = set("ACGUN")
PROTEIN_LETTERS = set("ACDEFGHIKLMNPQRSTVWYXBZUO")
SOURCES = ("rpi2241", "npinter")


def read_records(path):
    """Read a three-line file: header, protein sequence, RNA sequence.

    Returns a list of ``(header, protein, rna)``. A file whose line count is not a multiple of
    three, or whose records do not alternate as expected, is an error rather than something to
    repair silently.
    """
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    if len(lines) % 3:
        raise ValueError(f"{path}: {len(lines)} non-empty lines is not a multiple of three")
    records = []
    for index in range(0, len(lines), 3):
        header, protein, rna = lines[index:index + 3]
        if not header.startswith(">"):
            raise ValueError(f"{path}: record {index // 3} does not begin with a header")
        if protein.startswith(">") or rna.startswith(">"):
            raise ValueError(f"{path}: record {index // 3} is missing one of its sequences")
        records.append((header[1:], protein.upper(), rna.upper()))
    return records


def _rna(sequence):
    """One alphabet for RNA, whether the file wrote cDNA or RNA."""
    return sequence.replace("T", "U")


def _malformed(kind, sequence):
    """Letters this alphabet does not have, or nothing at all: the record cannot be used."""
    allowed = RNA_LETTERS if kind == "rna" else PROTEIN_LETTERS
    return sorted(set(sequence) - allowed) if sequence else ["<empty>"]


def _entity(registry, name, sequence, kind):
    """Register one entity, refusing to let one name carry two different sequences."""
    previous = registry.get(name)
    if previous is not None and previous != sequence:
        raise ValueError(f"{name}: the same identifier carries two different {kind} sequences")
    registry[name] = sequence
    return name


def _groups(sequences):
    """Group entities by exact sequence, so identical sequences cannot be split apart."""
    return {name: "seq:" + hashlib.sha256(sequence.encode()).hexdigest()[:16]
            for name, sequence in sequences.items()}


def _frame(name, rows, rna_sequences, protein_sequences, note):
    table = pd.DataFrame(rows, columns=["left", "right", "label", "provenance"])
    duplicated = int(table.duplicated(["left", "right"]).sum())
    if duplicated:
        # A pair recorded twice is one pair; conflicting labels are reported rather than picked.
        clashes = table.groupby(["left", "right"])["label"].nunique()
        conflicting = int((clashes > 1).sum())
        if conflicting:
            raise ValueError(f"{name}: {conflicting} pairs carry both labels; resolve the "
                             "sources before building a table")
        table = table.drop_duplicates(["left", "right"]).reset_index(drop=True)
    frame = PairFrame(name, table, rna_sequences, protein_sequences,
                      _groups(rna_sequences), _groups(protein_sequences))
    frame.note = note
    frame.duplicates_removed = duplicated
    return frame


def load_rpi2241(data_dir):
    """The structure-derived benchmark: 2,241 positives and its 2,241 constructed negatives.

    Identifiers are PDB chains (``1drz_A`` for the protein, ``1drz_B`` for the RNA), so one
    sequence can appear under several names; the exact-sequence groups hold those together.
    """
    data_dir = Path(data_dir)
    rna_sequences, protein_sequences, rows, dropped = {}, {}, [], []
    for file_name, label, provenance in (
            ("Positive_Fasta_RPI2241.txt", 1, "measured_positive"),
            ("Negative_Fasta_RPI2241.txt", 0, "artificial_sampled")):
        for header, protein, rna in read_records(data_dir / file_name):
            parts = header.split(",")
            if len(parts) != 2:
                raise ValueError(f"{file_name}: header {header!r} is not a chain pair")
            protein_name, rna_name = parts
            rna = _rna(rna)
            if _malformed("rna", rna) or _malformed("protein", protein):
                dropped.append(header)
                continue
            rows.append((_entity(rna_sequences, rna_name, rna, "rna"),
                         _entity(protein_sequences, protein_name, protein, "protein"),
                         label, provenance))
    frame = _frame("rpi2241", rows, rna_sequences, protein_sequences,
                  "Positives are RNA and protein chains of one structural complex. The supplied "
                   "negatives pair a protein from one complex with an RNA from another: nobody "
                   "measured them, and they are predictable from how often each chain appears.")
    frame.dropped_records = dropped
    return frame


def load_npinter(data_dir, *, include_supplied_negatives=True):
    """The curated interactions, with the supplied non-binder negatives if wanted.

    The two sets are unbalanced by roughly twelve to one, so average precision read off this
    table is close to the positive rate by construction. ``sample_negatives`` is the way to a
    table whose classes are comparable.
    """
    data_dir = Path(data_dir)
    rna_sequences, protein_sequences, rows, dropped = {}, {}, [], []
    species = {}
    for header, protein, rna in read_records(data_dir / "NPInter_UniqSeq.txt"):
        parts = header.split(",")
        if len(parts) < 4:
            raise ValueError(f"NPInter_UniqSeq.txt: header {header!r} is too short")
        rna_name, protein_name = parts[1], parts[3]
        rna = _rna(rna)
        if _malformed("rna", rna) or _malformed("protein", protein):
            dropped.append(header)
            continue
        species[rna_name] = parts[4] if len(parts) > 4 else "unknown"
        rows.append((_entity(rna_sequences, rna_name, rna, "rna"),
                     _entity(protein_sequences, protein_name, protein, "protein"),
                     1, "measured_positive"))
    if include_supplied_negatives:
        for index, (header, protein, rna) in enumerate(read_records(data_dir / "NRBP_pairs.txt")):
            # These headers name the protein only; the RNA it was paired with has no identifier.
            rna = _rna(rna)
            if _malformed("rna", rna) or _malformed("protein", protein):
                dropped.append(header)
                continue
            rows.append((_entity(rna_sequences, f"NRBP_rna_{index:04d}", rna, "rna"),
                         _entity(protein_sequences, header, protein, "protein"),
                         0, "artificial_sampled"))
    frame = _frame("npinter", rows, rna_sequences, protein_sequences,
                   "Positives are curated interactions. The supplied negatives pair proteins "
                   "with no recorded RNA binding against unrelated RNAs, and there are twelve "
                   "times fewer of them than positives.")
    frame.species = species
    frame.dropped_records = dropped
    return frame


def sample_negatives(frame, *, ratio=1.0, seed=0, match_degree=True, name=None):
    """Build negatives from pairs the sources never record, keeping each entity's degree.

    A negative drawn uniformly is predictable from how often each entity appears, so a model
    can score well without reading a sequence. With ``match_degree``, each entity receives
    negatives in proportion to its positives, which removes that shortcut: the degree baseline
    then scores near chance by construction, and any remaining signal has to come from the
    pair.

    Returns a new frame holding the positives of ``frame`` and the sampled negatives only; the
    sources' own constructed negatives are left behind, since mixing two negative definitions
    in one table makes a score impossible to interpret.
    """
    rng = np.random.default_rng(seed)
    positives = frame.pairs[frame.pairs["label"] == 1]
    if positives.empty:
        raise ValueError(f"{frame.name}: no positives to match")
    known = set(zip(frame.pairs["left"], frame.pairs["right"]))
    lefts = positives["left"].value_counts()
    rights = sorted(set(positives["right"]))
    right_weight = positives["right"].value_counts().reindex(rights).fillna(0).to_numpy()
    right_weight = (right_weight / right_weight.sum() if match_degree
                    else np.full(len(rights), 1 / len(rights)))

    wanted_total = int(round(ratio * len(positives)))
    rows, built = [], set()
    for left, count in lefts.items():
        # Each RNA gets negatives in proportion to its own positives, so the two classes
        # cannot be told apart by how often an entity appears.
        wanted = max(1, int(round(ratio * count))) if match_degree else 0
        for _ in range(wanted):
            for _attempt in range(64):
                right = rights[int(rng.choice(len(rights), p=right_weight))]
                if (left, right) not in known and (left, right) not in built:
                    built.add((left, right))
                    rows.append((left, right, 0, "artificial_sampled"))
                    break
    while not match_degree and len(rows) < wanted_total:
        left = positives["left"].iloc[int(rng.integers(len(positives)))]
        right = rights[int(rng.integers(len(rights)))]
        if (left, right) not in known and (left, right) not in built:
            built.add((left, right))
            rows.append((left, right, 0, "artificial_sampled"))

    table = pd.concat([positives, pd.DataFrame(rows, columns=positives.columns)],
                      ignore_index=True)
    built_frame = PairFrame(name or f"{frame.name}-sampled", table,
                            frame.left_sequences, frame.right_sequences,
                            frame.left_groups, frame.right_groups, frame.kind,
                            frame.left_modality, frame.right_modality)
    built_frame.note = (f"Positives from {frame.name}. Negatives sampled from pairs no source "
                        f"records, {'with' if match_degree else 'without'} degree matching, "
                        f"seed {seed}. No negative here was measured.")
    built_frame.negative_sampling = {"ratio": ratio, "seed": int(seed),
                                     "match_degree": bool(match_degree),
                                     "negatives": len(rows)}
    return built_frame



def load_affinity(path, *, name, endpoint, log_transform=False, drop_duplicates=True):
    """Build an affinity table from a drug-target file of ID, SMILES, ID, sequence, value.

    This is the shape the Therapeutics Data Commons releases use for DAVIS and KIBA: one row
    per measured pair, the ligand as a SMILES string and the target as its amino-acid sequence.
    The left side is the protein and the right side the ligand, so the same splits, controls and
    arms apply as for the binding tasks.

    ``log_transform`` converts a dissociation constant in nanomolar to its negative logarithm,
    which is the scale affinity models are fitted and read on; it is recorded in ``endpoint``
    rather than left implicit. One table carries one endpoint: mixing constants measured by
    different assays into a single column is what the plan forbids, and ``PairFrame`` refuses it.
    """
    table = pd.read_csv(path, sep="\t")
    expected = {"ID1", "X1", "ID2", "X2", "Y"}
    missing = expected - set(table.columns)
    if missing:
        raise ValueError(f"{path}: affinity file is missing {sorted(missing)}")
    table = table.dropna(subset=["X1", "X2", "Y"])
    proteins = {str(row.ID2).strip('"'): str(row.X2).strip('"').upper()
                for row in table.itertuples(index=False)}
    ligands = {str(row.ID1).strip('"'): str(row.X1).strip('"')
               for row in table.itertuples(index=False)}
    values = table["Y"].astype(float).to_numpy()
    if log_transform:
        if (values <= 0).any():
            raise ValueError(f"{path}: a non-positive constant cannot be log-transformed")
        values = -np.log10(values * 1e-9)
    rows = pd.DataFrame({"left": [str(v).strip('"') for v in table["ID2"]],
                         "right": [str(v).strip('"') for v in table["ID1"]],
                         "label": values,
                         "provenance": "measured_positive",
                         "endpoint": endpoint})
    duplicated = 0
    if drop_duplicates:
        # One pair measured twice is averaged, not counted twice, and the count is recorded.
        duplicated = int(rows.duplicated(["left", "right"]).sum())
        if duplicated:
            rows = (rows.groupby(["left", "right"], as_index=False)
                    .agg({"label": "mean", "provenance": "first", "endpoint": "first"}))
    frame = PairFrame(name, rows, proteins, ligands,
                      _groups(proteins), _groups(ligands), "affinity", "protein", "ligand")
    frame.note = (f"Measured {endpoint} per protein-ligand pair. Proteins are amino-acid "
                  "sequences, ligands are SMILES strings, and groups are exact sequence or "
                  "exact SMILES identity.")
    frame.duplicates_removed = duplicated
    return frame


def describe(frame):
    """The numbers to read before trusting anything fitted on this table."""
    table = frame.pairs
    positives = table[table["label"] == 1]
    degrees = {side: table[side].value_counts() for side in ("left", "right")}
    summary = {"name": frame.name, "kind": frame.kind, "pairs": int(len(table)),
               "left_entities": int(table["left"].nunique()),
               "right_entities": int(table["right"].nunique()),
               "left_groups": int(len(set(frame.groups("left")))),
               "right_groups": int(len(set(frame.groups("right")))),
               "pairs_per_left": [int(degrees["left"].min()), int(degrees["left"].max())],
               "pairs_per_right": [int(degrees["right"].min()), int(degrees["right"].max())],
               "provenance": frame.label_provenance_counts(),
               "density": round(len(table) / (table["left"].nunique()
                                              * table["right"].nunique()), 6),
               "note": getattr(frame, "note", "")}
    if frame.kind == "binding":
        summary.update({"positives": int((table["label"] == 1).sum()),
                        "negatives": int((table["label"] == 0).sum()),
                        "positive_rate": round(float(table["label"].mean()), 4),
                        "positives_per_left_median":
                            float(positives["left"].value_counts().median())})
    else:
        values = table["label"].astype(float)
        floor = float(values.min())
        summary.update({"endpoint": str(table["endpoint"].iloc[0]),
                        "label_range": [round(floor, 4), round(float(values.max()), 4)],
                        "label_mean": round(float(values.mean()), 4),
                        "label_sd": round(float(values.std()), 4),
                        # A value piled up at the floor is a censoring limit, not a measurement.
                        "at_floor": int((values == floor).sum()),
                        "at_floor_share": round(float((values == floor).mean()), 4)})
    return summary


# What a side's strings are written as, by modality. A ligand side holds SMILES, which is
# not a biological sequence and does not belong in a file called .fasta.
SIDE_FILES = {"dna": ("dna.fasta", "ACGT, uppercase"),
              "rna": ("rna.fasta", "T mapped to U"),
              "protein": ("protein.fasta", "amino acids, as published"),
              "ligand": ("ligand.smi", "SMILES, as published")}


def side_files(frame):
    """The file each side is written to, keyed by side, with the two sides never colliding."""
    chosen = {}
    for side in pair_tasks.SIDES:
        name, _ = SIDE_FILES[frame.modality(side)]
        chosen[side] = name
    if chosen["left"] == chosen["right"]:
        chosen = {side: f"{side}_{chosen[side]}" for side in chosen}
    return chosen


def write_pair_dataset(frame, directory):
    """Save a pair table as the three files a later run reads: pairs, sequences, metadata."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frame.pairs.to_csv(directory / "pairs.csv", index=False)
    files = side_files(frame)
    for side in pair_tasks.SIDES:
        sequences = frame.sequences(side)
        with open(directory / files[side], "w", encoding="utf-8") as handle:
            for name in sorted(sequences):
                handle.write(f">{name}\n{sequences[name]}\n")
    # Groups decide what a disjoint split may separate, so they are saved rather than
    # re-derived. DAP-seq groups its regions by 10 kb genomic bin and its factors by family;
    # re-deriving groups from sequence identity on read would quietly replace both, and the
    # split would then put a factor's relatives on either side of it.
    stored_groups = {side: {name: frame.group_of(side, name)
                            for name in sorted(frame.sequences(side))}
                     for side in pair_tasks.SIDES}
    (directory / "groups.json").write_text(json.dumps(stored_groups, indent=2) + "\n",
                                           encoding="utf-8")
    declared = {side: (frame.left_groups if side == "left" else frame.right_groups)
                for side in pair_tasks.SIDES}
    metadata = {"schema": "pair-dataset.v2", "name": frame.name, "kind": frame.kind,
                "summary": describe(frame),
                "groups": {"file": "groups.json",
                           "rule": {side: ("declared by the builder" if declared[side]
                                           else "exact sequence identity")
                                    for side in pair_tasks.SIDES},
                           "counts": {side: len(set(stored_groups[side].values()))
                                      for side in pair_tasks.SIDES},
                           "known_gap": "similar but non-identical sequences are not grouped "
                                        "unless the builder grouped them; cluster by identity "
                                        "before the reported runs"},
                "sides": {side: {"modality": frame.modality(side), "file": files[side],
                                 "written_as": SIDE_FILES[frame.modality(side)][1],
                                 "entities": len(frame.sequences(side))}
                          for side in pair_tasks.SIDES},
                "negative_sampling": getattr(frame, "negative_sampling", None),
                "construction": getattr(frame, "construction", None),
                "duplicates_removed": getattr(frame, "duplicates_removed", 0),
                "dropped_records": {"count": len(getattr(frame, "dropped_records", [])),
                                    "reason": "a sequence carried letters its alphabet does not "
                                              "have, or was empty; the record was not repaired",
                                    "headers": getattr(frame, "dropped_records", [])[:20]},
                "note": getattr(frame, "note", "")}
    (directory / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                            encoding="utf-8")
    return metadata


def _read_entities(path):
    """Read a name-to-string file, whichever alphabet it holds."""
    entries, name = {}, None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            name = line[1:]
        elif name is not None:
            entries[name] = entries.get(name, "") + line.strip()
    return entries


def read_pair_dataset(directory):
    """Load a saved pair dataset back into a frame.

    Both schema versions are read. A v1 directory predates the recorded side modalities and
    always held RNA on the left and protein on the right, which is what it is read as.
    """
    directory = Path(directory)
    metadata = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
    schema = metadata.get("schema")
    if schema not in ("pair-dataset.v1", "pair-dataset.v2"):
        raise ValueError(f"{directory}: unsupported pair dataset schema {schema!r}")
    if schema == "pair-dataset.v1":
        sides = {"left": {"modality": "rna", "file": "rna.fasta"},
                 "right": {"modality": "protein", "file": "protein.fasta"}}
    else:
        sides = metadata["sides"]
    sequences = {side: _read_entities(directory / sides[side]["file"])
                 for side in pair_tasks.SIDES}
    # Entity names can look numeric (a ligand is named by its PubChem CID), and a name read as
    # a number would not match the name its sequence is stored under.
    table = pd.read_csv(directory / "pairs.csv", dtype={"left": str, "right": str})
    groups_path = directory / "groups.json"
    if groups_path.exists():
        stored_groups = json.loads(groups_path.read_text(encoding="utf-8"))
        groups = {side: dict(stored_groups[side]) for side in pair_tasks.SIDES}
    else:
        # A v1 directory saved no groups; exact sequence identity is what it was built with.
        groups = {side: _groups(sequences[side]) for side in pair_tasks.SIDES}
    frame = PairFrame(metadata["name"], table, sequences["left"], sequences["right"],
                      groups["left"], groups["right"],
                      metadata.get("kind", "binding"),
                      sides["left"]["modality"], sides["right"]["modality"])
    frame.note = metadata.get("note", "")
    return frame


def main(data_dir, out_dir):
    """Build every table from a RPIMotif Data folder and report what each one holds."""
    out_dir = Path(out_dir)
    built = {}
    for frame in (load_rpi2241(data_dir),
                  load_npinter(data_dir),
                  sample_negatives(load_npinter(data_dir, include_supplied_negatives=False),
                                   name="npinter-matched")):
        metadata = write_pair_dataset(frame, out_dir / frame.name)
        built[frame.name] = metadata["summary"]
        print(f"\n{frame.name}: {metadata['summary']['pairs']} pairs, "
              f"{metadata['summary']['positives']} positive, "
              f"{metadata['summary']['negatives']} negative")
        print(f"  {metadata['summary']['left_entities']} RNAs in "
              f"{metadata['summary']['left_groups']} groups, "
              f"{metadata['summary']['right_entities']} proteins in "
              f"{metadata['summary']['right_groups']} groups")
        print(f"  provenance: {metadata['summary']['provenance']}")
    return built


if __name__ == "__main__":  # pragma: no cover - command line entry
    import sys

    main(sys.argv[1], sys.argv[2])
