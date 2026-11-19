"""Build DNA-with-transcription-factor pair tables from DAP-seq peaks and a genome.

This is the proposed addition to the fusion plan's partner tasks: a DNA region and the factor
that binds it are two different molecules, so the pair carries information neither sequence
holds alone. Unlike the RNA-protein benchmarks, nobody publishes this as a pair table. The
construction is ours, and the choices below are the experiment's design, not plumbing.

**Where the three inputs come from.** Peaks are the narrowPeak files of the DAP-seq release
(GEO GSE60143), one per experiment, named for the factor's family and either its AGI locus or
its gene symbol. Protein sequences come from the TAIR10 proteome, matched on locus or symbol,
keeping the longest isoform per gene. Region sequences are cut from the TAIR10 chromosomes.

**A positive is a window, not a peak.** Peaks differ in width across factors and experiments, so
a model could learn the factor from the window length alone. Every region is therefore a fixed
window centred on the peak's summit, and the width is recorded with the table.

**Negatives are the design decision.** A negative drawn uniformly from the genome is separable
from a peak by base composition alone, which a model will find instead of the factor's motif.
Each positive therefore gets one negative from the same chromosome whose GC content matches it
within a stated tolerance and which overlaps no peak of that factor in any replicate, including
the peaks ranked below the cut. That makes the two classes equal in number per factor, so a
model cannot score by learning which factors have many peaks, and comparable in composition, so
it cannot score on GC alone.

**Groups are families and genomic bins.** A factor-disjoint split that separated two members of
one family would still leak, because paralogues share motifs, so factors are grouped by the
family named in the experiment. Regions are grouped into fixed genomic bins, so two windows a
few hundred bases apart cannot land on opposite sides of a split.

**What this cannot do.** DAP-seq is in vitro: it measures what a factor binds on naked DNA,
without chromatin. It therefore says what a factor can bind, not what it does bind in a cell,
and no result from this table should be read as in vivo occupancy.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .pair_tasks import PairFrame

LOCUS = re.compile(r"^AT[1-5MC]G\d{5}$", re.IGNORECASE)
EXPERIMENT = re.compile(
    r"GSM\d+_(?P<assay>DAPSeq|ChIPSeq)-(?P<family>[^_]+)_(?P<system>[^-]+)-"
    r"(?P<factor>.+?)_(?P<ecotype>col|colamp)[^-]*-chr")
WINDOW = 201
BIN = 10_000
GC_TOLERANCE = 0.05


def read_proteome(path):
    """Longest protein per gene, indexed by AGI locus and by gene symbol."""
    by_locus, by_symbol = {}, {}
    opener = gzip.open if str(path).endswith(".gz") else open

    def store(header, sequence):
        gene = re.search(r"gene:(AT[1-5MC]G\d{5})", header)
        if not gene or not sequence:
            return
        locus = gene.group(1).upper()
        if len(sequence) > len(by_locus.get(locus, "")):
            by_locus[locus] = sequence
        symbol = re.search(r"gene_symbol:(\S+)", header)
        if symbol:
            key = symbol.group(1).upper()
            if len(sequence) > len(by_symbol.get(key, "")):
                by_symbol[key] = sequence

    with opener(path, "rt") as handle:
        header, parts = None, []
        for line in handle:
            if line.startswith(">"):
                if header:
                    store(header, "".join(parts))
                header, parts = line, []
            else:
                parts.append(line.strip())
        if header:
            store(header, "".join(parts))
    return {"locus": by_locus, "symbol": by_symbol}


def resolve_factor(name, proteome):
    """Map an experiment's factor name to a locus and its protein sequence.

    Names in the release are a locus in either case, a gene symbol, or a symbol carrying the
    species prefix. A name that resolves to nothing is reported rather than guessed at.
    """
    candidates = [name, name.upper()]
    if name.upper().startswith("AT") and not LOCUS.match(name):
        candidates.append(name[2:].upper())      # AtIDD11 -> IDD11
    for candidate in candidates:
        if LOCUS.match(candidate):
            sequence = proteome["locus"].get(candidate.upper())
            if sequence:
                return candidate.upper(), sequence
        sequence = proteome["symbol"].get(candidate.upper())
        if sequence:
            return candidate.upper(), sequence
    return None, None


def read_genome(directory):
    """Chromosome sequences, upper case, keyed by the name the peak files use."""
    chromosomes = {}
    for path in sorted(Path(directory).glob("chr*.fa.gz")):
        with gzip.open(path, "rt") as handle:
            name, parts = None, []
            for line in handle:
                if line.startswith(">"):
                    name = line[1:].split()[0]
                else:
                    parts.append(line.strip())
        chromosomes[f"chr{name}"] = "".join(parts).upper()
    return chromosomes


def read_peaks(path):
    """One narrowPeak file: chromosome, summit position and score, strongest first."""
    rows = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5 or not fields[1].isdigit():
                continue
            start, end, score = int(fields[1]), int(fields[2]), float(fields[4])
            offset = int(fields[9]) if len(fields) > 9 and fields[9].lstrip("-").isdigit() else -1
            summit = start + offset if offset >= 0 else (start + end) // 2
            rows.append((fields[0], summit, score))
    rows.sort(key=lambda row: -row[2])
    return rows


def _window(chromosome, centre, width):
    half = width // 2
    start = centre - half
    if start < 0 or start + width > len(chromosome):
        return None, None
    return start, chromosome[start:start + width]


def _gc(sequence):
    return (sequence.count("G") + sequence.count("C")) / max(len(sequence), 1)


def build_dapseq_pairs(peak_dir, genome_dir, proteome_path, *, factors=40, peaks_per_factor=200,
                       window=WINDOW, seed=0, assay="DAPSeq", gc_tolerance=GC_TOLERANCE,
                       name="dapseq"):
    """Build the pair table: windows under a factor's strongest peaks, and matched negatives.

    ``factors`` caps how many factors are used, strongest-covered first, so a pilot can be
    built without reading every experiment. Replicates of one factor are merged, and a pair
    built twice is kept once.
    """
    rng = np.random.default_rng(seed)
    proteome = read_proteome(proteome_path)
    chromosomes = read_genome(genome_dir)

    # One entry per factor, carrying every replicate's peaks, so exclusion uses all of them.
    collected, unresolved = {}, []
    for path in sorted(Path(peak_dir).glob("*.narrowPeak.gz")):
        match = EXPERIMENT.match(path.name)
        if not match or match.group("assay") != assay:
            continue
        locus, sequence = resolve_factor(match.group("factor"), proteome)
        if not sequence:
            unresolved.append(match.group("factor"))
            continue
        entry = collected.setdefault(locus, {"family": match.group("family"),
                                             "sequence": sequence, "peaks": [], "files": 0})
        entry["peaks"].extend(read_peaks(path))
        entry["files"] += 1
    chosen = sorted(collected.items(), key=lambda item: -len(item[1]["peaks"]))[:factors]

    rows, regions, proteins, region_groups = [], {}, {}, {}
    for locus, entry in chosen:
        proteins[locus] = entry["sequence"]
        peaks = sorted(entry["peaks"], key=lambda peak: -peak[2])
        # Every peak of this factor is excluded from its negatives, not only the ones used.
        occupied = {}
        for chromosome, summit, _ in entry["peaks"]:
            occupied.setdefault(chromosome, []).append(summit)
        occupied = {key: np.sort(np.asarray(value)) for key, value in occupied.items()}
        used = 0
        for chromosome, summit, _ in peaks:
            if used >= peaks_per_factor or chromosome not in chromosomes:
                continue
            start, sequence = _window(chromosomes[chromosome], summit, window)
            if sequence is None or "N" in sequence:
                continue
            region = f"{chromosome}:{start}"
            regions[region] = sequence
            region_groups[region] = f"{chromosome}:{start // BIN}"
            rows.append((region, locus, 1, "measured_positive"))
            # A negative of matched composition, from the same chromosome, away from any peak.
            target, chosen_negative = _gc(sequence), None
            for _attempt in range(200):
                centre = int(rng.integers(window, len(chromosomes[chromosome]) - window))
                nearby = occupied.get(chromosome)
                if nearby is not None and nearby.size:
                    index = int(np.searchsorted(nearby, centre))
                    neighbours = nearby[max(0, index - 1):index + 1]
                    if neighbours.size and np.min(np.abs(neighbours - centre)) < window:
                        continue
                negative_start, negative = _window(chromosomes[chromosome], centre, window)
                if negative is None or "N" in negative:
                    continue
                if abs(_gc(negative) - target) <= gc_tolerance:
                    chosen_negative = (f"{chromosome}:{negative_start}", negative_start, negative)
                    break
            if chosen_negative is None:
                continue
            region_id, negative_start, negative = chosen_negative
            regions[region_id] = negative
            region_groups[region_id] = f"{chromosome}:{negative_start // BIN}"
            rows.append((region_id, locus, 0, "artificial_sampled"))
            used += 1

    table = pd.DataFrame(rows, columns=["left", "right", "label", "provenance"])
    duplicates = int(table.duplicated(["left", "right"]).sum())
    table = table.drop_duplicates(["left", "right"]).reset_index(drop=True)
    # Paralogues share motifs, so a factor-disjoint split groups by family, not by factor.
    families = {locus: f"family:{entry['family']}" for locus, entry in chosen}
    frame = PairFrame(name, table, regions, proteins, region_groups, families,
                      "binding", "dna", "protein")
    frame.note = (f"Positives are {window} bp windows centred on DAP-seq peak summits, the "
                  f"strongest {peaks_per_factor} per factor. Each has one negative from the "
                  "same chromosome, GC-matched and clear of every peak of that factor. "
                  "DAP-seq is in vitro: this is what a factor can bind, not what it occupies "
                  "in a cell.")
    frame.construction = {"window": window, "peaks_per_factor": peaks_per_factor,
                          "gc_tolerance": gc_tolerance, "bin": BIN, "seed": int(seed),
                          "factors": len(chosen), "duplicates_removed": duplicates,
                          "unresolved_experiments": len(unresolved),
                          "region_groups": "genomic bins", "factor_groups": "family"}
    return frame


def main(peak_dir, genome_dir, proteome_path, out_dir, factors=40, peaks_per_factor=200):
    from .pair_sources import write_pair_dataset

    frame = build_dapseq_pairs(peak_dir, genome_dir, proteome_path, factors=int(factors),
                               peaks_per_factor=int(peaks_per_factor))
    metadata = write_pair_dataset(frame, out_dir)
    summary = metadata["summary"]
    print(f"{frame.name}: {summary['pairs']} pairs, {summary['positives']} positive, "
          f"{summary['negatives']} negative")
    print(f"  {summary['left_entities']} regions in {summary['left_groups']} genomic bins, "
          f"{summary['right_entities']} factors in {summary['right_groups']} families")
    print(f"  construction: {frame.construction}")
    return metadata


if __name__ == "__main__":  # pragma: no cover - command line entry
    import sys

    main(*sys.argv[1:])
