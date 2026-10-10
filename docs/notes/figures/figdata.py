"""Stage 1 values used by the figure scripts.

Every number here is a saved output of Code/Stage1_stability.ipynb or
Code/Stage1_gtex.ipynb (the October 6 runs recorded in docs/run-records.md).
The 66-pair tables are read from CSVs recomputed read-only from the cached
embeddings in Code/stage1_embeddings/; they reproduce the saved outputs.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.environ.get("STAGE1_TABLES", HERE)

# ---------------------------------------------------------------- encoders
KEYS = ["nt500m", "ntv2_100m", "dnabert2", "hyenadna",
        "rnafm", "rinalmo", "mrnafm", "calm",
        "esm2_8m", "esm2_35m", "esm2_150m", "protbert"]

# key: (label, short label, modality, token unit, embedding width)
ENCODERS = {
    "nt500m":    ("NT 500M human-ref", "NT 500M", "DNA", "6-mer", 1280),
    "ntv2_100m": ("NT v2 100M multi-species", "NT v2 100M", "DNA", "6-mer", 512),
    "dnabert2":  ("DNABERT-2", "DNABERT-2", "DNA", "byte-pair", 768),
    "hyenadna":  ("HyenaDNA large", "HyenaDNA", "DNA", "nucleotide", 256),
    "rnafm":     ("RNA-FM", "RNA-FM", "RNA", "nucleotide", 640),
    "rinalmo":   ("RiNALMo 150M", "RiNALMo", "RNA", "nucleotide", 640),
    "mrnafm":    ("mRNA-FM", "mRNA-FM", "RNA", "codon", 1280),
    "calm":      ("CaLM", "CaLM", "RNA", "codon", 768),
    "esm2_8m":   ("ESM-2 8M", "ESM-2 8M", "protein", "amino acid", 320),
    "esm2_35m":  ("ESM-2 35M", "ESM-2 35M", "protein", "amino acid", 480),
    "esm2_150m": ("ESM-2 150M", "ESM-2 150M", "protein", "amino acid", 640),
    "protbert":  ("ProtBERT", "ProtBERT", "protein", "amino acid", 1024),
}
LABEL = {k: v[0] for k, v in ENCODERS.items()}
SHORT = {k: v[1] for k, v in ENCODERS.items()}
MOD = {k: v[2] for k, v in ENCODERS.items()}
TOKEN = {k: v[3] for k, v in ENCODERS.items()}
WIDTH = {k: v[4] for k, v in ENCODERS.items()}

# ------------------------------------------------- Track A probe results
# pinned five-fold mean, fold SD, mean over ten further fold assignments
STABILITY = {
    "nt500m": (0.032, 0.043, 0.046), "ntv2_100m": (0.080, 0.032, 0.074),
    "dnabert2": (0.054, 0.031, 0.057), "hyenadna": (0.026, 0.017, 0.019),
    "rnafm": (0.052, 0.032, 0.046), "rinalmo": (0.078, 0.021, 0.079),
    "mrnafm": (0.127, 0.032, 0.110), "calm": (0.127, 0.058, 0.132),
    "esm2_8m": (0.127, 0.044, 0.131), "esm2_35m": (0.114, 0.049, 0.123),
    "esm2_150m": (0.144, 0.035, 0.142), "protbert": (0.143, 0.041, 0.135),
}
GTEX = {
    "nt500m": (0.034, 0.025, 0.039), "ntv2_100m": (0.036, 0.030, 0.041),
    "dnabert2": (0.033, 0.028, 0.047), "hyenadna": (0.031, 0.028, 0.038),
    "rnafm": (0.149, 0.037, 0.161), "rinalmo": (0.149, 0.047, 0.150),
    "mrnafm": (0.172, 0.044, 0.176), "calm": (0.120, 0.032, 0.119),
    "esm2_8m": (0.091, 0.022, 0.099), "esm2_35m": (0.104, 0.031, 0.113),
    "esm2_150m": (0.130, 0.043, 0.138), "protbert": (0.163, 0.040, 0.164),
}
STAB_LENGTH = (0.003, 0.009, 0.004)        # sequence length only
GTEX_LENGTH = (0.242, 0.036, 0.238)        # four transcript-length features
GTEX_COMPOSITION = (0.239, 0.011, 0.244)   # 92 composition features
STAB_CEILING = 0.585   # sequence-only ceiling set by repeated-sequence label noise

# ------------------------------------------------ layer-wise CKA (n = 194)
# Reported in the stability notebook's layer-wise section. Ranges span the
# ESM-2 models where a group contains more than one pair.
LAYERWISE = [
    # label, early peak (lo, hi), where, final (lo, hi), reference?
    ("ESM-2 with ESM-2", (0.982, 0.994), "state 0", (0.590, 0.815), True),
    ("RNA-FM with RiNALMo", (0.989, 0.989), "state 0", (0.834, 0.834), True),
    ("CaLM with ESM-2", (0.847, 0.874), "CaLM state 1", (0.469, 0.480), False),
    ("NT 500M with RNA-FM", (0.764, 0.764), "DNA state 14", (0.596, 0.596), False),
    ("mRNA-FM with ESM-2", (0.650, 0.689), "mRNA-FM state 2", (0.052, 0.068), False),
]
MRNAFM_MID = (0.650, 0.780)     # states 2-5, against every other encoder
MRNAFM_FINAL_CAP = 0.244        # final state, every encoder except ProtBERT
MRNAFM_FINAL_PROTBERT = 0.362
LAYERWISE_N = 194

# --------------------------------------------------------- pair groupings
DNA = ["nt500m", "ntv2_100m", "dnabert2", "hyenadna"]
NUC_RNA = ["rnafm", "rinalmo"]
CODON_RNA = ["mrnafm", "calm"]
ESM = ["esm2_8m", "esm2_35m", "esm2_150m"]
PROTEIN = ESM + ["protbert"]
_ORDER = {k: i for i, k in enumerate(KEYS)}


def pairs(a_keys, b_keys):
    out = set()
    for a in a_keys:
        for b in b_keys:
            if a == b:
                continue
            x, y = sorted([a, b], key=_ORDER.get)
            out.add(f"{x}|{y}")
    return sorted(out, key=lambda p: (_ORDER[p.split("|")[0]], _ORDER[p.split("|")[1]]))


GROUPS = {
    "Within ESM-2": pairs(ESM, ESM),
    "ProtBERT with ESM-2": pairs(["protbert"], ESM),
    "Within DNA": pairs(DNA, DNA),
    "RNA-FM with RiNALMo": pairs(NUC_RNA, NUC_RNA),
    "mRNA-FM with CaLM": pairs(["mrnafm"], ["calm"]),
    "CaLM with protein": pairs(["calm"], PROTEIN),
    "mRNA-FM with protein": pairs(["mrnafm"], PROTEIN),
    "RNA-FM/RiNALMo with CaLM": pairs(NUC_RNA, ["calm"]),
    "RNA-FM/RiNALMo with mRNA-FM": pairs(NUC_RNA, ["mrnafm"]),
    "RNA-FM/RiNALMo with protein": pairs(NUC_RNA, PROTEIN),
    "DNA with RNA-FM/RiNALMo": pairs(DNA, NUC_RNA),
    "DNA with CaLM": pairs(DNA, ["calm"]),
    "DNA with mRNA-FM": pairs(DNA, ["mrnafm"]),
    "DNA with protein": pairs(DNA, PROTEIN),
}


def pair_kind(pair):
    """Coarse pair type used to colour the fusion-gain scatter."""
    a, b = pair.split("|")
    ma, mb = MOD[a], MOD[b]
    if ma == mb:
        return "within modality"
    if "DNA" in (ma, mb):
        return "DNA with RNA or protein"
    codon = {a, b} & set(CODON_RNA)
    if codon and "protein" in (ma, mb):
        return "codon RNA with protein"
    return "nucleotide RNA with protein"


# ------------------------------------------------------------ CSV tables
def load():
    """Return (stability, gtex) dicts of DataFrames for the 66-pair tables."""
    out = {}
    for name in ("stab", "gtex"):
        geom = pd.read_csv(f"{TABLES}/{name}_geom.csv", index_col=0)
        gain = pd.read_csv(f"{TABLES}/{name}_gain.csv").set_index("pair")
        share = pd.read_csv(f"{TABLES}/{name}_share.csv", index_col=0).iloc[:, 0]
        geom["gain"] = gain.loc[geom.index, "mean_gain"]
        geom["n_pos"] = gain.loc[geom.index, "n_pos"]
        geom["kind"] = [pair_kind(p) for p in geom.index]
        out[name] = {"pairs": geom, "share": share}
    return out["stab"], out["gtex"]
