"""Build phaseB_readout.json for make_phaseB_full_figure.py from the saved paired reports.

Sources: Code/stage4/results/phaseB/analyses/paired-report-v1/*.json (Chase's saved-prediction
report, seed 0, 2,000 sequence-grouped draws) for every head score, best-single reference and
interval; each swapped encoder's native single-encoder test score from its seed-0 fit JSON for
the five stability swaps the earlier readout lacked. Every value already in the earlier readout
is checked against the report and must agree.

    python build_phaseB_readout.py <old_readout.json> <report_dir> <out.json>
"""
import glob, json, os, sys

old = json.load(open(sys.argv[1]))
reports = {os.path.basename(f).split("__")[0]: json.load(open(f))["analysis"]
           for f in glob.glob(os.path.join(sys.argv[2], "*.json"))}
# Native single-encoder test scores, seed 0, from results/phaseB/mrna_stability/<encoder>/.
NEW_ALONE = {"nt500m": 0.5368, "hyenadna": 0.5229, "rinalmo": 0.5600, "esm2_35m": 0.5352,
             "protbert": 0.5427}
HEADS = ("concat", "mil", "cross")
out = json.loads(json.dumps(old))
checked = added = 0
for b, a in reports.items():
    scale_digits = 1 if b == "ecoli_proteins" else 3
    rows = {}
    for r in a["per_seed"]:
        if r["seed"] != 0 or r["kind"] not in ("complementarity", "replacement"):
            continue
        head = r["comparison"].split(" vs ")[0]
        rows[(r["encoder"], head, "best" if r["kind"] == "complementarity" else "original")] = r
    for k in sorted({e for e, _, _ in rows}):
        iv = {h: {} for h in HEADS}
        tab = {}
        for h in HEADS:
            for which in ("best", "original"):
                r = rows[(k, h, which)]
                iv[h][which] = {"d": round(r["difference"], 3), "lo": round(r["ci_low"], 3),
                                "hi": round(r["ci_high"], 3), "p": round(r["first_wins"], 3),
                                "resolved": bool(r["ci_low"] > 0 or r["ci_high"] < 0),
                                "ref": r["second"].split("/", 1)[1] if which == "best" else None}
            best = rows[(k, h, "best")]
            tab[h] = round(best["first_score"], scale_digits if b == "ecoli_proteins" else 3)
            tab[f"{h}_d"] = round(best["difference"], scale_digits if b == "ecoli_proteins" else 3)
            tab["best_single"] = best["second"].split("/", 1)[1]
            tab["best_score"] = round(best["second_score"], scale_digits if b == "ecoli_proteins" else 3)
        if k in old["intervals"].get(b, {}):
            for h in HEADS:
                for which in ("best", "original"):
                    o, n = old["intervals"][b][k][h][which], iv[h][which]
                    assert all(abs(o[f] - n[f]) < 1e-9 for f in ("d", "lo", "hi")) and o["resolved"] == n["resolved"], (b, k, h, which, o, n)
                    checked += 1
        else:
            assert b == "mrna_stability" and k in NEW_ALONE, (b, k)
            out["intervals"].setdefault(b, {})[k] = iv
            out["table"][b][k] = {"alone": NEW_ALONE[k], **tab}
            added += 1
json.dump(out, open(sys.argv[3], "w"), indent=1)
print(f"checked {checked} existing intervals against the report: all agree; added {added} swaps")
