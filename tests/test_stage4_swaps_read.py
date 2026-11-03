"""Exercise encoder-swap reading cells against fixtures, with producers blocked."""

from contextlib import redirect_stdout
import datetime
import io
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from shared_code.fusion import swap_plan, swap_results


RECIPE = dict(lr=3e-5, weight_decay=1e-5, batch_size=32, max_epochs=500,
              early_stopping_patience=20, plateau_patience=5, entropy_lambda=0.0,
              channels=100, tconv="equation", seed=0)
ORIGINAL = dict(dna="ntv2_100m", rna="rnafm", prot="esm2_8m")


def forbidden(*args, **kwargs):
    raise AssertionError("Read-saved mode attempted computation or publication")


class SwapNotebookReadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.code = self.root / "Code"
        self.code.mkdir()
        self.cells = json.loads((ROOT / "Code/stage4/tier2_encoder_swaps.ipynb").read_text(
            encoding="utf-8"))["cells"]

    def result(self, relative, variant, **changes):
        path = self.code / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(benchmark="cov_vac_degradation", variant=variant,
                    config={**RECIPE, "entropy_lambda": 0.01 if variant == "mil" else 0.0},
                    epochs=4, best_epoch=2, seconds=1.0, val_score=0.4, test_score=0.5)
        data.update(changes)
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def execute_reading(self):
        labels = (*ORIGINAL.values(), "nt500m", "hyenadna", "rinalmo", "mrnafm",
                  "calm", "esm2_35m", "esm2_150m", "protbert")
        shown = []
        namespace = dict(
            CODE_DIR=self.code, DEVICE="cpu", Path=Path, np=np, pd=pd, json=json,
            datetime=datetime, time=time, display=shown.append,
            ENCODERS={key: SimpleNamespace(label=key) for key in labels},
            datasets=SimpleNamespace(blf_benchmark_rows=forbidden),
            blf=SimpleNamespace(ENCODER_OF_MODALITY=ORIGINAL,
                                BenchmarkStreams=SimpleNamespace(open=forbidden)),
            tokens=SimpleNamespace(DEFAULT_CHUNK=256, extract=forbidden,
                                   inspect_cache=forbidden, tokens_for_length=forbidden,
                                   CacheVerification=forbidden,
                                   grid_resolution=lambda k: "codon" if k in
                                   {"mrnafm", "calm", "esm2_8m", "esm2_35m",
                                    "esm2_150m", "protbert"} else "nucleotide"),
            train=SimpleNamespace(
                TrainConfig=lambda **kw: SimpleNamespace(**{**RECIPE, **kw}),
                Result=SimpleNamespace(load=lambda p: SimpleNamespace(
                    **json.loads(Path(p).read_text(encoding="utf-8")))),
                run=forbidden, fit=forbidden, seed_averaged_bootstrap=forbidden),
            swap_results=swap_results, swap_plan=swap_plan,
        )
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        output = io.StringIO()
        # Imports/device discovery are not research; inject their interfaces so this
        # test needs neither pretrained models nor a particular PyTorch installation.
        omitted = {"swaps-imports", "swaps-paths", "1a72c669"}
        with redirect_stdout(output), patch.object(pd.DataFrame, "to_csv", forbidden), \
                patch.object(Path, "mkdir", forbidden), patch.object(Path, "write_text", forbidden):
            for cell in self.cells:
                if cell["cell_type"] == "code" and cell["id"] not in omitted:
                    exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)
        self.assertTrue(namespace["READ_SAVED_RESULTS"])
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})
        self.assertEqual(namespace["STREAMS"], {})
        self.assertEqual(namespace["RESOLVED"], {})
        return namespace, shown, output.getvalue()

    def test_empty_checkout_reports_gaps_and_finishes_without_producers(self):
        namespace, shown, output = self.execute_reading()
        self.assertIn("Training skipped", output)
        self.assertIn("Stage 1 alignment unavailable", output)
        self.assertIn("Paired intervals not recalculated", output)
        self.assertTrue(namespace["result_coverage"]["missing"].str.len().gt(0).all())

    def test_existing_scores_read_without_caches_and_without_recalculation(self):
        for variant in ("unimodal:dna", "unimodal:rna", "unimodal:prot", "concat", "mil", "cross"):
            self.result(f"stage4/results/phaseA/cov_vac_degradation/{variant.replace(':', '-')}.json",
                        variant)
        for variant in ("unimodal:dna", "unimodal-grid:dna", "concat", "mil", "cross"):
            self.result(f"stage4/results/phaseB/cov_vac_degradation/nt500m/{variant.replace(':', '-')}.json",
                        variant, test_score=0.7)
        namespace, shown, output = self.execute_reading()
        table = namespace["summary"]("cov_vac_degradation")
        self.assertAlmostEqual(table.loc[("dna", "nt500m"), "concat"], 0.7)
        self.assertIn("metadata unavailable", str(namespace["checks"]))

    def test_ambiguous_or_malformed_records_report_without_replacement(self):
        for name in ("first", "duplicate"):
            self.result(f"stage4/results/phaseA/cov_vac_degradation/{name}.json", "mil")
        path = self.result("stage4/results/phaseB/cov_vac_degradation/nt500m/broken.json", "concat")
        path.write_text("not JSON", encoding="utf-8")
        namespace, shown, output = self.execute_reading()
        self.assertIn("seed-0 MIL selection unavailable", str(namespace["status"]))
        self.assertIn("Ambiguous saved results", str(namespace["FIT_PLANS"]["cov_vac_degradation"].issues))
        self.assertFalse(namespace["result_issues"].empty)

    def test_missing_mil_selection_does_not_hide_independent_scores(self):
        for variant in ("unimodal:dna", "unimodal:rna", "unimodal:prot", "concat"):
            self.result(f"stage4/results/phaseA/cov_vac_degradation/{variant.replace(':', '-')}.json",
                        variant)
        self.result("stage4/results/phaseB/cov_vac_degradation/nt500m/concat.json", "concat",
                    test_score=0.7)
        namespace, _, _ = self.execute_reading()
        table = namespace["summary"]("cov_vac_degradation")
        self.assertEqual(table.loc[("none", "BioLangFusion's own"), "concat"], 0.5)
        self.assertEqual(table.loc[("dna", "nt500m"), "concat"], 0.7)
        self.assertTrue(np.isnan(table.loc[("dna", "nt500m"), "mil"]))

    def test_duplicated_variant_does_not_hide_unique_scores(self):
        for name in ("first", "duplicate"):
            self.result(f"stage4/results/phaseA/cov_vac_degradation/{name}.json", "mil")
        self.result("stage4/results/phaseA/cov_vac_degradation/concat.json", "concat")
        namespace, _, _ = self.execute_reading()
        table = namespace["summary"]("cov_vac_degradation")
        self.assertEqual(table.loc[("none", "BioLangFusion's own"), "concat"], 0.5)
        self.assertTrue(np.isnan(table.loc[("none", "BioLangFusion's own"), "mil"]))


if __name__ == "__main__":
    unittest.main()
