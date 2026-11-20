"""Exercise Git byte preservation with invented files in temporary repositories.

These checks never add to the project's index, create commits, inspect research
arrays or alter results. LF index fixtures model the existing canonical blobs;
the returned archives instead retain their exact original bytes in Git.
"""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
GIT = shutil.which("git")
CHECKOUT_SETTINGS = (
    ("true", "lf"), ("true", "crlf"),
    ("false", "lf"), ("false", "crlf"),
    ("input", "lf"), ("input", "crlf"),
)


@unittest.skipUnless(GIT, "Git is required for checkout-byte checks")
class GitEvidenceBytesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix=".git-evidence-", dir=ROOT / "tests")
        self.addCleanup(temporary.cleanup)
        self.repository = Path(temporary.name).resolve()
        # An inherited worktree/index or personal filter must never redirect a
        # synthetic Git operation into this project's metadata.
        self.environment = {
            key: value for key, value in os.environ.items()
            if not key.startswith("GIT_")
        }
        self.environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
        self.git("init", "--quiet")
        self.put(".gitattributes", (ROOT / ".gitattributes").read_bytes())
        self.git("add", "--", ".gitattributes", autocrlf="false")

    def git(self, *arguments, data=None, autocrlf="false", eol="lf"):
        result = subprocess.run(
            [GIT, "-c", f"core.autocrlf={autocrlf}", "-c", f"core.eol={eol}",
             "-c", "core.safecrlf=false", "-c", "core.longpaths=true", *arguments],
            cwd=self.repository, env=self.environment, input=data,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        )
        return result.stdout

    def put(self, name, content):
        path = self.repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def seed_index_blob(self, name, content):
        """Install an invented pre-existing blob without any text conversion."""
        object_id = self.git("hash-object", "-w", "--stdin", data=content).decode().strip()
        self.git("update-index", "--add", "--cacheinfo", f"100644,{object_id},{name}")
        return object_id

    def check_checkout_bytes(self, expected):
        for autocrlf, eol in CHECKOUT_SETTINGS:
            with self.subTest(autocrlf=autocrlf, eol=eol):
                # Removing only these invented working files forces checkout to
                # apply the configured policy rather than reuse a cached stat.
                for name in expected:
                    (self.repository / name).unlink(missing_ok=True)
                self.git("checkout-index", "--all", "--force", autocrlf=autocrlf, eol=eol)
                for name, content in expected.items():
                    with self.subTest(path=name):
                        actual = (self.repository / name).read_bytes()
                        self.assertEqual(actual, content)
                        self.assertEqual(hashlib.sha256(actual).digest(),
                                         hashlib.sha256(content).digest())

    def test_returned_evidence_retains_mixed_original_bytes_in_blobs_and_checkouts(self):
        evidence = {
            "Code/stage4/results/newton_h100_seeds012/outputs/phaseA/fit.json":
                b'{\r\n  "synthetic": true\r\n}\r\n',
            "Code/stage4/results/newton_h100_seeds012/provenance/sources/analysis.py":
                b"# original mixed endings\r\nvalue = 1\n",
            "Code/stage4/results/newton_h100_seeds012/provenance/old.sbatch.example":
                b"#!/bin/bash\r\n# historical evidence, not the maintained launcher\r\n",
            "Code/stage4/results/decision_inspections/fixture/executed.ipynb":
                b'{"cells": []}\n',
            "Code/stage4/results/region_aware_v1/fixture/input-audit-v1.json":
                b'{"fixture": true}\r\n',
        }
        for name, content in evidence.items():
            self.put(name, content)
        for autocrlf in ("true", "false", "input"):
            self.git("add", "--", *evidence, autocrlf=autocrlf)
            for name, content in evidence.items():
                self.assertEqual(self.git("show", f":{name}"), content)
        self.check_checkout_bytes(evidence)

    def test_historical_lf_blobs_checkout_as_recorded_crlf_without_blob_changes(self):
        names = (
            "Code/stage3/results/fungal_expression/fungal_expression__concat__lambda0.0__ch100__equation__seed0.json",
            "Code/stage3/results/fungal_expression/fungal_expression__concat__lambda0.0__ch100__equation__seed1.json",
            "Code/stage3/results/fungal_expression/records/equation-verification-20261020T034236791322Z.json",
            "Code/stage4/results/phaseA/ecoli_proteins/ecoli_proteins__mil__lambda0.01__ch100__equation__seed0.json",
            "Code/stage4/results/phaseA/ecoli_proteins/ecoli_proteins__mil__lambda0.01__ch100__equation__seed2.json",
            "Code/stage4/results/phaseB/cov_vac_degradation/calm/cov_vac_degradation__concat__lambda0.0__ch100__equation__seed0.json",
            "Code/stage4/results/phaseB/analyses/paired-report-v1/cov_vac_degradation__9c0f130491382c1bc2ad.json",
        )
        content = b'{\n  "synthetic": true\n}\n'
        for name in names:
            self.seed_index_blob(name, content)
        self.check_checkout_bytes({name: content.replace(b"\n", b"\r\n") for name in names})
        for name in names:
            self.assertEqual(self.git("show", f":{name}"), content)
            self.git("add", "--", name, autocrlf="true")
            self.assertEqual(self.git("show", f":{name}"), content)

    def test_original_lf_run_records_keep_original_bytes(self):
        names = (
            "Code/stage4/results/phaseB/runs/cov-vac-mrnafm-seed0/cov_vac_degradation/mrnafm/cov_vac_degradation__concat__lambda0.0__ch100__equation__seed0.json",
            "Code/stage4/results/phaseB/runs/cov-vac-calm-seed0/cov_vac_degradation/calm/cov_vac_degradation__concat__lambda0.0__ch100__equation__seed0.json",
        )
        content = b'{"synthetic": true}\n'
        for name in names:
            self.put(name, content)
            self.git("add", "--", name, autocrlf="true")
            self.assertEqual(self.git("show", f":{name}"), content)
        self.check_checkout_bytes({name: content for name in names})

    def test_future_stage4_seeds_and_reports_keep_their_original_lf_bytes(self):
        names = (
            "Code/stage4/results/phaseA/fungal_expression/new_model__seed0.json",
            "Code/stage4/results/phaseA/fungal_expression/new_model__seed1.json",
            "Code/stage4/results/phaseA/ecoli_proteins/ecoli_proteins__mil__lambda0.01__ch99__equation__seed2.json",
            "Code/stage4/results/phaseB/fixture/new_encoder/fit__seed1.json",
            "Code/stage4/results/phaseB/analyses/paired-report-v1/new_report.json",
        )
        content = b'{"future_fixture": true}\n'
        for name in names:
            self.put(name, content)
        self.git("add", "--", *names, autocrlf="true")
        for name in names:
            self.assertEqual(self.git("show", f":{name}"), content)
        self.check_checkout_bytes({name: content for name in names})

    def test_four_canonical_csv_lf_blobs_checkout_as_recorded_crlf(self):
        names = (
            "datasets/CoV_Vaccine_Degradation.csv", "datasets/E.Coli_proteins.csv",
            "datasets/Fungal_expression.csv", "datasets/mRNA_Stability.csv",
        )
        content = b"Sequence,Value\nACGT,1\n"
        for name in names:
            self.seed_index_blob(name, content)
        self.check_checkout_bytes({name: content.replace(b"\n", b"\r\n") for name in names})
        self.git("add", "--", *names, autocrlf="false")
        for name in names:
            self.assertEqual(self.git("show", f":{name}"), content)

    def test_maintained_slurm_scripts_store_and_checkout_lf(self):
        names = (
            "execution/arcc/encoder_swaps/encoder_swaps_preflight.sbatch",
            "execution/arcc/encoder_swaps/newton_ernie.sbatch.example",
            "execution/arcc/encoder_swaps/newton_report.sbatch.example",
            "execution/arcc/encoder_swaps/newton_swaps.sbatch.example",
            "execution/arcc/grouped_stability/grouped_stability_preflight.sbatch",
            "execution/arcc/grouped_stability/newton_grouped_stability.sbatch.example",
            "execution/arcc/fusion_studies/fusion_studies_preflight.sbatch",
            "execution/arcc/fusion_studies/newton_fusion_study.sbatch.example",
        )
        content = b"#!/bin/bash\n# synthetic fixture only\n"
        for name in names:
            self.put(name, content.replace(b"\n", b"\r\n"))
        self.git("add", "--", *names, autocrlf="true")
        for name in names:
            self.assertEqual(self.git("show", f":{name}"), content)
        self.check_checkout_bytes({name: content for name in names})

    def test_unrelated_source_and_data_have_no_new_attributes(self):
        names = (
            "README.md", "Code/shared_code/analysis.py", "Code/stage1/example.ipynb",
            "Code/stage3/results/other/fit.json", "datasets/unrelated.csv",
            "Code/stage3/results/fungal_expression/new_model__seed0.json",
            "Code/stage3/results/fungal_expression/new_model__seed1.json",
        )
        output = self.git("check-attr", "-z", "text", "eol", "--", *names)
        fields = output.rstrip(b"\0").split(b"\0")
        self.assertEqual(len(fields), len(names) * 2 * 3)
        for index in range(0, len(fields), 3):
            self.assertEqual(fields[index + 2], b"unspecified", fields[index].decode())


if __name__ == "__main__":
    unittest.main()
