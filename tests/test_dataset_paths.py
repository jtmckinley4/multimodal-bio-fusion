"""Check dataset directory defaults and explicit overrides without research data."""

from contextlib import contextmanager
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Code"))

# Preserve the notebook environment's NumPy-before-PyTorch import order on Windows.
import numpy as np  # noqa: F401, E402
import pandas as pd  # noqa: E402
from shared_code import datasets  # noqa: E402


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


class DatasetPathTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame({
            "Sequence": ["ATGAAACCCGGGTTT", "ATGCCCGGGAAATTT"],
            "Value": [1.0, 2.0],
            "Split": ["train", "test"],
        })

    def check_readers(self, **options):
        key = "mrna_stability"
        expected = self.fixture()
        pd.testing.assert_frame_equal(datasets.sample_rows(key, **options), expected)
        loaded = datasets.load_dataset(key, **options)
        pd.testing.assert_frame_equal(loaded.rows, expected)
        train, test = datasets.official_split_sample(key, 1, 1, **options)
        pd.testing.assert_frame_equal(train.rows, expected.iloc[:1])
        pd.testing.assert_frame_equal(test.rows, expected.iloc[1:])
        self.assertEqual(datasets.audit(key, **options)["Rows"], 2)

    def test_defaults_target_repository_datasets_from_each_working_directory(self):
        # Intercept the file boundary so this check needs no research CSVs.
        expected_path = REPO / "datasets" / "mRNA_Stability.csv"
        with tempfile.TemporaryDirectory() as outside:
            for cwd in (REPO, REPO / "Code", Path(outside)):
                with self.subTest(cwd=cwd), working_directory(cwd):
                    def read_fixture(path, *args, **kwargs):
                        self.assertEqual(Path(path), expected_path)
                        return self.fixture()
                    with patch.object(datasets.pd, "read_csv", side_effect=read_fixture) as read:
                        self.check_readers()
                        self.assertEqual(read.call_count, 4)

    def test_explicit_absolute_and_relative_directories_read_the_supplied_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            supplied = root / "inputs"
            supplied.mkdir()
            self.fixture().to_csv(supplied / "mRNA_Stability.csv", index=False)
            with working_directory(root):
                for data_dir in (supplied, "inputs"):
                    with self.subTest(data_dir=data_dir):
                        self.check_readers(data_dir=data_dir)


if __name__ == "__main__":
    unittest.main()
