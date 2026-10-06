"""Check GTEx coordinate conversion without data, model, or network access."""

from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Code"))

import pandas as pd  # noqa: E402

# Import the builder as a module: its __main__ path must not run. Guard the I/O
# boundaries as well so this test cannot initiate research work during import.
with patch("urllib.request.urlretrieve", side_effect=AssertionError("No downloads")), \
     patch("urllib.request.urlopen", side_effect=AssertionError("No requests")), \
     patch.object(pd, "read_csv", side_effect=AssertionError("No research CSV reads")):
    import build_gtex_pilot as builder  # noqa: E402


def original_regions(rows, window):
    """The pre-extraction expressions; complements independent known answers."""
    plus = rows["strand"] == "+"
    tss = (rows["start"] - 1).where(plus, rows["end"] - 1)
    lo = (tss - window // 2).where(plus, tss - (window - window // 2) + 1)
    hi = (tss + (window - window // 2)).where(plus, tss + window // 2 + 1)
    regions = [
        f"{c.replace('chr', '')}:{a + 1}..{b}:{1 if s == '+' else -1}"
        for c, a, b, s in zip(rows["chr"], lo, hi, rows["strand"])
    ]
    return tss, regions


class TranscriptRegionTests(unittest.TestCase):
    def rows(self):
        return pd.DataFrame({
            "chr": ["chr1", "chr2"],
            "start": [101, 101],
            "end": [200, 200],
            "strand": ["+", "-"],
        }, index=[17, 4])

    def test_even_window_hand_worked_endpoints(self):
        tss, regions = builder.transcript_regions(self.rows(), window=6)
        pd.testing.assert_series_equal(tss, pd.Series([100, 199], index=[17, 4], name="start"))
        self.assertEqual(regions, ["1:98..103:1", "2:198..203:-1"])

    def test_odd_window_hand_worked_endpoints(self):
        tss, regions = builder.transcript_regions(self.rows(), window=5)
        self.assertEqual(tss.tolist(), [100, 199])
        self.assertEqual(regions, ["1:99..103:1", "2:198..202:-1"])

    def test_requested_length_and_oriented_tss_index(self):
        rows = self.rows()
        for window in [1, 2, 5, 6, 6000, 6001]:
            with self.subTest(window=window):
                tss, regions = builder.transcript_regions(rows, window=window)
                for region, tss_zero_based in zip(regions, tss):
                    match = re.fullmatch(r"[^:]+:(-?\d+)\.\.(-?\d+):(-?1)", region)
                    self.assertIsNotNone(match)
                    low, high, strand = map(int, match.groups())
                    self.assertEqual(high - low + 1, window)
                    tss_one_based = tss_zero_based + 1
                    # A reverse-strand response starts at high, then decreases.
                    offset = tss_one_based - low if strand == 1 else high - tss_one_based
                    self.assertEqual(offset, window // 2)

    def test_default_window_remains_6000(self):
        rows = self.rows().assign(start=10001, end=12000)
        tss, regions = builder.transcript_regions(rows)
        self.assertEqual(builder.WINDOW, 6000)
        self.assertEqual(tss.tolist(), [10000, 11999])
        self.assertEqual(regions, ["1:7001..13000:1", "2:9001..15000:-1"])

    def test_chromosome_boundary_requests_are_not_clamped(self):
        rows = self.rows().assign(start=1, end=2)
        tss, regions = builder.transcript_regions(rows, window=6)
        self.assertEqual(tss.tolist(), [0, 1])
        self.assertEqual(regions, ["1:-2..3:1", "2:0..5:-1"])
        # No chromosome lengths are supplied, so upper endpoints remain as-is.
        _, reverse_regions = builder.transcript_regions(rows.iloc[[1]].assign(end=250), window=6)
        self.assertEqual(reverse_regions, ["2:248..253:-1"])

    def test_rows_index_and_unrelated_fields_are_preserved(self):
        rows = self.rows().assign(transcript_id=["tx-b", "tx-a"], label=[0.25, 0.75])
        before = rows.copy(deep=True)
        tss, _ = builder.transcript_regions(rows, window=5)
        pd.testing.assert_frame_equal(rows, before)
        self.assertEqual(tss.index.tolist(), [17, 4])

    def test_representative_rows_match_original_expressions(self):
        rows = pd.DataFrame({
            "chr": ["chr1", "chrX", "7", "chrMT", "chrchr2"],
            "start": [1, 500, 10001, 15, 42],
            "end": [90, 750, 12000, 140, 99],
            "strand": ["+", "-", "+", "-", "?"],
        }, index=[9, 3, 8, 8, 2])
        # Filtering and strand validation remain the existing caller's concern.
        for window in [1, 5, 6, 6000, 6001]:
            with self.subTest(window=window):
                expected_tss, expected_regions = original_regions(rows, window)
                tss, regions = builder.transcript_regions(rows, window)
                pd.testing.assert_series_equal(tss, expected_tss)
                self.assertEqual(regions, expected_regions)
                self.assertEqual(len(regions), len(rows))

    def test_empty_rows_preserve_empty_series_and_region_list(self):
        rows = self.rows().iloc[:0]
        expected_tss, _ = original_regions(rows, 6)
        tss, regions = builder.transcript_regions(rows, 6)
        pd.testing.assert_series_equal(tss, expected_tss)
        self.assertEqual(regions, [])


if __name__ == "__main__":
    unittest.main()
