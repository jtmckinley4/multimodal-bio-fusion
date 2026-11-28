"""Check encoder-swap figures using only synthetic saved-report summaries."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from shared_code.fusion import swap_figures


FUNGAL = "fungal_expression"
ECOLI = "ecoli_proteins"
LABELS = {
    "first": "Fixture A",
    "second": "Fixture B",
    "third": "Fixture C",
    "fourth": "Fixture D",
    "fifth": "Fixture E",
}
INTERVAL_CASES = (
    ("first", 0.2, 0.1, 0.3, (), ""),
    ("second", -0.2, -0.3, -0.1, (), ""),
    ("third", 0.05, 0.0, 0.1, (), ""),
    ("fourth", 0.08, None, None, (), "undefined resampled score"),
    ("fifth", None, None, None, (2,), ""),
)


def summary_row(encoder, kind, comparison, first, second, difference, low, high, *,
                missing_seeds=(), interval_issue=""):
    return {
        "encoder": encoder,
        "kind": kind,
        "comparison": comparison,
        "first": f"{encoder}/{first}",
        "second": second,
        "difference": difference,
        "ci_low": low,
        "ci_high": high,
        "interval_issue": interval_issue,
        "status": "incomplete" if missing_seeds else "complete",
        "available_seeds": [seed for seed in (0, 1, 2) if seed not in missing_seeds],
        "missing_seeds": list(missing_seeds),
    }


def saved_report(benchmark, units, rows):
    return {
        "identity": {"benchmark": benchmark, "seeds": [0, 1, 2]},
        "analysis": {"units": units, "seed_summary": rows, "per_seed": [], "issues": []},
    }


class SwapFigureTests(unittest.TestCase):
    def figure(self, plot, reports, labels=LABELS):
        before = deepcopy((reports, labels))
        figure = plot(reports, labels)
        self.addCleanup(plt.close, figure)
        self.assertIsInstance(figure, Figure)
        self.assertEqual([axis.get_title() for axis in figure.axes],
                         ["CoV-Vac", "Fungal expression", "E. coli", "mRNA stability"])
        self.assertEqual((reports, labels), before)
        return figure

    def test_native_and_fusion_grids_select_their_own_contrasts_and_units(self):
        def rows(scale):
            return [
                summary_row("first", "representation", "replacement single vs original single",
                            "unimodal:rna", "original/unimodal:rna",
                            0.123 * scale, 0.1 * scale, 0.15 * scale),
                summary_row("first", "complementarity", "concat vs validation-selected single",
                            "concat", "first/unimodal-grid:rna",
                            -0.067 * scale, -0.1 * scale, -0.05 * scale),
                summary_row("first", "replacement", "concat vs original concat",
                            "concat", "original/concat",
                            0.9 * scale, 0.8 * scale, 0.95 * scale),
                summary_row("first", "method", "cross vs concat", "cross", "first/concat",
                            0.4 * scale, 0.39 * scale, 0.41 * scale),
            ]

        reports = {
            FUNGAL: saved_report(FUNGAL, "Spearman", rows(1)),
            ECOLI: saved_report(ECOLI, "accuracy percentage points", rows(100)),
        }
        for plot, expected in (
            (swap_figures.plot_native_swap_gains, ("+0.123", "+12.30")),
            (swap_figures.plot_fusion_gains, ("-0.067", "-6.70")),
        ):
            with self.subTest(plot=plot.__name__):
                figure = self.figure(plot, reports, {"first": LABELS["first"]})
                for axis, result, units in zip((figure.axes[1], figure.axes[2]), expected,
                                                ("Spearman", "accuracy percentage points")):
                    values = [text.get_text().replace("−", "-") for text in axis.texts]
                    deltas = {value for value in values if value.startswith(("+", "-"))}
                    self.assertEqual(deltas, {result})
                    self.assertIn(units.lower(), axis.get_xlabel().lower())

    def test_native_grid_distinguishes_interval_statuses_without_inventing_missing_scores(self):
        rows = [
            summary_row(encoder, "representation", "replacement single vs original single",
                        "unimodal:rna", "original/unimodal:rna", difference, low, high,
                        missing_seeds=missing, interval_issue=issue)
            for encoder, difference, low, high, missing, issue in INTERVAL_CASES
        ]
        figure = self.figure(swap_figures.plot_native_swap_gains,
                             {FUNGAL: saved_report(FUNGAL, "Spearman", rows)})
        # The leftmost panel displays the encoder labels shared across panels.
        row_labels = [label.get_text() for label in figure.axes[0].get_yticklabels()]
        self.assertEqual(row_labels, list(LABELS.values()))
        axis = figure.axes[1]
        plotted = [text.get_text().replace("−", "-") for text in axis.texts]
        self.assertEqual(set(plotted), {"+0.200", "-0.200", "+0.050", "+0.080", "—"})

        image, = axis.images
        rendered_colors = image.to_rgba(image.get_array())
        legend, = figure.legends
        colors = {
            label.get_text(): handle.get_facecolor()
            for label, handle in zip(legend.get_texts(), legend.legend_handles)
        }
        statuses = ["Above zero", "Below zero", "Includes zero", "No interval", "Missing pair"]
        self.assertEqual(len({colors[status] for status in statuses}), len(statuses))
        self.assertEqual([tuple(color) for color in rendered_colors.reshape(-1, 4)],
                         [colors[status] for status in statuses])

    def test_head_intervals_keep_direction_unavailability_and_missing_pairs_distinct(self):
        rows = [
            summary_row(encoder, "method", f"{head} vs concat", head, f"{encoder}/concat",
                        difference, low, high, missing_seeds=missing, interval_issue=issue)
            for (encoder, difference, low, high, missing, issue), head
            in zip(INTERVAL_CASES, ("mil", "mil", "cross", "cross", "mil"))
        ]
        rows.append(summary_row("fifth", "complementarity", "mil vs validation-selected single",
                                "mil", "fifth/unimodal:rna", 0.6, 0.5, 0.7))
        figure = self.figure(swap_figures.plot_head_comparisons,
                             {FUNGAL: saved_report(FUNGAL, "Spearman", rows)})
        axis = figure.axes[1]
        points = {line.get_xdata()[0]: line for line in axis.lines if len(line.get_xdata()) == 1}
        self.assertEqual(set(points), {0.2, -0.2, 0.05, 0.08})
        self.assertTrue(any(tuple(line.get_xdata()) == (0, 0) for line in axis.lines))
        for difference in (0.2, -0.2):
            point = points[difference]
            self.assertEqual(to_rgba(point.get_markerfacecolor()), to_rgba(point.get_markeredgecolor()))
        unclear = points[0.05]
        self.assertNotEqual(to_rgba(unclear.get_markerfacecolor()), to_rgba(unclear.get_markeredgecolor()))
        self.assertEqual(points[0.08].get_marker(), "x")
        self.assertTrue(any("missing pair" in text.get_text().lower() for text in axis.texts))

        bounds = {
            (segment[0][0], segment[1][0])
            for artist in axis.collections if isinstance(artist, LineCollection)
            for segment in artist.get_segments()
        }
        self.assertEqual(bounds, {(0.1, 0.3), (-0.3, -0.1), (0.0, 0.1)})
        self.assertIn("Spearman", axis.get_xlabel())


if __name__ == "__main__":
    unittest.main()
