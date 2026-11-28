"""
Draw encoder-swap comparisons from the saved Newton three-seed reports.

The notebook loads and verifies the reports. These functions use the recorded
differences and interval bounds directly and return figures for display.
"""

from collections import Counter
from math import isfinite


NEWTON_BENCHMARKS = {
    "cov_vac_degradation": "CoV-Vac",
    "fungal_expression": "Fungal expression",
    "ecoli_proteins": "E. coli",
    "mrna_stability": "mRNA stability",
}

_TRAINING_SEEDS = (0, 1, 2)
_INTERVAL_COLORS = {
    "Above zero": "#B9D9F2",
    "Below zero": "#F5C4AA",
    "Includes zero": "#E2E5E9",
    "No interval": "#F6E5B8",
    "Missing pair": "#F8F8F8",
}
_HEAD_COMPARISONS = (
    ("cross vs concat", "Cross-attention", "Cross", "#326BA5", -0.17),
    ("mil vs concat", "MIL gated fusion", "MIL", "#9A4F22", 0.17),
)


def plot_native_swap_gains(reports, encoder_labels):
    """Show replacement native single minus original native single."""
    return _plot_gain_grid(
        reports, encoder_labels, kind="representation",
        columns=(("replacement single vs original single", ""),),
        title="Native replacement minus original single encoder",
    )


def plot_fusion_gains(reports, encoder_labels):
    """Show each fusion head minus its validation-selected single encoder."""
    return _plot_gain_grid(
        reports, encoder_labels, kind="complementarity",
        columns=(
            ("concat vs validation-selected single", "Concat"),
            ("mil vs validation-selected single", "MIL"),
            ("cross vs validation-selected single", "Cross-attn."),
        ),
        title="Fusion head minus validation-selected single encoder",
        show_counts=True,
    )


def plot_head_comparisons(reports, encoder_labels):
    """Show paired intervals for each head minus same-input concatenation."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    fig, axes = _benchmark_panels(
        encoder_labels, "Fusion head minus same-input concatenation", bottom=0.34,
    )
    for ax, benchmark in zip(axes, NEWTON_BENCHMARKS):
        comparisons = _summary_rows(reports, benchmark, "method")
        ax.axvline(0, color="#6B7280", linewidth=1)
        count_lines = ["Ahead / includes zero / behind"]
        for comparison, _, short_label, color, offset in _HEAD_COMPARISONS:
            counts = Counter()
            for index, encoder in enumerate(encoder_labels):
                difference, low, high, status = _interval(comparisons.get((encoder, comparison)))
                counts[status] += 1
                y = index + offset
                if status == "Missing pair":
                    ax.text(0.98, y, status, transform=ax.get_yaxis_transform(),
                            ha="right", va="center", color=color, fontsize=8)
                    continue
                if status != "No interval":
                    ax.hlines(y, low, high, color=color, linewidth=1.2)
                ax.plot(
                    difference, y, linestyle="none", color=color, markersize=5,
                    marker="x" if status == "No interval" else "o",
                    markerfacecolor="white" if status == "Includes zero" else color,
                )
            count_lines.append(
                f"{short_label}   {counts['Above zero']} / {counts['Includes zero']} / {counts['Below zero']}"
            )
        ax.set_xlabel(_units(reports, benchmark))
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.text(0.5, -0.27, "\n".join(count_lines), transform=ax.transAxes,
                ha="center", va="top", fontsize=8)

    handles = [Line2D([], [], color=color, label=label) for _, label, _, color, _ in _HEAD_COMPARISONS]
    handles.extend([
        Line2D([], [], color="#222222", marker="o", linestyle="none", label="Interval excludes zero"),
        Line2D([], [], color="#222222", marker="o", linestyle="none",
               markerfacecolor="white", label="Interval includes zero"),
        Line2D([], [], color="#222222", marker="x", linestyle="none", label="No interval"),
    ])
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               title="95% paired intervals")
    return fig


def _benchmark_panels(encoder_labels, title, *, bottom):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(NEWTON_BENCHMARKS), sharey=True,
        figsize=(13, max(4, 2.2 + 0.4 * len(encoder_labels))),
    )
    fig.subplots_adjust(left=0.16, right=0.985, top=0.78, bottom=bottom, wspace=0.34)
    fig.suptitle(f"{title}\nMean difference across training seeds 0, 1, and 2")
    axes[0].set_yticks(range(len(encoder_labels)), labels=list(encoder_labels.values()))
    axes[0].set_ylim(len(encoder_labels) - 0.5, -0.5)
    for ax, label in zip(axes, NEWTON_BENCHMARKS.values()):
        ax.set_title(label)
    return fig, axes


def _units(reports, benchmark):
    report = reports.get(benchmark)
    if report is not None and report["analysis"].get("units"):
        return report["analysis"]["units"].capitalize()
    return "Accuracy percentage points" if benchmark == "ecoli_proteins" else "Spearman"


def _summary_rows(reports, benchmark, kind):
    report = reports.get(benchmark)
    if report is None or tuple(report["identity"]["seeds"]) != _TRAINING_SEEDS:
        return {}
    return {
        (row["encoder"], row["comparison"]): row
        for row in report["analysis"]["seed_summary"]
        if row["kind"] == kind
    }


def _interval(row):
    if row is None or row["status"] != "complete" or tuple(row["available_seeds"]) != _TRAINING_SEEDS:
        return None, None, None, "Missing pair"
    difference, low, high = (row.get(field) for field in ("difference", "ci_low", "ci_high"))
    if difference is None or not isfinite(difference):
        return None, None, None, "Missing pair"
    if low is None or high is None or not (isfinite(low) and isfinite(high)) or low > high:
        return difference, None, None, "No interval"
    # Classify the interval before formatting rounded values for display.
    if low > 0:
        return difference, low, high, "Above zero"
    if high < 0:
        return difference, low, high, "Below zero"
    return difference, low, high, "Includes zero"


def _plot_gain_grid(reports, encoder_labels, *, kind, columns, title, show_counts=False):
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    fig, axes = _benchmark_panels(encoder_labels, title, bottom=0.32 if show_counts else 0.24)
    colors = ListedColormap(list(_INTERVAL_COLORS.values()))
    color_index = {status: index for index, status in enumerate(_INTERVAL_COLORS)}
    for ax, benchmark in zip(axes, NEWTON_BENCHMARKS):
        comparisons = _summary_rows(reports, benchmark, kind)
        color_grid = []
        counts = Counter()
        precision = 2 if benchmark == "ecoli_proteins" else 3
        for y, encoder in enumerate(encoder_labels):
            color_row = []
            for x, (comparison, _) in enumerate(columns):
                difference, _, _, status = _interval(comparisons.get((encoder, comparison)))
                color_row.append(color_index[status])
                counts[status] += 1
                # Leave missing comparisons visibly empty of numeric values.
                label = "—" if difference is None else f"{difference:+.{precision}f}"
                ax.text(x, y, label, ha="center", va="center", fontsize=9)
            color_grid.append(color_row)

        ax.imshow(color_grid, cmap=colors, vmin=0, vmax=len(_INTERVAL_COLORS) - 1,
                  aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(columns)), labels=[label for _, label in columns])
        ax.set_xticks([i - 0.5 for i in range(1, len(columns))], minor=True)
        ax.set_yticks([i - 0.5 for i in range(1, len(encoder_labels))], minor=True)
        ax.grid(which="minor", color="white", linewidth=1)
        ax.tick_params(which="both", length=0)

        caption = _units(reports, benchmark)
        if show_counts:
            caption += (f"\n{counts['Above zero']} ahead · {counts['Below zero']} behind"
                        f"\n{counts['Includes zero']} include zero")
            for status in ("No interval", "Missing pair"):
                if counts[status]:
                    caption += f"\n{counts[status]} {status.lower()}"
        ax.set_xlabel(caption)

    handles = [Patch(facecolor=color, edgecolor="#AAAAAA", label=status)
               for status, color in _INTERVAL_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               title="95% paired intervals")
    return fig
