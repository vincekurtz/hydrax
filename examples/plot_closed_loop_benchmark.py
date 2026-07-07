"""Plot closed-loop spline-type benchmark results.

Reads the JSON produced by ``closed_loop_benchmark.py`` and draws a grouped
bar chart: one group per environment, one bar per spline type. All
plotting/formatting lives here so it can be iterated on without re-running
the benchmark.

Because cumulative-cost magnitudes differ by orders of magnitude across
environments, bar heights are normalized so that each environment's
zero-order-hold cost is 1.0. Bars below 1.0 therefore mean the spline type
beat zero-order hold on that environment; raw costs are printed to the console.

Optional flags::

    --data PATH        Input JSON (default: closed_loop_benchmark_data.json)
    --save PATH        Save the figure to PATH instead of displaying it
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_DATA = "closed_loop_benchmark_data.json"

# Display order, labels, and colors for spline types (validated categorical
# slots: blue / aqua / yellow). "zero" is the normalization baseline.
BASELINE = "zero"
SPLINE_LABELS = {
    "zero": "Zero-order hold",
    "linear": "Linear",
    "cubic": "Cubic",
    "none": "No spline (per-step)",
}
SPLINE_COLORS = {
    "zero": "#2a78d6",
    "linear": "#1baf7a",
    "cubic": "#eda100",
    "none": "#008300",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATA,
        metavar="PATH",
        help=f"Input JSON path (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        metavar="PATH",
        help="Save figure to PATH instead of displaying it",
    )
    return parser.parse_args()


def load_final_costs(data: dict) -> tuple[list[str], list[str], dict]:
    """Extract (envs, spline_types, final_costs) from loaded JSON."""
    envs = data["meta"]["envs"]
    spline_types = data["meta"]["spline_types"]
    final_costs = {
        env: {s: data["results"][env][s]["final_cost"] for s in spline_types}
        for env in envs
    }
    return envs, spline_types, final_costs


def plot_bar_chart(
    envs: list[str],
    spline_types: list[str],
    final_costs: dict,
    save: str | None,
) -> None:
    """Draw a grouped bar chart of normalized cost per spline type."""
    x = np.arange(len(envs))
    n_splines = len(spline_types)
    bar_width = 0.8 / n_splines

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(envs) + 2), 5))

    for i, spline_type in enumerate(spline_types):
        offsets = x + (i - (n_splines - 1) / 2) * bar_width
        # Normalize each environment to its zero-order-hold cost.
        heights = [
            final_costs[env][spline_type] / final_costs[env][BASELINE]
            for env in envs
        ]
        bars = ax.bar(
            offsets,
            heights,
            bar_width * 0.92,
            label=SPLINE_LABELS.get(spline_type, spline_type),
            color=SPLINE_COLORS.get(spline_type),
            zorder=3,
        )
        # Direct value labels (relief rule: aqua/yellow are low-contrast).
        for rect in bars:
            ax.annotate(
                f"{rect.get_height():.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#3a3a38",
            )

    # Zero-order-hold baseline reference at 1.0.
    ax.axhline(1.0, color="#9a9a95", linewidth=1.0, linestyle="--", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=15, ha="right")
    ax.set_ylabel("Cumulative cost (normalized to zero-order hold)")
    ax.set_title("CEM closed-loop performance by spline type")
    ax.legend(title="Spline type", frameon=False)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=150)
        print(f"Figure saved to {save}")
    else:
        plt.show()


def main() -> None:
    """Load benchmark data and plot the grouped bar chart."""
    args = parse_args()

    with open(args.data) as f:
        data = json.load(f)

    envs, spline_types, final_costs = load_final_costs(data)

    # Echo the raw (un-normalized) costs.
    print(f"Raw cumulative costs from {args.data}:\n")
    for env in envs:
        print(f"{env}:")
        for spline_type in spline_types:
            print(
                f"  {spline_type:8s} {final_costs[env][spline_type]:12.4f}"
            )
        print()

    plot_bar_chart(envs, spline_types, final_costs, args.save)


if __name__ == "__main__":
    main()
