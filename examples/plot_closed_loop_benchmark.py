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

# Left-to-right display order within each environment group. "zero" is the
# normalization baseline. Any spline type not listed here is appended after.
SPLINE_ORDER = ["none", "zero", "linear", "cubic"]
BASELINE = "zero"
SPLINE_LABELS = {
    "none": "No spline",
    "zero": "Zero-order hold",
    "linear": "Linear",
    "cubic": "Cubic",
}
SPLINE_COLORS = {
    "none": "#e34948",
    "zero": "#2a78d6",
    "linear": "#1baf7a",
    "cubic": "#eda100",
}


def order_spline_types(spline_types: list[str]) -> list[str]:
    """Return spline types in the fixed display order, unknowns appended."""
    known = [s for s in SPLINE_ORDER if s in spline_types]
    extra = [s for s in spline_types if s not in SPLINE_ORDER]
    return known + extra


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
    controller: str = "Controller",
) -> None:
    """Draw a grouped bar chart of normalized cost per spline type."""
    spline_types = order_spline_types(spline_types)
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
        ax.bar(
            offsets,
            heights,
            bar_width * 0.92,
            label=SPLINE_LABELS.get(spline_type, spline_type),
            color=SPLINE_COLORS.get(spline_type),
            zorder=3,
        )

    # Zero-order-hold baseline reference at 1.0.
    ax.axhline(1.0, color="#9a9a95", linewidth=1.0, linestyle="--", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=15, ha="right")
    ax.set_ylabel("Cumulative cost (normalized to zero-order hold)")
    ax.set_title(f"{controller} closed-loop performance by spline type")
    ax.legend(title="Spline type", framealpha=0.9)
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

    controller = data["meta"].get("controller", "Controller")
    plot_bar_chart(envs, spline_types, final_costs, args.save, controller)


if __name__ == "__main__":
    main()
