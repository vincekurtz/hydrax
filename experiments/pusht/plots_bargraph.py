##
#
# Heatmap and alternative visualizations for mean total cost
#
##

import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})

##################################################################
# LOAD ALL RUNS
##################################################################

data_dir = "experiments/pusht/data"
h5_files = sorted(glob.glob(f"{data_dir}/run_*.h5"))
print(f"Found {len(h5_files)} runs in {data_dir}/\n")

risk_order = ["worst", "average", "best"]
risk_labels = {"worst": "Pessimistic", "average": "Average", "best": "Optimistic"}

grouped = defaultdict(list)
for filepath in h5_files:
    with h5py.File(filepath, "r") as f:
        args = dict(f["experiment_args"].attrs)
        total_cost = f["metrics/total_cost"][:]
        grouped[(args["risk_strategy"], args["num_randomizations"])].append(total_cost.mean())

all_risk = set(k[0] for k in grouped)
risk_strategies = [r for r in risk_order if r in all_risk]
all_nr = sorted(set(k[1] for k in grouped))

N_expected = 20
means = {}
ses = {}
all_values = {}
for rs in risk_strategies:
    means[rs] = []
    ses[rs] = []
    all_values[rs] = []
    for nr in all_nr:
        values = grouped[(rs, nr)]
        assert len(values) == N_expected, f"Expected {N_expected} runs for ({rs}, {nr}), got {len(values)}"
        means[rs].append(np.mean(values))
        ses[rs].append(np.std(values) / np.sqrt(len(values)))
        all_values[rs].append(values)

risk_colors = {
    "worst":   np.array([216,  27,  96]) / 255,  # red
    "average": np.array([ 30, 136, 229]) / 255,  # blue
    "best":    np.array([255, 193,   7]) / 255,   # yellow
}

##################################################################
# 1. HEATMAP (original)
##################################################################

matrix = np.array([means[rs] for rs in risk_strategies])
matrix_se = np.array([ses[rs] for rs in risk_strategies])

fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(matrix, cmap="viridis")
ax.set_xticks(range(len(all_nr)))
ax.set_xticklabels([int(nr) for nr in all_nr])
ax.set_yticks(range(len(risk_strategies)))
ax.set_yticklabels([risk_labels[rs] for rs in risk_strategies])
ax.set_xlabel(r"Num. Randomizations, $R$")
for i in range(len(risk_strategies)):
    for j in range(len(all_nr)):
        ax.text(j, i, f"{matrix[i, j]:.3f}\n$\\pm${matrix_se[i, j]:.3f}",
                ha="center", va="center", color="w")
fig.colorbar(im, ax=ax)
plt.tight_layout()

##################################################################
# 2. GROUPED BAR CHART
##################################################################

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(all_nr))
width = 0.25

for i, rs in enumerate(risk_strategies):
    ax.bar(x + i * width, means[rs], width, yerr=ses[rs],
           label=risk_labels[rs], capsize=3, color=risk_colors[rs])

ax.set_xticks(x + width)
ax.set_xticklabels([int(nr) for nr in all_nr])
ax.set_xlabel(r"Num. Randomizations, $R$")
ax.set_ylabel(r"Mean Total Cost")
all_means = [v for rs in risk_strategies for v in means[rs]]
all_ses_vals = [v for rs in risk_strategies for v in ses[rs]]
y_min = min(m - s for m, s in zip(all_means, all_ses_vals))
y_max = max(m + s for m, s in zip(all_means, all_ses_vals))
margin = (y_max - y_min) * 0.05
ax.set_ylim(bottom=y_min - margin, top=y_max + margin * 5)
ax.legend(ncol=3)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()

plt.show()
