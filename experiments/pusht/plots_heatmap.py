##
#
# Heatmap: mean total cost (risk_strategy x num_randomizations)
#
##

import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
})

##################################################################
# LOAD ALL RUNS
##################################################################

data_dir = "experiments/pusht/data"
h5_files = sorted(glob.glob(f"{data_dir}/run_*.h5"))
print(f"Found {len(h5_files)} runs in {data_dir}/\n")

##################################################################
# HEATMAP
##################################################################

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
matrix = np.zeros((len(risk_strategies), len(all_nr)))
matrix_se = np.zeros((len(risk_strategies), len(all_nr)))
for i, rs in enumerate(risk_strategies):
    for j, nr in enumerate(all_nr):
        values = grouped[(rs, nr)]
        assert len(values) == N_expected, f"Expected {N_expected} runs for ({rs}, {nr}), got {len(values)}"
        matrix[i, j] = np.mean(values)
        matrix_se[i, j] = np.std(values) / np.sqrt(len(values))

fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(matrix, cmap="viridis")
ax.set_xticks(range(len(all_nr)))
ax.set_xticklabels(all_nr)
ax.set_yticks(range(len(risk_strategies)))
ax.set_yticklabels([risk_labels[rs] for rs in risk_strategies])
ax.set_xlabel("num_randomizations")
ax.set_ylabel("risk_strategy")
ax.set_title("Mean Total Cost")
for i in range(len(risk_strategies)):
    for j in range(len(all_nr)):
        ax.text(j, i, f"{matrix[i, j]:.3f}\n±{matrix_se[i, j]:.3f}", ha="center", va="center", color="w")
fig.colorbar(im, ax=ax)
plt.tight_layout()

plt.show()
