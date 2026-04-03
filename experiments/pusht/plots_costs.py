##
#
# 2x2 position error vs time for nr=0,4,16,32
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

runs = []
for filepath in h5_files:
    with h5py.File(filepath, "r") as f:
        run = {
            "experiment_args": dict(f["experiment_args"].attrs),
            "position_cost": f["metrics/position_cost"][:],
            "sim_dt": f["trajectory"].attrs["sim_dt"],
        }
        runs.append(run)

##################################################################
# 2x2 PLOT: position_cost vs time for nr=0,4,16,32
##################################################################

nr_values = [0, 4, 16, 32]

risk_order = ["worst", "average", "best"]
all_risk = set(run["experiment_args"]["risk_strategy"] for run in runs)
risk_strategies = [r for r in risk_order if r in all_risk]
risk_labels = {"worst": "Pessimistic", "average": "Average", "best": "Optimistic"}
risk_colors = {
    "worst":   np.array([216,  27,  96]) / 255,  # red
    "average": np.array([ 30, 136, 229]) / 255,  # blue
    "best":    np.array([255, 193,   7]) / 255,   # yellow
}


grouped = defaultdict(list)
for run in runs:
    args = run["experiment_args"]
    key = (args["risk_strategy"], args["num_randomizations"])
    grouped[key].append(run["position_cost"])

fig, axes = plt.subplots(2, 2, figsize=(7, 5), sharey=True)

for idx, nr in enumerate(nr_values):
    ax = axes[idx // 2, idx % 2]
    for rs in risk_strategies:
        key = (rs, nr)
        if key in grouped:
            all_traces = np.sqrt(np.array(grouped[key]))
            sim_dt = runs[0]["sim_dt"]
            t = np.arange(all_traces.shape[1]) * sim_dt
            mean_trace = all_traces.mean(axis=0)
            se_trace = all_traces.std(axis=0) / np.sqrt(all_traces.shape[0])
            c = risk_colors[rs]
            ax.plot(t, mean_trace, label=risk_labels[rs], color=c)
            ax.fill_between(t, mean_trace - se_trace, mean_trace + se_trace, alpha=0.2, color=c)
    ax.set_title(rf"$R ={nr}$")
    if idx < 2:  # top row: hide x-axis tick labels
        ax.set_xticklabels([])
    else:  # bottom row only
        ax.set_xlabel(r"Time (s)")
    if idx % 2 == 0:  # left column only
        ax.set_ylabel(r"$\|\mathbf{p}_{b}^{\rm des} - \mathbf{p}_{b}\|$")
    if idx == 1:  # top-right only
        ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
