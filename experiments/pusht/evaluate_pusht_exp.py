##
#
# Evaluate the PUSHT experiments.
#
##

import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

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
            "filepath": filepath,
            # experiment args
            "experiment_args": dict(f["experiment_args"].attrs),
            # controller options
            "ctrl_options": dict(f["ctrl_options"].attrs),
            # metrics (scalar attrs)
            "total_wall_time": f["metrics"].attrs["total_wall_time"],
            "termination_time": f["metrics"].attrs["termination_time"],
            # metrics (time series)
            "position_cost": f["metrics/position_cost"][:],
            "orientation_cost": f["metrics/orientation_cost"][:],
            "close_to_block_cost": f["metrics/close_to_block_cost"][:],
            "control_cost": f["metrics/control_cost"][:],
            "total_cost": f["metrics/total_cost"][:],
            # trajectory
            "sim_dt": f["trajectory"].attrs["sim_dt"],
            "ctrl_dt": f["trajectory"].attrs["ctrl_dt"],
            "qpos": f["trajectory/qpos"][:],
            "qvel": f["trajectory/qvel"][:],
            "ctrl": f["trajectory/ctrl"][:],
        }
        runs.append(run)

# Print summary of loaded runs
for run in runs:
    args = run["experiment_args"]
    print(
        f"  {run['filepath']}: "
        f"nr={args['num_randomizations']}, "
        f"ctrl_seed={args['ctrl_seed']}, "
        f"sim_seed={args['sim_seed']}, "
        f"risk={args['risk_strategy']}"
    )


##################################################################
# PLOT DATA
##################################################################

from collections import defaultdict

# Get initial_pos as a hashable tuple for grouping
def get_initial_pos_key(run):
    ip = run["experiment_args"]["initial_pos"]
    return tuple(np.round(ip, 4))

# Extract unique initial positions (in order of first appearance)
seen = set()
initial_positions = []
for run in runs:
    key = get_initial_pos_key(run)
    if key not in seen:
        seen.add(key)
        initial_positions.append(key)

# Extract unique axes (sorted)
risk_order = ["worst", "average", "best"]
all_risk = set(run["experiment_args"]["risk_strategy"] for run in runs)
risk_strategies = [r for r in risk_order if r in all_risk]
risk_strategies += sorted(all_risk - set(risk_order))
num_randomizations = sorted(set(run["experiment_args"]["num_randomizations"] for run in runs))

cost_terms = ["position_cost", "orientation_cost", "close_to_block_cost", "total_cost"]

# Produce one set of plots per initial condition
for ip in initial_positions:
    ip_label = f"q0=[{', '.join(f'{v:.2f}' for v in ip)}]"
    ip_runs = [r for r in runs if get_initial_pos_key(r) == ip]

    # ---- Matrix heatmap ----
    grouped = defaultdict(list)
    for run in ip_runs:
        args = run["experiment_args"]
        key = (args["risk_strategy"], args["num_randomizations"])
        grouped[key].append(run["total_cost"].mean())

    matrix = np.zeros((len(risk_strategies), len(num_randomizations)))
    matrix_std = np.zeros((len(risk_strategies), len(num_randomizations)))
    for i, rs in enumerate(risk_strategies):
        for j, nr in enumerate(num_randomizations):
            values = grouped[(rs, nr)]
            matrix[i, j] = np.mean(values)
            matrix_std[i, j] = np.std(values)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(len(num_randomizations)))
    ax.set_xticklabels(num_randomizations)
    ax.set_yticks(range(len(risk_strategies)))
    ax.set_yticklabels(risk_strategies)
    ax.set_xlabel("num_randomizations")
    ax.set_ylabel("risk_strategy")
    ax.set_title(f"Mean Total Cost — {ip_label}")
    for i in range(len(risk_strategies)):
        for j in range(len(num_randomizations)):
            ax.text(j, i, f"{matrix[i, j]:.3f}\n±{matrix_std[i, j]:.3f}", ha="center", va="center", color="w")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    # ---- Cost terms over time ----
    grouped_ts = defaultdict(lambda: defaultdict(list))
    for run in ip_runs:
        args = run["experiment_args"]
        key = (args["risk_strategy"], args["num_randomizations"])
        for term in cost_terms:
            grouped_ts[key][term].append(run[term])

    nrows = len(cost_terms)
    ncols = len(num_randomizations)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey="row")

    for i, term in enumerate(cost_terms):
        for j, nr in enumerate(num_randomizations):
            ax = axes[i, j]
            for rs in risk_strategies:
                key = (rs, nr)
                if key in grouped_ts:
                    all_traces = np.array(grouped_ts[key][term])
                    sim_dt = ip_runs[0]["sim_dt"]
                    t = np.arange(all_traces.shape[1]) * sim_dt
                    mean_trace = all_traces.mean(axis=0)
                    std_trace = all_traces.std(axis=0)
                    ax.plot(t, mean_trace, label=rs)
                    ax.fill_between(t, mean_trace - std_trace, mean_trace + std_trace, alpha=0.2)
            if i == 0:
                ax.set_title(f"nr={nr}")
            if i == nrows - 1:
                ax.set_xlabel("Time (s)")
            if j == 0:
                ax.set_ylabel(term)
            ax.legend()
            ax.grid(True, alpha=0.3)

    fig.suptitle(f"Cost terms over time — {ip_label}")
    plt.tight_layout()

plt.show()
