##
#
# Evaluate all your runs
#
##

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt


##################################################################
# LOAD DATA
##################################################################

data_dir = "experiments/data"

runs = {}
for filename in sorted(os.listdir(data_dir)):
    if not filename.endswith(".h5"):
        continue
    run_id = filename.replace(".h5", "")

    with h5py.File(os.path.join(data_dir, filename), "r") as f:
        run = {}
        run["experiment_args"] = dict(f["experiment_args"].attrs)
        run["cem_options"] = dict(f["cem_options"].attrs)
        # metrics: datasets (arrays) + attrs (scalars)
        run["metrics"] = {k: v[()] for k, v in f["metrics"].items()}
        run["metrics"].update(dict(f["metrics"].attrs))
        # trajectory: datasets + attrs
        run["trajectory"] = {k: v[()] for k, v in f["trajectory"].items()}
        run["trajectory"]["sim_dt"] = f["trajectory"].attrs["sim_dt"]
        run["trajectory"]["ctrl_dt"] = f["trajectory"].attrs["ctrl_dt"]

    runs[run_id] = run

print(f"Loaded {len(runs)} runs: {list(runs.keys())}")

