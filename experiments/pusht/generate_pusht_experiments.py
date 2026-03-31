##
#
# Experiment variables
#
##

import os
import argparse

##################################################################
# EXPERIMENT VARIABLES
##################################################################

# how many times to randomize
num_randomizations = [0, 4, 32, 128]

# seeds for DR of the sim model and the controller
sim_seed  = [100, 101, 102, 103, 104] # seed for simulation DR
ctrl_seed = [0, 1, 2, 3, 4]       # seed for control DR (if num_randomizations >= 2)

# list of risk strategies to try
risk_strategy = ["average", "worst", "best"]

# duration (seconds)
duration = 7.0

##################################################################
# EXPERIMENT SETUP
##################################################################

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Parse experiment variables for the push-T task."
)
parser.add_argument(
    "--num_gpu",
    type=int,
    default=1,
    help="Number of GPUs to round-robin experiments across.",
)
args = parser.parse_args()

num_gpus = args.num_gpu
assert num_gpus >= 1, "num_gpu must be at least 1."

assert all(isinstance(nr, int) and nr >= 0 for nr in num_randomizations), "num_randomizations must be non-negative integers."
assert not set(sim_seed) & set(ctrl_seed), "sim_seed and ctrl_seed must not overlap, otherwise the sim model will be exactly one of the control models."


##################################################################
# GENERATE EXPERIMENTS
##################################################################

print(f"\nNumber of GPUs: {num_gpus}")

print("\nExperiment variables:")
print(f"  Number of randomizations: {num_randomizations}")
print(f"  Sim seeds: {sim_seed}")
print(f"  Ctrl seeds: {ctrl_seed}")
print(f"  Risk strategies: {risk_strategy}")

# enumerate every combination of variables
experiments = []
for rs in risk_strategy:
    for nr in num_randomizations:
        ctrl_seeds = ctrl_seed if nr >= 2 else [0]
        for ss in sim_seed:
            for cs in ctrl_seeds:
                experiments.append({
                    "num_randomizations": nr,
                    "ctrl_seed": cs,
                    "sim_seed": ss,
                    "risk_strategy": rs,
                    "duration": duration,
                })

# function to convert experiment dict to command-line arguments
def experiment_to_args(exp, run_id):
    args = [f"--run_id {run_id}"]
    for key, val in exp.items():
        if key == "num_randomizations" and not val:
            pass
        elif key == "ctrl_seed" and exp.get("num_randomizations", 0) < 2:
            pass
        elif val is not None:
            args.append(f"--{key} {val}")
    return " ".join(args)

# assign IDs and round-robin across GPU slots
print(f"\nTotal experiments: {len(experiments)}")
gpu_commands = {i: [] for i in range(num_gpus)}  # gpu_slot -> list of commands
for i, exp in enumerate(experiments):
    run_id = f"{i + 1:03d}"               # 001, 002, etc.
    gpu_slot = i % num_gpus               # round-robin assign GPU slots
    cmd = f"CUDA_VISIBLE_DEVICES=$GPU_ID uv run examples/pusht_headless.py {experiment_to_args(exp, run_id)}"
    gpu_commands[gpu_slot].append((run_id, cmd))
    print(f"  [{run_id}] (GPU {gpu_slot:02d}) {cmd}")

# write experiment registry .txt
os.makedirs("experiments/pusht/run", exist_ok=True)
with open("experiments/pusht/run/experiment_registry.txt", "w") as f:
    for slot_cmds in gpu_commands.values():
        for run_id, cmd in slot_cmds:
            f.write(f"[{run_id}] {cmd}\n")

# write one .sh per GPU slot, each takes GPU_ID as first argument
os.makedirs("experiments/pusht/run", exist_ok=True)
for slot, cmds in gpu_commands.items():
    filename = f"experiments/pusht/run/run_experiments_{slot:02d}.sh"
    with open(filename, "w") as f:
        f.write("#!/bin/bash\n")
        f.write('GPU_ID=${1:?\"Usage: bash $0 <GPU_ID>\"}\n')
        f.write(f'echo "Running slot {slot:02d} on GPU $GPU_ID"\n\n')
        f.write(" &&\n".join(f'echo ">>> [{run_id}] {cmd}" &&\n{cmd}' for run_id, cmd in cmds))
        f.write("\n")
    print(f"Wrote {filename}")

print(f"\nWrote experiments/pusht/run/experiment_registry.txt")
print(f"Usage: bash experiments/pusht/run/run_experiments_00.sh <GPU_ID>")
