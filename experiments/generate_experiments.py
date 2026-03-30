##
#
# Experimenta variables
#
##

import os
import argparse

##################################################################
# EXPERIMENT VARIABLES
##################################################################

# which motions G1 motions to test
motion = ["Lafan1/mocap/UnitreeG1/walk1_subject1.npz"]

# how many times to randomize and level of randomization
num_randomizations = [0, 2, 4, 6, 8]
level_randomization = [0.2, 0.4, 0.6, 1.0]

# list of risk strategies to try
risk_strategy = ["average", "worst", "best"]

# duration (seconds)
duration = 300.0

# use warp by default since it's faster
use_warp = True


##################################################################
# EXPERIMENT SETUP
##################################################################

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Parse experiment variables for the humanoid mocap."
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

# parse num_randomizations and level_randomization
# level_randomization is only used when num_randomizations > 1 (alg_base clamps
# to 1 and only randomizes when > 1), so only validate it in that case.
assert all(isinstance(nr, int) and nr >= 0 for nr in num_randomizations), "num_randomizations must be non-negative integers."
if any(nr > 1 for nr in num_randomizations):
    assert all(isinstance(lr, float) and 0.0 < lr <= 1.0 for lr in level_randomization), "level_randomization must be in (0.0, 1.0]."


##################################################################
# GENERATE EXPERIMENTS
##################################################################

print(f"\nNumber of GPUs: {num_gpus}")

print("\nExperiment variables:")
print(f"  Motions: {motion}")
print(f"  Number of randomizations: {num_randomizations}")
print(f"  Levels of randomization: {level_randomization}")
print(f"  Risk strategies: {risk_strategy}")

# enumerate every combination of variables
experiments = []
for m in motion:
    for rs in risk_strategy:
        for nr in num_randomizations:
            if nr <= 1:
                experiments.append({
                    "reference_filename": m,
                    "num_randomizations": nr,
                    "level_randomization": None,
                    "risk_strategy": rs,
                    "warp": use_warp,
                    "duration": duration,
                })
            else:
                for lr in level_randomization:
                    experiments.append({
                        "reference_filename": m,
                        "num_randomizations": nr,
                        "level_randomization": lr,
                        "risk_strategy": rs,
                        "warp": use_warp,
                        "duration": duration,
                    })

# function to convert experiment dict to command-line arguments
def experiment_to_args(exp, run_id):
    args = [f"--run_id {run_id}"]
    for key, val in exp.items():
        if key == "warp":
            if val:
                args.append("--warp")
        elif key in ("num_randomizations", "level_randomization") and not val:
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
    cmd = f"CUDA_VISIBLE_DEVICES=$GPU_ID uv run examples/humanoid_mocap_headless.py {experiment_to_args(exp, run_id)}"
    gpu_commands[gpu_slot].append(cmd)
    print(f"  [{run_id}] (GPU {gpu_slot:02d}) {cmd}")

# write experiment registry .txt
os.makedirs("experiments/run", exist_ok=True)
with open("experiments/run/experiment_registry.txt", "w") as f:
    for slot_cmds in gpu_commands.values():
        for cmd in slot_cmds:
            f.write(f"{cmd}\n")

# write one .sh per GPU slot, each takes GPU_ID as first argument
os.makedirs("experiments/run", exist_ok=True)
for slot, cmds in gpu_commands.items():
    filename = f"experiments/run/run_experiments_{slot:02d}.sh"
    with open(filename, "w") as f:
        f.write("#!/bin/bash\n")
        f.write('GPU_ID=${1:?\"Usage: bash $0 <GPU_ID>\"}\n')
        f.write(f'echo "Running slot {slot:02d} on GPU $GPU_ID"\n\n')
        f.write(" &&\n".join(cmd for cmd in cmds))
        f.write("\n")
    print(f"Wrote {filename}")

print(f"\nWrote experiments/run/experiment_registry.txt")
print(f"Usage: bash experiments/run/run_experiments_00.sh <GPU_ID>")