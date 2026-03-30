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
num_randomizations = [0, 2, 4, 6]
level_randomization = [0.1, 0.4, 0.8]

# list of risk strategies to try
risk_strategy = ["average"]

# use warp
use_warp = True

# duration (seconds)
duration = 60.0


##################################################################
# EXPERIMENT SETUP
##################################################################

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Parse experiment variables for the humanoid mocap."
)
parser.add_argument(
    "--gpu_id",
    type=str,
    default="0",
    help="GPU ID to use for the experiments, e.g., '0, 2, 7'.",
)
args = parser.parse_args()

# parse GPU list
gpu_list = [gpu.strip() for gpu in args.gpu_id.split(",")]
gpu_count = len(gpu_list)
assert all(gpu.isdigit() and int(gpu) >= 0 for gpu in gpu_list), "GPU IDs must be non-negative integers."

# parse num_randomizations and level_randomization
# level_randomization is only used when num_randomizations > 1 (alg_base clamps
# to 1 and only randomizes when > 1), so only validate it in that case.
assert all(isinstance(nr, int) and nr >= 0 for nr in num_randomizations), "num_randomizations must be non-negative integers."
if any(nr > 1 for nr in num_randomizations):
    assert all(isinstance(lr, float) and 0.0 < lr <= 1.0 for lr in level_randomization), "level_randomization must be in (0.0, 1.0]."


##################################################################
# GENERATE EXPERIMENTS
##################################################################

print(f"\nRequested GPU list:")
print(f"  Number of GPUs: {gpu_count}")
print(f"  GPU IDs: {gpu_list}")

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

# assign IDs and GPUs
print(f"\nTotal experiments: {len(experiments)}")
commands = {}  # run_id -> (gpu, command)
for i, exp in enumerate(experiments):
    run_id = f"{i + 1:03d}"         # 001, 002, etc.
    gpu = gpu_list[i % gpu_count]   # round-robin assign GPUs
    cmd = f"CUDA_VISIBLE_DEVICES={gpu} uv run examples/humanoid_mocap_headless.py {experiment_to_args(exp, run_id)}"
    commands[run_id] = (gpu, cmd)
    print(f"  [{run_id}] {cmd}")

# write experiment registry .txt
os.makedirs("experiments/setup", exist_ok=True)
with open("experiments/setup/experiment_registry.txt", "w") as f:
    for run_id, (_, cmd) in commands.items():
        f.write(f"{cmd}\n")

# write .sh grouped by GPU (each GPU's experiments run sequentially, GPUs in parallel)
gpu_commands = {gpu: [] for gpu in gpu_list}
for run_id, (gpu, cmd) in commands.items():
    gpu_commands[gpu].append(cmd)

with open("experiments/run_experiments.sh", "w") as f:
    f.write("#!/bin/bash\n\n")
    for gpu, cmds in gpu_commands.items():
        f.write(f"# GPU {gpu}\n(\n")
        f.write(" &&\n".join(f"  {cmd}" for cmd in cmds))
        f.write("\n) &\n\n")
    f.write("wait  # wait for all GPUs to finish\n")

print(f"\nWrote experiments/setup/experiment_registry.txt")
print(f"Wrote experiments/run_experiments.sh")