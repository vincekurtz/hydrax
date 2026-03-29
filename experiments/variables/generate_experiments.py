##
#
# Experimenta variables
#
##

import os
import numpy as np
import argparse

##################################################################
# EXPERIMENT VARIABLES
##################################################################

# which motions G1 motions to test
motion = ["Lafan1/mocap/UnitreeG1/walk1_subject1.npz"]

# how many times to randomize and level of randomization
num_randomizations = [0, 1, 2, 3]
level_randomization = [0.1, 0.5, 1.0]

# list of risk strategies to try
risk_strategy = ["average", "worst", "best"]

# use warp
use_warp = True

# duration (seconds)
duration = 420.0

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

##################################################################
# PROCESS VARIABLES
##################################################################

print(f"\nRequested GPU list:")
print(f"  Number of GPUs: {gpu_count}")
print(f"  GPU IDs: {gpu_list}")

print("\nExperiment variables:")
print(f"  Motions: {motion}")
print(f"  Number of randomizations: {num_randomizations}")
print(f"  Levels of randomization: {level_randomization}")
print(f"  Risk strategies: {risk_strategy}")

