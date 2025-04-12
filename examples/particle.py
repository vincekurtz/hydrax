import argparse

import evosax
import mujoco

from hydrax.algs import MPPI, Evosax, PredictiveSampling
from hydrax.risk import WorstCase
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.particle import Particle

"""
Run an interactive simulation of the particle tracking task.

Double click on the green target, then drag it around with [ctrl + right-click].
"""

# Define the task (cost and dynamics)
task = Particle()

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the particle tracking task."
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("cmaes", help="CMA-ES")
subparsers.add_parser(
    "samr", help="Genetic Algorithm with Self-Adaptation Mutation Rate (SAMR)"
)
subparsers.add_parser("de", help="Differential Evolution")
subparsers.add_parser("gld", help="Gradient-Less Descent")
subparsers.add_parser("rs", help="Uniform Random Search")
args = parser.parse_args()

# Set the controller based on command-line arguments
if args.algorithm == "ps" or args.algorithm is None:
    print("Running predictive sampling")
    ctrl = PredictiveSampling(
        task,
        num_samples=16,
        noise_level=0.1,
        num_randomizations=10,
        risk_strategy=WorstCase(),
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=11,
    )

elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=16,
        noise_level=0.3,
        temperature=0.01,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=11,
    )

elif args.algorithm == "cmaes":
    print("Running CMA-ES")
    ctrl = Evosax(
        task,
        evosax.Sep_CMA_ES,
        num_samples=16,
        elite_ratio=0.5,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=11,
    )

elif args.algorithm == "samr":
    print("Running genetic algorithm with Self-Adaptation Mutation Rate (SAMR)")
    ctrl = Evosax(
        task,
        evosax.SAMR_GA,
        num_samples=16,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=11,
    )

elif args.algorithm == "de":
    print("Running Differential Evolution (DE)")
    ctrl = Evosax(
        task,
        evosax.DE,
        num_samples=16,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=11,
    )

elif args.algorithm == "gld":
    print("Running Gradient-Less Descent (GLD)")
    ctrl = Evosax(
        task,
        evosax.GLD,
        num_samples=16,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=11,
    )

elif args.algorithm == "rs":
    print("Running uniform random search")
    es_params = evosax.strategies.random.EvoParams(
        range_min=-1.0,
        range_max=1.0,
    )
    ctrl = Evosax(
        task,
        evosax.RandomSearch,
        num_samples=16,
        es_params=es_params,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=11,
    )
else:
    parser.error("Invalid algorithm")

# Define the model used for simulation
mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    show_traces=False,
    max_traces=5,
)
