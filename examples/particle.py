import argparse

import jax
import mujoco
from evosax.algorithms.distribution_based import (
    CMA_ES,
    GradientlessDescent,
    Open_ES,
    RandomSearch,
    SimulatedAnnealing,
    xNES,
)

from hydrax.algs import CEM, DIAL, MPPI, Evosax, PredictiveSampling
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
subparsers.add_parser("cem", help="Cross-Entropy Method")
subparsers.add_parser("cmaes", help="CMA-ES")
subparsers.add_parser("openes", help="OpenAI-ES")
subparsers.add_parser("sa", help="Simulated Annealing")
subparsers.add_parser("xnes", help="Exponential Natural Evolution Strategy")
subparsers.add_parser("gld", help="Gradient-Less Descent")
subparsers.add_parser("rs", help="Uniform Random Search")
subparsers.add_parser(
    "dial", help="Diffusion-Inspired Annealing for Legged MPC (DIAL)"
)
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

elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task,
        num_samples=32,
        num_elites=8,
        sigma_start=0.3,
        sigma_min=0.05,
        explore_fraction=0.5,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=11,
    )

elif args.algorithm == "cmaes":
    print("Running CMA-ES")
    ctrl = Evosax(
        task,
        CMA_ES,
        num_samples=16,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=11,
    )

elif args.algorithm == "gld":
    print("Running Gradient-Less Descent (GLD)")
    ctrl = Evosax(
        task,
        GradientlessDescent,
        num_samples=16,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=11,
    )

elif args.algorithm == "openes":
    print("Running OpenAI-ES")
    ctrl = Evosax(
        task,
        Open_ES,
        num_samples=16,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=11,
    )

elif args.algorithm == "sa":
    print("Running Simulated Annealing")
    ctrl = Evosax(
        task,
        SimulatedAnnealing,
        num_samples=16,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=11,
    )

elif args.algorithm == "xnes":
    print("Running Exponential Natural Evolution Strategy")
    ctrl = Evosax(
        task,
        xNES,
        num_samples=16,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=11,
    )

elif args.algorithm == "rs":
    print("Running uniform random search")
    sampling_fn = lambda key: jax.random.uniform(
        key, shape=(11 * 2), minval=-1.0, maxval=1.0
    )
    ctrl = Evosax(
        task,
        RandomSearch,
        sampling_fn=sampling_fn,
        num_samples=16,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=11,
    )

elif args.algorithm == "dial":
    print("Running Diffusion-Inspired Annealing for Legged MPC (DIAL)")
    ctrl = DIAL(
        task,
        num_samples=16,
        noise_level=0.4,
        beta_opt_iter=1.0,
        beta_horizon=1.0,
        temperature=0.001,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=11,
        iterations=5,
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
