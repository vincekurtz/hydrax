import argparse

import mujoco

from hydrax.algs import CEM, MPPI, MTP, PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.bugtrap import BugTrap

"""
Run an interactive simulation of the bug-trap navigation task.

Double click on the green target, then drag it around with [ctrl + right-click].
The starting pointmass is placed inside a U-shaped barrier that creates a
local minimum for purely local samplers.
"""

parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the bug-trap task."
)
parser.add_argument(
    "--warp",
    action="store_true",
    help="Whether to use the (experimental) MjWarp backend. (default: False)",
    required=False,
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("cem", help="Cross-Entropy Method")
subparsers.add_parser("mtp", help="Model Tensor Planning")
args = parser.parse_args()

task = BugTrap(impl="warp" if args.warp else "jax")

if args.algorithm == "ps" or args.algorithm is None:
    print("Running predictive sampling")
    ctrl = PredictiveSampling(
        task,
        num_samples=32,
        noise_level=1.0,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
    )

elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=32,
        noise_level=1.0,
        temperature=0.01,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
    )

elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task,
        num_samples=32,
        num_elites=1,
        sigma_start=1.0,
        sigma_min=0.5,
        explore_fraction=0.5,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
    )

elif args.algorithm == "mtp":
    print("Running MTP")
    ctrl = MTP(
        task,
        num_samples=32,
        m_pts=4,
        num_elites=1,
        sigma_start=0.7,
        sigma_min=0.5,
        sigma_max=1.0,
        beta=1.0,
        alpha=0.1,
        mtp_interpolation="akima",
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
    )
else:
    parser.error("Invalid algorithm")

mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)
mj_data.qpos[:2] = [-0.15, 0.0]
mj_data.mocap_pos[0] = [0.25, 0.0, 0.01]

run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
)
