import argparse

import mujoco

from hydrax.algs import CEM, MPPI, MTP, PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.navigation import Navigation

"""
Run an interactive simulation of the U-maze navigation task.

Double click on the green target, then drag it around with [ctrl + right-click].
The starting pointmass is placed inside a U-shaped barrier so that local
samplers (PS / MPPI / CEM) tend to stall against the inner wall, whereas
MTP's tensor sampler routes around it.
"""

parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the navigation task."
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

task = Navigation(impl="warp" if args.warp else "jax")

# All baselines share the same sample budget (32 rollouts), planning
# horizon (0.5 s), and spline resolution (11 knots) so the comparison
# is fair and representative short horizon to show MTP exploration capabilities.
# Local samplers (PS, MPPI, CEM) are tuned with aggressive noise
# but still cannot route around the U-wall.

if args.algorithm == "ps" or args.algorithm is None:
    print("Running predictive sampling")
    ctrl = PredictiveSampling(
        task,
        num_samples=32,
        noise_level=1.0,
        plan_horizon=0.5,
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
        plan_horizon=0.5,
        spline_type="zero",
        num_knots=11,
    )

elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task,
        num_samples=32,
        num_elites=2,
        sigma_start=1.0,
        sigma_min=0.5,
        explore_fraction=0.5,
        plan_horizon=0.5,
        spline_type="zero",
        num_knots=11,
    )

elif args.algorithm == "mtp":
    print("Running MTP")
    ctrl = MTP(
        task,
        num_samples=32,
        m_pts=5,
        n_per_layer=50,
        num_elites=2,
        sigma_start=0.7,
        sigma_min=0.5,
        sigma_max=1.0,
        beta=1.0,
        alpha=0.1,
        mtp_interpolation="bspline",
        plan_horizon=0.5,
        spline_type="zero",
        num_knots=11,
    )
else:
    parser.error("Invalid algorithm")

mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)
mj_data.qpos[:2] = [-0.2, 0.0]
mj_data.mocap_pos[0] = [0.25, 0.0, 0.01]

run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    show_traces=True,
    max_traces=5,
)
