"""Closed-loop MPC spline-type benchmark for Hydrax tasks (data generation).

Runs the Predictive Sampling controller in closed loop across one or more task
environments, comparing four control-interpolation spline types:

    * ``zero``   - zero-order hold (piecewise constant)
    * ``linear`` - linear interpolation
    * ``cubic``  - cubic spline
    * ``none``   - no spline: zero-order hold with one knot per simulation
      step over the planning horizon, i.e. a full-resolution control tape

The zero/linear/cubic cases share each environment's tuned knot count; ``none``
overrides it with one knot per simulation step (see ``PARAMS`` and
``make_controller``).

Per-environment controller parameters (num_samples, noise_level, plan_horizon,
num_knots) are taken from the corresponding single-task example scripts in
``examples/`` rather than a one-size-fits-all default.

For each (environment, spline type) pair the controller is run from the same
initial state and random seed. The full cumulative-cost curve, the final
cumulative cost, and wall-clock time are recorded and written to a JSON file.

This script only *generates* data. Use ``plot_closed_loop_benchmark.py`` to
draw figures from the saved JSON, so plotting/formatting can be iterated on
without re-running the (expensive) benchmark.

Optional flags::

    --envs A B C       Task environments (default: a fast low-dim subset)
    --warp             Use MjWarp backend (default: False)
    --time T           Total benchmark time in seconds (default: 5.0)
    --frequency F      Replanning frequency in Hz (default: 50.0)
    --out PATH         JSON output path (default: closed_loop_benchmark_data.json)
"""

import argparse
import json
import time

from hydrax.algs import PredictiveSampling
from hydrax.benchmarking import run_closed_loop_benchmark
from hydrax.task_base import Task
from hydrax.tasks.cart_pole import CartPole
from hydrax.tasks.crane import Crane
from hydrax.tasks.cube import CubeRotation
from hydrax.tasks.double_cart_pole import DoubleCartPole
from hydrax.tasks.humanoid_standup import HumanoidStandup
from hydrax.tasks.particle import Particle
from hydrax.tasks.pendulum import Pendulum
from hydrax.tasks.pusht import PushT
from hydrax.tasks.walker import Walker

ENVIRONMENTS = {
    "pendulum": Pendulum,
    "cart_pole": CartPole,
    "double_cart_pole": DoubleCartPole,
    "particle": Particle,
    "crane": Crane,
    "cube": CubeRotation,
    "pusht": PushT,
    "walker": Walker,
    "humanoid_standup": HumanoidStandup,
}

# Per-environment Predictive Sampling parameters, gathered from the individual
# single-task example scripts in examples/ (num_knots is the base used for the
# zero/linear/cubic spline types; "none" overrides it, see make_controller).
PARAMS = {
    # env               samples  noise  horizon  knots   (source example)
    "pendulum":         dict(num_samples=32,   noise_level=0.1,  plan_horizon=1.0,  num_knots=11),
    "cart_pole":        dict(num_samples=128,  noise_level=0.3,  plan_horizon=1.0,  num_knots=4),
    "double_cart_pole": dict(num_samples=1024, noise_level=0.3,  plan_horizon=1.0,  num_knots=4),
    "particle":         dict(num_samples=16,   noise_level=0.1,  plan_horizon=0.25, num_knots=11),
    "crane":            dict(num_samples=8,    noise_level=0.05, plan_horizon=0.8,  num_knots=3),
    "cube":             dict(num_samples=32,   noise_level=0.2,  plan_horizon=0.25, num_knots=4),
    "pusht":            dict(num_samples=128,  noise_level=0.4,  plan_horizon=0.5,  num_knots=6),
    "walker":           dict(num_samples=128,  noise_level=0.5,  plan_horizon=0.6,  num_knots=5),
    # humanoid_standup has no Predictive Sampling example; horizon/knots taken
    # from its MPPI-CMA example, with a standard noise level.
    "humanoid_standup": dict(num_samples=128,  noise_level=0.3,  plan_horizon=1.0,  num_knots=4),
}

# Fast, low-dimensional environments used when --envs is not given.
DEFAULT_ENVS = ["pendulum", "cart_pole", "double_cart_pole", "particle", "pusht"]

# Spline types to compare, in fixed order. "none" is not a real spline type: it
# is a zero-order hold with one knot per simulation step (see make_controller).
SPLINE_TYPES = ["zero", "linear", "cubic", "none"]

DEFAULT_OUT = "closed_loop_benchmark_data.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--envs",
        type=str,
        nargs="+",
        default=DEFAULT_ENVS,
        choices=sorted(ENVIRONMENTS.keys()),
        metavar="ENV",
        help="Task environments to benchmark (default: a fast subset)",
    )
    parser.add_argument(
        "--warp",
        action="store_true",
        help="Use the experimental MjWarp backend (default: False)",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=5.0,
        help="Total benchmark time in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=50.0,
        help="Replanning frequency in Hz (default: 50.0)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUT,
        metavar="PATH",
        help=f"JSON output path (default: {DEFAULT_OUT})",
    )
    return parser.parse_args()


def make_task(env_name: str, warp: bool) -> Task:
    """Construct the selected task environment."""
    impl = "warp" if warp else "jax"
    return ENVIRONMENTS[env_name](impl=impl)


def make_controller(
    task: Task, env_name: str, spline_type: str
) -> PredictiveSampling:
    """Construct a Predictive Sampling controller for a task and spline type.

    Uses the per-environment parameters in ``PARAMS``. The pseudo-type
    ``"none"`` (no spline) is realized as a zero-order hold with one knot per
    simulation step over the planning horizon, giving a full-resolution
    control tape.
    """
    params = PARAMS[env_name]
    plan_horizon = params["plan_horizon"]

    if spline_type == "none":
        actual_spline_type = "zero"
        num_knots = max(int(round(plan_horizon / float(task.dt))), 1)
    else:
        actual_spline_type = spline_type
        num_knots = params["num_knots"]

    return PredictiveSampling(
        task,
        num_samples=params["num_samples"],
        noise_level=params["noise_level"],
        plan_horizon=plan_horizon,
        spline_type=actual_spline_type,
        num_knots=num_knots,
    )


def main() -> None:
    """Run the Predictive Sampling spline benchmark and write results to JSON."""
    args = parse_args()

    total_time = args.time
    replan_frequency = args.frequency
    seed = 1

    print(
        f"Benchmarking Predictive Sampling across {len(args.envs)} "
        f"environment(s) and {len(SPLINE_TYPES)} spline type(s) for "
        f"total_time={total_time:.2f}s at {replan_frequency:.1f} Hz.\n"
    )

    data = {
        "meta": {
            "controller": "Predictive Sampling",
            "envs": list(args.envs),
            "spline_types": list(SPLINE_TYPES),
            "total_time": total_time,
            "replan_frequency": replan_frequency,
            "seed": seed,
            "warp": bool(args.warp),
            "params": {env: PARAMS[env] for env in args.envs},
        },
        # results[env][spline_type] = {final_cost, elapsed, times, costs}
        "results": {},
    }

    for env_name in args.envs:
        task = make_task(env_name, args.warp)
        initial_state = task.make_data()
        data["results"][env_name] = {}
        print(f"{env_name}:")
        for spline_type in SPLINE_TYPES:
            ctrl = make_controller(task, env_name, spline_type)
            t0 = time.perf_counter()
            result = run_closed_loop_benchmark(
                ctrl,
                initial_state,
                total_time=total_time,
                replan_frequency=replan_frequency,
                seed=seed,
            )
            result.costs.block_until_ready()
            elapsed = time.perf_counter() - t0

            final_cost = float(result.costs[-1])
            data["results"][env_name][spline_type] = {
                "final_cost": final_cost,
                "elapsed": elapsed,
                "times": [float(t) for t in result.times],
                "costs": [float(c) for c in result.costs],
            }
            print(
                f"  {spline_type:8s} cumulative cost: {final_cost:12.4f}"
                f"  time: {elapsed:.2f}s"
            )
        print()

    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Benchmark data written to {args.out}")


if __name__ == "__main__":
    main()
