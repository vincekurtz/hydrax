"""Closed-loop MPC spline-type benchmark for Hydrax tasks (data generation).

Runs the CEM controller in closed loop across one or more task environments,
comparing three control-interpolation spline types:

    * ``zero``   - zero-order hold (piecewise constant)
    * ``linear`` - linear interpolation
    * ``cubic``  - cubic spline

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
    --samples N        Number of rollout samples (default: 128)
    --out PATH         JSON output path (default: closed_loop_benchmark_data.json)
"""

import argparse
import json
import time

from hydrax.algs import CEM
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

# Fast, low-dimensional environments used when --envs is not given.
DEFAULT_ENVS = ["pendulum", "cart_pole", "double_cart_pole", "particle", "pusht"]

# Spline types to compare, in fixed order.
SPLINE_TYPES = ["zero", "linear", "cubic"]

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
        "--samples",
        type=int,
        default=128,
        help="Number of rollout samples (default: 128)",
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


def make_cem(task: Task, spline_type: str, num_samples: int) -> CEM:
    """Construct a CEM controller for a given task and spline type."""
    return CEM(
        task,
        num_samples=num_samples,
        num_elites=max(4, num_samples // 8),
        sigma_start=0.3,
        sigma_min=0.05,
        plan_horizon=1.0,
        spline_type=spline_type,
        num_knots=8,
    )


def main() -> None:
    """Run the CEM spline-type benchmark and write results to JSON."""
    args = parse_args()

    total_time = args.time
    replan_frequency = args.frequency
    num_samples = args.samples
    seed = 1

    print(
        f"Benchmarking CEM across {len(args.envs)} environment(s) and "
        f"{len(SPLINE_TYPES)} spline type(s) for total_time="
        f"{total_time:.2f}s at {replan_frequency:.1f} Hz and "
        f"{num_samples} samples.\n"
    )

    data = {
        "meta": {
            "controller": "CEM",
            "envs": list(args.envs),
            "spline_types": list(SPLINE_TYPES),
            "total_time": total_time,
            "replan_frequency": replan_frequency,
            "num_samples": num_samples,
            "seed": seed,
            "warp": bool(args.warp),
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
            ctrl = make_cem(task, spline_type, num_samples)
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
