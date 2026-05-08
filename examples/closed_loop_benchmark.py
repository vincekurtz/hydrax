"""Generic closed-loop MPC benchmark for Hydrax tasks.

Each algorithm is run from the same initial state and random seed. At each
replanning step, the controller optimizes from the current state, then the
system is advanced forward in closed loop. The cumulative realized cost is
recorded over time.

Optional flags::

    --env NAME         Task environment (default: pendulum)
    --warp             Use MjWarp backend (default: False)
    --time T           Total benchmark time in seconds (default: 5.0)
    --frequency F      Replanning frequency in Hz (default: 50.0)
    --samples N        Number of rollout samples per algorithm (default: 2048)
    --save PATH        Save the figure to PATH instead of displaying it
"""

import argparse
import time

import matplotlib.pyplot as plt

from hydrax.algs import CEM, DIAL, MPPI, MppiCma, PredictiveSampling
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        type=str,
        default="pendulum",
        choices=sorted(ENVIRONMENTS.keys()),
        help="Task environment to benchmark (default: pendulum)",
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
        help="Number of rollout samples per algorithm (default: 128)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        metavar="PATH",
        help="Save figure to PATH instead of displaying it",
    )
    return parser.parse_args()


def make_task(env_name: str, warp: bool) -> Task:
    """Construct the selected task environment."""
    impl = "warp" if warp else "jax"
    return ENVIRONMENTS[env_name](impl=impl)


def main() -> None:
    """Run the closed-loop benchmark and plot cumulative realized cost."""
    args = parse_args()

    total_time = args.time
    replan_frequency = args.frequency
    num_samples = args.samples
    seed = 1

    # Shared spline / horizon settings
    shared = {
        "plan_horizon": 1.0,
        "spline_type": "zero",
        "num_knots": 8,
    }

    task = make_task(args.env, args.warp)
    initial_state = task.make_data()

    controllers = {
        "Predictive Sampling": PredictiveSampling(
            task,
            num_samples=num_samples,
            noise_level=0.3,
            **shared,
        ),
        "MPPI": MPPI(
            task,
            num_samples=num_samples,
            noise_level=0.3,
            temperature=0.1,
            **shared,
        ),
        "CEM": CEM(
            task,
            num_samples=num_samples,
            num_elites=max(4, num_samples // 8),
            sigma_start=0.3,
            sigma_min=0.05,
            **shared,
        ),
        "DIAL": DIAL(
            task,
            num_samples=num_samples,
            noise_level=0.3,
            beta_opt_iter=1.0,
            beta_horizon=2.0,
            temperature=0.1,
            **shared,
        ),
        "MPPI-CMA": MppiCma(
            task,
            num_samples=num_samples,
            initial_noise_level=0.3,
            temperature=0.1,
            minimum_noise_level=0.0,
            covariance_adaptation_rate=0.1,
            **shared,
        ),
    }

    print(
        f"Benchmarking env='{args.env}' with {len(controllers)} algorithms "
        f"for total_time={total_time:.2f}s at {replan_frequency:.1f} Hz "
        f"and {num_samples} samples.\n"
    )

    results = {}
    for name, ctrl in controllers.items():
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

        results[name] = (result, elapsed)
        print(
            f"  {name:20s} cumulative cost: "
            f"{float(result.costs[-1]):.4f}"
            f"  time: {elapsed:.2f}s"
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (result, _) in results.items():
        ax.plot(result.times, result.costs, label=name, linewidth=1.5)

    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("Cumulative realized cost")
    ax.set_title(f"Closed-loop MPC benchmark - {args.env}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if args.save is not None:
        fig.savefig(args.save, dpi=150)
        print(f"\nFigure saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()