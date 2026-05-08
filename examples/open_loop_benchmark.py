"""Generic open-loop convergence benchmark for Hydrax tasks.

Each algorithm is run for a fixed number of iterations from the same initial
state and random seed. The mean rollout cost is recorded at every iteration
and plotted as a convergence curve.

Usage::

    python examples/pendulum_open_loop_benchmark.py --env pendulum

Optional flags::

    --env NAME       Task environment (default: pendulum)
    --warp           Use MjWarp backend (default: False)
    --iterations N   Number of optimization iterations (default: 100)
    --samples N      Number of rollout samples per algorithm (default: 128)
    --save PATH      Save the figure to PATH instead of displaying it
"""

import argparse
import time

import matplotlib.pyplot as plt

from hydrax.algs import CEM, DIAL, MPPI, MppiCma, PredictiveSampling
from hydrax.benchmarking import run_open_loop_benchmark
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
        "--iterations",
        type=int,
        default=100,
        help="Number of optimization iterations per algorithm (default: 100)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2048,
        help="Number of rollout samples per algorithm (default: 2048)",
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
    """Run the open-loop benchmark and plot convergence curves."""
    args = parse_args()

    iterations = args.iterations
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
        f"for {iterations} iterations and {num_samples} samples.\n"
    )

    results = {}
    for name, ctrl in controllers.items():
        t0 = time.perf_counter()
        result = run_open_loop_benchmark(
            ctrl, initial_state, iterations=iterations, seed=seed
        )
        result.mean_costs.block_until_ready()
        elapsed = time.perf_counter() - t0

        results[name] = (result, elapsed)
        print(
            f"  {name:20s} final mean: {float(result.mean_costs[-1]):.4f}"
            f"  best: {float(result.best_costs.min()):.4f}"
            f"  time: {elapsed:.2f}s"
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    iterations_axis = range(1, iterations + 1)

    for name, (result, _) in results.items():
        ax.plot(iterations_axis, result.mean_costs, label=name, linewidth=1.5)

    ax.set_xlabel("Optimization iteration")
    ax.set_ylabel("Cost")
    ax.set_title(f"Open-loop optimization convergence - {args.env}")
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
