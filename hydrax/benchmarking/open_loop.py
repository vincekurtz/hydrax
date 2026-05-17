r"""Open-loop trajectory optimization benchmark.

Runs each (task, algorithm, seed) combination by calling optimize() N
times from a fixed initial state, then evaluating the cost of the
final mean trajectory (no noise). Reports that cost and wall time.

Example:
    uv run python -m hydrax.benchmarking.open_loop \
        --tasks pendulum --num-trials 3 --num-iterations 50 \
        --output /tmp/open_loop.csv
"""

import argparse
import sys
from typing import Any, Callable, Dict

from hydrax.benchmarking import _cli
from hydrax.benchmarking.runner import run_open_loop
from hydrax.benchmarking.tasks import TaskSpec

FIELDS = [
    "task",
    "algorithm",
    "seed",
    "num_samples",
    "num_iterations",
    "cost",
    "wall_time_s",
    "status",
    "error",
]


def _trial_factory(
    num_iterations: int,
) -> Callable[[TaskSpec, str, int, int], Dict[str, Any]]:
    """Build a trial closure that captures num_iterations."""

    def _trial(
        spec: TaskSpec, alg_name: str, seed: int, num_samples: int
    ) -> Dict[str, Any]:
        task, ctrl = _cli.make_controller_and_task(
            spec, alg_name, seed, num_samples
        )
        qpos, qvel, mocap_pos = _cli.sample_initial_condition(spec, seed)
        final_cost, wall = run_open_loop(
            task=task,
            controller=ctrl,
            qpos=qpos,
            qvel=qvel,
            mocap_pos=mocap_pos,
            num_iterations=num_iterations,
            seed=seed,
        )
        return {
            "cost": final_cost,
            "wall_time_s": wall,
            "num_iterations": num_iterations,
        }

    return _trial


def main() -> None:
    """CLI entry point; extra --num-iterations flag."""
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--num-iterations", type=int, default=50)
    extra, remaining = base.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    args = _cli.parse_args(__doc__.splitlines()[0])
    _cli.run_grid(args, FIELDS, _trial_factory(extra.num_iterations))


if __name__ == "__main__":
    main()
