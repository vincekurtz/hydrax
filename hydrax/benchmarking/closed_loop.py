r"""Closed-loop MPC benchmark.

Runs each (task, algorithm, seed) combination as a headless MPC
simulation and reports cumulative running cost over the episode. Each
seed picks both the initial condition and the algorithm's PRNG, so
seeds are one-to-one with initial conditions.

Example:
    uv run python -m hydrax.benchmarking.closed_loop \
        --tasks pendulum cart_pole --num-trials 3 \
        --output /tmp/closed_loop.csv
"""

from typing import Any, Dict

from hydrax.benchmarking import _cli
from hydrax.benchmarking.runner import run_closed_loop
from hydrax.benchmarking.tasks import TaskSpec

FIELDS = [
    "task",
    "algorithm",
    "seed",
    "num_samples",
    "cost",
    "wall_time_s",
    "status",
    "error",
]


def _trial(
    spec: TaskSpec, alg_name: str, seed: int, num_samples: int
) -> Dict[str, Any]:
    """Run one closed-loop trial and return metric fields."""
    task, ctrl = _cli.make_controller_and_task(
        spec, alg_name, seed, num_samples
    )
    qpos, qvel, mocap_pos = _cli.sample_initial_condition(spec, seed)
    total_cost, wall = run_closed_loop(
        task=task,
        controller=ctrl,
        qpos=qpos,
        qvel=qvel,
        mocap_pos=mocap_pos,
        control_frequency=spec.control_frequency,
        episode_length=spec.episode_length,
        seed=seed,
    )
    return {"cost": total_cost, "wall_time_s": wall}


def main() -> None:
    """CLI entry point."""
    args = _cli.parse_args(__doc__.splitlines()[0])
    _cli.run_grid(args, FIELDS, _trial)


if __name__ == "__main__":
    main()
