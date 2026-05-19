"""Closed-loop benchmark with uniform task setup and uniform hparams.

Every (task, algorithm) trial uses the same task-level setup
(`plan_horizon=1.0 s`, `num_knots=8`, `spline_type=cubic`) and the same
per-algorithm hyperparameter set, with randomized ICs across trials.
"""

import argparse
import csv
import os
import time
from typing import Any, Dict, List

from hydrax.benchmarking.runner import run_closed_loop
from hydrax.benchmarking.sweep import build_controller
from hydrax.benchmarking.tasks import TASKS, TaskSpec
from hydrax.benchmarking.uniform_hparams import (
    UNIFORM_HPARAMS,
    UNIFORM_TASK_SETUP,
)

ALL_ALGS = ["ps", "mppi", "cem", "dial", "cmaes", "mppi_cma", "er_cma"]

FIELDNAMES = [
    "task",
    "algorithm",
    "seed",
    "num_samples",
    "cost",
    "wall_time_s",
    "status",
    "error",
]


def write_header_if_needed(path: str) -> None:
    """Create the CSV with a header row if it doesn't already exist."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def append_row(path: str, row: Dict[str, Any]) -> None:
    """Append one CSV row."""
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)


def run_trial(
    spec: TaskSpec,
    alg_name: str,
    hparams: Dict[str, Any],
    num_samples: int,
    seed: int,
) -> Dict[str, Any]:
    """Build a uniform-setup controller, sample a random IC, run an episode."""
    import jax  # noqa: PLC0415

    task = spec.factory()
    rng = jax.random.key(seed)
    qpos, qvel, mocap_pos = spec.ic_sampler(rng)
    ctrl = build_controller(
        alg_name,
        task,
        hparams,
        num_samples=num_samples,
        plan_horizon=UNIFORM_TASK_SETUP["plan_horizon"],
        num_knots=UNIFORM_TASK_SETUP["num_knots"],
        spline_type=UNIFORM_TASK_SETUP["spline_type"],
        seed=seed,
    )
    cost, wall_time = run_closed_loop(
        task,
        ctrl,
        qpos,
        qvel,
        mocap_pos,
        control_frequency=spec.control_frequency,
        episode_length=spec.episode_length,
        seed=seed,
    )
    return {"cost": cost, "wall_time_s": wall_time}


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description="Closed-loop benchmark with uniform task setup + hparams."
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASKS.keys()),
        choices=list(TASKS.keys()),
        help="Tasks to evaluate.",
    )
    p.add_argument(
        "--algorithms",
        nargs="+",
        default=ALL_ALGS,
        choices=ALL_ALGS,
        help="Algorithms to evaluate.",
    )
    p.add_argument(
        "--num-trials",
        type=int,
        default=10,
        help="Trials per (task, algorithm). Each trial uses a distinct seed.",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=256,
        help="Number of samples per optimize() call.",
    )
    p.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Offset added to per-trial seeds for reproducible re-runs.",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="CSV path for results (rows appended).",
    )
    args = p.parse_args()

    write_header_if_needed(args.output)
    total = len(args.tasks) * len(args.algorithms) * args.num_trials
    done = 0
    start = time.time()

    for task_name in args.tasks:
        spec = TASKS[task_name]
        for alg_name in args.algorithms:
            hparams = UNIFORM_HPARAMS[alg_name]
            for trial in range(args.num_trials):
                seed = args.seed_offset + trial
                done += 1
                print(
                    f"[{done}/{total}] task={task_name} alg={alg_name} "
                    f"seed={seed} ...",
                    flush=True,
                )
                try:
                    metrics = run_trial(
                        spec, alg_name, hparams, args.num_samples, seed
                    )
                    status, error = "ok", ""
                except Exception as e:  # noqa: BLE001
                    metrics = {
                        "cost": float("nan"),
                        "wall_time_s": float("nan"),
                    }
                    status = "error"
                    error = repr(e)
                    print(f"  ERROR: {error}", flush=True)

                row = {
                    "task": task_name,
                    "algorithm": alg_name,
                    "seed": seed,
                    "num_samples": args.num_samples,
                    "status": status,
                    "error": error,
                    **metrics,
                }
                append_row(args.output, row)
                cost = metrics.get("cost", float("nan"))
                wt = metrics.get("wall_time_s", float("nan"))
                print(f"  -> cost={cost:.4f}, wall_time={wt:.2f}s", flush=True)

    elapsed = time.time() - start
    print(f"Done. {total} trials in {elapsed:.1f}s. Output: {args.output}")


if __name__ == "__main__":
    main()


__all__: List[str] = ["main"]
