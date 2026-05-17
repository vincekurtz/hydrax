"""Shared CLI argument parsing and trial loop for the benchmark scripts."""

import argparse
import csv
import os
import time
from typing import Any, Callable, Dict, List, Tuple

import jax

from hydrax.alg_base import SamplingBasedController
from hydrax.benchmarking.algorithms import ALGORITHMS, make_algorithm
from hydrax.benchmarking.tasks import TASKS, TaskSpec
from hydrax.task_base import Task

TrialFn = Callable[[TaskSpec, str, int, int], Dict[str, Any]]


def parse_args(description: str) -> argparse.Namespace:
    """Parse the shared CLI args used by both benchmark scripts."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASKS.keys()),
        choices=list(TASKS.keys()),
        help="Tasks to benchmark.",
    )
    p.add_argument(
        "--algorithms",
        nargs="+",
        default=list(ALGORITHMS.keys()),
        choices=list(ALGORITHMS.keys()),
        help="Algorithms to benchmark.",
    )
    p.add_argument(
        "--num-trials",
        type=int,
        default=5,
        help=(
            "Trials per (task, algorithm). Each trial uses a distinct "
            "seed and initial-condition pairing."
        ),
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=128,
        help=(
            "Number of samples per optimize() call. Held fixed across "
            "algorithms."
        ),
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
        help="Path to append per-trial CSV rows.",
    )
    return p.parse_args()


def write_header_if_needed(path: str, fieldnames: List[str]) -> None:
    """Create the CSV with a header row if it doesn't already exist."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def append_row(path: str, row: Dict[str, Any], fieldnames: List[str]) -> None:
    """Append one CSV row in dict form."""
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow(row)


def run_grid(
    args: argparse.Namespace,
    fieldnames: List[str],
    trial_fn: TrialFn,
) -> None:
    """Iterate over (task, algorithm, trial) and append rows to CSV.

    `trial_fn(task_spec, algorithm_name, seed, num_samples)` returns a
    dict of CSV fields other than the grid coordinates, which are filled
    in here.
    """
    write_header_if_needed(args.output, fieldnames)
    total = len(args.tasks) * len(args.algorithms) * args.num_trials
    done = 0
    overall_start = time.time()
    for task_name in args.tasks:
        spec = TASKS[task_name]
        for alg_name in args.algorithms:
            for trial in range(args.num_trials):
                seed = args.seed_offset + trial
                done += 1
                print(
                    f"[{done}/{total}] task={task_name} "
                    f"alg={alg_name} seed={seed} ...",
                    flush=True,
                )
                try:
                    metrics = trial_fn(spec, alg_name, seed, args.num_samples)
                    status = "ok"
                    error = ""
                except Exception as e:  # noqa: BLE001
                    metrics = {
                        "cost": float("nan"),
                        "wall_time_s": float("nan"),
                    }
                    status = "error"
                    error = repr(e)
                    print(f"  ERROR: {error}")

                row = {
                    "task": task_name,
                    "algorithm": alg_name,
                    "seed": seed,
                    "num_samples": args.num_samples,
                    "status": status,
                    "error": error,
                    **metrics,
                }
                append_row(args.output, row, fieldnames)
                cost = metrics.get("cost", float("nan"))
                wt = metrics.get("wall_time_s", float("nan"))
                print(f"  -> cost={cost:.4f}, wall_time={wt:.2f}s")
    elapsed = time.time() - overall_start
    print(f"Done. {total} trials in {elapsed:.1f}s. Output: {args.output}")


def make_controller_and_task(
    spec: TaskSpec, alg_name: str, seed: int, num_samples: int
) -> Tuple[Task, SamplingBasedController]:
    """Construct a fresh task + controller for one trial."""
    task = spec.factory()
    ctrl = make_algorithm(
        alg_name,
        task,
        num_samples=num_samples,
        plan_horizon=spec.plan_horizon,
        num_knots=spec.num_knots,
        spline_type=spec.spline_type,
        seed=seed,
    )
    return task, ctrl


def sample_initial_condition(spec: TaskSpec, seed: int):
    """Deterministically sample an initial condition for the given seed."""
    rng = jax.random.key(seed)
    return spec.ic_sampler(rng)
