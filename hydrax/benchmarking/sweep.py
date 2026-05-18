"""Hyperparameter sweep across (task, algorithm, hyperparams).

Each trial uses a single fixed initial condition (matching examples/) and a
single seed, so the cost number isolates the effect of hyperparameter
choices rather than IC randomness.

The sweep runs in two phases per task:
    Phase 1: PS, MPPI, CEM grids (independent).
    Phase 2: DIAL, MPPI-CMA, ER-CMA grids, fixing noise_level + temperature
             at MPPI's per-task best from Phase 1.

CMA-ES is intentionally excluded (no scalar hyperparameters exposed).
"""

import argparse
import csv
import itertools
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from evosax.algorithms.distribution_based import CMA_ES

from hydrax.alg_base import SamplingBasedController
from hydrax.algs import (
    CEM,
    DIAL,
    MPPI,
    ErCma,
    Evosax,
    MppiCma,
    PredictiveSampling,
)
from hydrax.benchmarking.runner import run_closed_loop
from hydrax.benchmarking.tasks import TASKS, TaskSpec
from hydrax.task_base import Task

# Columns written to the output CSV. Hyperparameters not used by a given
# algorithm are left blank.
HPARAM_COLS = [
    "noise_level",
    "temperature",
    "sigma_start",
    "sigma_min",
    "num_elites",
    "explore_fraction",
    "beta_opt_iter",
    "beta_horizon",
    "initial_noise_level",
    "minimum_noise_level",
    "maximum_noise_level",
    "covariance_adaptation_rate",
    "initial_entropy_bonus",
    "final_entropy_bonus",
]

FIELDNAMES = (
    ["task", "algorithm", "seed", "num_samples"]
    + HPARAM_COLS
    + ["cost", "wall_time_s", "status", "error"]
)


# --- Hyperparameter grids -----------------------------------------------


def ps_grid() -> List[Dict[str, Any]]:
    """PS grid: noise_level only."""
    return [{"noise_level": nl} for nl in [0.05, 0.1, 0.3, 0.5, 1.0]]


def mppi_grid() -> List[Dict[str, Any]]:
    """MPPI grid: noise_level x temperature."""
    return [
        {"noise_level": nl, "temperature": t}
        for nl, t in itertools.product(
            [0.1, 0.3, 0.5, 1.0], [0.01, 0.1, 0.5, 1.0]
        )
    ]


def cem_grid() -> List[Dict[str, Any]]:
    """CEM grid: sigma_start x num_elites; other params fixed."""
    return [
        {
            "sigma_start": s,
            "num_elites": e,
            "sigma_min": 1e-3,
            "explore_fraction": 0.0,
        }
        for s, e in itertools.product([0.1, 0.3, 0.5, 1.0], [4, 8, 16])
    ]


def dial_grid(noise_level: float, temperature: float) -> List[Dict[str, Any]]:
    """DIAL grid: beta_opt_iter x beta_horizon at MPPI's best (noise, temp)."""
    return [
        {
            "noise_level": noise_level,
            "temperature": temperature,
            "beta_opt_iter": boi,
            "beta_horizon": bh,
        }
        for boi, bh in itertools.product([0.5, 1.0, 2.0], [0.5, 1.0, 2.0])
    ]


def mppi_cma_grid(noise_level: float) -> List[Dict[str, Any]]:
    """MPPI-CMA grid: covariance_adaptation_rate x temperature."""
    return [
        {
            "initial_noise_level": noise_level,
            "minimum_noise_level": 1e-3,
            "temperature": t,
            "covariance_adaptation_rate": alpha,
        }
        for alpha, t in itertools.product(
            [0.05, 0.1, 0.3, 1.0], [0.01, 0.1, 0.5, 1.0]
        )
    ]


def er_cma_grid(noise_level: float) -> List[Dict[str, Any]]:
    """ER-CMA grid: entropy bonuses x covariance_adaptation_rate x temperature.

    Skips (initial=0, final=0) (equivalent to MPPI-CMA) and any (i, f)
    where f < i (entropy should not decrease across the horizon).
    """
    bonuses = [0.0, 0.1, 0.3, 0.5]
    alphas = [0.05, 0.1, 0.3, 1.0]
    temps = [0.01, 0.1, 0.5, 1.0]
    configs = []
    for ieb, feb in itertools.product(bonuses, bonuses):
        if feb < ieb:
            continue
        if ieb == 0.0 and feb == 0.0:
            continue
        for alpha, t in itertools.product(alphas, temps):
            configs.append(
                {
                    "initial_noise_level": noise_level,
                    "minimum_noise_level": 1e-3,
                    "maximum_noise_level": 1.0,
                    "temperature": t,
                    "initial_entropy_bonus": ieb,
                    "final_entropy_bonus": feb,
                    "covariance_adaptation_rate": alpha,
                }
            )
    return configs


# --- Algorithm constructors --------------------------------------------


def build_controller(  # noqa: PLR0911
    name: str,
    task: Task,
    hparams: Dict[str, Any],
    num_samples: int,
    plan_horizon: float,
    num_knots: int,
    spline_type: str,
    seed: int,
) -> SamplingBasedController:
    """Construct an algorithm instance from a hyperparameter dict."""
    common = dict(
        task=task,
        num_samples=num_samples,
        plan_horizon=plan_horizon,
        spline_type=spline_type,
        num_knots=num_knots,
        seed=seed,
    )
    if name == "ps":
        return PredictiveSampling(noise_level=hparams["noise_level"], **common)
    if name == "mppi":
        return MPPI(
            noise_level=hparams["noise_level"],
            temperature=hparams["temperature"],
            **common,
        )
    if name == "cem":
        return CEM(
            num_elites=hparams["num_elites"],
            sigma_start=hparams["sigma_start"],
            sigma_min=hparams["sigma_min"],
            explore_fraction=hparams["explore_fraction"],
            **common,
        )
    if name == "dial":
        return DIAL(
            noise_level=hparams["noise_level"],
            beta_opt_iter=hparams["beta_opt_iter"],
            beta_horizon=hparams["beta_horizon"],
            temperature=hparams["temperature"],
            **common,
        )
    if name == "cmaes":
        return Evosax(optimizer=CMA_ES, **common)
    if name == "mppi_cma":
        return MppiCma(
            initial_noise_level=hparams["initial_noise_level"],
            minimum_noise_level=hparams["minimum_noise_level"],
            covariance_adaptation_rate=hparams["covariance_adaptation_rate"],
            temperature=hparams["temperature"],
            **common,
        )
    if name == "er_cma":
        return ErCma(
            initial_noise_level=hparams["initial_noise_level"],
            minimum_noise_level=hparams["minimum_noise_level"],
            maximum_noise_level=hparams["maximum_noise_level"],
            initial_entropy_bonus=hparams["initial_entropy_bonus"],
            final_entropy_bonus=hparams["final_entropy_bonus"],
            covariance_adaptation_rate=hparams["covariance_adaptation_rate"],
            temperature=hparams["temperature"],
            **common,
        )
    raise KeyError(f"Unknown algorithm '{name}'")


# --- Sweep driver -------------------------------------------------------


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


def run_one_trial(
    spec: TaskSpec,
    alg_name: str,
    hparams: Dict[str, Any],
    num_samples: int,
    seed: int,
) -> Tuple[float, float]:
    """Construct algorithm and run one closed-loop episode."""
    task = spec.factory()
    qpos, qvel, mocap_pos = spec.fixed_ic(task)
    ctrl = build_controller(
        alg_name,
        task,
        hparams,
        num_samples=num_samples,
        plan_horizon=spec.plan_horizon,
        num_knots=spec.num_knots,
        spline_type=spec.spline_type,
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
    return cost, wall_time


def run_config(
    spec: TaskSpec,
    task_name: str,
    alg_name: str,
    hparams: Dict[str, Any],
    num_samples: int,
    seed: int,
    output_path: str,
) -> Optional[float]:
    """Run one (task, algorithm, hparams) config; append a row; return cost."""
    try:
        cost, wall_time = run_one_trial(
            spec, alg_name, hparams, num_samples, seed
        )
        status, error = "ok", ""
    except Exception as e:  # noqa: BLE001
        cost = float("nan")
        wall_time = float("nan")
        status = "error"
        error = repr(e)
        print(f"    ERROR: {error}", flush=True)

    row = {
        "task": task_name,
        "algorithm": alg_name,
        "seed": seed,
        "num_samples": num_samples,
        "cost": cost,
        "wall_time_s": wall_time,
        "status": status,
        "error": error,
    }
    for col in HPARAM_COLS:
        row[col] = hparams.get(col, "")
    append_row(output_path, row)
    print(
        f"    {alg_name:9s} {hparams} -> cost={cost:.4f}, t={wall_time:.2f}s",
        flush=True,
    )
    return None if status == "error" else cost


def sweep_task(
    task_name: str,
    spec: TaskSpec,
    num_samples: int,
    seed: int,
    output_path: str,
) -> None:
    """Run the full two-phase sweep for a single task."""
    print(f"\n=== {task_name} ===", flush=True)

    # Phase 1: independent grids.
    print(f"  [phase 1] PS ({len(ps_grid())} configs)", flush=True)
    for hp in ps_grid():
        run_config(spec, task_name, "ps", hp, num_samples, seed, output_path)

    print(f"  [phase 1] MPPI ({len(mppi_grid())} configs)", flush=True)
    mppi_costs: List[Tuple[float, Dict[str, Any]]] = []
    for hp in mppi_grid():
        cost = run_config(
            spec, task_name, "mppi", hp, num_samples, seed, output_path
        )
        if cost is not None:
            mppi_costs.append((cost, hp))

    print(f"  [phase 1] CEM ({len(cem_grid())} configs)", flush=True)
    for hp in cem_grid():
        run_config(spec, task_name, "cem", hp, num_samples, seed, output_path)

    if not mppi_costs:
        print(
            f"  [phase 2] SKIPPED ({task_name}): no successful MPPI trial",
            flush=True,
        )
        return

    best_cost, best_mppi = min(mppi_costs, key=lambda x: x[0])
    nl, temp = best_mppi["noise_level"], best_mppi["temperature"]
    print(
        f"  [phase 2] best MPPI: noise_level={nl}, temperature={temp} "
        f"(cost={best_cost:.4f})",
        flush=True,
    )

    print(f"  [phase 2] DIAL ({len(dial_grid(nl, temp))} configs)", flush=True)
    for hp in dial_grid(nl, temp):
        run_config(spec, task_name, "dial", hp, num_samples, seed, output_path)

    print(
        f"  [phase 2] MPPI-CMA ({len(mppi_cma_grid(nl))} configs)",
        flush=True,
    )
    for hp in mppi_cma_grid(nl):
        run_config(
            spec, task_name, "mppi_cma", hp, num_samples, seed, output_path
        )

    print(
        f"  [phase 2] ER-CMA ({len(er_cma_grid(nl))} configs)",
        flush=True,
    )
    for hp in er_cma_grid(nl):
        run_config(
            spec, task_name, "er_cma", hp, num_samples, seed, output_path
        )


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Hyperparameter sweep.")
    p.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASKS.keys()),
        choices=list(TASKS.keys()),
        help="Tasks to sweep.",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=256,
        help="Number of samples per optimize() call.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Single seed used for all trials.",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="CSV path for sweep results (rows appended).",
    )
    args = p.parse_args()

    write_header_if_needed(args.output)
    start = time.time()
    for task_name in args.tasks:
        sweep_task(
            task_name,
            TASKS[task_name],
            args.num_samples,
            args.seed,
            args.output,
        )
    elapsed = time.time() - start
    print(f"\nDone. Total wall time: {elapsed:.1f}s. Output: {args.output}")


if __name__ == "__main__":
    main()
