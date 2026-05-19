"""Single hyperparameter set per algorithm, applied across all tasks.

Used together with a uniform task-level setup (1 s plan horizon, 8 knots,
cubic spline) to compare algorithms on an apples-to-apples footing without
per-task tuning.
"""

from typing import Any, Dict

UNIFORM_HPARAMS: Dict[str, Dict[str, Any]] = {
    "ps": {"noise_level": 0.3},
    "mppi": {"noise_level": 0.3, "temperature": 0.1},
    "cem": {
        "sigma_start": 0.3,
        "sigma_min": 1e-3,
        "num_elites": 8,
        "explore_fraction": 0.0,
    },
    "dial": {
        "noise_level": 0.3,
        "temperature": 0.1,
        "beta_opt_iter": 0.5,
        "beta_horizon": 2.0,
    },
    "cmaes": {},
    "mppi_cma": {
        "initial_noise_level": 0.3,
        "minimum_noise_level": 1e-3,
        "temperature": 0.1,
        "covariance_adaptation_rate": 0.1,
    },
    "er_cma": {
        "initial_noise_level": 0.3,
        "minimum_noise_level": 1e-3,
        "maximum_noise_level": 1.0,
        "temperature": 0.1,
        "covariance_adaptation_rate": 0.1,
        "initial_entropy_bonus": 0.5,
        "final_entropy_bonus": 0.5,
    },
}

UNIFORM_TASK_SETUP: Dict[str, Any] = {
    "plan_horizon": 1.0,
    "num_knots": 8,
    "spline_type": "cubic",
}
