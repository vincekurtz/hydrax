"""Per-(task, algorithm) tuned hyperparameters from `results/sweep_all.csv`.

These are the configurations that minimized closed-loop cost in the
single-seed, fixed-IC sweep. They are used by the validation runner to
measure performance across randomized ICs and seeds.

CMA-ES (Evosax) is included with an empty hparam dict; the Evosax
defaults are treated as "tuned" for the purposes of this benchmark.
"""

from typing import Any, Dict

TUNED_HPARAMS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "pendulum": {
        "cmaes": {},
        "ps": {"noise_level": 1.0},
        "mppi": {"noise_level": 1.0, "temperature": 0.01},
        "cem": {
            "sigma_start": 0.5,
            "sigma_min": 0.001,
            "num_elites": 4,
            "explore_fraction": 0.0,
        },
        "dial": {
            "noise_level": 1.0,
            "temperature": 0.01,
            "beta_opt_iter": 0.5,
            "beta_horizon": 2.0,
        },
        "mppi_cma": {
            "initial_noise_level": 1.0,
            "minimum_noise_level": 0.001,
            "temperature": 0.01,
            "covariance_adaptation_rate": 0.05,
        },
        "er_cma": {
            "initial_noise_level": 1.0,
            "minimum_noise_level": 0.001,
            "maximum_noise_level": 1.0,
            "temperature": 0.01,
            "covariance_adaptation_rate": 0.05,
            "initial_entropy_bonus": 0.5,
            "final_entropy_bonus": 0.5,
        },
    },
    "cart_pole": {
        "cmaes": {},
        "ps": {"noise_level": 0.3},
        "mppi": {"noise_level": 0.3, "temperature": 0.01},
        "cem": {
            "sigma_start": 0.1,
            "sigma_min": 0.001,
            "num_elites": 4,
            "explore_fraction": 0.0,
        },
        "dial": {
            "noise_level": 0.3,
            "temperature": 0.01,
            "beta_opt_iter": 0.5,
            "beta_horizon": 2.0,
        },
        "mppi_cma": {
            "initial_noise_level": 0.3,
            "minimum_noise_level": 0.001,
            "temperature": 0.01,
            "covariance_adaptation_rate": 0.05,
        },
        "er_cma": {
            "initial_noise_level": 0.3,
            "minimum_noise_level": 0.001,
            "maximum_noise_level": 1.0,
            "temperature": 0.01,
            "covariance_adaptation_rate": 0.1,
            "initial_entropy_bonus": 0.3,
            "final_entropy_bonus": 0.5,
        },
    },
    "double_cart_pole": {
        "cmaes": {},
        "ps": {"noise_level": 0.1},
        "mppi": {"noise_level": 0.1, "temperature": 0.01},
        "cem": {
            "sigma_start": 0.1,
            "sigma_min": 0.001,
            "num_elites": 4,
            "explore_fraction": 0.0,
        },
        "dial": {
            "noise_level": 0.1,
            "temperature": 0.01,
            "beta_opt_iter": 0.5,
            "beta_horizon": 2.0,
        },
        "mppi_cma": {
            "initial_noise_level": 0.1,
            "minimum_noise_level": 0.001,
            "temperature": 0.01,
            "covariance_adaptation_rate": 0.1,
        },
        "er_cma": {
            "initial_noise_level": 0.1,
            "minimum_noise_level": 0.001,
            "maximum_noise_level": 1.0,
            "temperature": 0.01,
            "covariance_adaptation_rate": 0.1,
            "initial_entropy_bonus": 0.1,
            "final_entropy_bonus": 0.5,
        },
    },
    "particle": {
        "cmaes": {},
        "ps": {"noise_level": 0.1},
        "mppi": {"noise_level": 1.0, "temperature": 0.01},
        "cem": {
            "sigma_start": 0.1,
            "sigma_min": 0.001,
            "num_elites": 4,
            "explore_fraction": 0.0,
        },
        "dial": {
            "noise_level": 1.0,
            "temperature": 0.01,
            "beta_opt_iter": 0.5,
            "beta_horizon": 2.0,
        },
        "mppi_cma": {
            "initial_noise_level": 1.0,
            "minimum_noise_level": 0.001,
            "temperature": 0.01,
            "covariance_adaptation_rate": 0.05,
        },
        "er_cma": {
            "initial_noise_level": 1.0,
            "minimum_noise_level": 0.001,
            "maximum_noise_level": 1.0,
            "temperature": 0.01,
            "covariance_adaptation_rate": 0.05,
            "initial_entropy_bonus": 0.1,
            "final_entropy_bonus": 0.1,
        },
    },
    "pusht": {
        "cmaes": {},
        "ps": {"noise_level": 0.3},
        "mppi": {"noise_level": 0.3, "temperature": 0.01},
        "cem": {
            "sigma_start": 0.3,
            "sigma_min": 0.001,
            "num_elites": 16,
            "explore_fraction": 0.0,
        },
        "dial": {
            "noise_level": 0.3,
            "temperature": 0.01,
            "beta_opt_iter": 0.5,
            "beta_horizon": 2.0,
        },
        "mppi_cma": {
            "initial_noise_level": 0.3,
            "minimum_noise_level": 0.001,
            "temperature": 0.01,
            "covariance_adaptation_rate": 0.05,
        },
        "er_cma": {
            "initial_noise_level": 0.3,
            "minimum_noise_level": 0.001,
            "maximum_noise_level": 1.0,
            "temperature": 0.01,
            "covariance_adaptation_rate": 0.1,
            "initial_entropy_bonus": 0.5,
            "final_entropy_bonus": 0.5,
        },
    },
    "walker": {
        "cmaes": {},
        "ps": {"noise_level": 0.5},
        "mppi": {"noise_level": 0.5, "temperature": 0.01},
        "cem": {
            "sigma_start": 0.5,
            "sigma_min": 0.001,
            "num_elites": 8,
            "explore_fraction": 0.0,
        },
        "dial": {
            "noise_level": 0.5,
            "temperature": 0.01,
            "beta_opt_iter": 0.5,
            "beta_horizon": 2.0,
        },
        "mppi_cma": {
            "initial_noise_level": 0.5,
            "minimum_noise_level": 0.001,
            "temperature": 0.1,
            "covariance_adaptation_rate": 0.05,
        },
        "er_cma": {
            "initial_noise_level": 0.5,
            "minimum_noise_level": 0.001,
            "maximum_noise_level": 1.0,
            "temperature": 0.1,
            "covariance_adaptation_rate": 0.1,
            "initial_entropy_bonus": 0.3,
            "final_entropy_bonus": 0.5,
        },
    },
    "cube": {
        "cmaes": {},
        "ps": {"noise_level": 1.0},
        "mppi": {"noise_level": 1.0, "temperature": 0.01},
        "cem": {
            "sigma_start": 1.0,
            "sigma_min": 0.001,
            "num_elites": 8,
            "explore_fraction": 0.0,
        },
        "dial": {
            "noise_level": 1.0,
            "temperature": 0.01,
            "beta_opt_iter": 2.0,
            "beta_horizon": 2.0,
        },
        "mppi_cma": {
            "initial_noise_level": 1.0,
            "minimum_noise_level": 0.001,
            "temperature": 0.01,
            "covariance_adaptation_rate": 0.05,
        },
        "er_cma": {
            "initial_noise_level": 1.0,
            "minimum_noise_level": 0.001,
            "maximum_noise_level": 1.0,
            "temperature": 0.01,
            "covariance_adaptation_rate": 0.3,
            "initial_entropy_bonus": 0.0,
            "final_entropy_bonus": 0.5,
        },
    },
    "humanoid_standup": {
        "cmaes": {},
        "ps": {"noise_level": 0.05},
        "mppi": {"noise_level": 1.0, "temperature": 0.5},
        "cem": {
            "sigma_start": 0.5,
            "sigma_min": 0.001,
            "num_elites": 16,
            "explore_fraction": 0.0,
        },
        "dial": {
            "noise_level": 1.0,
            "temperature": 0.5,
            "beta_opt_iter": 0.5,
            "beta_horizon": 2.0,
        },
        "mppi_cma": {
            "initial_noise_level": 1.0,
            "minimum_noise_level": 0.001,
            "temperature": 0.5,
            "covariance_adaptation_rate": 0.1,
        },
        "er_cma": {
            "initial_noise_level": 1.0,
            "minimum_noise_level": 0.001,
            "maximum_noise_level": 1.0,
            "temperature": 0.5,
            "covariance_adaptation_rate": 0.1,
            "initial_entropy_bonus": 0.5,
            "final_entropy_bonus": 0.5,
        },
    },
}
