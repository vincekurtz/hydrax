"""Controller-related utilities for benchmarking."""

from typing import Dict, Any, Type

import evosax

from hydrax.algs.cem import CEM
from hydrax.algs.mppi import MPPI
from hydrax.algs.predictive_sampling import PredictiveSampling
from hydrax.algs.evosax import Evosax
from hydrax.risk import WorstCase, AverageCost


def get_default_controller_configs() -> Dict[str, Dict[str, Any]]:
    """Get default configurations for all controllers.

    Returns:
        Dictionary mapping controller names to their configurations.
    """
    return {
        "PredictiveSampling": {
            "class": PredictiveSampling,
            "params": {
                "num_samples": 512,
                "noise_level": 0.1,
                "num_randomizations": 10,
                "plan_horizon": 0.5,
                "spline_type": "zero",
                "num_knots": 11,
            },
        },
        "MPPI": {
            "class": MPPI,
            "params": {
                "num_samples": 512,
                "noise_level": 0.1,
                "temperature": 0.01,
                "plan_horizon": 0.25,
                "spline_type": "zero",
                "num_knots": 11,
            },
        },
        "CEM": {
            "class": CEM,
            "params": {
                "num_samples": 512,
                "num_elites": 20,
                "sigma_start": 0.3,
                "sigma_min": 0.05,
                "explore_fraction": 0.5,
                "plan_horizon": 0.25,
                "spline_type": "zero",
                "num_knots": 11,
            },
        },
        # Uncomment to include Evosax controllers
        # "CMA_ES": {
        #     "class": Evosax,
        #     "params": {
        #         "optimizer": evosax.CMA_ES,
        #         "num_samples": 16,
        #         "plan_horizon": 0.25,
        #         "spline_type": "zero",
        #         "num_knots": 11,
        #     },
        # },
        # "Sep_CMA_ES": {
        #     "class": Evosax,
        #     "params": {
        #         "optimizer": evosax.Sep_CMA_ES,
        #         "num_samples": 16,
        #         "plan_horizon": 0.25,
        #         "spline_type": "zero",
        #         "num_knots": 11,
        #     },
        # },
    }
