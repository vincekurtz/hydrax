from typing import Literal, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task


@dataclass
class CEMParams(SamplingParams):
    """Policy parameters for the cross-entropy method.

    Attributes:
        mean: The mean of the control distribution, μ = [u₀, u₁, ..., ].
        cov: The (diagonal) covariance of the control distribution.
        rng: The pseudo-random number generator key.
    """

    cov: jax.Array


class CEM(SamplingBasedController):
    """Cross-entropy method with diagonal covariance."""

    def __init__(
        self,
        task: Task,
        num_samples: int,
        num_elites: int,
        sigma_start: float,
        sigma_min: float,
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
    ) -> None:
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control sequences to sample.
            num_elites: The number of elite samples to keep at each iteration.
            sigma_start: The initial standard deviation for the controls.
            sigma_min: The minimum standard deviation for the controls.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
                           Defaults to average cost.
            seed: The random seed for domain randomization.
            plan_horizon: The time horizon for the rollout in seconds.
            spline_type: The type of spline used for control interpolation.
                         Defaults to "zero" (zero-order hold).
            num_knots: The number of knots in the control spline.
        """
        super().__init__(
            task,
            num_randomizations=num_randomizations,
            risk_strategy=risk_strategy,
            seed=seed,
            plan_horizon=plan_horizon,
            spline_type=spline_type,
            num_knots=num_knots,
        )
        self.num_samples = num_samples
        self.sigma_min = sigma_min
        self.sigma_start = sigma_start
        self.num_elites = num_elites

    def init_params(self, seed: int = 0) -> CEMParams:
        """Initialize the policy parameters."""
        _params = super().init_params(seed)
        cov = jnp.full_like(_params.mean, self.sigma_start)
        return CEMParams(
            tk=_params.tk, mean=_params.mean, cov=cov, rng=_params.rng
        )

    def sample_knots(self, params: CEMParams) -> Tuple[jax.Array, CEMParams]:
        """Sample a control sequence."""
        rng, sample_rng = jax.random.split(params.rng)
        noise = jax.random.normal(
            sample_rng,
            (
                self.num_samples,
                self.num_knots,
                self.task.model.nu,
            ),
        )
        controls = params.mean + params.cov * noise
        return controls, params.replace(rng=rng)

    def update_params(
        self, params: CEMParams, rollouts: Trajectory
    ) -> CEMParams:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps

        # Sort the costs and get the indices of the elites.
        indices = jnp.argsort(costs)
        elites = indices[: self.num_elites]

        # The new proposal distribution is a Gaussian fit to the elites.
        mean = jnp.mean(rollouts.knots[elites], axis=0)
        cov = jnp.maximum(
            jnp.std(rollouts.knots[elites], axis=0), self.sigma_min
        )
        return params.replace(mean=mean, cov=cov)
