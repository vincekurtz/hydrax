from typing import Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base import SamplingBasedController, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task


@dataclass
class CEMParams:
    """Policy parameters for the cross-entropy method.

    Attributes:
        mean: The mean of the control distribution, μ = [u₀, u₁, ..., ].
        cov: The (diagonal) covariance of the control distribution.
        rng: The pseudo-random number generator key.
    """

    mean: jax.Array
    cov: jax.Array
    rng: jax.Array


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
    ):
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
        """
        super().__init__(task, num_randomizations, risk_strategy, seed)
        self.num_samples = num_samples
        self.sigma_min = sigma_min
        self.sigma_start = sigma_start
        self.num_elites = num_elites

    def init_params(self, seed: int = 0) -> CEMParams:
        """Initialize the policy parameters."""
        rng = jax.random.key(seed)
        mean = jnp.zeros((self.task.planning_horizon, self.task.model.nu))
        cov = jnp.full_like(mean, self.sigma_start)
        return CEMParams(mean=mean, cov=cov, rng=rng)

    def sample_controls(self, params: CEMParams) -> Tuple[jax.Array, CEMParams]:
        """Sample a control sequence."""
        rng, sample_rng = jax.random.split(params.rng)
        noise = jax.random.normal(
            sample_rng,
            (
                self.num_samples,
                self.task.planning_horizon,
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
        mean = jnp.mean(rollouts.controls[elites], axis=0)
        cov = jnp.maximum(
            jnp.std(rollouts.controls[elites], axis=0), self.sigma_min
        )

        return params.replace(mean=mean, cov=cov)

    def get_action(self, params: CEMParams, t: float) -> jax.Array:
        """Get the control action for the current time step, zero order hold."""
        idx_float = t / self.task.dt  # zero order hold
        idx = jnp.floor(idx_float).astype(jnp.int32)
        return params.mean[idx]
