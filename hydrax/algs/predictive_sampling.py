from typing import Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base import SamplingBasedController, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task


@dataclass
class PSParams:
    """Policy parameters for predictive sampling.

    Attributes:
        mean: The mean of the control distribution, μ = [u₀, u₁, ..., ].
        rng: The pseudo-random number generator key.
    """

    mean: jax.Array
    rng: jax.Array


class PredictiveSampling(SamplingBasedController):
    """A simple implementation of https://arxiv.org/abs/2212.00541."""

    def __init__(
        self,
        task: Task,
        num_samples: int,
        noise_level: float,
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
    ):
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control tapes to sample.
            noise_level: The scale of Gaussian noise to add to sampled controls.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
                           Defaults to average cost.
            seed: The random seed for domain randomization.
        """
        super().__init__(task, num_randomizations, risk_strategy, seed)
        self.noise_level = noise_level
        self.num_samples = num_samples

    def init_params(self, seed: int = 0) -> PSParams:
        """Initialize the policy parameters."""
        rng = jax.random.key(seed)
        mean = jnp.zeros((self.task.planning_horizon, self.task.model.nu))
        return PSParams(mean=mean, rng=rng)

    def sample_controls(self, params: PSParams) -> Tuple[jax.Array, PSParams]:
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
        controls = params.mean + self.noise_level * noise

        # The original mean of the distribution is included as a sample
        controls = jnp.append(params.mean[None, ...], controls, axis=0)

        return controls, params.replace(rng=rng)

    def update_params(self, params: PSParams, rollouts: Trajectory) -> PSParams:
        """Update the policy parameters by choosing the lowest-cost rollout."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        best_idx = jnp.argmin(costs)
        mean = rollouts.controls[best_idx]
        return params.replace(mean=mean)

    def get_action(self, params: PSParams, t: float) -> jax.Array:
        """Get the control action for the current time step, zero order hold."""
        idx_float = t / self.task.dt  # zero order hold
        idx = jnp.floor(idx_float).astype(jnp.int32)
        return params.mean[idx]
