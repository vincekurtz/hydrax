from typing import Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base import SamplingBasedController, Trajectory
from hydrax.task_base import Task


@dataclass
class MPPIParams:
    """Policy parameters for model-predictive path integral control.

    Attributes:
        mean: The mean of the control distribution, μ = [u₀, u₁, ..., ].
        rng: The pseudo-random number generator key.
    """

    mean: jax.Array
    rng: jax.Array


class MPPI(SamplingBasedController):
    """Model-predictive path integral control.

    Implements "MPPI-generic" as described in https://arxiv.org/abs/2409.07563.
    Unlike the original MPPI derivation, this does not assume stochastic,
    control-affine dynamics or a separable cost function that is quadratic in
    control.
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        noise_level: float,
        temperature: float,
    ):
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control sequences to sample.
            noise_level: The scale of Gaussian noise to add to sampled controls.
            temperature: The temperature parameter λ. Higher values take a more
                         even average over the samples.
        """
        super().__init__(task)
        self.noise_level = noise_level
        self.num_samples = num_samples
        self.temperature = temperature

    def init_params(self, seed: int = 0) -> MPPIParams:
        """Initialize the policy parameters."""
        rng = jax.random.key(seed)
        mean = jnp.zeros((self.task.planning_horizon - 1, self.task.model.nu))
        return MPPIParams(mean=mean, rng=rng)

    def sample_controls(
        self, params: MPPIParams
    ) -> Tuple[jax.Array, MPPIParams]:
        """Sample a control sequence."""
        rng, sample_rng = jax.random.split(params.rng)
        noise = jax.random.normal(
            sample_rng,
            (
                self.num_samples,
                self.task.planning_horizon - 1,
                self.task.model.nu,
            ),
        )
        controls = params.mean + self.noise_level * noise
        return controls, params.replace(rng=rng)

    def update_params(
        self, params: MPPIParams, rollouts: Trajectory
    ) -> MPPIParams:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)
        # N.B. jax.nn.softmax takes care of details like baseline subtraction.
        weights = jax.nn.softmax(-costs / self.temperature, axis=0)
        mean = jnp.sum(weights[:, None, None] * rollouts.controls, axis=0)
        return params.replace(mean=mean)

    def get_action(self, params: MPPIParams, t: float) -> jax.Array:
        """Get the control action for the current time step, zero order hold."""
        idx_float = t / self.task.dt  # zero order hold
        idx = jnp.floor(idx_float).astype(jnp.int32)
        return params.mean[idx]
