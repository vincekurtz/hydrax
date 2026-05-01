from typing import Literal, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task


@dataclass
class MPPICMAParams(SamplingParams):
    """Policy parameters for MPPI with Covariance Matrix Adaptation (CMA).

    Attributes:
        tk: The knot times of the control spline.
        mean: The mean of the control spline knot distribution, μ = [u₀, ...],
              with shape (num_knots, control_dim).
        cov: The covariance of the control spline knot distribution, with shape
             (num_knots, control_dim, control_dim).
        rng: The pseudo-random number generator key.
    """
    tk: jax.Array
    mean: jax.Array
    cov: jax.Array
    rng: jax.Array


class MPPICMA(SamplingBasedController):
    """Model-predictive path integral control with covariance matrix adaptation.

    Implements the block-diagonal variant of MPPI-CMA, as described in
    https://arxiv.org/abs/2506.22087.
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        temperature: float,
        initial_noise_level: float,
        covariance_adaptation_rate: float = 0.1,
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
    ) -> None:
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control sequences to sample.
            temperature: The temperature parameter λ. Higher values take a more
                         even average over the samples.
            initial_noise_level: The initial covariance of the control
                         distribution.
            covariance_adaptation_rate: The learning rate for covariance
                                        adaptation.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
                           Defaults to average cost.
            seed: The random seed for domain randomization.
            plan_horizon: The time horizon for the rollout in seconds.
            spline_type: The type of spline used for control interpolation.
                         Defaults to "zero" (zero-order hold).
            num_knots: The number of knots in the control spline.
            iterations: The number of optimization iterations to perform.
        """
        super().__init__(
            task,
            num_randomizations=num_randomizations,
            risk_strategy=risk_strategy,
            seed=seed,
            plan_horizon=plan_horizon,
            spline_type=spline_type,
            num_knots=num_knots,
            iterations=iterations,
        )
        self.initial_noise_level = initial_noise_level
        self.alpha = covariance_adaptation_rate
        self.num_samples = num_samples
        self.temperature = temperature

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> MPPICMAParams:
        """Initialize the policy parameters."""
        _params = super().init_params(initial_knots, seed)

        cov = jnp.eye(self.task.model.nu) * self.initial_noise_level
        cov = jnp.tile(cov[None], (self.num_knots, 1, 1))

        return MPPICMAParams(
            tk=_params.tk, mean=_params.mean, cov=cov, rng=_params.rng
        )

    def sample_knots(
        self, params: MPPICMAParams
    ) -> Tuple[jax.Array, MPPICMAParams]:
        """Sample a control sequence."""
        rng, sample_rng = jax.random.split(params.rng)
        noise = jax.random.multivariate_normal(
            sample_rng,
            mean=jnp.zeros(self.task.model.nu),
            cov=params.cov,
            shape=(self.num_samples, self.num_knots),
        )  # shape (num_samples, num_knots, control_dim)
        controls = params.mean + self.initial_noise_level * noise
        return controls, params.replace(rng=rng)

    def update_params(
        self, params: MPPICMAParams, rollouts: Trajectory
    ) -> MPPICMAParams:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        # N.B. jax.nn.softmax takes care of details like baseline subtraction.
        weights = jax.nn.softmax(-costs / self.temperature, axis=0)

        # Difference between samples and mean,
        # shape (num_samples, num_knots, control_dim)
        delta = rollouts.knots - params.mean

        # Outer product of deltas, 
        # shape (num_samples, num_knots, control_dim, control_dim)
        outer_product = jnp.einsum("ijk,ijl->ijkl", delta, delta)

        # CMA update
        new_cov = jnp.einsum("i,ijkl->jkl", weights, outer_product)
        new_cov = (1 - self.alpha) * params.cov + self.alpha * new_cov

        # Mean update (same as standard MPPI)
        mean = jnp.sum(weights[:, None, None] * rollouts.knots, axis=0)

        return params.replace(mean=mean, cov=new_cov)
