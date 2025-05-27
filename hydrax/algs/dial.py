from typing import Literal, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task


@dataclass
class DIALParams(SamplingParams):
    """Policy parameters for Diffusion-Inspired Annealing for Legged MPC (DIAL).

    Same as SamplingParams, but with a different name for clarity.

    Attributes:
        tk: The knot times of the control spline.
        opt_iteration: The optimization iteration number.
        mean: The mean of the control spline knot distribution, μ = [u₀, ...].
        rng: The pseudo-random number generator key.
    """


class DIAL(SamplingBasedController):
    """DIAL MPC based on https://arxiv.org/abs/2409.15610"""

    def __init__(
        self,
        task: Task,
        num_samples: int,
        beta_opt_iter: float,
        beta_horizon: float,
        temperature: float,
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
            beta_opt_iter: The temperature parameter β₁ used in the noise schedule
                          for annealing the control sequence.
            beta_horizon: The temperature parameter β₂ used in the noise schedule
                              for annealing the planning horizon.
            temperature: The MPPI temperature parameter λ. Higher values take a more
                         even average over the samples.
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
            iterations=iterations,
        )
        self.beta_opt_iter = beta_opt_iter
        self.beta_horizon = beta_horizon
        self.num_samples = num_samples
        self.temperature = temperature

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> DIALParams:
        """Initialize the policy parameters."""
        _params = super().init_params(initial_knots, seed)

        return DIALParams(
            tk=_params.tk,
            opt_iteration=0,
            mean=_params.mean,
            rng=_params.rng,
        )

    def sample_knots(self, params: DIALParams) -> Tuple[jax.Array, DIALParams]:
        rng, sample_rng = jax.random.split(params.rng)

        noise = jax.random.normal(
            sample_rng,
            (self.num_samples, self.num_knots, self.task.model.nu),
        )

        noise_level = jnp.exp(
            -(params.opt_iteration) / (self.beta_opt_iter * self.iterations)
            - (self.num_knots - jnp.arange(self.num_knots))
            / (self.beta_horizon * self.num_knots)
        )

        controls = params.mean + noise_level[None, :, None] * noise
        return controls, params.replace(rng=rng)

    def update_params(
        self, params: DIALParams, rollouts: Trajectory
    ) -> DIALParams:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        # N.B. jax.nn.softmax takes care of details like baseline subtraction.
        weights = jax.nn.softmax(-costs / self.temperature, axis=0)
        mean = jnp.sum(weights[:, None, None] * rollouts.knots, axis=0)
        return params.replace(mean=mean)
