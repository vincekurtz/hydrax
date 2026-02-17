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
        mean: The mean of the control spline knot distribution, μ = [u₀, ...].
        rng: The pseudo-random number generator key.
        opt_iteration: The optimization iteration number.
    """

    opt_iteration: int


class DIAL(SamplingBasedController):
    """Diffusion-Inspired Annealing for Legged MPC (DIAL) based on https://arxiv.org/abs/2409.15610.

    DIAL-MPC is MPPI with a dual-loop, annealed sampling covariance that:
        - Decreases across optimisation iterations (trajectory-level annealing).
        - Increases along the planning horizon (action-level annealing).

    The noise level is given by:

        σ[i,h] = σ₀ * exp(-i/(β₁*N) - (H-h)/(β₂*H))

    where:
        - σ₀ is the tunable `noise_level`,
        - β₁ is the tunable `beta_opt_iter`,
        - β₂ is the tunable `beta_horizon`,
        - i in {0,...,N-1} is the optimisation iteration,
        - h in {0,...,H} indexes the knot along the horizon, and
        - N is the number of iterations and H is the number of knots.
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        noise_level: float,
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
            noise_level: The initial noise level σ₀ for the sampling covariance.
            beta_opt_iter: The temperature parameter β₁ for the trajectory-level
                          annealing. Higher values will result in less
                          annealing over optimisation iterations (exploration).
            beta_horizon: The temperature parameter β₂ for the action-level
                          annealing. Higher values will result in less
                          variation over the planning horizon (exploitation).
            temperature: The MPPI temperature parameter λ. Higher values take a
                         more even average over the samples.
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
        self.noise_level = noise_level
        self.beta_opt_iter = beta_opt_iter
        self.beta_horizon = beta_horizon
        assert self.beta_opt_iter > 0.0, "beta_opt_iter must be positive"
        assert self.beta_horizon > 0.0, "beta_horizon must be positive"
        self.num_samples = num_samples
        self.temperature = temperature

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> DIALParams:
        """Initialize the policy parameters."""
        _params = super().init_params(initial_knots, seed)

        return DIALParams(
            tk=_params.tk, mean=_params.mean, rng=_params.rng, opt_iteration=0
        )

    def sample_knots(self, params: DIALParams) -> Tuple[jax.Array, DIALParams]:
        """Sample control knots.

        Anneals noise and adds it to the mean control sequence, then increments
        the optimisation iteration number.
        """
        rng, sample_rng = jax.random.split(params.rng)

        noise = jax.random.normal(
            sample_rng,
            (self.num_samples, self.num_knots, self.task.model.nu),
        )

        noise_level = self.noise_level * jnp.exp(
            -(params.opt_iteration) / (self.beta_opt_iter * self.iterations)
            - (self.num_knots - 1 - jnp.arange(self.num_knots))
            / (self.beta_horizon * self.num_knots)
        )

        controls = params.mean + noise_level[None, :, None] * noise

        # Increment opt_iteration, wrapping after maximum iterations reached
        return controls, params.replace(
            opt_iteration=(params.opt_iteration + 1) % self.iterations,
            rng=rng,
        )

    def update_params(
        self, params: DIALParams, rollouts: Trajectory
    ) -> DIALParams:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        # N.B. jax.nn.softmax takes care of details like baseline subtraction.
        weights = jax.nn.softmax(-costs / self.temperature, axis=0)
        mean = jnp.sum(weights[:, None, None] * rollouts.knots, axis=0)
        return params.replace(mean=mean)
