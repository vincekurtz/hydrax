from typing import Literal, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task


@dataclass
class CBOParams(SamplingParams):
    """Policy parameters for consensus-based optimization.

    Attributes:
        tk: The knot times of the control spline.
        mean: The consensus of control spline knots, μ = [u₀, ...].
        rng: The pseudo-random number generator key.
        samples: The sampled control spline knots.
    """

    samples: jax.Array  # shape (N, K, nu)


class CBO(SamplingBasedController):
    """Consensus-based optimization.

    CBO (https://en.wikipedia.org/wiki/Consensus_based_optimization) simulates
    a swarm of particles Uᵢ according to a stochastic differential equation

        dUᵢ = - λ (Uᵢ - μ) dt + σ |Uᵢ - μ| dBᵢ,

    where μ = ∑ᵢ wᵢ Uᵢ is the consensus, wᵢ = exp(-J(Uᵢ)/T) / ∑ⱼ exp(-J(Uⱼ)/T)
    is the normalized weight of each particle, J(Uᵢ) is the cost of the control
    sequence Uᵢ, and Bᵢ is Brownian noise.

    This algorithm has global convergence properties for nonconvex problems in
    the mean-field (infinite particle) limit (https://arxiv.org/abs/2103.15130).
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        initial_noise_level: float,
        temperature: float,
        consensus_weight: float,
        noise_weight: float,
        step_size: float = 0.1,
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
            initial_noise_level: The scale of Gaussian noise for the initial
                                 control samples.
            temperature: The temperature parameter T. Higher values take a more
                         even average over the samples.
            consensus_weight: The weight λ on the consensus term in the SDE.
            noise_weight: The weight σ on the noise term in the SDE.
            step_size: The step size for Euler-Maruyama integration of the SDE.
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
        self.num_samples = num_samples
        self.temperature = temperature
        self.consensus_weight = consensus_weight
        self.noise_weight = noise_weight
        self.step_size = step_size

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> CBOParams:
        """Initialize the policy parameters."""
        _params = super().init_params(initial_knots, seed)

        rng, sample_rng = jax.random.split(_params.rng)
        noise = jax.random.normal(
            sample_rng,
            (
                self.num_samples,
                self.num_knots,
                self.task.model.nu,
            ),
        )
        samples = _params.mean + self.initial_noise_level * noise

        return CBOParams(
            tk=_params.tk, mean=_params.mean, rng=rng, samples=samples
        )

    def sample_knots(self, params: CBOParams) -> Tuple[jax.Array, CBOParams]:
        """Return the current particle positions."""
        return params.samples, params

    def update_params(
        self, params: CBOParams, rollouts: Trajectory
    ) -> CBOParams:
        """Update the consensus and take an Euler-Maruyama step of the SDE."""
        # Compute the consensus point μ = ∑ᵢ wᵢ Uᵢ
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        weights = jax.nn.softmax(-costs / self.temperature, axis=0)
        mean = jnp.sum(weights[:, None, None] * rollouts.knots, axis=0)

        # Step the SDE for each particle:
        #  Uᵢ ← Uᵢ - λ (Uᵢ - μ) Δt + σ |Uᵢ - μ| √Δt ξᵢ, ξᵢ ~ N(0, I)
        rng, noise_rng = jax.random.split(params.rng)
        noise = jax.random.normal(noise_rng, rollouts.knots.shape)
        deviation = rollouts.knots - mean
        samples = (
            rollouts.knots
            - self.consensus_weight * deviation * self.step_size
            + self.noise_weight
            * jnp.abs(deviation)  # Ansiotropic CBO, regular norm also possible
            * jnp.sqrt(self.step_size)
            * noise
        )

        return params.replace(mean=mean, rng=rng, samples=samples)
