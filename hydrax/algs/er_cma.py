from typing import Literal, Optional, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task


@dataclass
class ErCmaParams(SamplingParams):
    """Policy parameters for entropy-regularized covariance matrix adaptation.

    Attributes:
        tk: The knot times of the control spline.
        mean: The mean of the control spline knot distribution, μ = [u₀, ...],
              with shape (num_knots, control_dim).
        rng: The pseudo-random number generator key.
        covariance: The covariance of the control spline knot distribution, with
                    shape (num_knots, control_dim, control_dim).
    """

    covariance: jax.Array


class ErCma(SamplingBasedController):
    """Entropy-regularized covariance matrix adaptation."""

    def __init__(
        self,
        task: Task,
        num_samples: int,
        initial_noise_level: float,
        temperature: float,
        minimum_noise_level: Optional[float] = None,
        maximum_noise_level: Optional[float] = None,
        initial_entropy_bonus: float = 0.0,
        final_entropy_bonus: float = 0.0,
        covariance_adaptation_rate: float = 0.1,
        num_randomizations: int = 1,
        risk_strategy: Optional[RiskStrategy] = None,
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
            initial_noise_level: The initial standard deviation of the control
                         distribution.
            temperature: The temperature parameter λ. Higher values take a more
                         even average over the samples.
            minimum_noise_level: The minimum noise level, enforced by bounding
                         the eigenvalues of the covariance matrix. Defaults to
                         initial_noise_level.
            maximum_noise_level: The maximum noise level, enforced by bounding
                         the eigenvalues of the covariance matrix. Defaults to
                         initial_noise_level.
            initial_entropy_bonus: Entropy bonus at the start of the rollout,
                                   must be in [0, 1). Higher values encourage
                                   more exploration.
            final_entropy_bonus: Entropy bonus at the end of the rollout,
                                 must be in [0, 1). Higher values encourage more
                                 exploration.
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
        self.minimum_noise_level = (
            minimum_noise_level
            if minimum_noise_level is not None
            else initial_noise_level
        )
        self.maximum_noise_level = (
            maximum_noise_level            
            if maximum_noise_level is not None
            else initial_noise_level
        )

        assert 0.0 <= initial_entropy_bonus < 1.0, (
            "initial_entropy_bonus must be in [0, 1)"
        )
        assert 0.0 <= final_entropy_bonus < 1.0, (
            "final_entropy_bonus must be in [0, 1)"
        )
        self.initial_entropy_bonus = initial_entropy_bonus
        self.final_entropy_bonus = final_entropy_bonus

        self.alpha = covariance_adaptation_rate
        self.num_samples = num_samples
        self.temperature = temperature

    def _clamp_eigenvalues(
        self, cov: jax.Array, min_eig: jax.Array, max_eig: jax.Array
    ) -> jax.Array:
        """Impose a minimum and maximum eigenvalue on a covariance matrix.

        Args:
            cov: A covariance matrix, shape (control_dim, control_dim).
            min_eig: The minimum eigenvalue to impose (scalar).
            max_eig: The maximum eigenvalue to impose (scalar).

        Returns:
            The clamped covariance matrix, with eigenvalues at least min_eig.
        """
        eigvals, eigvecs = jnp.linalg.eigh(cov)
        clamped_eigvals = jnp.clip(eigvals, min_eig, max_eig)
        clamped_cov = (eigvecs * clamped_eigvals) @ eigvecs.T
        return clamped_cov

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> ErCmaParams:
        """Initialize the policy parameters."""
        _params = super().init_params(initial_knots, seed)

        cov = jnp.eye(self.task.model.nu) * self.initial_noise_level**2
        cov = jnp.tile(cov[None], (self.num_knots, 1, 1))

        return ErCmaParams(
            tk=_params.tk, mean=_params.mean, covariance=cov, rng=_params.rng
        )

    def sample_knots(
        self, params: ErCmaParams
    ) -> Tuple[jax.Array, ErCmaParams]:
        """Sample a control sequence."""
        rng, sample_rng = jax.random.split(params.rng)
        noise = jax.random.multivariate_normal(
            sample_rng,
            mean=jnp.zeros(self.task.model.nu),
            cov=params.covariance,
            shape=(self.num_samples, self.num_knots),
        )  # shape (num_samples, num_knots, control_dim)
        controls = params.mean + noise
        return controls, params.replace(rng=rng)

    def update_params(
        self, params: ErCmaParams, rollouts: Trajectory
    ) -> ErCmaParams:
        """Update the mean with MPPI and the covariance with CMA."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        # N.B. jax.nn.softmax takes care of details like baseline subtraction.
        weights = jax.nn.softmax(-costs / self.temperature, axis=0)

        # Difference between samples and mean,
        # shape (num_samples, num_knots, control_dim)
        delta = rollouts.knots - params.mean

        # Outer product of deltas,
        # shape (num_samples, num_knots, control_dim, control_dim)
        outer_product = jnp.einsum("ijk,ijl->ijkl", delta, delta)

        # Standard CMA update
        new_cov = jnp.einsum("i,ijkl->jkl", weights, outer_product)

        # Entropy regularization 
        beta = jnp.linspace(
            self.initial_entropy_bonus, self.final_entropy_bonus, self.num_knots
        )
        new_cov = jnp.einsum("j,jkl->jkl", 1.0 / (1.0 - beta), new_cov)

        # EMA smoothing
        new_cov = (1 - self.alpha) * params.covariance + self.alpha * new_cov

        # Clamp eigenvalues to enforce minimum and maximum noise levels
        new_cov = jax.vmap(self._clamp_eigenvalues, in_axes=(0, None, None))(
            new_cov, self.minimum_noise_level**2, self.maximum_noise_level**2
        )

        # Mean update (same as standard MPPI)
        mean = jnp.sum(weights[:, None, None] * rollouts.knots, axis=0)

        return params.replace(mean=mean, covariance=new_cov)
