"""Model Tensor Planning (MTP) controller.

Implements the sampling-based MPC framework of Le et al. 2026
(arxiv 2505.01059), which generates globally-diverse trajectory
candidates via structured tensor sampling over a randomised M-partite
graph and mixes them with a local CEM-style distribution.
"""

from typing import Literal, Optional, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task
from hydrax.utils.spline import (
    compute_b_spline_matrix,
    poly_akima,
    poly_interpolation,
)

MTPInterpolationType = Literal["akima", "bspline", "linear"]


@dataclass
class MTPParams(SamplingParams):
    """Policy parameters for Model Tensor Planning.

    Attributes:
        tk: The knot times of the control spline.
        mean: Mean of the local CEM knot distribution, μ = [u₀, ...].
        rng: The pseudo-random number generator key.
        cov: Diagonal standard deviation of the local CEM distribution.
        best_knots: Best spline knots from the previous optimisation step.
    """

    cov: jax.Array
    best_knots: jax.Array


class MTP(SamplingBasedController):
    """Model Tensor Planning sampling-based MPC.

    MTP draws a fraction ``beta`` of the rollouts from a structured
    M-partite tensor sampler (random paths through ``M`` waypoints with
    ``N`` candidates each, smoothed by an Akima/B-spline/linear
    interpolation) and the remaining rollouts from a local Gaussian
    around the current mean. Elite rollouts update the local mean and
    covariance with a CEM-style softmax weighting.
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        m_pts: int = 3,
        n_per_layer: int = 50,
        degree: int = 2,
        num_elites: int = 5,
        sigma_start: float = 0.5,
        sigma_min: float = 0.1,
        sigma_max: float = 1.0,
        temperature: float = 0.1,
        beta: float = 0.1,
        alpha: float = 0.5,
        mtp_interpolation: MTPInterpolationType = "akima",
        num_randomizations: int = 1,
        risk_strategy: Optional[RiskStrategy] = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
    ) -> None:
        """Initialise the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: Total number of control sequences to evaluate per
                iteration. Must be at least ``num_elites + 1`` (one slot is
                always reserved for the previous best).
            m_pts: Number of tensor waypoints (graph depth).
            n_per_layer: Number of candidate values per waypoint
                (graph width).
            degree: B-spline degree, only consulted when
                ``mtp_interpolation='bspline'``. Must be ``>= 2``.
            num_elites: Number of elite rollouts kept per iteration.
            sigma_start: Initial standard deviation of the local Gaussian.
            sigma_min: Lower clip on the local standard deviation.
            sigma_max: Upper clip on the local standard deviation. Also
                used to derive a finite sampling range when actuator
                bounds are infinite.
            temperature: Softmax temperature λ used for elite weighting.
            beta: Fraction of rollouts drawn from the tensor sampler. The
                remainder are drawn from the local Gaussian.
            alpha: CEM smoothing weight. ``alpha = 0`` applies the new
                statistics in full; ``alpha = 1`` keeps the old ones.
            mtp_interpolation: Smoothing applied to each tensor path
                before it is used as a control sequence.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different
                randomizations. Defaults to average cost.
            seed: The random seed for domain randomization.
            plan_horizon: The time horizon for the rollout in seconds.
            spline_type: The type of spline used for control interpolation.
                Defaults to "zero" (zero-order hold).
            num_knots: The number of knots in the control spline.
            iterations: The number of optimization iterations to perform.
        """
        if degree < 2:
            raise ValueError(f"degree must be at least 2, got {degree}.")
        if num_elites < 1:
            raise ValueError(
                f"num_elites must be at least 1, got {num_elites}."
            )
        if num_elites >= num_samples:
            raise ValueError(
                f"num_elites ({num_elites}) must be strictly less than "
                f"num_samples ({num_samples})."
            )
        if mtp_interpolation == "bspline" and m_pts < degree + 1:
            raise ValueError(
                f"B-spline interpolation requires m_pts >= degree + 1 "
                f"(got m_pts={m_pts}, degree={degree}). With m_pts < "
                f"degree+1 the valid B-spline parameter domain collapses "
                f"to a point and the tensor branch produces flat samples. "
                f"Use 'akima' or 'linear' for small m_pts, or increase "
                f"m_pts."
            )

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

        self.num_samples = num_samples
        self.m_pts = m_pts
        self.n_per_layer = n_per_layer
        self.degree = degree
        self.num_elites = num_elites
        self.sigma_start = sigma_start
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.temperature = temperature
        self.beta = beta
        self.alpha = alpha
        self.mtp_interpolation = mtp_interpolation

        # Always reserve one slot for the previous best, then split the
        # remainder between the tensor branch and the local Gaussian.
        mtp_samples = int(round(num_samples * beta))
        self.mtp_samples = min(max(mtp_samples, 0), num_samples - 1)
        self.cem_samples = num_samples - self.mtp_samples - 1

        # Pre-baked spline parameters. These are independent of the rng so
        # they can be allocated once at init time and reused across every
        # planning step.
        self._aknots = jnp.linspace(1.0, m_pts, m_pts)
        bknots = jnp.arange(m_pts + degree + 1)
        self._bmat = compute_b_spline_matrix(bknots, degree, num_knots)

        # Linear branch: pre-bake a (num_knots, m_pts) interpolation matrix
        # so the branch becomes a single einsum, mirroring the bspline
        # branch and avoiding any per-step searchsorted overhead.
        if m_pts > 1:
            n_total = -(-num_knots // (m_pts - 1)) * (m_pts - 1)  # ceil-div
            t_lin = jnp.linspace(0.0, m_pts - 1, n_total + 1)[:-1]
            i_idx = jnp.clip(jnp.floor(t_lin).astype(jnp.int32), 0, m_pts - 2)
            s = t_lin - i_idx
            cols = jnp.arange(m_pts)
            self._linmat = jnp.where(
                cols[None, :] == i_idx[:, None], 1.0 - s[:, None], 0.0
            ) + jnp.where(cols[None, :] == i_idx[:, None] + 1, s[:, None], 0.0)
        else:
            self._linmat = jnp.ones((num_knots, 1))

        # Number of evaluation points per Akima segment so the total knot
        # count covers ``num_knots`` (we then slice off any overhang).
        self._akima_pts_per_seg = (
            -(-num_knots // (m_pts - 1)) if m_pts > 1 else num_knots
        )

    def init_params(
        self, initial_knots: Optional[jax.Array] = None, seed: int = 0
    ) -> MTPParams:
        """Initialise the policy parameters."""
        _params = super().init_params(initial_knots, seed)
        cov = jnp.full_like(_params.mean, self.sigma_start)
        best_knots = jnp.zeros_like(_params.mean)
        return MTPParams(
            tk=_params.tk,
            mean=_params.mean,
            rng=_params.rng,
            cov=cov,
            best_knots=best_knots,
        )

    def _interp_paths(self, paths: jax.Array) -> jax.Array:
        """Smooth tensor paths into ``(B, num_knots, nu)`` knot tensors."""
        if self.mtp_interpolation == "akima":
            a = jax.vmap(poly_akima, in_axes=(None, 0))(self._aknots, paths)
            return poly_interpolation(a, self._akima_pts_per_seg)[
                :, : self.num_knots
            ]
        if self.mtp_interpolation == "bspline":
            return jnp.einsum("bmd,hm->bhd", paths, self._bmat)
        if self.mtp_interpolation == "linear":
            return jnp.einsum("bmd,hm->bhd", paths, self._linmat)[
                :, : self.num_knots
            ]
        raise ValueError(
            f"Invalid MTP interpolation: {self.mtp_interpolation}. "
            "Expected one of ['akima', 'bspline', 'linear']."
        )

    def sample_knots(self, params: MTPParams) -> Tuple[jax.Array, MTPParams]:
        """Sample control spline knots using tensor + local Gaussian mixing.

        Each call returns exactly ``num_samples`` knot sequences:
        the previous-best slot, ``mtp_samples`` tensor-sampled
        sequences, and ``cem_samples`` local Gaussian perturbations
        of the current mean.
        """
        rng = params.rng
        nu = self.task.model.nu
        best = params.best_knots[None, ...]  # (1, num_knots, nu)

        if self.mtp_samples >= 1:
            rng, tensor_rng = jax.random.split(rng)

            # Hydrax marks unbounded actuators with ±inf; substitute a
            # finite range derived from the current mean ± k·sigma_max so
            # ``jax.random.uniform`` never returns NaN.
            k = 3.0
            mean_lo = jnp.min(params.mean, axis=0) - k * self.sigma_max
            mean_hi = jnp.max(params.mean, axis=0) + k * self.sigma_max
            u_min = jnp.where(
                jnp.isfinite(self.task.u_min), self.task.u_min, mean_lo
            )
            u_max = jnp.where(
                jnp.isfinite(self.task.u_max), self.task.u_max, mean_hi
            )

            # Sample tensor paths directly. Mathematically identical to
            # building a (m_pts, n_per_layer, nu) graph and gathering
            # along independent layer indices, but skips the
            # m_pts·n_per_layer·nu materialisation, one RNG split, and
            # the gather op.
            paths = jax.random.uniform(
                tensor_rng,
                (self.mtp_samples, self.m_pts, nu),
                minval=u_min,
                maxval=u_max,
            )
            mtp_knots = self._interp_paths(paths)
            all_knots = jnp.concatenate([best, mtp_knots], axis=0)
        else:
            all_knots = best

        if self.cem_samples > 0:
            rng, sample_rng = jax.random.split(rng)
            noise = jax.random.normal(
                sample_rng, (self.cem_samples, self.num_knots, nu)
            )
            cem_knots = params.mean + params.cov * noise
            all_knots = jnp.concatenate([all_knots, cem_knots], axis=0)

        return all_knots, params.replace(rng=rng)

    def update_params(
        self, params: MTPParams, rollouts: Trajectory
    ) -> MTPParams:
        """Refit the local Gaussian with weighted elite statistics."""
        costs = jnp.sum(rollouts.costs, axis=1)  # (num_samples,)

        # Top-K elite selection in O(N + K log K).
        _, elite_idx = jax.lax.top_k(-costs, self.num_elites)
        elite_knots = rollouts.knots[elite_idx]
        elite_costs = costs[elite_idx]

        # Subtract the elite minimum to keep softmax exponents bounded.
        # softmax is logsumexp-stable on its own, so no NaN guard is
        # needed: any NaN in ``elite_costs`` means an upstream cost bug
        # we want to surface, not silently hide.
        baseline = jnp.min(elite_costs)
        weights = jax.nn.softmax(
            -(elite_costs - baseline) / self.temperature, axis=0
        )

        # Weighted mean and Bessel-corrected std (the (E-1)/E correction
        # for uniform weights, generalised to weighted samples).
        mean = jnp.sum(weights[:, None, None] * elite_knots, axis=0)
        var = jnp.sum(
            weights[:, None, None] * (elite_knots - mean) ** 2, axis=0
        )
        bessel = 1.0 / jnp.maximum(1.0 - jnp.sum(weights**2), 1e-6)
        cov = jnp.sqrt(var * bessel)
        cov = jnp.clip(cov, self.sigma_min, self.sigma_max)

        # Momentum smoothing. Variance update is the standard CMA convex
        # blend ``α·old + (1-α)·new`` (algebraically identical to the
        # earlier ``new + α·(old-new)`` form but reads as the standard
        # convention and saves one fused op).
        mean = mean + self.alpha * (params.mean - mean)
        new_var = cov**2
        old_var = params.cov**2
        cov = jnp.sqrt(self.alpha * old_var + (1.0 - self.alpha) * new_var)

        best_knots = rollouts.knots[elite_idx[0]]
        return params.replace(mean=mean, cov=cov, best_knots=best_knots)
