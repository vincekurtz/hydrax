from typing import Any, Literal, Tuple

import evosax
import jax
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task

# Generic types for evosax
EvoParams = Any
EvoState = Any


@dataclass
class EvosaxParams(SamplingParams):
    """Policy parameters for evosax optimizers.

    Attributes:
        mean: The mean of the control spline knot distribution, μ = [u₀, ...].
        opt_state: The state of the evosax optimizer (covariance, etc.).
        rng: The pseudo-random number generator key.
    """

    opt_state: EvoState


class Evosax(SamplingBasedController):
    """A generic controller that allows us to use any evosax optimizer.

    See https://github.com/RobertTLange/evosax/ for details and a list of
    available optimizers.
    """

    def __init__(
        self,
        task: Task,
        optimizer: evosax.Strategy,
        num_samples: int,
        es_params: EvoParams = None,
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        **kwargs,
    ) -> None:
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            optimizer: The evosax optimizer to use.
            num_samples: The number of control tapes to sample.
            es_params: The parameters for the evosax optimizer.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
                           Defaults to average cost.
            seed: The random seed for domain randomization.
            plan_horizon: The time horizon for the rollout in seconds.
            spline_type: The type of spline used for control interpolation.
                         Defaults to "zero" (zero-order hold).
            num_knots: The number of knots in the control spline.
            **kwargs: Additional keyword arguments for the optimizer.
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

        self.strategy = optimizer(
            popsize=num_samples,
            num_dims=task.model.nu * self.num_knots,
            **kwargs,
        )

        if es_params is None:
            es_params = self.strategy.default_params
        self.es_params = es_params

    def init_params(self, seed: int = 0) -> EvosaxParams:
        """Initialize the policy parameters."""
        _params = super().init_params(seed)
        rng, init_rng = jax.random.split(_params.rng)
        opt_state = self.strategy.initialize(init_rng, self.es_params)
        return EvosaxParams(
            tk=_params.tk, mean=_params.mean, opt_state=opt_state, rng=rng
        )

    def sample_knots(
        self, params: EvosaxParams
    ) -> Tuple[jax.Array, EvosaxParams]:
        """Sample control sequences from the proposal distribution."""
        rng, sample_rng = jax.random.split(params.rng)
        x, opt_state = self.strategy.ask(
            sample_rng, params.opt_state, self.es_params
        )

        # evosax works with vectors of decision variables, so we reshape U to
        # [batch_size, num_knots, nu].
        controls = jnp.reshape(
            x,
            (
                self.strategy.popsize,
                self.num_knots,
                self.task.model.nu,
            ),
        )

        return controls, params.replace(opt_state=opt_state, rng=rng)

    def update_params(
        self, params: EvosaxParams, rollouts: Trajectory
    ) -> EvosaxParams:
        """Update the policy parameters based on the rollouts."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        x = jnp.reshape(rollouts.knots, (self.strategy.popsize, -1))
        opt_state = self.strategy.tell(
            x, costs, params.opt_state, self.es_params
        )

        best_idx = jnp.argmin(costs)
        best_knots = rollouts.knots[best_idx]

        # By default, opt_state stores the best member ever, rather than the
        # best member from the current generation. We want to just use the best
        # member from this generation, since the cost landscape is constantly
        # changing.
        opt_state = opt_state.replace(
            best_member=x[best_idx], best_fitness=costs[best_idx]
        )
        return params.replace(mean=best_knots, opt_state=opt_state)
