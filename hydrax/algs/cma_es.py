from typing import Tuple

import evosax
import jax
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base import SamplingBasedController, Trajectory
from hydrax.task_base import Task


@dataclass
class CMAESParams:
    """Policy parameters for CMA-ES.

    Attributes:
        controls: The latest control sequence, U = [u₀, u₁, ..., ].
        opt_state: The state of the CMA-ES optimizer (covariance, etc.).
        rng: The pseudo-random number generator key.
    """

    controls: jax.Array
    opt_state: evosax.strategies.cma_es.EvoState
    rng: jax.Array


class CMAES(SamplingBasedController):
    """Covariance Matrix Adaptation Evolution Strategy (CMA-ES) controller."""

    def __init__(self, task: Task, num_samples: int, elite_ratio: float = 0.5):
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control tapes to sample.
            elite_ratio: The ratio of top-performing samples to keep.
        """
        super().__init__(task)

        self.strategy = evosax.CMA_ES(
            popsize=num_samples,
            num_dims=task.model.nu * (task.planning_horizon - 1),
            elite_ratio=elite_ratio,
        )

        # TODO: consider exposing these evolution strategy parameters
        self.es_params = self.strategy.default_params

    def init_params(self, seed: int = 0) -> CMAESParams:
        """Initialize the policy parameters."""
        rng = jax.random.key(seed)
        rng, init_rng = jax.random.split(rng)
        controls = jnp.zeros(
            (self.task.planning_horizon - 1, self.task.model.nu)
        )
        opt_state = self.strategy.initialize(init_rng, self.es_params)
        return CMAESParams(controls=controls, opt_state=opt_state, rng=rng)

    def sample_controls(
        self, params: CMAESParams
    ) -> Tuple[jax.Array, CMAESParams]:
        """Sample control sequences from the proposal distribution."""
        rng, sample_rng = jax.random.split(params.rng)
        x, opt_state = self.strategy.ask(
            sample_rng, params.opt_state, self.es_params
        )

        # evosax works with vectors of decision variables, so we reshape U to
        # [batch_size, horizon - 1, nu].
        controls = jnp.reshape(
            x,
            (
                self.strategy.popsize,
                self.task.planning_horizon - 1,
                self.task.model.nu,
            ),
        )

        return controls, params.replace(opt_state=opt_state, rng=rng)

    def update_params(
        self, params: CMAESParams, rollouts: Trajectory
    ) -> CMAESParams:
        """Update the policy parameters based on the rollouts."""
        costs = jnp.sum(rollouts.costs, axis=1)
        x = jnp.reshape(rollouts.controls, (self.strategy.popsize, -1))
        opt_state = self.strategy.tell(
            x, costs, params.opt_state, self.es_params
        )
        best_controls = opt_state.best_member.reshape(
            (self.task.planning_horizon - 1, self.task.model.nu)
        )
        return params.replace(controls=best_controls, opt_state=opt_state)

    def get_action(self, params: CMAESParams, t: float) -> jax.Array:
        """Get the control action for the current time step, zero order hold."""
        idx_float = t / self.task.dt  # zero order hold
        idx = jnp.floor(idx_float).astype(jnp.int32)
        return params.controls[idx]
