from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from mujoco import mjx

from hydrax.task_base import Task


@dataclass
class Trajectory:
    """Data class for storing rollout data.

    Attributes:
        controls: Control actions for each time step (size N - 1).
        costs: Costs associated with each time step (size N).
        observations: Observations at each time step (size N).
        trace_sites: Positions of trace sites at each time step (size N).
    """

    controls: jax.Array
    costs: jax.Array
    observations: jax.Array
    trace_sites: jax.Array

    def __len__(self):
        """Return the number of time steps in the trajectory (N)."""
        return self.costs.shape[-1]


class SamplingBasedController(ABC):
    """An abstract sampling-based MPC algorithm interface."""

    def __init__(self, task: Task):
        """Initialize the MPC controller.

        Args:
            task: The task instance defining the dynamics and costs.
        """
        self.task = task

    def optimize(self, state: mjx.Data, params: Any) -> Tuple[Any, Trajectory]:
        """Perform an optimization step to update the policy parameters.

        Args:
            state: The initial state x₀.
            params: The current policy parameters, U ~ π(params).

        Returns:
            Updated policy parameters
            Rollouts used to update the parameters
        """
        controls, params = self.sample_controls(params)
        controls = jnp.clip(controls, -self.task.u_max, self.task.u_max)
        rollouts = self.eval_rollouts(state, controls)
        params = self.update_params(params, rollouts)
        return params, rollouts

    @partial(jax.vmap, in_axes=(None, None, 0))
    def eval_rollouts(self, state: mjx.Data, controls: jax.Array) -> Trajectory:
        """Rollout control sequences (in parallel) and compute the costs.

        Args:
            state: The initial state x₀.
            controls: The control sequences, size (num rollouts, horizon - 1).

        Returns:
            A Trajectory object containing the costs, controls, observations.
        """

        def _scan_fn(
            x: mjx.Data, u: jax.Array
        ) -> Tuple[mjx.Data, Tuple[jax.Array, jax.Array]]:
            """Compute the cost and observation, then advance the state."""
            x = mjx.forward(self.task.model, x)  # compute site positions
            cost = self.task.dt * self.task.running_cost(x, u)
            obs = self.task.get_obs(x)
            sites = self.task.get_trace_sites(x)

            # Advance the state for several steps, zero-order hold on control
            x = jax.lax.fori_loop(
                0,
                self.task.sim_steps_per_control_step,
                lambda _, x: mjx.step(self.task.model, x),
                x.replace(ctrl=u),
            )

            return x, (cost, obs, sites)

        final_state, (costs, observations, trace_sites) = jax.lax.scan(
            _scan_fn, state, controls
        )
        final_cost = self.task.terminal_cost(final_state)
        final_obs = self.task.get_obs(final_state)
        final_trace_sites = self.task.get_trace_sites(final_state)

        costs = jnp.append(costs, final_cost)
        observations = jnp.vstack([observations, final_obs])
        trace_sites = jnp.append(trace_sites, final_trace_sites[None], axis=0)

        return Trajectory(
            controls=controls,
            costs=costs,
            observations=observations,
            trace_sites=trace_sites,
        )

    @abstractmethod
    def init_params(self) -> Any:
        """Initialize the policy parameters, U = [u₀, u₁, ... ] ~ π(params).

        Returns:
            The initial policy parameters.
        """
        pass

    @abstractmethod
    def sample_controls(self, params: Any) -> Tuple[jax.Array, Any]:
        """Sample a set of control sequences U ~ π(params).

        Args:
            params: Parameters of the policy distribution (e.g., mean, std).

        Returns:
            A control sequences U, size (num rollouts, horizon - 1).
            Updated parameters (e.g., with a new PRNG key).
        """
        pass

    @abstractmethod
    def update_params(self, params: Any, rollouts: Trajectory) -> Any:
        """Update the policy parameters π(params) using the rollouts.

        Args:
            params: The current policy parameters.
            rollouts: The rollouts obtained from the current policy.

        Returns:
            The updated policy parameters.
        """
        pass

    @abstractmethod
    def get_action(self, params: Any, t: float) -> jax.Array:
        """Get the control action at a given point along the trajectory.

        Args:
            params: The policy parameters, U ~ π(params).
            t: The time (in seconds) from the start of the trajectory.

        Returns:
            The control action u(t).
        """
        pass
