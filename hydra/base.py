from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Sequence, Tuple

import jax
import jax.numpy as jnp
import mujoco
from flax.struct import dataclass
from mujoco import mjx


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


class Task(ABC):
    """An abstract task interface, defining the dynamics and cost functions.

    The task is a discrete-time optimal control problem of the form

        minᵤ ϕ(x_T) + ∑ₜ ℓ(xₜ, uₜ)
        s.t. xₜ₊₁ = f(xₜ, uₜ)

    where the dynamics f(xₜ, uₜ) are defined by a MuJoCo model, and the costs
    ℓ(xₜ, uₜ) and ϕ(x_T) are defined by the task instance itself.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        planning_horizon: int,
        sim_steps_per_control_step: int,
        u_max: float = jnp.inf,
        trace_sites: Sequence[str] = [],
    ):
        """Set the model and simulation parameters.

        Args:
            mj_model: The MuJoCo model to use for simulation.
            planning_horizon: The number of control steps to plan over.
            sim_steps_per_control_step: The number of simulation steps to take
                                        for each control step.
            u_max: The maximum control input.
            trace_sites: A list of site names to visualize with traces.

        Note: many other simulator parameters, e.g., simulator time step,
              Newton iterations, etc., are set in the model itself.
        """
        assert isinstance(mj_model, mujoco.MjModel)
        self.model = mjx.put_model(mj_model)
        self.planning_horizon = planning_horizon
        self.sim_steps_per_control_step = sim_steps_per_control_step
        self.u_max = u_max

        # Timestep for each control step
        self.dt = mj_model.opt.timestep * sim_steps_per_control_step

        # Get site IDs for points we want to trace
        self.trace_site_ids = jnp.array(
            [
                mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
                for name in trace_sites
            ]
        )

    @abstractmethod
    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ).

        Args:
            state: The current state xₜ.
            control: The control action uₜ.

        Returns:
            The scalar running cost ℓ(xₜ, uₜ)
        """
        pass

    @abstractmethod
    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T).

        Args:
            state: The final state x_T.

        Returns:
            The scalar terminal cost ϕ(x_T).
        """
        pass

    def get_obs(self, state: mjx.Data) -> jax.Array:
        """Get the observation vector at the current time step.

        Args:
            state: The current state xₜ.

        Returns:
            The observation vector yₜ.
        """
        # The default is to return the full state as the observation
        return jnp.concatenate([state.qpos, state.qvel])

    def get_trace_sites(self, state: mjx.Data) -> jax.Array:
        """Get the positions of the trace sites at the current time step.

        Args:
            state: The current state xₜ.

        Returns:
            The positions of the trace sites at the current time step.
        """
        if len(self.trace_site_ids) == 0:
            return jnp.zeros((0, 3))

        return state.site_xpos[self.trace_site_ids]


class SamplingBasedController(ABC):
    """An abstract sampling-based MPC interface."""

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
