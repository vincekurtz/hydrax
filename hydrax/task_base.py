from abc import ABC, abstractmethod
from typing import Dict, Sequence

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx


class Task(ABC):
    """An abstract task interface, defining the dynamics and cost functions.

    The task is a discrete-time optimal control problem of the form

        minᵤ ϕ(x_{T+1}) + ∑ₜ ℓ(xₜ, uₜ)
        s.t. xₜ₊₁ = f(xₜ, uₜ)

    where the dynamics f(xₜ, uₜ) are defined by a MuJoCo model, and the costs
    ℓ(xₜ, uₜ) and ϕ(x_{T+1}) are defined by the task instance itself.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        planning_horizon: int,
        sim_steps_per_control_step: int,
        trace_sites: Sequence[str] = [],
    ):
        """Set the model and simulation parameters.

        Args:
            mj_model: The MuJoCo model to use for simulation.
            planning_horizon: The number of control steps (T) to plan over.
            sim_steps_per_control_step: The number of simulation steps to take
                                        for each control step.
            trace_sites: A list of site names to visualize with traces.

        Note: many other simulator parameters, e.g., simulator time step,
              Newton iterations, etc., are set in the model itself.
        """
        assert isinstance(mj_model, mujoco.MjModel)
        self.model = mjx.put_model(mj_model)
        self.planning_horizon = planning_horizon
        self.sim_steps_per_control_step = sim_steps_per_control_step

        # Set actuator limits
        self.u_min = jnp.where(
            mj_model.actuator_ctrllimited,
            mj_model.actuator_ctrlrange[:, 0],
            -jnp.inf,
        )
        self.u_max = jnp.where(
            mj_model.actuator_ctrllimited,
            mj_model.actuator_ctrlrange[:, 1],
            jnp.inf,
        )

        # Timestep for each control step
        self.dt = mj_model.opt.timestep * sim_steps_per_control_step

        # Get site IDs for points we want to trace
        self.trace_site_ids = jnp.array(
            [mj_model.site(name).id for name in trace_sites]
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

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Generate randomized model parameters for domain randomization.

        Returns a dictionary of randomized model parameters, that can be used
        with `mjx.Model.tree_replace` to create a new randomized model.

        For example, we might set the `model.geom_friction` values by returning
        `{"geom_friction": new_frictions, ...}`.

        The default behavior is to return an empty dictionary, which means no
        randomization is applied.

        Args:
            rng: A random number generator key.

        Returns:
            A dictionary of randomized model parameters.
        """
        return {}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Generate randomized data elements for domain randomization.

        This is the place where we could randomize the initial state and other
        `data` elements. Like `domain_randomize_model`, this method should
        return a dictionary that can be used with `mjx.Data.tree_replace`.

        Args:
            data: The base data instance holding the current state.
            rng: A random number generator key.

        Returns:
            A dictionary of randomized data elements.
        """
        return {}
