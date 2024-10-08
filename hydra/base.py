from abc import ABC, abstractmethod
from typing import Any

import jax
from flax.struct import dataclass
from mujoco import mjx


@dataclass
class Trajectory:
    """Data class for storing rollout data.

    Attributes:
        controls: Control actions for each time step (size N - 1).
        costs: Costs associated with each time step (size N).
        observations: Observations at each time step (size N).
    """

    controls: jax.Array
    costs: jax.Array
    observations: jax.Array

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
        model: mjx.Model,
        planning_horizon: int,
        sim_steps_per_control_step: int,
    ):
        """Set the model and simulation parameters.

        Args:
            model: The MuJoCo model to use for simulation.
            planning_horizon: The number of control steps to plan over.
            sim_steps_per_control_step: The number of simulation steps to take
                                        for each control step.

        Note: many other simulator parameters, e.g., simulator time step,
              Newton iterations, etc., are set in the model itself.
        """
        self.model = model
        self.planning_horizon = planning_horizon
        self.sim_steps_per_control_step = sim_steps_per_control_step

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


class SamplingBasedMPC(ABC):
    """An abstract sampling-based MPC interface."""

    def __init__(self, task: Task):
        """Initialize the MPC controller.

        Args:
            task: The task instance defining the dynamics and costs.
        """
        self.task = task

    def eval_rollouts(self, state: mjx.Data, controls: jax.Array) -> Trajectory:
        """Rollout control sequences (in parallel) and compute the costs.

        Args:
            state: The initial state x₀.
            controls: The control sequences, size (num rollouts, horizon - 1).

        Returns:
            A Trajectory object containing the costs, controls, observations.
        """
        return NotImplementedError

    @abstractmethod
    def sample_controls(self, params: Any) -> jax.Array:
        """Sample a control sequence U ~ π(params).

        Args:
            params: Parameters of the policy distribution (e.g. mean, std).

        Returns:
            A control sequence U, size (horizon - 1).
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
