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
        return self.costs.shape[-1]


class Task(ABC):
    def __init__(
        self,
        model: mjx.Model,
        planning_horizon: int,
        sim_steps_per_control_step: int,
    ):
        self.model = model
        self.planning_horizon = planning_horizon
        self.sim_steps_per_control_step = sim_steps_per_control_step

    @abstractmethod
    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        pass

    @abstractmethod
    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        pass


def rollout(task: Task, state: mjx.Data, controls: jax.Array) -> Trajectory:
    pass


class SamplingBasedMPC(ABC):
    def __init__(self, task: Task, num_samples):
        self.task = task
        self.num_samples = num_samples

    def eval_rollouts(self, state: mjx.Data, controls: jax.Array) -> Trajectory:
        pass

    @abstractmethod
    def sample_controls(self, params: Any) -> jax.Array:
        pass

    @abstractmethod
    def update_params(self, params: Any, rollouts: Trajectory) -> Any:
        pass

    @abstractmethod
    def get_action(self, params: Any, t: float) -> jax.Array:
        pass


def run_mpc(
    system: mjx.Model,
    task: Task,
    mpc_rate: float,
    total_sim_time: float,
    headless: bool,
) -> mjx.Data:
    pass
