import jax
from mujoco import mjx
from flax.struct import dataclass
from abc import ABC, abstractmethod
from typing import Any, Tuple

@dataclass
class Trajectory:
    controls: jax.Array
    costs: jax.Array
    observations: jax.Array

@dataclass
class Solution:
    samples: Trajectory
    best_controls: jax.Array

class Task(ABC):
    def __init__(self, model: mjx.Model, planning_horizon: int, sim_steps_per_control_step: int):
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

    @abstractmethod
    def sample_controls(self, previous: Solution, info: Any) -> Tuple[jax.Array, Any]:
        pass

    @abstractmethod
    def pick_best(self, state: mjx.Data, controls: jax.Array, info: Any) -> Tuple[Solution, Any]:
        pass

    def solver_step(self, previous: Solution, state: mjx.Data, info: Any) -> Tuple[Solution, Any]:
        controls, info = self.sample_controls(previous, info)
        solution, info = self.pick_best(state, controls, info)
        return solution, info


def run_mpc(system: mjx.Model, task: Task, mpc_rate: float, total_sim_time: float, headless: bool) -> mjx.Data:
    pass
