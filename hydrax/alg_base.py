from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Literal, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from mujoco import mjx

from hydrax.risk import AverageCost, RiskStrategy
from hydrax.task_base import Task
from hydrax.utils.spline import get_interp_func


@dataclass
class Trajectory:
    """Data class for storing rollout data.

    Attributes:
        controls: Control actions for each time step (size H).
        knots: Control spline knots (size num_knots).
        costs: Costs associated with each time step (size H+1).
        trace_sites: Positions of trace sites at each time step (size H+1).
    """

    controls: jax.Array
    knots: jax.Array
    costs: jax.Array
    trace_sites: jax.Array

    def __len__(self):
        """Return the number of time steps in the trajectory (T)."""
        return self.costs.shape[-1] - 1


@dataclass
class SamplingParams:
    """Parameters for sampling-based control algorithms.

    Attributes:
        mean: The mean of the control spline knot distribution, μ = [u₀, ...].
        rng: The pseudo-random number generator key.
    """

    mean: jax.Array
    rng: jax.Array


class SamplingBasedController(ABC):
    """An abstract sampling-based MPC algorithm interface."""

    def __init__(
        self,
        task: Task,
        num_randomizations: int,
        risk_strategy: RiskStrategy,
        seed: int,
        T: float,
        dt: float,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
    ) -> None:
        """Initialize the MPC controller.

        Args:
            task: The task instance defining the dynamics and costs.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
            seed: The random seed for domain randomization.
            T: The time horizon for the rollout in seconds.
            dt: The time step for the controller in seconds.
            spline_type: The type of spline used for control interpolation.
                         Defaults to "zero" (zero-order hold).
            num_knots: The number of knots in the control spline.
        """
        self.task = task
        self.num_randomizations = max(num_randomizations, 1)

        # Risk strategy defaults to average cost
        if risk_strategy is None:
            risk_strategy = AverageCost()
        self.risk_strategy = risk_strategy

        # time-related variables
        self.T = T
        self.dt = dt
        self.H = int(round(self.T / self.dt))  # number of control steps

        sim_dt = self.task.dt
        self.sim_steps_per_control_step = int(round(self.dt / sim_dt))

        # Spline setup for control interpolation
        self.spline_type = spline_type
        self.num_knots = num_knots
        self.tk = jnp.linspace(0.0, self.T, self.num_knots)  # knot times
        self.interp_func = get_interp_func(spline_type)

        # Use a single model (no domain randomization) by default
        self.model = task.model
        self.randomized_axes = None

        if self.num_randomizations > 1:
            # Make domain randomized models
            rng = jax.random.key(seed)
            rng, subrng = jax.random.split(rng)
            subrngs = jax.random.split(subrng, num_randomizations)
            randomizations = jax.vmap(self.task.domain_randomize_model)(subrngs)
            self.model = self.task.model.tree_replace(randomizations)

            # Keep track of which elements of the model have randomization
            self.randomized_axes = jax.tree.map(lambda x: None, self.task.model)
            self.randomized_axes = self.randomized_axes.tree_replace(
                {key: 0 for key in randomizations.keys()}
            )

    def optimize(self, state: mjx.Data, params: Any) -> Tuple[Any, Trajectory]:
        """Perform an optimization step to update the policy parameters.

        Args:
            state: The initial state x₀.
            params: The current policy parameters, U ~ π(params).

        Returns:
            Updated policy parameters
            Rollouts used to update the parameters
        """
        # Sample random control sequences from spline knots
        knots, params = self.sample_knots(params)
        knots = jnp.clip(
            knots, self.task.u_min, self.task.u_max
        )  # (num_rollouts, num_knots, nu)

        # Roll out the control sequences, applying domain randomizations and
        # combining costs using self.risk_strategy.
        rng, dr_rng = jax.random.split(params.rng)
        rollouts = self.rollout_with_randomizations(state, knots, dr_rng)
        params = params.replace(rng=rng)

        # Update the policy parameters based on the combined costs
        params = self.update_params(params, rollouts)
        return params, rollouts

    def rollout_with_randomizations(
        self,
        state: mjx.Data,
        knots: jax.Array,
        rng: jax.Array,
    ) -> Trajectory:
        """Compute rollout costs, applying domain randomizations.

        Args:
            state: The initial state x₀.
            knots: The control spline knots, (num rollouts, num_knots, nu).
            rng: The random number generator key for randomizing initial states.

        Returns:
            A Trajectory object containing the control, costs, and trace sites.
            Costs are aggregated over domains using the given risk strategy.
        """
        # Set the initial state for each rollout.
        states = jax.vmap(lambda _, x: x, in_axes=(0, None))(
            jnp.arange(self.num_randomizations), state
        )

        if self.num_randomizations > 1:
            # Randomize the initial states for each domain randomization
            subrngs = jax.random.split(rng, self.num_randomizations)
            randomizations = jax.vmap(self.task.domain_randomize_data)(
                states, subrngs
            )
            states = states.tree_replace(randomizations)

        # compute the control sequence from the knots
        tq = jnp.linspace(0.0, self.T - self.dt, self.H)  # ctrl query times
        controls = self.interp_func(tq, self.tk, knots)  # (num_rollouts, H, nu)

        # Apply the control sequences, parallelized over both rollouts and
        # domain randomizations.
        _, rollouts = jax.vmap(
            self.eval_rollouts, in_axes=(self.randomized_axes, 0, None, None)
        )(self.model, states, controls, knots)

        # Combine the costs from different domain randomizations using the
        # specified risk strategy.
        costs = self.risk_strategy.combine_costs(rollouts.costs)
        controls = rollouts.controls[0]  # identical over randomizations
        knots = rollouts.knots[0]  # identical over randomizations
        trace_sites = rollouts.trace_sites[0]  # visualization only, take 1st
        return rollouts.replace(
            costs=costs, controls=controls, knots=knots, trace_sites=trace_sites
        )

    @partial(jax.vmap, in_axes=(None, None, None, 0, 0))
    def eval_rollouts(
        self,
        model: mjx.Model,
        state: mjx.Data,
        controls: jax.Array,
        knots: jax.Array,
    ) -> Tuple[mjx.Data, Trajectory]:
        """Rollout control sequences (in parallel) and compute the costs.

        Args:
            model: The mujoco dynamics model to use.
            state: The initial state x₀.
            controls: The control sequences, (num rollouts, H, nu).
            knots: The control spline knots, (num rollouts, num_knots, nu).

        Returns:
            The states (stacked) experienced during the rollouts.
            A Trajectory object containing the control, costs, and trace sites.
        """

        def _scan_fn(
            x: mjx.Data, u: jax.Array
        ) -> Tuple[mjx.Data, Tuple[mjx.Data, jax.Array, jax.Array]]:
            """Compute the cost and observation, then advance the state."""
            x = mjx.forward(model, x)  # compute site positions
            cost = self.dt * self.task.running_cost(x, u)
            sites = self.task.get_trace_sites(x)

            # Advance the state for several steps, zero-order hold on control
            x = jax.lax.fori_loop(
                0,
                self.sim_steps_per_control_step,
                lambda _, x: mjx.step(model, x),
                x.replace(ctrl=u),
            )

            return x, (x, cost, sites)

        final_state, (states, costs, trace_sites) = jax.lax.scan(
            _scan_fn, state, controls
        )
        final_cost = self.task.terminal_cost(final_state)
        final_trace_sites = self.task.get_trace_sites(final_state)

        costs = jnp.append(costs, final_cost)
        trace_sites = jnp.append(trace_sites, final_trace_sites[None], axis=0)

        return states, Trajectory(
            controls=controls,
            knots=knots,
            costs=costs,
            trace_sites=trace_sites,
        )

    def init_params(self, seed: int = 0) -> Any:
        """Initialize the policy parameters, U = [u₀, u₁, ... ] ~ π(params).

        Args:
            seed: The random seed for initializing the policy parameters.

        Returns:
            The initial policy parameters.
        """
        rng = jax.random.key(seed)
        mean = jnp.zeros((self.num_knots, self.task.model.nu))
        return SamplingParams(mean=mean, rng=rng)

    @abstractmethod
    def sample_knots(self, params: Any) -> Tuple[jax.Array, Any]:
        """Sample a set of control spline knots U ~ π(params).

        Args:
            params: Parameters of the policy distribution (e.g., mean, std).

        Returns:
            Control spline knots U, size (num rollouts, num_knots).
            Updated parameters (e.g., with a new PRNG key).
        """

    @abstractmethod
    def update_params(self, params: Any, rollouts: Trajectory) -> Any:
        """Update the policy parameters π(params) using the rollouts.

        Args:
            params: The current policy parameters.
            rollouts: The rollouts obtained from the current policy.

        Returns:
            The updated policy parameters.
        """

    def get_action(self, params: SamplingParams, t: float) -> jax.Array:
        """Get the control action at a given point along the trajectory.

        Args:
            params: The policy parameters, U ~ π(params).
            t: The time (in seconds) from the start of the trajectory.

        Returns:
            The control action u(t).
        """
        knots = params.mean[None, ...]  # (1, num_knots, nu)
        tq = jnp.array([t])  # query time
        u = self.interp_func(tq, self.tk, knots)[0, 0]  # (nu,)
        return u
