# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Hydrax** is a GPU-accelerated sampling-based model predictive control (MPC) framework built with JAX and MuJoCo MJX. It solves optimal control problems by sampling control sequences, rolling out trajectories in parallel on the GPU, and updating a parameterized control policy.

## Commands

```bash
uv sync                                      # Install dependencies
uv run pytest                                # Run all tests
uv run pytest tests/test_pendulum.py        # Run a single test file
uv run python examples/pendulum.py mppi     # Run an interactive example
uv run ruff check                           # Lint
uv run ruff format                          # Format
```

Dependencies require Python 3.12+ and CUDA 13. `uv` is the package manager (preferred over conda).

## Architecture

### Core Abstractions

**`Task` (`hydrax/task_base.py`)** — abstract base class for an optimal control problem. Users subclass this to define:
- `running_cost(state, control)` and `terminal_cost(state)` — the objective
- `domain_randomize_model()` / `domain_randomize_data()` — optional domain randomization

The task wraps a `mujoco.MjModel` via `mjx.put_model()`. Concrete tasks live in `hydrax/tasks/` and load XML from `hydrax/models/`.

**`SamplingBasedController` (`hydrax/alg_base.py`)** — abstract base for all sampling algorithms. Users subclass this and implement:
- `sample_knots(params)` — draw candidate control knot sequences
- `update_params(params, rollouts)` — update the policy distribution from rollouts

The base class provides `optimize()` (the main MPC loop), `rollout_with_randomizations()`, and `eval_rollouts()` (vmapped over samples). All rollouts are jit-compiled and vmapped for GPU parallelism.

**`SamplingParams`** — dataclass holding the current policy state: `tk` (knot times), `mean` (mean control knots), and `rng` (JAX PRNG key). The PRNG key lives here because JAX requires explicit key threading.

**`Trajectory`** — dataclass returned from rollouts: `controls`, `knots`, `costs`, `trace_sites`.

**`RiskStrategy` (`hydrax/risk.py`)** — aggregates costs across domain randomizations. Options include `AverageCost`, `WorstCase`, `ExponentialWeightedAverage`, `ValueAtRisk`, `ConditionalValueAtRisk`.

### Control Parameterization

Controls are parameterized as spline knots, not as a flat sequence of actions. The knots are interpolated to the full control horizon via `hydrax/utils/spline.py` (using `interpax`). Supported `spline_type` values: `"zero"` (ZOH), `"linear"`, `"cubic"`. Knot times `tk` advance each MPC step for warm-starting.

### Algorithm Implementations (`hydrax/algs/`)

| File | Algorithm | Description |
|------|-----------|-------------|
| `predictive_sampling.py` | Predictive Sampling | Select minimum-cost rollout |
| `mppi.py` | MPPI | Exponentially weighted average of samples |
| `cem.py` | CEM | Fit Gaussian to top-k samples |
| `dial.py` | DIAL-MPC | MPPI with dual-loop annealing |
| `evosax.py` | Evosax | Wrapper for 30+ evolution strategies |
| `mppi_cma.py` | MPPI-CMA | MPPI with CMA-ES adaptive distribution |
| `er_cma.py` | ER-CMA | Entropy-regularized CMA |

### Simulation (`hydrax/simulation/`)

- `deterministic.py` — `run_interactive()`: synchronous MPC + mujoco.viewer for interactive demos
- `asynchronous.py` — async simulation loop for more realistic timing

### Examples (`examples/`)

Each example file (e.g., `pendulum.py`) accepts an algorithm name as a CLI argument and constructs the task, algorithm, and simulation loop. They serve as the primary reference for wiring everything together.

## Key Patterns

- **GPU parallelism**: `jax.vmap` over samples and domain randomizations; everything is jitted.
- **Domain randomization**: override `domain_randomize_*` in `Task`; costs are combined via `risk_strategy`.
- **Warm-starting**: `optimize()` re-evaluates the old spline at new knot times each step.
- **Ruff**: strict linting with Google docstring style and required type annotations.
