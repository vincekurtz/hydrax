# Hyperparameter sweep — reproducibility reference

This document captures every piece of information needed to reproduce the
`results/sweep_all.csv` sweep without relying on the
`hydrax/benchmarking/` directory. It assumes hydrax's algorithm classes
(`PredictiveSampling`, `MPPI`, `CEM`, `DIAL`, `Evosax`, `MppiCma`, `ErCma`)
and task classes (`Pendulum`, `CartPole`, ...) are available and unchanged
from commits in the `er_cma` branch.

## 1. Environment

- Python 3.12+, CUDA 13.
- Dependency manager: `uv sync`.
- Single GPU. The sweep is sequential (no parallelism across configs); a
  fresh `Task` and controller are constructed per trial.
- The sweep is fully deterministic given a single seed (default `0`),
  modulo any nondeterminism in `mjx.step` / GPU reductions.

## 2. Shared experimental setup

| Setting | Value |
|---|---|
| `num_samples` per `optimize()` | **256** |
| `iterations` per `optimize()` | **1** (algorithm default) |
| `num_randomizations` | **1** (default; no domain randomization) |
| `risk_strategy` | `None` (defaults to AverageCost) |
| Seed | **0** (used for both controller RNG and `seed` arg) |
| Episodes per (task, algorithm, config) | **1** |
| Cost metric | Cumulative running cost across the episode: `Σ task.running_cost(state, u) * task.dt` over all simulation steps |
| Wall-time metric | Excludes JIT compilation (a full warmup runs first with `block_until_ready`) |

## 3. Per-task configuration

For each task, the sweep uses the same per-task `plan_horizon`, `num_knots`,
`spline_type`, `control_frequency`, and `episode_length` for all algorithms.

| task | factory | plan_horizon (s) | num_knots | spline_type | control_freq (Hz) | episode_length (s) | dt (s) | nq | nv | nu |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| pendulum | `Pendulum` | 1.0 | 11 | zero | 50 | 3.0 | 0.02 | 1 | 1 | 1 |
| cart_pole | `CartPole` | 1.0 | 4 | cubic | 50 | 4.0 | 0.01 | 2 | 2 | 1 |
| double_cart_pole | `DoubleCartPole` | 1.0 | 4 | cubic | 50 | 5.0 | 0.01 | 3 | 3 | 1 |
| particle | `Particle` | 0.25 | 11 | zero | 50 | 2.0 | 0.01 | 2 | 2 | 2 |
| pusht | `PushT` | 0.5 | 6 | zero | 50 | 4.0 | 0.01 | 5 | 5 | 2 |
| walker | `Walker` | 0.6 | 5 | zero | 50 | 3.0 | 0.01 | 9 | 9 | 6 |
| cube | `CubeRotation` | 0.25 | 4 | zero | 25 | 3.0 | 0.01 | 23 | 22 | 16 |
| humanoid_standup | `HumanoidStandup` | 0.6 | 4 | zero | 50 | 3.0 | 0.02 | 36 | 35 | 29 |

`dt`, `nq`, `nv`, `nu` come from the loaded `mj_model` and are listed for
sanity-checking; the sweep does not override them.

### Derived per-task control loop counts

For each task:

- `sim_per_ctrl = round(1 / control_frequency / dt)`
- `num_ctrl_steps = round(episode_length / dt / sim_per_ctrl)`

| task | sim_per_ctrl | num_ctrl_steps |
|---|---:|---:|
| pendulum | 1 | 150 |
| cart_pole | 2 | 200 |
| double_cart_pole | 2 | 250 |
| particle | 2 | 100 |
| pusht | 2 | 200 |
| walker | 2 | 150 |
| cube | 4 | 75 |
| humanoid_standup | 1 | 150 |

## 4. Fixed initial conditions

Each trial uses a single deterministic initial condition matching the
`examples/` scripts. These ICs are identical across algorithms and configs
for a given task. `mocap_pos` is left at the MuJoCo XML default unless
otherwise noted.

| task | qpos | qvel | mocap_pos |
|---|---|---|---|
| pendulum | `[0.0]` | `[0.0]` | (no mocap) |
| cart_pole | `[0.0, 0.0]` | `[0.0, 0.0]` | (no mocap) |
| double_cart_pole | `[0.0, 0.0, 0.0]` | `[0.0, 0.0, 0.0]` | (no mocap) |
| particle | `[0.0, 0.0]` | `[0.0, 0.0]` | `[[0.25, 0.0, 0.01]]` (XML default) |
| pusht | `[0.1, 0.1, 1.3, 0.0, 0.0]` | `[0, 0, 0, 0, 0]` | `[[0.0, 0.0, 0.009]]` (XML default) |
| walker | `[0]*9` | `[0]*9` | (no mocap) |
| cube | XML default qpos (see below) | `[0]*22` | `[[0.325, 0.17, 0.0475]]` (XML default) |
| humanoid_standup | `keyframe("stand").qpos` with `qpos[3:7] = [0.7, 0, -0.7, 0]` (see below) | `[0]*35` | (no mocap) |

**Cube default qpos (23 elements)**:

```
[-0.8, 0.0, -0.8, -0.8, -0.8, 0.0, -0.8, -0.8,
 -0.8, 0.0, -0.8, -0.8, -0.8, -0.8, -0.8, 0.0,
  0.11, 0.0, 0.1,         # cube position
  1.0, 0.0, 0.0, 0.0]     # cube quaternion
```

**Humanoid_standup fixed qpos (36 elements)**:

```
[0.0, 0.0, 0.79,           # base position
 0.7, 0.0, -0.7, 0.0,      # knocked-over base orientation (quat)
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.2, 0.2, 0.0, 1.28, 0.0, 0.0, 0.0,
 0.2, -0.2, 0.0, 1.28, 0.0, 0.0, 0.0]
```

(Equivalent to `qpos = MjModel.keyframe("stand").qpos.copy()`, then
`qpos[3:7] = [0.7, 0.0, -0.7, 0.0]`.)

## 5. Hyperparameter grids

### 5.1 Fixed algorithm hyperparameters

Algorithm constructor arguments not listed in the per-algorithm grids
below take these fixed values:

- All algorithms: `num_samples=256`, `plan_horizon`/`num_knots`/`spline_type`
  per task, `seed=0`, `iterations=1` (default), `num_randomizations=1`
  (default), `risk_strategy=None` (default).
- `CEM`: `sigma_min=1e-3`, `explore_fraction=0.0`.
- `MppiCma`: `minimum_noise_level=1e-3`.
- `ErCma`: `minimum_noise_level=1e-3`, `maximum_noise_level=1.0`.

### 5.2 Phase 1 grids (independent per task)

**PS (5 configs)** — sweep `noise_level`:

```
noise_level ∈ {0.05, 0.1, 0.3, 0.5, 1.0}
```

**MPPI (16 configs)** — full grid:

```
noise_level ∈ {0.1, 0.3, 0.5, 1.0}
temperature ∈ {0.01, 0.1, 0.5, 1.0}
```

**CEM (12 configs)** — full grid:

```
sigma_start ∈ {0.1, 0.3, 0.5, 1.0}
num_elites  ∈ {4, 8, 16}
```

### 5.3 Phase 2 grids (per-task: use MPPI's best `noise_level` from Phase 1)

After Phase 1, the per-task minimum-cost MPPI config is selected and its
`noise_level` (denoted `NL*` below) is reused as the *fixed* initial noise
level for Phase 2 algorithms. The per-task `NL*` and the per-task best
MPPI temperature (`T_mppi*`) used by DIAL are:

| task | NL\* | T_mppi\* | best MPPI cost |
|---|---:|---:|---:|
| pendulum | 1.0 | 0.01 | 7.9980 |
| cart_pole | 0.3 | 0.01 | 5.1924 |
| double_cart_pole | 0.1 | 0.01 | 19.3256 |
| particle | 1.0 | 0.01 | 0.1180 |
| pusht | 0.3 | 0.01 | 0.5215 |
| walker | 0.5 | 0.01 | 1.5600 |
| cube | 1.0 | 0.01 | 0.1141 |
| humanoid_standup | 1.0 | 0.5 | 32.0432 |

**DIAL (9 configs)** — `noise_level = NL*`, `temperature = T_mppi*`,
sweep both betas:

```
beta_opt_iter ∈ {0.5, 1.0, 2.0}
beta_horizon  ∈ {0.5, 1.0, 2.0}
```

**MPPI-CMA (16 configs)** — `initial_noise_level = NL*`, sweep `(α, T)`:

```
covariance_adaptation_rate ∈ {0.05, 0.1, 0.3, 1.0}
temperature                ∈ {0.01, 0.1, 0.5, 1.0}
```

**ER-CMA (144 configs)** — `initial_noise_level = NL*`, sweep
`(α, T, ent_init, ent_final)` over the full grid, subject to two
constraints:

```
covariance_adaptation_rate ∈ {0.05, 0.1, 0.3, 1.0}
temperature                ∈ {0.01, 0.1, 0.5, 1.0}
initial_entropy_bonus      ∈ {0.0, 0.1, 0.3, 0.5}
final_entropy_bonus        ∈ {0.0, 0.1, 0.3, 0.5}
```

Excluded combinations:
- `final_entropy_bonus < initial_entropy_bonus` (entropy should not
  decrease across the horizon).
- `initial_entropy_bonus == 0 AND final_entropy_bonus == 0` (equivalent
  to MPPI-CMA).

This leaves 9 valid `(initial, final)` pairs:

```
(0.0, 0.1), (0.0, 0.3), (0.0, 0.5),
(0.1, 0.1), (0.1, 0.3), (0.1, 0.5),
(0.3, 0.3), (0.3, 0.5),
(0.5, 0.5)
```

→ 9 × 4 (α) × 4 (T) = 144 configs.

### 5.4 Per-task config counts

| algorithm | configs per task |
|---|---:|
| PS | 5 |
| MPPI | 16 |
| CEM | 12 |
| DIAL | 9 |
| MPPI-CMA | 16 |
| ER-CMA | 144 |
| **total** | **202** |

CMA-ES (Evosax wrapper) is **not** included in the sweep — no scalar
hyperparameters are exposed by the wrapper.

Across 8 tasks: **1616 total trials**.

## 6. Per-trial procedure

For each `(task, algorithm, hparams)` tuple:

1. **Construct task**: call the factory (e.g., `Pendulum()`).
2. **Build initial state**: pull `(qpos, qvel, mocap_pos)` from §4.
3. **Construct controller**: instantiate the algorithm with the
   hyperparameters from §5.1 + the chosen grid point, plus the per-task
   `plan_horizon`, `num_knots`, `spline_type`, and `seed=0`.
4. **Initialize MJX state**:
   - `data = task.make_data()` (uses model defaults).
   - Override `qpos`, `qvel` with §4 values.
   - Override `mocap_pos` only if §4 specifies an override (otherwise
     keep the default from `make_data`).
5. **Initialize policy params**: `params = controller.init_params(seed=0)`.
6. **Build the JIT'd MPC step function** (closure over `controller`,
   `task`, `dt = task.dt`, `sim_per_ctrl`):

   ```python
   @jax.jit
   def mpc_step(state, params):
       params, _ = controller.optimize(state, params)  # discard rollouts
       tq = jnp.arange(sim_per_ctrl) * dt + state.time
       controls = controller.interp_func(
           tq, params.tk, params.mean[None, ...])[0]

       def body(carry, u):
           carry = carry.replace(ctrl=u)
           carry = mjx.step(task.model, carry)
           return carry, task.running_cost(carry, u) * dt

       state, costs = jax.lax.scan(body, state, controls)
       return state, params, jnp.sum(costs)
   ```

7. **Warm up JIT**: call `mpc_step` once on the initial state, then
   `jax.block_until_ready(...)` on the returned cost to ensure
   compilation finishes before timing starts.
8. **Re-initialize state and policy params** to the same values as steps
   4–5 (the warmup call mutates them).
9. **Run the episode**: loop `for _ in range(num_ctrl_steps)`, each
   iteration calling `mpc_step` and accumulating `segment_cost` into
   `total_cost`.
10. **Report**:
    - `cost = float(np.asarray(total_cost))` (blocks on device to flush).
    - `wall_time = time.time() - start` (started just before the loop).

### Notes on the procedure

- **`controller.optimize(state, params)` discards rollouts** (the second
  return) so XLA can skip allocating large output buffers. Three things
  are fused inside `mpc_step` (plan + interpolate + simulate) so XLA can
  optimize across the boundaries and avoid two extra kernel dispatches
  per control step.
- **No domain randomization** is used (`num_randomizations=1`,
  `risk_strategy=None`).
- **Spline interpolation** uses `controller.interp_func`, which is the
  task's chosen `spline_type` (`zero`, `linear`, or `cubic`).
- **The same `seed=0`** is passed to the controller constructor AND to
  `init_params`. The trial-level seed therefore controls both
  algorithm RNG and (in the more general randomized-IC setting) IC; in
  this sweep the IC is fixed and only the algorithm RNG depends on it.

## 7. Sweep driver: two phases per task

For each task (in registry order: `pendulum`, `cart_pole`,
`double_cart_pole`, `particle`, `pusht`, `walker`, `cube`,
`humanoid_standup`):

**Phase 1** — run independent grids:

1. Run all 5 PS configs.
2. Run all 16 MPPI configs; track which produced the minimum cost.
3. Run all 12 CEM configs.

After Phase 1, look up `(NL*, T_mppi*)` from the per-task min-cost MPPI
row (§5.3 table).

**Phase 2** — run grids that fix `NL*`:

4. Run all 9 DIAL configs (with `noise_level=NL*` and
   `temperature=T_mppi*`).
5. Run all 16 MPPI-CMA configs (with `initial_noise_level=NL*`).
6. Run all 144 ER-CMA configs (with `initial_noise_level=NL*`).

All results — successful or errored — are appended row-by-row to a CSV.

## 8. CSV schema

`results/sweep_all.csv` columns:

```
task, algorithm, seed, num_samples,
noise_level, temperature, sigma_start, sigma_min, num_elites,
explore_fraction, beta_opt_iter, beta_horizon,
initial_noise_level, minimum_noise_level, maximum_noise_level,
covariance_adaptation_rate, initial_entropy_bonus, final_entropy_bonus,
cost, wall_time_s, status, error
```

- Hyperparameter columns not used by a given algorithm are left blank.
- `status` ∈ {`ok`, `error`}; on error, `cost` and `wall_time_s` are
  `NaN` and `error` contains `repr(exception)`.

Row count for the full sweep: 1616 data rows + 1 header = 1617 lines.

## 9. Selected results (sanity-check anchors)

These exact values came out of the sweep used to produce the optimal
hyperparameter tables. A reimplementation should reproduce them to
within ~1e-3 (any larger drift indicates a divergence in setup).

| task | best MPPI (cost, NL, T) | best ER-CMA cost |
|---|---|---:|
| pendulum | 7.9980 at (1.0, 0.01) | 8.1556 |
| cart_pole | 5.1924 at (0.3, 0.01) | 5.1243 |
| double_cart_pole | 19.3256 at (0.1, 0.01) | 19.0390 |
| particle | 0.1180 at (1.0, 0.01) | 0.1180 |
| pusht | 0.5215 at (0.3, 0.01) | 0.5083 |
| walker | 1.5600 at (0.5, 0.01) | 1.2608 |
| cube | 0.1141 at (1.0, 0.01) | 0.0867 |
| humanoid_standup | 32.0432 at (1.0, 0.5) | 31.5025 |

Total wall time: **45,643.8 s** (~12.7 h) on a single GPU for all 1616
configs, dominated by per-trial JIT compilation (~2.5 s per fresh
controller) — see notes in `sweep_all_results.md` for per-algorithm
behavior.
