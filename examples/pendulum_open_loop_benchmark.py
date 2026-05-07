"""Open-loop convergence benchmark for the pendulum swingup task.

Each algorithm is run for a fixed number of iterations from the same initial
state and random seed.  The mean rollout cost is recorded at every iteration
and plotted as a convergence curve.

Usage::

    python examples/pendulum_open_loop_benchmark.py

Optional flags::

    --iterations N   Number of optimization iterations (default: 100)
    --samples N      Number of rollout samples per algorithm (default: 128)
    --save PATH      Save the figure to PATH instead of displaying it
"""

import argparse
import time

import matplotlib.pyplot as plt

from hydrax.algs import CEM, DIAL, MPPI, MppiCma, PredictiveSampling
from hydrax.benchmarking import run_open_loop_benchmark
from hydrax.tasks.pendulum import Pendulum

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--iterations",
    type=int,
    default=100,
    help="Number of optimization iterations per algorithm (default: 100)",
)
parser.add_argument(
    "--samples",
    type=int,
    default=128,
    help="Number of rollout samples per algorithm (default: 128)",
)
parser.add_argument(
    "--save",
    type=str,
    default=None,
    metavar="PATH",
    help="Save figure to PATH instead of displaying it",
)
args = parser.parse_args()

ITERATIONS = args.iterations
NUM_SAMPLES = args.samples
SEED = 0

# Shared spline / horizon settings
SHARED = dict(
    plan_horizon=1.0,
    spline_type="cubic",
    num_knots=10,
)

# ---------------------------------------------------------------------------
# Task and initial state
# ---------------------------------------------------------------------------

task = Pendulum(impl="warp")
initial_state = task.make_data()

# ---------------------------------------------------------------------------
# Controllers to compare
# ---------------------------------------------------------------------------

controllers = {
    "Predictive Sampling": PredictiveSampling(
        task,
        num_samples=NUM_SAMPLES,
        noise_level=0.5,
        **SHARED,
    ),
    "MPPI": MPPI(
        task,
        num_samples=NUM_SAMPLES,
        noise_level=0.5,
        temperature=0.1,
        **SHARED,
    ),
    "CEM": CEM(
        task,
        num_samples=NUM_SAMPLES,
        num_elites=max(4, NUM_SAMPLES // 8),
        sigma_start=0.5,
        sigma_min=0.05,
        **SHARED,
    ),
    "DIAL": DIAL(
        task,
        num_samples=NUM_SAMPLES,
        noise_level=0.5,
        beta_opt_iter=1.0,
        beta_horizon=2.0,
        temperature=0.1,
        **SHARED,
    ),
    "MPPI-CMA": MppiCma(
        task,
        num_samples=NUM_SAMPLES,
        initial_noise_level=0.5,
        temperature=0.1,
        minimum_noise_level=0.0,
        covariance_adaptation_rate=0.1,
        **SHARED,
    ),
}

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------

print(
    f"Benchmarking {len(controllers)} algorithms for {ITERATIONS} iterations "
    f"with {NUM_SAMPLES} samples each.\n"
)

results = {}
for name, ctrl in controllers.items():
    t0 = time.perf_counter()
    result = run_open_loop_benchmark(
        ctrl, initial_state, iterations=ITERATIONS, seed=SEED
    )
    elapsed = time.perf_counter() - t0

    # Block on JAX result to get accurate timing
    result.mean_costs.block_until_ready()

    results[name] = (result, elapsed)
    print(
        f"  {name:20s}  final mean cost: {float(result.mean_costs[-1]):.4f}"
        f"  best observed: {float(result.best_costs.min()):.4f}"
        f"  time: {elapsed:.2f}s"
    )

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))
iterations_axis = range(1, ITERATIONS + 1)

for name, (result, _) in results.items():
    ax.plot(iterations_axis, result.best_costs, label=name, linewidth=1.5)

ax.set_xlabel("Optimization iteration")
ax.set_ylabel("Cost")
ax.set_title("Open-loop optimization convergence — Pendulum swingup")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()

if args.save is not None:
    fig.savefig(args.save, dpi=150)
    print(f"\nFigure saved to {args.save}")
else:
    plt.show()
