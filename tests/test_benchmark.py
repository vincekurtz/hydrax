import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from hydrax.algs.predictive_sampling import PredictiveSampling
from hydrax.benchmarking import benchmark
from hydrax.tasks.pendulum import Pendulum


def test_benchmark_shapes() -> None:
    """Benchmark returns arrays of the documented shapes."""
    task = Pendulum()
    ctrl = PredictiveSampling(
        task,
        num_samples=8,
        noise_level=0.1,
        plan_horizon=0.5,
        spline_type="zero",
        num_knots=5,
    )

    state0 = mjx.make_data(task.model)
    num_steps = 10

    running, rollouts = benchmark(task, ctrl, state0, num_steps=num_steps)

    assert running.shape == (num_steps,)
    assert rollouts.shape == (num_steps, ctrl.num_samples)
    assert jnp.all(jnp.isfinite(running))
    assert jnp.all(jnp.isfinite(rollouts))


def test_benchmark_save(tmp_path: Path) -> None:
    """Benchmark can save its results to an .npz file."""
    task = Pendulum()
    ctrl = PredictiveSampling(
        task,
        num_samples=4,
        noise_level=0.1,
        plan_horizon=0.5,
        spline_type="zero",
        num_knots=5,
    )

    state0 = mjx.make_data(task.model)
    num_steps = 5
    path = str(tmp_path / "bench.npz")

    running, rollouts = benchmark(
        task, ctrl, state0, num_steps=num_steps, save_path=path
    )

    assert os.path.exists(path)
    loaded = np.load(path)
    np.testing.assert_array_equal(loaded["running_costs"], np.asarray(running))
    np.testing.assert_array_equal(loaded["rollout_costs"], np.asarray(rollouts))


def test_benchmark_progress() -> None:
    """Closed-loop simulation actually advances state.time across steps."""
    task = Pendulum()
    ctrl = PredictiveSampling(
        task,
        num_samples=8,
        noise_level=0.1,
        plan_horizon=0.5,
        spline_type="zero",
        num_knots=5,
    )

    # Start with a non-trivial state so the running cost is non-zero.
    state0 = mjx.make_data(task.model)
    state0 = state0.replace(qpos=jnp.array([1.0]), qvel=jnp.array([0.0]))

    num_steps = 15
    running, rollouts = benchmark(task, ctrl, state0, num_steps=num_steps)

    # Costs should not all be the same constant — state is evolving.
    assert float(jnp.std(running)) > 0.0
    assert float(jnp.std(rollouts[:, 0])) > 0.0


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    test_benchmark_shapes()
    with tempfile.TemporaryDirectory() as d:
        test_benchmark_save(Path(d))
    test_benchmark_progress()
    print("All benchmark tests passed.")
