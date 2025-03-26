import time
from typing import Tuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.tasks.humanoid_mocap import HumanoidMocap
from hydrax.tasks.humanoid_standup import HumanoidStandup


def test_mjx_model() -> None:
    """Test that the MJX model runs without crashing."""
    rng = jax.random.key(0)

    mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/g1/scene.xml")
    model = mjx.put_model(mj_model)
    data = mjx.make_data(model)

    nu = mj_model.nu
    assert mj_model.nu + 6 == mj_model.nv

    assert isinstance(model, mjx.Model)
    assert isinstance(data, mjx.Data)

    @jax.jit
    def step(data: mjx.Data, rng: jax.Array) -> Tuple[mjx.Data, jax.Array]:
        """Do a single step of the forward dyanmics with a random input."""
        rng, sample_rng = jax.random.split(rng)
        u = jax.random.uniform(sample_rng, (nu,), minval=-1.0, maxval=1.0)
        data = data.replace(ctrl=u)
        return mjx.step(model, data), rng

    st = time.time()
    data, rng = step(data, rng)
    print(f"Time to jit: {time.time() - st:.3f}s")

    st = time.time()
    for _ in range(100):
        data, rng = step(data, rng)
    run_time = time.time() - st
    print(f"Time to run 100 steps: {run_time:.3f}s")
    sim_time = model.opt.timestep * 100
    print(f"Realtime rate: {sim_time / run_time:.3f}x")

    assert not jnp.any(jnp.isnan(data.qpos))
    assert not jnp.any(jnp.isnan(data.qvel))


def test_standup() -> None:
    """Test the humanoid standup task."""
    task = HumanoidStandup()
    assert task.orientation_sensor_id >= 0
    assert task.torso_id >= 0

    state = mjx.make_data(task.model)
    assert isinstance(state, mjx.Data)

    # Check sensor measurements
    state = mjx.forward(task.model, state)
    pz = task._get_torso_height(state)
    w = task._get_torso_orientation(state)
    assert pz > 0.0
    assert w.shape == (3,)

    ell = task.running_cost(state, jnp.zeros(task.model.nu))
    assert ell.shape == ()
    assert ell > 0.0

    phi = task.terminal_cost(state)
    assert phi.shape == ()
    assert phi > 0.0


def test_mocap() -> None:
    """Test the humanoid mocap task."""
    task = HumanoidMocap()
    assert task.reference is not None

    state = mjx.make_data(task.model)
    assert isinstance(state, mjx.Data)

    ell = task.running_cost(state, jnp.zeros(task.model.nu))
    assert ell.shape == ()
    assert ell > 0.0

    phi = task.terminal_cost(state)
    assert phi.shape == ()
    assert phi > 0.0


if __name__ == "__main__":
    test_mjx_model()
    test_standup()
    test_mocap()
