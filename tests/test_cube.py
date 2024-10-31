import time
from typing import Tuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.tasks.cube import CubeRotation


def test_mjx_model() -> None:
    """Test that the MJX model runs without crashing."""
    rng = jax.random.key(0)

    mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/cube/scene.xml")
    model = mjx.put_model(mj_model)
    data = mjx.make_data(model)

    nu = mj_model.nu
    assert nu == 16
    assert mj_model.nq == 16 + 7  # hand and floating cube
    assert mj_model.nv == 16 + 6

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


def test_task() -> None:
    """Set up the cube rotation task."""
    task = CubeRotation()

    state = mjx.make_data(task.model)
    assert isinstance(state, mjx.Data)
    state = state.replace(mocap_quat=jnp.array([[0.0, 1.0, 0.0, 0.0]]))
    state = jax.jit(mjx.forward)(task.model, state)

    # Check cube position relative to the target grasp position
    cube_position = task._get_cube_position_err(state)
    assert jnp.all(cube_position == jnp.array([0.0, 0.0, 0.07]))

    # Check cube orientation relative to the target grasp orientation
    cube_orientation = task._get_cube_orientation_err(state)
    assert cube_orientation.shape == (3,)
    assert jnp.allclose(cube_orientation, jnp.array([-jnp.pi, 0.0, 0.0]))

    # Test observation
    obs = task.get_obs(state)
    assert jnp.all(obs[0:3] == cube_position)
    assert jnp.all(obs[3:6] == cube_orientation)
    assert obs.shape == (6 + 16 + task.model.nv,)


if __name__ == "__main__":
    test_mjx_model()
    test_task()
