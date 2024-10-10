import time

import jax
import mujoco
from mujoco import mjx

from hydrax import ROOT


def test_mjx_model() -> None:
    """Test that the MJX model runs without crashing."""
    mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/leap/leap_rh.xml")
    model = mjx.put_model(mj_model)
    data = mjx.make_data(model)

    nu = mj_model.nu
    assert nu == 16

    assert isinstance(model, mjx.Model)
    assert isinstance(data, mjx.Data)

    @jax.jit
    def step(data: mjx.Data, u: jax.Array) -> mjx.Data:
        """Do a single step of the forward dyanmics."""
        data = data.replace(ctrl=u)
        return mjx.step(model, data)

    st = time.time()
    data = step(data, jax.numpy.zeros(nu))
    print(f"Time to jit: {time.time() - st:.3f}s")


if __name__ == "__main__":
    test_mjx_model()
