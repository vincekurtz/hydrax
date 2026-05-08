import jax.numpy as jnp
from mujoco import mjx

from hydrax.algs.er_cma import ErCma
from hydrax.tasks.pendulum import Pendulum


def test_params_update() -> None:
    """Test that the ErCMA parameter update works."""
    task = Pendulum()
    opt = ErCma(
        task,
        num_samples=32,
        initial_noise_level=0.1,
        minimum_noise_level=0.05,
        maximum_noise_level=0.2,
        initial_entropy_bonus=0.0,
        final_entropy_bonus=0.0,
        covariance_adaptation_rate=0.1,
        temperature=0.01,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
    )

    params = opt.init_params()

    assert params.mean.shape == (opt.num_knots, task.model.nu)
    assert params.covariance.shape == (
        opt.num_knots,
        task.model.nu,
        task.model.nu,
    )

    knots, params = opt.sample_knots(params)
    assert knots.shape == (opt.num_samples, opt.num_knots, task.model.nu)

    state = mjx.make_data(task.model)
    new_params, _ = opt.optimize(state, params)

    assert new_params.mean.shape == (opt.num_knots, task.model.nu)
    assert new_params.covariance.shape == (
        opt.num_knots,
        task.model.nu,
        task.model.nu,
    )
    assert not jnp.allclose(new_params.covariance, params.covariance)


if __name__ == "__main__":
    test_params_update()