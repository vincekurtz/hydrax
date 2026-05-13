import jax
import jax.numpy as jnp
import pytest
from mujoco import mjx

from hydrax.algs.mtp import MTP
from hydrax.tasks.pendulum import Pendulum


def test_open_loop() -> None:
    """Use MTP for open-loop pendulum swingup."""
    task = Pendulum()
    opt = MTP(
        task,
        num_samples=32,
        m_pts=3,
        n_per_layer=50,
        num_elites=4,
        sigma_start=1.0,
        sigma_min=0.1,
        sigma_max=2.0,
        beta=0.5,
        alpha=0.5,
        mtp_interpolation="akima",
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
    )
    jit_opt = jax.jit(opt.optimize)

    state = mjx.make_data(task.model)
    params = opt.init_params()

    for _ in range(100):
        params, _ = jit_opt(state, params)

    knots = params.mean[None]
    tk = jnp.linspace(0.0, opt.plan_horizon, opt.num_knots)
    tq = jnp.linspace(0.0, opt.plan_horizon - opt.dt, opt.ctrl_steps)
    controls = opt.interp_func(tq, tk, knots)
    _, final_rollout = jax.jit(opt.eval_rollouts)(
        task.model, state, controls, knots
    )

    total_cost = jnp.sum(final_rollout.costs[0])
    assert total_cost <= 9.0
    assert jnp.all(params.cov >= opt.sigma_min**2)
    assert jnp.all(params.cov <= opt.sigma_max**2)


@pytest.mark.parametrize("mtp_interpolation", ["akima", "bspline", "linear"])
@pytest.mark.parametrize("m_pts", [2, 3, 5])
def test_sample_knots_shape(mtp_interpolation: str, m_pts: int) -> None:
    """Knot sampler returns the right shape across interpolation/m_pts combos."""
    if mtp_interpolation == "bspline" and m_pts < 3:
        pytest.skip("bspline requires m_pts >= degree + 1")

    task = Pendulum()
    num_samples = 16
    num_knots = 11
    opt = MTP(
        task,
        num_samples=num_samples,
        m_pts=m_pts,
        n_per_layer=20,
        num_elites=4,
        sigma_start=0.5,
        sigma_min=0.1,
        sigma_max=1.0,
        beta=0.5,
        mtp_interpolation=mtp_interpolation,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=num_knots,
    )
    params = opt.init_params(seed=0)
    knots, _ = opt.sample_knots(params)
    assert knots.shape == (num_samples, num_knots, task.model.nu)


def test_bspline_invalid_m_pts_raises() -> None:
    """Constructing MTP with bspline + m_pts < degree+1 raises."""
    task = Pendulum()
    with pytest.raises(ValueError, match="degree"):
        MTP(
            task,
            num_samples=16,
            m_pts=2,
            num_elites=2,
            mtp_interpolation="bspline",
            degree=2,
            plan_horizon=1.0,
            spline_type="zero",
            num_knots=11,
        )


def test_num_elites_too_large_raises() -> None:
    """Constructing MTP with num_elites >= num_samples raises."""
    task = Pendulum()
    with pytest.raises(ValueError, match="num_elites"):
        MTP(
            task,
            num_samples=8,
            num_elites=8,
            plan_horizon=1.0,
            spline_type="zero",
            num_knots=11,
        )


@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_beta_extremes_keep_sample_count(beta: float) -> None:
    """Beta in {0.0, 1.0} still produces exactly num_samples rollouts."""
    task = Pendulum()
    num_samples = 16
    opt = MTP(
        task,
        num_samples=num_samples,
        m_pts=3,
        num_elites=4,
        beta=beta,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
    )
    params = opt.init_params(seed=0)
    knots, _ = opt.sample_knots(params)
    assert knots.shape[0] == num_samples


if __name__ == "__main__":
    test_open_loop()
    for interp in ["akima", "bspline", "linear"]:
        for m in [2, 3, 5]:
            if interp == "bspline" and m < 3:
                continue
            test_sample_knots_shape(interp, m)
    test_bspline_invalid_m_pts_raises()
    test_num_elites_too_large_raises()
    for b in [0.0, 1.0]:
        test_beta_extremes_keep_sample_count(b)
    print("All MTP tests passed!")
