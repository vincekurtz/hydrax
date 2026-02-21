from hydrax.algs import PredictiveSampling
from hydrax.open_loop import trajectory_optimization
from hydrax.tasks.cart_pole import CartPole


def test_traj_opt() -> None:
    """Smoke test for open-loop cart-pole swingup."""
    task = CartPole()
    ctrl = PredictiveSampling(
        task,
        num_samples=128,
        noise_level=0.3,
        spline_type="cubic",
        plan_horizon=2.0,
        num_knots=4,
    )

    initial_state = ctrl.task.make_data()
    traj = trajectory_optimization(ctrl, initial_state, iterations=5)

    assert traj.qpos.shape == (200, 2)


if __name__ == "__main__":
    test_traj_opt()
