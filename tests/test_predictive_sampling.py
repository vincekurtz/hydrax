import jax
from mujoco import mjx

from hydra.algs.predictive_sampling import PredictiveSampling
from hydra.tasks.pendulum import Pendulum


def test_predictive_sampling() -> None:
    """Test the PredictiveSampling algorithm."""
    task = Pendulum()
    opt = PredictiveSampling(task, num_samples=10, noise_level=0.1)

    # Initialize the policy parameters
    params = opt.init_params()
    assert params.mean.shape == (task.planning_horizon - 1, 1)
    assert isinstance(params.rng, jax._src.prng.PRNGKeyArray)

    # Sample control sequences from the policy
    controls, new_params = opt.sample_controls(params)
    assert controls.shape == (opt.num_samples + 1, task.planning_horizon - 1, 1)
    assert new_params.rng != params.rng

    # Roll out the control sequences
    state = mjx.make_data(task.model)
    rollouts = opt.eval_rollouts(state, controls)

    assert rollouts.costs.shape == (
        opt.num_samples + 1,
        task.planning_horizon,
    )
    assert rollouts.observations.shape == (
        opt.num_samples + 1,
        task.planning_horizon,
        2,
    )
    assert rollouts.controls.shape == (
        opt.num_samples + 1,
        task.planning_horizon - 1,
        1,
    )


if __name__ == "__main__":
    test_predictive_sampling()
