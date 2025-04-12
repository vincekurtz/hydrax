import mujoco

from hydrax.algs import PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.double_cart_pole import DoubleCartPole

"""
Run an interactive simulation of a double pendulum on a cart. Only the cart
is actuated, and the goal is to swing up the pendulum and balance it upright.
"""

# Define the task (cost and dynamics)
task = DoubleCartPole()

# Set up the controller
ctrl = PredictiveSampling(
    task,
    num_samples=1024,
    noise_level=0.3,
    plan_horizon=1.0,
    spline_type="cubic",
    num_knots=4,
)

# Define the model used for simulation
mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=20,
    fixed_camera_id=0,
    show_traces=True,
    max_traces=1,
)
