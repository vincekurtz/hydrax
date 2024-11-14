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
ctrl = PredictiveSampling(task, num_samples=512, noise_level=0.5)

# Define the model used for simulation
mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)
mj_data.qpos[1] = 3.14  # Set the pole to be facing down

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    fixed_camera_id=0,
    show_traces=True,
    max_traces=1,
)
