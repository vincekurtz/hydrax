import mujoco

from hydrax.algs import PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.cart_pole import CartPole

"""
Run an interactive simulation of a cart-pole swingup
"""

# Define the task (cost and dynamics)
task = CartPole()

# Set up the controller
ctrl = PredictiveSampling(
    task,
    num_samples=128,
    noise_level=0.3,
    spline_type="cubic",
    T=1.0,
    dt=0.1,
    num_knots=11,
)

# Define the model used for simulation
mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)

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
