import mujoco

from hydrax import ROOT
from hydrax.algs import PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.cart_pole import CartPole

"""
Run an interactive simulation of a cart-pole swingup
"""

# Define the task (cost and dynamics)
task = CartPole()

# Set up the controller
ctrl = PredictiveSampling(task, num_samples=128, noise_level=0.3)

# Define the model used for simulation
mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/cart_pole/scene.xml")
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
