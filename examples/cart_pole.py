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
ctrl = PredictiveSampling(task, num_samples=512, noise_level=0.5)

# Define the model used for simulation
mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/cart_pole/scene.xml")
mj_data = mujoco.MjData(mj_model)

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    show_traces=True,
    max_traces=1,
)
