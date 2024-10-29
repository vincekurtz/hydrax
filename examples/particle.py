import sys

import evosax
import mujoco

from hydrax import ROOT
from hydrax.algs import MPPI, Evosax, PredictiveSampling
from hydrax.risk import WorstCase
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.particle import Particle

"""
Run an interactive simulation of the particle tracking task.

Double click on the green target, then drag it around with [ctrl + right-click].
"""

# Define the task (cost and dynamics)
task = Particle()

# Set the controller based on command-line arguments
if len(sys.argv) == 1 or sys.argv[1] == "ps":
    print("Running predictive sampling")
    ctrl = PredictiveSampling(
        task,
        num_samples=16,
        noise_level=0.1,
        num_randomizations=10,
        risk_strategy=WorstCase(),
    )

elif sys.argv[1] == "mppi":
    print("Running MPPI")
    ctrl = MPPI(task, num_samples=16, noise_level=0.3, temperature=0.01)

elif sys.argv[1] == "cmaes":
    print("Running CMA-ES")
    ctrl = Evosax(task, evosax.Sep_CMA_ES, num_samples=16, elite_ratio=0.5)

elif sys.argv[1] == "samr":
    print("Running genetic algorithm with Self-Adaptation Mutation Rate (SAMR)")
    ctrl = Evosax(task, evosax.SAMR_GA, num_samples=16)

elif sys.argv[1] == "de":
    print("Running Differential Evolution (DE)")
    ctrl = Evosax(task, evosax.DE, num_samples=16)

elif sys.argv[1] == "gld":
    print("Running Gradient-Less Descent (GLD)")
    ctrl = Evosax(task, evosax.GLD, num_samples=16)

elif sys.argv[1] == "rs":
    print("Running uniform random search")
    es_params = evosax.strategies.random.EvoParams(
        range_min=-1.0,
        range_max=1.0,
    )
    ctrl = Evosax(
        task, evosax.RandomSearch, num_samples=16, es_params=es_params
    )

else:
    print("Usage: python particle.py [ps|mppi|cmaes|samr|de|gld|rs]")
    sys.exit(1)

# Define the model used for simulation
mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/particle/scene.xml")
mj_data = mujoco.MjData(mj_model)

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    show_traces=True,
    max_traces=5,
)
