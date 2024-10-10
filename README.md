# Hydrax

Sampling-based model predictive control on GPU with JAX and
[MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html).

![A planar walker running MPPI](img/walker.gif)

## About

Implements several sampling-based MPC algorithms, including
[predictive sampling](https://arxiv.org/abs/2212.00541) and
[MPPI](https://arxiv.org/abs/1707.02342), on GPU.

This software is heavily inspired by
[MJPC](https://github.com/google-deepmind/mujoco_mpc), but focuses exclusively
on sampling-based algorithms, and runs on hardware accelerators via JAX and MJX.

## Setup (conda)

Set up a conda env with cuda support (first time only):

```bash
conda env create -f environment.yml
```

Enter the conda env:

```bash
conda activate hydrax
```

Install the package and dependencies:

```bash
pip install -e . --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

(Optional) set up pre-commit hooks:

```bash
pre-commit autoupdate
pre-commit install
```

(Optional) run unit tests:

```bash
pytest
```

## Usage

Launch an interactive pendulum swingup simulation with predictive sampling:

```bash
python examples/pendulum.py ps
```

Launch an interactive planar walker simulation (shown above) with MPPI:

```bash
python examples/walker mppi
```

Other demos can be found in the `examples` folder.

### Design your own task

Hydrax considers optimal control problems of the form

```math
\begin{align}
\min_{u_t} & \sum_{t=0}^{T-1} \ell(x_t, u_t) + \phi(x_T), \\
\mathrm{s.t.}~& x_{t+1} = f(x_t, u_t),
\end{align}
```
where $x_t$ is the system state and $u_t$ is the control input at time $t$, and
the system dynamics $f(x_t, u_t)$ are defined by a mujoco MJX model.

To design a new task, you'll need to specify the cost ($\ell$, $\phi$) and the
dynamics ($f$). You can do this by creating a new class that inherits from
[`hydrax.task_base.Task`](hydrax/task_base.py):

```python
class MyNewTask(Task):
    def __init__(self):
        # Create or load a mujoco model defining the dynamics
        mj_model = ...
        super().__init__(mj_model, ...)

    def running_cost(self, x: mjx.Data, u: jax.Array) -> jax.Array:
        # Implement the running cost here
        return ...

    def terminal_cost(self, x: jax.Array) -> jax.Array:
        # Implement the terminal cost here
        return ...
```


The dynamics ($f$) are specified by a `mujoco.MjModel` that is passed to the
constructor. For the cost, simply implement the `running_cost` ($\ell$) and
`terminal_cost` ($\phi$) abstract methods.

See [`hydrax.tasks`](hydrax/tasks) for some example task implementations.
