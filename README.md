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

## Design your own task

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
    def __init__(self, ...):
        # Create or load a mujoco model defining the dynamics
        mj_model = ...
        super().__init__(mj_model, ...)

    def running_cost(self, x: mjx.Data, u: jax.Array) -> float:
        # Implement the running cost (l) here
        return ...

    def terminal_cost(self, x: jax.Array) -> float:
        # Implement the terminal cost (phi) here
        return ...
```


The dynamics ($f$) are specified by a `mujoco.MjModel` that is passed to the
constructor. Other constructor arguments specify the planning horizon $T$, any
control limits, and other details.

For the cost, simply implement the `running_cost` ($\ell$) and
`terminal_cost` ($\phi$) methods.

See [`hydrax.tasks`](hydrax/tasks) for some example task implementations.

## Implement your own control algorithm

Hydrax considers sampling-based MPC algorithms that follow the following
[generic structure](https://arxiv.org/abs/2409.14562):

![Generic sampling-based MPC algorithm block](img/spc_alg.png)

The meaning of the parameters $\theta$ differ depending on the algorithm. In
predictive sampling, for example, $\theta$ is the mean of a Gaussian distribution
that the controls $U = [u_0, u_1, ...]$ are sampled from.

To implement a new planning algorithm, you'll need to inherit from
[`hydrax.alg_base.SamplingBasedController`](hydrax/alg_base.py) and implement
the four methods shown below:

```python
class MyControlAlgorithm(SamplingBasedController):

    def init_params(self) -> Any:
        # Initialize the policy parameters (theta).
        return params

    def sample_controls(self, params: Any) -> Tuple[jax.Array, Any]:
        # Sample control sequences U from the policy. Return the samples
        # and the (updated) parameters.
        ...
        return controls, params

    def update_params(self, params: Any, rollouts: Trajectory) -> Any:
        # Update the policy parameters (theta) based on the trajectory data
        # (costs, controls, observations, etc) stored in the rollouts.
        ...
        return new_params

    def get_action(self, params: Any, t: float) -> Any:
        # Return the control action applied t seconds into the trajectory.
        ...
        return u
```

These four methods define a unique sampling-based MPC algorithm. Hydrax takes
care of the rest, including parallelizing rollouts on GPU and collecting the
rollout data in a [`Trajectory`](hydrax/alg_base.py) object.

**Note**: because of
[the way JAX handles randomness](https://jax.readthedocs.io/en/latest/random-numbers.html)
in a functional programming paradigm, we assume the PRNG key is stored as one of
parameters $\theta$. This is why `sample_controls` returns updated parameters
along with the control samples $U^{(1:N)}$.

For some examples, take a look at [`hydrax.algs`](hydrax/algs).
