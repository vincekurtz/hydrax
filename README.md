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
