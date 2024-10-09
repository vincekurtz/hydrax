# hydrax

Sampling-based model predictive control on GPU with JAX and
[MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html)

## Setup (Conda)

Set up a conda env with Cuda support (first time only):

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

Set up pre-commit hooks:

```bash
pre-commit autoupdate
pre-commit install
```

## Usage

Run unit tests:

```bash
pytest
```

Other demos can be found in the `examples` folder.
