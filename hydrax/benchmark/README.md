# Hydrax Controller Benchmark

Simple utility to benchmark Hydrax control algorithms on a specific task.

## Usage

### Basic Usage

Run the benchmark on a specific task:

```bash
python hydrax/benchmark/run_benchmark.py --task Pendulum
```

### Customize Episodes and Steps

```bash
python hydrax/benchmark/run_benchmark.py --task CartPole --episodes 5 --steps 1000
```

### Options

- `--task`: Specify a task to benchmark (REQUIRED)
- `--episodes`: Number of episodes (default: 3) 
- `--steps`: Steps per episode (default: 500)

Results are saved to `hydrax/benchmark/results/`