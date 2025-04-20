# Hydrax Controller Benchmark

Simple utility to benchmark Hydrax control algorithms across various tasks.

## Usage

### Basic Usage

Run the benchmark on all tasks with all controllers:

```bash
python hydrax/benchmark/run_benchmark.py
```

### Benchmark a Specific Task

```bash
python hydrax/benchmark/run_benchmark.py --task Pendulum
```

### Customize Episodes and Steps

```bash
python hydrax/benchmark/run_benchmark.py --task CartPole --episodes 5 --steps 1000
```

### Options

- `--task`: Specify a task to benchmark
- `--episodes`: Number of episodes (default: 1) 
- `--steps`: Steps per episode (default: 500)

Results are saved to `hydrax/benchmark/results/`