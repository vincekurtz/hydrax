# Hydrax Controller Benchmark

Simple utility to benchmark Hydrax control algorithms on a specific task.

## Usage

Run the benchmark on a specific task:

```bash
python hydrax/benchmark/run_benchmark.py --task Pendulum
```

### Options

- `--task`: Specify a task to benchmark (REQUIRED)
- `--steps`: Steps per episode (default: 500)

Results are saved to `hydrax/benchmark/results/`