#!/usr/bin/env python3
"""Main entry point for benchmarking Hydrax controllers."""

import argparse
import time
from pathlib import Path

from hydrax import ROOT
from runner import run_full_benchmark, run_single_task_benchmark
from plotting import plot_results, plot_cost_over_time


def main():
    """Run the benchmark script."""
    # Set up output directory
    results_dir = Path(ROOT) / "benchmark" / "results"
    results_dir.mkdir(exist_ok=True)

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Benchmark controllers on tasks"
    )
    parser.add_argument(
        "--task", type=str, help="Run benchmark for a specific task only"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes per benchmark",
    )
    parser.add_argument(
        "--steps", type=int, default=500, help="Number of steps per episode"
    )
    args = parser.parse_args()

    start_time = time.time()

    # Run benchmark for a single task or all tasks
    if args.task:
        results_df, all_results = run_single_task_benchmark(
            args.task, num_episodes=args.episodes, total_steps=args.steps
        )
        print(f"Single task benchmark for {args.task} complete!")
    else:
        # Run the full benchmark
        results_df, all_results = run_full_benchmark(
            num_episodes=args.episodes, total_steps=args.steps
        )

    # Plot the results
    plot_results(results_df)
    plot_cost_over_time(all_results)

    end_time = time.time()
    print(
        f"Benchmark complete in {end_time - start_time:.2f} seconds! Results saved to {results_dir}"
    )


if __name__ == "__main__":
    main()
