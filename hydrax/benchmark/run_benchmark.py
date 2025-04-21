#!/usr/bin/env python3
"""Main entry point for benchmarking Hydrax controllers on a specific task."""

import argparse
import time
from pathlib import Path

from hydrax import ROOT
from runner import run_task_benchmark
from plotting import plot_results, plot_cost_over_time
from runner import get_all_tasks


def main():
    """Run the benchmark script."""
    # Set up output directory
    results_dir = Path(ROOT) / "benchmark" / "results"
    results_dir.mkdir(exist_ok=True)

    # Get available tasks for help message
    all_tasks = get_all_tasks()
    available_tasks = ", ".join(all_tasks.keys())

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Benchmark controllers on a specific task"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help=f"Task to benchmark. Available tasks: {available_tasks}",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes per benchmark",
    )
    parser.add_argument(
        "--steps", type=int, default=500, help="Number of steps per episode"
    )
    args = parser.parse_args()

    start_time = time.time()

    # Validate task
    if args.task not in all_tasks:
        print(
            f"Error: Task '{args.task}' not found. Available tasks: {available_tasks}"
        )
        return

    # Run benchmark for the specified task
    results_df, all_results = run_task_benchmark(
        args.task, num_episodes=args.episodes, total_steps=args.steps
    )

    # Plot the results
    plot_results(results_df, args.task)
    plot_cost_over_time(all_results, args.task)

    end_time = time.time()
    print(
        f"Benchmark complete in {end_time - start_time:.2f} seconds! Results saved to {results_dir}"
    )

    # Print summary table
    print("\nPerformance Summary:")
    summary = results_df[["controller", "avg_cost", "avg_plan_time"]]
    summary = summary.sort_values("avg_cost")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
