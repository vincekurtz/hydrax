#!/usr/bin/env python3
"""Main entry point for benchmarking Hydrax controllers on a specific task."""

import argparse
import time
from pathlib import Path

from hydrax import ROOT
from runner import run_task_benchmark, get_all_tasks
from plotting import plot_results, plot_cost_over_time


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
        "--steps",
        type=int,
        default=500,
        help="Number of steps to run each controller",
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
    results = run_task_benchmark(args.task, total_steps=args.steps)

    # Plot the results
    plot_results(results, args.task)
    plot_cost_over_time(results, args.task)

    end_time = time.time()
    print(
        f"Benchmark complete in {end_time - start_time:.2f} seconds! Results saved to {results_dir}"
    )

    # Print summary table without pandas
    print("\nPerformance Summary:")
    print(
        f"{'Controller':<20} {'Avg Cost':<15} {'Final Cost':<15} {'Avg Plan Time (s)':<20} "
    )
    print("-" * 75)

    # Sort by average cost
    sorted_results = sorted(
        results, key=lambda x: x.get("avg_cost", float("inf"))
    )

    for result in sorted_results:
        if "controller" in result and "avg_cost" in result:
            controller = result["controller"]
            avg_cost = result["avg_cost"]
            final_cost = result["costs"][-1]
            avg_plan_time = result["avg_plan_time"]
            print(
                f"{controller:<20} {avg_cost:<15.6f} {final_cost:<15.6f} {avg_plan_time:<20.6f} "
            )


if __name__ == "__main__":
    main()
