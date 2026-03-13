#!/usr/bin/env python
"""
Analyze hyperparameter search results and rank by Background Rejection at 50% signal efficiency
"""

import argparse
import re
from pathlib import Path
import numpy as np


def parse_results_file(file_path):
    """Parse the hyperparameter search results summary file"""
    if not file_path.exists():
        print(f"Error: Results file not found at {file_path}")
        return []

    results = []
    current_run = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith("Run: "):
                if current_run is not None:
                    results.append(current_run)
                current_run = {"name": line.split("Run: ")[1]}

            elif "Temperature:" in line and current_run:
                current_run["temperature"] = line.split("Temperature: ")[1]

            elif "Alpha_KD:" in line and current_run:
                current_run["alpha"] = line.split("Alpha_KD: ")[1]

            elif "Background Rejection @ 50% efficiency:" in line and current_run:
                br_str = line.split("efficiency: ")[1]
                current_run["br"] = float(br_str)

            elif "AUC (Student):" in line and current_run:
                auc_str = line.split("AUC (Student): ")[1]
                current_run["auc"] = float(auc_str)

            elif "Saved to:" in line and current_run:
                current_run["path"] = line.split("Saved to: ")[1]

    # Add last run
    if current_run is not None:
        results.append(current_run)

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze hyperparameter search results")
    parser.add_argument(
        "--results_file",
        type=str,
        default="checkpoints/transformer_search/hyperparameter_search_results.txt",
        help="Path to hyperparameter search results file"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Number of top results to display"
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        default="br",
        choices=["br", "auc"],
        help="Sort by: br (Background Rejection) or auc"
    )
    args = parser.parse_args()

    results_file = Path(args.results_file)
    results = parse_results_file(results_file)

    if not results:
        print("No results found!")
        return

    # Sort results
    results_sorted = sorted(results, key=lambda x: x.get(args.sort_by, 0), reverse=True)

    print(f"\n{'='*100}")
    print(f"HYPERPARAMETER SEARCH RESULTS - Sorted by {args.sort_by.upper()}")
    print(f"{'='*100}\n")
    print(f"Total runs: {len(results_sorted)}")
    print(f"\nTop {args.top_n} configurations:\n")

    # Print header
    print(f"{'Rank':<6} {'Run Name':<45} {'BR@50%':<10} {'AUC':<10} {'Temperature':<25} {'Alpha':<25}")
    print("-" * 120)

    # Print top N results
    for i, result in enumerate(results_sorted[:args.top_n], 1):
        name = result.get("name", "Unknown")
        br = result.get("br", 0.0)
        auc = result.get("auc", 0.0)
        temp = result.get("temperature", "N/A")
        alpha = result.get("alpha", "N/A")

        print(f"{i:<6} {name:<45} {br:<10.2f} {auc:<10.4f} {temp:<25} {alpha:<25}")

    # Print statistics
    brs = [r.get("br", 0) for r in results_sorted]
    aucs = [r.get("auc", 0) for r in results_sorted]

    print(f"\n{'='*100}")
    print("STATISTICS")
    print(f"{'='*100}")
    print(f"\nBackground Rejection @ 50% efficiency:")
    print(f"  Best:   {max(brs):.2f}")
    print(f"  Worst:  {min(brs):.2f}")
    print(f"  Mean:   {np.mean(brs):.2f}")
    print(f"  Median: {np.median(brs):.2f}")
    print(f"  Std:    {np.std(brs):.2f}")

    print(f"\nAUC:")
    print(f"  Best:   {max(aucs):.4f}")
    print(f"  Worst:  {min(aucs):.4f}")
    print(f"  Mean:   {np.mean(aucs):.4f}")
    print(f"  Median: {np.median(aucs):.4f}")
    print(f"  Std:    {np.std(aucs):.4f}")

    # Print best configuration details
    best = results_sorted[0]
    print(f"\n{'='*100}")
    print("BEST CONFIGURATION")
    print(f"{'='*100}")
    print(f"  Run name: {best.get('name', 'Unknown')}")
    print(f"  Background Rejection @ 50%: {best.get('br', 0):.2f}")
    print(f"  AUC: {best.get('auc', 0):.4f}")
    print(f"  Temperature: {best.get('temperature', 'N/A')}")
    print(f"  Alpha: {best.get('alpha', 'N/A')}")
    print(f"  Results saved to: {best.get('path', 'N/A')}")
    print()


if __name__ == "__main__":
    main()
