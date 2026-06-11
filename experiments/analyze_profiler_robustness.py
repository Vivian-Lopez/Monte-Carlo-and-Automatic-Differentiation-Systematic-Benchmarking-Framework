#!/usr/bin/env python3
"""
Post-hoc robustness audit for the SHA cloud profiler.

The headline report table focuses on the n2-standard-4 slice of
sha_cloud_profiler_v4. This script reuses the same metric definitions across
all instance-type slices stored in results/profiler_vs_grid.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


MAX_M = 100_000
FULL_TYPES = {"profiler_selected", "grid_search_full", "sha_selected"}
OLD_DECISIONS = {"selected_old", "selected_both"}
SHA_DECISIONS = {"selected_sha", "selected_both"}


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def pareto_frontier(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    frontier: list[dict[str, str]] = []
    for row in rows:
        runtime = as_float(row["mean_runtime_ms"])
        cost = as_float(row["cost_per_run"])
        dominated = False
        for other in rows:
            if other is row:
                continue
            other_runtime = as_float(other["mean_runtime_ms"])
            other_cost = as_float(other["cost_per_run"])
            no_worse = other_runtime <= runtime and other_cost <= cost
            strictly_better = other_runtime < runtime or other_cost < cost
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return frontier


def config_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["workload_type"], row["engine"], row["ad_mode"]


def selected_keys(rows: Iterable[dict[str, str]], decisions: set[str]) -> set[tuple[str, str, str]]:
    return {
        config_key(row)
        for row in rows
        if row["profiler_decision"] in decisions and int(row["M"]) == MAX_M
    }


def analyse_instance(rows: list[dict[str, str]], instance: str) -> dict[str, object]:
    instance_rows = [
        row
        for row in rows
        if row["instance_type"] == instance
        and row["experiment_type"] in FULL_TYPES
        and int(row["M"]) in {10_000, 50_000, 100_000}
    ]
    max_rows = [row for row in instance_rows if int(row["M"]) == MAX_M]
    old_rows = [row for row in instance_rows if row["profiler_decision"] in OLD_DECISIONS]
    sha_rows = [row for row in instance_rows if row["profiler_decision"] in SHA_DECISIONS]

    pareto_keys: set[tuple[str, str, str]] = set()
    workloads = sorted({row["workload_type"] for row in max_rows})
    for workload in workloads:
        workload_rows = [row for row in max_rows if row["workload_type"] == workload]
        pareto_keys.update(config_key(row) for row in pareto_frontier(workload_rows))

    old_keys = selected_keys(old_rows, OLD_DECISIONS)
    sha_keys = selected_keys(sha_rows, SHA_DECISIONS)
    best_runtime = min(max_rows, key=lambda row: as_float(row["mean_runtime_ms"]))
    best_cost = min(max_rows, key=lambda row: as_float(row["cost_per_run"]))
    sha_max_rows = [row for row in sha_rows if int(row["M"]) == MAX_M]
    old_max_rows = [row for row in old_rows if int(row["M"]) == MAX_M]
    sha_best = min(sha_max_rows, key=lambda row: as_float(row["mean_runtime_ms"]))
    old_best = min(old_max_rows, key=lambda row: as_float(row["mean_runtime_ms"]))

    full_runs = len(instance_rows)
    sha_runs = len(sha_rows)
    old_runs = len(old_rows)
    return {
        "instance_type": instance,
        "candidate_configurations": len({config_key(row) for row in max_rows}),
        "full_runs": full_runs,
        "old_runs": old_runs,
        "sha_runs": sha_runs,
        "sha_saved_pct": 100.0 * (1.0 - sha_runs / full_runs),
        "old_pareto_recovery_pct": 100.0 * len(pareto_keys & old_keys) / len(pareto_keys),
        "sha_pareto_recovery_pct": 100.0 * len(pareto_keys & sha_keys) / len(pareto_keys),
        "sha_runtime_regret_pct": 100.0
        * (as_float(sha_best["mean_runtime_ms"]) - as_float(best_runtime["mean_runtime_ms"]))
        / as_float(best_runtime["mean_runtime_ms"]),
        "sha_cost_regret_pct": 100.0
        * (as_float(sha_best["cost_per_run"]) - as_float(best_cost["cost_per_run"]))
        / as_float(best_cost["cost_per_run"]),
        "best_config": "/".join(config_key(best_runtime)),
        "sha_best_config": "/".join(config_key(sha_best)),
        "old_best_config": "/".join(config_key(old_best)),
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("\\begin{table}[htbp]\n")
        handle.write("\\centering\n\\small\n")
        handle.write("\\caption{Cross-instance robustness audit for SHA on sha\\_cloud\\_profiler\\_v4.}\n")
        handle.write("\\label{tab:profiler-cross-instance}\n")
        handle.write("\\begin{tabularx}{\\textwidth}{lrrrrr}\n")
        handle.write("\\toprule\n")
        handle.write(
            "\\textbf{Instance} & \\textbf{SHA runs} & \\textbf{Saved} & "
            "\\textbf{SHA Pareto} & \\textbf{SHA regret} & \\textbf{Old Pareto} \\\\\n"
        )
        handle.write("\\midrule\n")
        for row in rows:
            handle.write(
                f"{row['instance_type']} & "
                f"{int(row['sha_runs'])} & "
                f"{float(row['sha_saved_pct']):.1f}\\% & "
                f"{float(row['sha_pareto_recovery_pct']):.1f}\\% & "
                f"+{float(row['sha_runtime_regret_pct']):.1f}\\% & "
                f"{float(row['old_pareto_recovery_pct']):.1f}\\% \\\\\n"
            )
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabularx}\n")
        handle.write("\\end{table}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/profiler_vs_grid.csv")
    parser.add_argument("--experiment-id", default="sha_cloud_profiler_v4")
    parser.add_argument("--csv-out", default="results/profiler_cross_instance.csv")
    parser.add_argument(
        "--tex-out",
        default="Final_Year_Project_Report/evaluation/tables/profiler_cross_instance.tex",
    )
    args = parser.parse_args()

    with Path(args.input).open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["experiment_id"] == args.experiment_id and row["instance_type"]
        ]

    instances = sorted({row["instance_type"] for row in rows})
    summary = [analyse_instance(rows, instance) for instance in instances]
    write_csv(summary, Path(args.csv_out))
    write_latex(summary, Path(args.tex_out))

    perfect = sum(1 for row in summary if float(row["sha_pareto_recovery_pct"]) == 100.0)
    print(f"Analysed {len(summary)} instance grids; SHA perfect recovery on {perfect}/{len(summary)}.")
    print(f"Wrote {args.csv_out}")
    print(f"Wrote {args.tex_out}")


if __name__ == "__main__":
    main()
