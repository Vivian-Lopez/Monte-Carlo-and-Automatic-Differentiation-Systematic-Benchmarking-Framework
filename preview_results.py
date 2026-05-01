#!/usr/bin/env python3
"""
Preview benchmark results from SQLite database.
Displays results grouped by experiment type with clean formatting.

Usage:
    python preview_results.py              # show all experiments
    python preview_results.py european_baseline
    python preview_results.py --summary    # only show summary stats
"""

import sys
import sqlite3
from pathlib import Path
from collections import defaultdict

DB_PATH = Path(__file__).parent / "results" / "benchmarks.db"


def connect_db():
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        print("Run an experiment first: python experiments/run_european_baseline.py")
        sys.exit(1)
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    return con


def print_header(text):
    print(f"\n{'=' * 80}")
    print(f"  {text}")
    print(f"{'=' * 80}\n")


def fmt(v, spec=".4f"):
    return f"{v:{spec}}" if v is not None else "n/a"


def show_experiment_type(con, exp_type):
    rows = con.execute("""
        SELECT engine, ad_mode, M,
               mean_runtime_ms, std_runtime_ms,
               result_value, analytical_price, rel_price_error,
               greek_delta, greek_vega, greek_rho,
               analytical_delta, analytical_vega, analytical_rho,
               throughput_paths_per_sec, memory_peak_mb,
               ad_overhead_ratio, baseline_mean_ms
        FROM runs
        WHERE experiment_type = ?
        ORDER BY M, engine, ad_mode
    """, (exp_type,)).fetchall()

    if not rows:
        print(f"  No runs found for experiment_type='{exp_type}'")
        return

    print_header(f"Experiment: {exp_type}")

    by_M = defaultdict(list)
    for r in rows:
        by_M[r["M"]].append(r)

    for M in sorted(by_M.keys()):
        group = by_M[M]
        print(f"  M = {M:,} paths")
        print(f"  {'─' * 76}")
        print(f"  {'Engine':<8} {'AD':<9} {'ms (mean±std)':<16} {'Price':<12} "
              f"{'Err%':<8} {'Paths/s':<14} {'Mem MB':<8}")
        print(f"  {'─' * 76}")

        for r in group:
            ms_str = f"{r['mean_runtime_ms']:.3f} ± {r['std_runtime_ms']:.3f}"
            err_str = f"{r['rel_price_error']*100:.3f}%" if r['rel_price_error'] is not None else "n/a"
            paths_s = f"{r['throughput_paths_per_sec']:>12,.0f}" if r['throughput_paths_per_sec'] else "n/a"
            mem = fmt(r['memory_peak_mb'], ".1f")

            print(f"  {r['engine']:<8} {r['ad_mode']:<9} {ms_str:<16} "
                  f"{r['result_value']:<12.6f} {err_str:<8} {paths_s:<14} {mem}")

            if r['ad_overhead_ratio'] is not None and r['ad_mode'] != 'none':
                print(f"    └─ AD overhead: {r['ad_overhead_ratio']:.2f}×  "
                      f"(baseline no-AD: {fmt(r['baseline_mean_ms'], '.3f')} ms)")

        # Greeks block for AD runs
        ad_runs = [r for r in group if r['ad_mode'] != 'none' and r['greek_delta'] is not None]
        if ad_runs:
            print(f"\n  Greeks at M={M:,}")
            print(f"  {'─' * 76}")
            print(f"  {'AD mode':<10} {'Delta':>12} {'Vega':>12} {'Rho':>12}")
            print(f"  {'─' * 76}")
            for r in ad_runs:
                print(f"  {r['ad_mode']:<10} {r['greek_delta']:>12.6f} "
                      f"{r['greek_vega']:>12.5f} {r['greek_rho']:>12.5f}")
            # Analytical row
            sample = ad_runs[0]
            if sample['analytical_delta'] is not None:
                print(f"  {'analytical':<10} {sample['analytical_delta']:>12.6f} "
                      f"{sample['analytical_vega']:>12.5f} {sample['analytical_rho']:>12.5f}")
                # Error row for first AD run
                for r in ad_runs:
                    d_err = abs(r['greek_delta'] - sample['analytical_delta'])
                    v_err = abs(r['greek_vega']  - sample['analytical_vega'])
                    rho_err = abs(r['greek_rho'] - sample['analytical_rho'])
                    print(f"  abs err ({r['ad_mode']:<8}) {d_err:>12.2e} "
                          f"{v_err:>12.2e} {rho_err:>12.2e}")
        print()


def show_summary(con):
    print_header("Summary across all runs")

    rows = con.execute("""
        SELECT experiment_type,
               COUNT(*)                        AS n_runs,
               COUNT(DISTINCT engine)          AS n_engines,
               MIN(mean_runtime_ms)            AS fastest_ms,
               MAX(mean_runtime_ms)            AS slowest_ms,
               MIN(rel_price_error)            AS best_err,
               MAX(rel_price_error)            AS worst_err,
               AVG(rel_price_error)            AS avg_err
        FROM runs
        WHERE rel_price_error IS NOT NULL
        GROUP BY experiment_type
        ORDER BY experiment_type
    """).fetchall()

    print(f"  {'Experiment':<30} {'Runs':<6} {'Eng':<5} "
          f"{'Fastest ms':<12} {'Slowest ms':<12} {'Best err%':<11} {'Worst err%':<11}")
    print(f"  {'─' * 76}")
    for r in rows:
        print(f"  {r['experiment_type']:<30} {r['n_runs']:<6} {r['n_engines']:<5} "
              f"{r['fastest_ms']:<12.3f} {r['slowest_ms']:<12.3f} "
              f"{r['best_err']*100:<11.4f} {r['worst_err']*100:<11.4f}")

    total = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    exp_types = con.execute(
        "SELECT DISTINCT experiment_type FROM runs ORDER BY experiment_type"
    ).fetchall()
    print(f"\n  Total stored runs : {total}")
    print(f"  Experiment types  : {', '.join(r[0] for r in exp_types)}\n")


def main():
    con = connect_db()
    args = sys.argv[1:]
    summary_only = "--summary" in args
    exp_filter = next((a for a in args if not a.startswith("--")), None)

    show_summary(con)

    if not summary_only:
        if exp_filter:
            show_experiment_type(con, exp_filter)
        else:
            exp_types = con.execute(
                "SELECT DISTINCT experiment_type FROM runs ORDER BY experiment_type"
            ).fetchall()
            for (exp_type,) in exp_types:
                show_experiment_type(con, exp_type)

    con.close()


if __name__ == "__main__":
    main()
