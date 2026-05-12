"""
Cloud Cost / Performance Analysis
===================================
Queries the SQLite database for rows from the local vol cloud profile
experiment (experiment_type = 'localvol_cloud_profile') and prints:

  1. Cost-per-million-paths table by (instance_type, engine, ad_mode, M)
  2. Throughput table (paths/sec)
  3. Cost-efficiency ranking (throughput per dollar-per-hour)
  4. Best-instance recommendation per (engine, M)

Usage
-----
  # Analyse all localvol_cloud_profile rows
  python experiments/run_cloud_cost_analysis.py

  # Filter to a single experiment run
  python experiments/run_cloud_cost_analysis.py --experiment-id <uuid>

  # Compare specific instance types
  python experiments/run_cloud_cost_analysis.py \\
      --instance-types n2-standard-8 t2d-standard-8
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmarking.storage.database import _DB_PATH


def _fmt(v: Optional[float], fmt: str = ".4f") -> str:
    return f"{v:{fmt}}" if v is not None else "    n/a "


def _query(
    db_path: Path,
    experiment_id: Optional[str],
    instance_types: Optional[list[str]],
) -> list[dict]:
    """Return rows from the localvol_cloud_profile experiment."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    conditions = ["experiment_type = 'localvol_cloud_profile'",
                  "status = 'completed'"]
    params: list = []

    if experiment_id:
        conditions.append("experiment_id = ?")
        params.append(experiment_id)
    if instance_types:
        placeholders = ",".join("?" * len(instance_types))
        conditions.append(f"instance_type IN ({placeholders})")
        params.extend(instance_types)

    sql = f"""
        SELECT
            instance_type,
            engine,
            ad_mode,
            M,
            mean_runtime_ms,
            throughput_paths_per_sec,
            cost_per_run,
            result_value,
            memory_peak_mb,
            ad_overhead_ratio,
            cpu_count,
            cpu_model
        FROM runs
        WHERE {' AND '.join(conditions)}
        ORDER BY instance_type, engine, ad_mode, M
    """
    cur.execute(sql, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def _cost_per_million(cost_per_run: Optional[float], M: int) -> Optional[float]:
    if cost_per_run is None or M == 0:
        return None
    return cost_per_run / M * 1_000_000


def _print_throughput_table(rows: list[dict]) -> None:
    print()
    print("  " + "=" * 95)
    print("  Throughput  (paths / second)")
    print("  " + "=" * 95)
    hdr = (
        f"  {'Instance':<20}  {'Engine':<6}  {'AD':>8}  {'M':>7}  "
        f"{'Paths/s':>13}  {'Mean ms':>9}  {'Overhead':>8}  {'Mem MB':>7}"
    )
    print(hdr)
    print("  " + "-" * 95)
    for r in rows:
        oh = f"{r['ad_overhead_ratio']:.2f}x" if r["ad_mode"] != "none" else "  1.00x"
        print(
            f"  {(r['instance_type'] or 'local'):<20}  {r['engine']:<6}  "
            f"{r['ad_mode']:>8}  {r['M']:>7,}  "
            f"{_fmt(r['throughput_paths_per_sec'], ',.0f'):>13}  "
            f"{_fmt(r['mean_runtime_ms'], '.3f'):>9}  "
            f"{oh:>8}  "
            f"{_fmt(r['memory_peak_mb'], '.1f'):>7}"
        )


def _print_cost_table(rows: list[dict]) -> None:
    print()
    print("  " + "=" * 95)
    print("  Cost per million paths  (USD)")
    print("  " + "=" * 95)
    hdr = (
        f"  {'Instance':<20}  {'Engine':<6}  {'AD':>8}  {'M':>7}  "
        f"{'$/M paths':>11}  {'$/run':>12}  {'Paths/s':>13}"
    )
    print(hdr)
    print("  " + "-" * 95)
    for r in rows:
        cpm = _cost_per_million(r["cost_per_run"], r["M"])
        cpm_s  = f"${cpm:.4f}"  if cpm  is not None else "       n/a"
        cost_s = f"${r['cost_per_run']:.2e}" if r["cost_per_run"] is not None else "          n/a"
        print(
            f"  {(r['instance_type'] or 'local'):<20}  {r['engine']:<6}  "
            f"{r['ad_mode']:>8}  {r['M']:>7,}  "
            f"{cpm_s:>11}  {cost_s:>12}  "
            f"{_fmt(r['throughput_paths_per_sec'], ',.0f'):>13}"
        )


def _print_efficiency_ranking(rows: list[dict]) -> None:
    """Throughput / hourly_rate = paths per dollar — higher is better."""
    rows_with_cost = [r for r in rows if r["cost_per_run"] is not None and r["M"] > 0]
    if not rows_with_cost:
        print("\n  [no cost data — run on a GCP VM or pass --hourly-rate to capture costs]")
        return

    print()
    print("  " + "=" * 95)
    print("  Cost-efficiency ranking  (paths per dollar)")
    print("  Higher = cheaper per unit of compute")
    print("  " + "=" * 95)

    # Compute paths per dollar = throughput / (cost_per_run / mean_runtime_s * 3600)
    # Equivalently: throughput * 3600 / (cost_per_run * 1000 / mean_runtime_ms)
    # Simplest: paths_per_dollar = M / cost_per_run  (cost covers exactly one run of M paths)
    augmented = []
    for r in rows_with_cost:
        ppd = r["M"] / r["cost_per_run"] if r["cost_per_run"] > 0 else None
        augmented.append({**r, "paths_per_dollar": ppd})

    augmented.sort(key=lambda x: -(x["paths_per_dollar"] or 0))

    hdr = (
        f"  {'Instance':<20}  {'Engine':<6}  {'AD':>8}  {'M':>7}  "
        f"{'Paths/$':>14}  {'$/M paths':>11}"
    )
    print(hdr)
    print("  " + "-" * 95)
    for r in augmented:
        ppd = r["paths_per_dollar"]
        cpm = _cost_per_million(r["cost_per_run"], r["M"])
        ppd_s = f"{ppd:,.0f}" if ppd is not None else "       n/a"
        cpm_s = f"${cpm:.4f}" if cpm is not None else "       n/a"
        print(
            f"  {(r['instance_type'] or 'local'):<20}  {r['engine']:<6}  "
            f"{r['ad_mode']:>8}  {r['M']:>7,}  "
            f"{ppd_s:>14}  {cpm_s:>11}"
        )


def _print_recommendations(rows: list[dict]) -> None:
    """Per (engine, M): which instance type has lowest cost per million paths?"""
    rows_with_cost = [r for r in rows if r["cost_per_run"] is not None and r["M"] > 0]
    if not rows_with_cost:
        return

    # Group by (engine, ad_mode, M)
    groups: dict[tuple, list[dict]] = {}
    for r in rows_with_cost:
        key = (r["engine"], r["ad_mode"], r["M"])
        groups.setdefault(key, []).append(r)

    print()
    print("  " + "=" * 95)
    print("  Best instance per (engine, AD mode, M)  — lowest cost-per-million-paths")
    print("  " + "=" * 95)
    hdr = (
        f"  {'Engine':<6}  {'AD':>8}  {'M':>7}  "
        f"{'Best instance':<20}  {'$/M paths':>11}  {'Paths/s':>13}  "
        f"{'vs. runner-up':>14}"
    )
    print(hdr)
    print("  " + "-" * 95)

    for (engine, ad_mode, M), group in sorted(groups.items()):
        ranked = sorted(group, key=lambda x: _cost_per_million(x["cost_per_run"], x["M"]) or float("inf"))
        best   = ranked[0]
        cpm    = _cost_per_million(best["cost_per_run"], best["M"])
        cpm_s  = f"${cpm:.4f}" if cpm is not None else "       n/a"

        vs_s = ""
        if len(ranked) > 1:
            second     = ranked[1]
            second_cpm = _cost_per_million(second["cost_per_run"], second["M"])
            if cpm and second_cpm and cpm > 0:
                ratio = second_cpm / cpm
                vs_s  = f"{ratio:.2f}x more expensive"

        print(
            f"  {engine:<6}  {ad_mode:>8}  {M:>7,}  "
            f"{(best['instance_type'] or 'local'):<20}  {cpm_s:>11}  "
            f"{_fmt(best['throughput_paths_per_sec'], ',.0f'):>13}  "
            f"{vs_s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Cloud cost/performance analysis")
    parser.add_argument("--experiment-id", default=None,
                        help="Filter to a single experiment UUID")
    parser.add_argument("--instance-types", nargs="+", default=None,
                        metavar="TYPE",
                        help="Filter to specific instance types")
    parser.add_argument("--db-path", default=None,
                        help=f"Path to SQLite DB (default: {_DB_PATH})")
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else _DB_PATH
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("Run run_localvol_cloud.py first to generate data.")
        sys.exit(1)

    rows = _query(db_path, args.experiment_id, args.instance_types)

    if not rows:
        print("\n  No rows found for experiment_type='localvol_cloud_profile'.")
        print("  Run:  python experiments/run_localvol_cloud.py  to generate data.")
        sys.exit(0)

    instance_types = sorted({r["instance_type"] or "local" for r in rows})
    engines        = sorted({r["engine"] for r in rows})
    m_values       = sorted({r["M"] for r in rows})

    print()
    print("=" * 80)
    print("  Cloud Cost / Performance Analysis")
    print("=" * 80)
    print(f"  Rows          : {len(rows)}")
    print(f"  Instance types: {instance_types}")
    print(f"  Engines       : {engines}")
    print(f"  M values      : {m_values}")
    print(f"  DB path       : {db_path}")

    _print_throughput_table(rows)
    _print_cost_table(rows)
    _print_efficiency_ranking(rows)
    _print_recommendations(rows)

    print()


if __name__ == "__main__":
    main()
