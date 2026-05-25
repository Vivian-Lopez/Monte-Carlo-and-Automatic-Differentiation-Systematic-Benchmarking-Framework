"""
Budget-Aware Profiler vs Naive Grid-Search Experiment
======================================================
Runs cheap probe benchmarks to select promising (workload, engine, ad_mode)
configurations, then compares those selections against a full naive grid-search
using cost/runtime Pareto frontiers.

Experiment tags in the DB
--------------------------
  probe_run          - cheap small-M probe runs
  grid_search_full   - full naive grid across all M values
  profiler_selected  - full runs for profiler-selected configs only

Usage
-----
  # Probe only (fast smoke test)
  python experiments/run_profiler_vs_grid.py \\
      --workloads european asian --engines cpu jax \\
      --probe-only --m-probe 1000 --runs-probe 2 --warmup-probe 1

  # Full local experiment
  python experiments/run_profiler_vs_grid.py \\
      --workloads european european_local_vol asian \\
      --engines cpu jax \\
      --m-values 10000 50000 100000 \\
      --runs 5 --warmup 2 \\
      --m-probe 1000 --runs-probe 2 --warmup-probe 1 \\
      --top-k 3
"""

from __future__ import annotations

import argparse
import math
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmarking.core.config import (
    EuropeanOptionConfig, EuropeanLocalVolConfig, AsianOptionConfig,
    WorkloadConfig,
)
from benchmarking.runner.runner import BenchmarkRunner
from benchmarking.storage.database import BenchmarkDB
from benchmarking.analysis.pareto import compute_pareto_frontier
from benchmarking.cloud.metadata import get_instance_metadata
from benchmarking.cloud.pricing import get_hourly_rate, compute_cost_per_run

# ---------------------------------------------------------------------------
# Engine loading — compiled engines silently skipped if absent
# ---------------------------------------------------------------------------
try:
    from benchmarking.workloads.mc_cpp import CPPMonteCarloEngine
    _CPP = CPPMonteCarloEngine()
except Exception:
    _CPP = None

try:
    from benchmarking.workloads.mc_rust import RustMonteCarloEngine
    _RUST = RustMonteCarloEngine()
except Exception:
    _RUST = None

from benchmarking.workloads.mc_cpu import CPUMonteCarloEngine
from benchmarking.workloads.mc_jax import JAXMonteCarloEngine

_ENGINES = {
    "cpu":  CPUMonteCarloEngine(),
    "jax":  JAXMonteCarloEngine(),
    "cpp":  _CPP,
    "rust": _RUST,
}

# AD modes supported per engine key
_ENGINE_AD_MODES = {
    "cpu":  ["none"],
    "jax":  ["none", "forward", "reverse"],
    "cpp":  ["none"],
    "rust": ["none"],
}

# Which workloads each engine supports (compiled engines do not support asian)
_ENGINE_WORKLOADS = {
    "cpu":  {"european", "european_local_vol", "asian"},
    "jax":  {"european", "european_local_vol", "asian"},
    "cpp":  {"european", "european_local_vol"},
    "rust": {"european", "european_local_vol"},
}

# Language/backend metadata for DB storage
_ENGINE_META = {
    "cpu":  ("python", "numpy"),
    "jax":  ("python", "xla"),
    "cpp":  ("cpp",    "openmp"),
    "rust": ("rust",   "rayon"),
}


# ---------------------------------------------------------------------------
# Config factories
# ---------------------------------------------------------------------------

def make_config(workload: str, M: int, seed: int = 42) -> WorkloadConfig:
    if workload == "european":
        return EuropeanOptionConfig(M=M, seed=seed)
    elif workload == "european_local_vol":
        return EuropeanLocalVolConfig(M=M, seed=seed)
    elif workload == "asian":
        return AsianOptionConfig(M=M, seed=seed)
    else:
        raise ValueError(f"Unknown workload: {workload!r}")


# ---------------------------------------------------------------------------
# Single benchmark run helper
# ---------------------------------------------------------------------------

def run_one(
    engine_key: str,
    workload: str,
    M: int,
    ad_mode: str,
    num_warmup: int,
    num_runs: int,
    seed: int = 42,
) -> Optional[Dict[str, Any]]:
    """
    Run a single benchmark configuration. Returns a result dict or None on error.
    """
    engine_obj = _ENGINES.get(engine_key)
    if engine_obj is None:
        return None
    if workload not in _ENGINE_WORKLOADS.get(engine_key, set()):
        return None
    if ad_mode not in _ENGINE_AD_MODES.get(engine_key, []):
        return None

    config = make_config(workload, M, seed=seed)
    runner = BenchmarkRunner(engine_obj, name=f"{engine_key}_{workload}")
    try:
        res = runner.run(config, num_warmup=num_warmup, num_runs=num_runs, ad_mode=ad_mode)
    except Exception as exc:
        print(f"  [SKIP] {engine_key}/{workload}/{ad_mode}/M={M}: {exc}")
        return None

    return {
        "engine":           engine_key,
        "workload":         workload,
        "ad_mode":          ad_mode,
        "M":                M,
        "result":           res.result,
        "mean_runtime_ms":  res.mean_runtime * 1000.0,
        "std_runtime_ms":   res.std_runtime * 1000.0,
        "min_runtime_ms":   res.min_runtime * 1000.0,
        "max_runtime_ms":   res.max_runtime * 1000.0,
        "throughput":       res.throughput_paths_per_sec,
        "ad_overhead_ratio": res.ad_overhead_ratio,
        "memory_peak_mb":   res.memory_peak_mb,
        "greeks":           res.greeks,
        "config":           config,
        "metadata":         res.metadata,
        "cost_per_run":     0.0,  # filled in below if pricing available
    }


# ---------------------------------------------------------------------------
# Probe scoring heuristic
# ---------------------------------------------------------------------------

def _probe_score(row: Dict[str, Any]) -> float:
    """
    Lower score = more promising.
    score = mean_runtime_ms * (1 + stability_penalty) * (1 + error_penalty)
    stability_penalty = std/mean (coefficient of variation), capped at 2.
    error_penalty = 1 if result is non-positive or NaN, else 0.
    """
    rt = row["mean_runtime_ms"]
    if rt <= 0 or not math.isfinite(rt):
        return float("inf")
    std = row["std_runtime_ms"]
    stability = min(std / rt, 2.0) if rt > 0 else 2.0
    result_val = row["result"]
    error_pen = 0.0
    if result_val is None or not math.isfinite(result_val) or result_val < 0:
        error_pen = 1.0
    return rt * (1.0 + stability) * (1.0 + error_pen)


# ---------------------------------------------------------------------------
# DB save helper
# ---------------------------------------------------------------------------

def save_to_db(
    db: BenchmarkDB,
    row: Dict[str, Any],
    experiment_id: str,
    experiment_type: str,
    hourly_rate: Optional[float],
) -> None:
    meta = row["metadata"]
    cfg  = row["config"]
    cfg_dict = cfg.to_dict()
    cfg_dict["config_hash"] = cfg.config_hash()

    cost = None
    if hourly_rate and row["mean_runtime_ms"] > 0:
        cost = compute_cost_per_run(row["mean_runtime_ms"], hourly_rate)
        row["cost_per_run"] = cost

    greeks = row.get("greeks") or {}
    lang, backend = _ENGINE_META.get(row["engine"], ("unknown", "unknown"))

    db.store_run_full(
        config_dict=cfg_dict,
        engine=row["engine"],
        ad_mode=row["ad_mode"],
        experiment_id=experiment_id,
        experiment_type=experiment_type,
        mean_runtime_ms=row["mean_runtime_ms"],
        std_runtime_ms=row["std_runtime_ms"],
        min_runtime_ms=row["min_runtime_ms"],
        max_runtime_ms=row["max_runtime_ms"],
        throughput_paths_per_sec=row["throughput"],
        ad_overhead_ratio=row["ad_overhead_ratio"],
        result_value=row["result"],
        greek_delta=greeks.get("delta"),
        greek_vega=greeks.get("vega"),
        memory_peak_mb=row["memory_peak_mb"],
        language=lang,
        backend=backend,
        cpu_model=meta.get("cpu_model"),
        cpu_architecture=meta.get("cpu_architecture"),
        cpu_count=meta.get("cpu_count"),
        memory_gb=meta.get("memory_gb"),
        platform=meta.get("platform"),
        python_version=meta.get("python_version"),
        numpy_version=meta.get("numpy_version"),
        jax_version=meta.get("jax_version"),
        cost_per_run=cost,
    )


# ---------------------------------------------------------------------------
# Candidate key helpers
# ---------------------------------------------------------------------------

def _config_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    """Unique key for a (workload, engine, ad_mode) triple."""
    return (row["workload"], row["engine"], row["ad_mode"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Budget-aware profiler vs naive grid search")
    parser.add_argument("--workloads",    nargs="+", default=["european", "european_local_vol", "asian"])
    parser.add_argument("--engines",      nargs="+", default=["cpu", "jax"])
    parser.add_argument("--m-probe",      type=int,  default=1_000)
    parser.add_argument("--m-values",     nargs="+", type=int, default=[10_000, 50_000, 100_000])
    parser.add_argument("--runs-probe",   type=int,  default=2)
    parser.add_argument("--warmup-probe", type=int,  default=1)
    parser.add_argument("--runs",         type=int,  default=5)
    parser.add_argument("--warmup",       type=int,  default=2)
    parser.add_argument("--top-k",        type=int,  default=3)
    parser.add_argument("--probe-only",   action="store_true")
    parser.add_argument("--db-path",      default="results/benchmarks.db")
    parser.add_argument("--summary-path", default="results/profiler_summary.txt")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = BenchmarkDB(db_path)

    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Cloud pricing (local runs get rate=None, cost=0)
    gcp_meta = get_instance_metadata()
    instance_type = gcp_meta.get("instance_type")
    hourly_rate: Optional[float] = None
    if instance_type:
        try:
            hourly_rate = get_hourly_rate(instance_type)
        except Exception:
            pass

    experiment_id = str(uuid.uuid4())
    lines: List[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    log("=" * 70)
    log("Budget-Aware Profiler vs Naive Grid Search")
    log(f"Experiment ID : {experiment_id}")
    log(f"Workloads     : {args.workloads}")
    log(f"Engines       : {args.engines}")
    log(f"Probe M       : {args.m_probe}")
    log(f"Full M values : {args.m_values}")
    log(f"Probe runs    : {args.runs_probe}  warmup={args.warmup_probe}")
    log(f"Full runs     : {args.runs}  warmup={args.warmup}")
    log(f"Top-K         : {args.top_k}")
    log(f"Instance      : {instance_type or 'local'}")
    log(f"Hourly rate   : {'${:.4f}'.format(hourly_rate) if hourly_rate else 'n/a (local)'}")
    log("=" * 70)

    # -----------------------------------------------------------------------
    # Step 1: Build candidate list
    # -----------------------------------------------------------------------
    candidates: List[Tuple[str, str, str]] = []  # (workload, engine, ad_mode)
    for wl in args.workloads:
        for eng in args.engines:
            if eng not in _ENGINES:
                continue
            if _ENGINES[eng] is None:
                continue
            if wl not in _ENGINE_WORKLOADS.get(eng, set()):
                continue
            for ad in _ENGINE_AD_MODES.get(eng, ["none"]):
                candidates.append((wl, eng, ad))

    log(f"\nTotal candidates: {len(candidates)}")

    # -----------------------------------------------------------------------
    # Step 2: Probe runs
    # -----------------------------------------------------------------------
    log("\n--- PROBE RUNS (M={}) ---".format(args.m_probe))
    probe_results: List[Dict[str, Any]] = []
    for wl, eng, ad in candidates:
        print(f"  probe {eng}/{wl}/{ad} ...", end=" ", flush=True)
        row = run_one(eng, wl, args.m_probe, ad, args.warmup_probe, args.runs_probe)
        if row is None:
            print("skipped")
            continue
        print(f"{row['mean_runtime_ms']:.2f} ms  price={row['result']:.4f}")
        probe_results.append(row)
        save_to_db(db, row, experiment_id, "probe_run", hourly_rate)

    log(f"\nProbe runs completed: {len(probe_results)} / {len(candidates)}")

    if args.probe_only:
        log("\n[probe-only mode] Stopping after probe.")
        _write_summary(summary_path, lines)
        return

    # -----------------------------------------------------------------------
    # Step 3: Score and select promising configurations
    # -----------------------------------------------------------------------
    log("\n--- PROFILER SELECTION ---")

    # Group probe results by workload
    by_workload: Dict[str, List[Dict[str, Any]]] = {}
    for row in probe_results:
        by_workload.setdefault(row["workload"], []).append(row)

    selected_keys: Dict[str, List[Tuple[str, str, str]]] = {}  # workload -> list of (wl,eng,ad)

    for wl, wl_rows in by_workload.items():
        # Score each row
        scored = sorted(wl_rows, key=_probe_score)

        # Remove dominated configs on (runtime, cost) — cost=0 locally so
        # dominance is purely on runtime; keep as many as top_k after filtering.
        # For local runs cost_per_run=0 for all, so Pareto=all; use score order.
        pareto = compute_pareto_frontier(wl_rows, x_key="mean_runtime_ms", y_key="cost_per_run")
        if len(pareto) == 0:
            pareto = scored  # fallback: all valid rows

        # Take top-K by score from the non-dominated set
        pareto_scored = sorted(pareto, key=_probe_score)
        top = pareto_scored[:args.top_k]

        # Ensure fastest and cheapest are always included (may already be in top)
        fastest = min(wl_rows, key=lambda r: r["mean_runtime_ms"], default=None)
        if fastest and _config_key(fastest) not in [_config_key(t) for t in top]:
            top.append(fastest)

        selected_keys[wl] = [_config_key(r) for r in top]

        log(f"\n  {wl}: {len(wl_rows)} probed -> {len(top)} selected")
        for r in top:
            log(f"    {r['engine']}/{r['ad_mode']}  score={_probe_score(r):.2f}  "
                f"rt={r['mean_runtime_ms']:.2f}ms  price={r['result']:.4f}")

    total_selected = sum(len(v) for v in selected_keys.values())
    total_full_configs = len(candidates) * len(args.m_values)
    profiler_full_configs = total_selected * len(args.m_values)
    runs_saved = total_full_configs - profiler_full_configs

    # -----------------------------------------------------------------------
    # Step 4: Full naive grid-search
    # -----------------------------------------------------------------------
    log("\n--- FULL NAIVE GRID SEARCH ---")
    log(f"Running {len(candidates)} configs x {len(args.m_values)} M values = {total_full_configs} runs")

    full_results: Dict[Tuple, Dict[str, Any]] = {}  # (wl,eng,ad,M) -> row

    for wl, eng, ad in candidates:
        for M in args.m_values:
            print(f"  grid {eng}/{wl}/{ad} M={M} ...", end=" ", flush=True)
            row = run_one(eng, wl, M, ad, args.warmup, args.runs)
            if row is None:
                print("skipped")
                continue
            print(f"{row['mean_runtime_ms']:.2f} ms")
            full_results[(wl, eng, ad, M)] = row
            # Mark as profiler_selected if this config was chosen
            exp_type = "grid_search_full"
            if (wl, eng, ad) in selected_keys.get(wl, []):
                exp_type = "profiler_selected"
            save_to_db(db, row, experiment_id, exp_type, hourly_rate)

    log(f"\nFull grid completed: {len(full_results)} runs stored")

    # -----------------------------------------------------------------------
    # Step 5: Pareto frontier per workload on full results
    # -----------------------------------------------------------------------
    log("\n--- PARETO FRONTIER (FULL GRID) ---")

    pareto_keys_by_workload: Dict[str, List[Tuple[str, str, str]]] = {}

    for wl in args.workloads:
        wl_full = [r for (w, e, a, m), r in full_results.items() if w == wl]
        if not wl_full:
            continue
        # Use M=max for Pareto (most representative)
        max_M = max(args.m_values)
        wl_max = [r for (w, e, a, m), r in full_results.items() if w == wl and m == max_M]
        if not wl_max:
            wl_max = wl_full

        # If all cost_per_run are 0 (local), use runtime only as x-axis
        # by setting a synthetic cost proportional to runtime * M
        for r in wl_max:
            if r["cost_per_run"] == 0.0:
                r = dict(r)
                r["cost_per_run"] = r["mean_runtime_ms"] * max_M / 1e9  # tiny synthetic cost

        pareto = compute_pareto_frontier(wl_max, x_key="mean_runtime_ms", y_key="cost_per_run")
        pareto_keys = [_config_key(r) for r in pareto]
        pareto_keys_by_workload[wl] = pareto_keys

        log(f"\n  {wl} Pareto frontier ({len(pareto)} / {len(wl_max)} configs at M={max_M}):")
        for r in sorted(pareto, key=lambda x: x["mean_runtime_ms"]):
            log(f"    {r['engine']}/{r['ad_mode']}  rt={r['mean_runtime_ms']:.2f}ms")

    # -----------------------------------------------------------------------
    # Step 6: Overlap report
    # -----------------------------------------------------------------------
    log("\n--- PROFILER vs PARETO OVERLAP ---")

    total_pareto = 0
    total_recovered = 0
    total_missed = 0

    for wl in args.workloads:
        sel  = set(selected_keys.get(wl, []))
        par  = set(pareto_keys_by_workload.get(wl, []))
        recovered = sel & par
        missed    = par - sel
        total_pareto    += len(par)
        total_recovered += len(recovered)
        total_missed    += len(missed)
        log(f"\n  {wl}:")
        log(f"    Pareto frontier size       : {len(par)}")
        log(f"    Profiler selected          : {len(sel)}")
        log(f"    Recovered Pareto configs   : {len(recovered)}")
        log(f"    Missed Pareto configs      : {len(missed)}")
        if missed:
            log(f"    Missed: {[f'{e}/{a}' for w,e,a in missed]}")

    pct_saved = 100.0 * runs_saved / total_full_configs if total_full_configs > 0 else 0.0
    pct_recovered = 100.0 * total_recovered / total_pareto if total_pareto > 0 else 0.0

    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"Total full-grid configs         : {total_full_configs}")
    log(f"Profiler-selected full configs  : {profiler_full_configs}")
    log(f"Full runs saved                 : {runs_saved}  ({pct_saved:.1f}%)")
    log(f"Total Pareto-optimal configs    : {total_pareto}")
    log(f"Pareto configs recovered        : {total_recovered}  ({pct_recovered:.1f}%)")
    log(f"Pareto configs missed           : {total_missed}")
    log("=" * 70)

    _write_summary(summary_path, lines)
    print(f"\nSummary written to {summary_path}")
    print(f"Results stored in {db_path}")


def _write_summary(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
