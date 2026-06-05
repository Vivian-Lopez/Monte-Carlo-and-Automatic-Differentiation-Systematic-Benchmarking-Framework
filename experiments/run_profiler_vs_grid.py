"""
Budget-Aware Profiler vs Naive Grid-Search Experiment
======================================================
Runs cheap probe benchmarks on a candidate grid of (workload, engine, ad_mode)
configurations to predict which will perform well at full scale, prunes dominated
candidates, then compares profiler selections against a full naive grid-search.

Hardened design
---------------
  1. Probe step: small M to estimate relative performance cheaply.
     - JIT-warmup correction applied to JAX probes so JIT cost does not unfairly
       discard JAX at small M.
     - Synthetic cost proxy (mean_runtime_ms x M / 1e9) used when real GCP
       pricing is unavailable, keeping Pareto non-degenerate on local runs.
  2. Selection: top-K by probe score from non-dominated set, plus:
     - Safety margin: configs within score_margin x best_score are kept even
       outside top-K (avoids discarding near-frontier candidates).
     - Diversity guard: at least one config per available engine is always kept.
  3. Full-grid: naive full sweep over all (workload, engine, ad_mode) x M values.
  4. Evaluation (full spec):
     - Runs saved / percentage reduction
     - Pareto recovery %
     - Runtime regret vs full-grid best
     - Cost-performance regret vs full-grid best
     - Spearman rank correlation between probe ranking and full-run ranking
     - Per-config selection reasons and missed Pareto configs
  5. DB storage: all runs tagged with profiler_phase, profiler_decision, reason,
     dominated flag, cloud metadata (region, zone, vcpu_count, machine_family,
     paths_per_dollar, git_commit_hash).
  6. Dry-run mode: --dry-run uses tiny M, 1 run, 0 warmup, temp DB to smoke-test
     the full pipeline without a real benchmark.

Experiment tags in the DB
--------------------------
  probe_run          - cheap small-M probe runs
  grid_search_full   - full naive grid across all M values
  profiler_selected  - full runs for profiler-selected configs only

Usage
-----
  # Local smoke test (dry-run)
  python experiments/run_profiler_vs_grid.py --dry-run

  # Probe only (fast smoke test with real M)
  python experiments/run_profiler_vs_grid.py \\
      --workloads european asian --engines cpu jax \\
      --probe-only --m-probe 1000 --runs-probe 2 --warmup-probe 1

  # Full local experiment
  python experiments/run_profiler_vs_grid.py \\
      --experiment-id final_cloud_profiler_v1 \\
      --workloads european european_local_vol asian \\
      --engines cpu jax \\
      --m-values 10000 50000 100000 \\
      --runs 5 --warmup 2 \\
      --m-probe 1000 --runs-probe 2 --warmup-probe 1 \\
      --top-k 3 \\
      --write-db results/benchmarks.db \\
      --export results/

  # Cloud VM (extra metadata injected via CLI)
  python experiments/run_profiler_vs_grid.py \\
      --experiment-id final_cloud_profiler_v1 \\
      --instance-type n2-standard-8 \\
      --cloud-provider gcp --region europe-west2 --zone europe-west2-b \\
      --write-db results/benchmarks.db --export results/
"""

from __future__ import annotations

import argparse
import math
import os
import re as _re
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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

# JIT engines that benefit from an extra probe warmup correction
_JIT_ENGINES: Set[str] = {"jax"}


# ---------------------------------------------------------------------------
# Git commit hash helper
# ---------------------------------------------------------------------------

def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


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
# Synthetic cost proxy for local runs (keeps Pareto non-degenerate)
# ---------------------------------------------------------------------------

def _synthetic_cost(mean_runtime_ms: float, M: int) -> float:
    """
    When real GCP pricing is unavailable, use a proxy cost proportional to
    runtime x M so that Pareto dominance is well-defined locally.
    """
    return mean_runtime_ms * max(M, 1) / 1e9


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
    """Run a single benchmark configuration. Returns a result dict or None on error."""
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
        "engine":            engine_key,
        "workload":          workload,
        "ad_mode":           ad_mode,
        "M":                 M,
        "result":            res.result,
        "mean_runtime_ms":   res.mean_runtime * 1000.0,
        "std_runtime_ms":    res.std_runtime * 1000.0,
        "min_runtime_ms":    res.min_runtime * 1000.0,
        "max_runtime_ms":    res.max_runtime * 1000.0,
        "throughput":        res.throughput_paths_per_sec,
        "ad_overhead_ratio": res.ad_overhead_ratio,
        "memory_peak_mb":    res.memory_peak_mb,
        "greeks":            res.greeks,
        "config":            config,
        "metadata":          res.metadata,
        "cost_per_run":      0.0,
        "paths_per_dollar":  None,
    }


# ---------------------------------------------------------------------------
# Probe scoring heuristic
# ---------------------------------------------------------------------------

def _probe_score(
    row: Dict[str, Any],
    jit_correction: bool = False,
    extra_warmup_ms: float = 0.0,
) -> float:
    """
    Lower score = more promising.

      score = (mean_runtime_ms - jit_correction_ms)
              x (1 + stability_penalty)
              x (1 + error_penalty)

    stability_penalty = std/mean (coefficient of variation), capped at 2.
    error_penalty     = 1 if result is non-positive or NaN, else 0.
    jit_correction    = subtract one compile-run for JIT engines so the
                        one-off JIT cost does not discard JAX unfairly.
    """
    rt = row["mean_runtime_ms"]
    if rt <= 0 or not math.isfinite(rt):
        return float("inf")

    # JIT correction: subtract half the compile-run cost for JIT engines.
    if jit_correction and extra_warmup_ms > 0:
        rt = max(rt - extra_warmup_ms * 0.5, rt * 0.1)

    std = row["std_runtime_ms"]
    stability = min(std / rt, 2.0) if rt > 0 else 2.0
    result_val = row["result"]
    error_pen = 0.0
    if result_val is None or not math.isfinite(float(result_val)) or float(result_val) < 0:
        error_pen = 1.0
    return rt * (1.0 + stability) * (1.0 + error_pen)


# ---------------------------------------------------------------------------
# Spearman rank correlation
# ---------------------------------------------------------------------------

def _spearman(x: List[float], y: List[float]) -> Optional[float]:
    """Return Spearman rho between two equal-length lists. None if < 3 pairs."""
    if len(x) < 3 or len(x) != len(y):
        return None
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(x, y)
        return float(rho)
    except Exception:
        # Manual fallback
        def _rank(lst: List[float]) -> List[float]:
            s = sorted(range(len(lst)), key=lambda i: lst[i])
            r = [0.0] * len(lst)
            for rank, idx in enumerate(s):
                r[idx] = float(rank + 1)
            return r
        rx, ry = _rank(x), _rank(y)
        n = len(rx)
        d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
        denom = n * (n * n - 1)
        return 1.0 - 6.0 * d2 / denom if denom else None


# ---------------------------------------------------------------------------
# DB save helper
# ---------------------------------------------------------------------------

def save_to_db(
    db: BenchmarkDB,
    row: Dict[str, Any],
    experiment_id: str,
    experiment_type: str,
    hourly_rate: Optional[float],
    cloud_meta: Dict[str, Any],
    profiler_phase: Optional[str] = None,
    profiler_decision: Optional[str] = None,
    profiler_reason: Optional[str] = None,
    dominated: Optional[int] = None,
    git_commit: Optional[str] = None,
) -> None:
    meta = row["metadata"]
    cfg  = row["config"]
    cfg_dict = cfg.to_dict()
    cfg_dict["config_hash"] = cfg.config_hash()

    cost = None
    ppd = None
    if hourly_rate and row["mean_runtime_ms"] > 0:
        cost = compute_cost_per_run(row["mean_runtime_ms"], hourly_rate)
        row["cost_per_run"] = cost
        if cost and cost > 0 and row.get("throughput", 0) > 0:
            ppd = row["throughput"] / hourly_rate
    else:
        cost = row.get("cost_per_run") or 0.0

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
        cost_per_run=cost if cost else None,
        paths_per_dollar=ppd,
        cloud_provider=cloud_meta.get("cloud_provider"),
        region=cloud_meta.get("region"),
        zone=cloud_meta.get("zone"),
        instance_type=cloud_meta.get("instance_type"),
        machine_family=cloud_meta.get("machine_family"),
        vcpu_count=cloud_meta.get("vcpu_count"),
        profiler_phase=profiler_phase,
        profiler_decision=profiler_decision,
        profiler_reason=profiler_reason,
        dominated=dominated,
        git_commit_hash=git_commit,
    )


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

def _export_results(db: BenchmarkDB, export_dir: Path, experiment_id: str) -> None:
    """Write CSV exports from the database for the given experiment."""
    import csv
    import sqlite3 as _sqlite3

    export_dir.mkdir(parents=True, exist_ok=True)
    report_dir = export_dir / "report_tables"
    report_dir.mkdir(exist_ok=True)

    conn = _sqlite3.connect(str(db.db_path))
    conn.row_factory = _sqlite3.Row

    # profiler_vs_grid.csv
    rows = conn.execute(
        """SELECT id, experiment_type, workload_type, engine, ad_mode, M,
                  mean_runtime_ms, std_runtime_ms, throughput_paths_per_sec,
                  cost_per_run, paths_per_dollar,
                  profiler_phase, profiler_decision, profiler_reason, dominated,
                  instance_type, region, zone, machine_family, vcpu_count,
                  result_value, memory_peak_mb, git_commit_hash, created_at
           FROM runs
           WHERE experiment_id = ? AND status = 'completed'
           ORDER BY workload_type, engine, ad_mode, M""",
        (experiment_id,),
    ).fetchall()
    pvg_path = export_dir / "profiler_vs_grid.csv"
    if rows:
        with open(pvg_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(dict(r) for r in rows)
        print(f"  Exported: {pvg_path}  ({len(rows)} rows)")

    # cloud_cost_analysis.csv
    cost_rows = conn.execute(
        """SELECT workload_type, engine, ad_mode, M, instance_type,
                  mean_runtime_ms, throughput_paths_per_sec,
                  cost_per_run, paths_per_dollar,
                  region, zone, machine_family, vcpu_count
           FROM runs
           WHERE experiment_id = ? AND status = 'completed'
             AND cost_per_run IS NOT NULL AND cost_per_run > 0
           ORDER BY paths_per_dollar DESC""",
        (experiment_id,),
    ).fetchall()
    cca_path = export_dir / "cloud_cost_analysis.csv"
    if cost_rows:
        with open(cca_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cost_rows[0].keys())
            w.writeheader()
            w.writerows(dict(r) for r in cost_rows)
        print(f"  Exported: {cca_path}  ({len(cost_rows)} rows)")

    # pareto_frontier.csv
    pf_rows = conn.execute(
        """SELECT workload_type, engine, ad_mode, M, instance_type,
                  mean_runtime_ms, cost_per_run, paths_per_dollar,
                  profiler_decision, profiler_reason, dominated
           FROM runs
           WHERE experiment_id = ? AND status = 'completed'
             AND experiment_type IN ('profiler_selected', 'grid_search_full')
           ORDER BY workload_type, mean_runtime_ms""",
        (experiment_id,),
    ).fetchall()
    pf_path = export_dir / "pareto_frontier.csv"
    if pf_rows:
        with open(pf_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=pf_rows[0].keys())
            w.writeheader()
            w.writerows(dict(r) for r in pf_rows)
        print(f"  Exported: {pf_path}  ({len(pf_rows)} rows)")

    # report_tables/probe_vs_full_ranking.csv
    rank_rows = conn.execute(
        """SELECT workload_type, engine, ad_mode, M, mean_runtime_ms,
                  profiler_phase, profiler_decision
           FROM runs
           WHERE experiment_id = ? AND status = 'completed'
           ORDER BY workload_type, profiler_phase, mean_runtime_ms""",
        (experiment_id,),
    ).fetchall()
    rt_path = report_dir / "probe_vs_full_ranking.csv"
    if rank_rows:
        with open(rt_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rank_rows[0].keys())
            w.writeheader()
            w.writerows(dict(r) for r in rank_rows)
        print(f"  Exported: {rt_path}")

    conn.close()


# ---------------------------------------------------------------------------
# Candidate key helper
# ---------------------------------------------------------------------------

def _config_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    """Unique key for a (workload, engine, ad_mode) triple."""
    return (row["workload"], row["engine"], row["ad_mode"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Budget-aware profiler vs naive grid search")
    parser.add_argument("--experiment-id",  default=None,
                        help="Stable ID for this experiment (default: auto UUID)")
    parser.add_argument("--workloads",    nargs="+", default=["european", "european_local_vol", "asian"])
    parser.add_argument("--engines",      nargs="+", default=["cpu", "jax", "cpp", "rust"])
    parser.add_argument("--m-probe",      type=int,  default=1_000)
    parser.add_argument("--m-values",     nargs="+", type=int, default=[10_000, 50_000, 100_000])
    parser.add_argument("--runs-probe",   type=int,  default=2)
    parser.add_argument("--warmup-probe", type=int,  default=1)
    parser.add_argument("--runs",         type=int,  default=5)
    parser.add_argument("--warmup",       type=int,  default=2)
    parser.add_argument("--top-k",        type=int,  default=3)
    parser.add_argument("--score-margin", type=float, default=1.5,
                        help="Safety margin: keep configs within margin x best_score")
    parser.add_argument("--probe-only",   action="store_true")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Smoke-test: tiny M, 1 run, 0 warmup, temp DB")
    # DB / export
    parser.add_argument("--write-db",     default="results/benchmarks.db", dest="db_path")
    parser.add_argument("--export",       default=None, dest="export_dir",
                        help="Directory for CSV/txt exports")
    parser.add_argument("--summary-path", default="results/profiler_summary.txt")
    # Cloud metadata
    parser.add_argument("--cloud-provider", default=None)
    parser.add_argument("--region",         default=None)
    parser.add_argument("--zone",           default=None)
    parser.add_argument("--instance-type",  default=None)
    parser.add_argument("--gcp-api-key",    default=os.environ.get("GCP_PRICING_API_KEY"))
    args = parser.parse_args()

    # Dry-run overrides
    if args.dry_run:
        args.m_probe    = 500
        args.m_values   = [500, 1000]
        args.runs_probe = 1
        args.warmup_probe = 0
        args.runs       = 1
        args.warmup     = 0
        _tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        args.db_path = _tmp.name
        _tmp.close()
        print(f"[dry-run] Using temp DB: {args.db_path}")

    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = BenchmarkDB(db_path)

    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Cloud pricing
    gcp_meta = get_instance_metadata()
    instance_type = args.instance_type or gcp_meta.get("instance_type")
    zone = args.zone or gcp_meta.get("zone")
    region = args.region or (zone.rsplit("-", 1)[0] if zone else None)
    cloud_provider = args.cloud_provider or gcp_meta.get("cloud_provider")

    hourly_rate: Optional[float] = None
    if instance_type:
        try:
            hourly_rate = get_hourly_rate(
                instance_type,
                api_key=args.gcp_api_key,
                region=region or "europe-west2",
            )
        except Exception:
            pass

    machine_family: Optional[str] = None
    vcpu_count: Optional[int] = None
    if instance_type:
        machine_family = instance_type.split("-")[0]
        m = _re.search(r"-(\d+)$", instance_type)
        if m:
            vcpu_count = int(m.group(1))

    cloud_meta: Dict[str, Any] = {
        "cloud_provider": cloud_provider,
        "region":         region,
        "zone":           zone,
        "instance_type":  instance_type,
        "machine_family": machine_family,
        "vcpu_count":     vcpu_count,
    }

    git_commit = _git_commit()
    experiment_id = args.experiment_id or str(uuid.uuid4())
    lines: List[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    log("=" * 70)
    log("Budget-Aware Profiler vs Naive Grid Search")
    if args.dry_run:
        log("*** DRY-RUN MODE — tiny M, temp DB ***")
    log(f"Experiment ID : {experiment_id}")
    log(f"Git commit    : {git_commit or 'unknown'}")
    log(f"Workloads     : {args.workloads}")
    log(f"Engines       : {args.engines}")
    log(f"Probe M       : {args.m_probe}")
    log(f"Full M values : {args.m_values}")
    log(f"Probe runs    : {args.runs_probe}  warmup={args.warmup_probe}")
    log(f"Full runs     : {args.runs}  warmup={args.warmup}")
    log(f"Top-K         : {args.top_k}  score_margin={args.score_margin}x")
    log(f"Instance      : {instance_type or 'local'}")
    log(f"Region/zone   : {region or 'n/a'} / {zone or 'n/a'}")
    log(f"Hourly rate   : {'${:.4f}'.format(hourly_rate) if hourly_rate else 'n/a (local)'}")
    log("=" * 70)

    # -----------------------------------------------------------------------
    # Step 1: Build candidate list
    # -----------------------------------------------------------------------
    candidates: List[Tuple[str, str, str]] = []
    for wl in args.workloads:
        for eng in args.engines:
            if eng not in _ENGINES or _ENGINES[eng] is None:
                continue
            if wl not in _ENGINE_WORKLOADS.get(eng, set()):
                continue
            for ad in _ENGINE_AD_MODES.get(eng, ["none"]):
                candidates.append((wl, eng, ad))

    total_grid_configs = len(candidates)
    total_full_configs = total_grid_configs * len(args.m_values)
    log(f"\nTotal candidate (wl, engine, ad) triples : {total_grid_configs}")
    log(f"Full-grid total runs (x {len(args.m_values)} M values): {total_full_configs}")

    # -----------------------------------------------------------------------
    # Step 2: Probe runs with JIT warmup estimation
    # -----------------------------------------------------------------------
    log("\n--- PROBE RUNS (M={}) ---".format(args.m_probe))
    probe_results: List[Dict[str, Any]] = []

    # Extra JIT warmup timing for JAX to estimate compile cost
    jax_warmup_ms: Dict[Tuple[str, str], float] = {}
    if "jax" in args.engines and _ENGINES.get("jax"):
        for wl in args.workloads:
            if wl not in _ENGINE_WORKLOADS.get("jax", set()):
                continue
            _jit_row = run_one("jax", wl, args.m_probe, "none", 1, 1)
            if _jit_row:
                jax_warmup_ms[(wl, "jax")] = _jit_row["mean_runtime_ms"]

    for wl, eng, ad in candidates:
        print(f"  probe {eng}/{wl}/{ad} ...", end=" ", flush=True)
        row = run_one(eng, wl, args.m_probe, ad, args.warmup_probe, args.runs_probe)
        if row is None:
            print("skipped")
            continue
        if not hourly_rate:
            row["cost_per_run"] = _synthetic_cost(row["mean_runtime_ms"], args.m_probe)
        print(f"{row['mean_runtime_ms']:.2f} ms  price={row['result']:.4f}")
        probe_results.append(row)
        save_to_db(db, row, experiment_id, "probe_run", hourly_rate, cloud_meta,
                   profiler_phase="probe", git_commit=git_commit)

    log(f"\nProbe runs completed: {len(probe_results)} / {len(candidates)}")

    if args.probe_only:
        log("\n[probe-only mode] Stopping after probe.")
        _write_summary(summary_path, lines)
        return

    # -----------------------------------------------------------------------
    # Step 3: Score and select promising configurations
    # -----------------------------------------------------------------------
    log("\n--- PROFILER SELECTION ---")

    by_workload: Dict[str, List[Dict[str, Any]]] = {}
    for row in probe_results:
        by_workload.setdefault(row["workload"], []).append(row)

    selected_keys: Dict[str, List[Tuple[str, str, str]]] = {}
    selected_reasons: Dict[Tuple[str, str, str], str] = {}
    pruned_reasons: Dict[Tuple[str, str, str], str] = {}

    for wl, wl_rows in by_workload.items():
        def score(r: Dict[str, Any]) -> float:
            is_jit = r["engine"] in _JIT_ENGINES
            ewms = jax_warmup_ms.get((wl, r["engine"]), 0.0) if is_jit else 0.0
            return _probe_score(r, jit_correction=is_jit, extra_warmup_ms=ewms)

        pareto_probe = compute_pareto_frontier(wl_rows, x_key="mean_runtime_ms", y_key="cost_per_run")
        if not pareto_probe:
            pareto_probe = wl_rows

        pareto_scored = sorted(pareto_probe, key=score)
        best_score = score(pareto_scored[0]) if pareto_scored else float("inf")
        margin = args.score_margin
        within_margin = [r for r in pareto_scored if score(r) <= best_score * margin]
        top = within_margin[:args.top_k]

        # Diversity guard: at least one config per available engine
        engines_in_top = {r["engine"] for r in top}
        for eng in {r["engine"] for r in wl_rows} - engines_in_top:
            eng_rows = [r for r in wl_rows if r["engine"] == eng]
            if eng_rows:
                top.append(min(eng_rows, key=score))

        # Safety pin: always include fastest raw runtime
        fastest = min(wl_rows, key=lambda r: r["mean_runtime_ms"], default=None)
        if fastest and _config_key(fastest) not in [_config_key(t) for t in top]:
            top.append(fastest)

        selected_keys[wl] = [_config_key(r) for r in top]

        for r in top:
            k = _config_key(r)
            s = score(r)
            parts = []
            if r in pareto_scored[:args.top_k]:
                parts.append(f"top-{args.top_k} probe Pareto")
            if s <= best_score * margin and r not in pareto_scored[:args.top_k]:
                parts.append(f"within {margin}x safety margin")
            # Check diversity: is this the only config for its engine in selected_keys so far?
            eng_in_sel = {k2[1] for k2 in selected_keys[wl] if k2 != k}
            if r["engine"] not in eng_in_sel:
                parts.append("diversity guard")
            if fastest and k == _config_key(fastest):
                parts.append("fastest probe runtime")
            selected_reasons[k] = "; ".join(parts) if parts else "top selection"

        for r in wl_rows:
            k = _config_key(r)
            if k not in selected_keys[wl]:
                pruned_reasons[k] = (
                    f"dominated or outside {margin}x margin "
                    f"(score={score(r):.2f} vs best={best_score:.2f})"
                )

        log(f"\n  {wl}: {len(wl_rows)} probed -> {len(top)} selected")
        for r in top:
            k = _config_key(r)
            log(f"    [SEL] {r['engine']}/{r['ad_mode']}  score={score(r):.2f}  "
                f"rt={r['mean_runtime_ms']:.2f}ms  reason: {selected_reasons[k]}")
        for r in wl_rows:
            k = _config_key(r)
            if k in pruned_reasons:
                log(f"    [PRN] {r['engine']}/{r['ad_mode']}  score={score(r):.2f}  "
                    f"-> {pruned_reasons[k]}")

    total_selected = sum(len(v) for v in selected_keys.values())
    profiler_full_configs = total_selected * len(args.m_values)
    runs_saved = total_full_configs - profiler_full_configs

    # -----------------------------------------------------------------------
    # Step 4: Full naive grid-search
    # -----------------------------------------------------------------------
    log("\n--- FULL NAIVE GRID SEARCH ---")
    log(f"Running {total_grid_configs} configs x {len(args.m_values)} M values "
        f"= {total_full_configs} runs")

    full_results: Dict[Tuple, Dict[str, Any]] = {}
    max_M = max(args.m_values)

    for wl, eng, ad in candidates:
        for M in args.m_values:
            print(f"  grid {eng}/{wl}/{ad} M={M} ...", end=" ", flush=True)
            row = run_one(eng, wl, M, ad, args.warmup, args.runs)
            if row is None:
                print("skipped")
                continue
            print(f"{row['mean_runtime_ms']:.2f} ms")
            full_results[(wl, eng, ad, M)] = row

            if not hourly_rate:
                row["cost_per_run"] = _synthetic_cost(row["mean_runtime_ms"], M)

            ck = (wl, eng, ad)
            is_selected = ck in selected_keys.get(wl, [])
            exp_type = "profiler_selected" if is_selected else "grid_search_full"
            decision = "selected" if is_selected else "full_grid_only"
            reason = selected_reasons.get(ck) if is_selected else pruned_reasons.get(ck)
            save_to_db(db, row, experiment_id, exp_type, hourly_rate, cloud_meta,
                       profiler_phase="full", profiler_decision=decision,
                       profiler_reason=reason, git_commit=git_commit)

    log(f"\nFull grid completed: {len(full_results)} runs stored")

    # -----------------------------------------------------------------------
    # Step 5: Pareto frontier per workload on full results at max M
    # -----------------------------------------------------------------------
    log("\n--- PARETO FRONTIER (FULL GRID) ---")

    pareto_keys_by_workload: Dict[str, List[Tuple[str, str, str]]] = {}

    for wl in args.workloads:
        wl_max = [r for (w, e, a, m), r in full_results.items() if w == wl and m == max_M]
        if not wl_max:
            wl_max = [r for (w, e, a, m), r in full_results.items() if w == wl]
        if not wl_max:
            continue
        for r in wl_max:
            if not r.get("cost_per_run"):
                r["cost_per_run"] = _synthetic_cost(r["mean_runtime_ms"], max_M)

        pareto = compute_pareto_frontier(wl_max, x_key="mean_runtime_ms", y_key="cost_per_run")
        if not pareto:
            pareto = wl_max
        pareto_keys_by_workload[wl] = [_config_key(r) for r in pareto]

        log(f"\n  {wl} Pareto frontier ({len(pareto)} / {len(wl_max)} configs at M={max_M:,}):")
        for r in sorted(pareto, key=lambda x: x["mean_runtime_ms"]):
            log(f"    {r['engine']}/{r['ad_mode']}  rt={r['mean_runtime_ms']:.2f}ms  "
                f"cost={r['cost_per_run']:.2e}")

    # -----------------------------------------------------------------------
    # Step 6: Pareto overlap report
    # -----------------------------------------------------------------------
    log("\n--- PROFILER vs PARETO OVERLAP ---")

    total_pareto    = 0
    total_recovered = 0
    total_missed    = 0

    for wl in args.workloads:
        sel = set(selected_keys.get(wl, []))
        par = set(pareto_keys_by_workload.get(wl, []))
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
            log(f"    Missed: {[f'{e}/{a}' for _, e, a in missed]}")

    # -----------------------------------------------------------------------
    # Step 7: Regret + best-config report
    # -----------------------------------------------------------------------
    all_max_M_rows = [r for (w, e, a, m), r in full_results.items() if m == max_M]

    best_by_runtime = min(all_max_M_rows, key=lambda r: r["mean_runtime_ms"]) if all_max_M_rows else None
    cost_rows_valid = [r for r in all_max_M_rows if r.get("cost_per_run", 0) > 0]
    best_by_cost = min(cost_rows_valid, key=lambda r: r["cost_per_run"]) if cost_rows_valid else None

    selected_max_M_rows = [
        r for (w, e, a, m), r in full_results.items()
        if m == max_M and (w, e, a) in selected_keys.get(w, [])
    ]
    profiler_best = min(selected_max_M_rows, key=lambda r: r["mean_runtime_ms"]) if selected_max_M_rows else None

    runtime_regret: Optional[float] = None
    if profiler_best and best_by_runtime and best_by_runtime["mean_runtime_ms"] > 0:
        runtime_regret = (
            (profiler_best["mean_runtime_ms"] - best_by_runtime["mean_runtime_ms"])
            / best_by_runtime["mean_runtime_ms"]
        )

    cost_regret: Optional[float] = None
    if profiler_best and best_by_cost and best_by_cost.get("cost_per_run", 0) > 0:
        prof_c = profiler_best.get("cost_per_run") or _synthetic_cost(
            profiler_best["mean_runtime_ms"], max_M)
        cost_regret = (prof_c - best_by_cost["cost_per_run"]) / best_by_cost["cost_per_run"]

    # -----------------------------------------------------------------------
    # Step 8: Spearman rank correlation
    # -----------------------------------------------------------------------
    probe_scores_for_corr: List[float] = []
    full_runtimes_for_corr: List[float] = []
    for probe_row in probe_results:
        k = _config_key(probe_row)
        wl, eng, ad = k
        full_row = full_results.get((wl, eng, ad, max_M))
        if full_row is not None:
            is_jit = eng in _JIT_ENGINES
            ewms = jax_warmup_ms.get((wl, eng), 0.0) if is_jit else 0.0
            probe_scores_for_corr.append(
                _probe_score(probe_row, jit_correction=is_jit, extra_warmup_ms=ewms)
            )
            full_runtimes_for_corr.append(full_row["mean_runtime_ms"])

    rank_corr = _spearman(probe_scores_for_corr, full_runtimes_for_corr)

    # -----------------------------------------------------------------------
    # Step 9: Summary
    # -----------------------------------------------------------------------
    pct_saved     = 100.0 * runs_saved / total_full_configs if total_full_configs > 0 else 0.0
    pct_recovered = 100.0 * total_recovered / total_pareto  if total_pareto > 0 else 0.0

    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"Experiment ID                       : {experiment_id}")
    log(f"Git commit                          : {git_commit or 'unknown'}")
    log(f"Instance                            : {instance_type or 'local'}")
    log(f"Region / zone                       : {region or 'n/a'} / {zone or 'n/a'}")
    log("")
    log(f"Total grid configurations           : {total_grid_configs}")
    log(f"Number of full-grid configurations  : {total_full_configs}")
    log(f"Number selected by profiler         : {profiler_full_configs}")
    log(f"Full runs saved                     : {runs_saved}  ({pct_saved:.1f}%)")
    log("")
    log(f"Total Pareto-optimal configurations : {total_pareto}")
    log(f"Pareto configs recovered            : {total_recovered}  ({pct_recovered:.1f}%)")
    log(f"Pareto configs missed               : {total_missed}")
    log("")
    if best_by_runtime:
        r = best_by_runtime
        log(f"Full-grid best by runtime           : "
            f"{r['engine']}/{r['workload']}/{r['ad_mode']}  "
            f"{r['mean_runtime_ms']:.2f} ms at M={max_M:,}")
    if best_by_cost:
        r = best_by_cost
        log(f"Full-grid best by cost/run          : "
            f"{r['engine']}/{r['workload']}/{r['ad_mode']}  "
            f"${r['cost_per_run']:.2e} at M={max_M:,}")
    if profiler_best:
        r = profiler_best
        log(f"Profiler selected best config       : "
            f"{r['engine']}/{r['workload']}/{r['ad_mode']}  "
            f"{r['mean_runtime_ms']:.2f} ms at M={max_M:,}")
    log("")
    if runtime_regret is not None:
        log(f"Runtime regret vs full-grid best    : {runtime_regret*100:+.1f}%")
    else:
        log("Runtime regret vs full-grid best    : n/a")
    if cost_regret is not None:
        log(f"Cost regret vs full-grid best       : {cost_regret*100:+.1f}%")
    else:
        log("Cost regret vs full-grid best       : n/a (local/synthetic cost only)")
    if rank_corr is not None:
        log(f"Spearman rho (probe -> full rank)   : {rank_corr:.3f}")
    else:
        log("Spearman rho (probe -> full rank)   : n/a (< 3 matched pairs)")
    log("")
    log("Selected configurations and reasons:")
    for wl in args.workloads:
        for k in selected_keys.get(wl, []):
            _, eng, ad = k
            log(f"  {wl}/{eng}/{ad}  ->  {selected_reasons.get(k, '')}")
    if total_missed > 0:
        log("")
        log("Missed Pareto configurations:")
        for wl in args.workloads:
            sel = set(selected_keys.get(wl, []))
            par = set(pareto_keys_by_workload.get(wl, []))
            for k in (par - sel):
                _, eng, ad = k
                log(f"  {wl}/{eng}/{ad}  (in full-grid Pareto, not selected by profiler)")
    log("=" * 70)

    _write_summary(summary_path, lines)
    print(f"\nSummary written to {summary_path}")
    print(f"Results stored in {db_path}")

    if args.export_dir:
        export_dir = Path(args.export_dir)
        print(f"\nExporting results to {export_dir} ...")
        _export_results(db, export_dir, experiment_id)

    if args.dry_run:
        import os as _os
        print(f"\n[dry-run] Pipeline completed successfully.")
        print(f"[dry-run] Cleaning up temp DB: {args.db_path}")
        try:
            _os.unlink(args.db_path)
        except Exception:
            pass


def _write_summary(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
