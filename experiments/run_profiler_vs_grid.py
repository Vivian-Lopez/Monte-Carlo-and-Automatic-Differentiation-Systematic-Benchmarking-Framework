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
from benchmarking.analysis.pareto import compute_pareto_frontier, ci_overlap_select
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
# Scaling-law utilities
# ---------------------------------------------------------------------------

def _fit_scaling_law(
    m_points: List[int],
    t_points: List[float],
) -> Optional[Tuple[float, float]]:
    """
    Fit t(M) = alpha*M + beta using n≥2 (M, runtime_ms) observations.

    For n==2 solves exactly; for n≥3 uses ordinary least squares (numpy
    polyfit) which reduces sensitivity to a single noisy measurement.

    Returns (alpha, beta) or None if degenerate.
    alpha: slope  — ms per simulation path (pure compute cost)
    beta:  intercept — JIT/startup overhead (ms)
    Serial fraction at scale M: beta / (alpha*M + beta)
    """
    import numpy as _np_fit
    n = len(m_points)
    if n < 2:
        return None
    m_arr = _np_fit.array(m_points, dtype=float)
    t_arr = _np_fit.array(t_points, dtype=float)
    if len(_np_fit.unique(m_arr)) < 2:
        return None
    try:
        coeffs = _np_fit.polyfit(m_arr, t_arr, 1)
        return (float(coeffs[0]), float(coeffs[1]))
    except Exception:
        return None


def _serial_fraction(alpha: float, beta: float, M: int) -> Optional[float]:
    """
    Amdahl serial fraction: fraction of runtime that cannot be parallelised
    (startup/overhead) at a given M.

    Returns None if the denominator is zero or negative (unphysical fit).
    """
    total = alpha * M + beta
    if total <= 0:
        return None
    frac = beta / total
    # Clamp to [0, 1] — a negative beta means the fit extrapolated past zero
    return max(0.0, min(1.0, frac))


# ---------------------------------------------------------------------------
# Successive Halving (SHA)
# ---------------------------------------------------------------------------

def _successive_halving(
    candidates: List[Tuple[str, str, str]],  # (workload, engine, ad_mode)
    m_levels: List[int],
    runs_per_level: List[int],
    warmup_per_level: List[int],
    eta: float,
    jax_warmup_ms: Dict[Tuple[str, str], float],
    hourly_rate: Optional[float],
    db: "BenchmarkDB",
    experiment_id: str,
    cloud_meta: Dict[str, Any],
    git_commit: Optional[str],
    log_fn,
) -> Tuple[
    List[Dict[str, Any]],                     # all sha rows (every round)
    Dict[Tuple[str, str, str], Tuple[float, float]],  # scaling_law per config_key
    List[Tuple[str, str, str]],               # final SHA-selected keys
]:
    """
    Run Successive Halving across *m_levels*.

    Round 0 is the cheap probe; subsequent rounds run on surviving configs
    only.  Within each workload, configs are pruned using CI-overlap plus
    a hard cap of ceil(n / eta) survivors per round.

    Returns:
      all_sha_rows  : flat list of every run dict across all rounds
      scaling_laws  : map from (workload, engine, ad_mode) to (alpha, beta)
      selected_keys : final set of surviving (workload, engine, ad_mode) keys
    """
    import math as _math

    # Working set: active (wl, eng, ad) triples per workload
    active_by_wl: Dict[str, List[Tuple[str, str, str]]] = {}
    for (wl, eng, ad) in candidates:
        active_by_wl.setdefault(wl, []).append((wl, eng, ad))

    all_sha_rows: List[Dict[str, Any]] = []
    # rows_by_key[round_idx][(wl, eng, ad)] = result_row
    rows_by_round: List[Dict[Tuple[str, str, str], Dict[str, Any]]] = []

    for round_idx, (M, n_runs, n_warmup) in enumerate(
        zip(m_levels, runs_per_level, warmup_per_level)
    ):
        round_rows: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        log_fn(f"\n--- SHA ROUND {round_idx}  M={M:,}  runs={n_runs}  warmup={n_warmup} ---")

        for wl, active_keys in active_by_wl.items():
            for (wl2, eng, ad) in active_keys:
                k = (wl2, eng, ad)
                print(f"  sha[{round_idx}] {eng}/{wl2}/{ad} M={M} ...", end=" ", flush=True)
                row = run_one(eng, wl2, M, ad, n_warmup, n_runs)
                if row is None:
                    print("skipped")
                    continue
                if not hourly_rate:
                    row["cost_per_run"] = _synthetic_cost(row["mean_runtime_ms"], M)
                print(f"{row['mean_runtime_ms']:.2f} ms")
                round_rows[k] = row
                all_sha_rows.append(row)
                phase_tag = f"sha_round_{round_idx}"
                save_to_db(
                    db, row, experiment_id, "sha_probe", hourly_rate, cloud_meta,
                    profiler_phase=phase_tag, git_commit=git_commit,
                    sha_round=round_idx, sha_eliminated=0,
                )

        rows_by_round.append(round_rows)

        # Pruning: only prune after round 0 (need ≥ 2 rounds to use CI)
        if round_idx == len(m_levels) - 1:
            # Final round — no pruning, all survivors selected
            break

        new_active_by_wl: Dict[str, List[Tuple[str, str, str]]] = {}
        for wl, active_keys in active_by_wl.items():
            wl_rows = [round_rows[k] for k in active_keys if k in round_rows]
            if not wl_rows:
                continue

            # CI-overlap pruning: use JIT-corrected runtimes so JIT engines
            # (JAX) are not unfairly eliminated at small M where JIT cost
            # inflates their measured runtime.
            def _jit_corrected_rt(r: Dict[str, Any]) -> float:
                is_jit = r["engine"] in _JIT_ENGINES
                if not is_jit:
                    return r["mean_runtime_ms"]
                jit_ms = jax_warmup_ms.get((r["workload"], r["engine"]), 0.0)
                return max(r["mean_runtime_ms"] - jit_ms, r["mean_runtime_ms"] * 0.1)

            wl_rows_corrected = [
                {**r, "mean_runtime_ms": _jit_corrected_rt(r)}
                for r in wl_rows
            ]

            # CI-overlap pruning against the best config
            survivors_rows = ci_overlap_select(
                wl_rows_corrected, n_runs=n_runs,
                key="mean_runtime_ms", std_key="std_runtime_ms",
            )
            # Map back to original (uncorrected) rows for logging/DB
            survivor_corrected_keys = {_config_key(r) for r in survivors_rows}
            survivors_rows = [r for r in wl_rows if _config_key(r) in survivor_corrected_keys]

            # Hard budget cap: ceil(n / eta) — ensures geometric budget decay
            cap = _math.ceil(len(wl_rows) / eta)
            if len(survivors_rows) > cap:
                survivors_rows = survivors_rows[:cap]

            survivor_keys = {_config_key(r) for r in survivors_rows}
            # Mark eliminated configs in DB
            eliminated = [k for k in active_keys if k in round_rows and k not in survivor_keys]
            for k in eliminated:
                elim_row = round_rows[k]
                phase_tag = f"sha_round_{round_idx}"
                save_to_db(
                    db, elim_row, experiment_id, "sha_probe", hourly_rate, cloud_meta,
                    profiler_phase=phase_tag, git_commit=git_commit,
                    sha_round=round_idx, sha_eliminated=1,
                )

            new_active_by_wl[wl] = [k for k in active_keys if k in survivor_keys]

            def _score_for_log(r):
                is_jit = r["engine"] in _JIT_ENGINES
                ewms = jax_warmup_ms.get((r["workload"], r["engine"]), 0.0) if is_jit else 0.0
                return _probe_score(r, jit_correction=is_jit, extra_warmup_ms=ewms)

            log_fn(f"  {wl}: {len(wl_rows)} active -> {len(new_active_by_wl.get(wl, []))} survivors")
            for r in survivors_rows[:cap]:
                log_fn(f"    [SHA-KEEP] {r['engine']}/{r['ad_mode']}  "
                       f"rt={r['mean_runtime_ms']:.2f}ms  score={_score_for_log(r):.2f}")
            for k in eliminated:
                r = round_rows[k]
                log_fn(f"    [SHA-ELIM] {r['engine']}/{r['ad_mode']}  "
                       f"rt={r['mean_runtime_ms']:.2f}ms  (CI-pruned)")

        active_by_wl = new_active_by_wl

    # ── Fit scaling laws using ALL available rounds' data (OLS n≥2) ──────────
    # Collect (M, runtime_ms) pairs per config across every round it appeared.
    scaling_laws: Dict[Tuple[str, str, str], Tuple[float, float]] = {}
    _points_per_key: Dict[Tuple[str, str, str], Tuple[List[int], List[float]]] = {}
    for r_idx, round_rows in enumerate(rows_by_round):
        M_r = m_levels[r_idx]
        for k, row in round_rows.items():
            ms_list, t_list = _points_per_key.setdefault(k, ([], []))
            ms_list.append(M_r)
            t_list.append(row["mean_runtime_ms"])
    for k, (ms_list, t_list) in _points_per_key.items():
        fit = _fit_scaling_law(ms_list, t_list)
        if fit:
            scaling_laws[k] = fit

    # ── Final selected keys: all survivors from last active round ────────────
    selected_keys_sha: List[Tuple[str, str, str]] = []
    for keys in active_by_wl.values():
        selected_keys_sha.extend(keys)

    # ── Crossover reinstatement guard ────────────────────────────────────────
    # Problem: JIT engines (JAX) are expensive at small M but have a shallower
    # compute slope than CPU once JIT is amortised.  SHA may prune JAX at
    # round-0 because its raw runtime looks large, then miss it at large M.
    #
    # Fix: after SHA finishes, for each workload:
    #   1. For every eliminated config that has a scaling-law fit (≥2 points),
    #      predict its runtime at max_M.
    #   2. If the prediction beats the current winner at max_M (within a
    #      CROSSOVER_MARGIN tolerance), reinstate the config.
    #   3. Also reinstate configs where round-0 JIT-correction-adjusted slope
    #      predicts they would win (single-point estimate for round-0-only elim).
    CROSSOVER_MARGIN = 1.10   # 10% tolerance — avoids reinstating near-ties
    max_M_sha = m_levels[-1]

    # Collect all eliminated keys (appeared in round-0 but not in final selection)
    all_round0_keys = set(rows_by_round[0].keys()) if rows_by_round else set()
    selected_set = set(selected_keys_sha)

    # Group by workload
    elim_by_wl: Dict[str, List[Tuple[str, str, str]]] = {}
    for k in all_round0_keys:
        wl = k[0]
        if k not in selected_set:
            elim_by_wl.setdefault(wl, []).append(k)

    reinstated: List[Tuple[str, str, str]] = []
    for wl, elim_keys in elim_by_wl.items():
        # Find winner(s) for this workload
        wl_selected = [k for k in selected_set if k[0] == wl]
        if not wl_selected:
            continue

        # Predict winner runtime at max_M
        winner_rt_at_max: Optional[float] = None
        for wk in wl_selected:
            if wk in scaling_laws:
                alpha_w, beta_w = scaling_laws[wk]
                pred_w = alpha_w * max_M_sha + beta_w
                if pred_w > 0 and (winner_rt_at_max is None or pred_w < winner_rt_at_max):
                    winner_rt_at_max = pred_w
        if winner_rt_at_max is None:
            # Fallback: use last measured runtime of the winner
            for wk in wl_selected:
                for r_idx in reversed(range(len(rows_by_round))):
                    if wk in rows_by_round[r_idx]:
                        winner_rt_at_max = rows_by_round[r_idx][wk]["mean_runtime_ms"]
                        break
                if winner_rt_at_max is not None:
                    break
        if winner_rt_at_max is None or winner_rt_at_max <= 0:
            continue

        for ek in elim_keys:
            # Predict eliminated config runtime at max_M
            pred_elim: Optional[float] = None
            if ek in scaling_laws:
                alpha_e, beta_e = scaling_laws[ek]
                pred_elim = alpha_e * max_M_sha + beta_e
            else:
                # Single-point JIT-corrected slope estimate from round-0
                row0 = rows_by_round[0].get(ek) if rows_by_round else None
                if row0 is not None and m_levels[0] > 0:
                    rt0 = row0["mean_runtime_ms"]
                    # Apply JIT correction for JIT engines
                    is_jit = ek[1] in _JIT_ENGINES
                    if is_jit:
                        jit_ms = jax_warmup_ms.get((ek[0], ek[1]), 0.0)
                        rt0 = max(rt0 - jit_ms, rt0 * 0.1)
                    # Linear prediction: t = (rt0 / M0) * max_M (zero-intercept)
                    pred_elim = (rt0 / m_levels[0]) * max_M_sha

            if pred_elim is None or pred_elim <= 0:
                continue

            if pred_elim < winner_rt_at_max * CROSSOVER_MARGIN:
                reinstated.append(ek)
                selected_keys_sha.append(ek)
                selected_set.add(ek)
                log_fn(f"  [SHA-REINSTATE] {ek[1]}/{ek[2]}  "
                       f"pred@M={max_M_sha:,}={pred_elim:.1f}ms  "
                       f"vs winner={winner_rt_at_max:.1f}ms  (crossover detected)")

    log_fn(f"\nSHA completed: {len(selected_keys_sha)} configs selected across all workloads"
           f"{f'  ({len(reinstated)} reinstated via crossover check)' if reinstated else ''}")
    log_fn(f"Scaling laws fitted: {len(scaling_laws)}")
    return all_sha_rows, scaling_laws, selected_keys_sha


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
    sha_round: Optional[int] = None,
    sha_eliminated: Optional[int] = None,
    scaling_law_alpha: Optional[float] = None,
    scaling_law_beta: Optional[float] = None,
    extrapolated_runtime_ms: Optional[float] = None,
    extrapolation_error_pct: Optional[float] = None,
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
        blas_backend=meta.get("blas_backend"),
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
        sha_round=sha_round,
        sha_eliminated=sha_eliminated,
        scaling_law_alpha=scaling_law_alpha,
        scaling_law_beta=scaling_law_beta,
        extrapolated_runtime_ms=extrapolated_runtime_ms,
        extrapolation_error_pct=extrapolation_error_pct,
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
                  sha_round, sha_eliminated,
                  scaling_law_alpha, scaling_law_beta,
                  extrapolated_runtime_ms, extrapolation_error_pct,
                  ad_overhead_ratio,
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

    # sha_progression.csv — SHA elimination trajectory + scaling law predictions
    sha_rows = conn.execute(
        """SELECT workload_type, engine, ad_mode, M, instance_type,
                  mean_runtime_ms, std_runtime_ms,
                  sha_round, sha_eliminated,
                  scaling_law_alpha, scaling_law_beta,
                  extrapolated_runtime_ms, extrapolation_error_pct,
                  ad_overhead_ratio, profiler_phase, profiler_decision
           FROM runs
           WHERE experiment_id = ? AND status = 'completed'
             AND sha_round IS NOT NULL
           ORDER BY sha_round, workload_type, engine, ad_mode""",
        (experiment_id,),
    ).fetchall()
    sha_path = export_dir / "sha_progression.csv"
    if sha_rows:
        with open(sha_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sha_rows[0].keys())
            w.writeheader()
            w.writerows(dict(r) for r in sha_rows)
        print(f"  Exported: {sha_path}  ({len(sha_rows)} rows)")

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
    # SHA parameters
    parser.add_argument("--sha-m-levels",        nargs="+", type=int, default=[1_000, 5_000, 25_000],
                        help="M levels for Successive Halving rounds (default: 1000 5000 25000)")
    parser.add_argument("--sha-eta",             type=float, default=2.0,
                        help="Halving rate: ceil(n/eta) survivors per round (default: 2.0)")
    parser.add_argument("--sha-runs-per-level",  nargs="+", type=int, default=[3, 5, 7],
                        help="Repetitions per SHA round (default: 3 5 7)")
    parser.add_argument("--sha-warmup-per-level",nargs="+", type=int, default=[1, 2, 2],
                        help="Warmup runs per SHA round (default: 1 2 2)")
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
        args.sha_m_levels        = [500, 1000]
        args.sha_runs_per_level  = [1, 1]
        args.sha_warmup_per_level= [0, 0]
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

    # Detect BLAS backend (answers RQ1 & RQ2 without extra runs)
    blas_backend: Optional[str] = None
    try:
        import numpy as _np_blas
        cfg_info = {}
        try:
            cfg_info = dict(_np_blas.show_config(mode="dicts").get("blas_opt_info", {}))
        except Exception:
            pass
        libraries = cfg_info.get("libraries", [])
        if any("mkl" in str(lib).lower() for lib in libraries):
            blas_backend = "mkl"
        elif any("openblas" in str(lib).lower() for lib in libraries):
            blas_backend = "openblas"
        else:
            # Fallback: inspect linked libraries via ldd on numpy core
            import subprocess as _sp, os as _os
            np_core = _os.path.dirname(_np_blas.__file__)
            try:
                ldd_out = _sp.check_output(
                    ["find", np_core, "-name", "*.so", "-exec", "ldd", "{}", ";"],
                    stderr=_sp.DEVNULL, timeout=5,
                ).decode()
                if "mkl" in ldd_out.lower():
                    blas_backend = "mkl"
                elif "openblas" in ldd_out.lower():
                    blas_backend = "openblas"
                else:
                    blas_backend = "unknown"
            except Exception:
                blas_backend = "unknown"
    except Exception:
        pass

    # Align SHA m_levels[0] with --m-probe for backward compatibility
    sha_m_levels = list(args.sha_m_levels)
    if sha_m_levels and sha_m_levels[0] != args.m_probe:
        sha_m_levels[0] = args.m_probe
    # Pad runs/warmup lists to match m_levels length
    sha_runs = list(args.sha_runs_per_level)
    sha_warmup = list(args.sha_warmup_per_level)
    while len(sha_runs) < len(sha_m_levels):
        sha_runs.append(sha_runs[-1] if sha_runs else args.runs_probe)
    while len(sha_warmup) < len(sha_m_levels):
        sha_warmup.append(sha_warmup[-1] if sha_warmup else args.warmup_probe)

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
    log("Budget-Aware Profiler vs Naive Grid Search (SHA edition)")
    if args.dry_run:
        log("*** DRY-RUN MODE — tiny M, temp DB ***")
    log(f"Experiment ID : {experiment_id}")
    log(f"Git commit    : {git_commit or 'unknown'}")
    log(f"Workloads     : {args.workloads}")
    log(f"Engines       : {args.engines}")
    log(f"BLAS backend  : {blas_backend or 'unknown'}")
    log(f"SHA M levels  : {sha_m_levels}  eta={args.sha_eta}")
    log(f"SHA runs/lvl  : {sha_runs}  warmup={sha_warmup}")
    log(f"Full M values : {args.m_values}")
    log(f"Full runs     : {args.runs}  warmup={args.warmup}")
    log(f"Old profiler  : top-k={args.top_k}  score_margin={args.score_margin}x")
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
    # Step 2: Successive Halving (SHA) — multi-fidelity probe
    # Round 0 is identical to the old probe, so SHA and old profiler share
    # the same round-0 data for a free side-by-side comparison.
    # -----------------------------------------------------------------------

    # Extra JIT warmup timing for JAX (used by both SHA and old profiler scoring)
    jax_warmup_ms: Dict[Tuple[str, str], float] = {}
    if "jax" in args.engines and _ENGINES.get("jax"):
        for wl in args.workloads:
            if wl not in _ENGINE_WORKLOADS.get("jax", set()):
                continue
            _jit_row = run_one("jax", wl, sha_m_levels[0], "none", 1, 1)
            if _jit_row:
                jax_warmup_ms[(wl, "jax")] = _jit_row["mean_runtime_ms"]

    sha_all_rows, scaling_laws, selected_keys_sha = _successive_halving(
        candidates=candidates,
        m_levels=sha_m_levels,
        runs_per_level=sha_runs,
        warmup_per_level=sha_warmup,
        eta=args.sha_eta,
        jax_warmup_ms=jax_warmup_ms,
        hourly_rate=hourly_rate,
        db=db,
        experiment_id=experiment_id,
        cloud_meta=cloud_meta,
        git_commit=git_commit,
        log_fn=log,
    )

    # Round-0 rows are the cheap probe — extract for old profiler side-by-side
    probe_results: List[Dict[str, Any]] = [
        r for r in sha_all_rows if r["M"] == sha_m_levels[0]
    ]

    log(f"\nSHA probe runs (round 0, M={sha_m_levels[0]}): {len(probe_results)} / {len(candidates)}")

    if args.probe_only:
        log("\n[probe-only mode] Stopping after SHA probe.")
        _write_summary(summary_path, lines)
        return

    # -----------------------------------------------------------------------
    # Step 3a: OLD profiler selection (from round-0 data only)
    # Kept for side-by-side comparison — uses same fixed-margin heuristic
    # -----------------------------------------------------------------------
    log("\n--- OLD PROFILER SELECTION (from SHA round-0 data) ---")

    by_workload: Dict[str, List[Dict[str, Any]]] = {}
    for row in probe_results:
        by_workload.setdefault(row["workload"], []).append(row)

    selected_keys: Dict[str, List[Tuple[str, str, str]]] = {}
    selected_reasons: Dict[Tuple[str, str, str], str] = {}
    pruned_reasons: Dict[Tuple[str, str, str], str] = {}

    for wl, wl_rows in by_workload.items():
        def score(r: Dict[str, Any], _wl: str = wl) -> float:
            is_jit = r["engine"] in _JIT_ENGINES
            ewms = jax_warmup_ms.get((_wl, r["engine"]), 0.0) if is_jit else 0.0
            return _probe_score(r, jit_correction=is_jit, extra_warmup_ms=ewms)

        pareto_probe = compute_pareto_frontier(wl_rows, x_key="mean_runtime_ms", y_key="cost_per_run")
        if not pareto_probe:
            pareto_probe = wl_rows

        pareto_scored = sorted(pareto_probe, key=score)
        best_score_val = score(pareto_scored[0]) if pareto_scored else float("inf")
        margin = args.score_margin
        within_margin = [r for r in pareto_scored if score(r) <= best_score_val * margin]
        top = within_margin[:args.top_k]

        # Diversity guard
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
            if s <= best_score_val * margin and r not in pareto_scored[:args.top_k]:
                parts.append(f"within {margin}x safety margin")
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
                    f"(score={score(r):.2f} vs best={best_score_val:.2f})"
                )

        log(f"\n  {wl}: {len(wl_rows)} probed -> {len(top)} selected (old profiler)")
        for r in top:
            k = _config_key(r)
            log(f"    [OLD-SEL] {r['engine']}/{r['ad_mode']}  score={score(r):.2f}  "
                f"rt={r['mean_runtime_ms']:.2f}ms  reason: {selected_reasons[k]}")

    total_selected = sum(len(v) for v in selected_keys.values())
    profiler_full_configs = total_selected * len(args.m_values)
    runs_saved = total_full_configs - profiler_full_configs

    # -----------------------------------------------------------------------
    # Step 3b: SHA selection summary
    # -----------------------------------------------------------------------
    log("\n--- SHA SELECTION SUMMARY ---")
    sha_selected_by_wl: Dict[str, List[Tuple[str, str, str]]] = {}
    for k in selected_keys_sha:
        wl2, eng, ad = k
        sha_selected_by_wl.setdefault(wl2, []).append(k)
    sha_total_selected = len(selected_keys_sha)
    sha_profiler_full_configs = sha_total_selected * len(args.m_values)
    sha_runs_saved = total_full_configs - sha_profiler_full_configs
    for wl in args.workloads:
        sha_sel = sha_selected_by_wl.get(wl, [])
        log(f"  {wl}: SHA selected {len(sha_sel)} configs:")
        for (wl2, eng, ad) in sha_sel:
            log(f"    [SHA-SEL] {eng}/{ad}")

    # Scaling law report (answers RQ5: Amdahl serial fraction)
    if scaling_laws:
        log("\n--- SCALING LAW FITS (t = alpha*M + beta) ---")
        log(f"  {'config':<35} {'alpha(ms/path)':>15} {'beta(ms)':>10} {'serial_frac@100k':>18}")
        max_M_for_sf = max(args.m_values)
        for k, (alpha, beta) in sorted(scaling_laws.items()):
            wl2, eng, ad = k
            sf = _serial_fraction(alpha, beta, max_M_for_sf)
            sf_str = f"{sf*100:.1f}%" if sf is not None else "n/a"
            warn = " ← STARTUP-DOMINATED" if (sf is not None and sf > 0.3) else ""
            log(f"  {wl2}/{eng}/{ad:<20} {alpha:>15.6f} {beta:>10.2f} {sf_str:>18}{warn}")

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
            ck = (wl, eng, ad)
            print(f"  grid {eng}/{wl}/{ad} M={M} ...", end=" ", flush=True)
            row = run_one(eng, wl, M, ad, args.warmup, args.runs)
            if row is None:
                print("skipped")
                continue
            print(f"{row['mean_runtime_ms']:.2f} ms")
            full_results[(wl, eng, ad, M)] = row

            if not hourly_rate:
                row["cost_per_run"] = _synthetic_cost(row["mean_runtime_ms"], M)

            # Compute extrapolation error for scaling law if this is max_M
            ext_err: Optional[float] = None
            sl_alpha: Optional[float] = None
            sl_beta: Optional[float] = None
            ext_pred: Optional[float] = None
            if ck in scaling_laws and M == max_M:
                sl_alpha, sl_beta = scaling_laws[ck]
                ext_pred = sl_alpha * M + sl_beta
                if row["mean_runtime_ms"] > 0:
                    ext_err = abs(ext_pred - row["mean_runtime_ms"]) / row["mean_runtime_ms"] * 100.0

            # per-vCPU throughput stored in metadata (answers RQ1)
            vc = cloud_meta.get("vcpu_count")
            if vc and vc > 0 and row.get("throughput", 0) > 0:
                row["throughput_per_vcpu"] = row["throughput"] / vc

            is_old_selected = ck in selected_keys.get(wl, [])
            is_sha_selected = ck in selected_keys_sha
            if is_old_selected and is_sha_selected:
                exp_type = "profiler_selected"
                decision = "selected_both"
            elif is_old_selected:
                exp_type = "profiler_selected"
                decision = "selected_old"
            elif is_sha_selected:
                exp_type = "sha_selected"
                decision = "selected_sha"
            else:
                exp_type = "grid_search_full"
                decision = "full_grid_only"
            reason = selected_reasons.get(ck) if is_old_selected else pruned_reasons.get(ck)
            save_to_db(
                db, row, experiment_id, exp_type, hourly_rate, cloud_meta,
                profiler_phase="full", profiler_decision=decision,
                profiler_reason=reason, git_commit=git_commit,
                scaling_law_alpha=sl_alpha, scaling_law_beta=sl_beta,
                extrapolated_runtime_ms=ext_pred, extrapolation_error_pct=ext_err,
            )

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
    # Step 6: Pareto overlap — evaluate BOTH old profiler and SHA
    # -----------------------------------------------------------------------
    log("\n--- PROFILER vs PARETO OVERLAP ---")

    total_pareto    = 0
    total_recovered = 0
    total_missed    = 0
    sha_total_pareto    = 0
    sha_total_recovered = 0
    sha_total_missed    = 0

    for wl in args.workloads:
        sel     = set(selected_keys.get(wl, []))
        sha_sel = set(sha_selected_by_wl.get(wl, []))
        par     = set(pareto_keys_by_workload.get(wl, []))

        recovered     = sel & par
        missed        = par - sel
        sha_recovered = sha_sel & par
        sha_missed    = par - sha_sel

        total_pareto        += len(par)
        total_recovered     += len(recovered)
        total_missed        += len(missed)
        sha_total_pareto    += len(par)
        sha_total_recovered += len(sha_recovered)
        sha_total_missed    += len(sha_missed)

        log(f"\n  {wl}:")
        log(f"    Pareto frontier size          : {len(par)}")
        log(f"    [OLD] selected / recovered / missed : {len(sel)} / {len(recovered)} / {len(missed)}")
        log(f"    [SHA] selected / recovered / missed : {len(sha_sel)} / {len(sha_recovered)} / {len(sha_missed)}")
        if missed:
            log(f"    [OLD] missed: {[f'{e}/{a}' for _, e, a in missed]}")
        if sha_missed:
            log(f"    [SHA] missed: {[f'{e}/{a}' for _, e, a in sha_missed]}")

    # -----------------------------------------------------------------------
    # Step 7: Regret + best-config report (both old profiler and SHA)
    # -----------------------------------------------------------------------
    all_max_M_rows = [r for (w, e, a, m), r in full_results.items() if m == max_M]

    best_by_runtime = min(all_max_M_rows, key=lambda r: r["mean_runtime_ms"]) if all_max_M_rows else None
    cost_rows_valid = [r for r in all_max_M_rows if r.get("cost_per_run", 0) > 0]
    best_by_cost = min(cost_rows_valid, key=lambda r: r["cost_per_run"]) if cost_rows_valid else None

    # Old profiler best
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

    # SHA best
    sha_max_M_rows = [
        r for (w, e, a, m), r in full_results.items()
        if m == max_M and (w, e, a) in selected_keys_sha
    ]
    sha_best = min(sha_max_M_rows, key=lambda r: r["mean_runtime_ms"]) if sha_max_M_rows else None

    sha_runtime_regret: Optional[float] = None
    if sha_best and best_by_runtime and best_by_runtime["mean_runtime_ms"] > 0:
        sha_runtime_regret = (
            (sha_best["mean_runtime_ms"] - best_by_runtime["mean_runtime_ms"])
            / best_by_runtime["mean_runtime_ms"]
        )

    sha_cost_regret: Optional[float] = None
    if sha_best and best_by_cost and best_by_cost.get("cost_per_run", 0) > 0:
        sha_c = sha_best.get("cost_per_run") or _synthetic_cost(sha_best["mean_runtime_ms"], max_M)
        sha_cost_regret = (sha_c - best_by_cost["cost_per_run"]) / best_by_cost["cost_per_run"]

    # AD overhead summary (answers RQ2)
    ad_overhead_rows = [
        r for (w, e, a, m), r in full_results.items()
        if m == max_M and a != "none" and r.get("ad_overhead_ratio") is not None
    ]

    # -----------------------------------------------------------------------
    # Step 8: Spearman rank correlation (probe round-0 → full grid at max_M)
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

    # SHA final-round scores vs full-grid runtimes (better predictor)
    sha_final_m = sha_m_levels[-1] if sha_m_levels else sha_m_levels[0]
    sha_last_round_rows = [r for r in sha_all_rows if r["M"] == sha_final_m]
    sha_scores_corr: List[float] = []
    sha_full_corr: List[float] = []
    for sha_row in sha_last_round_rows:
        k = _config_key(sha_row)
        wl, eng, ad = k
        full_row = full_results.get((wl, eng, ad, max_M))
        if full_row is not None:
            is_jit = eng in _JIT_ENGINES
            ewms = jax_warmup_ms.get((wl, eng), 0.0) if is_jit else 0.0
            sha_scores_corr.append(
                _probe_score(sha_row, jit_correction=is_jit, extra_warmup_ms=ewms)
            )
            sha_full_corr.append(full_row["mean_runtime_ms"])

    sha_rank_corr = _spearman(sha_scores_corr, sha_full_corr)

    # Extrapolation accuracy summary
    ext_errors = [
        r.get("ext_err") for r in [
            {"ext_err": (
                abs((scaling_laws[k][0]*max_M + scaling_laws[k][1]) - full_results[(k[0],k[1],k[2],max_M)]["mean_runtime_ms"])
                / full_results[(k[0],k[1],k[2],max_M)]["mean_runtime_ms"] * 100.0
            )}
            for k in scaling_laws
            if (k[0],k[1],k[2],max_M) in full_results
               and full_results[(k[0],k[1],k[2],max_M)]["mean_runtime_ms"] > 0
        ]
        if r.get("ext_err") is not None
    ]
    mean_ext_err = (sum(ext_errors) / len(ext_errors)) if ext_errors else None

    # -----------------------------------------------------------------------
    # Step 9: Side-by-side summary
    # -----------------------------------------------------------------------
    pct_saved     = 100.0 * runs_saved / total_full_configs if total_full_configs > 0 else 0.0
    sha_pct_saved = 100.0 * sha_runs_saved / total_full_configs if total_full_configs > 0 else 0.0
    pct_recovered     = 100.0 * total_recovered / total_pareto if total_pareto > 0 else 0.0
    sha_pct_recovered = 100.0 * sha_total_recovered / sha_total_pareto if sha_total_pareto > 0 else 0.0

    # SHA total probe budget (all rounds combined)
    sha_probe_paths = sum(
        r["M"] for r in sha_all_rows
    )
    grid_paths = sum(M * total_grid_configs for M in args.m_values)
    sha_budget_savings_pct = 100.0 * (1.0 - sha_probe_paths / grid_paths) if grid_paths > 0 else 0.0

    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"Experiment ID                       : {experiment_id}")
    log(f"Git commit                          : {git_commit or 'unknown'}")
    log(f"Instance                            : {instance_type or 'local'}")
    log(f"Region / zone                       : {region or 'n/a'} / {zone or 'n/a'}")
    log(f"BLAS backend                        : {blas_backend or 'unknown'}")
    log("")
    log(f"Total grid configurations           : {total_grid_configs}")
    log(f"Number of full-grid configurations  : {total_full_configs}")
    log("")
    log(f"{'Metric':<45} {'Old Profiler':>15} {'SHA':>15}")
    log(f"  {'-'*73}")
    log(f"  {'Deployed full runs selected':<43} {profiler_full_configs:>15} {sha_profiler_full_configs:>15}")
    log(f"  {'Full runs saved vs naive grid':<43} {runs_saved:>14} {sha_runs_saved:>14}")
    log(f"  {'% runs saved':<43} {pct_saved:>14.1f}% {sha_pct_saved:>14.1f}%")
    log(f"  {'Pareto recovery':<43} {pct_recovered:>14.1f}% {sha_pct_recovered:>14.1f}%")
    _old_rr = f"{runtime_regret*100:+.1f}%" if runtime_regret is not None else "n/a"
    _sha_rr = f"{sha_runtime_regret*100:+.1f}%" if sha_runtime_regret is not None else "n/a"
    log(f"  {'Runtime regret vs grid best':<43} {_old_rr:>15} {_sha_rr:>15}")
    _old_cr = f"{cost_regret*100:+.1f}%" if cost_regret is not None else "n/a"
    _sha_cr = f"{sha_cost_regret*100:+.1f}%" if sha_cost_regret is not None else "n/a"
    log(f"  {'Cost regret vs grid best':<43} {_old_cr:>15} {_sha_cr:>15}")
    _old_sp = f"{rank_corr:.3f}" if rank_corr is not None else "n/a"
    _sha_sp = f"{sha_rank_corr:.3f}" if sha_rank_corr is not None else "n/a"
    log(f"  {'Spearman ρ (probe → full rank)':<43} {_old_sp:>15} {_sha_sp:>15}")
    log(f"  {'SHA budget savings (probe vs grid)':<43} {'n/a':>15} {sha_budget_savings_pct:>14.1f}%")
    if mean_ext_err is not None:
        log(f"  {'Mean scaling-law extrap. error':<43} {'n/a':>15} {mean_ext_err:>14.1f}%")
    log("")
    if best_by_runtime:
        r = best_by_runtime
        log(f"Full-grid best by runtime  : {r['engine']}/{r['workload']}/{r['ad_mode']}  "
            f"{r['mean_runtime_ms']:.2f} ms at M={max_M:,}")
    if profiler_best:
        r = profiler_best
        log(f"Old profiler best          : {r['engine']}/{r['workload']}/{r['ad_mode']}  "
            f"{r['mean_runtime_ms']:.2f} ms at M={max_M:,}")
    if sha_best:
        r = sha_best
        log(f"SHA best                   : {r['engine']}/{r['workload']}/{r['ad_mode']}  "
            f"{r['mean_runtime_ms']:.2f} ms at M={max_M:,}")
    if ad_overhead_rows:
        log("")
        log("AD overhead (RQ2) at max M:")
        for r in sorted(ad_overhead_rows, key=lambda x: x.get("ad_overhead_ratio", 0)):
            ratio = r.get("ad_overhead_ratio", 0)
            log(f"  {r['engine']}/{r['workload']}/{r['ad_mode']:<12} overhead={ratio:.2f}x")
    log("")
    log("SHA selected configurations:")
    for wl in args.workloads:
        for (wl2, eng, ad) in sha_selected_by_wl.get(wl, []):
            log(f"  {wl2}/{eng}/{ad}")
    if sha_total_missed > 0:
        log("")
        log("Missed Pareto configurations [SHA]:")
        for wl in args.workloads:
            sha_sel2 = set(sha_selected_by_wl.get(wl, []))
            par2 = set(pareto_keys_by_workload.get(wl, []))
            for k in (par2 - sha_sel2):
                _, eng, ad = k
                log(f"  {wl}/{eng}/{ad}  (in full-grid Pareto, not selected by SHA)")
    if total_missed > 0:
        log("")
        log("Missed Pareto configurations [Old profiler]:")
        for wl in args.workloads:
            sel2 = set(selected_keys.get(wl, []))
            par2 = set(pareto_keys_by_workload.get(wl, []))
            for k in (par2 - sel2):
                _, eng, ad = k
                log(f"  {wl}/{eng}/{ad}")
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
