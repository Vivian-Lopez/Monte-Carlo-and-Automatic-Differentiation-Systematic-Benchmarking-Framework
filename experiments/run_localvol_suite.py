"""
Local Volatility Benchmark Suite  —  LV-0 through LV-10
=========================================================
Implements the full battery of local-vol experiments described in the
experimental plan.  Each experiment writes results to the shared SQLite
database (results/benchmarks.db) under a unique experiment_type tag so
results can be queried, filtered, and re-run independently.

Experiment IDs and DB experiment_type tags
-------------------------------------------
  LV-0   lv_smoke_test           All engines, M=1k, N=252.
                                  Not reported as a main result.
  LV-1   lv_engine_scalability   NumPy, JAX, C++, Rust; no-AD; M sweep.
                                  Primary runtime / language comparison.
  LV-2   lv_ad_m_scaling         JAX; none/forward/reverse; M sweep.
                                  Primary AD overhead result.
  LV-3   lv_n_scaling            JAX, C++, Rust; no-AD; N sweep (M=50k).
                                  Time-discretisation scaling.
  LV-4   lv_ad_n_scaling         JAX; none/forward/reverse; N sweep (M=25k).
                                  AD overhead vs simulation length.
  LV-5   lv_memory_pressure      JAX; all AD modes; M x N grid.
                                  Resource-utilisation profile.
  LV-6   lv_thread_strong        C++/Rust/JAX; no-AD; T=1,2,4; M=100k fixed.
                                  CPU strong-scaling on local vol.
  LV-7   lv_thread_weak          C++/Rust/JAX; no-AD; T=1,2,4; M=50k*T.
                                  CPU weak-scaling on local vol.
  LV-8   lv_surface_sensitivity  JAX + fastest compiled; flat/equity-skew/
                                  strong-smile surface presets; M=50k.
                                  Extensible workload demonstration.
  LV-9   lv_timing_stability     JAX (none+reverse), C++, Rust; 20 runs;
                                  M=100k.  Coefficient-of-variation report.
  LV-10  lv_stress_limit         Compiled no-AD + JAX reverse; M=500k,1M.
                                  Practical resource-limit probe.

Thread scaling (LV-6, LV-7)
-----------------------------
Uses subprocess-per-cell isolation so each cell starts with a fresh
OMP / Rayon thread pool.  The cell script is run_lv_thread_cell.py.

Usage
-----
    # Run all experiments (default)
    python experiments/run_localvol_suite.py

    # Run selected experiments
    python experiments/run_localvol_suite.py --experiments LV0 LV1 LV2

    # Thread-scaling only, custom thread list
    python experiments/run_localvol_suite.py --experiments LV6 LV7 --threads 1 2 4

    # Smoke test only
    python experiments/run_localvol_suite.py --experiments LV0

Notes
-----
  * JAX local-vol greeks returned as dict with keys:
    delta, d_a0, d_a1, d_a2, d_b1, d_sigma_min.
    Stored as: greek_delta = delta, greek_vega = d_sigma_min.
  * Cloud cost columns are NULL (local run, no GCP metadata).
  * The smoke test (LV-0) is stored in the DB but should be excluded from
    reported results (use WHERE experiment_type != 'lv_smoke_test').
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmarking.core.config import EuropeanLocalVolConfig
from benchmarking.runner.runner import BenchmarkRunner
from benchmarking.storage.database import BenchmarkDB

# ---------------------------------------------------------------------------
# Engine loading — optional compiled engines silently skipped if absent
# ---------------------------------------------------------------------------
try:
    from benchmarking.workloads.mc_cpp import CPPMonteCarloEngine
    _CPP: object = CPPMonteCarloEngine()
except Exception:
    _CPP = None

try:
    from benchmarking.workloads.mc_rust import RustMonteCarloEngine
    _RUST: object = RustMonteCarloEngine()
except Exception:
    _RUST = None

from benchmarking.workloads.mc_cpu import CPUMonteCarloEngine
from benchmarking.workloads.mc_jax import JAXMonteCarloEngine

_CPU = CPUMonteCarloEngine()
_JAX = JAXMonteCarloEngine()

_ENGINE_META: dict[str, tuple[str, str]] = {
    "cpu":  ("python", "numpy"),
    "jax":  ("python", "xla"),
    "cpp":  ("cpp",    "openmp"),
    "rust": ("rust",   "rayon"),
}

_ENGINE_INSTANCES: dict[str, object] = {
    "cpu":  _CPU,
    "jax":  _JAX,
    "cpp":  _CPP,
    "rust": _RUST,
}


def _avail(name: str) -> object:
    """Return engine instance if available, else None."""
    return _ENGINE_INSTANCES.get(name)


# ---------------------------------------------------------------------------
# Local-vol surface presets for LV-8
# ---------------------------------------------------------------------------
_sigma_min_default = 0.01
# a0 value giving constant sigma ≈ 0.20 under softplus parametrisation:
#   sigma_min + softplus(a0) = 0.20  =>  a0 = log(expm1(0.20 - sigma_min))
_a0_flat = math.log(math.expm1(0.20 - _sigma_min_default))

SURFACE_PRESETS: dict[str, list[float]] = {
    "flat":         [_a0_flat,  0.00,  0.00, 0.00],   # constant σ = 0.20
    "equity-skew":  [_a0_flat, -0.15,  0.05, 0.00],   # negative skew in log-moneyness
    "strong-smile": [_a0_flat,  0.00,  0.35, 0.00],   # symmetric convex smile
}

# ---------------------------------------------------------------------------
# Cell script for LV-6 / LV-7 thread experiments
# ---------------------------------------------------------------------------
_LV_THREAD_CELL = Path(__file__).parent / "run_lv_thread_cell.py"


# ---------------------------------------------------------------------------
# In-process benchmark cell helper
# ---------------------------------------------------------------------------
def _run_cell(
    *,
    experiment_id: str,
    experiment_type: str,
    eng_name: str,
    engine: object,
    config: EuropeanLocalVolConfig,
    ad_mode: str,
    warmup: int,
    runs: int,
    db: BenchmarkDB,
    rows: list,
    label: str = "",
) -> None:
    """Run one benchmark cell in-process, store result to DB, append to rows."""
    tag = label or f"{eng_name:<4} / {ad_mode:<8} / M={config.M:>8,} / N={config.N}"
    runner = BenchmarkRunner(
        engine,  # type: ignore[arg-type]
        name=f"{eng_name}/lv/{ad_mode}/M{config.M}/N{config.N}",
    )
    try:
        res = runner.run(config, num_warmup=warmup, num_runs=runs, ad_mode=ad_mode)
    except Exception as exc:
        print(f"  [{tag}] FAILED: {exc}")
        return

    mean_ms    = res.mean_runtime * 1000
    std_ms     = res.std_runtime  * 1000
    min_ms     = res.min_runtime  * 1000
    max_ms     = res.max_runtime  * 1000
    throughput = res.throughput_paths_per_sec
    overhead   = res.ad_overhead_ratio
    price      = res.result
    greeks     = res.greeks   # dict or None
    env        = res.metadata
    language, backend = _ENGINE_META.get(eng_name, ("python", "cpu"))

    # Map local-vol greeks to the three dedicated DB columns (best effort).
    # Full sensitivity vector is preserved in greeks_json (built internally
    # by store_run_full from the three scalar fields we pass here).
    g_delta = greeks.get("delta")       if greeks else None
    g_vega  = greeks.get("d_sigma_min") if greeks else None   # closest to vega

    db.store_run_full(
        config_dict=config.to_dict(),
        engine=eng_name,
        ad_mode=ad_mode,
        experiment_id=experiment_id,
        experiment_type=experiment_type,
        mean_runtime_ms=mean_ms,
        std_runtime_ms=std_ms,
        min_runtime_ms=min_ms,
        max_runtime_ms=max_ms,
        throughput_paths_per_sec=throughput,
        baseline_mean_ms=res.baseline_mean_ms,
        ad_overhead_ratio=overhead,
        result_value=price,
        greek_delta=g_delta,
        greek_vega=g_vega,
        memory_peak_mb=res.memory_peak_mb,
        language=language,
        backend=backend,
        cpu_model=env.get("cpu_model"),
        cpu_architecture=env.get("cpu_architecture"),
        cpu_count=env.get("cpu_count"),
        memory_gb=env.get("memory_gb"),
        platform=env.get("platform"),
        python_version=env.get("python_version"),
        numpy_version=env.get("numpy_version"),
        jax_version=env.get("jax_version"),
    )

    oh_str = f"{overhead:.2f}x" if ad_mode != "none" else "  1.00x"
    print(
        f"  [{tag}]  {mean_ms:>9.3f} ms ± {std_ms:.3f}  "
        f"overhead {oh_str}  price {price:.5f}  "
        f"{throughput / 1e6:.2f}M/s  mem {res.memory_peak_mb:.1f}MB"
    )

    rows.append({
        "experiment_type": experiment_type,
        "eng":      eng_name,
        "ad_mode":  ad_mode,
        "M":        config.M,
        "N":        config.N,
        "mean_ms":  mean_ms,
        "std_ms":   std_ms,
        "throughput": throughput,
        "overhead": overhead,
        "price":    price,
        "mem_mb":   res.memory_peak_mb,
    })


# ---------------------------------------------------------------------------
# Subprocess environment builder (for thread-scaling cells)
# ---------------------------------------------------------------------------
def _build_thread_env(n_threads: int) -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"]      = str(n_threads)
    env["OMP_DYNAMIC"]          = "FALSE"
    env["OMP_PROC_BIND"]        = "close"
    env["OMP_PLACES"]           = "cores"
    env["RAYON_NUM_THREADS"]    = str(n_threads)
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"]      = "1"
    env["NUMEXPR_NUM_THREADS"]  = "1"
    env["JAX_ENABLE_X64"]       = "1"  # ensure float64 in subprocess cells
    # Remove stale XLA thread flag that is no longer valid in JAX 0.9.x
    if "XLA_FLAGS" in env:
        cleaned = re.sub(
            r"--xla_cpu_multi_thread_eigen_intra_op_parallelism_threads=\d+",
            "", env["XLA_FLAGS"],
        ).strip()
        if cleaned:
            env["XLA_FLAGS"] = cleaned
        else:
            del env["XLA_FLAGS"]
    return env


def _run_thread_cell(
    *,
    engine: str,
    n_threads: int,
    M: int,
    N: int,
    regime: str,
    experiment_id: str,
    runs: int,
    warmup: int,
) -> Optional[dict]:
    """
    Launch one subprocess cell for LV thread-scaling.

    Returns the parsed JSON result dict on success, or None on failure.
    The subprocess writes directly to the DB; the caller only needs the
    dict for summary tables.
    """
    cmd = [
        sys.executable, str(_LV_THREAD_CELL),
        "--engine",        engine,
        "--threads",       str(n_threads),
        "--M",             str(M),
        "--N",             str(N),
        "--regime",        regime,
        "--experiment-id", experiment_id,
        "--runs",          str(runs),
        "--warmup",        str(warmup),
    ]
    env = _build_thread_env(n_threads)

    print(f"  Running [{engine}, T={n_threads}, M={M:,}, N={N}, {regime}] ...",
          end="", flush=True)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=900,
        )
    except subprocess.TimeoutExpired:
        print("  TIMEOUT (>900 s)")
        return None
    except Exception as exc:
        print(f"  LAUNCH ERROR: {exc}")
        return None

    # Forward non-noise stderr lines to the console
    if proc.stderr:
        for line in proc.stderr.strip().splitlines():
            if any(t in line for t in ("I0", "W0", "E0", "absl", "xla", "XLA",
                                        "jax", "JAX", "WARNING")):
                continue
            print(f"\n    [stderr] {line}", end="")

    # Parse the single JSON summary line emitted by the cell script
    for line in proc.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
                if not data.get("success"):
                    print(f"  FAILED: {data.get('error', 'unknown')}")
                    return None
                ms = data["mean_ms"]
                tp = data["throughput"]
                print(f"  => {ms:.3f} ms  {tp / 1e6:.2f}M paths/s")
                return data
            except json.JSONDecodeError:
                pass

    print(f"  No JSON output (exit {proc.returncode})")
    return None


def _print_thread_speedup(
    thread_rows: list[dict],
    threads: list[int],
    label: str,
) -> None:
    """Print throughput speedup table relative to T=1 for each engine."""
    if not thread_rows:
        return
    by_engine: dict[str, dict[int, float]] = defaultdict(dict)
    for r in thread_rows:
        if r:
            by_engine[r["engine"]][r["threads"]] = r["throughput"]
    if not by_engine:
        return
    print()
    print(f"  {label} speedup (throughput relative to T=1):")
    print("  " + "-" * 48)
    header = "  " + f"{'Engine':<6}  " + "  ".join(f"T={t}" for t in threads)
    print(header)
    print("  " + "-" * 48)
    for eng, tp_map in sorted(by_engine.items()):
        base = tp_map.get(threads[0])
        if not base:
            continue
        speedups = []
        for t in threads:
            tp = tp_map.get(t)
            speedups.append(f"{tp / base:.2f}x" if tp is not None else "  n/a")
        print("  " + f"{eng:<6}  " + "  ".join(f"{s:>6}" for s in speedups))


# ===========================================================================
# LV-0 — Smoke test
# ===========================================================================
def run_lv0(db: BenchmarkDB, rows: list, warmup: int = 1, runs: int = 1) -> None:
    """
    LV-0: Sanity check that all engines and DB path work.
    Not reported as a main experimental result.
    """
    exp_id   = str(uuid.uuid4())
    exp_type = "lv_smoke_test"
    config   = EuropeanLocalVolConfig(M=1_000, N=252)

    print("\n" + "=" * 72)
    print("  LV-0  Smoke Test  (M=1,000, N=252 — not a reported result)")
    print("=" * 72)

    engines = [("cpu", _CPU), ("jax", _JAX)]
    if _CPP  is not None: engines.append(("cpp",  _CPP))
    if _RUST is not None: engines.append(("rust", _RUST))

    for eng_name, engine in engines:
        modes = ["none", "forward", "reverse"] if eng_name == "jax" else ["none"]
        for ad_mode in modes:
            _run_cell(
                experiment_id=exp_id, experiment_type=exp_type,
                eng_name=eng_name, engine=engine, config=config,
                ad_mode=ad_mode, warmup=warmup, runs=runs,
                db=db, rows=rows,
            )

    print(f"\n  [LV-0] experiment_id: {exp_id}")


# ===========================================================================
# LV-1 — Main engine + M scalability
# ===========================================================================
def run_lv1(db: BenchmarkDB, rows: list, warmup: int = 3, runs: int = 7) -> None:
    """
    LV-1: All engines, no AD, M sweep.
    Primary runtime / language comparison result.
    """
    exp_id   = str(uuid.uuid4())
    exp_type = "lv_engine_scalability"
    M_values = [10_000, 25_000, 50_000, 100_000, 250_000, 500_000]
    N        = 252

    print("\n" + "=" * 72)
    print("  LV-1  Engine + M Scalability  (no AD, N=252)")
    print(f"  M values : {M_values}")
    print(f"  Runs: {runs}  Warmup: {warmup}")
    print("=" * 72)

    engines = [("cpu", _CPU), ("jax", _JAX)]
    if _CPP  is not None: engines.append(("cpp",  _CPP))
    if _RUST is not None: engines.append(("rust", _RUST))

    for M in M_values:
        config = EuropeanLocalVolConfig(M=M, N=N)
        for eng_name, engine in engines:
            _run_cell(
                experiment_id=exp_id, experiment_type=exp_type,
                eng_name=eng_name, engine=engine, config=config,
                ad_mode="none", warmup=warmup, runs=runs,
                db=db, rows=rows,
            )

    print(f"\n  [LV-1] experiment_id: {exp_id}")


# ===========================================================================
# LV-2 — JAX AD M-scaling
# ===========================================================================
def run_lv2(db: BenchmarkDB, rows: list, warmup: int = 3, runs: int = 7) -> None:
    """
    LV-2: JAX only, three AD modes, M sweep.
    Primary AD overhead result.
    """
    exp_id   = str(uuid.uuid4())
    exp_type = "lv_ad_m_scaling"
    M_values = [5_000, 10_000, 25_000, 50_000, 100_000, 250_000]
    N        = 252

    print("\n" + "=" * 72)
    print("  LV-2  JAX AD M-Scaling  (none / forward / reverse, N=252)")
    print(f"  M values : {M_values}")
    print(f"  Runs: {runs}  Warmup: {warmup}")
    print("=" * 72)

    for M in M_values:
        config = EuropeanLocalVolConfig(M=M, N=N)
        for ad_mode in ["none", "forward", "reverse"]:
            _run_cell(
                experiment_id=exp_id, experiment_type=exp_type,
                eng_name="jax", engine=_JAX, config=config,
                ad_mode=ad_mode, warmup=warmup, runs=runs,
                db=db, rows=rows,
            )

    print(f"\n  [LV-2] experiment_id: {exp_id}")


# ===========================================================================
# LV-3 — N-scaling (JAX, C++, Rust; M=50k)
# ===========================================================================
def run_lv3(db: BenchmarkDB, rows: list, warmup: int = 3, runs: int = 7) -> None:
    """
    LV-3: JAX, C++, Rust; no AD; N sweep at M=50k.
    Time-discretisation scaling.
    """
    exp_id   = str(uuid.uuid4())
    exp_type = "lv_n_scaling"
    N_values = [12, 52, 126, 252, 504]
    M        = 50_000

    print("\n" + "=" * 72)
    print("  LV-3  N-Scaling  (M=50,000, no AD)")
    print(f"  N values : {N_values}")
    print(f"  Runs: {runs}  Warmup: {warmup}")
    print("=" * 72)

    engines = [("jax", _JAX)]
    if _CPP  is not None: engines.append(("cpp",  _CPP))
    if _RUST is not None: engines.append(("rust", _RUST))

    for N in N_values:
        config = EuropeanLocalVolConfig(M=M, N=N)
        for eng_name, engine in engines:
            _run_cell(
                experiment_id=exp_id, experiment_type=exp_type,
                eng_name=eng_name, engine=engine, config=config,
                ad_mode="none", warmup=warmup, runs=runs,
                db=db, rows=rows,
            )

    print(f"\n  [LV-3] experiment_id: {exp_id}")


# ===========================================================================
# LV-4 — JAX AD N-scaling (M=25k)
# ===========================================================================
def run_lv4(db: BenchmarkDB, rows: list, warmup: int = 3, runs: int = 5) -> None:
    """
    LV-4: JAX, three AD modes, N sweep at M=25k.
    AD overhead as a function of simulation length.
    """
    exp_id   = str(uuid.uuid4())
    exp_type = "lv_ad_n_scaling"
    N_values = [12, 52, 126, 252, 504]
    M        = 25_000

    print("\n" + "=" * 72)
    print("  LV-4  JAX AD N-Scaling  (M=25,000)")
    print(f"  N values : {N_values}")
    print(f"  Runs: {runs}  Warmup: {warmup}")
    print("=" * 72)

    for N in N_values:
        config = EuropeanLocalVolConfig(M=M, N=N)
        for ad_mode in ["none", "forward", "reverse"]:
            _run_cell(
                experiment_id=exp_id, experiment_type=exp_type,
                eng_name="jax", engine=_JAX, config=config,
                ad_mode=ad_mode, warmup=warmup, runs=runs,
                db=db, rows=rows,
            )

    print(f"\n  [LV-4] experiment_id: {exp_id}")


# ===========================================================================
# LV-5 — Memory pressure (JAX, large M × N)
# ===========================================================================
def run_lv5(db: BenchmarkDB, rows: list, warmup: int = 3, runs: int = 5) -> None:
    """
    LV-5: JAX; all AD modes; M × N grid.
    Resource-utilisation profile.  Large cells may fail on memory-limited
    machines — failures are reported and execution continues.
    """
    exp_id   = str(uuid.uuid4())
    exp_type = "lv_memory_pressure"
    M_values = [25_000, 50_000, 100_000, 250_000]
    N_values = [252, 504]

    print("\n" + "=" * 72)
    print("  LV-5  Memory Pressure  (JAX, all AD modes)")
    print(f"  M values : {M_values}")
    print(f"  N values : {N_values}")
    print(f"  Runs: {runs}  Warmup: {warmup}")
    print("=" * 72)

    for N in N_values:
        for M in M_values:
            config = EuropeanLocalVolConfig(M=M, N=N)
            for ad_mode in ["none", "forward", "reverse"]:
                _run_cell(
                    experiment_id=exp_id, experiment_type=exp_type,
                    eng_name="jax", engine=_JAX, config=config,
                    ad_mode=ad_mode, warmup=warmup, runs=runs,
                    db=db, rows=rows,
                )

    print(f"\n  [LV-5] experiment_id: {exp_id}")


# ===========================================================================
# LV-6 — Thread strong scaling (M=100k fixed, T = 1, 2, 4)
# ===========================================================================
def run_lv6(
    db: BenchmarkDB,
    rows: list,
    warmup: int = 2,
    runs: int = 5,
    threads: Optional[list[int]] = None,
) -> None:
    """
    LV-6: C++, Rust, JAX; no AD; strong scaling (M=100k, N=252).
    Subprocess per cell for clean thread-pool isolation.
    """
    if threads is None:
        threads = [1, 2, 4]
    exp_id = str(uuid.uuid4())
    M = 100_000
    N = 252

    print("\n" + "=" * 72)
    print("  LV-6  Thread Strong Scaling  (M=100,000, N=252, no AD)")
    print(f"  Threads: {threads}  Runs: {runs}  Warmup: {warmup}")
    print("=" * 72)

    thread_rows: list[dict] = []
    for eng_name in ("cpp", "rust", "jax"):
        if eng_name != "jax" and _avail(eng_name) is None:
            print(f"  [{eng_name}] not available — skipping")
            continue
        for t in threads:
            data = _run_thread_cell(
                engine=eng_name, n_threads=t, M=M, N=N,
                regime="strong", experiment_id=exp_id,
                runs=runs, warmup=warmup,
            )
            if data:
                thread_rows.append(data)
                rows.append({
                    "experiment_type": "lv_thread_strong",
                    "eng": eng_name, "ad_mode": "none",
                    "M": M, "N": N,
                    "mean_ms":    data["mean_ms"],
                    "std_ms":     data["std_ms"],
                    "throughput": data["throughput"],
                    "overhead":   1.0,
                    "price":      data["price"],
                    "mem_mb":     data.get("memory_peak_mb", 0.0),
                })

    _print_thread_speedup(thread_rows, threads, label="LV-6 strong")
    print(f"\n  [LV-6] experiment_id: {exp_id}")


# ===========================================================================
# LV-7 — Thread weak scaling (base_M=50k per thread, N=252)
# ===========================================================================
def run_lv7(
    db: BenchmarkDB,
    rows: list,
    warmup: int = 2,
    runs: int = 5,
    threads: Optional[list[int]] = None,
    base_M: int = 50_000,
) -> None:
    """
    LV-7: C++, Rust, JAX; no AD; weak scaling (M = base_M × threads, N=252).
    Subprocess per cell for clean thread-pool isolation.
    """
    if threads is None:
        threads = [1, 2, 4]
    exp_id = str(uuid.uuid4())
    N = 252

    print("\n" + "=" * 72)
    print(f"  LV-7  Thread Weak Scaling  (base_M={base_M:,}, N={N}, no AD)")
    print(f"  Threads: {threads}  Runs: {runs}  Warmup: {warmup}")
    print("=" * 72)

    thread_rows: list[dict] = []
    for eng_name in ("cpp", "rust", "jax"):
        if eng_name != "jax" and _avail(eng_name) is None:
            print(f"  [{eng_name}] not available — skipping")
            continue
        for t in threads:
            weak_M = base_M * t
            data = _run_thread_cell(
                engine=eng_name, n_threads=t, M=weak_M, N=N,
                regime="weak", experiment_id=exp_id,
                runs=runs, warmup=warmup,
            )
            if data:
                thread_rows.append(data)
                rows.append({
                    "experiment_type": "lv_thread_weak",
                    "eng": eng_name, "ad_mode": "none",
                    "M": weak_M, "N": N,
                    "mean_ms":    data["mean_ms"],
                    "std_ms":     data["std_ms"],
                    "throughput": data["throughput"],
                    "overhead":   1.0,
                    "price":      data["price"],
                    "mem_mb":     data.get("memory_peak_mb", 0.0),
                })

    _print_thread_speedup(thread_rows, threads, label="LV-7 weak")
    print(f"\n  [LV-7] experiment_id: {exp_id}")


# ===========================================================================
# LV-8 — Surface-shape sensitivity
# ===========================================================================
def run_lv8(db: BenchmarkDB, rows: list, warmup: int = 3, runs: int = 7) -> None:
    """
    LV-8: JAX (none + reverse) and fastest compiled engine (none);
    three surface presets (flat, equity-skew, strong-smile); M=50k, N=252.
    Demonstrates extensible workload definitions and model sensitivity.
    """
    exp_id   = str(uuid.uuid4())
    exp_type = "lv_surface_sensitivity"
    M = 50_000
    N = 252

    # Pick the fastest compiled engine available (prefer C++ then Rust)
    compiled_name: Optional[str] = None
    compiled_eng:  Optional[object] = None
    for name in ("cpp", "rust"):
        if _avail(name) is not None:
            compiled_name = name
            compiled_eng  = _avail(name)
            break

    print("\n" + "=" * 72)
    print("  LV-8  Surface-Shape Sensitivity  (M=50,000, N=252)")
    print(f"  Presets  : {list(SURFACE_PRESETS)}")
    print(f"  Compiled : {compiled_name or 'none (JAX only)'}")
    print(f"  Runs: {runs}  Warmup: {warmup}")
    print("=" * 72)

    for preset_name, theta in SURFACE_PRESETS.items():
        config = EuropeanLocalVolConfig(M=M, N=N, theta=theta)

        for ad_mode in ["none", "reverse"]:
            _run_cell(
                experiment_id=exp_id, experiment_type=exp_type,
                eng_name="jax", engine=_JAX, config=config,
                ad_mode=ad_mode, warmup=warmup, runs=runs,
                db=db, rows=rows,
                label=f"jax /{preset_name:<12}/{ad_mode}",
            )

        if compiled_eng is not None:
            _run_cell(
                experiment_id=exp_id, experiment_type=exp_type,
                eng_name=compiled_name, engine=compiled_eng, config=config,
                ad_mode="none", warmup=warmup, runs=runs,
                db=db, rows=rows,
                label=f"{compiled_name:<4}/{preset_name:<12}/none",
            )

    print(f"\n  [LV-8] experiment_id: {exp_id}")


# ===========================================================================
# LV-9 — Timing stability (20 timed runs)
# ===========================================================================
def run_lv9(db: BenchmarkDB, rows: list, warmup: int = 3, runs: int = 20) -> None:
    """
    LV-9: JAX (none + reverse), C++, Rust; 20 timed runs; M=100k, N=252.
    Reports coefficient of variation to assess benchmark credibility.
    """
    exp_id   = str(uuid.uuid4())
    exp_type = "lv_timing_stability"
    M = 100_000
    N = 252
    config = EuropeanLocalVolConfig(M=M, N=N)

    print("\n" + "=" * 72)
    print(f"  LV-9  Timing Stability  (M={M:,}, N={N}, {runs} timed runs each)")
    print("=" * 72)

    engines = [("jax", _JAX)]
    if _CPP  is not None: engines.append(("cpp",  _CPP))
    if _RUST is not None: engines.append(("rust", _RUST))

    for eng_name, engine in engines:
        modes = ["none", "reverse"] if eng_name == "jax" else ["none"]
        for ad_mode in modes:
            _run_cell(
                experiment_id=exp_id, experiment_type=exp_type,
                eng_name=eng_name, engine=engine, config=config,
                ad_mode=ad_mode, warmup=warmup, runs=runs,
                db=db, rows=rows,
            )

    # Coefficient of variation summary
    lv9_rows = [r for r in rows if r.get("experiment_type") == exp_type]
    if lv9_rows:
        print()
        print("  Coefficient of Variation (std / mean × 100):")
        for r in lv9_rows:
            cv = (r["std_ms"] / r["mean_ms"] * 100.0) if r["mean_ms"] > 0 else float("nan")
            print(f"    [{r['eng']:<4} / {r['ad_mode']:<8}]  "
                  f"mean={r['mean_ms']:.3f} ms  std={r['std_ms']:.4f} ms  CV={cv:.2f}%")

    print(f"\n  [LV-9] experiment_id: {exp_id}")


# ===========================================================================
# LV-10 — Stress limit (M = 500k, 1M)
# ===========================================================================
def run_lv10(db: BenchmarkDB, rows: list, warmup: int = 2, runs: int = 3) -> None:
    """
    LV-10: Fastest compiled no-AD engines + JAX reverse; M = {500k, 1M}, N=252.
    Probes the practical resource limit of the benchmarking machine.
    Large cells may OOM — failures are reported and execution continues.
    """
    exp_id   = str(uuid.uuid4())
    exp_type = "lv_stress_limit"
    M_values = [500_000, 1_000_000]
    N        = 252

    # Fastest compiled no-AD engines (prefer C++ then Rust; CPU as last resort)
    fast_engines: list[tuple[str, object]] = []
    for name in ("cpp", "rust"):
        if _avail(name) is not None:
            fast_engines.append((name, _avail(name)))
    if not fast_engines:
        fast_engines.append(("cpu", _CPU))   # fallback: NumPy

    print("\n" + "=" * 72)
    print("  LV-10  Stress Limit  (M=500k and M=1M, N=252)")
    print(f"  Compiled engines : {[n for n, _ in fast_engines]}")
    print(f"  JAX mode         : reverse")
    print(f"  Runs: {runs}  Warmup: {warmup}")
    print("=" * 72)

    for M in M_values:
        config = EuropeanLocalVolConfig(M=M, N=N)

        # Fast compiled engines: no AD
        for eng_name, engine in fast_engines:
            _run_cell(
                experiment_id=exp_id, experiment_type=exp_type,
                eng_name=eng_name, engine=engine, config=config,
                ad_mode="none", warmup=warmup, runs=runs,
                db=db, rows=rows,
            )

        # JAX reverse-mode AD
        _run_cell(
            experiment_id=exp_id, experiment_type=exp_type,
            eng_name="jax", engine=_JAX, config=config,
            ad_mode="reverse", warmup=warmup, runs=runs,
            db=db, rows=rows,
        )

    print(f"\n  [LV-10] experiment_id: {exp_id}")


# ===========================================================================
# Experiment registry
# ===========================================================================
_EXPERIMENTS: dict[str, tuple[str, object]] = {
    "LV0":  ("LV-0   Smoke test                (M=1k, all engines, 1/1)",      run_lv0),
    "LV1":  ("LV-1   Engine + M scalability    (no AD, N=252, 7/3)",           run_lv1),
    "LV2":  ("LV-2   JAX AD M-scaling          (none/fwd/rev, N=252, 7/3)",    run_lv2),
    "LV3":  ("LV-3   N-scaling                 (M=50k, no AD, 7/3)",           run_lv3),
    "LV4":  ("LV-4   JAX AD N-scaling          (M=25k, 5/3)",                  run_lv4),
    "LV5":  ("LV-5   Memory pressure           (JAX, M×N grid, 5/3)",          run_lv5),
    "LV6":  ("LV-6   Thread strong scaling     (M=100k, subprocess, 5/2)",     run_lv6),
    "LV7":  ("LV-7   Thread weak scaling       (base_M=50k, subprocess, 5/2)", run_lv7),
    "LV8":  ("LV-8   Surface-shape sensitivity (presets, M=50k, 7/3)",         run_lv8),
    "LV9":  ("LV-9   Timing stability          (20 runs, M=100k)",             run_lv9),
    "LV10": ("LV-10  Stress limit              (M=500k,1M, 3/2)",              run_lv10),
}


# ===========================================================================
# Entry point
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local Volatility Benchmark Suite (LV-0 through LV-10)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Run all experiments\n"
            "  python experiments/run_localvol_suite.py\n\n"
            "  # Run only selected experiments\n"
            "  python experiments/run_localvol_suite.py --experiments LV0 LV1 LV2\n\n"
            "  # Thread scaling only, custom thread list\n"
            "  python experiments/run_localvol_suite.py --experiments LV6 LV7 --threads 1 2 4\n"
        ),
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(_EXPERIMENTS),
        metavar="EXP",
        default=list(_EXPERIMENTS),
        help=(
            "Experiments to run (default: all). "
            "Choices: " + " ".join(_EXPERIMENTS)
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        metavar="T",
        help="Thread counts for LV-6 and LV-7 (default: 1 2 4)",
    )
    args = parser.parse_args()

    db   = BenchmarkDB()
    rows: list = []

    available_engines = ["cpu", "jax"]
    if _CPP  is not None: available_engines.append("cpp")
    if _RUST is not None: available_engines.append("rust")

    print()
    print("=" * 72)
    print("  Local Volatility Benchmark Suite")
    print("=" * 72)
    print(f"  Available engines : {available_engines}")
    print(f"  Experiments       : {args.experiments}")
    print(f"  Thread counts     : {args.threads}  (LV-6 / LV-7 only)")
    print(f"  DB                : {db.db_path}")
    print()

    for exp_key in args.experiments:
        description, run_fn = _EXPERIMENTS[exp_key]
        print(f"\n  >>> {description}")
        kwargs: dict = {}
        if exp_key in ("LV6", "LV7"):
            kwargs["threads"] = args.threads
        run_fn(db=db, rows=rows, **kwargs)  # type: ignore[call-arg]

    # Final summary
    print()
    print("=" * 72)
    print("  Suite complete")
    print("=" * 72)
    if rows:
        print("  Rows written by experiment_type:")
        for et, cnt in Counter(r.get("experiment_type", "?") for r in rows).most_common():
            print(f"    {et:<35} {cnt:>4} rows")
    print(f"\n  DB: {db.db_path}")
    print()


if __name__ == "__main__":
    main()
