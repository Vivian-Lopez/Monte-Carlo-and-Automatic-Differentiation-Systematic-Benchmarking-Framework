"""
Single-cell runner for the thread-scalability experiment.

This script is NOT meant to be called directly by users.  It is launched
as a fresh subprocess by run_thread_scalability.py for every
(engine, thread_count, M, regime) combination.

Why a subprocess per cell?
--------------------------
XLA reads its thread-pool size from XLA_FLAGS *before* any JAX computation
runs.  Once XLA has spawned its thread pool the flag is ignored for the
remainder of the process lifetime.  The same applies to OpenMP: the thread
count is fixed the first time a parallel region is entered.

Running all cells in one process and mutating os.environ between cells
therefore produces unreliable results — the first cell's thread count
"wins" for every subsequent cell.

Subprocess isolation guarantees that every cell starts with a clean process
whose XLA/OpenMP thread pool has never been initialised.

The parent (run_thread_scalability.py) sets the environment variables
*before* launching this process, so by the time Python/JAX imports happen
the correct value is already in os.environ.

Output
------
Prints exactly one JSON object to stdout on success, used by the
orchestrator to build summary tables without relying on shared state.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import psutil as _psutil
except ImportError as _e:
    raise SystemExit(
        "psutil is required for thread diagnostics.\n"
        "Install it with:  pip install psutil\n"
        f"Original error: {_e}"
    ) from _e

# ---------------------------------------------------------------------------
# These imports happen AFTER the parent has set env vars, ensuring
# OMP_NUM_THREADS and related variables are visible at process start.
# For OpenMP/C++ this gives reliable thread-count control.
# For JAX/XLA, subprocess isolation guarantees a clean runtime and the
# requested environment, but exact CPU worker thread control is
# runtime-dependent — we record observed OS thread counts for honest
# post-hoc interpretation.
# Note: runner.py imports JAX at module level, so JAX is already loaded
# before main() runs.  observed_threads_before_engine_load is measured
# after module imports but before the engine class is instantiated.
# ---------------------------------------------------------------------------
from benchmarking.core.config import EuropeanOptionConfig
from benchmarking.runner.runner import BenchmarkRunner
from benchmarking.storage.database import BenchmarkDB
from benchmarking.workloads.mc_cpu import european_analytical_greeks

_ENGINE_META = {
    "jax": ("python", "xla"),
    "cpp": ("cpp",    "openmp"),
}


def _thread_count() -> int:
    """Return number of OS threads in the current process (via psutil)."""
    try:
        return _psutil.Process().num_threads()
    except Exception:
        return -1


def _load_engine(name: str):
    if name == "jax":
        from benchmarking.workloads.mc_jax import JAXMonteCarloEngine
        return JAXMonteCarloEngine()
    if name == "cpp":
        from benchmarking.workloads.mc_cpp import CPPMonteCarloEngine
        return CPPMonteCarloEngine()
    raise ValueError(f"Unknown engine: {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-cell thread-scalability runner (launched by orchestrator)")
    parser.add_argument("--engine",        required=True, choices=["jax", "cpp"])
    parser.add_argument("--threads",       required=True, type=int)
    parser.add_argument("--M",             required=True, type=int)
    parser.add_argument("--regime",        required=True, choices=["strong", "weak"])
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--runs",          type=int, default=5)
    parser.add_argument("--warmup",        type=int, default=2)
    parser.add_argument("--oversubscribed", action="store_true",
                        help="Flag this cell as an oversubscription run")
    args = parser.parse_args()

    config = EuropeanOptionConfig(
        S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0,
        option_type="call", seed=42,
        M=args.M,
    )

    analytical   = european_analytical_greeks(config)
    ana_price    = analytical["price"]
    language, backend = _ENGINE_META.get(args.engine, ("python", "cpu"))

    # experiment_type encodes regime and whether oversubscribed
    experiment_type = f"thread_scalability_{args.regime}"
    if args.oversubscribed:
        experiment_type += "_oversubscribed"

    # Capture env vars set by orchestrator before any engine loads
    env_omp = os.environ.get("OMP_NUM_THREADS")
    env_xla = os.environ.get("XLA_FLAGS")
    threads_before = _thread_count()

    try:
        engine = _load_engine(args.engine)
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc),
                          "engine": args.engine, "threads": args.threads,
                          "M": args.M, "regime": args.regime}))
        sys.exit(1)

    threads_after_load = _thread_count()

    runner = BenchmarkRunner(
        engine,
        name=f"{args.engine}/thread_scalability/T{args.threads}",
    )
    try:
        res = runner.run(config, num_warmup=args.warmup,
                         num_runs=args.runs, ad_mode="none")
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc),
                          "engine": args.engine, "threads": args.threads,
                          "M": args.M, "regime": args.regime}))
        sys.exit(1)

    threads_after_run = _thread_count()
    threads_max = max(threads_before, threads_after_load, threads_after_run)

    mean_ms    = res.mean_runtime * 1000
    std_ms     = res.std_runtime  * 1000
    min_ms     = res.min_runtime  * 1000
    max_ms     = res.max_runtime  * 1000
    price      = res.result
    throughput = res.throughput_paths_per_sec
    rel_err    = abs(price - ana_price) / abs(ana_price) if ana_price else None
    env        = res.metadata

    db = BenchmarkDB()
    db.store_run_full(
        config_dict=config.to_dict(),
        engine=args.engine,
        ad_mode="none",
        experiment_id=args.experiment_id,
        experiment_type=experiment_type,
        mean_runtime_ms=mean_ms,
        std_runtime_ms=std_ms,
        min_runtime_ms=min_ms,
        max_runtime_ms=max_ms,
        throughput_paths_per_sec=throughput,
        ad_overhead_ratio=1.0,
        result_value=price,
        analytical_price=ana_price,
        abs_price_error=abs(price - ana_price),
        rel_price_error=rel_err,
        memory_peak_mb=res.memory_peak_mb,
        language=language,
        backend=backend,
        num_threads=args.threads,
        requested_threads=args.threads,
        observed_threads_before_engine_load=threads_before,
        observed_threads_after_engine_load=threads_after_load,
        observed_threads_after_run=threads_after_run,
        observed_threads_max=threads_max,
        env_omp_num_threads=env_omp,
        env_xla_flags=env_xla,
        cpu_model=env.get("cpu_model"),
        cpu_architecture=env.get("cpu_architecture"),
        cpu_count=env.get("cpu_count"),
        memory_gb=env.get("memory_gb"),
        platform=env.get("platform"),
        python_version=env.get("python_version"),
        numpy_version=env.get("numpy_version"),
        jax_version=env.get("jax_version"),
    )

    # One JSON summary line read by the orchestrator
    print(json.dumps({
        "success":        True,
        "engine":         args.engine,
        "threads":        args.threads,
        "M":              args.M,
        "regime":         args.regime,
        "oversubscribed": args.oversubscribed,
        "mean_ms":        round(mean_ms, 4),
        "std_ms":         round(std_ms, 4),
        "throughput":     round(throughput, 1),
        "price":          round(price, 8),
        "rel_err":        round(rel_err, 8) if rel_err is not None else None,
        "memory_peak_mb": round(res.memory_peak_mb, 2),
        # Thread diagnostics
        "requested_threads":                   args.threads,
        "observed_threads_before_engine_load": threads_before,
        "observed_threads_after_engine_load":  threads_after_load,
        "observed_threads_after_run":          threads_after_run,
        "observed_threads_max":                threads_max,
        "env_omp_num_threads":                 env_omp,
        "env_xla_flags":                       env_xla,
    }))


if __name__ == "__main__":
    main()
