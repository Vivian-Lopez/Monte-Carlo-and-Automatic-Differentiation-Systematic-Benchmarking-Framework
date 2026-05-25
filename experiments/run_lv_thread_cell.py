"""
Single-cell runner for local-vol thread-scalability experiments (LV-6 / LV-7).
================================================================================
Launched as a fresh subprocess by run_localvol_suite.py for every
(engine, thread_count, M, N) combination so that each cell starts with a
clean OMP / Rayon thread pool that has never been initialised.

Why a subprocess per cell?
--------------------------
OMP_NUM_THREADS is read by the OpenMP runtime on first parallel-region
entry; RAYON_NUM_THREADS is read by Rayon when it first builds its thread
pool.  Neither can be changed after that point.  Subprocess isolation
guarantees the requested value is already in the process environment before
any import-time or run-time initialisation occurs.

The parent orchestrator (run_localvol_suite.py) sets the environment
variables before launching this script.

Supported engines
-----------------
  jax   — JAX/XLA (OMP_NUM_THREADS set; actual XLA parallelism is
           runtime-dependent; observed OS thread counts are recorded for
           honest post-hoc interpretation)
  cpp   — C++ OpenMP (OMP_NUM_THREADS controls thread count reliably)
  rust  — Rust Rayon  (RAYON_NUM_THREADS controls thread count reliably)

Output
------
Prints exactly one JSON object to stdout on success.  The parent reads
this to build summary tables without relying on shared state.
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

    def _thread_count() -> int:
        try:
            return _psutil.Process().num_threads()
        except Exception:
            return -1
except ImportError:
    def _thread_count() -> int:  # type: ignore[misc]
        return -1

# These imports trigger JAX/OpenMP initialisation — they must happen *after*
# the parent has set the relevant env vars (i.e. at process start).
from benchmarking.core.config import EuropeanLocalVolConfig
from benchmarking.runner.runner import BenchmarkRunner
from benchmarking.storage.database import BenchmarkDB

_ENGINE_META = {
    "jax":  ("python", "xla"),
    "cpp":  ("cpp",    "openmp"),
    "rust": ("rust",   "rayon"),
}


def _load_engine(name: str):
    if name == "jax":
        from benchmarking.workloads.mc_jax import JAXMonteCarloEngine
        return JAXMonteCarloEngine()
    if name == "cpp":
        from benchmarking.workloads.mc_cpp import CPPMonteCarloEngine
        return CPPMonteCarloEngine()
    if name == "rust":
        from benchmarking.workloads.mc_rust import RustMonteCarloEngine
        return RustMonteCarloEngine()
    raise ValueError(f"Unknown engine: {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LV thread-cell subprocess runner (not for direct use)"
    )
    parser.add_argument("--engine",        required=True, choices=["jax", "cpp", "rust"])
    parser.add_argument("--threads",       required=True, type=int)
    parser.add_argument("--M",             required=True, type=int)
    parser.add_argument("--N",             required=True, type=int)
    parser.add_argument("--regime",        required=True, choices=["strong", "weak"])
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--runs",          type=int, default=5)
    parser.add_argument("--warmup",        type=int, default=2)
    args = parser.parse_args()

    config = EuropeanLocalVolConfig(M=args.M, N=args.N)
    experiment_type = f"lv_thread_{args.regime}"

    # Capture env vars set by orchestrator (recorded for diagnostics)
    env_omp   = os.environ.get("OMP_NUM_THREADS")
    env_xla   = os.environ.get("XLA_FLAGS")
    env_rayon = os.environ.get("RAYON_NUM_THREADS")
    threads_before = _thread_count()

    try:
        engine = _load_engine(args.engine)
    except Exception as exc:
        print(json.dumps({
            "success": False, "error": str(exc),
            "engine": args.engine, "threads": args.threads,
            "M": args.M, "N": args.N, "regime": args.regime,
        }))
        sys.exit(1)

    threads_after_load = _thread_count()
    language, backend = _ENGINE_META.get(args.engine, ("python", "cpu"))

    runner = BenchmarkRunner(
        engine,
        name=f"{args.engine}/lv_thread/T{args.threads}/M{args.M}/N{args.N}",
    )
    try:
        res = runner.run(config, num_warmup=args.warmup, num_runs=args.runs,
                         ad_mode="none")
    except Exception as exc:
        print(json.dumps({
            "success": False, "error": str(exc),
            "engine": args.engine, "threads": args.threads,
            "M": args.M, "N": args.N, "regime": args.regime,
        }))
        sys.exit(1)

    threads_after_run = _thread_count()
    threads_max = max(
        t for t in (threads_before, threads_after_load, threads_after_run) if t >= 0
    ) if any(t >= 0 for t in (threads_before, threads_after_load, threads_after_run)) else -1

    mean_ms    = res.mean_runtime * 1000
    std_ms     = res.std_runtime  * 1000
    min_ms     = res.min_runtime  * 1000
    max_ms     = res.max_runtime  * 1000
    throughput = res.throughput_paths_per_sec
    price      = res.result
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

    # Single JSON line consumed by the orchestrator
    print(json.dumps({
        "success":    True,
        "engine":     args.engine,
        "threads":    args.threads,
        "M":          args.M,
        "N":          args.N,
        "regime":     args.regime,
        "mean_ms":    round(mean_ms, 4),
        "std_ms":     round(std_ms, 4),
        "min_ms":     round(min_ms, 4),
        "max_ms":     round(max_ms, 4),
        "throughput": round(throughput, 1),
        "price":      round(price, 8),
        "memory_peak_mb":                      round(res.memory_peak_mb, 2),
        "requested_threads":                   args.threads,
        "observed_threads_before_engine_load": threads_before,
        "observed_threads_after_engine_load":  threads_after_load,
        "observed_threads_after_run":          threads_after_run,
        "observed_threads_max":                threads_max,
        "env_omp_num_threads":                 env_omp,
        "env_rayon_num_threads":               env_rayon,
        "env_xla_flags":                       env_xla,
    }))


if __name__ == "__main__":
    main()
