"""
Single-cell runner for the BLAS-backend comparison experiment.

NOT meant to be called directly.  Launched as a subprocess by
run_blas_comparison.py for every (blas_environment, M) combination.

Why a subprocess per cell?
--------------------------
NumPy's BLAS linkage is fixed at compile time — you cannot swap it at
runtime.  Each environment (bench_mkl, bench_openblas, …) has a different
Python executable pointing to a different NumPy build.  The orchestrator
calls this script via the correct interpreter so the BLAS library is the
one baked into that environment's NumPy.

This script also validates that BLAS detection succeeds before doing any
timing, so the orchestrator never silently stores rows with blas_backend=null.

Output
------
Prints exactly one JSON object to stdout on success; a JSON error object on
failure.  All stderr (warnings, import noise) is left for the orchestrator
to filter.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# BLAS detection — must succeed before any timing
# ---------------------------------------------------------------------------

def detect_blas() -> tuple[str, str]:
    """
    Return (blas_name, vectorization_flag) for the current process.

    blas_name is one of: "mkl", "openblas", "blis", "accelerate", "unknown".
    vectorization_flag is one of: "AVX-512", "AVX2", "AVX", "SSE2", "unknown".

    Detection strategy (in priority order):
    1. /proc/self/maps  — inspects *runtime*-loaded shared libraries after
                          importing numpy.  This is the most reliable method
                          because conda's BLAS metapackage swaps the
                          libcblas.so.3 symlink to point to MKL or OpenBLAS
                          regardless of what numpy was compiled against.
                          np.show_config() only reports the compile-time BLAS
                          and is therefore unreliable when conda has overridden
                          the runtime BLAS via symlink switching.
    2. np.show_config() — fallback for non-Linux or missing /proc.
    """
    import numpy as np

    blas_name = "unknown"

    # --- Strategy 1: runtime /proc/self/maps (Linux only) ---
    # Import numpy first so its shared libraries are loaded into the process,
    # then read /proc/self/maps to see exactly which BLAS .so is resident.
    try:
        maps = Path("/proc/self/maps").read_text()
        maps_lower = maps.lower()
        if "libmkl" in maps_lower or "mkl_rt" in maps_lower:
            blas_name = "mkl"
        elif "openblas" in maps_lower:
            blas_name = "openblas"
        elif "libblis" in maps_lower:
            blas_name = "blis"
        elif "accelerate" in maps_lower or "veclib" in maps_lower:
            blas_name = "accelerate"
    except Exception:
        pass

    # --- Strategy 2: np.show_config() text (fallback) ---
    if blas_name == "unknown":
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            np.show_config()
        text = buf.getvalue().lower()
        if "mkl" in text and "openblas" not in text:
            blas_name = "mkl"
        elif "openblas" in text:
            blas_name = "openblas"
        elif "blis" in text:
            blas_name = "blis"
        elif "accelerate" in text:
            blas_name = "accelerate"

    # Normalise vendor aliases (e.g. "mkl_rt" -> "mkl")
    if blas_name.startswith("mkl"):
        blas_name = "mkl"

    # Vectorisation: read CPU flags from /proc/cpuinfo (Linux) or fallback
    vec_flag = "unknown"
    try:
        flags_text = Path("/proc/cpuinfo").read_text().lower()
        if "avx512" in flags_text:
            vec_flag = "AVX-512"
        elif "avx2" in flags_text:
            vec_flag = "AVX2"
        elif " avx " in flags_text or flags_text.endswith("avx"):
            vec_flag = "AVX"
        elif "sse2" in flags_text:
            vec_flag = "SSE2"
    except Exception:
        pass

    return blas_name, vec_flag


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-cell BLAS-comparison runner (launched by orchestrator)")
    parser.add_argument("--M",             required=True, type=int)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--runs",          type=int, default=7)
    parser.add_argument("--warmup",        type=int, default=2)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Detect BLAS before any engine import — fail fast with structured error
    # ------------------------------------------------------------------
    try:
        blas_name, vec_flag = detect_blas()
    except Exception as exc:
        print(json.dumps({"success": False, "error": f"BLAS detection raised: {exc}",
                          "M": args.M}))
        sys.exit(1)

    if blas_name == "unknown":
        print(json.dumps({
            "success": False,
            "error": (
                "BLAS backend could not be identified.  "
                "Ensure this process is running under the correct conda environment "
                "and that numpy is installed there."
            ),
            "M": args.M,
        }))
        sys.exit(1)

    # ------------------------------------------------------------------
    # Late imports — after env check so import errors don't mask BLAS issues
    # ------------------------------------------------------------------
    from benchmarking.core.config import EuropeanOptionConfig
    from benchmarking.runner.runner import BenchmarkRunner
    from benchmarking.storage.database import BenchmarkDB
    from benchmarking.workloads.mc_cpu import CPUMonteCarloEngine, european_analytical_greeks

    config = EuropeanOptionConfig(
        S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0,
        option_type="call", seed=42,
        M=args.M,
    )

    analytical = european_analytical_greeks(config)
    ana_price  = analytical["price"]

    runner = BenchmarkRunner(CPUMonteCarloEngine(), name=f"cpu/blas_comparison/{blas_name}")

    try:
        res = runner.run(config, num_warmup=args.warmup, num_runs=args.runs, ad_mode="none")
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc), "M": args.M,
                          "blas_backend": blas_name}))
        sys.exit(1)

    mean_ms    = res.mean_runtime * 1000
    std_ms     = res.std_runtime  * 1000
    min_ms     = res.min_runtime  * 1000
    max_ms     = res.max_runtime  * 1000
    price      = res.result
    throughput = res.throughput_paths_per_sec
    rel_err    = abs(price - ana_price) / abs(ana_price) if ana_price else None
    env        = res.metadata

    import numpy as np
    numpy_version = np.__version__

    db = BenchmarkDB()
    run_id = db.store_run_full(
        config_dict=config.to_dict(),
        engine="cpu",
        ad_mode="none",
        experiment_id=args.experiment_id,
        experiment_type="blas_comparison",
        mean_runtime_ms=mean_ms,
        std_runtime_ms=std_ms,
        min_runtime_ms=min_ms,
        max_runtime_ms=max_ms,
        throughput_paths_per_sec=throughput,
        ad_overhead_ratio=1.0,
        result_value=price,
        analytical_price=ana_price,
        analytical_delta=analytical["delta"],
        analytical_vega=analytical["vega"],
        analytical_rho=analytical["rho"],
        abs_price_error=abs(price - ana_price),
        rel_price_error=rel_err,
        memory_peak_mb=res.memory_peak_mb,
        language="python",
        backend="numpy",
        num_threads=1,
        blas_backend=blas_name,
        vectorization_flag=vec_flag,
        cpu_model=env.get("cpu_model"),
        cpu_architecture=env.get("cpu_architecture"),
        cpu_count=env.get("cpu_count"),
        memory_gb=env.get("memory_gb"),
        platform=env.get("platform"),
        python_version=env.get("python_version"),
        numpy_version=numpy_version,
        jax_version=env.get("jax_version"),
    )

    print(json.dumps({
        "success":          True,
        "blas_backend":     blas_name,
        "vectorization":    vec_flag,
        "M":                args.M,
        "mean_ms":          round(mean_ms, 4),
        "std_ms":           round(std_ms, 4),
        "throughput":       round(throughput, 1),
        "price":            round(price, 8),
        "rel_err":          round(rel_err, 8) if rel_err is not None else None,
        "memory_peak_mb":   round(res.memory_peak_mb, 2),
        "numpy_version":    numpy_version,
        "run_id":           run_id,
    }))


if __name__ == "__main__":
    main()
