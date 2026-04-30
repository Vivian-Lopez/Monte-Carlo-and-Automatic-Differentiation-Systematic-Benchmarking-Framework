"""
European Option Baseline Benchmark
===================================
Compare NumPy CPU, JAX, and C++ (if available) on the European call option
with ad_mode=none.

Results are stored in SQLite (results/benchmarks.db).
No JSON middleman.

Usage
-----
    python experiments/run_european_baseline.py
"""

import argparse
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmarking.core.config import EuropeanOptionConfig
from benchmarking.runner.runner import BenchmarkRunner
from benchmarking.storage.database import BenchmarkDB
from benchmarking.workloads.mc_cpu import CPUMonteCarloEngine, european_analytical_greeks
from benchmarking.workloads.mc_jax import JAXMonteCarloEngine

# Optional C++ engine
try:
    from benchmarking.workloads.mc_cpp import CPPMonteCarloEngine
    _CPP = CPPMonteCarloEngine()
except Exception:
    _CPP = None

# ---------------------------------------------------------------------------
# Standard European call configuration (ATM, 1-year, σ=20%)
# ---------------------------------------------------------------------------
CONFIG = EuropeanOptionConfig(
    S0=100.0,
    K=100.0,
    r=0.05,
    sigma=0.20,
    T=1.0,
    option_type="call",
    M=10_000,
    seed=42,
)

NUM_WARMUP = 2   # overridden by --warmup
NUM_RUNS   = 7   # overridden by --runs

# Engine language / backend metadata for storage
_ENGINE_META = {
    "cpu":  ("python", "cpu"),
    "jax":  ("python", "xla"),
    "cpp":  ("cpp",    "openmp"),
}


def _abs_err(x, ref):
    return abs(x - ref) if ref is not None else None


def _rel_err(x, ref):
    return abs(x - ref) / abs(ref) if (ref is not None and ref != 0) else None


def _fmt(v, fmt=".4f"):
    return f"{v:{fmt}}" if v is not None else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description="European option baseline benchmark")
    parser.add_argument("--runs",   type=int, default=NUM_RUNS,   help="Timed repetitions (default: %(default)s)")
    parser.add_argument("--warmup", type=int, default=NUM_WARMUP, help="Warmup runs (default: %(default)s)")
    args = parser.parse_args()

    db = BenchmarkDB()
    experiment_id = str(uuid.uuid4())

    # Analytical reference
    analytical = european_analytical_greeks(CONFIG)
    ana_price  = analytical["price"]

    engines = [("cpu", CPUMonteCarloEngine()), ("jax", JAXMonteCarloEngine())]
    if _CPP is not None:
        engines.append(("cpp", _CPP))

    print()
    print("=" * 80)
    print("  European Option Baseline Benchmark")
    print("=" * 80)
    print(f"  Config  : S0={CONFIG.S0}, K={CONFIG.K}, r={CONFIG.r}, "
          f"σ={CONFIG.sigma}, T={CONFIG.T}, M={CONFIG.M:,}, seed={CONFIG.seed}")
    print(f"  Analytical call price: {ana_price:.6f}")
    print()

    rows = []
    for eng_name, engine in engines:
        runner = BenchmarkRunner(engine, name=f"{eng_name}/european")
        try:
            result = runner.run(CONFIG, num_warmup=args.warmup, num_runs=args.runs, ad_mode="none")
        except Exception as exc:
            print(f"  [{eng_name}] FAILED: {exc}")
            continue

        mean_ms  = result.mean_runtime * 1000
        std_ms   = result.std_runtime  * 1000
        min_ms   = result.min_runtime  * 1000
        max_ms   = result.max_runtime  * 1000
        price    = result.result
        throughput = result.throughput_paths_per_sec
        abs_err  = _abs_err(price, ana_price)
        rel_err  = _rel_err(price, ana_price)
        language, backend = _ENGINE_META.get(eng_name, ("python", "cpu"))
        env = result.metadata

        run_id = db.store_run_full(
            config_dict=CONFIG.to_dict(),
            engine=eng_name,
            ad_mode="none",
            experiment_id=experiment_id,
            experiment_type="european_baseline",
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
            abs_price_error=abs_err,
            rel_price_error=rel_err,
            memory_peak_mb=result.memory_peak_mb,
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

        rows.append({
            "engine":    eng_name,
            "price":     price,
            "mean_ms":   mean_ms,
            "std_ms":    std_ms,
            "throughput": throughput,
            "abs_err":   abs_err,
            "rel_err":   rel_err,
            "mem_mb":    result.memory_peak_mb,
            "run_id":    run_id,
        })

    # Print table
    hdr = f"  {'Engine':<8}  {'Price':>10}  {'Mean ms':>9}  {'Std ms':>7}  " \
          f"{'Paths/s':>12}  {'Abs Err':>9}  {'Rel Err':>9}  {'Mem MB':>7}  ID"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(
            f"  {r['engine']:<8}  {_fmt(r['price'], '.6f'):>10}  "
            f"{_fmt(r['mean_ms'], '.3f'):>9}  {_fmt(r['std_ms'], '.3f'):>7}  "
            f"{_fmt(r['throughput'], '.0f'):>12}  "
            f"{_fmt(r['abs_err'], '.4f'):>9}  {_fmt(r['rel_err'], '.4f'):>9}  "
            f"{_fmt(r['mem_mb'], '.1f'):>7}  {r['run_id'][:8]}"
        )

    print()
    print(f"  Analytical reference  : {ana_price:.6f}")
    print(f"  Results stored in     : {db.db_path}")
    print(f"  Experiment ID         : {experiment_id}")
    print()


if __name__ == "__main__":
    main()
