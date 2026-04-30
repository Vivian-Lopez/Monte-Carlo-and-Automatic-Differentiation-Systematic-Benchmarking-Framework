"""
European Option Scalability Benchmark
========================================
Sweep path count M over multiple engines with ad_mode=none.

Engines: NumPy CPU, JAX, C++ OpenMP (if available).
M values: 1 000, 5 000, 10 000, 50 000, 100 000.

Results are stored in SQLite (results/benchmarks.db).

Usage
-----
    python experiments/run_european_scalability.py
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

M_VALUES   = [1_000, 5_000, 10_000, 50_000, 100_000]
NUM_WARMUP = 2   # overridden by --warmup
NUM_RUNS   = 5   # overridden by --runs

BASE_CFG = dict(S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0,
                option_type="call", seed=42)

_ENGINE_META = {
    "cpu":  ("python", "cpu"),
    "jax":  ("python", "xla"),
    "cpp":  ("cpp",    "openmp"),
}


def _rel_err(x, ref):
    if x is None or ref is None or ref == 0:
        return None
    return abs(x - ref) / abs(ref)


def _fmt(v, fmt=".4f"):
    return f"{v:{fmt}}" if v is not None else "    n/a "


def main() -> None:
    parser = argparse.ArgumentParser(description="European option scalability benchmark")
    parser.add_argument("--runs",   type=int, default=NUM_RUNS,   help="Timed repetitions (default: %(default)s)")
    parser.add_argument("--warmup", type=int, default=NUM_WARMUP, help="Warmup runs (default: %(default)s)")
    args = parser.parse_args()

    db = BenchmarkDB()
    experiment_id = str(uuid.uuid4())

    engines = [("cpu", CPUMonteCarloEngine()), ("jax", JAXMonteCarloEngine())]
    if _CPP is not None:
        engines.append(("cpp", _CPP))

    print()
    print("=" * 80)
    print("  European Option Scalability Benchmark")
    print("=" * 80)
    print(f"  M values : {M_VALUES}")
    print(f"  Engines  : {[e for e, _ in engines]}")
    print(f"  Runs     : {args.runs}  Warmup: {args.warmup}")
    print()

    rows = []

    for M in M_VALUES:
        config = EuropeanOptionConfig(**BASE_CFG, M=M)
        analytical = european_analytical_greeks(config)
        ana_price  = analytical["price"]

        for eng_name, engine in engines:
            runner = BenchmarkRunner(engine, name=f"{eng_name}/european/M{M}")
            try:
                res = runner.run(config, num_warmup=args.warmup, num_runs=args.runs,
                                 ad_mode="none")
            except Exception as exc:
                print(f"  [{eng_name}, M={M}] FAILED: {exc}")
                continue

            mean_ms   = res.mean_runtime * 1000
            std_ms    = res.std_runtime  * 1000
            min_ms    = res.min_runtime  * 1000
            max_ms    = res.max_runtime  * 1000
            price     = res.result
            throughput = res.throughput_paths_per_sec
            rel_err   = _rel_err(price, ana_price)
            language, backend = _ENGINE_META.get(eng_name, ("python", "cpu"))
            env = res.metadata

            db.store_run_full(
                config_dict=config.to_dict(),
                engine=eng_name,
                ad_mode="none",
                experiment_id=experiment_id,
                experiment_type="european_scalability",
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
                "M": M, "engine": eng_name,
                "mean_ms": mean_ms, "std_ms": std_ms,
                "throughput": throughput,
                "price": price, "rel_err": rel_err,
                "mem_mb": res.memory_peak_mb,
            })

    # Print scalability table
    print("  Scalability Table")
    print("  " + "-" * 90)
    hdr = (f"  {'M':>7}  {'Engine':<8}  {'Mean ms':>9}  {'Std ms':>7}  "
           f"{'Paths/s':>12}  {'Price':>9}  {'Rel Err%':>9}  {'Mem MB':>7}")
    print(hdr)
    print("  " + "-" * 90)
    for r in rows:
        re = f"{r['rel_err']*100:.3f}" if r["rel_err"] is not None else "  n/a  "
        print(
            f"  {r['M']:>7}  {r['engine']:<8}  {r['mean_ms']:>9.3f}  "
            f"{r['std_ms']:>7.3f}  {r['throughput']:>12.0f}  "
            f"{r['price']:>9.5f}  {re:>9}  {_fmt(r['mem_mb'], '.1f'):>7}"
        )

    print()
    print(f"  Results stored in : {db.db_path}")
    print(f"  Experiment ID     : {experiment_id}")
    print()


if __name__ == "__main__":
    main()
