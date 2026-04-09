"""
Full benchmark suite: all workloads × all engines.

Matrix
------
  Workloads : european, asian, barrier, basket
  Engines   : cpu, jax  (cpp added automatically if built)
  AD modes  : none for all; forward + reverse for (jax, european) only

All results are persisted to results/benchmarks.db and visible in the frontend.

Usage
-----
    python experiments/run_benchmark_suite.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarking.core.config import (
    MCConfig, AsianOptionConfig, BarrierOptionConfig, BasketOptionConfig,
)
from benchmarking.runner.runner import BenchmarkRunner
from benchmarking.storage.database import BenchmarkDB
from benchmarking.workloads.mc_cpu import CPUMonteCarloEngine, black_scholes_call
from benchmarking.workloads.mc_jax import JAXMonteCarloEngine

# Try to load the optional C++ engine
try:
    from benchmarking.workloads.mc_cpp import CPPMonteCarloEngine
    _CPP = CPPMonteCarloEngine
except Exception:
    _CPP = None

# ── Benchmark configuration ───────────────────────────────────────────────

WORKLOAD_CONFIGS = {
    "european": MCConfig(
        S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, N=1, M=10000, seed=42),
    "asian":    AsianOptionConfig(
        S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, N=252,
        averaging="arithmetic", M=10000, seed=42),
    "barrier":  BarrierOptionConfig(
        S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, N=252,
        B=120.0, M=10000, seed=42),
    "basket":   BasketOptionConfig(
        S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, N=52,
        n_assets=3, M=10000, seed=42),
}

# AD modes to run per (engine, workload) pair; all others default to ["none"]
AD_MODES: dict[tuple[str, str], list[str]] = {
    ("jax", "european"): ["none", "forward", "reverse"],
}

# ── Helpers ───────────────────────────────────────────────────────────────

def _save_to_db(db: BenchmarkDB, config, engine_name: str, ad_mode: str, result) -> str:
    run_id = db.create_run(config.to_dict(), engine_name, ad_mode)
    db.mark_running(run_id)
    db.mark_completed(
        run_id=run_id,
        result_value=result.result,
        mean_runtime_ms=result.mean_runtime * 1000,
        std_runtime_ms=result.std_runtime * 1000,
        ad_overhead_ratio=result.ad_overhead_ratio,
    )
    return run_id


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    db = BenchmarkDB()
    bs_ref = black_scholes_call(WORKLOAD_CONFIGS["european"])
    total_ok = 0
    total_err = 0

    engines: dict[str, object] = {
        "cpu": CPUMonteCarloEngine(),
        "jax": JAXMonteCarloEngine(),
    }
    if _CPP is not None:
        engines["cpp"] = _CPP()

    print("=" * 80)
    print("  Monte Carlo Benchmark Suite")
    print("=" * 80)
    print(f"  European Black-Scholes reference price: ${bs_ref:.6f}")
    print(f"  Engines available: {list(engines.keys())}")
    print()

    for wl_name, config in WORKLOAD_CONFIGS.items():
        config.validate()
        print(f"  Workload: {wl_name}")
        print(f"  {'Engine':<8}  {'AD':<8}  {'Price':>10}  {'Mean ms':>8}  {'Std ms':>7}  ID")
        print(f"  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*8}")

        for eng_name, engine in engines.items():
            if not engine.supports(wl_name):  # type: ignore[attr-defined]
                continue
            runner = BenchmarkRunner(engine, name=f"{eng_name}/{wl_name}")
            ad_modes = AD_MODES.get((eng_name, wl_name), ["none"])

            for ad_mode in ad_modes:
                try:
                    result = runner.run(config, num_warmup=1, num_runs=5, ad_mode=ad_mode)
                    run_id = _save_to_db(db, config, eng_name, ad_mode, result)
                    print(
                        f"  {eng_name:<8}  {ad_mode:<8}  "
                        f"${result.result:>9.4f}  "
                        f"{result.mean_runtime * 1000:>7.2f}  "
                        f"{result.std_runtime * 1000:>6.2f}  "
                        f"{run_id[:8]}"
                    )
                    total_ok += 1
                except Exception as exc:
                    print(f"  {eng_name:<8}  {ad_mode:<8}  ERROR: {exc}")
                    total_err += 1
        print()

    print(f"✓ {total_ok} runs recorded in results/benchmarks.db"
          + (f"  ({total_err} errors)" if total_err else ""))
    print(f"  Open the dashboard:  cd frontend && npm run dev")


if __name__ == "__main__":
    main()
