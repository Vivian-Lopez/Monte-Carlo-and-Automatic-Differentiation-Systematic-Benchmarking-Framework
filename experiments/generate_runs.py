"""
Generate multiple benchmark runs for the dashboard.

Sweeps:
  1. Path-count scaling  — European option, CPU engine, M in {1k, 5k, 10k, 50k}
  2. Workload coverage   — all four workload types, CPU engine, M = 10 000

Results are written as JSON under results/ and recorded in results/benchmarks.db.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from benchmarking.core.config import (
    MCConfig, AsianOptionConfig, BarrierOptionConfig, BasketOptionConfig,
)
from benchmarking.runner.runner import BenchmarkRunner
from benchmarking.storage.database import BenchmarkDB
from benchmarking.workloads.mc_cpu import CPUMonteCarloEngine


def _save_to_db(db: BenchmarkDB, config, engine_name: str, ad_mode: str, result) -> str:
    """Persist a completed BenchmarkResult to SQLite and return the run ID."""
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


def generate_runs() -> None:
    os.makedirs("results", exist_ok=True)
    db = BenchmarkDB()
    runner = BenchmarkRunner(CPUMonteCarloEngine(), name="CPU")
    total = 0

    # ── 1. Path-count scaling sweep ───────────────────────────────────
    print("=" * 70)
    print("Sweep 1/2 — Path-count scaling  (European, CPU)")
    print("=" * 70)
    base = dict(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, N=1, seed=42)
    for M in [1000, 5000, 10000, 50000]:
        config = MCConfig(M=M, **base)
        config.validate()
        result = runner.run(config, num_warmup=1, num_runs=5, ad_mode="none")
        filename = f"results/benchmark_run_m{M:06d}.json"
        runner.save_results(result, filename)
        run_id = _save_to_db(db, config, "cpu", "none", result)
        print(f"  M={M:>6,}  price={result.result:.4f}  "
              f"mean={result.mean_runtime * 1000:.2f} ms  "
              f"throughput={M / result.mean_runtime / 1e6:.2f}M p/s  "
              f"id={run_id[:8]}")
        total += 1

    # ── 2. Workload coverage sweep ────────────────────────────────────
    print()
    print("=" * 70)
    print("Sweep 2/2 — Workload coverage  (all types, CPU, M=10 000)")
    print("=" * 70)
    workload_configs = [
        ("european", MCConfig(
            S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, N=1, M=10000, seed=42)),
        ("asian",    AsianOptionConfig(
            S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0,
            N=252, averaging="arithmetic", M=10000, seed=42)),
        ("barrier",  BarrierOptionConfig(
            S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0,
            N=252, B=120.0, M=10000, seed=42)),
        ("basket",   BasketOptionConfig(
            S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0,
            N=52, n_assets=3, M=10000, seed=42)),
    ]
    for label, config in workload_configs:
        config.validate()
        result = runner.run(config, num_warmup=1, num_runs=5, ad_mode="none")
        filename = f"results/benchmark_run_{label}.json"
        runner.save_results(result, filename)
        run_id = _save_to_db(db, config, "cpu", "none", result)
        print(f"  {label:<10}  price={result.result:.4f}  "
              f"mean={result.mean_runtime * 1000:.2f} ms  "
              f"id={run_id[:8]}")
        total += 1

    print()
    print(f"✓ {total} runs recorded — JSON: results/  DB: results/benchmarks.db")
    print(f"  Open the dashboard:  cd frontend && npm run dev")


if __name__ == "__main__":
    generate_runs()
