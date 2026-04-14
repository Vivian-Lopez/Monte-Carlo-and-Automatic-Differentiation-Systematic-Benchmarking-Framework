"""
Seed one synthetic CUDA run into the benchmark database for UI testing.

Use this in Codespaces (where PyCUDA is unavailable) so the GPU tab and
comparison table show realistic data rather than empty placeholders.

The seeded values are self-consistent with a CPU run at the same config:
  * result_value ≈ 10.45  (Black-Scholes closed-form for these params)
  * mean_runtime_ms = 1.2  (faster than NumPy CPU due to GPU parallelism)

Usage
-----
    python scripts/seed_cuda_run.py [--M <paths>]

The script is idempotent — re-running seeds an additional run (which will
update the averaged values in the dashboard but won't break anything).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmarking.core.config import EuropeanOptionConfig
from benchmarking.storage.database import BenchmarkDB


def seed(M: int = 10_000) -> str:
    """Insert one synthetic completed CUDA run and return its run ID."""
    config = EuropeanOptionConfig(
        S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0,
        option_type="call", M=M, seed=42,
    )
    config.validate()

    db = BenchmarkDB()
    run_id = db.create_run(config.to_dict(), engine="cuda", ad_mode="none")
    db.mark_running(run_id)
    db.mark_completed(
        run_id=run_id,
        # Black-Scholes closed-form ≈ 10.4506; Monte Carlo noise adds ~0.01
        result_value=10.4519,
        # GPU is faster than CPU (CPU ≈ 5–20 ms at M=10k, GPU ≈ 1–2 ms)
        mean_runtime_ms=1.2,
        std_runtime_ms=0.08,
        ad_overhead_ratio=1.0,
        greeks=None,
    )
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed a synthetic CUDA run for UI testing.")
    parser.add_argument("--M", type=int, default=10_000,
                        help="Number of Monte Carlo paths (default: 10000)")
    args = parser.parse_args()

    run_id = seed(args.M)
    print(f"Seeded synthetic CUDA run: {run_id}")
    print(f"  engine=cuda  workload=european  M={args.M:,}  seed=42")
    print(f"  result=10.4519  mean_runtime=1.2ms")
    print()
    print("Restart the Flask server (or wait for the next poll) then refresh the dashboard.")
    print("  GPU tab   → shows runtime / throughput / speedup charts")
    print("  Dashboard → shows CUDA in the comparison table")


if __name__ == "__main__":
    main()
