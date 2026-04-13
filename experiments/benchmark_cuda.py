"""
CPU vs CUDA Monte Carlo benchmark.

Runs European call pricing for increasing path counts (M), measuring:
  - Price from each engine
  - Wall-clock runtime (seconds)
  - Throughput (paths / second)
  - Relative error between CPU and CUDA

Results are saved to results/benchmark_cuda_<timestamp>.json.

Usage
-----
    python experiments/benchmark_cuda.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmarking.core.config import EuropeanOptionConfig
from benchmarking.workloads.mc_cpu import CPUMonteCarloEngine

# ── CUDA availability ─────────────────────────────────────────────────────

try:
    from benchmarking.workloads.mc_cuda import CUDAMonteCarloEngine
    _cuda_engine = CUDAMonteCarloEngine()
    _cuda_engine.run(EuropeanOptionConfig(M=128, seed=0))   # warm-up probe
    CUDA_AVAILABLE = True
except Exception as _exc:
    CUDA_AVAILABLE = False
    _CUDA_UNAVAIL_REASON = str(_exc)

# ── Experiment parameters ─────────────────────────────────────────────────

BASE_CONFIG = dict(
    S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0,
    option_type="call", seed=42,
)

PATH_COUNTS = [1_000, 10_000, 100_000, 1_000_000]

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────

def _sep(char: str = "-", width: int = 80) -> None:
    print(char * width)


def _run_timed(engine, config: EuropeanOptionConfig):
    """Return (price, elapsed_seconds). Wraps engine errors gracefully."""
    try:
        t0 = time.perf_counter()
        price, _ = engine.run(config)
        return price, time.perf_counter() - t0
    except Exception as exc:
        return None, None, str(exc)


def _throughput(M: int, elapsed_s: float | None) -> float | None:
    if elapsed_s is None or elapsed_s == 0:
        return None
    return M / elapsed_s


def _rel_error(cpu_price: float, cuda_price: float) -> float:
    if cpu_price == 0.0:
        return float("inf")
    return abs(cuda_price - cpu_price) / abs(cpu_price)

# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    if not CUDA_AVAILABLE:
        print("\nCUDA engine is NOT available on this machine.")
        print(f"Reason: {_CUDA_UNAVAIL_REASON}")
        print("\nInstall PyCUDA and ensure nvcc is on PATH:")
        print("    pip install pycuda")
        sys.exit(0)

    cpu_engine  = CPUMonteCarloEngine()
    cuda_engine = _cuda_engine

    _sep("=")
    print("  European Call — CPU vs CUDA Benchmark")
    _sep("=")
    print(f"  S0={BASE_CONFIG['S0']}  K={BASE_CONFIG['K']}  "
          f"r={BASE_CONFIG['r']}  σ={BASE_CONFIG['sigma']}  "
          f"T={BASE_CONFIG['T']}  seed={BASE_CONFIG['seed']}")
    _sep("=")

    header = (
        f"{'M':>10}  {'CPU price':>10}  {'CPU s':>8}  {'CPU paths/s':>13}  "
        f"{'CUDA price':>10}  {'CUDA s':>8}  {'CUDA paths/s':>13}  {'Rel err%':>9}"
    )
    print(header)
    _sep()

    records = []
    for M in PATH_COUNTS:
        config = EuropeanOptionConfig(M=M, **BASE_CONFIG)

        cpu_price,  cpu_t  = _run_timed(cpu_engine,  config)
        cuda_price, cuda_t = _run_timed(cuda_engine, config)

        cpu_tp  = _throughput(M, cpu_t)
        cuda_tp = _throughput(M, cuda_t)
        err     = _rel_error(cpu_price, cuda_price) if (cpu_price and cuda_price) else None

        warn = err is not None and err > 0.03

        def _fmt_price(v):  return f"{v:.6f}" if v is not None else "ERROR"
        def _fmt_t(v):      return f"{v:.4f}" if v is not None else "  —  "
        def _fmt_tp(v):     return f"{v:,.0f}" if v is not None else "  —  "
        def _fmt_err(v):    return f"{v * 100:.4f}%" if v is not None else "  —  "

        row = (
            f"{M:>10,}  {_fmt_price(cpu_price):>10}  {_fmt_t(cpu_t):>8}  "
            f"{_fmt_tp(cpu_tp):>13}  {_fmt_price(cuda_price):>10}  "
            f"{_fmt_t(cuda_t):>8}  {_fmt_tp(cuda_tp):>13}  {_fmt_err(err):>9}"
        )
        if warn:
            row += "  *** >3%"
        print(row)

        records.append({
            "M":               M,
            "cpu_price":       cpu_price,
            "cpu_runtime_s":   cpu_t,
            "cpu_throughput":  cpu_tp,
            "cuda_price":      cuda_price,
            "cuda_runtime_s":  cuda_t,
            "cuda_throughput": cuda_tp,
            "rel_error":       err,
        })

    _sep()

    # ── Save results ──────────────────────────────────────────────────────
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path  = RESULTS_DIR / f"benchmark_cuda_{timestamp}.json"

    payload = {
        "timestamp":    timestamp,
        "base_config":  BASE_CONFIG,
        "path_counts":  PATH_COUNTS,
        "results":      records,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
