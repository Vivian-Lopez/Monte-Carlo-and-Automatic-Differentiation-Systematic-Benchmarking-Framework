"""
Validation script: CUDAMonteCarloEngine vs CPUMonteCarloEngine.

Runs both engines over increasing path counts and compares:
  - Prices
  - Relative error (%)
  - Wall-clock runtime

Usage
-----
    python experiments/test_cuda_vs_cpu.py

If CUDA / PyCUDA is not available the script exits cleanly with a message.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarking.core.config import EuropeanOptionConfig
from benchmarking.workloads.mc_cpu import CPUMonteCarloEngine

# ── CUDA availability check ───────────────────────────────────────────────

try:
    from benchmarking.workloads.mc_cuda import CUDAMonteCarloEngine
    _cuda_engine = CUDAMonteCarloEngine()
    # Force a tiny compile + launch to confirm the GPU is actually usable.
    _probe = EuropeanOptionConfig(M=128, seed=0)
    _cuda_engine.run(_probe)
    CUDA_AVAILABLE = True
except Exception as exc:
    CUDA_AVAILABLE = False
    _cuda_unavailable_reason = str(exc)

# ── Experiment parameters ─────────────────────────────────────────────────

BASE_CONFIG = dict(
    S0=100.0,
    K=100.0,
    r=0.05,
    sigma=0.20,
    T=1.0,
    option_type="call",
    seed=42,
)

PATH_COUNTS = [1_000, 10_000, 100_000]

RELATIVE_ERROR_WARN_THRESHOLD = 0.03   # 3 %

# ── Helpers ───────────────────────────────────────────────────────────────

def _sep(char: str = "-", width: int = 72) -> None:
    print(char * width)


def _run_timed(engine, config):
    """Return (price, elapsed_seconds)."""
    t0 = time.perf_counter()
    price, _ = engine.run(config)
    return price, time.perf_counter() - t0


def _rel_error(cpu_price: float, cuda_price: float) -> float:
    if cpu_price == 0.0:
        return float("inf")
    return abs(cuda_price - cpu_price) / abs(cpu_price)

# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    if not CUDA_AVAILABLE:
        print("\nCUDA engine is NOT available on this machine.")
        print(f"Reason: {_cuda_unavailable_reason}")
        print("\nInstall PyCUDA and ensure the CUDA toolkit / nvcc are present:")
        print("    pip install pycuda")
        print("\nExiting.")
        sys.exit(0)

    cpu_engine  = CPUMonteCarloEngine()
    cuda_engine = _cuda_engine

    _sep("=")
    print("  European Call Option — CPU vs CUDA Monte Carlo Validation")
    _sep("=")
    print(f"  S0={BASE_CONFIG['S0']}  K={BASE_CONFIG['K']}  "
          f"r={BASE_CONFIG['r']}  sigma={BASE_CONFIG['sigma']}  "
          f"T={BASE_CONFIG['T']}  seed={BASE_CONFIG['seed']}")
    _sep("=")

    header = f"{'M':>10}  {'CPU price':>12}  {'CPU time':>10}  " \
             f"{'CUDA price':>12}  {'CUDA time':>10}  {'Rel error':>10}"
    print(header)
    _sep()

    any_warning = False

    for M in PATH_COUNTS:
        config = EuropeanOptionConfig(M=M, **BASE_CONFIG)

        cpu_price,  cpu_time  = _run_timed(cpu_engine,  config)
        cuda_price, cuda_time = _run_timed(cuda_engine, config)

        rel_err = _rel_error(cpu_price, cuda_price)
        warn    = rel_err > RELATIVE_ERROR_WARN_THRESHOLD

        row = (
            f"{M:>10,}  "
            f"{cpu_price:>12.6f}  "
            f"{cpu_time:>9.3f}s  "
            f"{cuda_price:>12.6f}  "
            f"{cuda_time:>9.3f}s  "
            f"{rel_err * 100:>9.4f}%"
        )
        if warn:
            row += "  *** WARNING: error > 3%"
            any_warning = True

        print(row)

    _sep()

    if any_warning:
        print("\n[WARNING] At least one result exceeded the 3% relative-error")
        print("threshold. This is expected for very low path counts (M=1 000)")
        print("due to Monte Carlo variance. Increase M to reduce error.\n")
    else:
        print("\nAll results within the 3% relative-error threshold.\n")


if __name__ == "__main__":
    main()
