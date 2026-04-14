"""
Validation script: CUDA forward-mode AD (pathwise Delta) vs Black-Scholes analytical.

Runs the CUDA engine with ad_mode='forward' for a range of M values and
compares the Monte Carlo Delta estimate against the exact Black-Scholes value.

Usage
-----
    python experiments/test_cuda_ad.py

Requirements
------------
    PyCUDA + CUDA toolkit (nvcc on PATH).  If CUDA is unavailable the script
    exits with a clear message rather than raising an obscure ImportError.

Pass criterion
--------------
    Relative error < 5% for M >= 10 000.
"""

from __future__ import annotations

import math
import sys

# ---------------------------------------------------------------------------
# Check for PyCUDA before importing anything else
# ---------------------------------------------------------------------------

def _probe_cuda() -> bool:
    """Return True only when PyCUDA is installed AND a GPU is reachable."""
    try:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
        if cuda.Device.count() == 0:
            return False
        return True
    except Exception:
        return False


if not _probe_cuda():
    print("CUDA not available in this environment (no GPU or PyCUDA missing).")
    print("Install PyCUDA and ensure nvcc is on PATH, then re-run.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Imports (only reached when CUDA is present)
# ---------------------------------------------------------------------------

from benchmarking.core.config import EuropeanOptionConfig
from benchmarking.workloads.mc_cuda import CUDAMonteCarloEngine
from benchmarking.workloads.mc_cpu import european_call_delta, black_scholes_call
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Analytical put delta (Black-Scholes)
# ---------------------------------------------------------------------------

def _bs_put_delta(config: EuropeanOptionConfig) -> float:
    """Analytical Delta for a European put: N(d1) - 1."""
    d1 = (
        math.log(config.S0 / config.K)
        + (config.r + 0.5 * config.sigma ** 2) * config.T
    ) / (config.sigma * math.sqrt(config.T))
    return norm.cdf(d1) - 1.0


# ---------------------------------------------------------------------------
# Reference config (ATM call)
# ---------------------------------------------------------------------------

BASE_CONFIG = dict(S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0, seed=42)
M_VALUES    = [1_000, 10_000, 100_000, 1_000_000]
TOLERANCE   = 0.05   # 5% relative error threshold

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rel_err(approx: float, exact: float) -> float:
    return abs(approx - exact) / max(abs(exact), 1e-12)


def _run_case(engine: CUDAMonteCarloEngine, config: EuropeanOptionConfig) -> tuple[float, float]:
    """Return (price, delta) from the CUDA forward-AD engine."""
    price, greeks = engine.run(config, ad_mode="forward")
    assert greeks is not None, "Expected greeks dict from ad_mode='forward'"
    assert "delta" in greeks,  "Expected 'delta' key in greeks dict"
    return price, greeks["delta"]


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def main() -> None:
    engine = CUDAMonteCarloEngine()

    # ── Call option ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("European CALL — CUDA pathwise Delta vs Black-Scholes")
    print("=" * 70)
    print(f"{'M':>10}  {'BS price':>10}  {'MC price':>10}  "
          f"{'BS delta':>10}  {'MC delta':>10}  {'rel err':>9}  {'status':>6}")
    print("-" * 70)

    call_pass = True
    for M in M_VALUES:
        config      = EuropeanOptionConfig(**BASE_CONFIG, M=M, option_type="call")
        bs_price    = black_scholes_call(config)
        bs_delta    = european_call_delta(config)
        mc_price, mc_delta = _run_case(engine, config)
        err         = _rel_err(mc_delta, bs_delta)
        ok          = err < TOLERANCE
        if M >= 10_000 and not ok:
            call_pass = False
        print(f"{M:>10,}  {bs_price:>10.5f}  {mc_price:>10.5f}  "
              f"{bs_delta:>10.6f}  {mc_delta:>10.6f}  {err:>8.2%}  "
              f"{'PASS' if ok else 'FAIL':>6}")

    # ── Put option ───────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("European PUT  — CUDA pathwise Delta vs Black-Scholes")
    print("=" * 70)
    print(f"{'M':>10}  {'BS price':>10}  {'MC price':>10}  "
          f"{'BS delta':>10}  {'MC delta':>10}  {'rel err':>9}  {'status':>6}")
    print("-" * 70)

    put_pass = True
    for M in M_VALUES:
        config      = EuropeanOptionConfig(**BASE_CONFIG, M=M, option_type="put")
        bs_price_p  = (BASE_CONFIG["K"] * math.exp(-BASE_CONFIG["r"] * BASE_CONFIG["T"])
                       * norm.cdf(-( (math.log(BASE_CONFIG["S0"]/BASE_CONFIG["K"])
                                       + (BASE_CONFIG["r"] + 0.5*BASE_CONFIG["sigma"]**2)*BASE_CONFIG["T"])
                                     / (BASE_CONFIG["sigma"]*math.sqrt(BASE_CONFIG["T"]))
                                     - BASE_CONFIG["sigma"]*math.sqrt(BASE_CONFIG["T"]) ))
                       - BASE_CONFIG["S0"] * norm.cdf(
                           -( (math.log(BASE_CONFIG["S0"]/BASE_CONFIG["K"])
                               + (BASE_CONFIG["r"] + 0.5*BASE_CONFIG["sigma"]**2)*BASE_CONFIG["T"])
                              / (BASE_CONFIG["sigma"]*math.sqrt(BASE_CONFIG["T"])) )
                       ))
        bs_delta_p  = _bs_put_delta(config)
        mc_price, mc_delta = _run_case(engine, config)
        err         = _rel_err(mc_delta, bs_delta_p)
        ok          = err < TOLERANCE
        if M >= 10_000 and not ok:
            put_pass = False
        print(f"{M:>10,}  {bs_price_p:>10.5f}  {mc_price:>10.5f}  "
              f"{bs_delta_p:>10.6f}  {mc_delta:>10.6f}  {err:>8.2%}  "
              f"{'PASS' if ok else 'FAIL':>6}")

    # ── ad_mode='none' regression: greeks dict must be None ─────────────────
    print()
    print("Regression — ad_mode='none' must not return greeks ...", end=" ")
    config_none = EuropeanOptionConfig(**BASE_CONFIG, M=10_000, option_type="call")
    price_none, greeks_none = engine.run(config_none, ad_mode="none")
    assert greeks_none is None, f"Expected None greeks for ad_mode='none', got {greeks_none}"
    print("OK")

    # ── unsupported ad_mode must raise ───────────────────────────────────────
    print("Regression — ad_mode='reverse' must raise NotImplementedError ...", end=" ")
    try:
        engine.run(config_none, ad_mode="reverse")
        print("FAIL (no exception raised)")
    except NotImplementedError:
        print("OK")

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    all_pass = call_pass and put_pass
    if all_pass:
        print("All validation checks PASSED (<5% relative error for M >= 10 000).")
    else:
        print("One or more validation checks FAILED. See table above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
