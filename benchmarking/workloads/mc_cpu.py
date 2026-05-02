import math
import random
import numpy as np
from scipy.stats import norm
from typing import Optional, Tuple
from benchmarking.core.config import (
    WorkloadConfig, EuropeanOptionConfig,
)
from benchmarking.core.engine import MonteCarloEngine, ADMode


class CPUMonteCarloEngine(MonteCarloEngine):
    """
    Pure-Python / NumPy CPU implementation supporting multiple option workloads.

    Dispatch is done by inspecting config.workload_type, so adding a new
    workload only requires adding a new _price_<type> method here.
    """

    SUPPORTED = {"european"}

    def supports(self, workload_type: str) -> bool:
        return workload_type in self.SUPPORTED

    def supported_ad_modes(self) -> Tuple[ADMode, ...]:
        return ("none",)

    def run(self, config: WorkloadConfig, ad_mode: ADMode = "none") -> Tuple[float, Optional[dict]]:
        if ad_mode != "none":
            raise NotImplementedError(
                f"CPUMonteCarloEngine does not support ad_mode={ad_mode!r}. "
                "Use the JAX engine for automatic differentiation."
            )
        if config.workload_type == "european":
            return (self._price_european(config), None)
        else:
            raise NotImplementedError(f"CPUMonteCarloEngine does not support workload '{config.workload_type}'")

    # ------------------------------------------------------------------
    # European option  (GBM, single-step exact)
    # ------------------------------------------------------------------

    def _price_european(self, config: EuropeanOptionConfig) -> float:
        rng = np.random.default_rng(config.seed)
        Z = rng.standard_normal(config.M)
        S_T = config.S0 * np.exp(
            (config.r - 0.5 * config.sigma ** 2) * config.T
            + config.sigma * math.sqrt(config.T) * Z
        )
        if config.option_type == "call":
            payoff = np.maximum(S_T - config.K, 0.0)
        else:
            payoff = np.maximum(config.K - S_T, 0.0)
        return float(math.exp(-config.r * config.T) * payoff.mean())

# ---------------------------------------------------------------------------
# Black-Scholes closed form (European only, for validation)
# ---------------------------------------------------------------------------

def black_scholes_call(config: EuropeanOptionConfig) -> float:
    """Black-Scholes closed-form price for a European call."""
    d1 = (math.log(config.S0 / config.K) +
          (config.r + 0.5 * config.sigma ** 2) * config.T) / \
         (config.sigma * math.sqrt(config.T))
    d2 = d1 - config.sigma * math.sqrt(config.T)
    return (config.S0 * norm.cdf(d1) -
            config.K * math.exp(-config.r * config.T) * norm.cdf(d2))


def black_scholes_put(config: EuropeanOptionConfig) -> float:
    """Black-Scholes closed-form price for a European put (via put-call parity)."""
    call = black_scholes_call(config)
    return call - config.S0 + config.K * math.exp(-config.r * config.T)


def european_call_delta(config: EuropeanOptionConfig) -> float:
    """Analytical delta for European call option."""
    d1 = (math.log(config.S0 / config.K) +
          (config.r + 0.5 * config.sigma ** 2) * config.T) / \
         (config.sigma * math.sqrt(config.T))
    return norm.cdf(d1)


def _bs_d1_d2(config: EuropeanOptionConfig):
    """Return (d1, d2) for a European option under GBM."""
    sqrt_T = math.sqrt(config.T)
    d1 = (math.log(config.S0 / config.K) +
          (config.r + 0.5 * config.sigma ** 2) * config.T) / \
         (config.sigma * sqrt_T)
    d2 = d1 - config.sigma * sqrt_T
    return d1, d2


def european_analytical_greeks(config: EuropeanOptionConfig) -> dict:
    """
    Return analytical Black-Scholes price and Greeks for a European option.

    Returns:
        dict with keys: price, delta, vega, rho
    """
    d1, d2 = _bs_d1_d2(config)
    sqrt_T = math.sqrt(config.T)
    disc = math.exp(-config.r * config.T)
    n_prime_d1 = norm.pdf(d1)  # standard normal PDF at d1

    if config.option_type == "call":
        price = config.S0 * norm.cdf(d1) - config.K * disc * norm.cdf(d2)
        delta = norm.cdf(d1)
        rho   = config.K * config.T * disc * norm.cdf(d2)
    else:  # put
        price = config.K * disc * norm.cdf(-d2) - config.S0 * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
        rho   = -config.K * config.T * disc * norm.cdf(-d2)

    # Vega is identical for call and put
    vega = config.S0 * n_prime_d1 * sqrt_T

    return {"price": price, "delta": delta, "vega": vega, "rho": rho}


# ---------------------------------------------------------------------------
# Legacy runner wrapper (kept for compatibility)
# ---------------------------------------------------------------------------

def monte_carlo_european_call(config: EuropeanOptionConfig, ad_mode: str = "none") -> float:
    price, _ = CPUMonteCarloEngine().run(config, ad_mode)
    return price