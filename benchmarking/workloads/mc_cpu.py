import math
import random
import numpy as np
from scipy.stats import norm
from typing import List, Optional, Tuple
from benchmarking.core.config import (
    WorkloadConfig, EuropeanOptionConfig, EuropeanLocalVolConfig, AsianOptionConfig,
)
from benchmarking.core.engine import MonteCarloEngine, ADMode


class CPUMonteCarloEngine(MonteCarloEngine):
    """
    Pure-Python / NumPy CPU implementation supporting multiple option workloads.

    Dispatch is done by inspecting config.workload_type, so adding a new
    workload only requires adding a new _price_<type> method here.
    """

    SUPPORTED = {"european", "european_local_vol", "asian"}

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
        elif config.workload_type == "european_local_vol":
            price = price_european_local_vol(
                S0=config.S0, K=config.K, r=config.r, T=config.T,
                M=config.M, N=config.N, sigma_min=config.sigma_min,
                theta=config.theta, option_type=config.option_type,
                seed=config.seed,
            )
            return (price, None)
        elif config.workload_type == "asian":
            return (self._price_asian(config), None)
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

    # ------------------------------------------------------------------
    # Asian arithmetic-average option (GBM log-Euler, N steps)
    # ------------------------------------------------------------------

    def _price_asian(self, config: AsianOptionConfig) -> float:
        rng = np.random.default_rng(config.seed)
        dt = config.T / config.N
        # Z shape: (N, M) — each column is one path
        Z = rng.standard_normal((config.N, config.M))
        log_drift = (config.r - 0.5 * config.sigma ** 2) * dt
        log_vol   = config.sigma * math.sqrt(dt)
        log_S = np.log(config.S0) + np.cumsum(log_drift + log_vol * Z, axis=0)
        S = np.exp(log_S)  # shape (N, M)
        A = S.mean(axis=0)  # arithmetic average over time steps, shape (M,)
        if config.option_type == "call":
            payoff = np.maximum(A - config.K, 0.0)
        else:
            payoff = np.maximum(config.K - A, 0.0)
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


# ===========================================================================
# Numerically stable softplus
# ===========================================================================

def softplus_np(z: np.ndarray) -> np.ndarray:
    """Element-wise softplus: log(1 + exp(z)), numerically stable.

    Uses: max(z, 0) + log1p(exp(-|z|))
    This avoids overflow for large positive z and catastrophic cancellation
    for large negative z.
    """
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


# ===========================================================================
# 4-parameter local volatility surface
# ===========================================================================

def local_vol(
    S: np.ndarray,
    t: float,
    S0: float,
    sigma_min: float,
    theta: List[float],
) -> np.ndarray:
    """
    Evaluate the 4-parameter local volatility sigma(S, t; theta).

        x   = log(S / S0)
        raw = a0 + a1*x + a2*x^2 + b1*t
        sigma = sigma_min + softplus(raw)

    Parameters
    ----------
    S        : current asset prices, shape (M,)
    t        : current time (years)
    S0       : initial asset price (used to normalise moneyness x)
    sigma_min: volatility floor (ensures sigma > sigma_min > 0)
    theta    : [a0, a1, a2, b1]

    Returns
    -------
    sigma : shape (M,), dtype float64
    """
    a0, a1, a2, b1 = theta
    x = np.log(S / S0)
    raw = a0 + a1 * x + a2 * (x ** 2) + b1 * t
    return sigma_min + softplus_np(raw)


# ===========================================================================
# Core pricing function — European option under local vol
# ===========================================================================

def price_european_local_vol(
    S0:          float,
    K:           float,
    r:           float,
    T:           float,
    M:           int,
    N:           int,
    sigma_min:   float,
    theta:       List[float],
    option_type: str   = "call",
    seed:        int   = 42,
) -> float:
    """
    Price a European option under a 4-parameter parametric local volatility
    model using log-Euler Monte Carlo discretisation.

    Dynamics:
        dS = r S dt + sigma(S, t; theta) S dW

    Log-Euler step:
        S_{n+1} = S_n * exp((r - 0.5*sigma_n^2)*dt + sigma_n*sqrt(dt)*Z_n)

    Returns the discounted expected payoff as a float, consistent with the
    MonteCarloEngine.run() contract.  BenchmarkRunner owns timing and SE.
    """
    dt      = T / N
    sqrt_dt = math.sqrt(dt)

    rng = np.random.default_rng(seed)
    Z   = rng.standard_normal((N, M))          # shape (N, M), float64

    S = np.full(M, S0, dtype=np.float64)

    for n in range(N):
        t_n     = n * dt
        sigma_n = local_vol(S, t_n, S0, sigma_min, theta)
        S       = S * np.exp(
            (r - 0.5 * sigma_n ** 2) * dt + sigma_n * sqrt_dt * Z[n]
        )

    if option_type == "call":
        payoff = np.maximum(S - K, 0.0)
    else:
        payoff = np.maximum(K - S, 0.0)

    disc = math.exp(-r * T)
    return float(disc * payoff.mean())


# ===========================================================================
# Constant-vol Black-Scholes validation helper
# ===========================================================================

def black_scholes_local_vol_constant(
    S0:        float,
    K:         float,
    r:         float,
    T:         float,
    sigma_min: float,
    a0:        float,
    option_type: str = "call",
) -> float:
    """
    Black-Scholes price for the constant-vol case of the local vol model.

    When a1 = a2 = b1 = 0, sigma is spatially and temporally flat:
        sigma = sigma_min + softplus(a0)

    This is used as a sanity check: the MC price should converge to this
    value as M increases.
    """
    sigma_const = sigma_min + (max(a0, 0.0) + math.log1p(math.exp(-abs(a0))))

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma_const ** 2) * T) / (sigma_const * sqrt_T)
    d2 = d1 - sigma_const * sqrt_T
    disc = math.exp(-r * T)

    if option_type == "call":
        return S0 * norm.cdf(d1) - K * disc * norm.cdf(d2)
    else:
        return K * disc * norm.cdf(-d2) - S0 * norm.cdf(-d1)