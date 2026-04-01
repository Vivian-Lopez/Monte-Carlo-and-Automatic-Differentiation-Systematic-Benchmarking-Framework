import math
import random
import numpy as np
from scipy.stats import norm
from benchmarking.core.config import (
    WorkloadConfig, EuropeanOptionConfig, AsianOptionConfig,
    BarrierOptionConfig, BasketOptionConfig,
)
from benchmarking.core.engine import MonteCarloEngine


class CPUMonteCarloEngine(MonteCarloEngine):
    """
    Pure-Python / NumPy CPU implementation supporting multiple option workloads.

    Dispatch is done by inspecting config.workload_type, so adding a new
    workload only requires adding a new _price_<type> method here.
    """

    SUPPORTED = {"european", "asian", "barrier", "basket"}

    def supports(self, workload_type: str) -> bool:
        return workload_type in self.SUPPORTED

    def run(self, config: WorkloadConfig, ad_mode: str = "none") -> float:
        wtype = config.workload_type
        if wtype == "european":
            return self._price_european(config)
        elif wtype == "asian":
            return self._price_asian(config)
        elif wtype == "barrier":
            return self._price_barrier(config)
        elif wtype == "basket":
            return self._price_basket(config)
        else:
            raise NotImplementedError(f"CPUMonteCarloEngine does not support workload '{wtype}'")

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
    # Asian option  (arithmetic / geometric average, multi-step)
    # ------------------------------------------------------------------

    def _price_asian(self, config: AsianOptionConfig) -> float:
        rng = np.random.default_rng(config.seed)
        dt = config.T / config.N
        Z = rng.standard_normal((config.M, config.N))
        # Build path matrix  (M × N)
        log_returns = (config.r - 0.5 * config.sigma ** 2) * dt + \
                      config.sigma * math.sqrt(dt) * Z
        log_S = np.log(config.S0) + np.cumsum(log_returns, axis=1)
        S_paths = np.exp(log_S)
        # Average
        if config.averaging == "arithmetic":
            avg = S_paths.mean(axis=1)
        else:
            avg = np.exp(np.log(S_paths).mean(axis=1))  # geometric
        if config.option_type == "call":
            payoff = np.maximum(avg - config.K, 0.0)
        else:
            payoff = np.maximum(config.K - avg, 0.0)
        return float(math.exp(-config.r * config.T) * payoff.mean())

    # ------------------------------------------------------------------
    # Barrier option  (knock-in / knock-out, up / down, multi-step)
    # ------------------------------------------------------------------

    def _price_barrier(self, config: BarrierOptionConfig) -> float:
        rng = np.random.default_rng(config.seed)
        dt = config.T / config.N
        Z = rng.standard_normal((config.M, config.N))
        log_returns = (config.r - 0.5 * config.sigma ** 2) * dt + \
                      config.sigma * math.sqrt(dt) * Z
        log_S = np.log(config.S0) + np.cumsum(log_returns, axis=1)
        S_paths = np.exp(log_S)
        S_T = S_paths[:, -1]

        if config.barrier_side == "up":
            breached = (S_paths >= config.B).any(axis=1)
        else:
            breached = (S_paths <= config.B).any(axis=1)

        if config.option_type == "call":
            vanilla = np.maximum(S_T - config.K, 0.0)
        else:
            vanilla = np.maximum(config.K - S_T, 0.0)

        if config.barrier_type == "knock_out":
            payoff = np.where(breached, 0.0, vanilla)
        else:  # knock_in
            payoff = np.where(breached, vanilla, 0.0)

        return float(math.exp(-config.r * config.T) * payoff.mean())

    # ------------------------------------------------------------------
    # Basket option  (equal-weight, correlated GBM, multi-step)
    # ------------------------------------------------------------------

    def _price_basket(self, config: BasketOptionConfig) -> float:
        rng = np.random.default_rng(config.seed)
        n = config.n_assets
        dt = config.T / config.N
        # Build correlation matrix
        rho_matrix = config.rho * np.ones((n, n)) + (1 - config.rho) * np.eye(n)
        L = np.linalg.cholesky(rho_matrix)
        # Z: (M, N, n)  — correlated normals
        Z_ind = rng.standard_normal((config.M, config.N, n))
        Z_corr = Z_ind @ L.T
        log_ret = (config.r - 0.5 * config.sigma ** 2) * dt + \
                  config.sigma * math.sqrt(dt) * Z_corr
        # S_paths: (M, N+1, n)
        log_S = np.log(config.S0) + np.cumsum(log_ret, axis=1)
        S_T = np.exp(log_S[:, -1, :])           # final prices  (M, n)
        basket_price = S_T.mean(axis=1)         # equal-weight basket
        if config.option_type == "call":
            payoff = np.maximum(basket_price - config.K, 0.0)
        else:
            payoff = np.maximum(config.K - basket_price, 0.0)
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


def european_call_delta(config: EuropeanOptionConfig) -> float:
    """Analytical delta for European call option."""
    d1 = (math.log(config.S0 / config.K) +
          (config.r + 0.5 * config.sigma ** 2) * config.T) / \
         (config.sigma * math.sqrt(config.T))
    return norm.cdf(d1)


# ---------------------------------------------------------------------------
# Legacy runner wrapper (kept for compatibility)
# ---------------------------------------------------------------------------

def monte_carlo_european_call(config: EuropeanOptionConfig, ad_mode: str = "none") -> float:
    return CPUMonteCarloEngine().run(config, ad_mode)