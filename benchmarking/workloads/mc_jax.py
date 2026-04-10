"""
JAX-based Monte Carlo engine with automatic differentiation support.

Dispatches on config.workload_type; adding a new workload requires only
adding a new _price_<type> method here plus the config class in config.py.
"""

import jax
import jax.numpy as jnp
from jax import grad
from benchmarking.core.config import (
    WorkloadConfig, EuropeanOptionConfig, AsianOptionConfig,
    BarrierOptionConfig, BasketOptionConfig,
)
from benchmarking.core.engine import MonteCarloEngine


class JAXMonteCarloEngine(MonteCarloEngine):
    """
    JAX-based vectorised Monte Carlo engine with optional AD.

    Supported workloads: european, asian, barrier, basket.
    AD (forward / reverse) is wired for European options;
    other workloads fall back to no-AD pricing.

    After a run with AD, the computed Greeks are stored in self.last_greeks.
    """

    SUPPORTED = {"european", "asian", "barrier", "basket"}

    def __init__(self):
        self.last_greeks: dict | None = None

    def supports(self, workload_type: str) -> bool:
        return workload_type in self.SUPPORTED

    def run(self, config: WorkloadConfig, ad_mode: str = "none") -> float:
        self.last_greeks = None
        wtype = config.workload_type
        if wtype == "european":
            return self._run_european(config, ad_mode)
        elif wtype == "asian":
            return float(self._price_asian(config))
        elif wtype == "barrier":
            return float(self._price_barrier(config))
        elif wtype == "basket":
            return float(self._price_basket(config))
        else:
            raise NotImplementedError(f"JAXMonteCarloEngine does not support '{wtype}'")

    # ------------------------------------------------------------------
    # European  (with AD support)
    # ------------------------------------------------------------------

    def _run_european(self, config: EuropeanOptionConfig, ad_mode: str) -> float:
        key = jax.random.PRNGKey(int(config.seed))
        Z = jax.random.normal(key, shape=(int(config.M),))

        if ad_mode == "none":
            return float(self._price_european(
                float(config.S0), float(config.K), float(config.r),
                float(config.sigma), float(config.T), Z, config.option_type
            ))
        elif ad_mode == "reverse":
            greeks = self._compute_greeks_reverse(config, Z)
            self.last_greeks = greeks
            return float(self._price_european(
                float(config.S0), float(config.K), float(config.r),
                float(config.sigma), float(config.T), Z, config.option_type
            ))
        elif ad_mode == "forward":
            greeks = self._compute_greeks_forward(config, Z)
            self.last_greeks = greeks
            return float(self._price_european(
                float(config.S0), float(config.K), float(config.r),
                float(config.sigma), float(config.T), Z, config.option_type
            ))
        else:
            raise ValueError(f"Unknown ad_mode: {ad_mode!r}")

    @staticmethod
    def _price_european(S0, K, r, sigma, T, Z, option_type="call"):
        """Price European option given pre-generated standard normals Z."""
        S_T = S0 * jnp.exp((r - 0.5 * sigma ** 2) * T + sigma * jnp.sqrt(T) * Z)
        if option_type == "call":
            payoff = jnp.maximum(S_T - K, 0.0)
        else:
            payoff = jnp.maximum(K - S_T, 0.0)
        return jnp.exp(-r * T) * jnp.mean(payoff)

    def _compute_greeks_reverse(self, config: EuropeanOptionConfig, Z) -> dict:
        """Compute Delta, Vega, Rho via reverse-mode AD (jax.grad)."""
        _K, _T, _opt = float(config.K), float(config.T), config.option_type

        def price_fn(S0, r, sigma):
            return self._price_european(S0, _K, r, sigma, _T, Z, _opt)

        S0, r, sigma = float(config.S0), float(config.r), float(config.sigma)
        d_S0 = float(grad(price_fn, argnums=0)(S0, r, sigma))
        d_r   = float(grad(price_fn, argnums=1)(S0, r, sigma))
        d_sig = float(grad(price_fn, argnums=2)(S0, r, sigma))
        return {"delta": d_S0, "rho": d_r, "vega": d_sig}

    def _compute_greeks_forward(self, config: EuropeanOptionConfig, Z) -> dict:
        """Compute Delta, Vega, Rho via forward-mode AD (jax.jvp)."""
        _K, _T, _opt = float(config.K), float(config.T), config.option_type
        S0, r, sigma = float(config.S0), float(config.r), float(config.sigma)
        primals = (S0, r, sigma)

        def price_fn(S0_arg, r_arg, sigma_arg):
            return self._price_european(S0_arg, _K, r_arg, sigma_arg, _T, Z, _opt)

        # jvp: one tangent vector per Greek
        _, d_S0 = jax.jvp(price_fn, primals, (1.0, 0.0, 0.0))
        _, d_r   = jax.jvp(price_fn, primals, (0.0, 1.0, 0.0))
        _, d_sig = jax.jvp(price_fn, primals, (0.0, 0.0, 1.0))
        return {"delta": float(d_S0), "rho": float(d_r), "vega": float(d_sig)}

    # ------------------------------------------------------------------
    # Asian  (arithmetic / geometric, multi-step)
    # ------------------------------------------------------------------

    def _price_asian(self, config: AsianOptionConfig):
        key = jax.random.PRNGKey(config.seed)
        dt = config.T / config.N
        Z = jax.random.normal(key, shape=(config.M, config.N))
        log_ret = (config.r - 0.5 * config.sigma ** 2) * dt + \
                  config.sigma * jnp.sqrt(dt) * Z
        log_S = jnp.log(config.S0) + jnp.cumsum(log_ret, axis=1)
        S_paths = jnp.exp(log_S)
        if config.averaging == "arithmetic":
            avg = S_paths.mean(axis=1)
        else:
            avg = jnp.exp(jnp.log(S_paths).mean(axis=1))
        if config.option_type == "call":
            payoff = jnp.maximum(avg - config.K, 0.0)
        else:
            payoff = jnp.maximum(config.K - avg, 0.0)
        return jnp.exp(-config.r * config.T) * jnp.mean(payoff)

    # ------------------------------------------------------------------
    # Barrier  (knock-in / knock-out, up / down, multi-step)
    # ------------------------------------------------------------------

    def _price_barrier(self, config: BarrierOptionConfig):
        key = jax.random.PRNGKey(config.seed)
        dt = config.T / config.N
        Z = jax.random.normal(key, shape=(config.M, config.N))
        log_ret = (config.r - 0.5 * config.sigma ** 2) * dt + \
                  config.sigma * jnp.sqrt(dt) * Z
        log_S = jnp.log(config.S0) + jnp.cumsum(log_ret, axis=1)
        S_paths = jnp.exp(log_S)
        S_T = S_paths[:, -1]

        if config.barrier_side == "up":
            breached = (S_paths >= config.B).any(axis=1)
        else:
            breached = (S_paths <= config.B).any(axis=1)

        if config.option_type == "call":
            vanilla = jnp.maximum(S_T - config.K, 0.0)
        else:
            vanilla = jnp.maximum(config.K - S_T, 0.0)

        if config.barrier_type == "knock_out":
            payoff = jnp.where(breached, 0.0, vanilla)
        else:
            payoff = jnp.where(breached, vanilla, 0.0)

        return jnp.exp(-config.r * config.T) * jnp.mean(payoff)

    # ------------------------------------------------------------------
    # Basket  (equal-weight, correlated GBM, multi-step)
    # ------------------------------------------------------------------

    def _price_basket(self, config: BasketOptionConfig):
        import numpy as np  # Cholesky via numpy (one-time setup cost)
        key = jax.random.PRNGKey(config.seed)
        n = config.n_assets
        dt = config.T / config.N
        rho_matrix = config.rho * jnp.ones((n, n)) + (1 - config.rho) * jnp.eye(n)
        L = jnp.array(np.linalg.cholesky(np.array(rho_matrix)))
        Z_ind = jax.random.normal(key, shape=(config.M, config.N, n))
        Z_corr = Z_ind @ L.T
        log_ret = (config.r - 0.5 * config.sigma ** 2) * dt + \
                  config.sigma * jnp.sqrt(dt) * Z_corr
        log_S = jnp.log(config.S0) + jnp.cumsum(log_ret, axis=1)
        S_T = jnp.exp(log_S[:, -1, :])
        basket = S_T.mean(axis=1)
        if config.option_type == "call":
            payoff = jnp.maximum(basket - config.K, 0.0)
        else:
            payoff = jnp.maximum(config.K - basket, 0.0)
        return jnp.exp(-config.r * config.T) * jnp.mean(payoff)


# ---------------------------------------------------------------------------
# Legacy function alias
# ---------------------------------------------------------------------------

def monte_carlo_european_call_jax(config, ad_mode: str = "none") -> float:
    return JAXMonteCarloEngine().run(config, ad_mode)
