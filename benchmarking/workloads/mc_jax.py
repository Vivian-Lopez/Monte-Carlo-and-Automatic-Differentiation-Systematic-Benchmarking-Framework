"""
JAX-based Monte Carlo engine with automatic differentiation support.

Dispatches on config.workload_type; adding a new workload requires only
adding a new _price_<type>_kernel and _run_<type> method here.

All pricing kernels accept scalar float arguments (S0, K, r, sigma, T, ...)
so that JAX can differentiate through them.  The shared _compute_greeks()
helper applies forward- or reverse-mode AD to any such function.

JAX / JIT notes
---------------
The European kernel is a module-level @jax.jit function so that JAX traces
and compiles it exactly once per (shape, dtype) signature.  Because Z is
passed as an argument (not captured in a per-call closure) the compiled XLA
computation is reused across every timed repetition.  The BenchmarkRunner's
warmup call triggers the first compilation; all timed calls hit the cache.

Reverse-mode AD correctness
----------------------------
One jax.grad call with argnums=(0, 1, 2) performs a single backward pass and
returns all three partial derivatives simultaneously.  The previous approach
of three separate jax.grad calls with argnums=0/1/2 was correct but launched
three independent backward passes, tripling the AD cost.
"""

import functools
import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Optional, Tuple
from benchmarking.core.config import (
    WorkloadConfig, EuropeanOptionConfig, AsianOptionConfig,
    BarrierOptionConfig, BasketOptionConfig,
)
from benchmarking.core.engine import MonteCarloEngine, ADMode


# ---------------------------------------------------------------------------
# Module-level JIT kernel for European option.
# Z, K, T are passed as arguments so JAX can cache the compiled computation
# across repeated calls with the same shapes.
# option_type is declared static so JAX emits specialised code for call/put.
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("option_type",))
def _european_kernel_jit(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    Z: jnp.ndarray,
    option_type: str = "call",
) -> jnp.ndarray:
    """JIT-compiled European GBM pricing kernel (scalar params + path array)."""
    S_T = S0 * jnp.exp((r - 0.5 * sigma ** 2) * T + sigma * jnp.sqrt(T) * Z)
    if option_type == "call":
        payoff = jnp.maximum(S_T - K, 0.0)
    else:
        payoff = jnp.maximum(K - S_T, 0.0)
    return jnp.exp(-r * T) * jnp.mean(payoff)


class JAXMonteCarloEngine(MonteCarloEngine):
    """
    JAX-based vectorised Monte Carlo engine with full AD support.

    All four workloads (European, Asian, Barrier, Basket) support forward-
    and reverse-mode AD.  Greeks computed are Delta (dP/dS0), Vega (dP/dσ),
    and Rho (dP/dr).

    Note on Barrier AD: the barrier condition uses jnp.where, which is
    differentiable but produces zero gradient at paths that cross the
    barrier.  This is equivalent to the pathwise estimator approach and
    is well-defined for smooth payoffs away from the barrier.
    """

    SUPPORTED = {"european", "asian", "barrier", "basket"}

    def supports(self, workload_type: str) -> bool:
        return workload_type in self.SUPPORTED

    def supported_ad_modes(self) -> Tuple[ADMode, ...]:
        return ("none", "forward", "reverse")

    def run(self, config: WorkloadConfig, ad_mode: ADMode = "none") -> Tuple[float, Optional[dict]]:
        wtype = config.workload_type
        if wtype == "european":
            return self._run_european(config, ad_mode)
        elif wtype == "asian":
            return self._run_asian(config, ad_mode)
        elif wtype == "barrier":
            return self._run_barrier(config, ad_mode)
        elif wtype == "basket":
            return self._run_basket(config, ad_mode)
        else:
            raise NotImplementedError(f"JAXMonteCarloEngine does not support '{wtype}'")

    # ------------------------------------------------------------------
    # Shared AD helper
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_greeks(
        price_fn: Callable,
        S0: float,
        r: float,
        sigma: float,
        ad_mode: str,
    ) -> dict:
        """
        Apply forward- or reverse-mode AD to price_fn(S0, r, sigma).

        price_fn must be a pure JAX function of exactly (S0, r, sigma) — all
        other parameters (Z, K, T, …) captured via closure.

        Reverse mode: one jax.grad call with argnums=(0, 1, 2) performs a
        single backward pass and returns (d/dS0, d/dr, d/dsigma) together.

        Forward mode: three jax.jvp calls, each probing one input direction.
        This is cheaper than reverse when the number of inputs is small (here 3)
        and avoids memory overhead of storing the full computation graph.
        """
        if ad_mode == "reverse":
            # Single backward pass — correct and efficient
            grad_fn = jax.grad(price_fn, argnums=(0, 1, 2))
            d_S0, d_r, d_sig = grad_fn(S0, r, sigma)
        else:  # forward
            primals = (S0, r, sigma)
            _, d_S0 = jax.jvp(price_fn, primals, (1.0, 0.0, 0.0))
            _, d_r   = jax.jvp(price_fn, primals, (0.0, 1.0, 0.0))
            _, d_sig = jax.jvp(price_fn, primals, (0.0, 0.0, 1.0))
        return {"delta": float(d_S0), "rho": float(d_r), "vega": float(d_sig)}

    # ------------------------------------------------------------------
    # European
    # ------------------------------------------------------------------

    def _run_european(self, config: EuropeanOptionConfig, ad_mode: str) -> Tuple[float, Optional[dict]]:
        key = jax.random.PRNGKey(int(config.seed))
        Z = jax.random.normal(key, shape=(int(config.M),))
        K, T, opt = float(config.K), float(config.T), config.option_type
        S0, r, sigma = float(config.S0), float(config.r), float(config.sigma)

        # price_fn closes over Z, K, T, opt and delegates to the module-level
        # JIT kernel so JAX reuses the compiled computation across timed runs.
        def price_fn(S0_: float, r_: float, sigma_: float) -> jnp.ndarray:
            return _european_kernel_jit(S0_, K, r_, sigma_, T, Z, opt)

        price = float(jax.block_until_ready(price_fn(S0, r, sigma)))
        greeks = self._compute_greeks(price_fn, S0, r, sigma, ad_mode) if ad_mode != "none" else None
        return (price, greeks)

    @staticmethod
    def _price_european_kernel(S0, K, r, sigma, T, Z, option_type="call"):
        """Non-JIT reference kernel (used by Asian/Barrier/Basket if needed)."""
        S_T = S0 * jnp.exp((r - 0.5 * sigma ** 2) * T + sigma * jnp.sqrt(T) * Z)
        payoff = jnp.maximum(S_T - K, 0.0) if option_type == "call" else jnp.maximum(K - S_T, 0.0)
        return jnp.exp(-r * T) * jnp.mean(payoff)

    # ------------------------------------------------------------------
    # Asian  (arithmetic / geometric, multi-step)
    # ------------------------------------------------------------------

    def _run_asian(self, config: AsianOptionConfig, ad_mode: str) -> Tuple[float, Optional[dict]]:
        key = jax.random.PRNGKey(int(config.seed))
        Z = jax.random.normal(key, shape=(int(config.M), int(config.N)))
        K, T, N = float(config.K), float(config.T), int(config.N)
        S0, r, sigma = float(config.S0), float(config.r), float(config.sigma)
        averaging, opt = config.averaging, config.option_type

        def price_fn(S0_, r_, sigma_):
            return self._price_asian_kernel(S0_, K, r_, sigma_, T, N, averaging, opt, Z)

        price = float(price_fn(S0, r, sigma))
        greeks = self._compute_greeks(price_fn, S0, r, sigma, ad_mode) if ad_mode != "none" else None
        return (price, greeks)

    @staticmethod
    def _price_asian_kernel(S0, K, r, sigma, T, N, averaging, option_type, Z):
        dt = T / N
        log_ret = (r - 0.5 * sigma ** 2) * dt + sigma * jnp.sqrt(dt) * Z
        log_S = jnp.log(S0) + jnp.cumsum(log_ret, axis=1)
        S_paths = jnp.exp(log_S)
        avg = S_paths.mean(axis=1) if averaging == "arithmetic" \
            else jnp.exp(jnp.log(S_paths).mean(axis=1))
        payoff = jnp.maximum(avg - K, 0.0) if option_type == "call" else jnp.maximum(K - avg, 0.0)
        return jnp.exp(-r * T) * jnp.mean(payoff)

    # ------------------------------------------------------------------
    # Barrier  (knock-in / knock-out, up / down, multi-step)
    # ------------------------------------------------------------------

    def _run_barrier(self, config: BarrierOptionConfig, ad_mode: str) -> Tuple[float, Optional[dict]]:
        key = jax.random.PRNGKey(int(config.seed))
        Z = jax.random.normal(key, shape=(int(config.M), int(config.N)))
        K, B, T, N = float(config.K), float(config.B), float(config.T), int(config.N)
        S0, r, sigma = float(config.S0), float(config.r), float(config.sigma)
        btype, bside, opt = config.barrier_type, config.barrier_side, config.option_type

        def price_fn(S0_, r_, sigma_):
            return self._price_barrier_kernel(S0_, K, B, r_, sigma_, T, N, btype, bside, opt, Z)

        price = float(price_fn(S0, r, sigma))
        greeks = self._compute_greeks(price_fn, S0, r, sigma, ad_mode) if ad_mode != "none" else None
        return (price, greeks)

    @staticmethod
    def _price_barrier_kernel(S0, K, B, r, sigma, T, N, barrier_type, barrier_side, option_type, Z):
        dt = T / N
        log_ret = (r - 0.5 * sigma ** 2) * dt + sigma * jnp.sqrt(dt) * Z
        log_S = jnp.log(S0) + jnp.cumsum(log_ret, axis=1)
        S_paths = jnp.exp(log_S)
        S_T = S_paths[:, -1]

        breached = (S_paths >= B).any(axis=1) if barrier_side == "up" \
            else (S_paths <= B).any(axis=1)
        vanilla = jnp.maximum(S_T - K, 0.0) if option_type == "call" \
            else jnp.maximum(K - S_T, 0.0)
        payoff = jnp.where(breached, 0.0, vanilla) if barrier_type == "knock_out" \
            else jnp.where(breached, vanilla, 0.0)
        return jnp.exp(-r * T) * jnp.mean(payoff)

    # ------------------------------------------------------------------
    # Basket  (equal-weight, correlated GBM, multi-step)
    # ------------------------------------------------------------------

    def _run_basket(self, config: BasketOptionConfig, ad_mode: str) -> Tuple[float, Optional[dict]]:
        key = jax.random.PRNGKey(int(config.seed))
        n = int(config.n_assets)
        rho_matrix = config.rho * np.ones((n, n)) + (1 - config.rho) * np.eye(n)
        L = jnp.array(np.linalg.cholesky(rho_matrix))  # precomputed — not differentiated
        Z_ind = jax.random.normal(key, shape=(int(config.M), int(config.N), n))
        Z_corr = Z_ind @ L.T
        K, T, N, opt = float(config.K), float(config.T), int(config.N), config.option_type
        S0, r, sigma = float(config.S0), float(config.r), float(config.sigma)

        def price_fn(S0_, r_, sigma_):
            return self._price_basket_kernel(S0_, K, r_, sigma_, T, N, opt, Z_corr)

        price = float(price_fn(S0, r, sigma))
        greeks = self._compute_greeks(price_fn, S0, r, sigma, ad_mode) if ad_mode != "none" else None
        return (price, greeks)

    @staticmethod
    def _price_basket_kernel(S0, K, r, sigma, T, N, option_type, Z_corr):
        dt = T / N
        log_ret = (r - 0.5 * sigma ** 2) * dt + sigma * jnp.sqrt(dt) * Z_corr
        log_S = jnp.log(S0) + jnp.cumsum(log_ret, axis=1)
        S_T = jnp.exp(log_S[:, -1, :])
        basket = S_T.mean(axis=1)
        payoff = jnp.maximum(basket - K, 0.0) if option_type == "call" \
            else jnp.maximum(K - basket, 0.0)
        return jnp.exp(-r * T) * jnp.mean(payoff)


# ---------------------------------------------------------------------------
# Legacy function alias
# ---------------------------------------------------------------------------

def monte_carlo_european_call_jax(config, ad_mode: str = "none") -> float:
    price, _ = JAXMonteCarloEngine().run(config, ad_mode)
    return price

