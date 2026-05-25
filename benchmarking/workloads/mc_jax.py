"""
JAX-based Monte Carlo engine with automatic differentiation support.

All pricing kernels accept scalar float arguments (S0, K, r, sigma, T)
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
jax.config.update("jax_enable_x64", True)  # needed for float64 local-vol paths
import jax.numpy as jnp
import numpy as np
from typing import Callable, List, Optional, Tuple
from benchmarking.core.config import (
    WorkloadConfig, EuropeanOptionConfig, EuropeanLocalVolConfig, AsianOptionConfig,
)
from benchmarking.core.engine import MonteCarloEngine, ADMode


# ---------------------------------------------------------------------------
# Module-level JIT kernel for European local vol option.
# Z has shape (N, M); theta components are scalar inputs so JAX can
# differentiate through them.  option_type is static.
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("option_type", "N"))
def _european_lv_kernel_jit(
    S0:        float,
    K:         float,
    r:         float,
    T:         float,
    sigma_min: float,
    a0:        float,
    a1:        float,
    a2:        float,
    b1:        float,
    Z:         jnp.ndarray,   # shape (N, M)
    N:         int,
    option_type: str = "call",
) -> jnp.ndarray:
    """JIT-compiled log-Euler MC under 4-parameter local vol.

    theta = [a0, a1, a2, b1] is unpacked to individual scalars so that
    jax.grad / jax.jvp can differentiate w.r.t. each component.
    """
    dt      = T / N
    sqrt_dt = jnp.sqrt(dt)

    def step(S, z_and_t):
        z, t_n = z_and_t
        x       = jnp.log(S / S0)
        raw     = a0 + a1 * x + a2 * x ** 2 + b1 * t_n
        sigma_n = sigma_min + jnp.maximum(raw, 0.0) + jnp.log1p(jnp.exp(-jnp.abs(raw)))
        S_next  = S * jnp.exp((r - 0.5 * sigma_n ** 2) * dt + sigma_n * sqrt_dt * z)
        return S_next, None

    t_grid = jnp.arange(N, dtype=jnp.float64) * dt   # shape (N,)
    S_T, _ = jax.lax.scan(step, jnp.full((Z.shape[1],), S0), (Z, t_grid))

    if option_type == "call":
        payoff = jnp.maximum(S_T - K, 0.0)
    else:
        payoff = jnp.maximum(K - S_T, 0.0)
    return jnp.exp(-r * T) * jnp.mean(payoff)


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

    European option pricing with forward- and reverse-mode AD.
    Greeks computed are Delta (dP/dS0), Vega (dP/dσ), and Rho (dP/dr).
    """

    SUPPORTED = {"european", "european_local_vol", "asian"}

    def supports(self, workload_type: str) -> bool:
        return workload_type in self.SUPPORTED

    def supported_ad_modes(self) -> Tuple[ADMode, ...]:
        return ("none", "forward", "reverse")

    def run(self, config: WorkloadConfig, ad_mode: ADMode = "none") -> Tuple[float, Optional[dict]]:
        if config.workload_type == "european":
            return self._run_european(config, ad_mode)
        elif config.workload_type == "european_local_vol":
            return self._run_european_local_vol(config, ad_mode)
        elif config.workload_type == "asian":
            return self._run_asian(config, ad_mode)
        else:
            raise NotImplementedError(f"JAXMonteCarloEngine does not support '{config.workload_type}'")

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
    # European local vol
    # ------------------------------------------------------------------

    def _run_european_local_vol(
        self, config: EuropeanLocalVolConfig, ad_mode: str
    ) -> Tuple[float, Optional[dict]]:
        key = jax.random.PRNGKey(int(config.seed))
        N   = int(config.N)
        M   = int(config.M)
        Z   = jax.random.normal(key, shape=(N, M), dtype=jnp.float64)  # (N, M)

        S0, K, r, T       = float(config.S0), float(config.K), float(config.r), float(config.T)
        sigma_min         = float(config.sigma_min)
        a0, a1, a2, b1    = [float(v) for v in config.theta]
        opt               = config.option_type

        def price_fn(
            S0_: float, a0_: float, a1_: float, a2_: float,
            b1_: float, sigma_min_: float,
        ) -> jnp.ndarray:
            return _european_lv_kernel_jit(
                S0_, K, r, T, sigma_min_, a0_, a1_, a2_, b1_, Z, N, opt
            )

        price = float(jax.block_until_ready(
            price_fn(S0, a0, a1, a2, b1, sigma_min)
        ))

        greeks = None
        if ad_mode != "none":
            greeks = self._compute_lv_greeks(
                price_fn, S0, a0, a1, a2, b1, sigma_min, ad_mode
            )
        return (price, greeks)

    @staticmethod
    def _compute_lv_greeks(
        price_fn: Callable,
        S0: float,
        a0: float, a1: float, a2: float, b1: float,
        sigma_min: float,
        ad_mode: str,
    ) -> dict:
        """Differentiate price w.r.t. S0 and theta = [a0, a1, a2, b1] and sigma_min."""
        argnums = (0, 1, 2, 3, 4, 5)   # S0, a0, a1, a2, b1, sigma_min
        primals = (S0, a0, a1, a2, b1, sigma_min)

        if ad_mode == "reverse":
            grad_fn = jax.grad(price_fn, argnums=argnums)
            grads   = grad_fn(*primals)
        else:  # forward
            n = len(primals)
            grads = []
            for i in range(n):
                tangent = tuple(1.0 if j == i else 0.0 for j in range(n))
                _, g = jax.jvp(price_fn, primals, tangent)
                grads.append(g)

        keys = ("delta", "d_a0", "d_a1", "d_a2", "d_b1", "d_sigma_min")
        return {k: float(v) for k, v in zip(keys, grads)}

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

    # ------------------------------------------------------------------
    # Asian arithmetic-average option (GBM log-Euler, N steps)
    # ------------------------------------------------------------------

    def _run_asian(self, config: AsianOptionConfig, ad_mode: str) -> Tuple[float, Optional[dict]]:
        key = jax.random.PRNGKey(int(config.seed))
        N, M = int(config.N), int(config.M)
        # Z shape: (N, M)
        Z = jax.random.normal(key, shape=(N, M))
        K, T, opt = float(config.K), float(config.T), config.option_type
        S0, r, sigma = float(config.S0), float(config.r), float(config.sigma)

        def price_fn(S0_: float, r_: float, sigma_: float) -> jnp.ndarray:
            dt = T / N
            log_drift = (r_ - 0.5 * sigma_ ** 2) * dt
            log_vol   = sigma_ * jnp.sqrt(dt)
            # scan over time steps: carry = log(S_t), accumulate path sum
            def step(log_s, z):
                log_s_next = log_s + log_drift + log_vol * z
                return log_s_next, jnp.exp(log_s_next)
            log_S0 = jnp.full((M,), jnp.log(S0_))
            _, S_path = jax.lax.scan(step, log_S0, Z)  # S_path shape (N, M)
            A = jnp.mean(S_path, axis=0)  # arithmetic average per path
            if opt == "call":
                payoff = jnp.maximum(A - K, 0.0)
            else:
                payoff = jnp.maximum(K - A, 0.0)
            return jnp.exp(-r_ * T) * jnp.mean(payoff)

        price = float(jax.block_until_ready(price_fn(S0, r, sigma)))
        greeks = self._compute_greeks(price_fn, S0, r, sigma, ad_mode) if ad_mode != "none" else None
        return (price, greeks)


# ---------------------------------------------------------------------------
# Legacy function alias
# ---------------------------------------------------------------------------

def monte_carlo_european_call_jax(config, ad_mode: str = "none") -> float:
    price, _ = JAXMonteCarloEngine().run(config, ad_mode)
    return price

