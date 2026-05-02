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
import jax.numpy as jnp
import numpy as np
from typing import Callable, Optional, Tuple
from benchmarking.core.config import (
    WorkloadConfig, EuropeanOptionConfig,
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

    European option pricing with forward- and reverse-mode AD.
    Greeks computed are Delta (dP/dS0), Vega (dP/dσ), and Rho (dP/dr).
    """

    SUPPORTED = {"european"}

    def supports(self, workload_type: str) -> bool:
        return workload_type in self.SUPPORTED

    def supported_ad_modes(self) -> Tuple[ADMode, ...]:
        return ("none", "forward", "reverse")

    def run(self, config: WorkloadConfig, ad_mode: ADMode = "none") -> Tuple[float, Optional[dict]]:
        if config.workload_type == "european":
            return self._run_european(config, ad_mode)
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


# ---------------------------------------------------------------------------
# Legacy function alias
# ---------------------------------------------------------------------------

def monte_carlo_european_call_jax(config, ad_mode: str = "none") -> float:
    price, _ = JAXMonteCarloEngine().run(config, ad_mode)
    return price

