"""
CUDA Monte Carlo engine for European option pricing via PyCUDA.

One CUDA thread = one Monte Carlo path.
Randomness is handled by the cuRAND device API (one curandState per thread).
No automatic differentiation is supported in this engine.

Build requirements
------------------
    pip install pycuda
    # CUDA toolkit must be installed and nvcc on PATH
"""

from __future__ import annotations

import math
import numpy as np
from typing import Optional, Tuple

from benchmarking.core.config import WorkloadConfig, EuropeanOptionConfig
from benchmarking.core.engine import MonteCarloEngine, ADMode

# ---------------------------------------------------------------------------
# CUDA kernel source
# ---------------------------------------------------------------------------
# Each thread:
#   1. Initialises its own curandState from (global_seed, thread_index).
#   2. Draws one Z ~ N(0,1) with curand_normal_double.
#   3. Advances the stock price one GBM step (exact, no discretisation error).
#   4. Writes the discounted payoff into global memory at payoffs[idx].
#
# The host then reduces payoffs[] with a plain np.mean — cheap for any
# realistic M and avoids a GPU reduction kernel.
# ---------------------------------------------------------------------------

_KERNEL_SOURCE = r"""
#include <curand_kernel.h>

extern "C" __global__ void mc_european(
    const double  S0,
    const double  K,
    const double  r,
    const double  sigma,
    const double  T,
    const int     is_call,   /* 1 = call, 0 = put */
    const int     M,         /* total number of paths */
    const unsigned long long seed,
    double * __restrict__ payoffs   /* output: one entry per path */
) {
    /* ---- thread → path mapping ---------------------------------------- */
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;   /* guard for last partial block               */

    /* ---- per-thread RNG state ----------------------------------------- */
    /*  curand_init(seed, sequence, offset, state)
     *  Using idx as the sequence number guarantees each thread draws from
     *  a statistically independent sub-sequence of the same generator,
     *  regardless of block/grid layout.                                    */
    curandState_t state;
    curand_init(seed, (unsigned long long)idx, 0ULL, &state);

    /* ---- one GBM step (exact log-normal) -------------------------------- */
    const double Z   = curand_normal_double(&state);          /* Z ~ N(0,1) */
    const double S_T = S0 * exp(
                           (r - 0.5 * sigma * sigma) * T
                           + sigma * sqrt(T) * Z
                       );

    /* ---- payoff --------------------------------------------------------- */
    const double raw    = is_call ? (S_T - K) : (K - S_T);
    payoffs[idx] = (raw > 0.0) ? raw : 0.0;
}
"""

# CUDA block size — 256 is a standard choice that occupies a warp multiple
# and leaves register pressure manageable given curandState overhead.
_BLOCK_SIZE = 256


class CUDAMonteCarloEngine(MonteCarloEngine):
    """
    PyCUDA-backed Monte Carlo engine for European option pricing.

    Design notes
    ------------
    * Kernel compilation happens once on first use (lazy) and is cached on
      the instance.  Subsequent calls pay no compilation cost.
    * All option parameters are passed as scalar kernel arguments; only the
      payoffs array is allocated in GPU global memory.
    * The CPU handles the final reduction (mean + discount) — at typical M
      values (<10 M paths) the transfer is negligible (<80 MB for float64)
      and keeps the implementation simple.
    """

    def __init__(self) -> None:
        self._module = None   # compiled pycuda.compiler.SourceModule, lazy

    # ------------------------------------------------------------------
    # MonteCarloEngine interface
    # ------------------------------------------------------------------

    def supports(self, workload_type: str) -> bool:
        return workload_type == "european"

    def supported_ad_modes(self) -> Tuple[ADMode, ...]:
        return ("none",)

    def run(
        self, config: WorkloadConfig, ad_mode: ADMode = "none"
    ) -> Tuple[float, Optional[dict]]:
        if ad_mode != "none":
            raise NotImplementedError(
                f"CUDAMonteCarloEngine does not support ad_mode={ad_mode!r}."
            )
        if not isinstance(config, EuropeanOptionConfig):
            raise NotImplementedError(
                f"CUDAMonteCarloEngine only supports EuropeanOptionConfig, "
                f"got {type(config).__name__}"
            )

        price = self._price_european(config)
        return (price, None)

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _get_kernel(self):
        """Lazily compile and cache the CUDA kernel."""
        if self._module is not None:
            return self._module

        try:
            import pycuda.autoinit          # noqa: F401 — initialises device
            import pycuda.compiler as compiler
        except ImportError as exc:
            raise RuntimeError(
                "PyCUDA is not installed. Install it with: pip install pycuda\n"
                "Also ensure the CUDA toolkit is installed and nvcc is on PATH."
            ) from exc

        # no_extern_c=True because curand_kernel.h uses C++ internally
        self._module = compiler.SourceModule(
            _KERNEL_SOURCE,
            no_extern_c=True,
            options=["--use_fast_math"],   # safe for exp/sqrt approximations
        )
        return self._module

    def _price_european(self, config: EuropeanOptionConfig) -> float:
        import pycuda.driver as cuda

        kernel = self._get_kernel().get_function("mc_european")

        M        = config.M
        is_call  = np.int32(1 if config.option_type == "call" else 0)
        seed     = np.uint64(config.seed)

        # ---- allocate GPU output buffer ---------------------------------
        # float64 (double) matches the CPU engine's precision.
        payoffs_gpu = cuda.mem_alloc(M * np.dtype(np.float64).itemsize)

        # ---- grid geometry ----------------------------------------------
        # Enough blocks so that blockDim.x * gridDim.x >= M.
        # The kernel guards against out-of-range threads.
        grid_x = math.ceil(M / _BLOCK_SIZE)

        # ---- launch kernel ----------------------------------------------
        # All scalar args are passed by value; pycuda marshals Python/numpy
        # scalars to the correct C types.
        kernel(
            np.float64(config.S0),
            np.float64(config.K),
            np.float64(config.r),
            np.float64(config.sigma),
            np.float64(config.T),
            is_call,
            np.int32(M),
            seed,
            payoffs_gpu,
            block=(_BLOCK_SIZE, 1, 1),
            grid=(grid_x, 1, 1),
        )

        # ---- copy results back to CPU and reduce ------------------------
        payoffs_cpu = np.empty(M, dtype=np.float64)
        cuda.memcpy_dtoh(payoffs_cpu, payoffs_gpu)

        discount = math.exp(-config.r * config.T)
        return float(discount * payoffs_cpu.mean())
