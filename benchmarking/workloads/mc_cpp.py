"""
C++ Monte Carlo engine wrapper.

Delegates European option pricing to the compiled cpp_mc extension module
which uses an OpenMP-parallelised Mersenne Twister implementation.

Build the extension first:
    cd benchmarking/cpp
    pip install -e . --no-build-isolation
or:
    python benchmarking/cpp/setup.py build_ext --inplace
"""

from typing import Optional, Tuple
from benchmarking.core.config import (
    WorkloadConfig, EuropeanOptionConfig, EuropeanLocalVolConfig,
)
from benchmarking.core.engine import MonteCarloEngine, ADMode


class CPPMonteCarloEngine(MonteCarloEngine):
    """
    OpenMP-parallelised C++ Monte Carlo engine.

    Supported workloads: european, european_local_vol.
    AD modes are not supported.
    """

    def supports(self, workload_type: str) -> bool:
        return workload_type in {"european", "european_local_vol"}

    def supported_ad_modes(self) -> Tuple[ADMode, ...]:
        return ("none",)

    def run(self, config: WorkloadConfig, ad_mode: ADMode = "none") -> Tuple[float, Optional[dict]]:
        if ad_mode != "none":
            raise NotImplementedError(
                f"CPPMonteCarloEngine does not support ad_mode={ad_mode!r}."
            )

        try:
            import cpp_mc  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "cpp_mc extension not found. "
                "Build it with: cd benchmarking/cpp && pip install -e . --no-build-isolation"
            ) from exc

        if config.workload_type == "european":
            if not isinstance(config, EuropeanOptionConfig):
                raise TypeError(f"Expected EuropeanOptionConfig, got {type(config).__name__}")
            is_call = 1 if config.option_type == "call" else 0
            price = cpp_mc.price_european(
                config.S0, config.K, config.r, config.sigma,
                config.T, config.M, config.seed, is_call,
            )

        elif config.workload_type == "european_local_vol":
            if not isinstance(config, EuropeanLocalVolConfig):
                raise TypeError(f"Expected EuropeanLocalVolConfig, got {type(config).__name__}")
            is_call = 1 if config.option_type == "call" else 0
            a0, a1, a2, b1 = config.theta
            price = cpp_mc.price_european_local_vol(
                config.S0, config.K, config.r, config.T,
                config.M, config.N, config.sigma_min,
                a0, a1, a2, b1,
                config.seed, is_call,
            )

        else:
            raise NotImplementedError(
                f"CPPMonteCarloEngine does not support workload '{config.workload_type}'"
            )

        return (float(price), None)
