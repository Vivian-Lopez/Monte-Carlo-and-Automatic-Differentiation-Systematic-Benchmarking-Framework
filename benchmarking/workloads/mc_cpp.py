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
from benchmarking.core.config import WorkloadConfig, EuropeanOptionConfig
from benchmarking.core.engine import MonteCarloEngine, ADMode


class CPPMonteCarloEngine(MonteCarloEngine):
    """
    OpenMP-parallelised C++ Monte Carlo engine.

    Supported workloads: european only (call and put).
    AD modes are not supported.
    """

    def supports(self, workload_type: str) -> bool:
        return workload_type == "european"

    def supported_ad_modes(self) -> Tuple[ADMode, ...]:
        return ("none",)

    def run(self, config: WorkloadConfig, ad_mode: ADMode = "none") -> Tuple[float, Optional[dict]]:
        if ad_mode != "none":
            raise NotImplementedError(
                f"CPPMonteCarloEngine does not support ad_mode={ad_mode!r}."
            )
        if not isinstance(config, EuropeanOptionConfig):
            raise NotImplementedError(
                f"CPPMonteCarloEngine only supports EuropeanOptionConfig, "
                f"got {type(config).__name__}"
            )

        try:
            from cpp_mc import price_european  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "cpp_mc extension not found. "
                "Build it with: cd benchmarking/cpp && pip install -e . --no-build-isolation"
            ) from exc

        is_call = 1 if config.option_type == "call" else 0

        return (price_european(
            config.S0,
            config.K,
            config.r,
            config.sigma,
            config.T,
            config.M,
            config.seed,
            is_call,
        ), None)
