"""
Rust Monte Carlo engine wrapper.

Delegates European option pricing to the compiled rust_mc extension module,
which uses Rayon-parallelised path simulation with SmallRng (Xoshiro128++).

Build the extension first (requires Rust + cargo):
    cd benchmarking/rust
    maturin develop --release
or install as a wheel:
    maturin build --release && pip install target/wheels/rust_mc-*.whl
"""

from typing import Optional, Tuple
from benchmarking.core.config import (
    WorkloadConfig, EuropeanOptionConfig, EuropeanLocalVolConfig,
)
from benchmarking.core.engine import MonteCarloEngine, ADMode


class RustMonteCarloEngine(MonteCarloEngine):
    """
    Rayon-parallelised Rust Monte Carlo engine.

    Supported workloads: european, european_local_vol.
    AD modes are not supported (no autodiff in the Rust kernel).
    """

    def supports(self, workload_type: str) -> bool:
        return workload_type in {"european", "european_local_vol"}

    def supported_ad_modes(self) -> Tuple[ADMode, ...]:
        return ("none",)

    def run(self, config: WorkloadConfig, ad_mode: ADMode = "none") -> Tuple[float, Optional[dict]]:
        if ad_mode != "none":
            raise NotImplementedError(
                f"RustMonteCarloEngine does not support ad_mode={ad_mode!r}."
            )

        try:
            import rust_mc  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "rust_mc extension not found. "
                "Build it with: cd benchmarking/rust && maturin develop --release"
            ) from exc

        if config.workload_type == "european":
            if not isinstance(config, EuropeanOptionConfig):
                raise TypeError(f"Expected EuropeanOptionConfig, got {type(config).__name__}")
            price = rust_mc.price_european(
                config.S0, config.K, config.r, config.sigma,
                config.T, config.M, config.seed,
                config.option_type == "call",
            )

        elif config.workload_type == "european_local_vol":
            if not isinstance(config, EuropeanLocalVolConfig):
                raise TypeError(f"Expected EuropeanLocalVolConfig, got {type(config).__name__}")
            a0, a1, a2, b1 = config.theta
            price = rust_mc.price_european_local_vol(
                config.S0, config.K, config.r, config.T,
                config.M, config.N, config.sigma_min,
                a0, a1, a2, b1,
                config.seed,
                config.option_type == "call",
            )

        else:
            raise NotImplementedError(
                f"RustMonteCarloEngine does not support workload '{config.workload_type}'"
            )

        return (float(price), None)
