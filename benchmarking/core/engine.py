"""Abstract engine interface for Monte Carlo benchmarking."""

from abc import ABC, abstractmethod
from typing import Literal, Tuple, Optional
from benchmarking.core.config import WorkloadConfig

ADMode = Literal["none", "forward", "reverse"]


class MonteCarloEngine(ABC):
    """
    Abstract base class for Monte Carlo simulation engines.

    Engines receive a WorkloadConfig subclass; they should inspect
    config.workload_type to dispatch to the correct simulation logic.

    ``run()`` returns a ``(price, greeks_or_None)`` tuple so that
    differentiation results travel with the price instead of through
    a mutable side-channel.
    """

    @abstractmethod
    def run(
        self, config: WorkloadConfig, ad_mode: ADMode = "none"
    ) -> Tuple[float, Optional[dict]]:
        """
        Execute a Monte Carlo simulation.

        Args:
            config: WorkloadConfig subclass (European, Asian, Barrier, Basket, …)
            ad_mode: Differentiation mode ("none", "forward", "reverse")

        Returns:
            (price, greeks) — greeks is None when ad_mode is "none" or
            unsupported by this engine.
        """
        pass

    def supports(self, workload_type: str) -> bool:
        """Return True if this engine supports the given workload type.
        Override in subclasses to declare supported workloads."""
        return workload_type == "european"

    def supported_ad_modes(self) -> Tuple[ADMode, ...]:
        """Return the AD modes this engine can execute.

        The default implementation declares only ``"none"``.  Override in
        engines that implement forward- or reverse-mode AD.
        """
        return ("none",)
