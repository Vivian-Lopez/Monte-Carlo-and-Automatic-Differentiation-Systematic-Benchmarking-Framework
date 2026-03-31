"""Abstract engine interface for Monte Carlo benchmarking."""

from abc import ABC, abstractmethod
from benchmarking.core.config import WorkloadConfig


class MonteCarloEngine(ABC):
    """
    Abstract base class for Monte Carlo simulation engines.

    Engines receive a WorkloadConfig subclass; they should inspect
    config.workload_type to dispatch to the correct simulation logic.
    """

    @abstractmethod
    def run(self, config: WorkloadConfig, ad_mode: str = "none") -> float:
        """
        Execute a Monte Carlo simulation.

        Args:
            config: WorkloadConfig subclass (European, Asian, Barrier, Basket, …)
            ad_mode: Differentiation mode ("none", "forward", "reverse")

        Returns:
            Numerical result (estimated option price)
        """
        pass

    def supports(self, workload_type: str) -> bool:
        """Return True if this engine supports the given workload type.
        Override in subclasses to declare supported workloads."""
        return workload_type == "european"
