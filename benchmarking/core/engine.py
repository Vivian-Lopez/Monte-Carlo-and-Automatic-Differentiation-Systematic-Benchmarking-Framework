"""Abstract engine interface for Monte Carlo benchmarking."""

from abc import ABC, abstractmethod
from benchmarking.core.config import MCConfig


class MonteCarloEngine(ABC):
    """
    Abstract base class for Monte Carlo simulation engines.
    
    All engines must implement the run() method, which takes a configuration
    and returns a single numerical result (e.g., option price).
    
    This allows swapping different implementations (CPU, GPU, JAX, etc.)
    without changing the runner or result collection logic.
    """
    
    @abstractmethod
    def run(self, config: MCConfig, ad_mode: str = "none") -> float:
        """
        Execute a Monte Carlo simulation.
        
        Args:
            config: Configuration for the simulation
            ad_mode: Differentiation mode ("none", "forward", "reverse")
            
        Returns:
            Numerical result (e.g., estimated option price)
        """
        pass
