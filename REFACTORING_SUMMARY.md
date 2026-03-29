"""
REFACTORING SUMMARY: Engine Abstraction

Key code changes and updated signatures.
"""

# ============================================================================
# 1. NEW ABSTRACT ENGINE CLASS
# ============================================================================
# File: benchmarking/core/engine.py

from abc import ABC, abstractmethod
from benchmarking.core.config import MCConfig

class MonteCarloEngine(ABC):
    """Abstract base class for Monte Carlo simulation engines."""
    
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


# ============================================================================
# 2. CONCRETE ENGINE IMPLEMENTATION
# ============================================================================
# File: benchmarking/workloads/mc_cpu.py (excerpt)

import math
import random
from benchmarking.core.engine import MonteCarloEngine

class CPUMonteCarloEngine(MonteCarloEngine):
    """Pure Python CPU implementation of Monte Carlo simulation."""
    
    def run(self, config: MCConfig, ad_mode: str = "none") -> float:
        """Execute geometric Brownian motion simulation on CPU."""
        random.seed(config.seed)
        payoff_sum = 0.0
        
        for _ in range(config.M):
            Z = random.gauss(0, 1)
            S_T = config.S0 * math.exp(
                (config.r - 0.5 * config.sigma**2) * config.T + 
                config.sigma * math.sqrt(config.T) * Z
            )
            payoff = max(S_T - config.K, 0)
            payoff_sum += payoff
        
        price = math.exp(-config.r * config.T) * payoff_sum / config.M
        return price


# Legacy function still available:
def monte_carlo_european_call(config: MCConfig, ad_mode: str = "none") -> float:
    """Legacy function interface (for backwards compatibility)."""
    # ... implementation unchanged


# ============================================================================
# 3. UPDATED RUNNER (forwards-compatible)
# ============================================================================
# File: benchmarking/runner/runner.py (key changes)

from typing import Union, Callable, List
from benchmarking.core.engine import MonteCarloEngine
from benchmarking.core.config import MCConfig
from benchmarking.core.result import BenchmarkResult

class BenchmarkRunner:
    """Benchmark runner supporting both engine and callable patterns."""
    
    def __init__(self, engine: Union[MonteCarloEngine, Callable], name: str = "unnamed"):
        """
        Accept both MonteCarloEngine instances and legacy callables.
        
        Args:
            engine: MonteCarloEngine instance OR Callable[[MCConfig, str], float]
            name: Workload name for metadata
        """
        self.engine = engine
        self.name = name
    
    def run(
        self,
        config: MCConfig,
        num_warmup: int = 1,
        num_runs: int = 5,
        ad_mode: str = "none"
    ) -> BenchmarkResult:
        """Run benchmark with warmup and timed executions."""
        config.validate()
        
        # Warmup (discarded)
        for _ in range(num_warmup):
            if isinstance(self.engine, MonteCarloEngine):
                self.engine.run(config, ad_mode)
            else:
                # Legacy: treat as Callable
                self.engine(config, ad_mode)
        
        # Timed runs
        runtimes: List[float] = []
        result: float = 0.0
        
        for _ in range(num_runs):
            start = time.perf_counter()
            if isinstance(self.engine, MonteCarloEngine):
                res = self.engine.run(config, ad_mode)
            else:
                # Legacy: treat as Callable
                res = self.engine(config, ad_mode)
            end = time.perf_counter()
            
            runtimes.append(end - start)
            result = res
        
        # Capture metadata and create result
        metadata = self.capture_environment()
        config_hash = config.config_hash()
        
        return BenchmarkResult.from_runs(
            config=config,
            result=result,
            runtimes=runtimes,
            config_hash=config_hash,
            metadata=metadata,
            ad_mode=ad_mode
        )
    
    # ... rest of runner unchanged


# ============================================================================
# 4. EXAMPLE: NEW PATTERN (Recommended)
# ============================================================================

from benchmarking.runner.runner import BenchmarkRunner
from benchmarking.workloads.mc_cpu import CPUMonteCarloEngine
from benchmarking.core.config import MCConfig

# Create engine
engine = CPUMonteCarloEngine()

# Create runner
runner = BenchmarkRunner(engine, name="CPU European Call")

# Configure and run
config = MCConfig(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, N=1, M=10000, seed=42)
result = runner.run(config, num_warmup=1, num_runs=5)

print(f"Price: ${result.result:.6f}")
print(f"Mean time: {result.mean_runtime*1000:.2f} ms")


# ============================================================================
# 5. EXAMPLE: LEGACY PATTERN (Still Works)
# ============================================================================

from benchmarking.runner.runner import BenchmarkRunner
from benchmarking.workloads.mc_cpu import monte_carlo_european_call

# Direct function (backwards compatible)
runner = BenchmarkRunner(monte_carlo_european_call)

# Same config and run
config = MCConfig(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, N=1, M=10000, seed=42)
result = runner.run(config)

print(f"Price: ${result.result:.6f}")

# ============================================================================
# SUMMARY
# ============================================================================
"""
CHANGES:
✓ Added: benchmarking/core/engine.py     (MonteCarloEngine ABC)
✓ Added: CPUMonteCarloEngine class        (in mc_cpu.py)
✓ Modified: BenchmarkRunner.__init__      (accepts Union[MonteCarloEngine, Callable])
✓ Modified: BenchmarkRunner.run()         (handles both patterns)
✓ Added: examples/engine_abstraction.py   (demonstration)
✓ Added: ENGINE_REFACTORING.md            (documentation)

UNCHANGED:
✓ MCConfig
✓ BenchmarkResult
✓ Result JSON format
✓ monte_carlo_european_call() function (legacy)
✓ black_scholes_call(), european_call_delta()
✓ experiments/run_experiment.py
✓ experiments/generate_runs.py
✓ Streamlit dashboard

BACKWARDS COMPATIBLE:
✓ Legacy scripts using monte_carlo_european_call still work
✓ No breaking changes to existing code
✓ New code can use MonteCarloEngine abstraction

EXTENSIBLE:
✓ Can add JAX, GPU, or other engines by inheriting MonteCarloEngine
✓ Runner automatically supports new engines
✓ No changes needed to runner or result handling
"""
