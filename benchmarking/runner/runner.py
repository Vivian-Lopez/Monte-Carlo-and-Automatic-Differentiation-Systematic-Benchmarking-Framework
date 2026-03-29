import time
import json
import platform
import sys
from typing import Callable, List
from datetime import datetime
from benchmarking.core.config import MCConfig
from benchmarking.core.result import BenchmarkResult

class BenchmarkRunner:
    """
    Runner for benchmarking Monte Carlo workloads with reproducibility and traceability.
    
    Implements key principles from Carlo.jl:
    - Explicit warm-up + repeated measurements
    - Structured metadata capture for reproducibility
    - Sound summary statistics (mean, std, min, max)
    - Configuration hashing for integrity checking
    
    Designed to be language-agnostic and support multiple engines.
    """
    
    def __init__(self, workload_func: Callable[[MCConfig, str], float], name: str = "unnamed"):
        """
        Initialize the runner with a workload function.
        
        Args:
            workload_func: Function that takes (MCConfig, ad_mode_str) and returns a float result
            name: Human-readable name for the workload (for metadata)
        """
        self.workload_func = workload_func
        self.name = name
    
    @staticmethod
    def capture_environment() -> dict:
        """
        Capture environment metadata for reproducibility verification.
        
        Records:
        - Python version and implementation
        - Platform and architecture
        - Timestamp
        - Framework version
        
        This allows detection of configuration drift across runs.
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "framework_version": "0.1.0",  # Semantic version
        }
    
    def run(
        self,
        config: MCConfig,
        num_warmup: int = 1,
        num_runs: int = 5,
        ad_mode: str = "none"
    ) -> BenchmarkResult:
        """
        Run the benchmark with warmup and timed executions.
        
        Implements best practices from benchmarking literature:
        - Warmup runs to stabilize state (alleviates JIT, cache effects)
        - Multiple timed runs to capture variability
        - Structured result capture with metadata
        
        Args:
            config: Configuration for the Monte Carlo simulation
            num_warmup: Number of warmup runs (discarded) (default: 1)
            num_runs: Number of timed runs used for statistics (default: 5)
            ad_mode: Differentiation mode for framework compatibility (default: "none")
            
        Returns:
            BenchmarkResult containing config, result, timings, statistics, and metadata
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate configuration
        config.validate()
        
        # Warmup runs (not timed, results discarded)
        for _ in range(num_warmup):
            self.workload_func(config, ad_mode)
        
        # Timed runs
        runtimes: List[float] = []
        result: float = 0.0
        
        for _ in range(num_runs):
            start = time.perf_counter()
            res = self.workload_func(config, ad_mode)
            end = time.perf_counter()
            
            runtimes.append(end - start)
            result = res  # Result should be deterministic given seed
        
        # Capture environment and compute config hash
        metadata = self.capture_environment()
        config_hash = config.config_hash()
        
        # Create result with full statistics
        return BenchmarkResult.from_runs(
            config=config,
            result=result,
            runtimes=runtimes,
            config_hash=config_hash,
            metadata=metadata,
            ad_mode=ad_mode
        )
    
    def save_results(self, result: BenchmarkResult, filename: str) -> None:
        """
        Save benchmark results to a JSON file with full traceability.
        
        Format:
        {
            "config": {...},
            "result": <price>,
            "runtimes": [<t1>, <t2>, ...],
            "statistics": {...},
            "config_hash": <hash>,
            "ad_mode": "none",
            "metadata": {...}
        }
        
        Args:
            result: BenchmarkResult to save
            filename: Output filename (JSON)
        """
        with open(filename, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)