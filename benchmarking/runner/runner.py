import time
import json
import platform
import sys
from typing import Union, Callable, List
from datetime import datetime
from benchmarking.core.config import WorkloadConfig
from benchmarking.core.result import BenchmarkResult
from benchmarking.core.engine import MonteCarloEngine

try:
    import psutil as _psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

try:
    import numpy as _np
    _NUMPY_VERSION = _np.__version__
except ImportError:
    _NUMPY_VERSION = "unavailable"

try:
    import jax as _jax
    _JAX_VERSION = _jax.__version__
except ImportError:
    _JAX_VERSION = "unavailable"


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
    
    def __init__(self, engine: Union[MonteCarloEngine, Callable], name: str = "unnamed"):
        """
        Initialize the runner with a Monte Carlo engine.
        
        Args:
            engine: MonteCarloEngine instance, or legacy Callable for backwards compatibility
                   (Callable should match signature: (MCConfig, str) -> float)
            name: Human-readable name for the workload (for metadata)
        """
        self.engine = engine
        self.name = name
    
    @staticmethod
    def capture_environment() -> dict:
        """
        Capture environment metadata for reproducibility verification.

        Records Python version, platform, architecture, CPU info, and
        framework versions (NumPy, JAX) for cross-environment comparison.
        """
        cpu_count = None
        cpu_model = platform.processor() or "unknown"
        memory_gb = None
        if _PSUTIL_AVAILABLE:
            try:
                cpu_count = _psutil.cpu_count(logical=True)
                memory_gb = round(_psutil.virtual_memory().total / (1024 ** 3), 2)
            except Exception:
                pass
        return {
            "timestamp": datetime.now().isoformat(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "processor": cpu_model,
            "cpu_model": cpu_model,
            "cpu_architecture": platform.machine(),
            "cpu_count": cpu_count,
            "memory_gb": memory_gb,
            "numpy_version": _NUMPY_VERSION,
            "jax_version": _JAX_VERSION,
            "framework_version": "0.1.0",
        }
    
    def run(
        self,
        config: WorkloadConfig,
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
        
        For AD modes (forward/reverse), also measures baseline (no-AD) performance
        to compute the AD overhead ratio.
        
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
        
        # Run with the specified AD mode
        runtimes, result, greeks, memory_peak_mb = self._run_with_timings(config, num_warmup, num_runs, ad_mode)
        
        # If AD mode is enabled, also measure baseline (no-AD) to compute overhead
        baseline_runtimes = None
        if ad_mode != "none":
            baseline_runtimes, _, _, _ = self._run_with_timings(config, num_warmup, num_runs, "none")
        
        # Capture environment and compute config hash
        metadata = self.capture_environment()
        config_hash = config.config_hash()
        
        # Compute AD overhead ratio and baseline mean if we have a baseline
        ad_overhead_ratio = 1.0
        baseline_mean_ms = None
        if baseline_runtimes is not None:
            baseline_mean = sum(baseline_runtimes) / len(baseline_runtimes)
            baseline_mean_ms = baseline_mean * 1000.0
            ad_mean = sum(runtimes) / len(runtimes)
            ad_overhead_ratio = ad_mean / baseline_mean if baseline_mean > 0 else 1.0

        # Throughput: paths per second using mean runtime
        mean_s = sum(runtimes) / len(runtimes)
        throughput = config.M / mean_s if mean_s > 0 else 0.0

        # Create result with full statistics
        return BenchmarkResult.from_runs(
            config=config,
            result=result,
            runtimes=runtimes,
            config_hash=config_hash,
            metadata=metadata,
            ad_mode=ad_mode,
            ad_overhead_ratio=ad_overhead_ratio,
            baseline_mean_ms=baseline_mean_ms,
            memory_peak_mb=memory_peak_mb,
            greeks=greeks,
            throughput_paths_per_sec=throughput,
        )
    
    def _run_with_timings(
        self,
        config: WorkloadConfig,
        num_warmup: int,
        num_runs: int,
        ad_mode: str
    ) -> tuple:
        """
        Run benchmark with given ad_mode and return timings + result + greeks + memory.
        
        Returns:
            (runtimes: List[float], result: float, greeks: dict | None, memory_peak_mb: float)
        """
        # Warmup runs (not timed, results discarded)
        for _ in range(num_warmup):
            if isinstance(self.engine, MonteCarloEngine):
                self.engine.run(config, ad_mode)
            else:
                self.engine(config, ad_mode)
        
        # Snapshot RSS before timed runs
        _proc = None
        if _PSUTIL_AVAILABLE:
            try:
                _proc = _psutil.Process()
                _rss_before = _proc.memory_info().rss
            except Exception:
                _proc = None

        # Timed runs
        runtimes: List[float] = []
        result: float = 0.0
        greeks = None
        _rss_peak = 0

        for _ in range(num_runs):
            start = time.perf_counter()
            if isinstance(self.engine, MonteCarloEngine):
                res = self.engine.run(config, ad_mode)
            else:
                res = self.engine(config, ad_mode)
            end = time.perf_counter()
            runtimes.append(end - start)
            if isinstance(res, tuple):
                result, greeks = res
            else:
                result = res
            if _proc is not None:
                try:
                    _rss_peak = max(_rss_peak, _proc.memory_info().rss)
                except Exception:
                    pass

        # Peak memory above baseline (in MB)
        if _proc is not None and _rss_peak > 0:
            memory_peak_mb = (_rss_peak - _rss_before) / (1024 * 1024)
            memory_peak_mb = max(memory_peak_mb, 0.0)
        else:
            memory_peak_mb = 0.0

        return runtimes, result, greeks, memory_peak_mb
    
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