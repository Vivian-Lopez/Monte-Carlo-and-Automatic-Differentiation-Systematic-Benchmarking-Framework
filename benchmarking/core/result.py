from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import statistics
import json
from .config import WorkloadConfig, config_from_dict

@dataclass
class BenchmarkResult:
    """
    Complete benchmark result with configuration, numerical result, timings, and statistics.
    
    Designed for reproducibility and analysis:
    - Stores the exact configuration used
    - Tracks all run times for post-processing
    - Computes summary statistics
    - Captures environment metadata
    """
    config: WorkloadConfig
    result: float  # Estimated option price
    runtimes: List[float]  # Individual run times (seconds)
    mean_runtime: float
    std_runtime: float
    min_runtime: float
    max_runtime: float
    config_hash: str  # For reproducibility verification
    metadata: Dict[str, Any]  # Environment info: version, timestamp, etc.
    ad_mode: str = "none"  # none, forward, or reverse
    ad_overhead_ratio: float = 1.0  # gradient_time / baseline_time
    gradient_time_ms: float = 0.0  # milliseconds
    baseline_mean_ms: float = 0.0  # baseline (no-AD) mean runtime in ms
    memory_peak_mb: float = 0.0  # megabytes
    ad_accuracy_error: float = 0.0  # relative error vs. analytical
    throughput_paths_per_sec: float = 0.0  # paths per second
    greeks: Optional[Dict[str, float]] = None
    
    @staticmethod
    def from_runs(
        config: WorkloadConfig,
        result: float,
        runtimes: List[float],
        config_hash: str,
        metadata: Dict[str, Any],
        ad_mode: str = "none",
        ad_overhead_ratio: float = 1.0,
        gradient_time_ms: float = 0.0,
        baseline_mean_ms: float = 0.0,
        memory_peak_mb: float = 0.0,
        ad_accuracy_error: float = 0.0,
        throughput_paths_per_sec: float = 0.0,
        greeks: Dict[str, float] = None,
    ) -> "BenchmarkResult":
        """Construct result from raw run data, computing statistics."""
        if not runtimes:
            raise ValueError("runtimes list cannot be empty")
        
        return BenchmarkResult(
            config=config,
            result=result,
            runtimes=runtimes,
            mean_runtime=statistics.mean(runtimes),
            std_runtime=statistics.stdev(runtimes) if len(runtimes) > 1 else 0.0,
            min_runtime=min(runtimes),
            max_runtime=max(runtimes),
            config_hash=config_hash,
            metadata=metadata,
            ad_mode=ad_mode,
            ad_overhead_ratio=ad_overhead_ratio,
            gradient_time_ms=gradient_time_ms,
            baseline_mean_ms=baseline_mean_ms,
            memory_peak_mb=memory_peak_mb,
            ad_accuracy_error=ad_accuracy_error,
            throughput_paths_per_sec=throughput_paths_per_sec,
            greeks=greeks,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "config": self.config.to_dict(),
            "result": self.result,
            "runtimes": self.runtimes,
            "statistics": {
                "mean_runtime": self.mean_runtime,
                "std_runtime": self.std_runtime,
                "min_runtime": self.min_runtime,
                "max_runtime": self.max_runtime,
            },
            "config_hash": self.config_hash,
            "ad_mode": self.ad_mode,
            "metadata": self.metadata,
            "ad_metrics": {
                "ad_overhead_ratio": self.ad_overhead_ratio,
                "gradient_time_ms": self.gradient_time_ms,
                "memory_peak_mb": self.memory_peak_mb,
                "ad_accuracy_error": self.ad_accuracy_error,
            },
            "greeks": self.greeks,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "BenchmarkResult":
        """Reconstruct from dictionary."""
        config = config_from_dict(data["config"])
        ad_metrics = data.get("ad_metrics", {})
        return BenchmarkResult.from_runs(
            config=config,
            result=data["result"],
            runtimes=data["runtimes"],
            config_hash=data["config_hash"],
            metadata=data["metadata"],
            ad_mode=data.get("ad_mode", "none"),
            ad_overhead_ratio=ad_metrics.get("ad_overhead_ratio", 1.0),
            gradient_time_ms=ad_metrics.get("gradient_time_ms", 0.0),
            memory_peak_mb=ad_metrics.get("memory_peak_mb", 0.0),
            ad_accuracy_error=ad_metrics.get("ad_accuracy_error", 0.0),
            greeks=data.get("greeks"),
        )