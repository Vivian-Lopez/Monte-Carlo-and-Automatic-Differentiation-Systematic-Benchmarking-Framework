from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import statistics
import json
from .config import MCConfig

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
    config: MCConfig
    result: float  # Estimated option price
    runtimes: List[float]  # Individual run times (seconds)
    mean_runtime: float
    std_runtime: float
    min_runtime: float
    max_runtime: float
    config_hash: str  # For reproducibility verification
    metadata: Dict[str, Any]  # Environment info: version, timestamp, etc.
    ad_mode: str = "none"  # none, forward, or reverse
    
    @staticmethod
    def from_runs(
        config: MCConfig,
        result: float,
        runtimes: List[float],
        config_hash: str,
        metadata: Dict[str, Any],
        ad_mode: str = "none"
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
            ad_mode=ad_mode
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "config": asdict(self.config),
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
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "BenchmarkResult":
        """Reconstruct from dictionary."""
        config = MCConfig(**data["config"])
        return BenchmarkResult.from_runs(
            config=config,
            result=data["result"],
            runtimes=data["runtimes"],
            config_hash=data["config_hash"],
            metadata=data["metadata"],
            ad_mode=data.get("ad_mode", "none")
        )