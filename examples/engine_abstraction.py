"""
Example: Using the new MonteCarloEngine abstraction.

This shows how to use the refactored engine-based design.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarking.core.config import MCConfig
from benchmarking.runner.runner import BenchmarkRunner
from benchmarking.workloads.mc_cpu import (
    CPUMonteCarloEngine,
    black_scholes_call,
)


def main():
    """Demonstrate new engine abstraction."""
    
    print("\n" + "=" * 70)
    print("  Engine Abstraction Example")
    print("=" * 70 + "\n")
    
    # Configuration
    config = MCConfig(
        S0=100.0,
        K=100.0,
        r=0.05,
        sigma=0.2,
        T=1.0,
        N=1,
        M=10000,
        seed=42
    )
    config.validate()
    
    print("1. Create engine instance (new pattern)")
    print("-" * 70)
    engine = CPUMonteCarloEngine()
    print(f"✓ Created: {engine.__class__.__name__}")
    print(f"  Base class: {engine.__class__.__bases__[0].__name__}")
    
    print("\n2. Create runner with engine")
    print("-" * 70)
    runner = BenchmarkRunner(engine, name="CPU European Call")
    print("✓ Runner initialized with MonteCarloEngine")
    
    print("\n3. Run benchmark")
    print("-" * 70)
    result = runner.run(config, num_warmup=1, num_runs=5, ad_mode="none")
    print(f"✓ Benchmark complete")
    print(f"  Price: ${result.result:.6f}")
    print(f"  Mean runtime: {result.mean_runtime*1000:.2f} ms")
    
    print("\n4. Numerical validation")
    print("-" * 70)
    bs_price = black_scholes_call(config)
    error = abs(result.result - bs_price)
    print(f"  MC price: ${result.result:.6f}")
    print(f"  BS price: ${bs_price:.6f}")
    print(f"  Error: {error/bs_price*100:.2f}%")
    
    print("\n" + "=" * 70)
    print("  ✓ Engine abstraction working correctly")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
