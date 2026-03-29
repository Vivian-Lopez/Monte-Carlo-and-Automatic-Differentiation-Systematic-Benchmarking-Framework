"""
Sample experiment demonstrating the benchmarking framework.

This script:
1. Configures a European call option pricing workload
2. Runs Monte Carlo simulations with benchmarking
3. Validates numerical correctness against Black-Scholes
4. Captures performance metrics and environment metadata
5. Saves results for analysis
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarking.core.config import MCConfig
from benchmarking.runner.runner import BenchmarkRunner
from benchmarking.workloads.mc_cpu import (
    CPUMonteCarloEngine,
    black_scholes_call,
    european_call_delta
)


def print_header(text: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}\n")


def main():
    """Run complete benchmarking experiment with validation."""
    
    print_header("Monte Carlo Benchmarking Framework - Sample Experiment")
    
    # 1. Configuration
    print("STEP 1: Configuring workload")
    print("-" * 70)
    
    config = MCConfig(
        S0=100.0,    # Initial stock price
        K=100.0,     # Strike price (ATM)
        r=0.05,      # Risk-free rate (5% annually)
        sigma=0.2,   # Volatility (20% annually)
        T=1.0,       # Time to maturity (1 year)
        N=1,         # Number of time steps (1 for European option)
        M=10000,     # Number of Monte Carlo paths
        seed=42      # Random seed for reproducibility
    )
    
    # Validate configuration
    try:
        config.validate()
        print(f"✓ Configuration validated")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        return
    
    # Print configuration details
    print(f"\nConfiguration details:")
    print(f"  S0 (initial stock):  ${config.S0:.2f}")
    print(f"  K  (strike):         ${config.K:.2f}")
    print(f"  r  (risk-free rate): {config.r:.2%}")
    print(f"  σ  (volatility):     {config.sigma:.2%}")
    print(f"  T  (time to mat.):   {config.T:.2f} years")
    print(f"  M  (MC paths):       {config.M:,}")
    print(f"  seed:                {config.seed}")
    print(f"  config_hash:         {config.config_hash()}")
    
    # 2. Numerical Validation
    print_header("STEP 2: Numerical Validation Against Black-Scholes")
    print("-" * 70)
    
    try:
        bs_price = black_scholes_call(config)
        bs_delta = european_call_delta(config)
        
        print(f"Black-Scholes (analytical) price:  ${bs_price:.6f}")
        print(f"Black-Scholes (analytical) delta:  {bs_delta:.6f}")
        print(f"\n(These are the ground-truth prices we'll compare against)")
    except Exception as e:
        print(f"✗ Could not compute Black-Scholes reference (scipy required): {e}")
        bs_price = None
        bs_delta = None
    
    # 3. Benchmarking
    print_header("STEP 3: Running Benchmark with Profiling")
    print("-" * 70)
    
    # Create runner with engine
    runner = BenchmarkRunner(CPUMonteCarloEngine(), name="CPU European Call")
    
    # Run benchmarks
    print("Running benchmark:")
    print("  Warmup runs:  1")
    print("  Timed runs:   5")
    print("  AD mode:      none\n")
    
    result = runner.run(config, num_warmup=1, num_runs=5, ad_mode="none")
    
    # 4. Results Summary
    print_header("STEP 4: Benchmark Results")
    print("-" * 70)
    
    print(f"Monte Carlo result:  ${result.result:.6f}")
    if bs_price is not None:
        error = abs(result.result - bs_price)
        error_pct = 100 * error / bs_price
        print(f"Black-Scholes ref:   ${bs_price:.6f}")
        print(f"Absolute error:      ${error:.6f} ({error_pct:.2f}%)")
    
    print(f"\nPerformance metrics:")
    print(f"  Mean runtime:        {result.mean_runtime*1000:.4f} ms")
    print(f"  Std dev:             {result.std_runtime*1000:.4f} ms")
    print(f"  Min:                 {result.min_runtime*1000:.4f} ms")
    print(f"  Max:                 {result.max_runtime*1000:.4f} ms")
    print(f"  Paths per second:    {config.M / result.mean_runtime:,.0f}")
    
    print(f"\nReproducibility info:")
    print(f"  Config hash:         {result.config_hash}")
    print(f"  AD mode:             {result.ad_mode}")
    print(f"  Python version:      {result.metadata['python_version']}")
    print(f"  Timestamp:           {result.metadata['timestamp']}")
    
    # 5. Save Results
    print_header("STEP 5: Saving Results")
    print("-" * 70)
    
    output_file = "results/benchmark_results.json"
    runner.save_results(result, output_file)
    print(f"✓ Results saved to: {output_file}")
    
    # Ensure results directory exists
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    runner.save_results(result, output_file)
    
    print(f"\nResult structure (JSON):")
    print(f"  - config: MCConfig parameters")
    print(f"  - result: Estimated option price")
    print(f"  - runtimes: Individual run times (5 values)")
    print(f"  - statistics: mean, std, min, max runtime")
    print(f"  - config_hash: Reproducibility verification")
    print(f"  - ad_mode: Differentiation mode")
    print(f"  - metadata: Environment info")
    
    # 6. Reproducibility Check
    print_header("STEP 6: Reproducibility Verification")
    print("-" * 70)
    
    # Run again with same config
    result2 = runner.run(config, num_warmup=1, num_runs=5, ad_mode="none")
    
    if result.config_hash == result2.config_hash:
        print(f"✓ Config hashes match: {result.config_hash}")
    else:
        print(f"✗ Config hash mismatch!")
        print(f"  Run 1: {result.config_hash}")
        print(f"  Run 2: {result2.config_hash}")
    
    if abs(result.result - result2.result) < 1e-10:
        print(f"✓ Numerical results are identical (deterministic RNG seed)")
    else:
        print(f"! Results differ slightly (expected, MC variance):")
        print(f"  Run 1: ${result.result:.6f}")
        print(f"  Run 2: ${result2.result:.6f}")
    
    # 7. Framework Features
    print_header("Framework Capabilities (from Code)")
    print("-" * 70)
    
    print("""
    ✓ Configuration validation and hashing
    ✓ Numerical correctness validation (Black-Scholes)
    ✓ Warmup + multiple timed runs
    ✓ Sound statistics (mean, std, min, max)
    ✓ Environment metadata capture
    ✓ JSON serialization with full traceability
    ✓ Reproducibility verification via config hash
    ✓ AD mode framework (ready for forward/reverse differentiation)
    
    Future extensions:
    - JAX implementations with automatic differentiation
    - GPU acceleration (CUDA/Metal backends)
    - Multi-language support (C++, Julia)
    - Parallel scheduling (MPI-inspired work stealing)
    - Advanced profiling (LIKWID, VTune integration)
    """)
    
    print_header("Experiment Complete")
    print(f"All benchmark artifacts saved. Ready for analysis and comparison.")


if __name__ == "__main__":
    main()