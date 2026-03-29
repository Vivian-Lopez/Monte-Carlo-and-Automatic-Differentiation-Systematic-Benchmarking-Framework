"""
Generate multiple benchmark runs for testing the dashboard.

This script runs the benchmark with different M (path count) values.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from benchmarking.core.config import MCConfig
from benchmarking.runner.runner import BenchmarkRunner
from benchmarking.workloads.mc_cpu import monte_carlo_european_call


def generate_runs():
    """Generate multiple benchmark runs with different path counts."""
    
    base_config = {
        'S0': 100.0,
        'K': 100.0,
        'r': 0.05,
        'sigma': 0.2,
        'T': 1.0,
        'N': 1,
        'seed': 42,
    }
    
    runner = BenchmarkRunner(monte_carlo_european_call, name="CPU European Call")
    
    # Different path counts to benchmark
    path_counts = [1000, 5000, 10000, 50000]
    
    print("=" * 70)
    print("Generating Multiple Benchmark Runs")
    print("=" * 70)
    print()
    
    for i, M in enumerate(path_counts, 1):
        print(f"[{i}/{len(path_counts)}] Running benchmark with M={M:,} paths...")
        
        config = MCConfig(M=M, **base_config)
        config.validate()
        
        # Run benchmark
        result = runner.run(config, num_warmup=1, num_runs=5, ad_mode="none")
        
        # Save with descriptive filename
        filename = f"benchmark_run_m{M:06d}.json"
        runner.save_results(result, filename)
        
        print(f"   ✓ Saved to: {filename}")
        print(f"   Price: ${result.result:.6f}")
        print(f"   Mean time: {result.mean_runtime*1000:.2f} ms")
        print(f"   Throughput: {M / result.mean_runtime / 1e6:.2f}M paths/s")
        print()
    
    print("=" * 70)
    print(f"✓ Generated {len(path_counts)} benchmark runs")
    print("  Open the Streamlit dashboard to visualize and compare:")
    print("    streamlit run fyp/frontend/app.py")
    print("=" * 70)


if __name__ == "__main__":
    generate_runs()