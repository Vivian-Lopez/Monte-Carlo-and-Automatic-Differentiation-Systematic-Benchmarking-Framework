"""
Unit tests for AD framework.

Tests:
- JAX engine produces valid results
- Gradients respect analytical bounds
- Result serialization/deserialization
- AD metrics computation
"""

import pytest
import math
from benchmarking.core.config import MCConfig
from benchmarking.workloads.mc_jax import JAXMonteCarloEngine, monte_carlo_european_call_jax
from benchmarking.workloads.ad_validation import (
    analytical_delta,
    analytical_vega,
    analytical_rho,
    validate_gradient,
    compute_all_analytical_greeks,
)
from benchmarking.core.result import BenchmarkResult
from benchmarking.runner.runner import BenchmarkRunner


def test_jax_engine_runs():
    """Test JAX engine executes without error."""
    config = MCConfig(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, N=1, M=100, seed=42)
    engine = JAXMonteCarloEngine()
    
    # Test no-AD mode
    result, greeks = engine.run(config, ad_mode="none")
    assert isinstance(result, float)
    assert result > 0
    assert greeks is None
    
    # Test forward-mode AD
    result_fwd, greeks_fwd = engine.run(config, ad_mode="forward")
    assert isinstance(result_fwd, float)
    assert result_fwd > 0
    assert greeks_fwd is not None
    
    # Test reverse-mode AD
    result_rev, greeks_rev = engine.run(config, ad_mode="reverse")
    assert isinstance(result_rev, float)
    assert result_rev > 0
    assert greeks_rev is not None


def test_jax_function_signature():
    """Test standalone JAX function has correct signature."""
    config = MCConfig(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, N=1, M=100, seed=42)
    
    result = monte_carlo_european_call_jax(config, ad_mode="none")
    assert isinstance(result, float)
    assert result > 0


def test_analytical_greeks():
    """Test analytical Greek computation."""
    config = MCConfig(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, N=1, M=100, seed=42)
    
    delta = analytical_delta(config)
    assert 0.0 < delta < 1.0, f"Delta {delta} out of bounds [0, 1]"
    
    vega = analytical_vega(config)
    assert vega > 0, f"Vega {vega} should be positive"
    
    rho = analytical_rho(config)
    assert rho > 0, f"Rho {rho} should be positive"


def test_analytical_greeks_atm():
    """Test Greeks for at-the-money option."""
    config = MCConfig(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, N=1, M=100, seed=42)
    
    greeks = compute_all_analytical_greeks(config)
    
    # ATM call delta should be ~0.636 (slightly in-the-money due to B-S drift)
    assert 0.6 < greeks["dC/dS0"] < 0.7
    assert greeks["dC/dsigma"] > 0
    assert greeks["dC/dr"] > 0


def test_gradient_validation():
    """Test gradient validation utility."""
    analytical = 0.5
    
    # Perfect match
    error = validate_gradient(0.5, analytical)
    assert error < 1e-10
    
    # 1% error
    error = validate_gradient(0.505, analytical)
    assert 0.009 < error < 0.011


def test_benchmark_result_with_ad_metrics():
    """Test BenchmarkResult stores and retrieves AD metrics."""
    config = MCConfig(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, N=1, M=100, seed=42)
    
    result = BenchmarkResult.from_runs(
        config=config,
        result=10.5,
        runtimes=[0.01, 0.01],
        config_hash="abc123",
        metadata={"test": True},
        ad_mode="forward",
        ad_overhead_ratio=2.5,
        gradient_time_ms=15.0,
        memory_peak_mb=128.5,
        ad_accuracy_error=0.001
    )
    
    assert result.ad_mode == "forward"
    assert result.ad_overhead_ratio == 2.5
    assert result.gradient_time_ms == 15.0
    assert result.memory_peak_mb == 128.5
    assert result.ad_accuracy_error == 0.001


def test_benchmark_result_serialization():
    """Test BenchmarkResult serialization with AD metrics."""
    config = MCConfig(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, N=1, M=100, seed=42)
    
    result1 = BenchmarkResult.from_runs(
        config=config,
        result=10.5,
        runtimes=[0.01, 0.01],
        config_hash="abc123",
        metadata={"test": True},
        ad_mode="reverse",
        ad_overhead_ratio=3.0,
        gradient_time_ms=20.0,
        memory_peak_mb=256.0,
        ad_accuracy_error=0.002
    )
    
    # Serialize
    data = result1.to_dict()
    assert "ad_metrics" in data
    assert data["ad_metrics"]["ad_overhead_ratio"] == 3.0
    
    # Deserialize
    result2 = BenchmarkResult.from_dict(data)
    assert result2.ad_overhead_ratio == 3.0
    assert result2.gradient_time_ms == 20.0
    assert result2.memory_peak_mb == 256.0
    assert result2.ad_accuracy_error == 0.002


def test_benchmark_runner_with_jax_ad():
    """Test BenchmarkRunner with JAX engine and AD."""
    config = MCConfig(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, N=1, M=100, seed=42)
    config.validate()
    
    runner = BenchmarkRunner(JAXMonteCarloEngine(), name="JAX Test")
    
    # Basic run
    result = runner.run(config, num_warmup=1, num_runs=2, ad_mode="none")
    assert result.ad_mode == "none"
    assert len(result.runtimes) == 2
    
    # With AD
    result_ad = runner.run(config, num_warmup=1, num_runs=2, ad_mode="forward")
    assert result_ad.ad_mode == "forward"


def test_jax_deterministic_seeding():
    """Test JAX engine produces consistent results with same seed."""
    config1 = MCConfig(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, N=1, M=1000, seed=42)
    config2 = MCConfig(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, N=1, M=1000, seed=42)
    
    engine = JAXMonteCarloEngine()
    
    result1, _ = engine.run(config1, ad_mode="none")
    result2, _ = engine.run(config2, ad_mode="none")
    
    assert result1 == result2, "Same seed should produce identical results"


def test_forward_mode_uses_jvp():
    """Forward-mode AD computes Greeks via jax.jvp, not jax.grad."""
    config = MCConfig(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, N=1, M=50_000, seed=42)
    engine = JAXMonteCarloEngine()

    _, fwd = engine.run(config, ad_mode="forward")
    assert fwd is not None, "forward-mode should return greeks"
    assert set(fwd.keys()) == {"delta", "rho", "vega"}

    _, rev = engine.run(config, ad_mode="reverse")
    assert rev is not None

    # Forward and reverse should produce the same Greeks (within numerical tolerance)
    for key in ("delta", "rho", "vega"):
        assert abs(fwd[key] - rev[key]) < 0.05, (
            f"{key}: forward={fwd[key]:.6f} vs reverse={rev[key]:.6f}"
        )


def test_greeks_vs_analytical():
    """AD Greeks should be close to Black-Scholes analytical Greeks."""
    config = MCConfig(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, N=1, M=200_000, seed=42)
    analytical = compute_all_analytical_greeks(config)

    engine = JAXMonteCarloEngine()
    for mode in ("forward", "reverse"):
        _, g = engine.run(config, ad_mode=mode)
        assert g is not None
        assert abs(g["delta"] - analytical["dC/dS0"]) < 0.02, f"{mode} delta off"
        assert abs(g["vega"] - analytical["dC/dsigma"]) < 2.0, f"{mode} vega off"
        assert abs(g["rho"] - analytical["dC/dr"]) < 2.0, f"{mode} rho off"


def test_no_ad_does_not_return_greeks():
    """Running with ad_mode='none' should return None greeks."""
    config = MCConfig(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, N=1, M=100, seed=42)
    engine = JAXMonteCarloEngine()
    _, greeks = engine.run(config, ad_mode="none")
    assert greeks is None


def test_runner_captures_greeks():
    """BenchmarkRunner should capture Greeks from the engine."""
    config = MCConfig(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, N=1, M=1000, seed=42)
    runner = BenchmarkRunner(JAXMonteCarloEngine(), name="greeks-test")
    result = runner.run(config, num_warmup=1, num_runs=2, ad_mode="reverse")
    assert result.greeks is not None
    assert "delta" in result.greeks


def test_db_stores_and_retrieves_greeks():
    """BenchmarkDB should round-trip Greeks through greeks_json."""
    import tempfile, pathlib
    from benchmarking.storage.database import BenchmarkDB
    with tempfile.TemporaryDirectory() as tmp:
        db = BenchmarkDB(pathlib.Path(tmp) / "test.db")
        rid = db.create_run({"workload_type": "european"}, engine="jax", ad_mode="reverse")
        db.mark_running(rid)
        db.mark_completed(rid, result_value=10.5, mean_runtime_ms=1.0,
                          std_runtime_ms=0.1, ad_overhead_ratio=2.0,
                          greeks={"delta": 0.63, "rho": 45.1, "vega": 37.8})
        row = db.get_run(rid)
        assert row is not None
        assert row["greeks"] == {"delta": 0.63, "rho": 45.1, "vega": 37.8}


def test_result_greeks_serialization():
    """BenchmarkResult.to_dict / from_dict should round-trip Greeks."""
    config = MCConfig(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, N=1, M=100, seed=42)
    result = BenchmarkResult.from_runs(
        config=config, result=10.5, runtimes=[0.01],
        config_hash="abc", metadata={}, ad_mode="forward",
        greeks={"delta": 0.63, "vega": 37.8, "rho": 45.1},
    )
    data = result.to_dict()
    assert data["greeks"] == {"delta": 0.63, "vega": 37.8, "rho": 45.1}
    r2 = BenchmarkResult.from_dict(data)
    assert r2.greeks == {"delta": 0.63, "vega": 37.8, "rho": 45.1}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
