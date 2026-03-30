"""
Validation utilities for automatic differentiation.

Compares computed gradients against analytical benchmarks to verify correctness.
"""

import math
from scipy.stats import norm
from benchmarking.core.config import MCConfig


def analytical_delta(config: MCConfig) -> float:
    """
    Analytical delta (dC/dS0) for European call option via Black-Scholes.
    
    Delta = N(d1), where:
    d1 = (ln(S0/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
    
    Args:
        config: MCConfig containing option parameters
        
    Returns:
        Delta value (N(d1))
    """
    sqrt_T = math.sqrt(config.T)
    d1 = (
        math.log(config.S0 / config.K) + 
        (config.r + 0.5 * config.sigma**2) * config.T
    ) / (config.sigma * sqrt_T)
    return float(norm.cdf(d1))


def analytical_vega(config: MCConfig) -> float:
    """
    Analytical vega (dC/dsigma) for European call option via Black-Scholes.
    
    Vega = S0 * N'(d1) * sqrt(T), where N'(x) is the PDF of standard normal.
    
    Args:
        config: MCConfig containing option parameters
        
    Returns:
        Vega value
    """
    sqrt_T = math.sqrt(config.T)
    d1 = (
        math.log(config.S0 / config.K) + 
        (config.r + 0.5 * config.sigma**2) * config.T
    ) / (config.sigma * sqrt_T)
    
    vega = config.S0 * norm.pdf(d1) * sqrt_T
    return float(vega)


def analytical_rho(config: MCConfig) -> float:
    """
    Analytical rho (dC/dr) for European call option via Black-Scholes.
    
    Rho = K * T * exp(-r*T) * N(d2), where:
    d2 = d1 - sigma*sqrt(T)
    
    Args:
        config: MCConfig containing option parameters
        
    Returns:
        Rho value
    """
    sqrt_T = math.sqrt(config.T)
    d1 = (
        math.log(config.S0 / config.K) + 
        (config.r + 0.5 * config.sigma**2) * config.T
    ) / (config.sigma * sqrt_T)
    d2 = d1 - config.sigma * sqrt_T
    
    rho = config.K * config.T * math.exp(-config.r * config.T) * norm.cdf(d2)
    return float(rho)


def validate_gradient(computed_gradient: float, analytical_gradient: float) -> float:
    """
    Validate computed gradient against analytical benchmark.
    
    Args:
        computed_gradient: Gradient from AD
        analytical_gradient: Known analytical value
        
    Returns:
        Relative error (|computed - analytical| / |analytical|)
    """
    if analytical_gradient == 0:
        return abs(computed_gradient)
    
    rel_error = abs(computed_gradient - analytical_gradient) / abs(analytical_gradient)
    return rel_error


def compute_all_analytical_greeks(config: MCConfig) -> dict:
    """
    Compute all analytical Greeks for reference.
    
    Returns:
        Dictionary with keys: "dC/dS0", "dC/dsigma", "dC/dr"
    """
    return {
        "dC/dS0": analytical_delta(config),
        "dC/dsigma": analytical_vega(config),
        "dC/dr": analytical_rho(config),
    }
