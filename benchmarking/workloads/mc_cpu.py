import math
import random
from benchmarking.core.config import MCConfig
from benchmarking.core.engine import MonteCarloEngine


class CPUMonteCarloEngine(MonteCarloEngine):
    """
    Pure Python CPU implementation of Monte Carlo simulation for European options.
    
    Uses geometric Brownian motion with Euler discretization.
    Implements deterministic seeding for reproducibility.
    """
    
    def run(self, config: MCConfig, ad_mode: str = "none") -> float:
        """
        Execute Monte Carlo simulation on CPU.
        
        Implements geometric Brownian motion: dS = r*S*dt + sigma*S*sqrt(dt)*dZ
        """
        random.seed(config.seed)
        payoff_sum = 0.0
        
        for _ in range(config.M):
            # Generate random normal variable for single step
            Z = random.gauss(0, 1)
            
            # Simulate stock price at maturity using geometric Brownian motion
            # S_T = S_0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
            S_T = config.S0 * math.exp(
                (config.r - 0.5 * config.sigma**2) * config.T + 
                config.sigma * math.sqrt(config.T) * Z
            )
            
            # Calculate payoff: max(S_T - K, 0)
            payoff = max(S_T - config.K, 0)
            payoff_sum += payoff
        
        # Discount the average payoff back to present value
        price = math.exp(-config.r * config.T) * payoff_sum / config.M
        return price


def monte_carlo_european_call(config: MCConfig, ad_mode: str = "none") -> float:
    """
    Monte Carlo simulation for European call option pricing using pure Python loops.
    
    Implements geometric Brownian motion: dS = r*S*dt + sigma*S*sqrt(dt)*dZ
    Uses explicit Euler discretization for single-step maturity (N=1 in practice).
    
    Args:
        config: MCConfig containing simulation parameters
        ad_mode: Differentiation mode ("none", "forward", "reverse") - currently unused
                 but parameter included for framework compatibility with future AD implementations
        
    Returns:
        Estimated option price (discounted average payoff)
    """
    random.seed(config.seed)
    payoff_sum = 0.0
    
    for _ in range(config.M):
        # Generate random normal variable for single step
        Z = random.gauss(0, 1)
        
        # Simulate stock price at maturity using geometric Brownian motion
        # S_T = S_0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
        S_T = config.S0 * math.exp(
            (config.r - 0.5 * config.sigma**2) * config.T + 
            config.sigma * math.sqrt(config.T) * Z
        )
        
        # Calculate payoff: max(S_T - K, 0)
        payoff = max(S_T - config.K, 0)
        payoff_sum += payoff
    
    # Discount the average payoff back to present value
    price = math.exp(-config.r * config.T) * payoff_sum / config.M
    return price


def black_scholes_call(config: MCConfig) -> float:
    """
    Analytical Black-Scholes price for European call option.
    
    Used to validate Monte Carlo results:
    C = S_0 * N(d1) - K * exp(-r*T) * N(d2)
    
    where:
    d1 = (ln(S_0/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    N(x) = cumulative standard normal distribution
    
    Args:
        config: MCConfig containing option parameters
        
    Returns:
        Exact option price under Black-Scholes assumptions
    """
    from scipy.stats import norm
    
    sqrt_T = math.sqrt(config.T)
    d1 = (
        math.log(config.S0 / config.K) + 
        (config.r + 0.5 * config.sigma**2) * config.T
    ) / (config.sigma * sqrt_T)
    d2 = d1 - config.sigma * sqrt_T
    
    call_price = (
        config.S0 * norm.cdf(d1) - 
        config.K * math.exp(-config.r * config.T) * norm.cdf(d2)
    )
    return call_price


def european_call_delta(config: MCConfig) -> float:
    """
    Analytical delta (dC/dS0) for European call option.
    
    Delta = N(d1), where d1 is as in Black-Scholes formula.
    This is useful for AD validation.
    
    Args:
        config: MCConfig containing option parameters
        
    Returns:
        Delta (first derivative w.r.t. S0)
    """
    from scipy.stats import norm
    
    sqrt_T = math.sqrt(config.T)
    d1 = (
        math.log(config.S0 / config.K) + 
        (config.r + 0.5 * config.sigma**2) * config.T
    ) / (config.sigma * sqrt_T)
    
    return norm.cdf(d1)