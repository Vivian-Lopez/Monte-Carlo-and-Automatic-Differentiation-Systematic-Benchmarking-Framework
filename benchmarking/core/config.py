from dataclasses import dataclass
import hashlib
import json

@dataclass
class MCConfig:
    """
    Monte Carlo simulation configuration.
    
    Parameters represent a European call option pricing scenario:
    - S0: Initial stock price
    - K: Strike price
    - r: Risk-free rate (annual)
    - sigma: Volatility (annual)
    - T: Time to maturity (years)
    - N: Number of time steps (reserved for future use in path-dependent options)
    - M: Number of Monte Carlo simulation paths
    - seed: Random seed for reproducibility
    """
    S0: float
    K: float
    r: float
    sigma: float
    T: float
    N: int
    M: int
    seed: int
    
    def validate(self) -> None:
        """Validate parameter ranges."""
        if self.S0 <= 0:
            raise ValueError(f"S0 must be positive, got {self.S0}")
        if self.K <= 0:
            raise ValueError(f"K must be positive, got {self.K}")
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.M <= 0:
            raise ValueError(f"M must be positive, got {self.M}")
        if self.N <= 0:
            raise ValueError(f"N must be positive, got {self.N}")
    
    def config_hash(self) -> str:
        """
        Compute a hash of the configuration for reproducibility verification.
        Allows detection of configuration mismatches across runs.
        """
        config_str = json.dumps(
            {
                "S0": self.S0,
                "K": self.K,
                "r": self.r,
                "sigma": self.sigma,
                "T": self.T,
                "N": self.N,
                "M": self.M,
                "seed": self.seed,
            },
            sort_keys=True
        )
        return hashlib.sha256(config_str.encode()).hexdigest()[:8]