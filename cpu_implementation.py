import random
import math

def estimate_option_call_payoff_with_monte_carlo(S0: float, K: float, r: float, sigma: float, T: float, N: int, M: int) -> float:
    dt = T/N
    sqrt_dt = math.sqrt(dt)
    discount_factor = math.exp(-r * T)
    payoff_sum = 0.0

    for _ in range(M):
        St = S0
        for _ in range(N):
            Z = random.gauss(0.0, 1.0)
            St = St * math.exp((r - 0.5 * sigma**2) * dt + sigma * sqrt_dt * Z)
        payoff_sum += max(St - K, 0)

    average_payoff = payoff_sum / M
    discounted_price = discount_factor * average_payoff
    return discounted_price

random.seed(42)

print(estimate_option_call_payoff_with_monte_carlo(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, N=252, M=10_000))