#pragma once

/**
 * cpu_engine.hpp
 * Header for the C++ Monte Carlo European option pricer.
 */

/**
 * Price a European call option using Monte Carlo simulation under GBM.
 *
 * Uses single-step exact GBM formula:
 *   S_T = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
 *
 * where Z ~ N(0,1).  Payoff is discounted at rate r.
 *
 * @param S0     Initial asset price
 * @param K      Strike price
 * @param r      Risk-free rate (continuously compounded)
 * @param sigma  Volatility
 * @param T      Time to maturity (years)
 * @param M      Number of Monte Carlo paths
 * @param seed   Base RNG seed (each OpenMP thread offsets by thread id)
 * @param is_call  1 for call, 0 for put
 * @return  Estimated option price
 */
double price_european(
    double S0,
    double K,
    double r,
    double sigma,
    double T,
    int    M,
    int    seed,
    int    is_call
);

/**
 * Price a European option under a 4-parameter parametric local volatility
 * model using log-Euler Monte Carlo discretisation.
 *
 * Dynamics: dS = r S dt + sigma(S,t;theta) S dW
 * Log-Euler: S_{n+1} = S_n * exp((r - 0.5*sigma_n^2)*dt + sigma_n*sqrt(dt)*Z_n)
 *
 * Local vol (stable softplus form):
 *   x_n     = log(S_n / S0)
 *   raw_n   = a0 + a1*x_n + a2*x_n^2 + b1*t_n
 *   sigma_n = sigma_min + max(raw,0) + log1p(exp(-|raw|))
 *
 * @param S0        Initial asset price
 * @param K         Strike price
 * @param r         Risk-free rate
 * @param T         Time to maturity (years)
 * @param M         Number of Monte Carlo paths
 * @param N         Number of time steps
 * @param sigma_min Volatility floor
 * @param a0,a1,a2,b1  Local vol polynomial coefficients
 * @param seed      Base RNG seed
 * @param is_call   1 for call, 0 for put
 * @return  Estimated option price
 */
double price_european_local_vol(
    double S0,
    double K,
    double r,
    double T,
    int    M,
    int    N,
    double sigma_min,
    double a0,
    double a1,
    double a2,
    double b1,
    int    seed,
    int    is_call
);
