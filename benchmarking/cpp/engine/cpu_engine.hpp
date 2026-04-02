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
