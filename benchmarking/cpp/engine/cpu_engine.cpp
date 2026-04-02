/**
 * cpu_engine.cpp
 *
 * C++ Monte Carlo European option pricer with OpenMP parallelisation.
 *
 * Algorithm
 * ---------
 *  Single-step exact GBM:
 *    S_T = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z),  Z ~ N(0,1)
 *
 *  Parallelisation strategy
 *  ------------------------
 *  Each OpenMP thread owns its own mt19937 RNG seeded as (seed + thread_id).
 *  A reduction accumulates partial sums safely without locks.
 *
 *  Reproducibility
 *  ---------------
 *  For a given (M, seed, n_threads), results are fully deterministic.
 *  Note: changing thread count changes the per-path RNG assignment, so
 *  numerical agreement with the Python engine requires running single-threaded
 *  (OMP_NUM_THREADS=1) or accepting a statistically equivalent but
 *  numerically different result when threaded.
 */

#include "cpu_engine.hpp"

#include <cmath>
#include <random>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#else
// Stub so code compiles without OpenMP
inline int omp_get_thread_num()  { return 0; }
inline int omp_get_max_threads() { return 1; }
#endif

double price_european(
    double S0,
    double K,
    double r,
    double sigma,
    double T,
    int    M,
    int    seed,
    int    is_call
) {
    const double drift   = (r - 0.5 * sigma * sigma) * T;
    const double vol_sqT = sigma * std::sqrt(T);
    const double disc    = std::exp(-r * T);

    double sum = 0.0;

#pragma omp parallel reduction(+:sum)
    {
        // Per-thread RNG — offset seed by thread id for independence
        const int tid = omp_get_thread_num();
        std::mt19937_64 rng(static_cast<uint64_t>(seed) + static_cast<uint64_t>(tid));
        std::normal_distribution<double> norm(0.0, 1.0);

#pragma omp for schedule(static)
        for (int i = 0; i < M; ++i) {
            const double Z   = norm(rng);
            const double S_T = S0 * std::exp(drift + vol_sqT * Z);
            const double raw = is_call ? S_T - K : K - S_T;
            sum += (raw > 0.0) ? raw : 0.0;
        }
    }

    return disc * sum / static_cast<double>(M);
}
