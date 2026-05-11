//! rust_mc — Rayon-parallelised Monte Carlo pricer for local volatility models.
//!
//! Exposed to Python via PyO3.  Each Rayon worker thread owns an independent
//! Xoshiro256++ RNG seeded as `seed ^ thread_index` so results are
//! deterministic for a fixed (M, seed, thread count) triple.
//!
//! Local volatility model
//! ----------------------
//! dS = r S dt + sigma(S, t; theta) S dW
//!
//! Log-Euler discretisation:
//!   S_{n+1} = S_n * exp((r - 0.5*sigma_n^2)*dt + sigma_n*sqrt(dt)*Z_n)
//!
//! Local vol (4-parameter polynomial + softplus floor):
//!   x_n     = ln(S_n / S0)
//!   raw_n   = a0 + a1*x_n + a2*x_n^2 + b1*t_n
//!   sigma_n = sigma_min + softplus(raw_n)
//!
//! Numerically stable softplus:
//!   softplus(z) = max(z, 0) + ln(1 + exp(-|z|))

use pyo3::prelude::*;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Type alias for the per-thread RNG.
// rand::rngs::SmallRng is a fast, non-cryptographic PRNG; it is Send + Sync
// so Rayon can move it across threads.
// ---------------------------------------------------------------------------
type Rng = rand::rngs::SmallRng;

// ---------------------------------------------------------------------------
// Numerically stable softplus: max(z,0) + log1p(exp(-|z|))
// ---------------------------------------------------------------------------
#[inline(always)]
fn softplus_stable(z: f64) -> f64 {
    // max(z, 0) + log1p(exp(-|z|))
    // Safe: for z >> 0, exp(-z) -> 0, so log1p(0) = 0, result = z
    //       for z << 0, exp(|z|) is large but log1p handles it
    //       |z| <= 0: both terms are O(1)
    z.max(0.0) + (-z.abs()).exp().ln_1p()
}

// ---------------------------------------------------------------------------
// Local volatility: sigma(S, t; S0, sigma_min, theta)
// ---------------------------------------------------------------------------
#[inline(always)]
fn local_vol(s: f64, t: f64, s0: f64, sigma_min: f64, a0: f64, a1: f64, a2: f64, b1: f64) -> f64 {
    let x = (s / s0).ln();
    let raw = a0 + a1 * x + a2 * x * x + b1 * t;
    sigma_min + softplus_stable(raw)
}

// ---------------------------------------------------------------------------
// Single-path log-Euler simulation — no allocation, pure scalar arithmetic
// ---------------------------------------------------------------------------
#[inline]
fn simulate_path(
    s0: f64, r: f64, _t: f64,
    n: usize, dt: f64, sqrt_dt: f64,
    sigma_min: f64, a0: f64, a1: f64, a2: f64, b1: f64,
    rng: &mut Rng,
) -> f64 {
    let mut s = s0;
    for step in 0..n {
        let t_n = step as f64 * dt;
        let sigma = local_vol(s, t_n, s0, sigma_min, a0, a1, a2, b1);
        let z: f64 = StandardNormal.sample(rng);
        s *= ((r - 0.5 * sigma * sigma) * dt + sigma * sqrt_dt * z).exp();
    }
    s
}

// ---------------------------------------------------------------------------
// Core pricer: European option under local vol
// ---------------------------------------------------------------------------
fn _price_european_local_vol(
    s0: f64, k: f64, r: f64, t: f64,
    m: usize, n: usize,
    sigma_min: f64,
    a0: f64, a1: f64, a2: f64, b1: f64,
    seed: u64,
    is_call: bool,
) -> f64 {
    let dt      = t / n as f64;
    let sqrt_dt = dt.sqrt();
    let disc    = (-r * t).exp();

    // Each Rayon worker thread gets a unique RNG seeded as seed ^ thread_index,
    // matching the C++ per-thread offset pattern.  Results are reproducible for
    // a fixed (M, seed, thread count) triple.
    let sum: f64 = (0..m)
        .into_par_iter()
        .map_init(
            || {
                let tid = rayon::current_thread_index().unwrap_or(0);
                Rng::seed_from_u64(seed ^ tid as u64)
            },
            |rng, _i| {
                let s_t = simulate_path(
                    s0, r, t, n, dt, sqrt_dt,
                    sigma_min, a0, a1, a2, b1, rng,
                );
                let raw = if is_call { s_t - k } else { k - s_t };
                raw.max(0.0)
            },
        )
        .sum();

    disc * sum / m as f64
}

// ---------------------------------------------------------------------------
// PyO3 bindings
// ---------------------------------------------------------------------------

/// Price a European option under a 4-parameter parametric local volatility
/// model using log-Euler Monte Carlo discretisation.
///
/// Parameters
/// ----------
/// s0, k, r, t   : standard option parameters (float)
/// m              : number of Monte Carlo paths (int)
/// n              : number of time steps (int)
/// sigma_min      : volatility floor (float)
/// a0, a1, a2, b1 : local vol polynomial coefficients — theta (float)
/// seed           : RNG seed for reproducibility (int)
/// is_call        : True for call, False for put (bool, default True)
///
/// Returns
/// -------
/// float — discounted expected payoff
#[pyfunction]
#[pyo3(signature = (s0, k, r, t, m, n, sigma_min, a0, a1, a2, b1, seed, is_call=true))]
fn price_european_local_vol(
    s0: f64, k: f64, r: f64, t: f64,
    m: usize, n: usize,
    sigma_min: f64,
    a0: f64, a1: f64, a2: f64, b1: f64,
    seed: u64,
    is_call: bool,
) -> f64 {
    _price_european_local_vol(s0, k, r, t, m, n, sigma_min, a0, a1, a2, b1, seed, is_call)
}

/// Price a European option under constant-volatility GBM (single-step exact).
///
/// Included as a baseline / sanity-check companion to price_european_local_vol.
///
/// Parameters
/// ----------
/// s0, k, r, sigma, t : standard option parameters
/// m                   : number of paths
/// seed                : RNG seed
/// is_call             : True for call (default), False for put
#[pyfunction]
#[pyo3(signature = (s0, k, r, sigma, t, m, seed, is_call=true))]
fn price_european(
    s0: f64, k: f64, r: f64, sigma: f64, t: f64,
    m: usize,
    seed: u64,
    is_call: bool,
) -> f64 {
    // Delegate to local vol with a1=a2=b1=0 and a0 = softplus_inv(sigma - sigma_min)
    // at sigma_min=0 — simpler to just inline the single-step formula directly.
    let drift   = (r - 0.5 * sigma * sigma) * t;
    let vol_sqt = sigma * t.sqrt();
    let disc    = (-r * t).exp();

    let sum: f64 = (0..m)
        .into_par_iter()
        .map_init(
            || {
                let tid = rayon::current_thread_index().unwrap_or(0);
                Rng::seed_from_u64(seed ^ tid as u64)
            },
            |rng, _i| {
                let z: f64 = StandardNormal.sample(rng);
                let s_t = s0 * (drift + vol_sqt * z).exp();
                let raw = if is_call { s_t - k } else { k - s_t };
                raw.max(0.0)
            },
        )
        .sum();

    disc * sum / m as f64
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
fn rust_mc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(price_european_local_vol, m)?)?;
    m.add_function(wrap_pyfunction!(price_european, m)?)?;
    Ok(())
}
