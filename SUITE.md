# Benchmark Suite Definition

This file defines the full intended benchmark suite.  
Items marked **Active** are included in the default experiment scripts.  
Items marked **Dormant** are implemented but excluded from default runs.  
Items marked **Future** are not yet implemented.

---

## Workloads

| Workload | Status | Description | Engines | Analytical Ref |
|---|---|---|---|---|
| European vanilla | **Active** | GBM, single-step exact, call/put | NumPy, JAX, C++ | Black-Scholes |
| Asian (arithmetic) | Dormant | Path-dependent arithmetic average | NumPy, JAX | None (compare engines) |
| Asian (geometric) | Dormant | Path-dependent geometric average | NumPy, JAX | Geometric closed-form |
| Barrier (knock-out/in) | Dormant | Up/down barrier, multi-step | NumPy, JAX | None |
| Digital / Binary | Future | Cash-or-nothing payoff | NumPy, JAX | Closed-form |
| Basket (equal weight) | Dormant | Correlated GBM, multi-asset | NumPy, JAX | None |
| High-dimensional basket | Future | 10–100 assets, correlated GBM | JAX, CUDA | None |
| Large-path throughput | Future | M = 1M–10M paths, European | NumPy, JAX, C++, CUDA | Black-Scholes |
| Long-horizon path | Future | T = 10–30 years, N = 3650 steps | NumPy, JAX | None |

---

## Engines

| Engine | Language | Backend | Status | Description |
|---|---|---|---|---|
| `cpu` (NumPy) | Python | CPU | **Active** | Vectorised NumPy reference |
| `jax` | Python | XLA (CPU) | **Active** | JAX JIT-compiled, AD-capable |
| `cpp` (OpenMP) | C++ | CPU OpenMP | Active (if built) | pybind11 multi-threaded |
| `cuda` | Python/CUDA | GPU | Dormant | PyCUDA kernel, GPU required |
| Rust | Rust | CPU | Future | `pyo3` binding |
| Mojo | Mojo | CPU/GPU | Future | Mojo language |
| JAX (GPU/TPU) | Python | XLA (GPU/TPU) | Future | Hardware required |

---

## AD Modes

| Mode | Status | Method | Notes |
|---|---|---|---|
| `none` | **Active** | Price only | All engines |
| `forward` | **Active** | `jax.jvp` (3 calls) | JAX only |
| `reverse` | **Active** | `jax.grad(argnums=(0,1,2))` — one backward pass | JAX only |
| Finite difference | Future | Bump-and-reprice | Engine-agnostic |
| Complex-step | Future | Complex perturbation | NumPy only |

---

## Experiment Dimensions

| Dimension | Current Values | Future Values |
|---|---|---|
| Path count M | 1k, 5k, 10k, 50k, 100k | 500k, 1M, 10M |
| Time steps N | 1 (European) | 52, 252 (path-dependent) |
| Seed | 42 | Multiple seeds for variance study |
| Option type | call | put |
| Moneyness | ATM (S0=K=100) | ITM, OTM |
| Volatility σ | 0.20 | 0.10, 0.30, 0.40 |
| Maturity T | 1.0 year | 0.25, 2.0, 5.0 |
| Thread count | system default | 1, 2, 4, 8, 16 |

---

## Planned Analysis Scripts

| Script | Status | Purpose |
|---|---|---|
| `run_european_baseline.py` | **Active** | Engines × European, ad_mode=none |
| `run_european_ad_analysis.py` | **Active** | JAX × M sweep × AD modes |
| `run_european_scalability.py` | **Active** | Engines × M sweep, ad_mode=none |
| `run_asian_baseline.py` | Future | Asian workload engine comparison |
| `run_barrier_baseline.py` | Future | Barrier workload engine comparison |
| `run_basket_baseline.py` | Future | Basket workload engine comparison |
| `run_thread_scaling.py` | Future | C++ thread count sweep |
| `run_large_path_throughput.py` | Future | M = 1M–10M throughput benchmark |
| `run_cloud_cost_analysis.py` | Future | Google Cloud instance comparison |

---

## Re-enabling a Dormant Workload

1. Import the config class in the experiment script, e.g.:
   ```python
   from benchmarking.core.config import AsianOptionConfig
   ```
2. Add an entry to the workload configs dict:
   ```python
   configs["asian"] = AsianOptionConfig(S0=100, K=100, r=0.05, sigma=0.20,
                                         T=1.0, N=252, averaging="arithmetic",
                                         M=10_000, seed=42)
   ```
3. The existing engine methods (`_run_asian`, `_run_barrier`, `_run_basket`)
   in `JAXMonteCarloEngine` and `CPUMonteCarloEngine` will handle the rest.
4. Note: analytical validation is not available for Asian/Barrier/Basket.
   Store `analytical_price=None` and omit error metrics.

---

## Success Criteria for Full Suite Activation

- [ ] European pipeline validated end-to-end (price + Greeks + errors + storage)
- [ ] SQLite schema stable (no further migrations)
- [ ] C++ engine tested on European scalability
- [ ] AD overhead reproducible across repeated runs (< 5% variation in ratio)
- [ ] Asian engine comparison (CPU vs JAX) ≤ 1% price deviation at M=50k
- [ ] Barrier engine comparison (CPU vs JAX) ≤ 2% price deviation at M=50k
- [ ] Thread scaling test for C++ (1–8 threads, European, M=100k)
