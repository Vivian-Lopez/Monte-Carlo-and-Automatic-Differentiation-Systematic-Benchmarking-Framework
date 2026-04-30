# Benchmark Methodology

## Why European Option Is the Validation Anchor

The European plain vanilla call/put under Geometric Brownian Motion (GBM)
has an exact closed-form solution (Black-Scholes, 1973).  This makes it the
ideal validation anchor for the entire benchmark suite:

- **Price correctness** — every engine's output can be verified against the
  analytical price at any path count.
- **Greek correctness** — analytical Delta, Vega, Rho provide ground truth for
  evaluating AD accuracy (forward vs reverse vs finite difference).
- **Zero modelling ambiguity** — the single-step exact simulation has no
  discretisation error, so observed error is purely Monte Carlo variance.

All other workloads (Asian, Barrier, Basket) lack closed-form solutions and
must be validated by comparing engines against each other.  Until the European
pipeline is fully stable, extending the suite would conflate correctness bugs
with modelling differences.

---

## Warmup Protocol

JAX (and any JIT-compiled backend) traces and compiles the computation on the
first call.  This compilation cost is **excluded from timed measurements**.

### How it works

1. The `BenchmarkRunner` performs `num_warmup` calls (default: **2**) before
   the timed loop begins.
2. For JAX, the European pricing kernel is defined as a module-level
   `@jax.jit` function.  Because the array shapes are fixed by `M`, the first
   warmup call compiles once.  All subsequent calls reuse the compiled XLA
   computation.
3. `jax.block_until_ready()` is called on the result of the pricing function
   to ensure the asynchronous JAX computation is complete before the timer
   stops.

**Key invariant:** warmup calls are never included in `runtimes`.

---

## Repetition Protocol

- **`num_runs = 5`** timed repetitions by default (AD analysis: 5, scalability: 5).
- The final result value is taken from the **last** timed run (all runs with
  the same seed produce the same result).
- Statistics reported: `mean_runtime_ms`, `std_runtime_ms`, `min_runtime_ms`,
  `max_runtime_ms`.
- `std_runtime_ms` measures **timing variability**, not MC variance.

---

## Fixed Seeds

Every run uses `seed=42` by default.  The seed is stored in the `runs` table.
Changing the seed changes the random sample but not the engine or config.
Reproducibility is verified via `config_hash` (SHA-256 of the serialised config).

---

## Runtime Metric

```
mean_runtime_ms = mean(runtimes) × 1000
```

Where `runtimes` is a list of `time.perf_counter()` wall-clock intervals
(seconds) from just before `engine.run(config, ad_mode)` to just after.

For JAX, the per-call timer includes any asynchronous dispatch latency but
**excludes compilation** (handled by warmup).

---

## Throughput Metric

```
throughput_paths_per_sec = M / (mean_runtime_ms / 1000)
```

This measures how many Monte Carlo paths the engine processes per second.
It normalises across different `M` values for scalability analysis.

---

## AD Overhead Ratio

When `ad_mode` is `forward` or `reverse`, the runner also executes a
`no-AD` baseline run:

```
ad_overhead_ratio = mean_runtime_AD / mean_runtime_baseline
```

A ratio of `2.0` means the AD run takes twice as long as the price-only run.
`baseline_mean_ms` stores the no-AD mean in milliseconds.

---

## Price Error Metrics

```
abs_price_error = |MC_price - BS_price|
rel_price_error = |MC_price - BS_price| / |BS_price|
```

These are stored per run.  For European options, `BS_price` is the
Black-Scholes closed-form call (or put) price.

---

## Greek Error Metrics

```
abs_delta_error = |MC_delta - BS_delta|
rel_delta_error = |MC_delta - BS_delta| / |BS_delta|
```

Similarly for `vega` and `rho`.

**Analytical formulas used:**
- Delta:  $\Delta = N(d_1)$ (call) / $N(d_1) - 1$ (put)
- Vega:   $\nu = S_0 \cdot N'(d_1) \cdot \sqrt{T}$  (identical for call and put)
- Rho:    $\rho = K \cdot T \cdot e^{-rT} \cdot N(d_2)$ (call) / $-K \cdot T \cdot e^{-rT} \cdot N(-d_2)$ (put)

where $d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$, $d_2 = d_1 - \sigma\sqrt{T}$.

---

## memory_peak_mb

Measured using `psutil.Process().memory_info().rss`:

```
memory_peak_mb = max(RSS during timed runs) - RSS before timed runs
```

This is a **process-level RSS measurement** — it includes Python interpreter
overhead and is not a precise per-call heap measurement.  It is suitable for
order-of-magnitude comparisons across engines and path counts.

If `psutil` is unavailable, the field is stored as `0.0`.

---

## Why JAX Compilation Is Excluded

JAX traces Python functions into XLA computations.  This is a one-time fixed
cost that scales with code complexity, not with `M`.  Including compilation in
timing would:

1. Make JAX look slower for small `M` even if its steady-state throughput is
   higher than NumPy.
2. Make results non-reproducible (compilation time varies with JVM JIT state).
3. Conflate software startup cost with computational efficiency.

The warmup protocol correctly isolates the steady-state runtime.

---

## Current Limitation: Other Workloads Disabled

Asian, Barrier, and Basket workloads are implemented but **not included in
the default experiment matrices**.  They will be re-enabled once:

1. The European schema and runner are validated end-to-end.
2. Cross-engine comparison logic is tested on a workload with a known answer.
3. The SQLite schema is confirmed stable (no further migrations needed).

This prevents untested code from polluting the result database.
