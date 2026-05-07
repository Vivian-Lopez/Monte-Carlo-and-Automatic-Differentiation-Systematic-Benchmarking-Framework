/**
 * pybind_module.cpp
 *
 * Minimal pybind11 bindings for the C++ Monte Carlo engine.
 * Exposes a single Python-callable function: price_european.
 *
 * Import from Python:
 *   from cpp_mc import price_european
 */

#include <pybind11/pybind11.h>
#include "../engine/cpu_engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cpp_mc, m) {
    m.doc() = "C++ Monte Carlo engine (OpenMP-parallelised) for benchmarking";

    m.def(
        "price_european",
        &price_european,
        py::arg("S0"),
        py::arg("K"),
        py::arg("r"),
        py::arg("sigma"),
        py::arg("T"),
        py::arg("M"),
        py::arg("seed"),
        py::arg("is_call") = 1,
        R"doc(
Price a European option via Monte Carlo under GBM.

Parameters
----------
S0      : float  — initial asset price
K       : float  — strike
r       : float  — risk-free rate (continuous)
sigma   : float  — volatility
T       : float  — time to maturity (years)
M       : int    — number of paths
seed    : int    — base RNG seed
is_call : int    — 1 for call (default), 0 for put

Returns
-------
float  — discounted expected payoff
        )doc"
    );

    m.def(
        "price_european_local_vol",
        &price_european_local_vol,
        py::arg("S0"),
        py::arg("K"),
        py::arg("r"),
        py::arg("T"),
        py::arg("M"),
        py::arg("N"),
        py::arg("sigma_min"),
        py::arg("a0"),
        py::arg("a1"),
        py::arg("a2"),
        py::arg("b1"),
        py::arg("seed"),
        py::arg("is_call") = 1,
        R"doc(
Price a European option via Monte Carlo under a 4-parameter local vol model.

Log-Euler discretisation with stable softplus local volatility:
    x_n     = log(S_n / S0)
    raw_n   = a0 + a1*x_n + a2*x_n^2 + b1*t_n
    sigma_n = sigma_min + softplus(raw_n)

Parameters
----------
S0, K, r, T : standard option parameters
M           : number of Monte Carlo paths
N           : number of time steps
sigma_min   : volatility floor
a0,a1,a2,b1 : local vol polynomial coefficients (theta)
seed        : base RNG seed
is_call     : 1 for call (default), 0 for put

Returns
-------
float  — discounted expected payoff
        )doc"
    );
}
