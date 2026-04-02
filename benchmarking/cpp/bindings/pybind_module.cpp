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
}
