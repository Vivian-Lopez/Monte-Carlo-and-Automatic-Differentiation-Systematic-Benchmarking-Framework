"""
setup.py — build the cpp_mc pybind11 extension.

Usage (from repo root or benchmarking/cpp/):
    pip install -e benchmarking/cpp/
or:
    python benchmarking/cpp/setup.py build_ext --inplace

The resulting .so is placed in benchmarking/cpp/ and importable as `cpp_mc`
once that directory (or the repo root after `pip install -e .`) is on sys.path.
"""

import os
import sys
from setuptools import setup, Extension

import pybind11

BASE = os.path.dirname(__file__)

ext = Extension(
    name="cpp_mc",
    sources=[
        os.path.join(BASE, "engine", "cpu_engine.cpp"),
        os.path.join(BASE, "bindings", "pybind_module.cpp"),
    ],
    include_dirs=[
        pybind11.get_include(),
        os.path.join(BASE, "engine"),
    ],
    extra_compile_args=[
        "-O3",
        "-march=native",
        "-fopenmp",
        "-std=c++17",
        "-fvisibility=hidden",   # reduces .so size, required by pybind11
    ],
    extra_link_args=["-fopenmp"],
    language="c++",
)

setup(
    name="cpp_mc",
    version="0.1.0",
    description="OpenMP-parallelised C++ Monte Carlo pricer (pybind11)",
    ext_modules=[ext],
    python_requires=">=3.9",
    install_requires=["pybind11>=2.11"],
)
