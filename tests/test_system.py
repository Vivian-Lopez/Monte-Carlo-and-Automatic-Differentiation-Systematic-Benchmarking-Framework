"""
End-to-end system tests for the Monte Carlo Benchmarking Engine.

Covers every layer a quant interacts with:
  1. Config  — validation, serialization, hashing, registry
  2. Engine  — numerical correctness for all 4 workloads × CPU & JAX
  3. Runner  — timing, reproducibility, result structure
  4. Storage — SQLite round-trip (create → run → complete → query)
  5. API     — Flask endpoint contracts (submit → poll → summary)
  6. E2E     — full flow: config → engine → runner → storage → API read-back
"""

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ── Layer 1: Config ───────────────────────────────────────────────────────

from benchmarking.core.config import (
    WorkloadConfig,
    EuropeanOptionConfig,
    AsianOptionConfig,
    BarrierOptionConfig,
    BasketOptionConfig,
    WORKLOAD_REGISTRY,
    config_from_dict,
    MCConfig,
)


class TestConfigValidation:
    """Validation rejects bad inputs and accepts good ones."""

    def test_european_valid(self):
        c = EuropeanOptionConfig(S0=100, K=100, r=0.05, sigma=0.2, T=1.0, M=1000, seed=1)
        c.validate()  # should not raise

    @pytest.mark.parametrize("field,value", [
        ("S0", -1), ("K", 0), ("sigma", 0), ("T", -0.5), ("M", 0),
    ])
    def test_european_rejects_bad_numerics(self, field, value):
        kwargs = dict(S0=100, K=100, r=0.05, sigma=0.2, T=1.0, M=1000, seed=1)
        kwargs[field] = value
        c = EuropeanOptionConfig(**kwargs)
        with pytest.raises(ValueError):
            c.validate()

    def test_european_rejects_bad_option_type(self):
        c = EuropeanOptionConfig(option_type="straddle")
        with pytest.raises(ValueError):
            c.validate()

    def test_asian_needs_N_gte_2(self):
        c = AsianOptionConfig(N=1)
        with pytest.raises(ValueError):
            c.validate()

    def test_asian_rejects_bad_averaging(self):
        c = AsianOptionConfig(averaging="harmonic")
        with pytest.raises(ValueError):
            c.validate()

    def test_barrier_rejects_bad_barrier_type(self):
        c = BarrierOptionConfig(barrier_type="knock_sideways")
        with pytest.raises(ValueError):
            c.validate()

    def test_basket_rejects_one_asset(self):
        c = BasketOptionConfig(n_assets=1)
        with pytest.raises(ValueError):
            c.validate()

    def test_basket_rejects_rho_out_of_range(self):
        c = BasketOptionConfig(rho=1.5)
        with pytest.raises(ValueError):
            c.validate()


class TestConfigSerialization:
    """to_dict / from_dict round-trip for every workload type."""

    @pytest.mark.parametrize("cls,extras", [
        (EuropeanOptionConfig, {}),
        (AsianOptionConfig, {"averaging": "geometric"}),
        (BarrierOptionConfig, {"B": 130.0, "barrier_type": "knock_in", "barrier_side": "down"}),
        (BasketOptionConfig, {"n_assets": 5, "rho": 0.3}),
    ])
    def test_round_trip(self, cls, extras):
        original = cls(**extras)
        d = original.to_dict()
        assert "workload_type" in d
        assert "SCHEMA" not in d      # SCHEMA must be stripped
        restored = cls.from_dict(d)
        assert restored.to_dict() == d

    def test_config_from_dict_dispatches(self):
        d = {"workload_type": "asian", "S0": 50.0, "averaging": "geometric"}
        c = config_from_dict(d)
        assert isinstance(c, AsianOptionConfig)
        assert c.S0 == 50.0
        assert c.averaging == "geometric"

    def test_config_from_dict_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown workload_type"):
            config_from_dict({"workload_type": "lookback"})


class TestConfigHash:
    """Deterministic hashing for reproducibility."""

    def test_same_config_same_hash(self):
        a = EuropeanOptionConfig(S0=100, K=100, r=0.05, sigma=0.2, T=1.0, M=1000, seed=42)
        b = EuropeanOptionConfig(S0=100, K=100, r=0.05, sigma=0.2, T=1.0, M=1000, seed=42)
        assert a.config_hash() == b.config_hash()

    def test_different_seed_different_hash(self):
        a = EuropeanOptionConfig(seed=42)
        b = EuropeanOptionConfig(seed=99)
        assert a.config_hash() != b.config_hash()

    def test_hash_is_8_hex_chars(self):
        h = EuropeanOptionConfig().config_hash()
        assert len(h) == 8
        int(h, 16)  # must be valid hex


class TestWorkloadRegistry:
    """Registry contains all four workload types."""

    def test_registry_keys(self):
        assert set(WORKLOAD_REGISTRY.keys()) == {"european", "asian", "barrier", "basket"}

    def test_all_have_schema(self):
        for wtype, cls in WORKLOAD_REGISTRY.items():
            instance = cls()
            assert isinstance(instance.SCHEMA, list), f"{wtype} SCHEMA is not a list"
            assert len(instance.SCHEMA) > 0, f"{wtype} SCHEMA is empty"
            for field in instance.SCHEMA:
                assert "key" in field and "label" in field and "type" in field

    def test_mcconfig_is_european(self):
        assert MCConfig is EuropeanOptionConfig


# ── Layer 2: Engine correctness ───────────────────────────────────────────

from benchmarking.workloads.mc_cpu import CPUMonteCarloEngine, black_scholes_call
from benchmarking.workloads.mc_jax import JAXMonteCarloEngine


class TestCPUEngineEuropean:
    """CPU engine European option prices converge to Black-Scholes."""

    @pytest.fixture()
    def engine(self):
        return CPUMonteCarloEngine()

    @pytest.fixture()
    def atm_config(self):
        return EuropeanOptionConfig(
            S0=100, K=100, r=0.05, sigma=0.2, T=1.0, M=50_000, seed=42,
        )

    def test_call_converges_to_bs(self, engine, atm_config):
        mc_price, _ = engine.run(atm_config)
        bs_price = black_scholes_call(atm_config)
        assert abs(mc_price - bs_price) / bs_price < 0.02  # <2% relative error

    def test_put_is_positive(self, engine):
        c = EuropeanOptionConfig(
            S0=100, K=120, r=0.05, sigma=0.2, T=1.0, option_type="put", M=10_000, seed=7,
        )
        price, _ = engine.run(c)
        assert price > 0

    def test_deterministic_with_seed(self, engine, atm_config):
        p1, _ = engine.run(atm_config)
        p2, _ = engine.run(atm_config)
        assert p1 == p2


class TestCPUEngineExotics:
    """CPU engine prices exotic workloads without error and with sensible values."""

    @pytest.fixture()
    def engine(self):
        return CPUMonteCarloEngine()

    def test_asian_arithmetic(self, engine):
        c = AsianOptionConfig(S0=100, K=100, r=0.05, sigma=0.2, T=1.0,
                              N=50, M=10_000, seed=42, averaging="arithmetic")
        price, _ = engine.run(c)
        assert 0 < price < 20  # reasonable range for ATM asian

    def test_asian_geometric(self, engine):
        c = AsianOptionConfig(S0=100, K=100, r=0.05, sigma=0.2, T=1.0,
                              N=50, M=10_000, seed=42, averaging="geometric")
        price, _ = engine.run(c)
        assert 0 < price < 20

    def test_asian_cheaper_than_european(self, engine):
        """Averaging reduces volatility → Asian call ≤ European call."""
        eu = EuropeanOptionConfig(S0=100, K=100, r=0.05, sigma=0.2, T=1.0, M=50_000, seed=42)
        asian = AsianOptionConfig(S0=100, K=100, r=0.05, sigma=0.2, T=1.0,
                                  N=252, M=50_000, seed=42, averaging="arithmetic")
        eu_price, _ = engine.run(eu)
        asian_price, _ = engine.run(asian)
        assert asian_price < eu_price

    def test_barrier_knock_out_leq_vanilla(self, engine):
        """Knock-out ≤ vanilla because some paths are eliminated."""
        eu = EuropeanOptionConfig(S0=100, K=100, r=0.05, sigma=0.2, T=1.0, M=50_000, seed=42)
        barrier = BarrierOptionConfig(
            S0=100, K=100, r=0.05, sigma=0.2, T=1.0, B=130, N=252,
            barrier_type="knock_out", barrier_side="up", M=50_000, seed=42,
        )
        eu_price, _ = engine.run(eu)
        barrier_price, _ = engine.run(barrier)
        assert barrier_price <= eu_price * 1.01  # 1% tolerance for MC noise

    def test_barrier_knock_in_plus_knock_out_eq_vanilla(self, engine):
        """In-out parity: knock_in + knock_out ≈ vanilla."""
        seed, M, N = 42, 50_000, 252
        base = dict(S0=100, K=100, r=0.05, sigma=0.2, T=1.0, B=120,
                    barrier_side="up", M=M, seed=seed, N=N)
        ki = BarrierOptionConfig(barrier_type="knock_in", **base)
        ko = BarrierOptionConfig(barrier_type="knock_out", **base)
        eu = EuropeanOptionConfig(S0=100, K=100, r=0.05, sigma=0.2, T=1.0, M=M, seed=seed)
        ki_price, _ = engine.run(ki)
        ko_price, _ = engine.run(ko)
        eu_price, _ = engine.run(eu)
        assert abs((ki_price + ko_price) - eu_price) / eu_price < 0.03

    def test_basket_runs(self, engine):
        c = BasketOptionConfig(n_assets=3, S0=100, K=100, r=0.05, sigma=0.2,
                               rho=0.5, T=1.0, N=12, M=10_000, seed=42)
        price, _ = engine.run(c)
        assert 0 < price < 30

    def test_basket_correlation_effect(self, engine):
        """Higher ρ → more correlated assets → basket behaves more like single asset."""
        base = dict(n_assets=3, S0=100, K=100, r=0.05, sigma=0.2, T=1.0,
                   N=12, M=50_000, seed=42)
        low_rho, _ = engine.run(BasketOptionConfig(rho=0.1, **base))
        high_rho, _ = engine.run(BasketOptionConfig(rho=0.9, **base))
        # Higher correlation → higher basket option price (less diversification benefit)
        assert high_rho > low_rho

    def test_supports_all_workloads(self, engine):
        for wtype in ("european", "asian", "barrier", "basket"):
            assert engine.supports(wtype)
        assert not engine.supports("lookback")

    def test_cpu_rejects_ad_mode(self, engine):
        """CPU engine raises NotImplementedError for non-none AD modes."""
        c = EuropeanOptionConfig(S0=100, K=100, r=0.05, sigma=0.2, T=1.0, M=1000, seed=1)
        for mode in ("forward", "reverse"):
            with pytest.raises(NotImplementedError):
                engine.run(c, ad_mode=mode)

    def test_cpu_supported_ad_modes(self, engine):
        assert engine.supported_ad_modes() == ("none",)


class TestJAXEngineEuropean:
    """JAX engine produces same-ballpark prices as CPU for European."""

    def test_jax_matches_cpu(self):
        c = EuropeanOptionConfig(S0=100, K=100, r=0.05, sigma=0.2, T=1.0, M=50_000, seed=42)
        cpu_price, _ = CPUMonteCarloEngine().run(c)
        jax_price, _ = JAXMonteCarloEngine().run(c, ad_mode="none")
        # Different RNG streams → prices won't be identical, but both close to BS
        bs = black_scholes_call(c)
        assert abs(cpu_price - bs) / bs < 0.02
        assert abs(jax_price - bs) / bs < 0.02

    def test_jax_ad_does_not_crash(self):
        c = EuropeanOptionConfig(S0=100, K=100, r=0.05, sigma=0.2, T=1.0, M=1000, seed=42)
        engine = JAXMonteCarloEngine()
        for mode in ("none", "forward", "reverse"):
            price, _ = engine.run(c, ad_mode=mode)
            assert isinstance(price, float)
            assert price > 0

    def test_jax_supported_ad_modes(self):
        engine = JAXMonteCarloEngine()
        modes = engine.supported_ad_modes()
        assert "none" in modes
        assert "forward" in modes
        assert "reverse" in modes


# ── Layer 3: Runner ───────────────────────────────────────────────────────

from benchmarking.runner.runner import BenchmarkRunner
from benchmarking.core.result import BenchmarkResult


class TestBenchmarkRunner:
    """Runner produces structured results with valid statistics."""

    @pytest.fixture()
    def result(self):
        engine = CPUMonteCarloEngine()
        runner = BenchmarkRunner(engine, name="test-cpu")
        config = EuropeanOptionConfig(S0=100, K=100, r=0.05, sigma=0.2,
                                      T=1.0, M=1000, seed=42)
        return runner.run(config, num_warmup=1, num_runs=3, ad_mode="none")

    def test_result_type(self, result):
        assert isinstance(result, BenchmarkResult)

    def test_result_has_correct_num_runtimes(self, result):
        assert len(result.runtimes) == 3

    def test_statistics_are_consistent(self, result):
        assert result.mean_runtime == pytest.approx(
            sum(result.runtimes) / len(result.runtimes), rel=1e-9)
        assert result.min_runtime == min(result.runtimes)
        assert result.max_runtime == max(result.runtimes)
        assert result.std_runtime >= 0

    def test_metadata_captured(self, result):
        assert "python_version" in result.metadata
        assert "platform" in result.metadata
        assert "timestamp" in result.metadata

    def test_config_hash_present(self, result):
        assert len(result.config_hash) == 8

    def test_result_serialization_round_trip(self, result):
        d = result.to_dict()
        restored = BenchmarkResult.from_dict(d)
        assert restored.result == result.result
        assert restored.runtimes == result.runtimes
        assert restored.config_hash == result.config_hash
        assert restored.ad_mode == result.ad_mode

    def test_save_and_load_json(self, result):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            runner = BenchmarkRunner(CPUMonteCarloEngine(), name="test")
            runner.save_results(result, path)
            with open(path) as f:
                data = json.load(f)
            assert data["result"] == result.result
            assert "statistics" in data
            assert "config" in data
            restored = BenchmarkResult.from_dict(data)
            assert restored.result == result.result
        finally:
            os.unlink(path)


# ── Layer 4: Storage ──────────────────────────────────────────────────────

from benchmarking.storage.database import BenchmarkDB


class TestBenchmarkDB:
    """SQLite storage layer: create → status transitions → query."""

    @pytest.fixture()
    def db(self, tmp_path):
        return BenchmarkDB(db_path=tmp_path / "test.db")

    @pytest.fixture()
    def sample_config(self):
        return EuropeanOptionConfig(S0=100, K=100, r=0.05, sigma=0.2,
                                    T=1.0, M=1000, seed=42).to_dict()

    def test_create_and_get(self, db, sample_config):
        run_id = db.create_run(sample_config, "cpu", "none")
        row = db.get_run(run_id)
        assert row is not None
        assert row["status"] == "pending"
        assert row["engine"] == "cpu"
        assert row["workload_type"] == "european"

    def test_status_transitions(self, db, sample_config):
        run_id = db.create_run(sample_config, "jax", "forward")
        db.mark_running(run_id)
        assert db.get_run(run_id)["status"] == "running"

        db.mark_completed(run_id, result_value=10.5, mean_runtime_ms=1.2,
                          std_runtime_ms=0.1, ad_overhead_ratio=2.0)
        row = db.get_run(run_id)
        assert row["status"] == "completed"
        assert row["result_value"] == pytest.approx(10.5)
        assert row["mean_runtime_ms"] == pytest.approx(1.2)

    def test_mark_failed(self, db, sample_config):
        run_id = db.create_run(sample_config, "cpu", "none")
        db.mark_running(run_id)
        db.mark_failed(run_id, "Division by zero")
        row = db.get_run(run_id)
        assert row["status"] == "failed"
        assert "Division by zero" in row["error_message"]

    def test_get_pending(self, db, sample_config):
        id1 = db.create_run(sample_config, "cpu", "none")
        id2 = db.create_run(sample_config, "jax", "none")
        db.mark_running(id1)  # no longer pending
        pending = db.get_pending_runs()
        assert len(pending) == 1
        assert pending[0]["id"] == id2

    def test_get_all_with_filters(self, db, sample_config):
        db.create_run(sample_config, "cpu", "none")
        asian_config = AsianOptionConfig(S0=100, K=100, r=0.05, sigma=0.2,
                                         T=1.0, N=50, M=1000, seed=42).to_dict()
        db.create_run(asian_config, "jax", "none")

        all_runs = db.get_all_runs()
        assert len(all_runs) == 2

        cpu_only = db.get_all_runs(engine="cpu")
        assert len(cpu_only) == 1
        assert cpu_only[0]["engine"] == "cpu"

        asian_only = db.get_all_runs(workload_type="asian")
        assert len(asian_only) == 1

    def test_summary(self, db, sample_config):
        id1 = db.create_run(sample_config, "cpu", "none")
        db.mark_running(id1)
        db.mark_completed(id1, 10.0, 1.0, 0.1, 1.0)

        id2 = db.create_run(sample_config, "jax", "none")
        db.mark_running(id2)
        db.mark_failed(id2, "test error")

        s = db.summary()
        assert s["total"] == 2
        assert s["completed"] == 1
        assert s["failed"] == 1
        assert s["fastest_ms"] == pytest.approx(1.0)

    def test_get_nonexistent_run(self, db):
        assert db.get_run("does-not-exist") is None


# ── Layer 5: API ──────────────────────────────────────────────────────────

from benchmarking.api.server import app as flask_app


class TestAPI:
    """Flask endpoint contracts."""

    @pytest.fixture()
    def client(self):
        flask_app.config["TESTING"] = True
        with flask_app.test_client() as c:
            yield c

    def test_get_workloads(self, client):
        r = client.get("/api/workloads")
        assert r.status_code == 200
        data = r.get_json()
        assert "european" in data
        assert "schema" in data["european"]
        assert len(data["european"]["schema"]) > 0

    def test_get_engines(self, client):
        r = client.get("/api/engines")
        assert r.status_code == 200
        data = r.get_json()
        assert "cpu" in data
        assert "jax" in data
        assert "european" in data["cpu"]["supported_workloads"]
        # Verify supported_ad_modes are exposed
        assert "supported_ad_modes" in data["cpu"]
        assert data["cpu"]["supported_ad_modes"] == ["none"]
        assert "forward" in data["jax"]["supported_ad_modes"]

    def test_submit_run_returns_201(self, client):
        payload = {
            "workload_type": "european",
            "engine": "cpu",
            "ad_mode": "none",
            "config": {
                "S0": 100, "K": 100, "r": 0.05, "sigma": 0.2,
                "T": 1.0, "M": 1000, "seed": 42, "option_type": "call",
            },
        }
        r = client.post("/api/runs", json=payload)
        assert r.status_code == 201
        data = r.get_json()
        assert "id" in data
        assert data["status"] == "pending"

    def test_submit_run_validates_config(self, client):
        payload = {
            "workload_type": "european",
            "engine": "cpu",
            "config": {"S0": -100, "K": 100, "r": 0.05, "sigma": 0.2, "T": 1.0,
                       "M": 1000, "seed": 42},
        }
        r = client.post("/api/runs", json=payload)
        assert r.status_code == 400

    def test_submit_run_rejects_unknown_engine(self, client):
        payload = {
            "workload_type": "european",
            "engine": "quantum",
            "config": {"S0": 100, "K": 100, "r": 0.05, "sigma": 0.2, "T": 1.0,
                       "M": 1000, "seed": 42},
        }
        r = client.post("/api/runs", json=payload)
        assert r.status_code == 400

    def test_submit_run_rejects_unsupported_ad_mode(self, client):
        """CPU engine + forward AD should be rejected at submission time."""
        payload = {
            "workload_type": "european",
            "engine": "cpu",
            "ad_mode": "forward",
            "config": {"S0": 100, "K": 100, "r": 0.05, "sigma": 0.2, "T": 1.0,
                       "M": 1000, "seed": 42},
        }
        r = client.post("/api/runs", json=payload)
        assert r.status_code == 400

    def test_get_run_by_id(self, client):
        # Submit first
        payload = {
            "workload_type": "european",
            "engine": "cpu",
            "config": {"S0": 100, "K": 100, "r": 0.05, "sigma": 0.2, "T": 1.0,
                       "M": 500, "seed": 1},
        }
        r = client.post("/api/runs", json=payload)
        run_id = r.get_json()["id"]
        # Fetch
        r2 = client.get(f"/api/runs/{run_id}")
        assert r2.status_code == 200
        assert r2.get_json()["id"] == run_id

    def test_get_run_404(self, client):
        r = client.get("/api/runs/nonexistent-id")
        assert r.status_code == 404

    def test_list_runs(self, client):
        r = client.get("/api/runs")
        assert r.status_code == 200
        assert isinstance(r.get_json(), list)

    def test_summary_endpoint(self, client):
        r = client.get("/api/summary")
        assert r.status_code == 200
        data = r.get_json()
        assert "total" in data
        assert "completed" in data
        assert "by_workload" in data
        assert "by_engine" in data


# ── Layer 6: End-to-end ──────────────────────────────────────────────────

class TestEndToEnd:
    """
    Full flow: define config → run engine → create result → store in DB
    → serialize to JSON → deserialize → verify values match.
    """

    def test_european_cpu_full_flow(self):
        """Quant defines European call → CPU engine → runner → DB → read back."""
        # 1. Define workload
        config = EuropeanOptionConfig(
            S0=100, K=100, r=0.05, sigma=0.2, T=1.0, M=10_000, seed=42,
        )
        config.validate()

        # 2. Run through engine
        engine = CPUMonteCarloEngine()
        runner = BenchmarkRunner(engine, name="e2e-test")
        result = runner.run(config, num_warmup=1, num_runs=3, ad_mode="none")

        # 3. Verify numerical correctness
        bs_price = black_scholes_call(config)
        assert abs(result.result - bs_price) / bs_price < 0.03

        # 4. Store in DB
        with tempfile.TemporaryDirectory() as tmp:
            db = BenchmarkDB(db_path=Path(tmp) / "test.db")
            run_id = db.create_run(config.to_dict(), "cpu", "none")
            db.mark_running(run_id)
            db.mark_completed(
                run_id=run_id,
                result_value=result.result,
                mean_runtime_ms=result.mean_runtime * 1000,
                std_runtime_ms=result.std_runtime * 1000,
                ad_overhead_ratio=result.ad_overhead_ratio,
            )

            # 5. Read back from DB
            row = db.get_run(run_id)
            assert row["status"] == "completed"
            assert row["result_value"] == pytest.approx(result.result)
            assert row["mean_runtime_ms"] == pytest.approx(result.mean_runtime * 1000)

            # 6. Verify config round-trips through DB
            stored_config = config_from_dict(row["config"])
            assert stored_config.config_hash() == config.config_hash()

    def test_asian_jax_full_flow(self):
        """Asian option through JAX engine end-to-end."""
        config = AsianOptionConfig(
            S0=100, K=100, r=0.05, sigma=0.2, T=1.0, N=50,
            averaging="arithmetic", M=5000, seed=7,
        )
        config.validate()

        engine = JAXMonteCarloEngine()
        runner = BenchmarkRunner(engine, name="e2e-asian-jax")
        result = runner.run(config, num_warmup=1, num_runs=2, ad_mode="none")

        assert result.result > 0
        assert len(result.runtimes) == 2

        # Serialize → deserialize round-trip
        d = result.to_dict()
        restored = BenchmarkResult.from_dict(d)
        assert restored.result == result.result
        assert restored.config.config_hash() == config.config_hash()

    def test_all_workloads_through_cpu(self):
        """Every workload type prices successfully through the CPU engine."""
        configs = {
            "european": EuropeanOptionConfig(M=1000, seed=1),
            "asian": AsianOptionConfig(N=10, M=1000, seed=1),
            "barrier": BarrierOptionConfig(N=10, M=1000, seed=1),
            "basket": BasketOptionConfig(n_assets=2, N=4, M=1000, seed=1),
        }
        engine = CPUMonteCarloEngine()
        for wtype, config in configs.items():
            config.validate()
            price, _ = engine.run(config)
            assert isinstance(price, float), f"{wtype} did not return float"
            assert price >= 0, f"{wtype} returned negative price: {price}"
