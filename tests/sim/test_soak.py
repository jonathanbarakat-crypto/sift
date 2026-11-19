"""Load-soak sim tests (SIMULATED — infrastructure validation, not
user-behavior evidence).

Small-N smoke soak (CI-sized: 20 users) + determinism of the seed-driven
report view + the SIFT_ENV=test hard guard.
"""

import json
import os

import pytest

from sift.sim.clock import DEFAULT_SEED
from sift.sim.report import deterministic_view, dump_json, render_text
from sift.sim.soak import run_soak
from sift.sim.world import assert_test_env


def test_soak_smoke_20_users(sim_env, tmp_path):
    """CI-sized smoke soak: 20 simulated users x 2 runs through 8 lanes.
    Asserts no timeouts/deadlocks/lane errors and a sane report."""
    res = run_soak(tmp_path / "soak", n_users=20, runs_per_user=2,
                   workers=8, seed=DEFAULT_SEED)
    sec = res.report["sections"]
    assert res.report["sim"] is True
    assert res.report["header"].startswith("SIMULATED")
    assert sec["assertions"]["no_timeouts"] is True
    assert sec["assertions"]["no_lane_errors"] is True
    assert sec["assertions"]["all_runs_completed"] is True
    assert sec["db_contention"]["ledger_trials_persisted"] == 40
    assert sec["db_contention"]["expected_trials"] == 40
    # the soak exercises the real mint path: at least some runs survive to
    # validated-green and mint a card (planted synthetic edge)
    assert sec["run_verdicts"]["survivor_validated_green"] >= 1
    assert sec["vault_capacity_pressure"]["cards_minted"] >= 1
    lat = sec["measured"]["latency_s"]
    assert lat["p50"] > 0 and lat["p95"] >= lat["p50"]
    assert sec["measured"]["queue"]["max_queue_depth"] <= 8
    # report text starts with the banner
    text = render_text(res.report)
    assert text.splitlines()[0].startswith("SIMULATED")


def test_soak_deterministic_view_same_seed(sim_env, tmp_path):
    """Same seed -> identical deterministic report view (sim decisions +
    DB-derived counts); wall-clock measurements are excluded from the view
    but reported under `measured`."""
    r1 = run_soak(tmp_path / "a", n_users=8, runs_per_user=1, workers=4, seed=42)
    r2 = run_soak(tmp_path / "b", n_users=8, runs_per_user=1, workers=4, seed=42)
    v1 = dump_json(deterministic_view(r1.report))
    v2 = dump_json(deterministic_view(r2.report))
    assert v1 == v2
    # the measured block exists but is not part of the deterministic view
    assert "measured" in r1.report["sections"]
    assert "measured" not in json.loads(v1)["sections"]


def test_soak_different_seed_differs(sim_env, tmp_path):
    r1 = run_soak(tmp_path / "a", n_users=8, runs_per_user=1, workers=4, seed=1)
    r2 = run_soak(tmp_path / "b", n_users=8, runs_per_user=1, workers=4, seed=2)
    v1 = dump_json(deterministic_view(r1.report))
    v2 = dump_json(deterministic_view(r2.report))
    assert v1 != v2


def test_sim_refuses_non_test_env(sim_env, monkeypatch, tmp_path):
    """Hard constraint: SIFT_ENV != test -> the sim refuses to run."""
    monkeypatch.setenv("SIFT_ENV", "production")
    with pytest.raises(RuntimeError, match="SIFT_ENV"):
        assert_test_env()
    with pytest.raises(RuntimeError, match="SIFT_ENV"):
        run_soak(tmp_path / "x", n_users=2, runs_per_user=1, workers=2)
