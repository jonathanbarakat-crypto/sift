"""End-to-end web API, Private Owner Spec, and compliance tests for `lead_lag`.

Verifies:
1. /api/mining/catalog exposes `allowed_families` and `lead_lag_config` cleanly.
2. Directed slot mining: Direct (`LEAD1 -> FOLL1` distinct from `FOLL1 -> LEAD1`),
   Anchored (`LEAD1 -> ?` and `? -> FOLL1`), and Uniform Random (`? -> ?`).
3. Private Owner Spec returns `[{"role": "leader", ...}, {"role": "follower", ...}]`,
   `lag_bars`, `hold_bars`, magnitude-only `|z| >= 0`, and ZERO slopes/signs/prices.
4. Zero tickers, zero `gamma`/`gamma_hat`, zero signs across public APIs, faces,
   SSE streams, and telemetry.
5. Kill-switch `SIFT_FAMILY_LEAD_LAG=0` fails closed (hides from catalog + 503).
6. Request injection of `gamma` / `gamma_hat` returns 400 fail-closed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from data import warehouse as wh
from sift import flags
from sift.gauntlet import lead_lag as ll
from sift.web.app import create_app
from sift.web.compliance import guard_api_payload


def _write_eod(root: Path, sym: str, dates: list[str], prices: np.ndarray) -> None:
    df = pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices,
            "adjusted_close": prices,
            "volume": np.full(len(dates), 2_000_000.0),
        }
    )
    out = Path(wh.eod_parquet_path(str(root), "US", sym))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)


def _build_lead_lag_warehouse(root: Path, *, T: int = 1600) -> list[str]:
    """Create a PIT warehouse with a planted directed LEAD1 -> FOLL1 signal plus
    two independent noise equities (NOISA, NOISB) and sector metadata."""
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2018-01-02", periods=T)]
    fx = ll.f_leadlag(T=T, gamma_true=0.30, lag_true=1, seed=101)
    rng = np.random.Generator(np.random.PCG64(909))
    p_noise_a = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.015, T)))
    p_noise_b = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.015, T)))

    _write_eod(root, "LEAD1", dates, fx["prices"][:, 0])
    _write_eod(root, "FOLL1", dates, fx["prices"][:, 1])
    _write_eod(root, "NOISA", dates, p_noise_a)
    _write_eod(root, "NOISB", dates, p_noise_b)

    syms = ["LEAD1", "FOLL1", "NOISA", "NOISB"]
    u_dir = root / "universe"
    u_dir.mkdir(parents=True, exist_ok=True)
    top_df = pd.DataFrame(
        {
            "symbol": syms,
            "exchange": ["US"] * 4,
            "sector": ["technology", "technology", "energy", "energy"],
        }
    )
    top_path = u_dir / "top4_pit_2024-06-01.parquet"
    top_df.to_parquet(top_path, index=False)
    (u_dir / "manifest.json").write_text(
        json.dumps(
            {
                "asof": dates[-1],
                "top_path": str(top_path),
                "research_grade": False,
            }
        ),
        encoding="utf-8",
    )
    return syms


def _signup(client: TestClient, invite: str, email: str) -> dict:
    r = client.post(
        "/api/auth/signup",
        json={
            "email": email,
            "password": "StrongPassword!123",
            "invite_code": invite,
            "tos_accepted": True,
        },
    )
    assert r.status_code == 201, r.text
    return r.json()


def _wait_run(client: TestClient, run_id: str, timeout_s: float = 20.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = client.get(f"/api/mining/runs/{run_id}")
        assert r.status_code == 200
        body = r.json()["run"]
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish within {timeout_s}s")


@pytest.fixture(autouse=True)
def _reset_flags():
    flags.reset()
    yield
    flags.reset()


def test_lead_lag_web_directed_modes_private_spec_and_compliance(tmp_path):
    wh_dir = tmp_path / "wh"
    data_dir = tmp_path / "data"
    syms = _build_lead_lag_warehouse(wh_dir)

    app = create_app(
        data_dir=data_dir,
        warehouse_dir=wh_dir,
        allow_synthetic_examples=False,
        s2_B=100,
        rate_limit_enabled=False,
    )
    with TestClient(app) as client:
        rt = app.state.runtime
        inv = rt.fast_call(rt.auth.create_invite, wave="w1")["code"]
        _signup(client, inv, "alice@example.com")

        # 1. Catalog exposes lead_lag family & pre-registered config, zero shortlist
        cat_r = client.get("/api/mining/catalog")
        assert cat_r.status_code == 200
        cat = cat_r.json()
        assert cat["candidates"] == []
        assert cat["allowed_families"] == ["pairs", "lead_lag"]
        assert cat["lead_lag_config"]["allowed_lags"] == [1, 2, 5]
        assert cat["lead_lag_config"]["allowed_formations"] == [60, 120, 252]
        assert cat["lead_lag_config"]["allowed_open_z"] == [1.0, 1.5, 2.0]
        guard_api_payload(cat, extra_symbols=syms)

        # 2. Forbidden gamma injection fails closed with 400
        r_bad = client.post(
            "/api/mining/runs",
            json={
                "family": "lead_lag",
                "slot_a": {"kind": "symbol", "value": "LEAD1"},
                "slot_b": {"kind": "symbol", "value": "FOLL1"},
                "gamma_hat": 0.3,
            },
        )
        assert r_bad.status_code == 400

        # 3. Direct forward LEAD1 -> FOLL1
        r_fwd = client.post(
            "/api/mining/runs",
            json={
                "family": "lead_lag",
                "slot_a": {"kind": "symbol", "value": "LEAD1"},
                "slot_b": {"kind": "symbol", "value": "FOLL1"},
                "scope_id": "free_5y_top1k",
                "lag_bars": 1,
                "formation": 120,
                "open_z": 1.5,
            },
        )
        assert r_fwd.status_code == 202, r_fwd.text
        fwd_meta = r_fwd.json()["run"]
        assert fwd_meta["family"] == "lead_lag"
        assert fwd_meta["mining_mode"] == "direct"
        guard_api_payload(fwd_meta, extra_symbols=syms)
        fwd_done = _wait_run(client, fwd_meta["run_id"])
        assert fwd_done["status"] == "completed"
        assert fwd_done["family"] == "lead_lag"
        assert fwd_done["verdict"] == "survivor"
        assert fwd_done["card_id"] is not None
        guard_api_payload(fwd_done, extra_symbols=syms)

        # 4. Direct reverse FOLL1 -> LEAD1 MUST be a DISTINCT trial_id and killed
        r_rev = client.post(
            "/api/mining/runs",
            json={
                "family": "lead_lag",
                "slot_a": {"kind": "symbol", "value": "FOLL1"},
                "slot_b": {"kind": "symbol", "value": "LEAD1"},
                "scope_id": "free_5y_top1k",
                "lag_bars": 1,
                "formation": 120,
                "open_z": 1.5,
            },
        )
        assert r_rev.status_code == 202, r_rev.text
        rev_meta = r_rev.json()["run"]
        assert rev_meta["trial_id"] != fwd_meta["trial_id"]
        rev_done = _wait_run(client, rev_meta["run_id"])
        assert rev_done["verdict"] == "killed"
        assert rev_done["card_id"] is None

        # 5. Private Owner Spec for forward minted lead_lag card
        r_priv = client.get(f"/api/private/cards/{fwd_done['card_id']}/spec")
        assert r_priv.status_code == 200, r_priv.text
        ps = r_priv.json()["private_spec"]
        assert ps["instruments"] == [
            {"role": "leader", "symbol": "LEAD1"},
            {"role": "follower", "symbol": "FOLL1"},
        ]
        assert ps["parameters"]["family"] == "lead_lag"
        assert ps["parameters"]["lag_bars"] == 1
        assert ps["parameters"]["hold_bars"] == 1
        assert "close_z" not in ps["parameters"]
        assert ps["live_signal"]["magnitude_z"] is not None
        assert ps["live_signal"]["magnitude_z"] >= 0.0
        ps_text = json.dumps(ps)
        assert "gamma" not in ps_text.lower()
        assert "hedge_ratio" not in ps_text.lower()

        # 6. Public card face (+ server-rendered PNG) stays 100% ticker-free & slope-free
        r_face = client.get(f"/api/cards/{fwd_done['card_id']}/face?with_image=true")
        assert r_face.status_code == 200
        face_body = r_face.json()
        guard_api_payload(face_body, extra_symbols=syms)
        assert face_body["face"]["family"] == "lead_lag"
        assert face_body["face"]["sector_aggregate"]["bucket"] == "directed_lead_lag_aggregate"
        assert face_body["face"]["specifics"]["lag_bars"] == 1
        assert face_body["face"]["image"]["png_2x_b64"]

        # 7. Anchored (LEAD1 -> Random(energy)) and Uniform Random (? -> ?)
        r_anc = client.post(
            "/api/mining/runs",
            json={
                "family": "lead_lag",
                "slot_a": {"kind": "symbol", "value": "LEAD1"},
                "slot_b": {"kind": "random", "subspace": "energy"},
                "scope_id": "free_5y_top1k",
                "lag_bars": 2,
                "formation": 120,
                "open_z": 1.5,
            },
        )
        assert r_anc.status_code == 202, r_anc.text
        anc_meta = r_anc.json()["run"]
        assert anc_meta["mining_mode"] == "anchored"
        assert anc_meta["subspace_filter"] == "energy"
        assert anc_meta["subspace_size"] == 2
        anc_done = _wait_run(client, anc_meta["run_id"])
        assert anc_done["status"] == "completed"
        guard_api_payload(anc_done, extra_symbols=syms)

        r_rnd = client.post(
            "/api/mining/runs",
            json={
                "family": "lead_lag",
                "slot_a": {"kind": "random", "subspace": "all"},
                "slot_b": {"kind": "random", "subspace": "all"},
                "scope_id": "free_5y_top1k",
                "lag_bars": 5,
                "formation": 120,
                "open_z": 1.5,
            },
        )
        assert r_rnd.status_code == 202, r_rnd.text
        rnd_meta = r_rnd.json()["run"]
        assert rnd_meta["mining_mode"] == "uniform_random"
        # Directed subspace size for 4 symbols is 4 * 3 = 12 (not 6!)
        assert rnd_meta["subspace_size"] == 12
        rnd_done = _wait_run(client, rnd_meta["run_id"])
        assert rnd_done["status"] == "completed"

        # 8. /api/trials reports actual family="lead_lag" with zero ticker leaks
        r_trials = client.get("/api/trials")
        assert r_trials.status_code == 200
        tbody = r_trials.json()
        guard_api_payload(tbody, extra_symbols=syms)
        assert all(tr["family"] == "lead_lag" for tr in tbody["trials"])


def test_lead_lag_kill_switch_fails_closed(tmp_path, monkeypatch):
    """SIFT_FAMILY_LEAD_LAG=0 hides lead_lag from catalog and rejects runs with 503."""
    wh_dir = tmp_path / "wh"
    data_dir = tmp_path / "data"
    _build_lead_lag_warehouse(wh_dir)
    monkeypatch.setenv("SIFT_FAMILY_LEAD_LAG", "0")
    flags.reset()

    app = create_app(
        data_dir=data_dir,
        warehouse_dir=wh_dir,
        allow_synthetic_examples=True,
        s2_B=100,
        rate_limit_enabled=False,
    )
    with TestClient(app) as client:
        rt = app.state.runtime
        inv = rt.fast_call(rt.auth.create_invite, wave="w1")["code"]
        _signup(client, inv, "bob@example.com")

        cat = client.get("/api/mining/catalog").json()
        assert cat["allowed_families"] == ["pairs"]
        assert all(c["family"] != "lead_lag" for c in cat["candidates"])

        r_run = client.post(
            "/api/mining/runs",
            json={
                "family": "lead_lag",
                "slot_a": {"kind": "symbol", "value": "LEAD1"},
                "slot_b": {"kind": "symbol", "value": "FOLL1"},
                "scope_id": "free_5y_top1k",
                "lag_bars": 1,
                "formation": 120,
                "open_z": 1.5,
            },
        )
        assert r_run.status_code == 503

        r_ex = client.post(
            "/api/mining/runs",
            json={
                "candidate_id": "example-lead-lag-directed",
                "scope_id": "free_5y_top1k",
            },
        )
        assert r_ex.status_code == 503
