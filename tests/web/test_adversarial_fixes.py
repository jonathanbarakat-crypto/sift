"""Regression tests reproducing & verifying every finding from the PR #68
adversarial security, compliance, concurrency, and gauntlet integrity review."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from data import warehouse as wh
from sift import flags
from sift.web.app import _safe_json, create_app
from sift.web.auth import MIN_INVITE_LEN, AuthError
from sift.web.compliance import ComplianceError, guard_api_payload
from sift.web.mining import REGISTRY_DEPTH_BARS


@pytest.fixture(autouse=True)
def _reset_flags():
    flags.reset()
    yield
    flags.reset()


def _seed_warehouse(root: Path, *, n_bars: int, delist_symbol: str | None = None) -> None:
    rng = np.random.Generator(np.random.PCG64(42))
    dates = pd.bdate_range("2008-01-02", periods=n_bars).strftime("%Y-%m-%d").tolist()
    lx = np.cumsum(rng.normal(0.0, 0.01, n_bars))
    e = rng.normal(0.0, 1.0, n_bars)
    s = np.zeros(n_bars)
    for t in range(1, n_bars):
        s[t] = 0.70 * s[t - 1] + 10.0 * e[t]
    series_map = {
        "AAPL": 150.0 * np.exp(lx),
        "MSFT": 150.0 * np.exp(lx + s / 100.0),
        "F": 12.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n_bars))),
    }
    if delist_symbol:
        # Delisted halfway through (< 50% of bars) — must be retained, not dropped!
        cut = n_bars // 3
        px_d = 40.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, cut)))
        df_d = pd.DataFrame({
            "date": dates[:cut],
            "open": px_d,
            "high": px_d,
            "low": px_d,
            "close": px_d,
            "adjusted_close": px_d,
            "volume": np.full(cut, 500_000.0),
            "dollar_volume": px_d * 500_000.0,
            "symbol": f"{delist_symbol}.US",
            "exchange": "US",
        })
        wh.write_parquet_atomic(df_d, wh.eod_parquet_path(str(root), "US", delist_symbol))

    for sym, px in series_map.items():
        df = pd.DataFrame({
            "date": dates,
            "open": px,
            "high": px,
            "low": px,
            "close": px,
            "adjusted_close": px,
            "volume": np.full(n_bars, 1_000_000.0),
            "dollar_volume": px * 1_000_000.0,
            "symbol": f"{sym}.US",
            "exchange": "US",
        })
        wh.write_parquet_atomic(df, wh.eod_parquet_path(str(root), "US", sym))
    wh.write_vintage(str(root))


def _signup(tc: TestClient, email: str) -> tuple[str, dict[str, str]]:
    rt = tc.app.state.runtime
    inv = rt.fast_call(rt.auth.create_invite, wave="wave-1", max_uses=1)
    r = tc.post(
        "/api/auth/signup",
        json={
            "email": email,
            "password": "correct-horse-battery",
            "invite_code": inv["code"],
            "tos_accepted": True,
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    return body["user"]["user_id"], {"Authorization": f"Bearer {body['session_token']}"}


# ============================================================================
# 1. Data & compliance leak regressions
# ============================================================================

def test_compliance_guard_never_echoes_blocked_token_in_500():
    """PoC #1: _safe_json must NEVER echo the blocked ticker/phrase in its 500 body."""
    resp = _safe_json({"ok": False, "error": "Unknown AAPL candidate"})
    assert resp.status_code == 500
    body = resp.body.decode("utf-8")
    assert "AAPL" not in body
    assert "compliance guard" in body.lower()


def test_compliance_guard_blocks_all_string_fields_nested_b64_and_warehouse_symbols():
    # `error`, `message`, `description` are now checked for banned phrases
    with pytest.raises(ComplianceError):
        guard_api_payload({"ok": False, "error": "guaranteed profit here"})
    with pytest.raises(ComplianceError):
        guard_api_payload({"ok": True, "description": "buy now"})
    # Arbitrary *_b64 keys and nested dicts under png_1x_b64 are NOT skipped
    with pytest.raises(ComplianceError):
        guard_api_payload({"ok": True, "evil_b64": {"ticker": "AAPL"}})
    with pytest.raises(ComplianceError):
        guard_api_payload({"ok": True, "png_1x_b64": {"ticker": "AAPL"}})
    # Single-letter & lowercase warehouse symbols blocked when active
    with pytest.raises(ComplianceError):
        guard_api_payload({"ok": True, "note": "look at aapl"}, extra_symbols={"AAPL", "F"})
    with pytest.raises(ComplianceError):
        guard_api_payload({"ok": True, "note": "watch F closely"}, extra_symbols={"AAPL", "F"})


def test_s0_level_hedge_hidden_and_telemetry_uses_hmac_commitment(tmp_path: Path):
    """Med findings: S0 statistic/threshold must be None; telemetry must NOT
    contain raw SHA-256 config_hash (uses keyed HMAC config_commitment instead)."""
    app = create_app(
        data_dir=tmp_path / "data",
        card_secret=b"hmac-test-secret",
        allow_synthetic_examples=True,
        s2_B=200,
    )
    with TestClient(app) as tc:
        _, h = _signup(tc, "s0@quant.io")
        r = tc.post("/api/mining/runs", headers=h, json={"example": True})
        assert r.status_code == 202
        run_id = r.json()["run"]["run_id"]
        for _ in range(100):
            snap = tc.get(f"/api/mining/runs/{run_id}", headers=h).json()["run"]
            if snap["status"] == "completed":
                break
            time.sleep(0.05)
        s0 = next(s for s in snap["stages"] if s["stage"] == "s0")
        assert s0["statistic"] is None
        assert s0["threshold"] is None

        # Telemetry trial_registered carries config_commitment (not raw config_hash)
        rt = tc.app.state.runtime
        evs = rt.fast_call(rt.telemetry.export_events, event_type="trial_registered")
        assert len(evs) == 1
        assert "config_hash" not in evs[0]["payload"]
        assert "config_commitment" in evs[0]["payload"]


def test_unknown_candidate_id_not_reflected_and_trial_404_is_uniform(tmp_path: Path):
    app = create_app(
        data_dir=tmp_path / "data",
        card_secret=b"oracle-secret",
        allow_synthetic_examples=True,
        s2_B=200,
    )
    with TestClient(app) as tc:
        _, h_a = _signup(tc, "a@quant.io")
        _, h_b = _signup(tc, "b@quant.io")

        # User-supplied ticker in candidate_id is NOT echoed back in 404
        r_bad = tc.post(
            "/api/mining/runs", headers=h_a, json={"candidate_id": "AAPL"}
        )
        assert r_bad.status_code == 404
        assert "AAPL" not in r_bad.text

        # Trial 404 oracle: unknown trial_id vs Alice's trial_id return identical body
        r_run = tc.post("/api/mining/runs", headers=h_a, json={"example": True})
        tid = r_run.json()["run"]["trial_id"]
        r_unknown = tc.get("/api/trials/trial-999999", headers=h_b)
        r_foreign = tc.get(f"/api/trials/{tid}", headers=h_b)
        assert r_unknown.status_code == 404
        assert r_foreign.status_code == 404
        assert r_unknown.json() == r_foreign.json()


# ============================================================================
# 2. Auth, CSRF, rate limits, ToS version, and PII physical separation/deletion
# ============================================================================

def test_auth_hardening_csrf_invite_length_tos_version_and_deletion(tmp_path: Path):
    data_dir = tmp_path / "data"
    app = create_app(
        data_dir=data_dir,
        card_secret=b"auth-hard-secret",
        allow_synthetic_examples=True,
        s2_B=200,
    )
    with TestClient(app) as tc:
        rt = tc.app.state.runtime

        # Short custom invite codes (< 12 chars) rejected
        with pytest.raises(AuthError):
            rt.fast_call(rt.auth.create_invite, code="SHORT1")
        assert MIN_INVITE_LEN == 12

        # text/plain CSRF POST rejected with 415
        r_csrf = tc.post(
            "/api/auth/login",
            content=json.dumps({"email": "x@y.io", "password": "pw"}),
            headers={"Content-Type": "text/plain"},
        )
        assert r_csrf.status_code == 415

        uid, headers = _signup(tc, "delete-me@quant.io")

        # Physical DB separation: telemetry.db has NO beta_user / email table
        assert (data_dir / "auth.db").exists()
        assert (data_dir / "telemetry.db").exists()
        assert (data_dir / "app.db").exists()
        tcon = sqlite3.connect(str(data_dir / "telemetry.db"))
        tables = {r[0] for r in tcon.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        tcon.close()
        assert "beta_user" not in tables
        assert tables == {"telemetry_event", "sqlite_sequence"} or "telemetry_event" in tables

        # Outdated ToS version fails closed on _require_user (403)
        def _downgrade_tos():
            rt.auth.con.execute(
                "UPDATE beta_user SET tos_version='old-v0' WHERE user_id=?", (uid,)
            )
            rt.auth.con.commit()

        rt.fast_call(_downgrade_tos)
        assert tc.get("/api/auth/me", headers=headers).status_code == 403

        def _restore_tos():
            rt.auth.con.execute(
                "UPDATE beta_user SET tos_version='beta-v1-day1' WHERE user_id=?", (uid,)
            )
            rt.auth.con.commit()

        rt.fast_call(_restore_tos)
        assert tc.get("/api/auth/me", headers=headers).status_code == 200

        # Account deletion erases email PII & invalidates session while preserving telemetry
        r_del = tc.post(
            "/api/auth/delete-account", headers=headers, json={"confirm": True}
        )
        assert r_del.status_code == 200 and r_del.json()["deleted"] is True
        assert tc.get("/api/auth/me", headers=headers).status_code == 401
        raw_auth = (data_dir / "auth.db").read_bytes()
        assert b"delete-me@quant.io" not in raw_auth
        # Telemetry remains intact (append-only, anonymized)
        assert rt.fast_call(rt.telemetry.count_events, "account_created") == 1


def test_example_string_false_rejected(tmp_path: Path):
    app = create_app(
        data_dir=tmp_path / "data",
        card_secret=b"bool-secret",
        allow_synthetic_examples=True,
        s2_B=200,
    )
    with TestClient(app) as tc:
        _, h = _signup(tc, "bool@quant.io")
        r = tc.post("/api/mining/runs", headers=h, json={"example": "false"})
        assert r.status_code == 400


# ============================================================================
# 3. Concurrency: non-blocking fast endpoints & coalesced duplicate runs
# ============================================================================

def test_mining_does_not_block_auth_me_and_coalesces_duplicate_runs(tmp_path: Path):
    """PoC #2 & #3:
    - /api/auth/me responds quickly while a mining run executes on _engine_pool.
    - Two concurrent runs of the exact same config coalesce onto one execution
      and BOTH complete cleanly (neither crashes with LedgerError)."""
    app = create_app(
        data_dir=tmp_path / "data",
        card_secret=b"conc-secret",
        allow_synthetic_examples=True,
        s2_B=300,
    )
    with TestClient(app) as tc:
        _, h_a = _signup(tc, "alice-c@quant.io")
        _, h_b = _signup(tc, "bob-c@quant.io")

        r1 = tc.post("/api/mining/runs", headers=h_a, json={"example": True})
        assert r1.status_code == 202
        run1 = r1.json()["run"]["run_id"]

        # While run1 is executing on _engine_pool, /api/auth/me stays fast!
        t0 = time.perf_counter()
        r_me = tc.get("/api/auth/me", headers=h_b)
        dt = time.perf_counter() - t0
        assert r_me.status_code == 200
        assert dt < 0.5, f"/api/auth/me blocked behind mining ({dt:.2f}s)"

        # User B immediately submits the identical config -> coalesces cleanly!
        r2 = tc.post("/api/mining/runs", headers=h_b, json={"example": True})
        assert r2.status_code == 202
        run2 = r2.json()["run"]["run_id"]
        assert r2.json()["run"]["cached"] is False  # no cross-user cache oracle

        # Wait for BOTH runs to complete (neither should fail!)
        for rid, hdr in ((run1, h_a), (run2, h_b)):
            for _ in range(150):
                s = tc.get(f"/api/mining/runs/{rid}", headers=hdr).json()["run"]
                if s["status"] in ("completed", "failed"):
                    assert s["status"] == "completed", s
                    assert s["verdict"] == "survivor"
                    break
                time.sleep(0.05)
            else:
                raise AssertionError(f"run {rid} timed out")

        # Alice owns the card; Bob does NOT receive Alice's card_id (non-transferable)
        s1 = tc.get(f"/api/mining/runs/{run1}", headers=h_a).json()["run"]
        s2 = tc.get(f"/api/mining/runs/{run2}", headers=h_b).json()["run"]
        assert s1["card_id"] is not None
        assert s2["card_id"] is None


def test_failed_run_sse_stream_terminates_cleanly(tmp_path: Path):
    """Med finding: replayed SSE stream for a failed/orphaned run MUST emit done."""
    app = create_app(
        data_dir=tmp_path / "data",
        card_secret=b"sse-term-secret",
        allow_synthetic_examples=True,
        s2_B=200,
    )
    with TestClient(app) as tc:
        uid, h = _signup(tc, "orphan@quant.io")
        rt = tc.app.state.runtime
        rt.fast_call(
            rt.runs.insert_run,
            run_id="run-orphan-1",
            user_id=uid,
            trial_id="trial-000001",
            candidate_id="example-pair-cointegrated",
            scope_id="free_5y_top1k",
            cached=False,
            status="failed",
            verdict="pending",
            survivor_class="none",
            error_message="Run interrupted before completion.",
        )
        with tc.stream("GET", "/api/mining/runs/run-orphan-1/stream", headers=h) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
        assert "event: error" in body
        assert "event: done" in body


# ============================================================================
# 4. Gauntlet integrity: S5 depth gate, frozen S3 epoch, holdout, delisted names
# ============================================================================

def test_gauntlet_integrity_s5_depth_frozen_epoch_holdout_and_delisted(tmp_path: Path):
    # Warehouse with 1600 bars (> 1260 + 252 holdout, but < 4032 registry_depth):
    # - delisted name (< 50% bars) must be retained in catalog panel
    # - 252-bar frozen holdout is locked and holdout_baseline persisted
    # - S3 epoch 1 is frozen with panel_corr & m_screened = N*(N-1)//2
    # - S5 MUST STAY PENDING (never validated-green without 4032 bars!)
    wh_short = tmp_path / "wh_short"
    _seed_warehouse(wh_short, n_bars=1600, delist_symbol="DELX")

    app = create_app(
        data_dir=tmp_path / "data_short",
        warehouse_dir=wh_short,
        card_secret=b"integrity-secret",
        allow_synthetic_examples=False,
        s2_B=200,
    )
    with TestClient(app) as tc:
        rt = tc.app.state.runtime
        # Delisted name retained in server-side catalog panel (not dropped by 50% filter)
        assert "DELX" in rt.catalog.forbidden_symbols
        assert rt.catalog._wh_delist_idx is not None
        delx_idx = rt.catalog._wh_symbols.index("DELX")
        assert rt.catalog._wh_delist_idx[delx_idx] < 1600
        assert rt.catalog._wh_m_screened == 4 * 3 // 2  # all 6 pairs screened

        _, h = _signup(tc, "quant-int@fund.io")
        cat = tc.get("/api/mining/catalog", headers=h).json()["candidates"]
        r = tc.post(
            "/api/mining/runs",
            headers=h,
            json={"candidate_id": cat[0]["candidate_id"], "scope_id": "free_5y_top1k"},
        )
        assert r.status_code == 202
        run_id = r.json()["run"]["run_id"]
        for _ in range(150):
            snap = tc.get(f"/api/mining/runs/{run_id}", headers=h).json()["run"]
            if snap["status"] == "completed":
                break
            time.sleep(0.05)
        assert snap["status"] == "completed"
        # Because n_bars=1600 < 4032 (REGISTRY_DEPTH_BARS), S5 MUST NOT run!
        s5 = next(s for s in snap["stages"] if s["stage"] == "s5")
        assert s5["status"] == "pending"
        assert snap["survivor_class"] != "validated-green"

        # Verify sift #59 admission epoch 1 was frozen with m_screened=6
        ep = rt.engine_call(rt.ledger.stratum_epoch_stats, "free_5y_top1k", 1)
        assert ep is not None
        assert ep["n_eff_raw"] == 6
        assert ep["method"] == "a3_avg_corr"


def test_warehouse_refuses_silent_scope_shrink(tmp_path: Path):
    """Med finding: a 400-bar warehouse MUST refuse free_5y_top1k (1260 bars)
    instead of silently shrinking the window under a false scope label."""
    wh_tiny = tmp_path / "wh_tiny"
    _seed_warehouse(wh_tiny, n_bars=400)

    app = create_app(
        data_dir=tmp_path / "data_tiny",
        warehouse_dir=wh_tiny,
        card_secret=b"shrink-secret",
        allow_synthetic_examples=False,
        s2_B=200,
    )
    with TestClient(app) as tc:
        _, h = _signup(tc, "shrink@fund.io")
        cat = tc.get("/api/mining/catalog", headers=h).json()["candidates"]
        r_5y = tc.post(
            "/api/mining/runs",
            headers=h,
            json={"candidate_id": cat[0]["candidate_id"], "scope_id": "free_5y_top1k"},
        )
        assert r_5y.status_code == 400
        assert REGISTRY_DEPTH_BARS == 4032
