"""Negative compliance tests proving ZERO raw prices, tickers, banned phrases,
directional calls, or email PII leak in any HTTP response or SSE stream.

Seeds a real temporary server-side parquet warehouse with realistic US equity
tickers (AAPL, MSFT, JPM, BAC) and asserts that NONE of those tickers, raw
prices, banned phrases, or emails ever appear in ANY HTTP response body,
SSE frame, telemetry export, or Gate-1 report.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from data import warehouse as wh
from sift import flags
from sift.cards import copy as card_copy
from sift.web.app import create_app
from sift.web.compliance import (
    FORBIDDEN_RESPONSE_KEYS,
    ComplianceError,
    guard_api_payload,
)
from sift.web.telemetry import gate1_report

REAL_TICKERS = ("AAPL", "MSFT", "JPM", "BAC")
SECRET_EMAIL = "quant.researcher@hedgefund.example"


@pytest.fixture(autouse=True)
def _reset_flags():
    flags.reset()
    yield
    flags.reset()


def _seed_parquet_warehouse(root: Path) -> None:
    """Create a realistic server-side EODHD parquet warehouse with real ticker names."""
    rng = np.random.Generator(np.random.PCG64(2026))
    dates = pd.bdate_range("2019-01-02", periods=1600).strftime("%Y-%m-%d").tolist()
    base_lx = np.cumsum(rng.normal(0.0, 0.01, len(dates)))
    # Plant AAPL/MSFT as a cointegrated pair, JPM/BAC as random walks.
    e = rng.normal(0.0, 1.0, len(dates))
    s = np.zeros(len(dates))
    for t in range(1, len(dates)):
        s[t] = 0.70 * s[t - 1] + 10.0 * e[t]
    series_map = {
        "AAPL": 150.0 * np.exp(base_lx),
        "MSFT": 150.0 * np.exp(base_lx + s / 100.0),
        "JPM": 120.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(dates)))),
        "BAC": 35.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(dates)))),
    }
    for sym, px in series_map.items():
        df = pd.DataFrame({
            "date": dates,
            "open": px * 0.999,
            "high": px * 1.005,
            "low": px * 0.995,
            "close": px,
            "adjusted_close": px,
            "volume": np.full(len(dates), 1_000_000.0),
            "dollar_volume": px * 1_000_000.0,
            "symbol": f"{sym}.US",
            "exchange": "US",
        })
        wh.write_parquet_atomic(df, wh.eod_parquet_path(str(root), "US", sym))
    wh.write_vintage(str(root))


def _assert_no_leaks(
    text: str,
    *,
    allow_legal_negations: bool = False,
    context: str = "",
) -> None:
    # 1. Zero real ticker symbols anywhere in raw response text
    for tk in REAL_TICKERS:
        assert not re.search(rf"\b{tk}\b", text), f"ticker {tk} leaked in {context}: {text[:300]}"
    # 2. Zero email PII anywhere in response text
    assert SECRET_EMAIL not in text, f"email leaked in {context}"
    assert "hedgefund.example" not in text, f"email domain leaked in {context}"
    # 3. Zero raw price / OHLCV keys
    for bad_key in ("adjusted_close", "raw_bars", "params_json", "ohlcv", "dollar_volume"):
        assert f'"{bad_key}"' not in text, f"forbidden key {bad_key} leaked in {context}"
    # 4. Zero banned phrases in parsed JSON structure (delegating to compliance guard)
    if text.strip().startswith("{"):
        parsed = json.loads(text)
        guard_api_payload(parsed)
        if not allow_legal_negations:
            low = text.lower()
            for word in ("buy", "sell", "long", "short"):
                assert not re.search(rf"\b{word}\b", low), (
                    f"directional word {word!r} leaked in {context}"
                )


def test_zero_ticker_price_banned_phrase_or_pii_leak_across_all_endpoints(tmp_path: Path):
    wh_root = tmp_path / "warehouse"
    _seed_parquet_warehouse(wh_root)

    app = create_app(
        data_dir=tmp_path / "data",
        warehouse_dir=wh_root,
        card_secret=b"compliance-test-secret",
        allow_synthetic_examples=False,
        s2_B=200,
    )
    with TestClient(app) as tc:
        rt = app.state.runtime
        inv = rt.call(rt.auth.create_invite, wave="wave-1", max_uses=2)

        # Onboarding & health
        r_ob = tc.get("/api/onboarding")
        assert r_ob.status_code == 200
        _assert_no_leaks(r_ob.text, allow_legal_negations=True, context="GET /api/onboarding")

        # Signup
        r_su = tc.post(
            "/api/auth/signup",
            json={
                "email": SECRET_EMAIL,
                "password": "very-strong-password-99",
                "invite_code": inv["code"],
                "tos_accepted": True,
            },
        )
        assert r_su.status_code == 201
        _assert_no_leaks(r_su.text, context="POST /api/auth/signup")
        token = r_su.json()["session_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # /me
        r_me = tc.get("/api/auth/me", headers=headers)
        assert r_me.status_code == 200
        _assert_no_leaks(r_me.text, context="GET /api/auth/me")

        # Catalog backed by real AAPL/MSFT/JPM/BAC parquet warehouse (zero pre-screened shortlist)
        r_cat = tc.get("/api/mining/catalog", headers=headers)
        assert r_cat.status_code == 200
        _assert_no_leaks(r_cat.text, context="GET /api/mining/catalog")
        cat_body = r_cat.json()
        assert cat_body["warehouse_ready"] is True
        assert cat_body["candidates"] == []
        assert len(cat_body["subspaces"]) >= 1

        # Run direct, anchored, and uniform_random slot modes and inspect SSE + poll + trials + cards
        slot_runs = [
            {
                "slot_a": {"kind": "symbol", "value": "AAPL"},
                "slot_b": {"kind": "symbol", "value": "MSFT"},
                "scope_id": "free_5y_top1k",
            },
            {
                "slot_a": {"kind": "symbol", "value": "JPM"},
                "slot_b": {"kind": "random", "subspace": "all"},
                "scope_id": "free_5y_top1k",
            },
            {
                "slot_a": {"kind": "random", "subspace": "all"},
                "slot_b": {"kind": "random", "subspace": "all"},
                "scope_id": "free_5y_top1k",
            },
        ]
        minted_card_ids = []
        for run_body in slot_runs:
            r_run = tc.post(
                "/api/mining/runs",
                headers=headers,
                json=run_body,
            )
            assert r_run.status_code == 202
            _assert_no_leaks(r_run.text, context="POST /api/mining/runs")
            run_id = r_run.json()["run"]["run_id"]

            # SSE stream must be completely clean
            with tc.stream("GET", f"/api/mining/runs/{run_id}/stream", headers=headers) as sse_resp:
                assert sse_resp.status_code == 200
                sse_body = "".join(sse_resp.iter_text())
            _assert_no_leaks(sse_body, context=f"SSE {run_id}")

            # Wait for completion and inspect poll snapshot
            for _ in range(100):
                r_poll = tc.get(f"/api/mining/runs/{run_id}", headers=headers)
                if r_poll.json()["run"]["status"] == "completed":
                    break
                time.sleep(0.05)
            _assert_no_leaks(
                r_poll.text, allow_legal_negations=True, context=f"GET /api/mining/runs/{run_id}"
            )
            cid = r_poll.json()["run"].get("card_id")
            if cid:
                minted_card_ids.append(cid)

        # Trial history, cards, vault, connections
        for endpoint in ("/api/trials", "/api/cards", "/api/vault", "/api/connections"):
            resp = tc.get(endpoint, headers=headers)
            assert resp.status_code == 200
            _assert_no_leaks(resp.text, allow_legal_negations=True, context=f"GET {endpoint}")

        if minted_card_ids:
            cid = minted_card_ids[0]
            r_face = tc.get(f"/api/cards/{cid}/face?with_image=true", headers=headers)
            assert r_face.status_code == 200
            _assert_no_leaks(r_face.text, allow_legal_negations=True, context="GET /api/cards/face")

            # Connection label carrying a ticker or banned phrase is rejected fail-closed!
            r_bad_ticker = tc.post(
                "/api/connections",
                headers=headers,
                json={"card_id": cid, "label": "watch AAPL spread"},
            )
            assert r_bad_ticker.status_code == 400
            r_bad_phrase = tc.post(
                "/api/connections",
                headers=headers,
                json={"card_id": cid, "label": "buy guaranteed profit now"},
            )
            assert r_bad_phrase.status_code == 400

        # Raw symbols/prices injected into mining run are refused fail-closed
        r_inject = tc.post(
            "/api/mining/runs",
            headers=headers,
            json={
                "slot_a": {"kind": "random", "subspace": "all"},
                "slot_b": {"kind": "random", "subspace": "all"},
                "tickers": ["AAPL", "MSFT"],
            },
        )
        assert r_inject.status_code == 400

        # Telemetry export & Gate-1 report contain ZERO email PII
        exported = json.dumps(rt.call(rt.telemetry.export_events))
        report = json.dumps(rt.call(gate1_report, rt.telemetry))
        assert SECRET_EMAIL not in exported
        assert SECRET_EMAIL not in report


def test_compliance_guard_unit_fail_closed():
    with pytest.raises(ComplianceError):
        guard_api_payload({"ok": True, "verdict_copy": "Buy AAPL for guaranteed profit"})
    with pytest.raises(ComplianceError):
        guard_api_payload({"ok": True, "prices": [100.0, 101.2]})
    with pytest.raises(ComplianceError):
        guard_api_payload({"ok": True, "ticker": "MSFT"})
    with pytest.raises(ComplianceError):
        guard_api_payload({"ok": True, "note": "Contact user@domain.com"})
    with pytest.raises(ComplianceError):
        guard_api_payload({"ok": True, "verdict_copy": "Check MSFT correlation"})
    assert "prices" in FORBIDDEN_RESPONSE_KEYS
    assert "ticker" in FORBIDDEN_RESPONSE_KEYS


def test_verdict_framing_copy_clean_and_flag_gated():
    for v, kw in (
        ("killed", {"killed_at": "s1"}),
        ("killed", {"killed_at": "s2"}),
        ("survivor", {"survivor_class": "validated-green"}),
        ("survivor", {"survivor_class": "on-watch-amber"}),
        ("survivor", {"survivor_class": "on-watch-amber-s4-evidence"}),
        ("pending", {}),
    ):
        txt = card_copy.verdict_framing(v, **kw)
        card_copy.assert_copy_clean(txt)
        guard_api_payload({"verdict_copy": txt})


def test_production_mode_enforces_eodhd_beta_license_gate(tmp_path: Path, monkeypatch):
    """docs/vendor-licensing.md §4.4: in production without EODHD written
    confirmation, server fails closed instead of serving EODHD artifacts."""
    wh_root = tmp_path / "warehouse"
    _seed_parquet_warehouse(wh_root)
    monkeypatch.setenv("SIFT_ENV", "production")
    monkeypatch.delenv("SIFT_EODHD_BETA_LICENSE_CONFIRMED", raising=False)
    monkeypatch.delenv("SIFT_ALLOW_SYNTHETIC_EXAMPLES", raising=False)

    app = create_app(
        data_dir=tmp_path / "data",
        warehouse_dir=wh_root,
        card_secret=b"prod-secret",
    )
    with TestClient(app) as tc:
        rt = app.state.runtime
        inv = rt.call(rt.auth.create_invite, wave="wave-1")
        su = tc.post(
            "/api/auth/signup",
            json={
                "email": "prod@quant.io",
                "password": "very-strong-password-99",
                "invite_code": inv["code"],
                "tos_accepted": True,
            },
        )
        headers = {"Authorization": f"Bearer {su.json()['session_token']}"}
        cat = tc.get("/api/mining/catalog", headers=headers)
        assert cat.status_code == 200
        assert cat.json()["candidates"] == []
        r_run = tc.post("/api/mining/runs", headers=headers, json={"example": True})
        assert r_run.status_code == 503
