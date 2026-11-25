"""Tests for SEC EDGAR sector pipeline, Unified Slot Mining Modes (direct /
anchored / uniform_random with zero hidden pre-screening), and Authenticated
Owner-Only Private Spec endpoints (v4.2 §5.1, docs/mining-modes-policy.md).
"""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from data import sectors as sector_meta
from data import universe as pit_universe
from data import warehouse as wh
from sift import flags
from sift.web.app import create_app
from sift.web.compliance import (
    PRIVATE_FORBIDDEN_KEYS,
    ComplianceError,
    guard_api_payload,
    guard_owner_private_payload,
)


@pytest.fixture(autouse=True)
def _reset_flags(monkeypatch):
    monkeypatch.setenv("SIFT_CARDS_V4_1", "1")
    monkeypatch.setenv("SIFT_DECKS_V4_2", "1")
    monkeypatch.setenv("SIFT_OWNER_PRIVATE_SPEC", "1")
    monkeypatch.setenv("SIFT_ENV", "test")
    flags.reset()
    yield
    flags.reset()


def _write_eod(wh_root: Path, sym: str, dates: pd.DatetimeIndex, px: np.ndarray, *, vol: float = 1_000_000.0) -> None:
    df = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "open": px * 0.999,
        "high": px * 1.005,
        "low": px * 0.995,
        "close": px,
        "adjusted_close": px,
        "volume": np.full(len(dates), float(vol)),
        "dollar_volume": px * float(vol),
        "symbol": f"{sym}.US",
        "exchange": "US",
    })
    wh.write_parquet_atomic(df, wh.eod_parquet_path(str(wh_root), "US", sym))
    wh.write_json_atomic([], wh.splits_json_path(str(wh_root), "US", sym))


def _seed_sectored_pit_warehouse(wh_root: Path) -> str:
    """Seed a PIT warehouse with planted cointegrated tech pair (AAPL/MSFT),
    uncorrelated financials pair (JPM/BAC), energy stock (XOM), and quarantined
    corrupt symbol (BADQ), plus SEC-style sector metadata."""
    rng = np.random.Generator(np.random.PCG64(2026))
    dates = pd.bdate_range("2019-01-02", periods=1600)
    asof = dates[-1].strftime("%Y-%m-%d")

    # AAPL / MSFT: planted cointegrated pair (survives S0-S2 -> mints card)
    base = np.cumsum(rng.normal(0.0, 0.01, len(dates)))
    e = rng.normal(0.0, 1.0, len(dates))
    s = np.zeros(len(dates))
    for t in range(1, len(dates)):
        s[t] = 0.70 * s[t - 1] + 10.0 * e[t]
    _write_eod(wh_root, "AAPL", dates, 150.0 * np.exp(base))
    _write_eod(wh_root, "MSFT", dates, 150.0 * np.exp(base + s / 100.0))

    # JPM / BAC: independent random walks (killed at S1/S2)
    _write_eod(wh_root, "JPM", dates, 120.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(dates)))))
    _write_eod(wh_root, "BAC", dates, 35.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(dates)))))

    # XOM: independent energy stock
    _write_eod(wh_root, "XOM", dates, 90.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(dates)))))

    # BADQ: sentinel price -> quarantined by data.universe
    badq = np.full(len(dates), 100.0)
    badq[50] = 999999.9999
    _write_eod(wh_root, "BADQ", dates, badq)

    # Seed meta/sectors/US.parquet via CSV helper (using official SEC SIC codes)
    csv_path = wh_root / "seed_sectors.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(
        "symbol,sic\n"
        "AAPL,3571\n"   # Electronic Computers -> technology
        "MSFT,7372\n"   # Prepackaged Software -> technology
        "JPM,6021\n"    # National Commercial Banks -> financials
        "BAC,6021\n"    # National Commercial Banks -> financials
        "XOM,2911\n",   # Petroleum Refining -> energy
        encoding="utf-8",
    )
    sec_df = sector_meta.build_sectors_from_csv(str(csv_path), exchange="US")
    wh.write_parquet_atomic(sec_df, sector_meta.sectors_parquet_path(str(wh_root), "US"))
    csv_path.unlink()

    rc = pit_universe.main(["--warehouse", str(wh_root), "--n", "20", "--asof", asof])
    assert rc == 0
    return asof


def _signup(tc: TestClient, email: str) -> dict:
    rt = tc.app.state.runtime
    inv = rt.call(rt.auth.create_invite, wave="wave-1", max_uses=1)
    r = tc.post(
        "/api/auth/signup",
        json={
            "email": email,
            "password": "very-strong-password-99",
            "invite_code": inv["code"],
            "tos_accepted": True,
        },
    )
    assert r.status_code == 201, r.text
    return r.json()


def _wait_run(tc: TestClient, run_id: str, headers: dict, timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = tc.get(f"/api/mining/runs/{run_id}", headers=headers)
        assert r.status_code == 200
        run = r.json()["run"]
        if run["status"] in ("completed", "failed"):
            return run
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish within {timeout}s")


# ============================================================================
# 1. Sector Metadata Pipeline (data/sectors.py + data/universe.py)
# ============================================================================


def test_sic_mapping_and_submissions_zip_and_universe_sector_column(tmp_path: Path):
    assert sector_meta.sic_to_sector_bucket(3571) == "technology"
    assert sector_meta.sic_to_sector_bucket("7372") == "technology"
    assert sector_meta.sic_to_sector_bucket(6021) == "financials"
    assert sector_meta.sic_to_sector_bucket(6798) == "real_estate"
    assert sector_meta.sic_to_sector_bucket(2834) == "healthcare"
    assert sector_meta.sic_to_sector_bucket(2911) == "energy"
    assert sector_meta.sic_to_sector_bucket(4911) == "utilities"
    assert sector_meta.sic_to_sector_bucket(4813) == "communication"
    assert sector_meta.sic_to_sector_bucket(None) == "unclassified"
    assert sector_meta.sic_to_sector_bucket("bad") == "unclassified"
    assert sector_meta.normalize_sector_bucket("Information Technology") == "technology"
    assert sector_meta.normalize_sector_bucket("Health Care") == "healthcare"
    assert sector_meta.normalize_sector_bucket("Real Estate") == "real_estate"

    # Build from synthetic SEC EDGAR bulk submissions.zip (active + delisted issuer)
    zpath = tmp_path / "submissions.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("CIK0000320193.json", json.dumps({"sic": "3571", "tickers": ["AAPL"]}))
        zf.writestr("CIK0000034088.json", json.dumps({"sic": "2911", "tickers": ["XOM"]}))
        zf.writestr("CIK0009999999.json", json.dumps({"sic": "6021", "tickers": ["OLDK"]}))
    df = sector_meta.build_sectors_from_submissions_zip(str(zpath), exchange="US")
    got = dict(zip(df["symbol"], df["sector"]))
    assert got == {"AAPL.US": "technology", "OLDK.US": "financials", "XOM.US": "energy"}

    # Verify universe.main populates the sector column and manifest sector_counts
    wh_root = tmp_path / "warehouse"
    asof = _seed_sectored_pit_warehouse(wh_root)
    top_df = pd.read_parquet(wh_root / "universe" / f"top20_pit_{asof}.parquet")
    assert "sector" in top_df.columns
    sec_by_sym = dict(zip(top_df["symbol"], top_df["sector"]))
    assert sec_by_sym["AAPL.US"] == "technology"
    assert sec_by_sym["MSFT.US"] == "technology"
    assert sec_by_sym["JPM.US"] == "financials"
    assert sec_by_sym["BAC.US"] == "financials"
    assert sec_by_sym["XOM.US"] == "energy"
    manifest = json.loads((wh_root / "universe" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sector_counts"] == {"technology": 2, "financials": 2, "energy": 1}


# ============================================================================
# 2. Private Compliance Guard Unit Tests (Fail-Closed)
# ============================================================================


def _valid_private_payload() -> dict:
    return {
        "ok": True,
        "private_spec": {
            "card_id": "card-abc123",
            "trial_id": "trial-abc123",
            "verdict": "survivor",
            "survivor_class": "on-watch-amber",
            "instruments": [
                {"role": "leg", "symbol": "AAPL"},
                {"role": "leg", "symbol": "MSFT"},
            ],
            "parameters": {
                "family": "pairs",
                "scope_id": "free_5y_top1k",
                "mining_mode": "direct",
                "subspace_filter": "direct",
                "subspace_size": 1,
                "formation": 60,
                "open_z": 2.0,
                "close_z": 0.5,
            },
            "live_signal": {
                "magnitude_z": 2.15,
                "band": "stretched",
                "label": "stretched beyond its usual range",
                "note": "magnitude is |z| only — no sign, no direction",
            },
            "guidance": "Check the pair in your own charting or broker tool.",
            "watermark": {
                "handle": "u-1234",
                "ts": 1700000000.0,
                "non_shareable": True,
                "note": "private owner view — do not share",
            },
            "disclaimer": (
                "Private owner view — do not share. SIFT does not trade; "
                "survival facts and magnitude only, never advice."
            ),
        },
    }


def test_guard_owner_private_payload_accepts_valid_and_rejects_every_red_line():
    extra = frozenset({"AAPL", "MSFT", "JPM", "BAC", "XOM", "BADQ"})
    ok_payload = _valid_private_payload()
    guard_owner_private_payload(ok_payload, allowed_symbols={"AAPL", "MSFT"}, extra_symbols=extra)

    # Confirm public guard_api_payload STILL rejects this private payload fail-closed!
    with pytest.raises(ComplianceError):
        guard_api_payload(ok_payload, extra_symbols=extra)

    # Forbidden keys: raw prices, vendor slopes/betas, sign/direction, P&L, params_json
    for bad_key in (
        "prices", "ohlcv", "adjusted_close", "params_json",
        "slope", "beta", "beta_hat", "beta_levels", "hedge_ratio",
        "z_signed", "signed_z", "sign", "direction", "pnl", "returns",
    ):
        assert bad_key in PRIVATE_FORBIDDEN_KEYS
        mut = _valid_private_payload()
        mut["private_spec"][bad_key] = 1.23
        with pytest.raises(ComplianceError):
            guard_owner_private_payload(mut, allowed_symbols={"AAPL", "MSFT"}, extra_symbols=extra)

    # Negative (signed) magnitude_z refused
    neg_z = _valid_private_payload()
    neg_z["private_spec"]["live_signal"]["magnitude_z"] = -2.15
    with pytest.raises(ComplianceError):
        guard_owner_private_payload(neg_z, allowed_symbols={"AAPL", "MSFT"}, extra_symbols=extra)

    # Unauthorized symbol inside instruments refused
    unauth = _valid_private_payload()
    unauth["private_spec"]["instruments"][0]["symbol"] = "JPM"
    with pytest.raises(ComplianceError):
        guard_owner_private_payload(unauth, allowed_symbols={"AAPL", "MSFT"}, extra_symbols=extra)

    # Ticker leaked outside instruments slot refused
    leak = _valid_private_payload()
    leak["private_spec"]["guidance"] = "Check AAPL in your charting tool."
    with pytest.raises(ComplianceError):
        guard_owner_private_payload(leak, allowed_symbols={"AAPL", "MSFT"}, extra_symbols=extra)

    # Email PII refused
    pii = _valid_private_payload()
    pii["private_spec"]["guidance"] = "Send alerts to quant@fund.io"
    with pytest.raises(ComplianceError):
        guard_owner_private_payload(pii, allowed_symbols={"AAPL", "MSFT"}, extra_symbols=extra)

    # Banned phrase / directional word refused
    dir_call = _valid_private_payload()
    dir_call["private_spec"]["guidance"] = "Buy the cheaper stock now."
    with pytest.raises(ComplianceError):
        guard_owner_private_payload(dir_call, allowed_symbols={"AAPL", "MSFT"}, extra_symbols=extra)

    # Missing non_shareable watermark refused
    no_wm = _valid_private_payload()
    no_wm["private_spec"]["watermark"]["non_shareable"] = False
    with pytest.raises(ComplianceError):
        guard_owner_private_payload(no_wm, allowed_symbols={"AAPL", "MSFT"}, extra_symbols=extra)

    # Oversized instruments list refused
    big = _valid_private_payload()
    big["private_spec"]["instruments"] = [{"role": "leg", "symbol": "AAPL"}] * 11
    with pytest.raises(ComplianceError):
        guard_owner_private_payload(big, allowed_symbols={"AAPL", "MSFT"}, extra_symbols=extra)

    # Directional role name refused
    bad_role = _valid_private_payload()
    bad_role["private_spec"]["instruments"][0]["role"] = "short"
    with pytest.raises(ComplianceError):
        guard_owner_private_payload(bad_role, allowed_symbols={"AAPL", "MSFT"}, extra_symbols=extra)


# ============================================================================
# 3. End-to-End Unified Slot Mining + Private Owner Spec + Isolation
# ============================================================================


def test_unified_slot_modes_private_spec_eligibility_isolation_and_kill_switch(tmp_path: Path):
    wh_root = tmp_path / "warehouse"
    _seed_sectored_pit_warehouse(wh_root)

    app = create_app(
        data_dir=tmp_path / "data",
        warehouse_dir=wh_root,
        card_secret=b"private-spec-test-secret",
        allow_synthetic_examples=False,
        s2_B=200,
    )
    with TestClient(app) as tc:
        # Unauthenticated private spec requests fail 401 (before any session cookie is set)
        assert tc.get("/api/private/cards/card-x/spec").status_code == 401
        assert tc.get("/api/private/trials/trial-x/spec").status_code == 401

        alice = _signup(tc, "alice@quant.io")
        bob = _signup(tc, "bob@quant.io")
        h_alice = {"Authorization": f"Bearer {alice['session_token']}"}
        h_bob = {"Authorization": f"Bearer {bob['session_token']}"}

        # Catalog has ZERO hidden pre-screened pair-001 shortlist, exposes populated subspaces only
        r_cat = tc.get("/api/mining/catalog", headers=h_alice)
        assert r_cat.status_code == 200
        cat = r_cat.json()
        assert cat["warehouse_ready"] is True
        assert cat["candidates"] == []  # zero pre-screened shortlist (mining-modes-policy.md §2)
        sub_map = {s["id"]: s["eligible_count"] for s in cat["subspaces"]}
        assert sub_map == {"all": 5, "technology": 2, "financials": 2, "energy": 1}
        for tk in ("AAPL", "MSFT", "JPM", "BAC", "XOM", "BADQ"):
            assert tk not in r_cat.text

        # Validation failures: quarantined BADQ, unknown ZZZZ, self-pair, forbidden prices/skip_s5
        for bad_body in (
            {"slot_a": {"kind": "symbol", "value": "BADQ"}, "slot_b": {"kind": "symbol", "value": "AAPL"}},
            {"slot_a": {"kind": "symbol", "value": "ZZZZ"}, "slot_b": {"kind": "symbol", "value": "AAPL"}},
            {"slot_a": {"kind": "symbol", "value": "AAPL"}, "slot_b": {"kind": "symbol", "value": "AAPL"}},
            {"slot_a": {"kind": "symbol", "value": "AAPL"}, "slot_b": {"kind": "symbol", "value": "MSFT"}, "skip_s5": True},
            {"slot_a": {"kind": "symbol", "value": "AAPL"}, "slot_b": {"kind": "symbol", "value": "MSFT"}, "prices": [1, 2]},
        ):
            r_bad = tc.post("/api/mining/runs", headers=h_alice, json=bad_body)
            assert r_bad.status_code == 400
            for tk in ("BADQ", "ZZZZ", "AAPL", "MSFT"):
                assert tk not in r_bad.text

        # ---- Mode 1: Direct Hypothesis (surviving cointegrated pair AAPL <-> MSFT) ----
        r_dir = tc.post(
            "/api/mining/runs",
            headers=h_alice,
            json={
                "slot_a": {"kind": "symbol", "value": "MSFT"},
                "slot_b": {"kind": "symbol", "value": "AAPL"},
                "scope_id": "free_5y_top1k",
            },
        )
        assert r_dir.status_code == 202
        dir_meta = r_dir.json()["run"]
        assert dir_meta["mining_mode"] == "direct"
        assert dir_meta["subspace_size"] == 1
        assert "AAPL" not in r_dir.text and "MSFT" not in r_dir.text
        dir_done = _wait_run(tc, dir_meta["run_id"], h_alice)
        assert dir_done["verdict"] == "survivor"
        assert dir_done["card_id"] is not None
        assert dir_done["private_spec_eligible"] is True
        card_id = dir_done["card_id"]
        trial_id = dir_done["trial_id"]

        # Canonical order dedupe: (AAPL, MSFT) hits the EXACT SAME trial_id as (MSFT, AAPL)
        r_dir2 = tc.post(
            "/api/mining/runs",
            headers=h_alice,
            json={
                "slot_a": {"kind": "symbol", "value": "AAPL"},
                "slot_b": {"kind": "symbol", "value": "MSFT"},
                "scope_id": "free_5y_top1k",
            },
        )
        assert r_dir2.status_code == 202
        assert r_dir2.json()["run"]["trial_id"] == trial_id
        assert r_dir2.json()["run"]["cached"] is True

        # Public card face + PNG remain 100% ticker-free!
        r_face = tc.get(f"/api/cards/{card_id}/face?with_image=true", headers=h_alice)
        assert r_face.status_code == 200
        face_body = r_face.json()
        guard_api_payload(face_body, extra_symbols={"AAPL", "MSFT", "JPM", "BAC", "XOM", "BADQ"})
        face_no_img = {k: v for k, v in face_body["face"].items() if k != "image"}
        face_txt = json.dumps(face_no_img)
        assert "AAPL" not in face_txt and "MSFT" not in face_txt

        # Private Owner Spec for card -> 200 OK with strict headers, exact symbols, zero slopes/prices/direction
        r_ps = tc.get(f"/api/private/cards/{card_id}/spec", headers=h_alice)
        assert r_ps.status_code == 200
        assert "no-store" in r_ps.headers.get("cache-control", "")
        assert "private" in r_ps.headers.get("cache-control", "")
        assert "noindex" in r_ps.headers.get("x-robots-tag", "")
        ps = r_ps.json()["private_spec"]
        assert [it["symbol"] for it in ps["instruments"]] == ["AAPL", "MSFT"]
        assert ps["parameters"]["mining_mode"] == "direct"
        assert ps["live_signal"]["magnitude_z"] is not None
        assert ps["live_signal"]["magnitude_z"] >= 0.0
        assert ps["watermark"]["non_shareable"] is True
        assert "does not trade" in ps["disclaimer"].lower()
        for bad_tok in ("beta_hat", "beta_levels", "hedge_ratio", "params_json", "adjusted_close"):
            assert bad_tok not in r_ps.text

        # Cross-user isolation: Bob CANNOT inspect Alice's private card or trial spec
        r_bob_card = tc.get(f"/api/private/cards/{card_id}/spec", headers=h_bob)
        assert r_bob_card.status_code == 403
        assert "AAPL" not in r_bob_card.text and "MSFT" not in r_bob_card.text
        r_bob_trial = tc.get(f"/api/private/trials/{trial_id}/spec", headers=h_bob)
        assert r_bob_trial.status_code == 404
        assert "AAPL" not in r_bob_trial.text and "MSFT" not in r_bob_trial.text

        # ---- Mode 1 (Killed Direct Hypothesis: JPM <-> BAC) -> STILL eligible for private spec ----
        r_kill_dir = tc.post(
            "/api/mining/runs",
            headers=h_alice,
            json={
                "slot_a": {"kind": "symbol", "value": "JPM"},
                "slot_b": {"kind": "symbol", "value": "BAC"},
                "scope_id": "free_5y_top1k",
            },
        )
        assert r_kill_dir.status_code == 202
        kd_done = _wait_run(tc, r_kill_dir.json()["run"]["run_id"], h_alice)
        assert kd_done["verdict"] == "killed"
        assert kd_done["private_spec_eligible"] is True
        r_kd_spec = tc.get(f"/api/private/trials/{kd_done['trial_id']}/spec", headers=h_alice)
        assert r_kd_spec.status_code == 200
        assert [it["symbol"] for it in r_kd_spec.json()["private_spec"]["instruments"]] == ["BAC", "JPM"]

        # ---- Mode 2: Anchored Search (XOM anchored + Random(financials) -> must draw JPM or BAC) ----
        r_anc = tc.post(
            "/api/mining/runs",
            headers=h_bob,
            json={
                "slot_a": {"kind": "symbol", "value": "XOM"},
                "slot_b": {"kind": "random", "subspace": "financials"},
                "scope_id": "free_5y_top1k",
            },
        )
        assert r_anc.status_code == 202
        anc_meta = r_anc.json()["run"]
        assert anc_meta["mining_mode"] == "anchored"
        assert anc_meta["subspace_filter"] == "financials"
        assert anc_meta["subspace_size"] == 2
        assert anc_meta["anchor_sector"] == "energy"
        for tk in ("XOM", "JPM", "BAC"):
            assert tk not in r_anc.text
        anc_done = _wait_run(tc, anc_meta["run_id"], h_bob)
        # Killed anchored trial MUST NOT reveal the server-drawn unknown leg (Decision 1)!
        if anc_done["verdict"] == "killed":
            assert anc_done["private_spec_eligible"] is False
            r_anc_spec = tc.get(f"/api/private/trials/{anc_done['trial_id']}/spec", headers=h_bob)
            assert r_anc_spec.status_code == 403
            for tk in ("XOM", "JPM", "BAC"):
                assert tk not in r_anc_spec.text

        # ---- Mode 3: Uniform Random (Random(technology) x Random(technology) -> AAPL/MSFT survivor) ----
        r_rnd = tc.post(
            "/api/mining/runs",
            headers=h_bob,
            json={
                "slot_a": {"kind": "random", "subspace": "technology"},
                "slot_b": {"kind": "random", "subspace": "technology"},
                "scope_id": "free_5y_top1k",
            },
        )
        assert r_rnd.status_code == 202
        rnd_meta = r_rnd.json()["run"]
        assert rnd_meta["mining_mode"] == "uniform_random"
        assert rnd_meta["subspace_filter"] == "technology"
        assert rnd_meta["subspace_size"] == 1  # C(2, 2) = 1 tech pair
        rnd_done = _wait_run(tc, rnd_meta["run_id"], h_bob)
        assert rnd_done["verdict"] == "survivor"
        assert rnd_done["private_spec_eligible"] is True
        r_rnd_spec = tc.get(f"/api/private/trials/{rnd_done['trial_id']}/spec", headers=h_bob)
        assert r_rnd_spec.status_code == 200
        assert [it["symbol"] for it in r_rnd_spec.json()["private_spec"]["instruments"]] == ["AAPL", "MSFT"]

        # ---- Kill-switch SIFT_OWNER_PRIVATE_SPEC=0 -> 503 fail-closed ----
        flags.set_flag(flags.FLAG_OWNER_PRIVATE_SPEC, False)
        assert tc.get(f"/api/private/cards/{card_id}/spec", headers=h_alice).status_code == 503
        assert tc.get(f"/api/private/trials/{trial_id}/spec", headers=h_alice).status_code == 503
        flags.set_flag(flags.FLAG_OWNER_PRIVATE_SPEC, True)

        # ---- Telemetry & general endpoints remain 100% ticker-free & email-free ----
        rt = app.state.runtime
        exported = json.dumps(rt.call(rt.telemetry.export_events))
        assert "private_spec_viewed" in exported
        for secret in ("AAPL", "MSFT", "JPM", "BAC", "XOM", "BADQ", "alice@quant.io", "bob@quant.io"):
            assert secret not in exported
