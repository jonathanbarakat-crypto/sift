"""Tests for PairCatalog PIT-universe and quarantine-screen enforcement (PR #69 integration)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from data import universe as pit_universe
from data import warehouse as wh
from sift.web.app import create_app
from sift.web.compliance import ComplianceError, guard_api_payload
from sift.web.mining import PairCatalog


@pytest.fixture(autouse=True)
def _enable_flags(monkeypatch):
    monkeypatch.setenv("SIFT_CARDS_V4_1", "1")
    monkeypatch.setenv("SIFT_DECKS_V4_2", "1")
    monkeypatch.setenv("SIFT_ENV", "test")


def _write_eod_symbol(
    wh_root: Path,
    symbol: str,
    dates: pd.DatetimeIndex,
    close: np.ndarray,
    *,
    volume: float = 500_000.0,
    adjusted_close: np.ndarray | None = None,
    splits: list[dict] | None = None,
) -> None:
    adj = close if adjusted_close is None else adjusted_close
    vol = np.full(len(dates), float(volume), dtype=float)
    df = pd.DataFrame(
        {
            "date": [d.strftime("%Y-%m-%d") for d in dates],
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adjusted_close": adj,
            "volume": vol,
            "dollar_volume": close * vol,
            "symbol": f"{symbol}.US",
            "exchange": "US",
        }
    )
    wh.write_parquet_atomic(df, wh.eod_parquet_path(str(wh_root), "US", symbol))
    if splits is not None:
        wh.write_json_atomic(splits, wh.splits_json_path(str(wh_root), "US", symbol))


def _seed_pit_warehouse(wh_root: Path, *, n_bars: int = 320, extra_post_asof: int = 10) -> str:
    """Seed a warehouse with clean, delisted, quarantined, and low-ADV symbols
    and run `data.universe.main` to produce real PIT + quarantine artifacts."""
    rng = np.random.Generator(np.random.PCG64(99))
    all_dates = pd.bdate_range("2022-01-03", periods=n_bars + extra_post_asof)
    asof = all_dates[n_bars - 1].strftime("%Y-%m-%d")

    f1 = np.cumsum(rng.normal(0.0, 0.01, len(all_dates)))
    aapl = 150.0 * np.exp(f1)
    msft = 200.0 * np.exp(f1 + rng.normal(0.0, 0.005, len(all_dates)))
    jpm = 120.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(all_dates))))

    # Clean common stocks passing quality floors + empty split files (verified)
    _write_eod_symbol(wh_root, "AAPL", all_dates, aapl, volume=500_000.0, splits=[])
    _write_eod_symbol(wh_root, "MSFT", all_dates, msft, volume=500_000.0, splits=[])
    _write_eod_symbol(wh_root, "JPM", all_dates, jpm, volume=500_000.0, splits=[])

    # Delisted common stock ending before asof (kept in PIT listing, trailing NaN preserved)
    delist_len = n_bars - 40
    dlist_p = 80.0 * np.exp(f1[:delist_len] + rng.normal(0.0, 0.008, delist_len))
    _write_eod_symbol(wh_root, "DLIST", all_dates[:delist_len], dlist_p, volume=400_000.0, splits=[])

    # Corrupt stock with EODHD sentinel 999999.9999 -> quarantined by data.universe
    badq = np.full(len(all_dates), 100.0)
    badq[50] = 999999.9999
    _write_eod_symbol(wh_root, "BADQ", all_dates, badq, volume=500_000.0, splits=[])

    # Low-volume micro-cap below $1M ADV floor -> excluded from top-N PIT listing
    lowv = np.full(len(all_dates), 10.0)
    _write_eod_symbol(wh_root, "LOWV", all_dates, lowv, volume=5_000.0, splits=[])

    rc = pit_universe.main(
        ["--warehouse", str(wh_root), "--n", "10", "--asof", asof]
    )
    assert rc == 0
    return asof


def test_catalog_honors_pit_manifest_excludes_quarantined_and_truncates_asof(tmp_path: Path):
    wh_root = tmp_path / "warehouse"
    asof = _seed_pit_warehouse(wh_root, n_bars=320, extra_post_asof=12)

    cat = PairCatalog(warehouse_dir=wh_root, allow_synthetic_examples=False)
    assert cat._wh_prices is not None
    assert cat._wh_delist_idx is not None

    # Only the 4 clean PIT symbols (including delisted DLIST) enter the catalog/panel
    assert set(cat._wh_symbols) == {"AAPL", "MSFT", "JPM", "DLIST"}
    assert "BADQ" not in cat._wh_symbols
    assert "LOWV" not in cat._wh_symbols

    # Delisted symbol retained with delist_idx < T (no survivorship bias)
    d_idx = cat._wh_symbols.index("DLIST")
    assert int(cat._wh_delist_idx[d_idx]) < cat._wh_prices.shape[0]

    # PIT truncation: the 12 post-asof bars were NOT loaded into the panel
    assert cat._wh_prices.shape[0] == 320

    # Quarantined and non-top-N symbols remain blocked by compliance guard
    assert {"BADQ", "LOWV", "AAPL", "DLIST"}.issubset(cat.forbidden_symbols)
    with pytest.raises(ComplianceError):
        guard_api_payload({"note": "check BADQ"}, extra_symbols=cat.forbidden_symbols)

    # Aggregate universe metadata carries zero ticker symbols
    summary = cat.universe_summary()
    assert summary["mode"] == "pit_manifest"
    assert summary["asof"] == asof
    assert summary["quarantined_excluded"] == 1
    assert summary["pit_listed_count"] == 4
    guard_api_payload({"universe": summary}, extra_symbols=cat.forbidden_symbols)


def test_catalog_subtracts_quarantined_even_if_present_in_top_parquet(tmp_path: Path):
    """Belt-and-suspenders: even if a quarantined ticker were injected into
    topN parquet, PairCatalog still excludes it."""
    wh_root = tmp_path / "warehouse"
    asof = _seed_pit_warehouse(wh_root, n_bars=320, extra_post_asof=0)

    top_path = wh_root / "universe" / f"top10_pit_{asof}.parquet"
    top_df = pd.read_parquet(top_path)
    injected = top_df.iloc[[0]].copy()
    injected["symbol"] = "BADQ.US"
    injected["rank"] = len(top_df) + 1
    wh.write_parquet_atomic(pd.concat([top_df, injected], ignore_index=True), str(top_path))

    cat = PairCatalog(warehouse_dir=wh_root, allow_synthetic_examples=False)
    assert "BADQ" not in cat._wh_symbols
    assert set(cat._wh_symbols) == {"AAPL", "MSFT", "JPM", "DLIST"}


def test_catalog_fails_closed_on_missing_quarantine_or_traversal_manifest(tmp_path: Path):
    # 1. Missing referenced quarantine file -> fail closed (never fall back to raw eod/)
    wh_root = tmp_path / "wh_missing_q"
    asof = _seed_pit_warehouse(wh_root, n_bars=320, extra_post_asof=0)
    (wh_root / "universe" / f"quarantined_{asof}.parquet").unlink()

    cat_missing = PairCatalog(warehouse_dir=wh_root, allow_synthetic_examples=False)
    assert cat_missing.list_public() == []
    assert cat_missing._wh_symbols == []

    # 2. Path traversal outside <root>/universe/ -> fail closed
    wh_trav = tmp_path / "wh_traversal"
    _seed_pit_warehouse(wh_trav, n_bars=320, extra_post_asof=0)
    manifest_p = wh_trav / "universe" / "manifest.json"
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    outside = tmp_path / "outside.parquet"
    pd.DataFrame({"symbol": ["AAPL.US"]}).to_parquet(outside, index=False)
    manifest["top_path"] = str(outside)
    manifest_p.write_text(json.dumps(manifest), encoding="utf-8")

    cat_trav = PairCatalog(warehouse_dir=wh_trav, allow_synthetic_examples=False)
    assert cat_trav.list_public() == []


def test_catalog_handles_relocated_warehouse_manifest_paths(tmp_path: Path):
    wh_root = tmp_path / "warehouse"
    asof = _seed_pit_warehouse(wh_root, n_bars=320, extra_post_asof=0)

    manifest_p = wh_root / "universe" / "manifest.json"
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    manifest["top_path"] = f"/other/host/warehouse/universe/top10_pit_{asof}.parquet"
    manifest["data_quality_screen"]["quarantined_path"] = (
        f"/other/host/warehouse/universe/quarantined_{asof}.parquet"
    )
    manifest_p.write_text(json.dumps(manifest), encoding="utf-8")

    cat = PairCatalog(warehouse_dir=wh_root, allow_synthetic_examples=False)
    assert set(cat._wh_symbols) == {"AAPL", "MSFT", "JPM", "DLIST"}
    assert "BADQ" not in cat._wh_symbols
    assert cat.universe_summary()["mode"] == "pit_manifest"


def test_production_requires_non_research_pit_manifest_and_exposes_clean_catalog(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("SIFT_ENV", "production")
    monkeypatch.setenv("SIFT_EODHD_BETA_LICENSE_CONFIRMED", "1")
    monkeypatch.delenv("SIFT_ALLOW_SYNTHETIC_EXAMPLES", raising=False)

    # 1. Raw eod/ without universe/manifest.json -> refused in production
    wh_raw = tmp_path / "wh_raw"
    dates = pd.bdate_range("2022-01-03", periods=300)
    _write_eod_symbol(wh_raw, "AAPL", dates, np.full(300, 150.0))
    _write_eod_symbol(wh_raw, "MSFT", dates, np.full(300, 200.0))
    cat_raw = PairCatalog(warehouse_dir=wh_raw)
    assert cat_raw.list_public() == []

    # 2. Research-grade manifest -> refused in production
    wh_res = tmp_path / "wh_research"
    _seed_pit_warehouse(wh_res, n_bars=320, extra_post_asof=0)
    mp = wh_res / "universe" / "manifest.json"
    m = json.loads(mp.read_text(encoding="utf-8"))
    m["research_grade"] = True
    mp.write_text(json.dumps(m), encoding="utf-8")
    cat_res = PairCatalog(warehouse_dir=wh_res)
    assert cat_res.list_public() == []

    # 3. Clean paid PIT manifest -> served via API with zero ticker leakage
    wh_prod = tmp_path / "wh_prod"
    asof = _seed_pit_warehouse(wh_prod, n_bars=320, extra_post_asof=0)
    app = create_app(
        data_dir=tmp_path / "data",
        warehouse_dir=wh_prod,
        card_secret=b"prod-pit-secret",
    )
    with TestClient(app) as tc:
        rt = app.state.runtime
        inv = rt.call(rt.auth.create_invite, wave="wave-1")
        su = tc.post(
            "/api/auth/signup",
            json={
                "email": "pit@quant.io",
                "password": "very-strong-password-99",
                "invite_code": inv["code"],
                "tos_accepted": True,
            },
        )
        assert su.status_code == 201
        headers = {"Authorization": f"Bearer {su.json()['session_token']}"}
        r_cat = tc.get("/api/mining/catalog", headers=headers)
        assert r_cat.status_code == 200
        body = r_cat.json()
        assert body["warehouse_ready"] is True
        assert body["candidates"] == []
        assert body["subspaces"] == [
            {"id": "all", "label": "Random (All eligible)", "eligible_count": 4}
        ]
        assert body["universe"] == {
            "mode": "pit_manifest",
            "asof": asof,
            "quarantined_excluded": 1,
            "pit_listed_count": 4,
        }
        for sym in ("AAPL", "MSFT", "JPM", "DLIST", "BADQ", "LOWV"):
            assert sym not in r_cat.text
