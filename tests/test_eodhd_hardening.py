"""Unit tests for PR #2 EODHD production-readiness hardening (P0-P2).

Covers:
  * P0.1: US OTC exchange tier filtering (Choice 1:A: drops PINK/OTCGREY/OTCCE/OTCBB,
          retains OTCMKTS/OTCQX/OTCQB; --include-otc opt-in)
  * P0.2: Internal dormancy / ticker-reuse gap screen (> 90 calendar days)
  * P0.3: Duplicate / copied pre-spinoff history clusters & PairCatalog/first_run skip
  * P0.4: Expanded US foreign-calendar holidays (1-year UK bank holidays) &
          single-bar causal volume spike screen (with IPO grace period)
  * P0.5: Multi-exchange sub-unit scaling (LSE GBX pence = 0.01 GBP, non-US calendar)
  * P1.1: Trailing-window PIT ADV ranking (--adv-window-bars)
  * P1.2: Independent cross-source spot audit (scripts/audit_warehouse_crosscheck.py)
  * P2.1: EODHD Bulk API nightly incremental update (--bulk-update, 100-call billing)
  * P2.2: Automated post-ingest validation gate (scripts/validate_warehouse.py)
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data import dollar_volume as dv
from data import ingest_eodhd, universe
from data import warehouse as wh
from data.security_type import is_common_stock_metadata
from scripts import audit_warehouse_crosscheck as crosscheck
from scripts import validate_warehouse as val_wh
from sift.web.mining import PairCatalog
from tests.test_dollar_volume_basis import _bars_from_triples, _write_eod, _write_splits
from tests.test_ingest_eodhd import FakeResponse, FakeSession


# --------------------------------------------------------------------------
# P0.1: OTC tier filtering (Choice 1:A)
# --------------------------------------------------------------------------
def test_noisy_otc_tiers_excluded_while_otcmkts_retained(tmp_path):
    assert not is_common_stock_metadata("HLAN", name="Heartland Banccorp", exchange="PINK")
    assert not is_common_stock_metadata("TLNE", name="Talen Energy", exchange="OTCGREY")
    assert not is_common_stock_metadata("FOO", name="Foo Corp", exchange="OTCBB")
    assert not is_common_stock_metadata("BAR", name="Bar Corp", exchange="OTCCE")
    assert is_common_stock_metadata("WYE", name="Wyeth", exchange="OTCMKTS")
    assert is_common_stock_metadata("FNMA", name="Fannie Mae", exchange="OTCQB")
    assert is_common_stock_metadata("SCTY", name="SolarCity", exchange="OTCQX")
    # --include-otc opt-in admits PINK while still rejecting unmapped Exchange=="US"
    assert is_common_stock_metadata("HLAN", name="Heartland Banccorp", exchange="PINK", include_otc=True)
    assert not is_common_stock_metadata("KLS", name="Kelsian Group", exchange="US", include_otc=True)

    root = str(tmp_path / "wh")
    for sym in ("AAPL", "WYE", "HLAN"):
        _write_eod(root, sym, _bars_from_triples([(100.0, 100.0, 1_000_000.0)] * 25, start="2026-09-01"))
        _write_splits(root, sym, [])
    wh.write_json_atomic([
        {"Code": "AAPL", "Exchange": "NASDAQ", "Name": "Apple Inc.", "Type": "Common Stock"},
        {"Code": "WYE", "Exchange": "OTCMKTS", "Name": "Wyeth", "Type": "Common Stock"},
        {"Code": "HLAN", "Exchange": "PINK", "Name": "Heartland Banccorp", "Type": "Common Stock"},
    ], os.path.join(root, "meta", "symbol-list", "US-active.json"))

    assert universe.main(["--warehouse", root, "--n", "10", "--asof", "2026-11-25"]) == 0
    ranked = wh.read_parquet(os.path.join(root, "universe", "ranked_2026-11-25.parquet"))
    assert set(ranked["symbol"]) == {"AAPL.US", "WYE.US"}
    manifest = wh.read_json(os.path.join(root, "universe", "manifest.json"))
    assert manifest["exchange_filter"]["include_otc"] is False
    assert "PINK" in manifest["exchange_filter"]["excluded_otc_exchanges"]


# --------------------------------------------------------------------------
# P0.2: Internal dormancy / ticker-reuse gap screen (> 90 calendar days)
# --------------------------------------------------------------------------
def test_internal_trading_gap_catches_recycled_ticker_but_allows_short_closure(tmp_path):
    root = str(tmp_path / "wh")
    # Recycled ticker with 150-day internal gap
    recycled = (
        _bars_from_triples([(50.0, 50.0, 1_000_000.0)] * 20, start="2023-01-03")
        + _bars_from_triples([(50.0, 50.0, 1_000_000.0)] * 20, start="2023-08-01")
    )
    # Normal stock with a 6-day closure (9/11 scale)
    normal = (
        _bars_from_triples([(50.0, 50.0, 1_000_000.0)] * 20, start="2023-01-03")
        + _bars_from_triples([(50.0, 50.0, 1_000_000.0)] * 20, start="2023-02-06")
    )
    _write_eod(root, "REUSED", recycled)
    _write_splits(root, "REUSED", [])
    _write_eod(root, "NORMAL", normal)
    _write_splits(root, "NORMAL", [])

    ranked, quarantine = universe.build_ranked_table(
        universe.load_all_bars(root), "2023-12-31",
        splits_by_symbol=universe.load_all_splits(root),
    )
    assert list(ranked["symbol"]) == ["NORMAL.US"]
    assert list(quarantine["symbol"]) == ["REUSED.US"]
    assert "internal_trading_gap" in str(quarantine.loc[0, "reason_codes"])


# --------------------------------------------------------------------------
# P0.3: Duplicate / copied pre-spinoff history clusters + PairCatalog skip
# --------------------------------------------------------------------------
def test_duplicate_history_clusters_detected_and_blocked_in_pair_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("SIFT_ENV", "test")
    root = tmp_path / "wh"
    rng = np.random.default_rng(42)
    days = pd.bdate_range("2022-01-03", periods=320)
    asof = days[-1].strftime("%Y-%m-%d")

    r_shared = rng.normal(0.0, 0.012, len(days))
    p_parent = 100.0 * np.exp(np.cumsum(r_shared))
    # Child copies parent's first 200 bars verbatim, then diverges
    r_child = r_shared.copy()
    r_child[200:] = rng.normal(0.0, 0.012, len(days) - 200)
    p_child = 100.0 * np.exp(np.cumsum(r_child))
    # Third clean stock correlated with parent (not identical!)
    p_other = 80.0 * np.exp(np.cumsum(0.7 * r_shared + rng.normal(0.0, 0.006, len(days))))

    def _rows(px):
        return [
            {"date": d.date().isoformat(), "open": float(c), "high": float(c),
             "low": float(c), "close": float(c), "adjusted_close": float(c), "volume": 500_000}
            for d, c in zip(days, px)
        ]

    _write_eod(str(root), "PARENT", _rows(p_parent))
    _write_splits(str(root), "PARENT", [])
    _write_eod(str(root), "CHILD", _rows(p_child))
    _write_splits(str(root), "CHILD", [])
    _write_eod(str(root), "OTHER", _rows(p_other))
    _write_splits(str(root), "OTHER", [])

    assert universe.main(["--warehouse", str(root), "--n", "10", "--asof", asof]) == 0
    manifest = wh.read_json(str(root / "universe" / "manifest.json"))
    dup_meta = manifest["duplicate_history_clusters"]
    assert dup_meta["clusters_count"] == 1
    assert dup_meta["path"] and os.path.exists(dup_meta["path"])
    dup_doc = wh.read_json(dup_meta["path"])
    assert dup_doc["clusters"] == [["CHILD.US", "PARENT.US"]]

    # PairCatalog must retain all 3 symbols in panel but NEVER pair (PARENT, CHILD)
    cat = PairCatalog(warehouse_dir=root, allow_synthetic_examples=False, max_warehouse_pairs=10)
    assert set(cat._wh_symbols) == {"PARENT", "CHILD", "OTHER"}
    pair_sets = [{cat._wh_symbols[i], cat._wh_symbols[j]} for i, j in cat._wh_pairs.values()]
    assert {"PARENT", "CHILD"} not in pair_sets
    assert {"PARENT", "OTHER"} in pair_sets


# --------------------------------------------------------------------------
# P0.4: 1-year UK foreign-calendar leak & single-bar causal volume spike
# --------------------------------------------------------------------------
def test_one_year_uk_calendar_leak_and_single_bar_volume_spike(tmp_path):
    root = str(tmp_path / "wh")
    # 1 full year (2024): drop all 4 UK non-US holidays (Easter Mon, 1st Mon May,
    # last Mon Aug, Dec 26) -> caught even with only 1 year of history
    days_2024 = pd.bdate_range("2024-01-02", "2024-12-31")
    uk_hols = {
        dv._easter_sunday(2024) + timedelta(days=1),
        dv._first_monday(2024, 5),
        dv._last_monday(2024, 8),
        date(2024, 12, 26),
    }
    uk_rows = [
        {"date": d.date().isoformat(), "open": 100.0, "high": 100.0, "low": 100.0,
         "close": 100.0, "adjusted_close": 100.0, "volume": 1_000_000}
        for d in days_2024 if d.date() not in uk_hols
    ]
    # Single-bar fat-finger print glitch on bar #70 ($30B vs $50M trailing median = 600x)
    glitch_triples = [(50.0, 50.0, 1_000_000.0)] * 80
    glitch_triples[70] = (50.0, 50.0, 600_000_000.0)
    # Legitimate S&P 500 inclusion day ($20B vs $100M median = 200x, BRK-B scale) must PASS
    sp500_triples = [(100.0, 100.0, 1_000_000.0)] * 80
    sp500_triples[70] = (100.0, 100.0, 200_000_000.0)
    # Legitimate IPO surge on bar #1 ($30B on day 2 -> inside 5-bar IPO grace window)
    ipo_triples = [(50.0, 50.0, 600_000_000.0)] * 2 + [(50.0, 50.0, 1_000_000.0)] * 78

    _write_eod(root, "UKONE", uk_rows)
    _write_splits(root, "UKONE", [])
    _write_eod(root, "GLITCH", _bars_from_triples(glitch_triples, start="2024-01-02"))
    _write_splits(root, "GLITCH", [])
    _write_eod(root, "SPINC", _bars_from_triples(sp500_triples, start="2024-01-02"))
    _write_splits(root, "SPINC", [])
    _write_eod(root, "IPOSURGE", _bars_from_triples(ipo_triples, start="2024-01-02"))
    _write_splits(root, "IPOSURGE", [])

    ranked, quarantine = universe.build_ranked_table(
        universe.load_all_bars(root), "2025-01-15",
        splits_by_symbol=universe.load_all_splits(root),
    )
    assert set(ranked["symbol"]) == {"IPOSURGE.US", "SPINC.US"}
    by_sym = dict(zip(quarantine["symbol"], quarantine["reason_codes"]))
    assert "foreign_calendar_mismatch" in by_sym["UKONE.US"]
    assert "single_bar_volume_spike" in by_sym["GLITCH.US"]


# --------------------------------------------------------------------------
# P0.5: Multi-exchange sub-unit scaling (LSE GBX pence = 0.01 GBP)
# --------------------------------------------------------------------------
def test_lse_subunit_scaling_and_non_us_calendar(tmp_path):
    assert wh.price_subunit_scale("LSE") == pytest.approx(0.01)
    assert wh.price_subunit_scale("US") == pytest.approx(1.0)
    assert wh.exchange_spec("LSE")["calendar_profile"] == "UK"

    # LSE stock quoted at 700 GBX pence (£7.00), 1M shares -> £7M/day (not £700M/day),
    # and missing UK holidays must NOT trigger US foreign_calendar_mismatch on LSE
    days = pd.bdate_range("2024-01-02", "2024-12-31")
    uk_hols = {
        dv._easter_sunday(2024) + timedelta(days=1),
        dv._first_monday(2024, 5),
        dv._last_monday(2024, 8),
        date(2024, 12, 26),
    }
    rows = [
        {"date": d.date().isoformat(), "open": 700.0, "high": 700.0, "low": 700.0,
         "close": 700.0, "adjusted_close": 700.0, "volume": 1_000_000,
         "dollar_volume": 7_000_000.0, "symbol": "HSBA.LSE", "exchange": "LSE"}
        for d in days if d.date() not in uk_hols
    ]
    df = pd.DataFrame(rows)[wh.EOD_COLUMNS]
    ranked, quarantine = universe.build_ranked_table(
        df, "2025-01-15", splits_by_symbol={"HSBA.LSE": []},
    )
    assert quarantine.empty
    assert list(ranked["symbol"]) == ["HSBA.LSE"]
    assert float(ranked.loc[0, "avg_dollar_volume"]) == pytest.approx(7_000_000.0)


# --------------------------------------------------------------------------
# P1.1: Trailing-window PIT ADV (--adv-window-bars)
# --------------------------------------------------------------------------
def test_trailing_adv_window_bars_ranks_by_recent_liquidity(tmp_path):
    root = str(tmp_path / "wh")
    # FADING: $100M/day for 30 bars, then $5M/day for last 10 bars (full mean ~$76M)
    fading = (
        _bars_from_triples([(100.0, 100.0, 1_000_000.0)] * 30, start="2025-01-02")
        + _bars_from_triples([(100.0, 100.0, 50_000.0)] * 10, start="2025-02-13")
    )
    # RISING: $5M/day for 30 bars, then $80M/day for last 10 bars (full mean ~$24M)
    rising = (
        _bars_from_triples([(100.0, 100.0, 50_000.0)] * 30, start="2025-01-02")
        + _bars_from_triples([(100.0, 100.0, 800_000.0)] * 10, start="2025-02-13")
    )
    _write_eod(root, "FADING", fading)
    _write_splits(root, "FADING", [])
    _write_eod(root, "RISING", rising)
    _write_splits(root, "RISING", [])

    # Full history: FADING beats RISING
    r_full, _ = universe.build_ranked_table(
        universe.load_all_bars(root), "2025-04-01",
        splits_by_symbol=universe.load_all_splits(root),
    )
    assert list(r_full["symbol"]) == ["FADING.US", "RISING.US"]

    # Trailing 10-bar window: RISING ($80M) beats FADING ($5M)
    r_10, _ = universe.build_ranked_table(
        universe.load_all_bars(root), "2025-04-01",
        splits_by_symbol=universe.load_all_splits(root),
        adv_window_bars=10,
    )
    assert list(r_10["symbol"]) == ["RISING.US", "FADING.US"]
    assert float(r_10.loc[0, "avg_dollar_volume"]) == pytest.approx(80_000_000.0)


# --------------------------------------------------------------------------
# P2.1: Bulk API nightly incremental update (--bulk-update)
# --------------------------------------------------------------------------
def test_bulk_update_appends_bars_refreshes_splits_and_charges_100_calls(tmp_path, monkeypatch):
    root = str(tmp_path / "wh")
    _write_eod(root, "AAPL", _bars_from_triples([(400.0, 400.0, 1_000_000.0)] * 5, start="2026-11-16"))
    _write_splits(root, "AAPL", [])
    _write_eod(root, "MSFT", _bars_from_triples([(300.0, 300.0, 1_000_000.0)] * 5, start="2026-11-16"))
    _write_splits(root, "MSFT", [])

    routes = [
        (lambda u, p: "/eod-bulk-last-day/US" in u and p.get("type") == "splits",
         FakeResponse(200, [{"code": "AAPL", "date": "2026-11-23", "split": "4.000000/1.000000"}])),
        (lambda u, p: "/splits/AAPL.US" in u,
         FakeResponse(200, [{"date": "2026-11-23", "split": "4.000000/1.000000"}])),
        (lambda u, p: "/eod-bulk-last-day/US" in u and "type" not in p,
         FakeResponse(200, [
             {"code": "AAPL", "date": "2026-11-23", "open": 100.0, "high": 101.0, "low": 99.0,
              "close": 100.0, "adjusted_close": 100.0, "volume": 4_000_000},
             {"code": "MSFT", "date": "2026-11-23", "open": 305.0, "high": 306.0, "low": 304.0,
              "close": 305.0, "adjusted_close": 305.0, "volume": 1_000_000},
             {"code": "UNKNOWN", "date": "2026-11-23", "open": 10.0, "high": 10.0, "low": 10.0,
              "close": 10.0, "adjusted_close": 10.0, "volume": 100},
         ])),
    ]
    sess = FakeSession(routes=routes)
    monkeypatch.setattr(ingest_eodhd, "_SESSION", sess)
    monkeypatch.setenv("SIFT_TEST_TOKEN", "test-token")

    rc = ingest_eodhd.main([
        "--warehouse", root, "--plan", "paid", "--bulk-update", "--bulk-date", "2026-11-23",
    ])
    assert rc == 0
    aapl_df = wh.read_parquet(wh.eod_parquet_path(root, "US", "AAPL"))
    assert len(aapl_df) == 6
    assert str(aapl_df["date"].max()) == "2026-11-23"
    # Splits refreshed for AAPL -> 2026-11-23 bar has $400M DV
    assert float(aapl_df["dollar_volume"].iloc[-1]) == pytest.approx(400_000_000.0)
    assert wh.read_json(wh.splits_json_path(root, "US", "AAPL")) == [
        {"date": "2026-11-23", "split": "4.000000/1.000000"}
    ]
    # Cost accounting: 100 (bulk splits) + 1 (AAPL splits) + 100 (bulk EOD) = 201 calls
    calls = wh.read_json(os.path.join(root, "state", "calls.json"))
    assert calls["calls_today"] == 201


# --------------------------------------------------------------------------
# P1.2 & P2.2: Cross-source audit & post-ingest validation gate scripts
# --------------------------------------------------------------------------
def test_validate_warehouse_and_crosscheck_scripts(tmp_path):
    root = tmp_path / "wh"
    days = pd.bdate_range("2025-01-02", periods=60)
    asof = days[-1].strftime("%Y-%m-%d")
    rng = np.random.default_rng(7)
    ref_rows = []
    for sym, base in (("AAPL", 200.0), ("MSFT", 350.0)):
        px = base * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(days))))
        rows = [
            {"date": d.date().isoformat(), "open": float(c), "high": float(c),
             "low": float(c), "close": float(c), "adjusted_close": float(c), "volume": 1_000_000}
            for d, c in zip(days, px)
        ]
        _write_eod(str(root), sym, rows)
        _write_splits(str(root), sym, [])
        for d, c in zip(days, px):
            ref_rows.append({"ticker": sym, "date": d.date().isoformat(), "closeadj": float(c)})

    wh.write_vintage(str(root), extra={"built_by": "ingest_eodhd", "plan": "paid", "vendor": "eodhd"})
    assert universe.main(["--warehouse", str(root), "--n", "10", "--asof", asof]) == 0

    # Validation gate passes on clean warehouse
    rep = val_wh.validate_warehouse(str(root), min_selected=2, bellwethers=("AAPL", "MSFT"))
    assert rep["ok"] is True

    # Missing bellwether fails closed
    rep_bad = val_wh.validate_warehouse(str(root), min_selected=2, bellwethers=("AAPL", "NVDA"))
    assert rep_bad["ok"] is False

    # Cross-source audit passes on matching reference
    ref_df = pd.DataFrame(ref_rows)
    ref_df.columns = ["symbol", "date", "ref_adj"]
    ref_df["date"] = pd.to_datetime(ref_df["date"]).dt.date
    audit_ok = crosscheck.audit_against_reference(str(root), ref_df, n_active=10, n_delisted=10)
    assert audit_ok["ok"] is True
    assert audit_ok["within_tol_frac"] == pytest.approx(1.0)

    # Inject 10% single-day divergence -> audit fails closed and flags outlier
    ref_df.loc[10, "ref_adj"] *= 1.12
    audit_fail = crosscheck.audit_against_reference(str(root), ref_df, n_active=10, n_delisted=10)
    assert audit_fail["ok"] is False
    assert len(audit_fail["gross_outliers"]) == 1
