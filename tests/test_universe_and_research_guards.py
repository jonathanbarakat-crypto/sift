"""Unit tests for universe quality floors, security-type filtering, and
legacy research-grade fail-closed guards.

These tests were preserved verbatim in coverage when ``data/yf_client.py`` and
``data/ingest_yf.py`` were retired: ``data/universe.py``, ``data/security_type.py``,
``sift/engine/first_run.py``, and ``sift/gauntlet/config.py`` continue to
enforce common-stock filtering, liquidity/min-bar quality floors, and
fail-closed labeling for any existing/legacy research-grade warehouses.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from data import universe
from data import warehouse as wh
from data.security_type import is_common_stock_symbol
from sift.engine import first_run as fr
from sift.gauntlet import config as cfg


def _make_bars_frame(symbol: str, start: str, n: int, base_close: float = 100.0,
                     vol: int = 1_000_000, exchange: str = "US") -> pd.DataFrame:
    days = pd.bdate_range(start, periods=n)
    full_sym = symbol if "." in symbol else f"{symbol}.{exchange}"
    return pd.DataFrame({
        "date": [d.date() for d in days],
        "open": [base_close] * n,
        "high": [base_close + 1.0] * n,
        "low": [base_close - 1.0] * n,
        "close": [base_close] * n,
        "adjusted_close": [base_close * 0.99] * n,
        "volume": [vol] * n,
        "dollar_volume": [base_close * vol] * n,
        "symbol": [full_sym] * n,
        "exchange": [exchange] * n,
    })[wh.EOD_COLUMNS]


# --------------------------------------------------------------------------
# universe: research vintage -> delist_idx unavailable, survivorship flagged
# --------------------------------------------------------------------------
def _seed_research_warehouse(root):
    eod = os.path.join(root, "eod", "US")
    os.makedirs(eod, exist_ok=True)
    for sym, n, close, vol in [("AAPL", 30, 200.0, 5_000_000),
                               ("DEAD", 5, 50.0, 1_000_000)]:
        bars = _make_bars_frame(sym, "2026-10-01", n, base_close=close, vol=vol)
        wh.write_parquet_atomic(bars, os.path.join(eod, f"{sym}.parquet"))
    wh.write_vintage(root, extra={"built_by": "ingest_yf", "vendor": "yfinance",
                                  "research_grade": True, "scope_id": "research_5y_top2",
                                  "period": "5y"})


def test_vintage_is_research_detection(tmp_path):
    root = str(tmp_path / "wh")
    _seed_research_warehouse(root)
    is_research, rec = universe.vintage_is_research(root)
    assert is_research is True and rec["vendor"] == "yfinance"
    # production vintage (no vendor/research_grade) -> not research
    root2 = str(tmp_path / "wh2")
    os.makedirs(os.path.join(root2, "eod", "US"))
    wh.write_vintage(root2, extra={"built_by": "ingest_eodhd"})
    is_research2, _ = universe.vintage_is_research(root2)
    assert is_research2 is False


def test_universe_research_vintage_flags_delist_idx_unavailable(tmp_path):
    root = str(tmp_path / "wh")
    _seed_research_warehouse(root)
    rc = universe.main(["--warehouse", root, "--n", "10", "--asof", "2026-11-15"])
    assert rc == 0
    ranked = wh.read_parquet(os.path.join(root, "universe", "ranked_2026-11-15.parquet"))
    assert "delist_idx_available" in ranked.columns
    assert ranked["delist_idx_available"].eq(False).all()  # never survivorship-free
    manifest = wh.read_json(os.path.join(root, "universe", "manifest.json"))
    assert manifest["research_grade"] is True
    assert manifest["vendor"] == "yfinance"
    assert manifest["delist_idx_available"] is False
    assert "BIASED" in manifest["survivorship_bias"]
    vrec = wh.read_json(os.path.join(root, "vintage.json"))
    assert vrec["research_grade"] is True and vrec["vendor"] == "yfinance"


def test_universe_production_vintage_keeps_delist_idx_available(tmp_path):
    root = str(tmp_path / "wh")
    eod = os.path.join(root, "eod", "US")
    os.makedirs(eod, exist_ok=True)
    bars = _make_bars_frame("AAPL", "2026-10-01", 30)
    wh.write_parquet_atomic(bars, os.path.join(eod, "AAPL.parquet"))
    wh.write_vintage(root, extra={"built_by": "ingest_eodhd"})  # no research flags
    rc = universe.main(["--warehouse", root, "--n", "10", "--asof", "2026-11-15"])
    assert rc == 0
    ranked = wh.read_parquet(os.path.join(root, "universe", "ranked_2026-11-15.parquet"))
    assert ranked["delist_idx_available"].eq(True).all()
    manifest = wh.read_json(os.path.join(root, "universe", "manifest.json"))
    assert manifest["research_grade"] is False
    assert manifest["delist_idx_available"] is True


# --------------------------------------------------------------------------
# engine scope-id mapping: --source yf -> research_* scope; eodhd unchanged
# --------------------------------------------------------------------------
def test_engine_scope_id_mapping():
    assert fr.scope_id_for_source("eodhd", "1y") == "free_1y_top1k"
    assert fr.scope_id_for_source("yf", "5y", top_n=500) == "research_5y_top500"
    assert fr.scope_id_for_source("yf", "5y", top_n=1000) == "research_5y_top1k"
    assert fr.scope_id_for_source("yf", "1y") == "research_1y"
    sid = fr.scope_id_for_source("yf", "5y", top_n=500)
    assert sid.startswith("research_") and sid not in ("free_1y_top1k", "pro_full", "free_5y_top1k")
    assert fr.source_is_research("yf") is True
    assert fr.source_is_research("eodhd") is False
    with pytest.raises(ValueError):
        fr.scope_id_for_source("unknown", "5y")


def test_engine_run_pairs_yf_source_labels_research(tmp_path):
    # minimal synthetic warehouse, run with source='yf' -> research scope id
    root = str(tmp_path / "wh")
    rng_symbols = ["AAA.US", "BBB.US"]
    rng = np.random.default_rng(3)
    eod = os.path.join(root, "eod", "US")
    os.makedirs(eod, exist_ok=True)
    for s in rng_symbols:
        px = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, 120)))
        rows = []
        for t in range(120):
            d = (pd.Timestamp("2026-06-01") + pd.Timedelta(days=t)).date()
            rows.append({"date": d, "open": px[t], "high": px[t], "low": px[t],
                         "close": px[t], "adjusted_close": px[t], "volume": 1e6,
                         "dollar_volume": px[t] * 1e6, "symbol": s, "exchange": "US"})
        wh.write_parquet_atomic(pd.DataFrame(rows), wh.eod_parquet_path(root, "US", s.replace(".US", "")))
    uroot = os.path.join(root, "universe")
    os.makedirs(uroot, exist_ok=True)
    uni = pd.DataFrame({"rank": [1, 2], "symbol": rng_symbols, "exchange": ["US"] * 2,
                        "avg_dollar_volume": [1e8, 1e8], "bars": [120, 120],
                        "first_date": ["2026-06-01"] * 2, "last_date": ["2026-10-28"] * 2,
                        "asof": "2026-11-15", "ended_before_asof": [False] * 2,
                        "delisted": [False] * 2, "delist_idx_available": [False] * 2,
                        "in_top_n": [True] * 2})
    uni_path = os.path.join(uroot, "top2_pit_test.parquet")
    wh.write_parquet_atomic(uni, uni_path)
    wh.write_vintage(root, extra={"built_by": "ingest_yf", "vendor": "yfinance",
                                  "research_grade": True, "scope_id": "research_5y_top2"})
    prices, kept, info = fr.load_price_matrix(root, rng_symbols, "US")
    assert prices.shape == (120, 2) and kept == rng_symbols and info["loaded"] == 2
    prices2, kept2, _ = fr.load_price_matrix(root, ["AAA", "BBB"], "US")
    assert prices2.shape == (120, 2)

    meas = fr.run_pairs(root, uni_path, str(tmp_path / "l.db"), max_pairs=1, formation=30,
                        s2_B=20, measurements_path=str(tmp_path / "m.json"),
                        max_universe=2, source="yf", window="5y")
    assert meas["scope"] == "research_5y_top2"
    assert meas["research_grade"] is True
    assert meas["source"] == "yf"
    m = json.load(open(str(tmp_path / "m.json")))
    assert m["scope"] == "research_5y_top2" and m["scope"] != "free_1y_top1k"
    assert "RESEARCH-GRADE" in m["note"] or "research" in m["note"].lower()


def test_config_research_scope_tau_and_T():
    assert cfg.tau_cal("research_5y_top1k") == 3.5
    assert cfg.tau_cal("research_5y_top500") == 3.5      # research_* -> 5y hurdle
    assert cfg.scope_T("research_5y_top1k") == 1260
    assert cfg.scope_T("research_5y_top500") == 1260      # fallback for arbitrary research ids
    # production scopes unchanged
    assert cfg.tau_cal("free_5y_top1k") == 3.5 and cfg.tau_cal("pro_full") == 3.0


# --------------------------------------------------------------------------
# universe-selection integrity: security-type classifier + quality floors
# --------------------------------------------------------------------------
def test_security_type_classifier_warrant_unit_right_preferred():
    excluded = ["AAC-WT", "AAC-WS", "FOO-W", "BAR-WU",   # warrants (dash form)
                "XYZ-UN", "ABC-UT", "QQQ-U",              # units
                "Q-RT", "Q-RS", "Q-R",                     # rights
                "PREF-P", "PREF-PF", "PREF-PR",           # preferred
                "AESPW", "APXTW",                          # trailing-W warrants (stem >= 3)
                "AACG.WS"]                                 # SEC dot form
    for s in excluded:
        assert not is_common_stock_symbol(s), s
    kept = ["AAPL", "MSFT", "BRK-B", "BRK.B", "BF-B", "CW", "GE"]
    for s in kept:
        assert is_common_stock_symbol(s), s


def _seed_bars_with_floors(root):
    eod = os.path.join(root, "eod", "US")
    os.makedirs(eod, exist_ok=True)
    specs = [
        ("AAPL", "2026-10-01", 30, 200.0, 5_000_000),   # ADV 1e9, 30 bars
        ("MICRO", "2026-10-01", 30, 2.0, 1_000),        # ADV 2e3
        ("ONEBAR", "2026-10-30", 1, 500.0, 1_000_000),  # 1 bar
        ("AESPW", "2026-10-01", 30, 1.0, 10_000_000),   # warrant, ADV 1e7
    ]
    for sym, start, n, close, vol in specs:
        bars = _make_bars_frame(sym, start, n, base_close=close, vol=vol)
        wh.write_parquet_atomic(bars, os.path.join(eod, f"{sym}.parquet"))


def test_universe_quality_floors_and_warrant_exclusion(tmp_path):
    root = str(tmp_path / "wh")
    _seed_bars_with_floors(root)
    rc = universe.main(["--warehouse", root, "--n", "10", "--asof", "2026-11-15"])
    assert rc == 0
    ranked = wh.read_parquet(os.path.join(root, "universe", "ranked_2026-11-15.parquet"))
    # only AAPL survives: MICRO (ADV floor), ONEBAR (min-bars), AESPW (warrant) dropped
    assert set(ranked["symbol"]) == {"AAPL.US"}
    manifest = wh.read_json(os.path.join(root, "universe", "manifest.json"))
    assert manifest["min_bars"] == universe.DEFAULT_MIN_BARS >= 20
    assert manifest["min_avg_dollar_volume"] == universe.DEFAULT_MIN_AVG_DOLLAR_VOLUME
    assert manifest["non_common_excluded"] == 1

    # escape hatch: explicit low floors keep the low-ADV / 1-bar names (warrants still out)
    rc = universe.main(["--warehouse", root, "--n", "10", "--asof", "2026-11-15",
                        "--min-bars", "1", "--min-avg-dollar-volume", "0"])
    assert rc == 0
    ranked2 = wh.read_parquet(os.path.join(root, "universe", "ranked_2026-11-15.parquet"))
    assert set(ranked2["symbol"]) == {"AAPL.US", "MICRO.US", "ONEBAR.US"}
