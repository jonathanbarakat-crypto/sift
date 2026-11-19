"""Unit tests for the yfinance (Yahoo) RESEARCH feed — no network.

The fetcher is a fake callable injected into YfClient, so these tests never
touch Yahoo. They cover: the client's retry/backoff + jitter on 429s, the
warehouse parquet round-trip (same schema/layout as the EODHD path), the
research-grade vintage manifest (vendor + research_grade + research_* scope
id), the scope-id mapping in the engine, and the universe's
survivorship/delist_idx flagging of research vintages.
"""
from __future__ import annotations

import json
import os
import random
from datetime import date, timedelta

import pandas as pd
import pytest

from data import warehouse as wh
from data import ingest_yf, universe
from data.yf_client import (
    DEFAULT_PERIOD, RESEARCH_GRADE, VENDOR, YfClient, YfEmpty, YfError, YfRateLimited,
    history_to_bars_frame,
)
from sift.engine import first_run as fr
from sift.gauntlet import config as cfg


# --------------------------------------------------------------------------
# fake yfinance fetcher
# --------------------------------------------------------------------------
def make_yf_frame(start: str, n: int, base_close: float = 100.0, vol: int = 1_000_000) -> pd.DataFrame:
    days = pd.bdate_range(start, periods=n)
    idx = pd.DatetimeIndex(days)  # tz-naive: the date column is what matters
    return pd.DataFrame({
        "Open": [base_close] * n, "High": [base_close + 1] * n,
        "Low": [base_close - 1] * n, "Close": [base_close] * n,
        "Adj Close": [base_close * 0.99] * n, "Volume": [vol] * n,
        "Dividends": [0.0] * n, "Stock Splits": [0.0] * n,
    }, index=idx)


class FakeFetcher:
    """(symbol, period, interval) -> DataFrame. scripted per symbol; can raise."""

    def __init__(self, frames=None, fail_times=None):
        self.frames = frames or {}
        self.fail_times = fail_times or {}   # symbol -> list of exceptions to raise first
        self.calls = []                       # (symbol, period, interval)

    def __call__(self, symbol, period, interval):
        self.calls.append((symbol, period, interval))
        fails = self.fail_times.get(symbol)
        if fails:
            raise fails.pop(0)
        df = self.frames.get(symbol)
        if df is None:
            return pd.DataFrame()  # unknown symbol -> empty
        return df.copy()


def make_client(fetcher, **kw):
    kw.setdefault("sleeper", lambda s: None)
    kw.setdefault("rng", random.Random(1234))
    return YfClient(fetcher=fetcher, **kw)


# --------------------------------------------------------------------------
# client: retry/backoff on 429 + jitter, then YfRateLimited
# --------------------------------------------------------------------------
def test_backoff_delay_exponential_with_jitter_bounds():
    c = make_client(FakeFetcher(), backoff_s=2.0, jitter=1.0)
    c.rng = random.Random(0)
    for attempt in range(6):
        d = c.backoff_delay(attempt)
        base = 2.0 * (2 ** attempt)
        assert 0.0 <= d <= 2.0 * base  # full jitter: base * U(0, 2)
    # deterministic given the seeded rng
    c2 = make_client(FakeFetcher(), backoff_s=2.0, jitter=1.0)
    c2.rng = random.Random(0)
    c3 = make_client(FakeFetcher(), backoff_s=2.0, jitter=1.0)
    c3.rng = random.Random(0)
    assert [c2.backoff_delay(a) for a in range(4)] == [c3.backoff_delay(a) for a in range(4)]


def test_history_retries_429_then_succeeds():
    f = FakeFetcher(
        frames={"AAPL": make_yf_frame("2026-01-01", 10)},
        fail_times={"AAPL": [Exception("429 Too Many Requests"), Exception("connection reset")]},
    )
    sleeps = []
    c = YfClient(fetcher=f, sleeper=sleeps.append, rng=random.Random(7), max_retries=4, backoff_s=2.0)
    df = c.history("AAPL", period="5y")
    assert len(df) == 10
    assert c.calls_made == 1
    assert c.retries == 2
    assert len(sleeps) == 2  # backed off twice before succeeding
    assert all(s > 0 for s in sleeps)


def test_history_rate_limited_after_max_retries():
    f = FakeFetcher(fail_times={"AAPL": [Exception("429 Too Many Requests")] * 10})
    c = make_client(f, max_retries=3, backoff_s=0.01)
    with pytest.raises(YfRateLimited) as ei:
        c.history("AAPL", period="5y")
    assert "rate-limited" in str(ei.value).lower() or "429" in str(ei.value)
    assert len(f.calls) == 4  # 1 initial + 3 retries, then give up


def test_history_non_retryable_error_raises_immediately():
    f = FakeFetcher(fail_times={"AAPL": [TypeError("bad arg")]})
    c = make_client(f, max_retries=5)
    with pytest.raises(YfRateLimited):
        c.history("AAPL", period="5y")
    assert len(f.calls) == 1  # no retries on a programming error


def test_history_empty_raises_yf_empty_no_retry():
    f = FakeFetcher(frames={})  # returns empty frame
    c = make_client(f, max_retries=5)
    with pytest.raises(YfEmpty):
        c.history("UNKNOWN", period="5y")
    assert len(f.calls) == 1


def test_invalid_period_rejected():
    c = make_client(FakeFetcher())
    with pytest.raises(YfError):
        c.history("AAPL", period="7y")


def test_download_many_sleeps_between_tickers_and_collects_errors():
    f = FakeFetcher(
        frames={"AAA": make_yf_frame("2026-01-01", 5), "BBB": make_yf_frame("2026-01-01", 5)},
        fail_times={"BBB": [Exception("429")] * 10},
    )
    sleeps = []
    c = YfClient(fetcher=f, sleeper=sleeps.append, rng=random.Random(1), max_retries=1, backoff_s=0.01)
    out = c.download_many(["AAA", "BBB", "CCC"], period="5y", sleep_s=2.0)
    assert len(out["AAA"]) == 5
    assert "BBB" in out["_errors"] and "CCC" in out["_errors"]
    # slept between tickers: once before BBB, once before CCC (i>0), plus backoff sleeps
    assert sleeps.count(2.0) == 2


# --------------------------------------------------------------------------
# frame normalization -> warehouse EOD schema (parquet round-trip)
# --------------------------------------------------------------------------
def test_history_to_bars_frame_tz_aware_index_matches_trading_date():
    # real yfinance returns tz-aware exchange-local timestamps; the UTC date
    # must equal the trading date for US listings
    days = pd.bdate_range("2026-01-01", periods=5, tz="America/New_York")
    df = pd.DataFrame({"Open": [1.0] * 5, "High": [1.0] * 5, "Low": [1.0] * 5,
                       "Close": [1.0] * 5, "Adj Close": [1.0] * 5, "Volume": [10] * 5},
                      index=pd.DatetimeIndex(days))
    bars = history_to_bars_frame(df, "AAPL", "US")
    assert [str(d) for d in bars["date"]] == ["2026-01-01", "2026-01-02", "2026-01-05",
                                               "2026-01-06", "2026-01-07"]


def test_history_to_bars_frame_schema_and_roundtrip(tmp_path):
    df = make_yf_frame("2026-01-01", 30, base_close=200.0, vol=5_000_000)
    bars = history_to_bars_frame(df, "AAPL", "US")
    assert list(bars.columns) == wh.EOD_COLUMNS
    assert len(bars) == 30
    assert bars["symbol"].iloc[0] == "AAPL.US"
    assert bars["exchange"].iloc[0] == "US"
    assert (bars["dollar_volume"] == bars["close"] * bars["volume"]).all()
    assert (bars["adjusted_close"] == bars["close"] * 0.99).all()
    # dates are plain dates, sorted
    assert str(bars["date"].iloc[0]) == "2026-01-01"
    assert list(bars["date"]) == sorted(bars["date"])

    path = wh.eod_parquet_path(str(tmp_path), "US", "AAPL")
    wh.write_parquet_atomic(bars, path)
    back = wh.read_parquet(path)
    assert back is not None and len(back) == 30
    assert list(back.columns) == wh.EOD_COLUMNS
    pd.testing.assert_frame_equal(back, bars)
    # same layout as the EODHD path: eod/<EX>/<SYMBOL>.parquet
    assert path.endswith(os.path.join("eod", "US", "AAPL.parquet"))


def test_history_to_bars_frame_adj_close_fallback():
    df = make_yf_frame("2026-01-01", 5)
    df = df.drop(columns=["Adj Close"])
    bars = history_to_bars_frame(df, "MSFT", "US")
    assert (bars["adjusted_close"] == bars["close"]).all()


def test_history_to_bars_frame_empty():
    bars = history_to_bars_frame(pd.DataFrame(), "X", "US")
    assert list(bars.columns) == wh.EOD_COLUMNS and bars.empty


# --------------------------------------------------------------------------
# ingest CLI end-to-end with a fake fetcher (resumable, research-grade vintage)
# --------------------------------------------------------------------------
def _patch_fetcher(monkeypatch, fetcher):
    monkeypatch.setattr(ingest_yf, "_FETCHER", fetcher)


def test_ingest_yf_end_to_end_resume_and_research_vintage(tmp_path, monkeypatch):
    root = str(tmp_path / "wh")
    # window end inside the fake data's coverage (fake frames end 2026-09-08,
    # bdate_range skips weekends) so the resume check treats them as covered
    to_date = "2026-09-08"
    frames = {
        "AAPL": make_yf_frame("2021-11-17", 1255, base_close=200.0, vol=5_000_000),
        "MSFT": make_yf_frame("2021-11-17", 1255, base_close=400.0, vol=2_000_000),
    }
    f = FakeFetcher(frames=frames)
    _patch_fetcher(monkeypatch, f)
    rc = ingest_yf.main(["--warehouse", root, "--tickers", "AAPL", "MSFT",
                         "--period", "5y", "--sleep", "0",
                         "--from", "2021-11-17", "--to", to_date])
    assert rc == 0
    assert f.calls == [("AAPL", "5y", "1d"), ("MSFT", "5y", "1d")]
    aapl = wh.read_parquet(wh.eod_parquet_path(root, "US", "AAPL"))
    assert aapl is not None and len(aapl) == 1255
    assert list(aapl.columns) == wh.EOD_COLUMNS

    # vintage marked research-grade with vendor + research scope id
    vrec = wh.read_json(os.path.join(root, "vintage.json"))
    assert vrec["vendor"] == VENDOR == "yfinance"
    assert vrec["research_grade"] is True
    assert vrec["scope_id"] == "research_5y"
    assert vrec["scope_id"].startswith("research_")
    assert vrec["scope_id"] not in ("free_1y_top1k", "pro_full")

    # rerun: parquet covers the window -> 0 network calls (resumable)
    rc = ingest_yf.main(["--warehouse", root, "--tickers", "AAPL", "MSFT",
                         "--period", "5y", "--sleep", "0",
                         "--from", "2021-11-17", "--to", to_date])
    assert rc == 0
    assert len(f.calls) == 2  # no re-fetch
    vrec2 = wh.read_json(os.path.join(root, "vintage.json"))
    assert vrec2["results"]["cached"] == 2


def test_ingest_yf_rate_limited_ticker_recorded_and_run_continues(tmp_path, monkeypatch):
    root = str(tmp_path / "wh")
    f = FakeFetcher(
        frames={"GOOD": make_yf_frame("2021-11-17", 100)},
        fail_times={"BAD": [Exception("429 Too Many Requests")] * 10},
    )
    _patch_fetcher(monkeypatch, f)
    rc = ingest_yf.main(["--warehouse", root, "--tickers", "BAD", "GOOD",
                         "--period", "5y", "--sleep", "0", "--max-retries", "2",
                         "--backoff", "0.001",
                         "--from", "2021-11-17", "--to", "2026-11-17"])
    assert rc == 0
    vrec = wh.read_json(os.path.join(root, "vintage.json"))
    assert vrec["results"]["error"] == 1 and vrec["results"]["fetched"] == 1
    assert "BAD" in vrec["errors"]
    assert wh.read_parquet(wh.eod_parquet_path(root, "US", "GOOD")) is not None


def test_scope_id_for_ingest():
    assert ingest_yf.scope_id_for("yf", "5y", 500) == "research_5y_top500"
    assert ingest_yf.scope_id_for("yf", "5y", None) == "research_5y"
    assert ingest_yf.scope_id_for("yfinance", "1y", 1000) == "research_1y_top1k"
    with pytest.raises(ValueError):
        ingest_yf.scope_id_for("eodhd", "5y", None)  # not a research source


# --------------------------------------------------------------------------
# universe: research vintage -> delist_idx unavailable, survivorship flagged
# --------------------------------------------------------------------------
def _seed_research_warehouse(root):
    eod = os.path.join(root, "eod", "US")
    os.makedirs(eod, exist_ok=True)
    for sym, n, close, vol in [("AAPL", 30, 200.0, 5_000_000),
                               ("DEAD", 5, 50.0, 1_000_000)]:
        rows = make_yf_frame("2026-10-01", n, base_close=close, vol=vol)
        bars = history_to_bars_frame(rows, sym, "US")
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
    assert (ranked["delist_idx_available"] == False).all()  # never survivorship-free
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
    rows = make_yf_frame("2026-10-01", 10)
    bars = history_to_bars_frame(rows, "AAPL", "US")
    wh.write_parquet_atomic(bars, os.path.join(eod, "AAPL.parquet"))
    wh.write_vintage(root, extra={"built_by": "ingest_eodhd"})  # no research flags
    rc = universe.main(["--warehouse", root, "--n", "10", "--asof", "2026-11-15"])
    assert rc == 0
    ranked = wh.read_parquet(os.path.join(root, "universe", "ranked_2026-11-15.parquet"))
    assert (ranked["delist_idx_available"] == True).all()
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
    import numpy as np
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
        # yfinance path writes the BARE yahoo ticker as the parquet name
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
    # suffixed universe symbols must resolve to the bare-ticker parquet files
    prices, kept, info = fr.load_price_matrix(root, rng_symbols, "US")
    assert prices.shape == (120, 2) and kept == rng_symbols and info["loaded"] == 2
    # bare symbols resolve too (Yahoo-style class shares keep their raw name)
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
# universe-selection integrity (hermes-tasks #182 re-file): --top must NOT
# truncate the seed list alphabetically; warrants/units/rights excluded from
# the seed and the ranking; alphabet sanity guard on the candidate pool;
# universe quality floors (min bars + avg dollar volume)
# --------------------------------------------------------------------------
from data.security_type import is_common_stock_symbol  # noqa: E402


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


def test_load_seed_list_file_excludes_non_common_types(tmp_path):
    p = tmp_path / "seed.txt"
    p.write_text("\n".join(["AAPL", "MSFT", "BRK.B", "AAC-WT", "FOO-UN", "Q-RT",
                            "AESPW", "# comment", ""]) + "\n")
    syms = ingest_yf.load_seed_list(f"file:{p}")
    assert syms == ["AAPL", "BRK.B", "MSFT"]


def _alpha_seed(n_per_letter: int = 4) -> list[str]:
    """A seed list spanning the whole alphabet (n_per_letter symbols per letter)."""
    return [f"{c}{c}{i}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for i in range(n_per_letter)]


def test_top_no_longer_truncates_seed_list(tmp_path, monkeypatch, capsys):
    root = str(tmp_path / "wh")
    seed = _alpha_seed()  # 104 symbols A-Z
    monkeypatch.setattr(ingest_yf, "load_seed_list", lambda kind: list(seed))
    f = FakeFetcher(frames={s: make_yf_frame("2026-01-01", 5) for s in seed})
    _patch_fetcher(monkeypatch, f)
    rc = ingest_yf.main(["--warehouse", root, "--seed-list", "sec", "--top", "5",
                         "--period", "5y", "--sleep", "0",
                         "--from", "2026-01-01", "--to", "2026-01-31"])
    assert rc == 0
    assert len(f.calls) == len(seed)                       # NO alphabetical truncation
    assert [s for s, _, _ in f.calls] != sorted(seed)[:5]
    vrec = wh.read_json(os.path.join(root, "vintage.json"))
    assert vrec["scope_id"] == "research_5y"               # --top no longer labels the scope
    assert "deprecat" in capsys.readouterr().out.lower()   # loud deprecation warning


def test_max_ingest_cap_samples_unbiasedly_not_alphabetically(tmp_path, monkeypatch):
    seed = _alpha_seed(8)  # 208 symbols A-Z
    monkeypatch.setattr(ingest_yf, "load_seed_list", lambda kind: list(seed))
    chosen_runs = []
    for run in range(2):  # deterministic across runs (fixed default seed)
        root = str(tmp_path / f"wh{run}")
        f = FakeFetcher(frames={s: make_yf_frame("2026-01-01", 5) for s in seed})
        _patch_fetcher(monkeypatch, f)
        rc = ingest_yf.main(["--warehouse", root, "--seed-list", "sec", "--max-ingest", "10",
                             "--period", "5y", "--sleep", "0",
                             "--from", "2026-01-01", "--to", "2026-01-31"])
        assert rc == 0
        chosen = [s for s, _, _ in f.calls]
        assert len(chosen) == 10
        assert chosen != sorted(seed)[:10]                 # not the alphabetical head
        assert len({s[0] for s in chosen}) >= 5            # spans the alphabet
        chosen_runs.append(chosen)
    assert chosen_runs[0] == chosen_runs[1]                # seeded -> reproducible


def test_alphabet_sanity_guard_refuses_degenerate_pool(tmp_path, monkeypatch, capsys):
    root = str(tmp_path / "wh")
    degenerate = [f"A{i:03d}" for i in range(200)]         # all first-letter A
    monkeypatch.setattr(ingest_yf, "load_seed_list", lambda kind: list(degenerate))
    f = FakeFetcher(frames={s: make_yf_frame("2026-01-01", 5) for s in degenerate})
    _patch_fetcher(monkeypatch, f)
    rc = ingest_yf.main(["--warehouse", root, "--seed-list", "sec",
                         "--period", "5y", "--sleep", "0",
                         "--from", "2026-01-01", "--to", "2026-01-31"])
    assert rc == 2                                          # refuse, don't silently bias
    assert f.calls == []                                    # refused before any network call
    assert not os.path.exists(os.path.join(root, "vintage.json"))
    assert "degenerate" in capsys.readouterr().out.lower()

    # control: an alphabet-spanning pool passes the guard
    root2 = str(tmp_path / "wh2")
    seed = _alpha_seed()
    monkeypatch.setattr(ingest_yf, "load_seed_list", lambda kind: list(seed))
    f2 = FakeFetcher(frames={s: make_yf_frame("2026-01-01", 5) for s in seed})
    _patch_fetcher(monkeypatch, f2)
    rc2 = ingest_yf.main(["--warehouse", root2, "--seed-list", "sec",
                          "--period", "5y", "--sleep", "0",
                          "--from", "2026-01-01", "--to", "2026-01-31"])
    assert rc2 == 0 and len(f2.calls) == len(seed)


def _seed_bars_with_floors(root):
    eod = os.path.join(root, "eod", "US")
    os.makedirs(eod, exist_ok=True)
    frames = {
        "AAPL": make_yf_frame("2026-10-01", 30, base_close=200.0, vol=5_000_000),  # ADV 1e9, 30 bars
        "MICRO": make_yf_frame("2026-10-01", 30, base_close=2.0, vol=1_000),       # ADV 2e3
        "ONEBAR": make_yf_frame("2026-10-30", 1, base_close=500.0, vol=1_000_000),  # 1 bar
        "AESPW": make_yf_frame("2026-10-01", 30, base_close=1.0, vol=10_000_000),   # warrant, ADV 1e7
    }
    for sym, df in frames.items():
        bars = history_to_bars_frame(df, sym, "US")
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
