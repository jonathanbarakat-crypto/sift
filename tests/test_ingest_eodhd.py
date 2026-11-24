"""Unit tests for the EODHD ingestion + PIT universe builder.

The HTTP layer is fully mocked: a FakeSession stands in for
requests.Session and serves canned EODHD responses while counting calls,
so the tests never touch the network and never need a real API token.
"""
from __future__ import annotations

import json
import os
from datetime import date, timedelta

import pandas as pd
import pytest

from data import warehouse as wh
from data.eodhd_client import (
    FREE_TIER_DAILY_CALLS, PAID_EOD_DAILY_CALLS, CallBudgetExceeded, DeepHistoryBlocked,
    EodhdClient, EodhdError, resolve_token,
)
from data import ingest_eodhd, universe


# --------------------------------------------------------------------------
# Mocked HTTP layer
# --------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, status_code: int, payload, url: str = "https://eodhd.com/api/x"):
        self.status_code = status_code
        self._payload = payload
        self.text = payload if isinstance(payload, str) else json.dumps(payload)
        self.url = url

    def json(self):
        if isinstance(self._payload, str):
            raise ValueError("not json")
        return self._payload


class FakeSession:
    """Duck-typed requests.Session. routes: list of (predicate, FakeResponse)."""

    def __init__(self, routes=None, default=None):
        self.routes = routes or []
        self.default = default if default is not None else FakeResponse(404, {"error": "no route"})
        self.requests = []  # (url, params)

    def get(self, url, params=None, timeout=None):
        params = dict(params or {})
        params.pop("api_token", None)  # token must never be inspected/logged
        self.requests.append((url, params))
        for pred, resp in self.routes:
            if pred(url, params):
                resp.url = url
                return resp
        return self.default


def bars(symbol_rows):
    return symbol_rows


def make_eod_rows(start: str, n: int, base_close: float = 100.0, vol: int = 1_000_000):
    """Business-day bars only (real exchanges don't trade weekends)."""
    d0 = pd.bdate_range(start, periods=n)[0].date()
    days = pd.bdate_range(start, periods=n)
    return [
        {"date": days[i].date().isoformat(), "open": base_close, "high": base_close + 1,
         "low": base_close - 1, "close": base_close, "adjusted_close": base_close,
         "volume": vol}
        for i in range(n)
    ]


EXCHANGES = [{"Code": "US", "Name": "NYSE/Nasdaq composite", "Country": "USA", "Currency": "USD"}]
SYMBOL_LIST_ACTIVE = [
    {"Code": "AAPL", "Name": "Apple", "Type": "Common Stock", "Exchange": "NASDAQ"},
    {"Code": "MSFT", "Name": "Microsoft", "Type": "Common Stock", "Exchange": "NASDAQ"},
    {"Code": "DEAD", "Name": "Delisted Co", "Type": "Common Stock", "Exchange": "NYSE"},
]


def std_routes(eod_map=None, delisted_403=True):
    eod_map = eod_map or {}
    routes = [
        (lambda u, p: u.endswith("/exchanges-list"), FakeResponse(200, EXCHANGES)),
        (lambda u, p: u.endswith("/exchange-symbol-list/US") and p.get("delisted") == "0",
         FakeResponse(200, SYMBOL_LIST_ACTIVE)),
        (lambda u, p: u.endswith("/exchange-symbol-list/US") and p.get("delisted") == "1",
         FakeResponse(403, {"error": "not available on free tier"}) if delisted_403 else FakeResponse(200, [])),
    ]
    for sym, rows in eod_map.items():
        routes.append((lambda u, p, s=sym: u.endswith(f"/eod/{s}"), FakeResponse(200, rows)))
    routes.append((lambda u, p: u.endswith("/eod/OLD.US"),
                   FakeResponse(403, {"error": "no data / limit"})))
    routes.append((lambda u, p: u.endswith("/real-time/AAPL.US"),
                   FakeResponse(200, {"code": "AAPL.US", "close": 101.0, "previousClose": 100.0,
                                      "change": 1.0, "change_p": 1.0, "volume": 123, "date": "2026-11-14"})))
    return routes


def make_client(tmp_path, session, max_calls=19, token="test-token-XYZ"):
    return EodhdClient(token=token, max_calls=max_calls,
                       state_file=str(tmp_path / "state" / "calls.json"), session=session)


# --------------------------------------------------------------------------
# token resolution — value never leaks
# --------------------------------------------------------------------------
def test_token_explicit_arg_wins():
    assert resolve_token("abc") == "abc"


def test_token_from_env(monkeypatch):
    monkeypatch.setenv("EODHD_TOKEN", "env-token")
    assert resolve_token(None) == "env-token"


def test_token_missing_raises_without_leaking(monkeypatch, tmp_path):
    monkeypatch.delenv("EODHD_TOKEN", raising=False)
    monkeypatch.setattr("subprocess.run", lambda *a, **k: type("R", (), {"returncode": 1, "stdout": ""})())
    with pytest.raises(EodhdError) as ei:
        resolve_token(None)
    assert "pass" in str(ei.value) and "eodhd/api-token" in str(ei.value)


def test_errors_never_contain_token(tmp_path):
    sess = FakeSession(default=FakeResponse(403, {"error": "limit"}))
    c = make_client(tmp_path, sess, token="super-secret-token")
    with pytest.raises(EodhdError) as ei:
        c.get("/eod/AAPL.US")
    assert "super-secret-token" not in str(ei.value)
    assert "super-secret-token" not in str(ei.value.url or "")


# --------------------------------------------------------------------------
# call budget
# --------------------------------------------------------------------------
def test_max_calls_above_free_limit_rejected(tmp_path):
    with pytest.raises(ValueError):
        make_client(tmp_path, FakeSession(), max_calls=FREE_TIER_DAILY_CALLS)
    with pytest.raises(ValueError):
        make_client(tmp_path, FakeSession(), max_calls=100)


def test_plan_paid_raises_cap_and_default_budget(tmp_path):
    sess = FakeSession()
    # paid plan default budget = plan limit - 1, no explicit max_calls
    c = EodhdClient(token="t", max_calls=None, state_file=str(tmp_path / "s" / "c.json"),
                   session=sess, plan="paid")
    assert c.max_calls == PAID_EOD_DAILY_CALLS - 1
    # paid plan accepts budgets up to its own limit, still rejects above it
    c2 = EodhdClient(token="t", max_calls=50_000, state_file=str(tmp_path / "s2" / "c.json"),
                    session=FakeSession(), plan="paid")
    assert c2.max_calls == 50_000
    with pytest.raises(ValueError):
        EodhdClient(token="t", max_calls=PAID_EOD_DAILY_CALLS,
                    state_file=str(tmp_path / "s3" / "c.json"), session=FakeSession(), plan="paid")
    # free plan still rejects paid-scale budgets
    with pytest.raises(ValueError):
        EodhdClient(token="t", max_calls=1000,
                    state_file=str(tmp_path / "s4" / "c.json"), session=FakeSession(), plan="free")
    with pytest.raises(ValueError):
        EodhdClient(token="t", plan="enterprise")  # unknown plan


def test_default_window_paid_starts_2000(tmp_path):
    import datetime as dt
    d_from, d_to = ingest_eodhd.default_window(today=dt.date(2026, 11, 20), plan="paid")
    assert d_from == "2000-01-01" and d_to == "2026-11-20"
    d_from, d_to = ingest_eodhd.default_window(today=dt.date(2026, 11, 20), plan="free")
    assert d_from == "2025-11-20" and d_to == "2026-11-20"


def test_ingest_paid_plan_ingests_delisted_and_lifts_cap(tmp_path, monkeypatch):
    root = str(tmp_path / "wh")
    delisted_list = [
        {"Code": "LEHMQ", "Name": "Lehman", "Type": "Common Stock", "Exchange": "NYSE"},
        {"Code": "SOMEETF", "Name": "Dead ETF", "Type": "ETF", "Exchange": "NYSE"},
    ]
    rows_map = {
        "AAPL.US": make_eod_rows("2026-10-01", 30, base_close=200.0, vol=5_000_000),
        "MSFT.US": make_eod_rows("2026-10-01", 30, base_close=400.0, vol=2_000_000),
        "DEAD.US": make_eod_rows("2026-10-01", 5, base_close=3.0, vol=10_000),
        "LEHMQ.US": make_eod_rows("2026-10-01", 5, base_close=1.5, vol=500_000),
    }
    routes = std_routes(rows_map, delisted_403=False)
    routes[2] = (lambda u, p: u.endswith("/exchange-symbol-list/US") and p.get("delisted") == "1",
                 FakeResponse(200, delisted_list))
    sess = FakeSession(routes=routes)
    _patch_client(monkeypatch, sess)
    rc = ingest_eodhd.main(["--warehouse", root, "--plan", "paid", "--max-calls", "50000",
                            "--from", "2026-10-01", "--to", "2026-11-11"])
    assert rc == 0
    assert wh.read_parquet(wh.eod_parquet_path(root, "US", "LEHMQ")) is not None
    assert wh.read_parquet(wh.eod_parquet_path(root, "US", "SOMEETF")) is None
    vrec = wh.read_json(os.path.join(root, "vintage.json"))
    assert vrec["plan"] == "paid"


def test_ingest_paid_plan_ignores_prior_free_tier_delisted_gap(tmp_path, monkeypatch):
    root = str(tmp_path / "wh")
    # First run on free tier records US-delisted.json with available_on_free_tier=False
    sess_free = FakeSession(routes=std_routes({
        "AAPL.US": make_eod_rows("2026-10-01", 30),
        "MSFT.US": make_eod_rows("2026-10-01", 30),
        "DEAD.US": make_eod_rows("2026-10-01", 5),
    }, delisted_403=True))
    _patch_client(monkeypatch, sess_free)
    assert ingest_eodhd.main(["--warehouse", root, "--plan", "free", "--max-calls", "19",
                              "--from", "2026-10-01", "--to", "2026-11-11"]) == 0
    gap = wh.read_json(os.path.join(root, "meta", "symbol-list", "US-delisted.json"))
    assert gap["available_on_free_tier"] is False

    # Upgrading to --plan paid must re-fetch delisted=1 without needing --force
    delisted_list = [{"Code": "LEHMQ", "Name": "Lehman", "Type": "Common Stock"}]
    routes_paid = std_routes({
        "LEHMQ.US": make_eod_rows("2026-10-01", 5),
    }, delisted_403=False)
    routes_paid[2] = (lambda u, p: u.endswith("/exchange-symbol-list/US") and p.get("delisted") == "1",
                      FakeResponse(200, delisted_list))
    sess_paid = FakeSession(routes=routes_paid)
    _patch_client(monkeypatch, sess_paid)
    assert ingest_eodhd.main(["--warehouse", root, "--plan", "paid", "--max-calls", "1000",
                              "--from", "2026-10-01", "--to", "2026-11-11"]) == 0
    assert wh.read_parquet(wh.eod_parquet_path(root, "US", "LEHMQ")) is not None
    cached_delisted = wh.read_json(os.path.join(root, "meta", "symbol-list", "US-delisted.json"))
    assert isinstance(cached_delisted, list) and cached_delisted[0]["Code"] == "LEHMQ"


def test_ingest_paid_plan_delisted_403_fails_closed(tmp_path, monkeypatch):
    root = str(tmp_path / "wh")
    sess = FakeSession(routes=std_routes(delisted_403=True))
    _patch_client(monkeypatch, sess)
    rc = ingest_eodhd.main(["--warehouse", root, "--plan", "paid", "--max-calls", "1000",
                            "--from", "2026-10-01", "--to", "2026-11-11"])
    assert rc == 1  # fail closed on paid plan if delisted=1 is refused


def test_ingest_cli_rejects_max_calls_above_free_cap(tmp_path, monkeypatch):
    root = str(tmp_path / "wh")
    _patch_client(monkeypatch, FakeSession())
    rc = ingest_eodhd.main(["--warehouse", root, "--plan", "free", "--max-calls", "100"])
    assert rc == 2


def test_budget_exceeded_raises_before_http(tmp_path):
    sess = FakeSession(routes=std_routes())
    c = make_client(tmp_path, sess, max_calls=2)
    c.get("/exchanges-list")
    c.get("/exchanges-list")
    with pytest.raises(CallBudgetExceeded):
        c.get("/exchanges-list")
    assert len(sess.requests) == 2  # third request never hit the network


def test_call_count_persisted_resume(tmp_path):
    sess = FakeSession(routes=std_routes())
    c1 = make_client(tmp_path, sess, max_calls=19)
    c1.get("/exchanges-list")
    assert c1.calls_made == 1
    # a new client (simulating a rerun after a crash) resumes the count
    c2 = make_client(tmp_path, sess, max_calls=19)
    assert c2.calls_made == 1
    assert c2.calls_remaining() == 18


def test_one_call_per_ticker_per_day(tmp_path):
    sess = FakeSession(routes=std_routes({"AAPL.US": make_eod_rows("2026-10-01", 30)}))
    c = make_client(tmp_path, sess)
    c.eod("AAPL.US", from_date="2026-10-01", to_date="2026-11-15")
    c.eod("AAPL.US", from_date="2026-10-01", to_date="2026-11-15")  # would be a 2nd call
    aapl_calls = [r for r in sess.requests if r[0].endswith("/eod/AAPL.US")]
    assert len(aapl_calls) == 2  # client itself allows it; the CLI skips via cache
    assert c.calls_made == 2


# --------------------------------------------------------------------------
# deep-history probes — clear error, expected, not bypassed
# --------------------------------------------------------------------------
def test_deep_history_403_raises_clear_error(tmp_path):
    sess = FakeSession(routes=std_routes())
    c = make_client(tmp_path, sess)
    with pytest.raises(DeepHistoryBlocked) as ei:
        c.eod("OLD.US", from_date="1990-01-01", to_date="1990-12-31", deep_history=True)
    msg = str(ei.value)
    assert "free tier" in msg and "1-year" in msg or "pre-1-year" in msg
    assert ei.value.status == 403


def test_generic_403_is_plain_error_not_deep_history(tmp_path):
    sess = FakeSession(routes=std_routes())
    c = make_client(tmp_path, sess)
    with pytest.raises(EodhdError) as ei:
        c.eod("OLD.US", from_date="2026-10-01", to_date="2026-11-01", deep_history=False)
    assert not isinstance(ei.value, DeepHistoryBlocked)
    assert ei.value.status == 403


# --------------------------------------------------------------------------
# ingestion CLI end-to-end with mocked HTTP
# --------------------------------------------------------------------------
def _patch_client(monkeypatch, session):
    monkeypatch.setattr(ingest_eodhd, "_SESSION", session)
    monkeypatch.setenv("SIFT_TEST_TOKEN", "test-token-XYZ")


def test_ingest_end_to_end_and_resume(tmp_path, monkeypatch):
    root = str(tmp_path / "wh")
    sess = FakeSession(routes=std_routes({
        "AAPL.US": make_eod_rows("2026-10-01", 30, base_close=200.0, vol=5_000_000),
        "MSFT.US": make_eod_rows("2026-10-01", 30, base_close=400.0, vol=2_000_000),
        "DEAD.US": make_eod_rows("2026-10-01", 5, base_close=3.0, vol=10_000),
    }))
    _patch_client(monkeypatch, sess)
    rc = ingest_eodhd.main(["--warehouse", root, "--exchange", "US", "--max-calls", "19",
                            "--from", "2026-10-01", "--to", "2026-11-11"])
    assert rc == 0
    # exchanges-list(1) + active list(1) + delisted 403(1) + 3 tickers = 6 calls
    assert len(sess.requests) == 6
    aapl = wh.read_parquet(wh.eod_parquet_path(root, "US", "AAPL"))
    assert aapl is not None and len(aapl) == 30
    assert list(aapl.columns) == wh.EOD_COLUMNS
    assert (aapl["dollar_volume"] == aapl["close"] * aapl["volume"]).all()
    assert aapl["symbol"].iloc[0] == "AAPL.US"
    assert wh.read_vintage(root) is not None
    # delisted-list gap recorded explicitly
    gap = wh.read_json(os.path.join(root, "meta", "symbol-list", "US-delisted.json"))
    assert gap and gap["available_on_free_tier"] is False and gap["http_status"] == 403

    # rerun with the same window as the first run: everything cached -> 0 calls
    # (DEAD's series ends 2026-10-07, within stale_days of the 2026-11-11 end,
    #  so it is treated as covered too — one call per ticker per day max)
    n_before = len(sess.requests)
    rc = ingest_eodhd.main(["--warehouse", root, "--max-calls", "19",
                            "--from", "2026-10-01", "--to", "2026-11-11"])
    assert rc == 0
    assert len(sess.requests) == n_before  # fully resumable, no re-spend


def test_ingest_max_calls_guard_stops_run(tmp_path, monkeypatch):
    root = str(tmp_path / "wh")
    many = [f"S{i:03d}" for i in range(10)]
    rows_map = {f"{s}.US": make_eod_rows("2026-10-01", 5) for s in many}
    # active symbol list returns our 10 names
    routes = std_routes(rows_map)
    routes[1] = (lambda u, p: u.endswith("/exchange-symbol-list/US") and p.get("delisted") == "0",
                 FakeResponse(200, [{"Code": s, "Type": "Common Stock"} for s in many]))
    sess = FakeSession(routes=routes)
    _patch_client(monkeypatch, sess)
    # budget 4: exchanges-list(1) + active list(1) + delisted 403(1) = 3 spent,
    # leaving exactly 1 for tickers; the guard must stop before a 2nd fetch
    rc = ingest_eodhd.main(["--warehouse", root, "--max-calls", "4",
                            "--from", "2026-10-01", "--to", "2026-11-15"])
    assert rc == 0
    eod_calls = [r for r in sess.requests if "/eod/" in r[0]]
    assert len(eod_calls) == 1  # guard stopped the run after 1 ticker fetch
    assert len(sess.requests) == 4  # budget fully consumed, never exceeded
    # resume with a fresh "day" (new state file date is same, but force refetch of remaining)
    sess2 = FakeSession(routes=routes)
    _patch_client(monkeypatch, sess2)
    rc = ingest_eodhd.main(["--warehouse", root, "--max-calls", "19", "--force",
                            "--from", "2026-10-01", "--to", "2026-11-15"])
    assert rc == 0


def test_ingest_deep_history_probe_exit_code(tmp_path, monkeypatch):
    root = str(tmp_path / "wh")
    sess = FakeSession(routes=std_routes())
    _patch_client(monkeypatch, sess)
    rc = ingest_eodhd.main(["--warehouse", root, "--probe-deep-history", "OLD",
                            "--max-calls", "19"])
    assert rc == 3  # DeepHistoryBlocked surfaced as clear error, not bypassed


def test_ingest_live_refresh(tmp_path, monkeypatch):
    root = str(tmp_path / "wh")
    sess = FakeSession(routes=std_routes({"AAPL.US": make_eod_rows("2026-10-01", 10)}))
    _patch_client(monkeypatch, sess)
    rc = ingest_eodhd.main(["--warehouse", root, "--tickers", "AAPL", "--refresh-live",
                            "--from", "2026-10-01", "--to", "2026-11-15"])
    assert rc == 0
    live = wh.read_parquet(wh.live_parquet_path(root, "AAPL.US"))
    assert live is not None and live["close"].iloc[0] == 101.0


def test_exchanges_list_only_one_call(tmp_path, monkeypatch):
    root = str(tmp_path / "wh")
    sess = FakeSession(routes=std_routes())
    _patch_client(monkeypatch, sess)
    rc = ingest_eodhd.main(["--warehouse", root, "--exchanges-list-only"])
    assert rc == 0 and len(sess.requests) == 1
    rc = ingest_eodhd.main(["--warehouse", root, "--exchanges-list-only"])
    assert rc == 0 and len(sess.requests) == 1  # cached second time


# --------------------------------------------------------------------------
# warehouse vintage hash
# --------------------------------------------------------------------------
def test_vintage_hash_changes_with_data_not_state(tmp_path):
    root = str(tmp_path / "wh")
    os.makedirs(os.path.join(root, "eod", "US"))
    df = pd.DataFrame([{"date": date(2026, 11, 1), "close": 1.0, "volume": 10,
                        "dollar_volume": 10.0, "symbol": "X.US", "exchange": "US",
                        "open": 1, "high": 1, "low": 1, "adjusted_close": 1}])
    wh.write_parquet_atomic(df, os.path.join(root, "eod", "US", "X.parquet"))
    v1 = wh.write_vintage(root)["vintage"]
    # state/ changes must NOT move the vintage
    os.makedirs(os.path.join(root, "state"))
    wh.write_json_atomic({"calls_today": 5}, os.path.join(root, "state", "calls.json"))
    v2 = wh.write_vintage(root)["vintage"]
    assert v1 == v2
    # data change must move the vintage
    df2 = df.copy(); df2["close"] = 2.0
    wh.write_parquet_atomic(df2, os.path.join(root, "eod", "US", "X.parquet"))
    v3 = wh.write_vintage(root)["vintage"]
    assert v3 != v1
    assert wh.read_vintage(root) == v3


# --------------------------------------------------------------------------
# PIT universe builder — no survivorship bias
# --------------------------------------------------------------------------
def _seed_bars(root):
    eod = os.path.join(root, "eod", "US")
    os.makedirs(eod, exist_ok=True)
    frames = {
        "AAPL": make_eod_rows("2026-10-01", 30, base_close=200.0, vol=5_000_000),   # big
        "MSFT": make_eod_rows("2026-10-01", 30, base_close=400.0, vol=2_000_000),   # mid
        "DEAD": make_eod_rows("2026-10-01", 5, base_close=50.0, vol=1_000_000),     # ends 2026-10-07
    }
    for sym, rows in frames.items():
        df = pd.DataFrame(rows)
        df["dollar_volume"] = df["close"] * df["volume"]
        df["symbol"] = f"{sym}.US"; df["exchange"] = "US"
        wh.write_parquet_atomic(df[wh.EOD_COLUMNS], os.path.join(eod, f"{sym}.parquet"))


def test_universe_pit_no_survivorship_bias(tmp_path):
    root = str(tmp_path / "wh")
    _seed_bars(root)
    asof = "2026-11-15"
    bars = universe.load_all_bars(root)
    ranked = universe.build_ranked_table(bars, asof)
    assert list(ranked["symbol"]) == ["AAPL.US", "MSFT.US", "DEAD.US"]  # by avg dollar vol
    dead = ranked[ranked["symbol"] == "DEAD.US"].iloc[0]
    assert dead["ended_before_asof"] is True or bool(dead["ended_before_asof"]) is True
    assert str(dead["last_date"]) == "2026-10-07"
    top = universe.build_pit_listing(ranked, 2)
    assert set(top["symbol"]) == {"AAPL.US", "MSFT.US"}

    # small N forces the ended name into the top-N listing: it must stay, flagged
    top1 = universe.build_pit_listing(ranked, 3)
    assert "DEAD.US" in set(top1["symbol"])
    assert bool(top1[top1["symbol"] == "DEAD.US"]["ended_before_asof"].iloc[0]) is True


def test_universe_asof_excludes_future_bars(tmp_path):
    root = str(tmp_path / "wh")
    _seed_bars(root)
    # as of 2026-10-03 only 3 bars exist per name; DEAD still present, AAPL adv over 3 bars
    ranked = universe.build_ranked_table(universe.load_all_bars(root), "2026-10-05")
    aapl = ranked[ranked["symbol"] == "AAPL.US"].iloc[0]
    assert int(aapl["bars"]) == 3
    assert float(aapl["avg_dollar_volume"]) == pytest.approx(200.0 * 5_000_000)
    # a bar dated after asof must not leak in
    assert all(str(d) <= "2026-10-05" for d in ranked["last_date"])


def test_universe_cli_writes_outputs_and_vintage(tmp_path):
    root = str(tmp_path / "wh")
    _seed_bars(root)
    # explicit escape hatch: these fixtures test ended-name flagging, not the
    # pre-registered quality floors (DEAD has only 5 bars -> below the default)
    rc = universe.main(["--warehouse", root, "--n", "2", "--asof", "2026-10-31",
                        "--min-bars", "1", "--min-avg-dollar-volume", "0"])
    assert rc == 0
    top = wh.read_parquet(os.path.join(root, "universe", "top2_pit_2026-10-31.parquet"))
    assert len(top) == 2 and top["rank"].tolist() == [1, 2]
    manifest = wh.read_json(os.path.join(root, "universe", "manifest.json"))
    assert manifest["symbols_ranked"] == 3
    # as of 2026-10-31 all series have ended (last bar 2026-10-30) — but with an
    # asof inside the window only DEAD (last bar 2026-10-07) has ended
    assert manifest["symbols_ended_before_asof"] == 3
    assert "survivorship_bias" in manifest and "none" in manifest["survivorship_bias"]
    assert wh.read_vintage(root) is not None


def test_universe_cli_ended_names_flagged(tmp_path):
    root = str(tmp_path / "wh")
    _seed_bars(root)
    # explicit escape hatch: ended-name flagging fixtures (DEAD has 5 bars)
    rc = universe.main(["--warehouse", root, "--n", "1000", "--asof", "2026-10-20",
                        "--min-bars", "1", "--min-avg-dollar-volume", "0"])
    assert rc == 0
    ranked = wh.read_parquet(os.path.join(root, "universe", "ranked_2026-10-20.parquet"))
    dead = ranked[ranked["symbol"] == "DEAD.US"].iloc[0]
    assert bool(dead["ended_before_asof"]) is True and str(dead["last_date"]) == "2026-10-07"
    live = ranked[ranked["symbol"] == "AAPL.US"].iloc[0]
    assert bool(live["ended_before_asof"]) is False
    manifest = wh.read_json(os.path.join(root, "universe", "manifest.json"))
    assert manifest["symbols_ended_before_asof"] == 1
