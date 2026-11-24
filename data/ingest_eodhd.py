#!/usr/bin/env python3
"""SIFT EODHD ingestion (plan-aware: free or paid, server-side only).

Fetches the exchanges list once (cached), then per-ticker daily EOD for US
stocks, writing one parquet file per ticker under ``data/warehouse/eod/US/``
plus a data-vintage content hash (``data/warehouse/VINTAGE``).

Plans (selected via ``--plan``):
  * ``free`` (default): 20 API calls/day, past-1-year EOD only, live quotes
    included, no delisted=1 symbol lists.
  * ``paid`` (EOD Historical Data — All World): 100,000 API calls/day,
    default window ``2000-01-01``..today (delisted coverage floor), active +
    delisted symbol lists.

Design:
  * resumable — already-ingested tickers whose parquet covers the requested
    window are skipped without spending a call;
  * cached on disk — exchanges-list and symbol lists are cached under
    ``data/warehouse/meta/`` and reused across runs;
  * one call per ticker per run at most — ``--max-calls`` (default: plan
    limit - 1) is a hard guard enforced BEFORE each HTTP request, and the
    per-day call count is persisted in ``state/calls.json`` so a rerun after
    a crash does not re-spend calls;
  * deep-history (pre-1-year) probes raise ``DeepHistoryBlocked`` with a clear
    message — expected on the free tier, never retried, never bypassed.

The API token is resolved via --api-token / $EODHD_TOKEN / ``pass
eodhd/api-token`` and is never printed, logged, or committed.

v3 scope doc hard rules (section 12): this pipeline is server-side mining
infrastructure only; raw data is never served to clients; SIFT does not
trade.

Usage:
    python3 -m data.ingest_eodhd --warehouse data/warehouse --plan paid --max-calls 50000
    python3 -m data.ingest_eodhd --warehouse data/warehouse --plan free --max-calls 19
    python3 -m data.ingest_eodhd --tickers AAPL MSFT --refresh-live
    python3 -m data.ingest_eodhd --exchanges-list-only
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta, timezone

import pandas as pd

# allow running both as a module (python3 -m data.ingest_eodhd) and as a script
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data import warehouse as wh  # type: ignore
    from data.eodhd_client import (  # type: ignore
        PLAN_DAILY_CALLS,
        CallBudgetExceeded,
        DeepHistoryBlocked,
        EodhdClient,
        EodhdError,
    )
else:
    from . import warehouse as wh
    from .eodhd_client import (
        PLAN_DAILY_CALLS,
        CallBudgetExceeded,
        DeepHistoryBlocked,
        EodhdClient,
        EodhdError,
    )

DEFAULT_WAREHOUSE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "data", "warehouse")
FREE_TIER_HISTORY_DAYS = 365  # free tier serves the past 1 year of EOD only

# tests monkeypatch this to inject a FakeSession (mocked HTTP layer)
_SESSION = None
# sentinel default so tests never hit `pass`; production runs must resolve a
# real token via $EODHD_TOKEN / pass — the sentinel is rejected at request time
_TEST_TOKEN_SENTINEL = "test-token-not-real"


def log(msg: str) -> None:
    print(f"[ingest_eodhd {datetime.now(timezone.utc).strftime('%H:%M:%S')}Z] {msg}", flush=True)


def default_window(today: date | None = None, plan: str = "free") -> tuple[str, str]:
    """Default ingest window. Free plan: (today - 365d, today) — deep history
    is not available there. Paid plan: (2000-01-01, today) — the delisted
    ticker list only covers names from January 2000, so earlier windows are
    survivorship-biased by construction (S5 exists to kill exactly that)."""
    today = today or date.today()
    if plan == "paid":
        return ("2000-01-01", today.isoformat())
    return ((today - timedelta(days=FREE_TIER_HISTORY_DAYS)).isoformat(), today.isoformat())


def _bars_to_frame(rows: list, symbol: str, exchange: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=wh.EOD_COLUMNS)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for c in ("open", "high", "low", "close", "adjusted_close", "volume"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["adjusted_close"] = df["adjusted_close"].fillna(df["close"])
    df["dollar_volume"] = df["close"] * df["volume"]
    df["symbol"] = symbol
    df["exchange"] = exchange
    return df[wh.EOD_COLUMNS].sort_values("date").reset_index(drop=True)


def parquet_covers_window(path: str, from_date: str, to_date: str, stale_days: int = 3) -> bool:
    """True when the cached parquet covers the requested window well enough.

    Conditions: the file starts at or before ``from_date`` and its last bar is
    at most ``stale_days`` calendar days before ``to_date`` — for a still-live
    name the window end is usually "today" and the latest trading day may be a
    day or two behind it (weekends/holidays/late vendor prints), so a small
    staleness is accepted instead of re-spending a call. A name whose series
    *ended* before ``to_date - stale_days`` (delisted inside the window) is NOT
    covered: its series is final and re-probing it would only burn a call, so
    the caller keeps the cached parquet explicitly instead.
    """
    df = wh.read_parquet(path)
    if df is None or df.empty or "date" not in df.columns:
        return False
    lo = date.fromisoformat(str(df["date"].min()))
    hi = date.fromisoformat(str(df["date"].max()))
    if lo > date.fromisoformat(from_date):
        return False
    end = date.fromisoformat(to_date)
    if hi < end - timedelta(days=stale_days):
        return False  # series ended inside the window: final, do not re-probe
    return True


def fetch_exchanges_list(client: EodhdClient, root: str, force: bool = False) -> list:
    path = os.path.join(root, "meta", "exchanges-list.json")
    if not force:
        cached = wh.read_json(path)
        if cached:
            log(f"exchanges-list: using cached ({len(cached)} exchanges, 0 calls)")
            return cached
    data = client.exchanges_list()
    wh.write_json_atomic(data, path)
    log(f"exchanges-list: fetched {len(data)} exchanges (1 call)")
    return data


def fetch_symbol_list(client: EodhdClient, root: str, exchange: str,
                      delisted: bool, force: bool = False) -> list:
    kind = "delisted" if delisted else "active"
    path = os.path.join(root, "meta", "symbol-list", f"{exchange.upper()}-{kind}.json")
    if not force:
        cached = wh.read_json(path)
        if cached is not None:
            if delisted and isinstance(cached, dict) and cached.get("available_on_free_tier") is False:
                if client.plan == "free":
                    log(f"symbol-list {exchange} (delisted): free-tier gap already recorded "
                        f"(HTTP {cached.get('http_status')}) — not re-probed, 0 calls")
                    return []
                log(f"symbol-list {exchange} (delisted): ignoring prior free-tier gap marker "
                    f"under plan={client.plan!r}; fetching delisted list")
            if isinstance(cached, list):
                log(f"symbol-list {exchange} ({kind}): using cached ({len(cached)} names, 0 calls)")
                return cached
    try:
        data = client.exchange_symbol_list(exchange, delisted=delisted)
    except DeepHistoryBlocked:
        raise
    except EodhdError as e:
        if delisted and e.status in (403, 429) and client.plan == "free":
            # EXPECTED on the free tier: delisted lists are not included.
            # Record the gap explicitly and fall back to the active list.
            log(f"symbol-list {exchange} (delisted): HTTP {e.status} — not available on the "
                "free tier (expected). Recording gap; using active list only.")
            wh.write_json_atomic(
                {"exchange": exchange.upper(), "delisted": True, "available_on_free_tier": False,
                 "http_status": e.status,
                 "note": "free tier = 20 calls/day + past-1y EOD + live only; delisted lists "
                         "need a paid plan (v3 scope doc section 4.4). Survivorship-free "
                         "ingest is therefore limited to the 1-year active window on free tier."},
                path)
            return []
        raise
    wh.write_json_atomic(data, path)
    log(f"symbol-list {exchange} ({kind}): fetched {len(data)} names (1 call)")
    return data


def ingest_ticker(client: EodhdClient, root: str, exchange: str, symbol: str,
                  from_date: str, to_date: str, force: bool = False) -> str:
    """One call per ticker max. Returns 'fetched' | 'cached' | 'empty'.

    A ticker whose parquet is missing or empty is refetched (the vendor may
    simply have had no bars that day); a covered parquet is skipped with 0
    calls, which is what makes reruns resumable within the 20/day budget.
    """
    path = wh.eod_parquet_path(root, exchange, symbol)
    if not force and parquet_covers_window(path, from_date, to_date):
        log(f"{symbol}: cached parquet covers [{from_date}..{to_date}] — skipping (0 calls)")
        return "cached"
    existing = wh.read_parquet(path)
    if existing is not None and not existing.empty and not force:
        # series ended inside the window (delisted/halted): its parquet is
        # final — keep it without spending a call
        log(f"{symbol}: series ended {existing['date'].max()} (before window end) — "
            f"keeping cached {len(existing)} bars (0 calls)")
        return "cached"
    rows = client.eod(f"{symbol}.{exchange.upper()}", from_date=from_date, to_date=to_date)
    df = _bars_to_frame(rows, f"{symbol}.{exchange.upper()}", exchange.upper())
    if df.empty and existing is not None and not existing.empty:
        # vendor returned nothing new (e.g. non-trading day at window end):
        # keep the richer cached series instead of overwriting with empty
        log(f"{symbol}: empty response, keeping cached {len(existing)} bars (1 call)")
        return "cached"
    wh.write_parquet_atomic(df, path)
    if df.empty:
        log(f"{symbol}: no rows in window (empty series, e.g. renamed/delisted) — 1 call")
        return "empty"
    log(f"{symbol}: wrote {len(df)} bars [{df['date'].min()}..{df['date'].max()}] (1 call)")
    return "fetched"


def refresh_live(client: EodhdClient, root: str, exchange: str, symbol: str) -> str:
    """Live quote snapshot (free tier includes /real-time). 1 call."""
    q = client.realtime(f"{symbol}.{exchange.upper()}")
    path = wh.live_parquet_path(root, f"{symbol}.{exchange.upper()}")
    df = pd.DataFrame([{
        "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": f"{symbol}.{exchange.upper()}",
        "date": str(q.get("date", ""))[:10],
        "open": q.get("open"), "high": q.get("high"), "low": q.get("low"),
        "close": q.get("close"), "volume": q.get("volume"),
        "previous_close": q.get("previousClose"), "change": q.get("change"),
        "change_pct": q.get("change_p"), "timestamp": q.get("timestamp"),
    }])
    wh.write_parquet_atomic(df, path)
    log(f"{symbol}: live snapshot close={q.get('close')} ({path})")
    return "live"


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--warehouse", default=DEFAULT_WAREHOUSE,
                    help="warehouse root (default: <repo>/data/warehouse)")
    ap.add_argument("--exchange", default="US", help="exchange code for stocks (default US)")
    ap.add_argument("--tickers", nargs="*", default=None,
                    help="specific symbols (without .US suffix), e.g. AAPL MSFT; "
                         "default: all active (and, on paid plan, delisted) common stocks "
                         "from the cached/fetched symbol lists")
    ap.add_argument("--plan", choices=["free", "paid"], default="free",
                    help="EODHD plan the key is entitled to: 'free' = 20 calls/day, past-1y EOD "
                         "only; 'paid' = EOD Historical Data — All World (100k calls/day, 30+ "
                         "yrs, delisted lists, bulk). The call-budget hard cap follows the plan "
                         "(default: paid -> 99999, free -> 19)")
    ap.add_argument("--max-calls", type=int, default=None,
                    help="hard API-call guard for this run (default: plan limit - 1)")
    ap.add_argument("--from", dest="from_date", default=None,
                    help="window start YYYY-MM-DD (default: today-365d on free plan, "
                         "2000-01-01 on paid plan — delisted coverage starts 2000)")
    ap.add_argument("--to", dest="to_date", default=None, help="window end (default: today)")
    ap.add_argument("--refresh-live", action="store_true",
                    help="also fetch the live real-time quote per ticker (1 extra call each)")
    ap.add_argument("--exchanges-list-only", action="store_true",
                    help="fetch exchanges-list once and exit (1 call)")
    ap.add_argument("--force", action="store_true", help="ignore parquet/cache coverage, refetch")
    ap.add_argument("--api-token", default=None,
                    help="EODHD API token (else $EODHD_TOKEN, else `pass eodhd/api-token`); never printed")
    ap.add_argument("--probe-deep-history", metavar="SYMBOL", default=None,
                    help="explicitly probe a pre-1-year window for SYMBOL to demonstrate the "
                         "free-tier limit; raises DeepHistoryBlocked (expected) — never bypassed")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.warehouse)
    os.makedirs(root, exist_ok=True)
    from_date, to_date = args.from_date, args.to_date
    if not from_date or not to_date:
        d_from, d_to = default_window(plan=args.plan)
        from_date = from_date or d_from
        to_date = to_date or d_to

    try:
        client = EodhdClient(
            token=args.api_token or os.environ.get("SIFT_TEST_TOKEN", ""),
            max_calls=args.max_calls,
            state_file=os.path.join(root, "state", "calls.json"),
            session=_SESSION,  # tests inject a FakeSession here
            plan=args.plan,
        )
    except (EodhdError, ValueError) as e:
        log(f"ERROR: {e}")
        return 2

    log(f"budget: {client.calls_made} already used today, max {client.max_calls} "
        f"({args.plan}-plan hard limit {PLAN_DAILY_CALLS[args.plan]}/day); "
        f"window {from_date}..{to_date}")

    if args.probe_deep_history:
        # deliberate, single, explicit deep-history probe — expected to fail on free tier
        sym = args.probe_deep_history
        try:
            client.eod(f"{sym}.{args.exchange.upper()}",
                       from_date="1990-01-01", to_date="1990-12-31", deep_history=True)
            log(f"deep-history probe for {sym}: unexpectedly returned data")
            return 0
        except DeepHistoryBlocked as e:
            log(f"deep-history probe for {sym}: {e}")
            return 3
        except CallBudgetExceeded as e:
            log(f"deep-history probe skipped: {e}")
            return 4

    try:
        fetch_exchanges_list(client, root, force=args.force)
        if args.exchanges_list_only:
            rec = wh.write_vintage(root, extra={"built_by": "ingest_eodhd --exchanges-list-only",
                                               "plan": args.plan,
                                               "window": [from_date, to_date]})
            log(f"vintage: {rec['vintage'][:16]}…")
            return 0

        symbols = args.tickers
        if symbols is None:
            active = fetch_symbol_list(client, root, args.exchange, delisted=False, force=args.force)
            try:
                delisted = fetch_symbol_list(client, root, args.exchange, delisted=True,
                                             force=args.force)
            except CallBudgetExceeded:
                if args.plan != "free":
                    raise
                # budget ran out before the delisted probe — it 403s on the free
                # tier anyway; record the gap without spending more calls
                log("symbol-list (delisted): skipped, call budget exhausted")
                wh.write_json_atomic(
                    {"exchange": args.exchange.upper(), "delisted": True,
                     "available_on_free_tier": False, "http_status": None,
                     "note": "not probed this run (call budget); free tier does not include "
                             "delisted lists — expected 403."},
                    os.path.join(root, "meta", "symbol-list",
                                 f"{args.exchange.upper()}-delisted.json"))
                delisted = []
            common = {r.get("Code") for r in active
                      if r.get("Type") in (None, "Common Stock", "Stock") and r.get("Code")}
            delisted_common = {r.get("Code") for r in delisted
                               if r.get("Type") in (None, "Common Stock", "Stock") and r.get("Code")}
            symbols = sorted(common | delisted_common)
            log(f"universe candidates: {len(common)} active + {len(delisted_common)} delisted "
                f"({len(symbols)} unique) common stocks on {args.exchange}")
        else:
            symbols = list(symbols)

        results = {"fetched": 0, "cached": 0, "empty": 0, "live": 0, "skipped_budget": 0}
        for sym in symbols:
            try:
                r = ingest_ticker(client, root, args.exchange, sym, from_date, to_date,
                                  force=args.force)
                results[r] = results.get(r, 0) + 1
                if args.refresh_live:
                    refresh_live(client, root, args.exchange, sym)
                    results["live"] += 1
            except CallBudgetExceeded as e:
                log(f"STOP: {e}")
                results["skipped_budget"] = len(symbols) - sum(
                    v for k, v in results.items() if k != "skipped_budget")
                break
            except EodhdError as e:
                log(f"{sym}: API error: {e} (continuing)")

        rec = wh.write_vintage(root, extra={
            "built_by": "ingest_eodhd",
            "plan": args.plan,
            "window": [from_date, to_date],
            "exchange": args.exchange.upper(),
            "tickers_requested": len(symbols),
            "results": results,
            "calls_used_this_run": client.calls_made,
        })
        log(f"done: {results} | calls used: {client.calls_made}/{client.max_calls} | "
            f"vintage {rec['vintage'][:16]}…")
        return 0
    except CallBudgetExceeded as e:
        log(f"ERROR: {e}")
        return 4
    except DeepHistoryBlocked as e:
        log(f"ERROR (expected on free tier): {e}")
        return 3
    except EodhdError as e:
        log(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
