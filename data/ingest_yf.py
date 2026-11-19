#!/usr/bin/env python3
"""SIFT research-grade yfinance (Yahoo) ingestion — RESEARCH ONLY, server-side.

Second data source alongside EODHD for INITIAL TESTING with ~5 years of
daily history. EODHD stays the production vendor (docs/vendor-licensing.md,
pull/26): Yahoo is not ToS-authorized for production redistribution, has no
delisted lists (survivorship-bias risk, v3 §4.5/§12.8), no SLA.

Mirrors data/ingest_eodhd.py's CLI shape and warehouse layout:

  * one parquet per ticker under ``<warehouse>/eod/<EX>/<SYMBOL>.parquet``
    (same schema as the EODHD path — warehouse.EOD_COLUMNS), so downstream
    code (universe builder, engine) is vendor-agnostic;
  * resumable — a ticker whose cached parquet already covers the requested
    window is skipped without any network call;
  * writes the data-vintage content hash (data/warehouse.py::write_vintage)
    with ``vendor: "yfinance"`` + ``research_grade: true`` + a research
    scope id (e.g. ``research_5y_top1k``) into vintage.json, so trials label
    the scope honestly — never free_1y_top1k / pro_full;
  * rate-limit friendly — Yahoo 429s: the client retries with exponential
    backoff + jitter (data/yf_client.py) and ``--sleep`` seconds pause
    between tickers; a persistently throttled ticker is recorded as an
    error and the run continues (resumable on the next run).

Seed list (active US names only — yfinance has NO delisted list, so the
research universe is ACTIVE-ONLY and survivorship-biased by construction;
data/universe.py flags research vintages accordingly):
  * ``--tickers AAPL MSFT ...`` explicit list, or
  * ``--seed-list sec`` (default): SEC EDGAR company_tickers_exchange.json
    (active US listings, fetched at runtime, never committed), or
  * ``--seed-list sp500``: Wikipedia S&P 500 constituents table, or
  * ``--seed-list file:<path>``: a committed-or-local newline list.

v3 scope doc hard rules (§12): server-side only; raw data never leaves the
server (the warehouse is gitignored); no tokens needed for Yahoo; nothing
secret is printed or committed; SIFT does not trade.

Usage:
    python3 -m data.ingest_yf --warehouse data/warehouse --period 5y            # full seed; universe.py cuts top-N by dollar volume
    python3 -m data.ingest_yf --tickers AAPL MSFT --period 5y --sleep 2
    python3 -m data.ingest_yf --seed-list sp500 --period 5y --max-ingest 500 --sleep 1.5

NOTE on universe size: --top is DEPRECATED (it truncated the alphabetically
sorted seed list — the A/B-only universe of hermes-tasks #182). Ingest the
full seed (or a seeded-random --max-ingest sample) and let data/universe.py
rank by point-in-time average dollar volume and cut top-N.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import date, datetime, timedelta, timezone

import pandas as pd

# allow running both as a module (python3 -m data.ingest_yf) and as a script
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data import warehouse as wh  # type: ignore
    from data.security_type import is_common_stock_symbol  # type: ignore
    from data.yf_client import (  # type: ignore
        DEFAULT_PERIOD, RESEARCH_GRADE, VENDOR, VALID_PERIODS, YfClient, YfError,
        YfRateLimited, history_to_bars_frame,
    )
else:
    from . import warehouse as wh
    from .security_type import is_common_stock_symbol
    from .yf_client import (
        DEFAULT_PERIOD, RESEARCH_GRADE, VENDOR, VALID_PERIODS, YfClient, YfError,
        YfRateLimited, history_to_bars_frame,
    )

# Default seed for --max-ingest shuffling: fixed so a run is reproducible
# (same seed list + same --max-ingest -> same candidate pool every time).
CAP_RNG_SEED = 20261119

DEFAULT_WAREHOUSE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "data", "warehouse")

# tests monkeypatch this to inject a fake fetcher (no network)
_FETCHER = None

SEC_TICKERS_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "sift-research contact@sift.local")
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def log(msg: str) -> None:
    print(f"[ingest_yf {datetime.now(timezone.utc).strftime('%H:%M:%S')}Z] {msg}", flush=True)


def _kfmt(n: int) -> str:
    return f"{n//1000}k" if n >= 1000 else str(n)


def scope_id_for(source: str, period: str, top_n: int | None) -> str:
    """Honest research scope id derived from source+window, e.g.
    ``research_5y_top500`` / ``research_5y`` — the ``research_`` prefix marks
    the vendor as research-grade (yfinance/Yahoo) and guarantees the id can
    never collide with a production scope (``free_1y_top1k``/``pro_full``).
    Top-N is rendered in the repo's ``top1k`` style (1000 -> ``1k``).
    """
    if source not in ("yf", "yfinance"):
        raise ValueError(f"unknown research source {source!r}")
    tag = period
    if top_n:
        tag += f"_top{_kfmt(top_n)}"
    return f"research_{tag}"


def period_to_days(period: str) -> int:
    """Approximate window length in calendar days for a yfinance period tag
    (used for parquet coverage checks)."""
    table = {"1d": 1, "5d": 5, "1mo": 31, "3mo": 92, "6mo": 183,
             "1y": 366, "2y": 731, "5y": 5 * 366, "10y": 10 * 366,
             "ytd": 366, "max": 40 * 366}
    return table[period]


def default_window(period: str, today: date | None = None) -> tuple[str, str]:
    today = today or date.today()
    return ((today - timedelta(days=period_to_days(period))).isoformat(), today.isoformat())


# --------------------------------------------------------------------------
# seed lists (active names only — no delisted list exists for yfinance)
# --------------------------------------------------------------------------
def _fetch_json(url: str, user_agent: str, timeout: int = 30):
    import gzip
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": user_agent,
                                                "Accept-Encoding": "gzip, deflate"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def filter_common_stock(symbols: list[str]) -> list[str]:
    """Drop warrants/units/rights/preferred from a seed list (symbol-pattern
    filter — Yahoo has no security-type field; mirrors the EODHD ingest's
    Type='Common Stock' filter, data/ingest_eodhd.py). Returns the kept
    symbols in their original order."""
    return [s for s in symbols if is_common_stock_symbol(s)]


def alphabet_sanity_ok(symbols: list[str],
                       min_letters: int = 10, max_single_letter_frac: float = 0.5) -> bool:
    """Guard against alphabetically truncated / degenerate candidate pools
    (the hermes-tasks #182 root cause: ``sorted(seed)[:top]`` produced an
    A-only pool). A healthy full US seed spans ~26 letters with no letter
    above ~50%. Returns False when the pool is degenerate."""
    if not symbols:
        return False
    letters = [s[0].upper() for s in symbols if s]
    if len(set(letters)) < min_letters:
        return False
    top_letter = max(letters.count(c) for c in set(letters))
    return top_letter / len(letters) <= max_single_letter_frac


def seed_list_sec() -> list[str]:
    """Active US listings from SEC EDGAR company_tickers_exchange.json
    (Nasdaq + NYSE + NYSE American/Arca), restricted to COMMON STOCK symbols
    (warrants/units/rights/preferred excluded by pattern — the SEC seed has
    no type field, so data/security_type.py classifies). Fetched at runtime,
    never committed."""
    d = _fetch_json(SEC_TICKERS_EXCHANGE_URL, SEC_USER_AGENT)
    data = d["data"] if isinstance(d, dict) and "data" in d else d
    wanted = {"nasdaq", "nyse", "nyse american", "nyse arca", "nyse mkt", "amex"}
    syms = sorted({row[2] for row in data if str(row[3]).lower() in wanted})
    common = filter_common_stock(syms)
    log(f"seed-list sec: {len(syms)} active US listings from SEC EDGAR, "
        f"{len(syms) - len(common)} non-common (warrant/unit/right/preferred) excluded, "
        f"{len(common)} common-stock candidates")
    return common


def seed_list_sp500() -> list[str]:
    """Current S&P 500 constituents from the Wikipedia table (one fetch).
    Research seed only — index membership changes; not a PIT constituent list."""
    import re
    import urllib.request
    req = urllib.request.Request(SP500_WIKI_URL, headers={"User-Agent": SEC_USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        html = r.read().decode("utf-8", "replace")
    # first wikitable on the page is the constituents table; pull the Symbol column
    m = re.search(r'<table[^>]*class="[^"]*wikitable[^"]*"[^>]*>(.*?)</table>', html, re.S)
    if not m:
        raise YfError("seed-list sp500: could not find the constituents table on Wikipedia")
    rows = re.findall(r"<tr>(.*?)</tr>", m.group(1), re.S)
    syms = []
    for row in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)
        if not cells:
            continue
        sym = re.sub(r"<[^>]+>", "", cells[0]).strip()
        if re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", sym or ""):
            syms.append(sym.replace(".", "-"))  # Yahoo convention: BRK.B -> BRK-B
    syms = sorted(set(syms))
    common = filter_common_stock(syms)
    log(f"seed-list sp500: {len(common)} constituents from Wikipedia"
        + (f" ({len(syms) - len(common)} non-common excluded)" if len(common) != len(syms) else ""))
    return common


def load_seed_list(kind: str) -> list[str]:
    if kind == "sec":
        return seed_list_sec()
    if kind == "sp500":
        return seed_list_sp500()
    if kind.startswith("file:"):
        path = kind.split(":", 1)[1]
        with open(path, "r", encoding="utf-8") as f:
            syms = sorted({ln.strip().upper() for ln in f if ln.strip() and not ln.startswith("#")})
        common = filter_common_stock(syms)
        if len(common) != len(syms):
            log(f"seed-list file:{path}: {len(syms)} symbols, "
                f"{len(syms) - len(common)} non-common excluded, {len(common)} kept")
        else:
            log(f"seed-list file:{path}: {len(syms)} symbols")
        return common
    raise YfError(f"unknown --seed-list {kind!r} (sec | sp500 | file:<path>)")


# --------------------------------------------------------------------------
# ingestion
# --------------------------------------------------------------------------
def parquet_covers_window(path: str, from_date: str, to_date: str, stale_days: int = 3) -> bool:
    """Same semantics as ingest_eodhd.parquet_covers_window: cached parquet
    starts at/before from_date and its last bar is within stale_days of
    to_date (or the series is final — ended inside the window)."""
    df = wh.read_parquet(path)
    if df is None or df.empty or "date" not in df.columns:
        return False
    lo = date.fromisoformat(str(df["date"].min()))
    hi = date.fromisoformat(str(df["date"].max()))
    if lo > date.fromisoformat(from_date):
        return False
    end = date.fromisoformat(to_date)
    if hi < end - timedelta(days=stale_days):
        return False
    return True


def ingest_ticker(client: YfClient, root: str, exchange: str, symbol: str,
                  from_date: str, to_date: str, period: str, force: bool = False) -> str:
    """Fetch one ticker (one history() call max per run). Returns
    'fetched' | 'cached' | 'empty' | 'error'."""
    path = wh.eod_parquet_path(root, exchange, symbol)
    if not force and parquet_covers_window(path, from_date, to_date):
        log(f"{symbol}: cached parquet covers [{from_date}..{to_date}] — skipping (0 calls)")
        return "cached"
    try:
        df = client.history(symbol, period=period)
    except YfRateLimited as e:
        log(f"{symbol}: RATE LIMITED after retries — recorded, continuing ({e})")
        return "error"
    except YfError as e:
        log(f"{symbol}: {type(e).__name__}: {e} (continuing)")
        return "error"
    bars = history_to_bars_frame(df, symbol, exchange)
    # clip to the requested window (period tags are approximate)
    if not bars.empty:
        bars = bars[(bars["date"] >= date.fromisoformat(from_date)) &
                    (bars["date"] <= date.fromisoformat(to_date))].reset_index(drop=True)
    if bars.empty:
        log(f"{symbol}: no bars in window [{from_date}..{to_date}] (1 call)")
        return "empty"
    wh.write_parquet_atomic(bars, path)
    log(f"{symbol}: wrote {len(bars)} bars [{bars['date'].min()}..{bars['date'].max()}] (1 call)")
    return "fetched"


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--warehouse", default=DEFAULT_WAREHOUSE,
                    help="warehouse root (default: <repo>/data/warehouse)")
    ap.add_argument("--exchange", default="US", help="exchange code for stocks (default US)")
    ap.add_argument("--tickers", nargs="*", default=None,
                    help="specific symbols (Yahoo style, e.g. AAPL MSFT BRK-B); "
                         "default: --seed-list filtered to --top N")
    ap.add_argument("--seed-list", default="sec", choices=["sec", "sp500"],
                    help="active-name seed list when --tickers is not given: "
                         "'sec' (SEC EDGAR active US listings) or 'sp500' (Wikipedia); "
                         "also accepts file:<path> via --seed-list-file")
    ap.add_argument("--seed-list-file", default=None,
                    help="path to a newline-separated symbol list (overrides --seed-list)")
    ap.add_argument("--top", type=int, default=None,
                    help="DEPRECATED: never truncated the seed alphabetically safely — "
                         "the seed list is alphabetically sorted, so the old "
                         "'symbols[:top]' kept only the first N symbols A-Z (an "
                         "A/B-only universe; hermes-tasks #182). Ignored for selection "
                         "(accepted with a warning); top-N selection is by POINT-IN-TIME "
                         "DOLLAR VOLUME in data/universe.py. Use --max-ingest for a "
                         "cost cap.")
    ap.add_argument("--max-ingest", type=int, default=None,
                    help="cost cap: ingest a seeded-RANDOM sample of at most N seed "
                         "symbols (default: all). The sample is reproducible "
                         "(fixed seed) and alphabetically unbiased — never the "
                         "alphabetical head.")
    ap.add_argument("--cap-seed", type=int, default=CAP_RNG_SEED,
                    help=f"RNG seed for --max-ingest sampling (default {CAP_RNG_SEED})")
    ap.add_argument("--period", default=DEFAULT_PERIOD, choices=VALID_PERIODS,
                    help=f"yfinance history period (default {DEFAULT_PERIOD})")
    ap.add_argument("--from", dest="from_date", default=None,
                    help="window start YYYY-MM-DD (default: today - period)")
    ap.add_argument("--to", dest="to_date", default=None, help="window end (default: today)")
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="seconds to sleep between tickers (Yahoo 429 friendliness; default 1.0)")
    ap.add_argument("--max-retries", type=int, default=4,
                    help="per-ticker retry attempts on 429/connection errors (default 4)")
    ap.add_argument("--backoff", type=float, default=2.0,
                    help="base backoff seconds, exponential + jitter (default 2.0)")
    ap.add_argument("--force", action="store_true", help="ignore parquet coverage, refetch")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.warehouse)
    os.makedirs(root, exist_ok=True)
    from_date, to_date = args.from_date, args.to_date
    if not from_date or not to_date:
        d_from, d_to = default_window(args.period)
        from_date = from_date or d_from
        to_date = to_date or d_to

    symbols = args.tickers
    if symbols is None:
        kind = f"file:{args.seed_list_file}" if args.seed_list_file else args.seed_list
        try:
            symbols = load_seed_list(kind)
        except YfError as e:
            log(f"ERROR: {e}")
            return 2
        if not alphabet_sanity_ok(symbols):
            log("ERROR: candidate seed pool is degenerate — first-letter distribution "
                "spans < 10 letters or one letter > 50% (alphabetical-truncation "
                "signature, hermes-tasks #182). Refusing to ingest a biased pool; "
                "check the seed list.")
            return 2
        if args.top:
            log("WARNING: --top is DEPRECATED and ignored for selection — the seed "
                "list is alphabetically sorted, so 'first N' would keep only A-Z "
                "head symbols (the alphabetical-truncation bias of hermes-tasks "
                "#182). Ingesting the FULL seed; top-N selection happens by "
                "point-in-time dollar volume in data/universe.py. For a cost cap "
                "use --max-ingest (seeded random sample, alphabetically unbiased).")
        if args.max_ingest and args.max_ingest < len(symbols):
            n_seed = len(symbols)
            rng = random.Random(args.cap_seed)
            symbols = sorted(rng.sample(symbols, args.max_ingest))
            log(f"--max-ingest {args.max_ingest}: seeded-random sample of "
                f"{args.max_ingest}/{n_seed} seed symbols "
                f"(seed={args.cap_seed}) — alphabetically unbiased cost cap")
    else:
        symbols = [s.upper() for s in symbols]

    scope_id = scope_id_for("yf", args.period, None)
    log(f"RESEARCH-ONLY feed: vendor={VENDOR} research_grade={RESEARCH_GRADE} "
        f"scope_id={scope_id} period={args.period} window=[{from_date}..{to_date}] "
        f"tickers={len(symbols)} sleep={args.sleep}s")

    client = YfClient(fetcher=_FETCHER, max_retries=args.max_retries, backoff_s=args.backoff)

    results = {"fetched": 0, "cached": 0, "empty": 0, "error": 0}
    errors = {}
    for i, sym in enumerate(symbols):
        if i > 0 and args.sleep > 0:
            client.sleeper(args.sleep)
        r = ingest_ticker(client, root, args.exchange, sym, from_date, to_date,
                          args.period, force=args.force)
        results[r] = results.get(r, 0) + 1
        if r == "error":
            errors[sym] = "rate-limited/empty-after-retries"

    rec = wh.write_vintage(root, extra={
        "built_by": "ingest_yf",
        "vendor": VENDOR,
        "research_grade": RESEARCH_GRADE,
        "scope_id": scope_id,
        "period": args.period,
        "window": [from_date, to_date],
        "exchange": args.exchange.upper(),
        "tickers_requested": len(symbols),
        "results": results,
        "errors": errors,
        "note": "yfinance/Yahoo RESEARCH feed — not ToS-authorized for production "
                "redistribution, no delisted lists (active-only, survivorship-biased), "
                "no SLA. EODHD remains the production vendor (docs/vendor-licensing.md).",
    })
    log(f"done: {results} | calls {client.calls_made} retries {client.retries} | "
        f"vintage {rec['vintage'][:16]}… scope={scope_id} research_grade=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
