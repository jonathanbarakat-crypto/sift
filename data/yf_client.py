"""yfinance (Yahoo Finance) research feed for the SIFT data pipeline.

RESEARCH-ONLY — never the production feed. EODHD remains the production
vendor (see docs/vendor-licensing.md, pull/26): Yahoo's Terms of Service do
not authorize redistribution of Yahoo data in a commercial product, and
yfinance scraping has NO SLA, NO delisted-name lists (survivorship-bias
risk, v3 §4.5/§12.8), and no support. Every vintage written through this
module is marked ``research_grade: true`` + ``vendor: yfinance`` in
``vintage.json`` so downstream trials label the scope honestly
(e.g. scope id ``research_5y_top1k`` — never ``free_1y_top1k``/``pro_full``).

Verified from the sandbox 2026-11-17: ``yf.Ticker('AAPL').history(period='5y')``
returns 1255 daily bars (2021-11-17..2026-11-17) with
Open/High/Low/Close/Adj Close/Volume/Dividends/Stock Splits.

Rate-limit friendliness: Yahoo answers 429 (and intermittently drops the
connection) under load. This client therefore
  * retries 429/5xx/connection errors with exponential backoff + full jitter
    (``backoff_s`` base, ``max_retries`` attempts), and
  * treats any persistent failure as :class:`YfRateLimited` after
    ``max_retries`` — the caller (ingest_yf) then sleeps ``--sleep`` seconds
    and continues with the next ticker, so one throttled name never kills a
    batch run.

No credentials are needed (Yahoo's chart API is keyless); nothing secret is
read, printed, or committed. The HTTP layer is injectable (``fetcher``) so
tests never touch the network.

v3 scope doc §12: server-side only — this module writes parquet to the
gitignored server-side warehouse; raw Yahoo data is never served to clients.
"""
from __future__ import annotations

import random
import time
from typing import Any, Callable, Optional

import pandas as pd

VENDOR = "yfinance"
RESEARCH_GRADE = True

# Yahoo history() period strings we accept (mirrors yfinance's own vocabulary)
VALID_PERIODS = ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
DEFAULT_PERIOD = "5y"


class YfError(RuntimeError):
    """Any yfinance/Yahoo failure."""


class YfRateLimited(YfError):
    """Yahoo throttled us (HTTP 429) or the connection failed persistently.

    Raised after ``max_retries`` backoff attempts. Callers should back off
    further (``--sleep``) and continue with the next ticker.
    """


class YfEmpty(YfError):
    """The ticker returned no bars for the requested window (e.g. halted,
    very recently listed, or an unknown symbol)."""


def _default_fetcher(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Real network path: yfinance Ticker.history. Imported lazily so the
    module (and its tests) import cleanly without yfinance installed."""
    import yfinance as yf  # lazy: keeps unit tests network-free and dep-light

    return yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)


class YfClient:
    """Thin yfinance wrapper with retry/backoff + jitter around history().

    Parameters mirror the EODHD client's shape where it makes sense:
    ``fetcher`` is injectable for tests (a callable
    ``(symbol, period, interval) -> DataFrame``); ``sleeper`` is injectable
    so backoff tests don't actually sleep; ``rng`` seeds the jitter.
    """

    def __init__(
        self,
        fetcher: Optional[Callable[[str, str, str], pd.DataFrame]] = None,
        sleeper: Callable[[float], None] = time.sleep,
        rng: Optional[random.Random] = None,
        max_retries: int = 4,
        backoff_s: float = 2.0,
        jitter: float = 1.0,
        interval: str = "1d",
    ):
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.fetcher = fetcher or _default_fetcher
        self.sleeper = sleeper
        self.rng = rng or random.Random()
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.jitter = jitter
        self.interval = interval
        self.calls_made = 0
        self.retries = 0

    # ------------------------------------------------------------------ retry
    def backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with full jitter: base * 2^attempt * U(1-j, 1+j).

        Pure function of ``attempt`` (0-based) — unit-tested directly.
        """
        base = self.backoff_s * (2 ** attempt)
        lo = max(0.0, 1.0 - self.jitter)
        hi = 1.0 + self.jitter
        return base * self.rng.uniform(lo, hi)

    def _is_retryable(self, exc: BaseException) -> bool:
        """429/5xx/connection-style failures are retryable; everything else
        (programming errors, bad args) is not."""
        name = type(exc).__name__.lower()
        text = str(exc).lower()
        if "429" in text or "too many requests" in text:
            return True
        if any(k in name for k in ("connection", "timeout", "chunked", "remote")):
            return True
        if any(k in text for k in ("connectionerror", "timed out", "connection reset",
                                    "chunkedencodingerror", "503", "502", "504", "500")):
            return True
        return False

    # ------------------------------------------------------------------ calls
    def history(self, symbol: str, period: str = DEFAULT_PERIOD) -> pd.DataFrame:
        """Fetch daily bars for ``symbol`` over ``period`` with retry/backoff.

        Raises YfRateLimited when Yahoo keeps throttling (after
        ``max_retries`` retries) and YfEmpty when the window has no bars.
        """
        if period not in VALID_PERIODS:
            raise YfError(
                f"invalid period {period!r}; expected one of {VALID_PERIODS} "
                "(yfinance vocabulary)"
            )
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            try:
                df = self.fetcher(symbol, period, self.interval)
                self.calls_made += 1
                if df is None or len(df) == 0:
                    raise YfEmpty(
                        f"{symbol}: no bars for period={period} (halted, recently "
                        "listed, or unknown symbol)"
                    )
                return df
            except YfEmpty:
                raise
            except Exception as e:  # noqa: BLE001 — classify, then re-raise
                last_exc = e
                if not self._is_retryable(e) or attempt >= self.max_retries:
                    break
                self.retries += 1
                self.sleeper(self.backoff_delay(attempt))
        raise YfRateLimited(
            f"{symbol}: Yahoo rate-limited/connection failed after "
            f"{self.max_retries + 1} attempts (last: {type(last_exc).__name__}: "
            f"{str(last_exc)[:200]}). Back off (--sleep) and retry later."
        ) from last_exc

    def download_many(self, symbols, period: str = DEFAULT_PERIOD,
                      sleep_s: float = 0.0) -> dict:
        """Fetch many tickers sequentially with an optional inter-ticker sleep.

        Returns ``{symbol: df}`` for successes; failures are collected in the
        returned dict under ``"_errors"`` as ``{symbol: error-message}`` so a
        batch run is resumable and never aborts on one bad name.
        """
        out: dict[str, Any] = {"_errors": {}}
        for i, sym in enumerate(symbols):
            if i > 0 and sleep_s > 0:
                self.sleeper(sleep_s)
            try:
                out[sym] = self.history(sym, period=period)
            except YfError as e:
                out["_errors"][sym] = str(e)
        return out


# ---------------------------------------------------------------------- frame
def history_to_bars_frame(df: pd.DataFrame, symbol: str, exchange: str = "US") -> pd.DataFrame:
    """Normalize a yfinance history() frame into the warehouse EOD schema
    (``warehouse.EOD_COLUMNS``): date/open/high/low/close/adjusted_close/
    volume/dollar_volume/symbol/exchange.

    Yahoo columns: Open/High/Low/Close/Adj Close/Volume (+Dividends/Splits,
    which are dropped — the warehouse schema is price+volume only, matching
    the EODHD ingest path so downstream code is vendor-agnostic).
    adjusted_close falls back to close when Yahoo omits Adj Close.
    """
    from data import warehouse as wh  # local import keeps the module standalone

    if df is None or len(df) == 0:
        return pd.DataFrame(columns=wh.EOD_COLUMNS)
    out = pd.DataFrame()
    idx = pd.to_datetime(df.index)
    # normalize to naive dates (Yahoo returns exchange-local tz-aware
    # timestamps; the warehouse stores plain dates like the EODHD path).
    # tz-naive input passes through unchanged; tz-aware is converted to UTC
    # then stripped (Yahoo daily bars are stamped at exchange midnight, so
    # the UTC date equals the trading date for US listings).
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    out["date"] = idx.normalize().date
    out["open"] = pd.to_numeric(df["Open"], errors="coerce").to_numpy()
    out["high"] = pd.to_numeric(df["High"], errors="coerce").to_numpy()
    out["low"] = pd.to_numeric(df["Low"], errors="coerce").to_numpy()
    out["close"] = pd.to_numeric(df["Close"], errors="coerce").to_numpy()
    if "Adj Close" in df.columns:
        out["adjusted_close"] = pd.to_numeric(df["Adj Close"], errors="coerce").to_numpy()
    else:
        out["adjusted_close"] = out["close"]
    out["adjusted_close"] = pd.to_numeric(out["adjusted_close"], errors="coerce").fillna(out["close"])
    out["volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0).to_numpy()
    out["dollar_volume"] = out["close"] * out["volume"]
    out["symbol"] = f"{symbol}.{exchange.upper()}"
    out["exchange"] = exchange.upper()
    out = out.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    return out[wh.EOD_COLUMNS]
