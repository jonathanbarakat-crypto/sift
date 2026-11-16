"""Minimal EODHD API client for the SIFT data pipeline.

Free tier (https://eodhd.com/help/plans, probed 2026-11-15, sift-research
docs/probe-plan.md S14/S15/S36):
  * 20 API calls per day (+ 500 welcome calls on a new account)
  * daily EOD history limited to the past 1 year
  * live / real-time quotes included
  * NO deep history, NO delisted=1 symbol lists, NO bulk endpoints
    (those 403/limit on the free tier — expected, see DeepHistoryBlocked)

Token resolution order (the value is never printed, logged, or committed):
  1. explicit ``token`` argument / --api-token flag
  2. EODHD_TOKEN environment variable
  3. ``pass eodhd/api-token`` (VM password store)

Every request goes through :meth:`EodhdClient.get`, which increments a
persisted call counter; ``max_calls`` is a hard guard that raises
:class:`CallBudgetExceeded` *before* the HTTP request is made, so a run can
never silently burn more of the daily budget than allowed.
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Optional

import requests

BASE_URL = "https://eodhd.com/api"
FREE_TIER_DAILY_CALLS = 20


class EodhdError(RuntimeError):
    """Any EODHD API failure. ``status`` is the HTTP code when known."""

    def __init__(self, message: str, status: Optional[int] = None, url: Optional[str] = None):
        super().__init__(message)
        self.status = status
        # never include the api_token in error text
        self.url = url.split("api_token=")[0] + "api_token=<redacted>" if url else None


class DeepHistoryBlocked(EodhdError):
    """Raised when the free tier refuses a pre-1-year (deep-history) probe.

    This is EXPECTED on the free tier (20 calls/day, past-1-year EOD only).
    It is raised with a clear message and must not be retried or bypassed —
    deep history needs a paid plan (see v3 scope doc section 4.4 emails).
    """


class CallBudgetExceeded(EodhdError):
    """Raised before an HTTP call would exceed the configured call budget."""


def resolve_token(token: Optional[str] = None) -> str:
    """Resolve the EODHD API token without ever exposing it.

    Order: explicit arg -> $EODHD_TOKEN -> ``pass eodhd/api-token``.
    Raises EodhdError with an actionable message when unavailable.
    """
    if token:
        return token
    env = os.environ.get("EODHD_TOKEN", "").strip()
    if env:
        return env
    err = ""
    try:
        out = subprocess.run(
            ["pass", "show", "eodhd/api-token"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        out = None
        err = str(e)
    if out is not None and out.returncode == 0 and out.stdout.strip():
        return out.stdout.strip()
    raise EodhdError(
        "no EODHD API token available: pass --api-token, set $EODHD_TOKEN, or "
        "store the free-tier key at `pass insert eodhd/api-token` "
        + (f"(pass error: {err})" if out is None else "(pass eodhd/api-token: not in the password store)")
    )


class EodhdClient:
    """EODHD client with a hard, persisted API-call budget.

    ``state_file`` records how many calls were made today (UTC date) so a
    crashed/interrupted run can be resumed without re-spending calls.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        max_calls: int = FREE_TIER_DAILY_CALLS - 1,
        state_file: Optional[str] = None,
        session: Optional[requests.Session] = None,
        timeout: int = 60,
    ):
        if max_calls < 0:
            raise ValueError("max_calls must be >= 0")
        if max_calls >= FREE_TIER_DAILY_CALLS:
            # free tier hard limit is 20/day; refuse to configure above it
            raise ValueError(
                f"max_calls={max_calls} exceeds the EODHD free-tier hard limit of "
                f"{FREE_TIER_DAILY_CALLS} calls/day; lower --max-calls"
            )
        self._token = token if token else resolve_token()
        self.max_calls = max_calls
        self.state_file = state_file
        self.session = session if session is not None else requests.Session()
        self.timeout = timeout
        self.calls_made = 0
        if state_file:
            self.calls_made = self._load_state().get("calls_today", 0)

    # ------------------------------------------------------------------ state
    def _state_key_today(self) -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _load_state(self) -> dict:
        if not self.state_file or not os.path.exists(self.state_file):
            return {"date": self._state_key_today(), "calls_today": 0}
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                st = json.load(f)
        except Exception:
            st = {}
        if st.get("date") != self._state_key_today():
            st = {"date": self._state_key_today(), "calls_today": 0}
        return st

    def _save_state(self) -> None:
        if not self.state_file:
            return
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        st = {"date": self._state_key_today(), "calls_today": self.calls_made}
        tmp = self.state_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(st, f)
        os.replace(tmp, self.state_file)

    def calls_remaining(self) -> int:
        return max(0, self.max_calls - self.calls_made)

    # ------------------------------------------------------------------ calls
    def get(self, path: str, params: Optional[dict] = None, deep_history: bool = False) -> Any:
        """GET ``BASE_URL + path`` with fmt=json&api_token=... .

        Counts against the budget BEFORE the request; raises
        CallBudgetExceeded when the budget is spent. ``deep_history=True``
        marks pre-1-year probes: a 403/limit there raises DeepHistoryBlocked
        (expected on the free tier — do not bypass).
        """
        if self.calls_remaining() <= 0:
            raise CallBudgetExceeded(
                f"API call budget exhausted: {self.calls_made}/{self.max_calls} calls used "
                f"(free tier hard limit {FREE_TIER_DAILY_CALLS}/day). "
                "Re-run tomorrow or lower the workload; cached parquet files are reused "
                "without spending calls."
            )
        q = {"fmt": "json", "api_token": self._token}
        if params:
            q.update({k: v for k, v in params.items() if v is not None})
        url = BASE_URL + path
        try:
            r = self.session.get(url, params=q, timeout=self.timeout)
        except requests.RequestException as e:
            raise EodhdError(f"EODHD request failed: {type(e).__name__}: {e}", url=url) from e
        self.calls_made += 1
        self._save_state()
        if r.status_code == 403 or r.status_code == 429:
            if deep_history:
                raise DeepHistoryBlocked(
                    f"EODHD free tier refused a deep-history (pre-1-year) request "
                    f"(HTTP {r.status_code}) for {path}. Expected on the free tier "
                    "(20 calls/day, past-1-year EOD only). Deep history requires a paid "
                    "plan — see v3 scope doc section 4.4. Not retried, not bypassed.",
                    status=r.status_code, url=r.url,
                )
            raise EodhdError(
                f"EODHD HTTP {r.status_code} for {path} (limit/entitlement). "
                "On the free tier this usually means the endpoint is not included "
                "(deep history, delisted lists, bulk) — not retried.",
                status=r.status_code, url=r.url,
            )
        if r.status_code != 200:
            raise EodhdError(f"EODHD HTTP {r.status_code} for {path}: {r.text[:200]}",
                             status=r.status_code, url=r.url)
        try:
            return r.json()
        except ValueError as e:
            raise EodhdError(f"EODHD returned non-JSON for {path}: {r.text[:200]}",
                             status=r.status_code, url=r.url) from e

    # ------------------------------------------------------------- endpoints
    def exchanges_list(self) -> list:
        """GET /exchanges-list — fetch once, cache on disk (1 call)."""
        data = self.get("/exchanges-list")
        if not isinstance(data, list):
            raise EodhdError("exchanges-list returned a non-list response")
        return data

    def exchange_symbol_list(self, exchange: str, delisted: bool = False) -> list:
        """GET /exchange-symbol-list/{EXCHANGE}[?delisted=1].

        delisted=1 is NOT available on the free tier (403) — callers should
        catch EodhdError/DeepHistoryBlocked and fall back to the active list.
        """
        params = {"delisted": "1" if delisted else "0"}
        data = self.get(f"/exchange-symbol-list/{exchange.upper()}", params=params)
        if not isinstance(data, list):
            raise EodhdError(f"exchange-symbol-list/{exchange} returned a non-list response")
        return data

    def eod(self, symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None,
            period: str = "d", deep_history: bool = False) -> list:
        """GET /eod/{SYMBOL}?from=&to=&period= — one call per ticker.

        Free tier: past 1 year of daily EOD only. Pass deep_history=True for
        pre-1-year windows to get a clear DeepHistoryBlocked error.
        """
        params = {"from": from_date, "to": to_date, "period": period}
        return self.get(f"/eod/{symbol}", params=params, deep_history=deep_history)

    def realtime(self, symbol: str) -> dict:
        """GET /real-time/{SYMBOL} — live quote (included on the free tier)."""
        data = self.get(f"/real-time/{symbol}")
        if not isinstance(data, dict):
            raise EodhdError(f"real-time/{symbol} returned a non-dict response")
        return data
