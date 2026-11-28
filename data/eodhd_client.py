"""Minimal EODHD API client for the SIFT data pipeline.

Plans (https://eodhd.com/help/plans, verified 2026-11-20 against a paid
"EOD Historical Data — All World" key):
  * Free: 20 API calls/day (+ 500 welcome calls), past-1-year EOD only,
    live/delayed quotes, NO deep history, NO delisted=1 symbol lists, NO bulk.
  * Paid EOD Historical Data — All World ($19.99/mo): 100,000 calls/day,
    30+ years of EOD history per ticker, delisted tickers + lists, bulk API.

The plan is selected with ``plan=`` ("free" | "paid"); the call-budget hard
cap defaults to the plan's daily limit so a run can never silently overspend.
``deep_history=True`` probes that 403 on the free tier raise
:class:`DeepHistoryBlocked` (expected there — never retried or bypassed).

Token resolution order (the value is never printed, logged, or committed):
  1. explicit ``token`` argument / --api-token flag
  2. EODHD_TOKEN environment variable
  3. ``pass eodhd/api-token`` (host password store)

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
PAID_EOD_DAILY_CALLS = 100_000   # EOD Historical Data — All World plan (verified 2026-11-20)
PLAN_DAILY_CALLS = {"free": FREE_TIER_DAILY_CALLS, "paid": PAID_EOD_DAILY_CALLS}


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
        "store the API key at `pass insert eodhd/api-token` "
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
        max_calls: Optional[int] = None,
        state_file: Optional[str] = None,
        session: Optional[requests.Session] = None,
        timeout: int = 60,
        plan: str = "free",
    ):
        if plan not in PLAN_DAILY_CALLS:
            raise ValueError(f"unknown plan {plan!r}; expected one of {sorted(PLAN_DAILY_CALLS)}")
        self.plan = plan
        plan_limit = PLAN_DAILY_CALLS[plan]
        if max_calls is None:
            # default: just under the plan's daily hard limit
            max_calls = plan_limit - 1
        if max_calls < 0:
            raise ValueError("max_calls must be >= 0")
        if max_calls >= plan_limit:
            # hard cap = the plan's daily limit; refuse to configure above it
            raise ValueError(
                f"max_calls={max_calls} exceeds the EODHD {plan}-plan hard limit of "
                f"{plan_limit} calls/day; lower --max-calls"
            )
        self._token = token if token else resolve_token()
        self.max_calls = max_calls
        self.state_file = state_file
        self.session = session if session is not None else requests.Session()
        self.timeout = timeout
        self.calls_made = 0
        self._state_date = self._state_key_today()
        if state_file:
            st = self._load_state()
            self._state_date = str(st.get("date") or self._state_key_today())
            self.calls_made = int(st.get("calls_today", 0))

    # ------------------------------------------------------------------ state
    def _state_key_today(self) -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _rollover_if_new_utc_day(self) -> None:
        today = self._state_key_today()
        if today != self._state_date:
            self._state_date = today
            self.calls_made = 0

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
        st = {"date": self._state_date, "calls_today": self.calls_made}
        tmp = self.state_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(st, f)
        os.replace(tmp, self.state_file)

    def calls_remaining(self) -> int:
        self._rollover_if_new_utc_day()
        return max(0, self.max_calls - self.calls_made)

    # ------------------------------------------------------------------ calls
    def get(
        self,
        path: str,
        params: Optional[dict] = None,
        deep_history: bool = False,
        cost: int = 1,
    ) -> Any:
        """GET ``BASE_URL + path`` with fmt=json&api_token=... .

        Counts ``cost`` against the budget BEFORE the request (standard endpoints
        cost 1 call; EODHD bulk endpoints cost 100 calls); raises
        CallBudgetExceeded when the budget would be exceeded. ``deep_history=True``
        marks pre-1-year probes: a 403/limit there raises DeepHistoryBlocked
        (expected on the free tier — do not bypass).
        """
        call_cost = max(int(cost), 1)
        if self.calls_remaining() < call_cost:
            raise CallBudgetExceeded(
                f"API call budget exhausted: {self.calls_made}/{self.max_calls} calls used "
                f"(need {call_cost}; {self.plan}-plan hard limit {PLAN_DAILY_CALLS[self.plan]}/day). "
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
        self.calls_made += call_cost
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
            if self.plan == "free":
                hint = (
                    "On the free tier this usually means the endpoint is not included "
                    "(deep history, delisted lists, bulk) — not retried."
                )
            else:
                hint = (
                    f"On the {self.plan} plan this usually means the API key lacks entitlement "
                    "for this endpoint or the daily limit was reached — not retried."
                )
            raise EodhdError(
                f"EODHD HTTP {r.status_code} for {path} (limit/entitlement). {hint}",
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

    def splits(self, symbol: str, from_date: Optional[str] = None,
               to_date: Optional[str] = None) -> list:
        """GET /splits/{SYMBOL} — historical split events (1 call per ticker).

        Included on both the paid ``EOD Historical Data — All World`` plan
        (full history back to 1970) and the free tier (past 1 year). Returns a
        list of ``{"date": "YYYY-MM-DD", "split": "<new>/<old>"}`` dicts
        (empty list when the symbol has no splits).
        """
        params = {"from": from_date, "to": to_date}
        data = self.get(f"/splits/{symbol}", params=params)
        if not isinstance(data, list):
            raise EodhdError(f"splits/{symbol} returned a non-list response")
        return data

    def realtime(self, symbol: str) -> dict:
        """GET /real-time/{SYMBOL} — live quote (included on the free tier)."""
        data = self.get(f"/real-time/{symbol}")
        if not isinstance(data, dict):
            raise EodhdError(f"real-time/{symbol} returned a non-dict response")
        return data

    def bulk_eod_last_day(self, exchange: str, date_str: Optional[str] = None) -> list:
        """GET /eod-bulk-last-day/{EXCHANGE} — entire exchange's latest (or dated)
        daily EOD bars in one call (EODHD bills 100 API calls per bulk request;
        paid plan only)."""
        params = {"date": date_str} if date_str else None
        data = self.get(f"/eod-bulk-last-day/{exchange.upper()}", params=params, cost=100)
        if not isinstance(data, list):
            raise EodhdError(f"eod-bulk-last-day/{exchange} returned a non-list response")
        return data

    def bulk_splits_last_day(self, exchange: str, date_str: Optional[str] = None) -> list:
        """GET /eod-bulk-last-day/{EXCHANGE}?type=splits — exchange-wide split
        events for a day in one bulk request (100 API calls; paid plan only)."""
        params: dict[str, Any] = {"type": "splits"}
        if date_str:
            params["date"] = date_str
        data = self.get(f"/eod-bulk-last-day/{exchange.upper()}", params=params, cost=100)
        if not isinstance(data, list):
            raise EodhdError(f"eod-bulk-last-day/{exchange}?type=splits returned a non-list response")
        return data
