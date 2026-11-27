"""Security-type classifier for US ticker symbols (research-seed filtering).

The SEC EDGAR company_tickers_exchange.json seed (and any Yahoo-derived
universe) contains warrants, units, rights and preferred shares alongside
common stock — e.g. ``AAC-WT`` (warrant), ``AAC-UN`` (unit), ``AESPW``
(trailing-W warrant), ``...-P`` (preferred). The EODHD production path
filters these server-side at ingest (``Type in ("Common Stock", "Stock")``,
data/ingest_eodhd.py); the Yahoo research path has no type field, so it
filters by symbol pattern instead — mirroring the EODHD intent.

Patterns (case-insensitive; separator is ``-`` (Yahoo) or ``.`` (SEC form)):

  * warrant:  suffix W / WS / WT / WU, or a bare symbol of length >= 5 ending
    in W (e.g. AESPW, APXTW). The length >= 5 guard keeps genuine common
    tickers that happen to end in W (LOW, ARW, BBW) — accepted tradeoff:
    a 5+ letter trailing-W name is a warrant in practice.
  * unit:     suffix U / UN / UT
  * right:    suffix R / RS / RT
  * preferred: suffix P / PF / PR
  * class shares (BRK-B, BF-B) and ordinary names (AAPL, MSFT, CW, GE) pass.
"""
from __future__ import annotations

import re

_WARRANT_SUFFIXES = {"W", "WS", "WT", "WU"}
_UNIT_SUFFIXES = {"U", "UN", "UT"}
_RIGHT_SUFFIXES = {"R", "RS", "RT"}
_PREFERRED_SUFFIXES = {"P", "PF", "PR"}
_WHEN_ISSUED_SUFFIXES = {"WI"}
_NON_COMMON_SUFFIXES = (
    _WARRANT_SUFFIXES
    | _UNIT_SUFFIXES
    | _RIGHT_SUFFIXES
    | _PREFERRED_SUFFIXES
    | _WHEN_ISSUED_SUFFIXES
)
_PREFERRED_SERIES_RE = re.compile(r"^P[A-Z]$")
# Eurex / CME-style futures contracts leaked into EODHD US symbol lists
# (e.g. FGBLZ26, FMWOM26, FXXPM26, FBTPM26, FOATZ26).
_FUTURES_CONTRACT_RE = re.compile(r"^F[A-Z]{3}[FGHJKMNQUVXZ]\d{2}$")

_NON_COMMON_NAME_MARKERS = (
    "future",
    "(cbt)",
    "(cme)",
    "when issued",
    "warrant",
    "preferred",
    " prf ",
    "series trust",
    "proshares trust",
    "tidal trust",
    "ishares",
    " etf",
)


def is_common_stock_symbol(symbol: str) -> bool:
    """True when ``symbol`` looks like a COMMON STOCK ticker (not a warrant,
    unit, right, preferred share, when-issued line, or futures contract)."""
    if not symbol:
        return False
    s = symbol.strip().upper()
    if not s:
        return False
    if _FUTURES_CONTRACT_RE.match(s):
        return False
    parts = re.split(r"[-.]", s)
    root = parts[0]
    if not root or not root.isalnum():
        return False
    segs = parts[1:]
    if segs:
        for seg in segs:
            if not seg or not seg.isalnum():
                return False
            if seg in _NON_COMMON_SUFFIXES or _PREFERRED_SERIES_RE.match(seg):
                return False
        # class-share suffixes (single letters like B, or digits) are common
        return True
    # bare symbol: trailing-W warrants are >= 5 chars (AESPW); shorter names
    # ending in W (LOW, ARW, BBW, CW) are ordinary common tickers
    if len(root) >= 5 and root.endswith("W"):
        return False
    return True


def is_common_stock_metadata(
    symbol: str,
    *,
    name: str | None = None,
    exchange: str | None = None,
    sec_type: str | None = None,
) -> bool:
    """Metadata-aware common-stock filter using EODHD symbol-list fields.

    Drops non-common ticker patterns plus EODHD's unmapped ``Exchange == "US"``
    catch-all bucket (where Eurex futures, foreign LSE/TSX/PSE/MOEX tickers,
    ETFs, and when-issued symbols land) and explicit non-equity name markers.
    When metadata fields are ``None`` (e.g. synthetic fixtures without meta/),
    behaves identically to :func:`is_common_stock_symbol`.
    """
    if not is_common_stock_symbol(symbol):
        return False
    if sec_type is not None:
        st = str(sec_type).strip()
        if st and st not in ("Common Stock", "Stock"):
            return False
    if exchange is not None:
        ex = str(exchange).strip().upper()
        if ex == "US":
            return False
    if name is not None:
        padded = f" {str(name).strip().lower()} "
        if any(marker in padded for marker in _NON_COMMON_NAME_MARKERS):
            return False
    return True
