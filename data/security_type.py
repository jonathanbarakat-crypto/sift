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

_WARRANT_SUFFIXES = {"W", "WS", "WT", "WU"}
_UNIT_SUFFIXES = {"U", "UN", "UT"}
_RIGHT_SUFFIXES = {"R", "RS", "RT"}
_PREFERRED_SUFFIXES = {"P", "PF", "PR"}
_NON_COMMON_SUFFIXES = (_WARRANT_SUFFIXES | _UNIT_SUFFIXES
                        | _RIGHT_SUFFIXES | _PREFERRED_SUFFIXES)


def is_common_stock_symbol(symbol: str) -> bool:
    """True when ``symbol`` looks like a COMMON STOCK ticker (not a warrant,
    unit, right, or preferred share)."""
    if not symbol:
        return False
    s = symbol.strip().upper()
    if not s:
        return False
    if "." in s:                      # SEC form: BRK.B / AACG.WS
        stem, _, suffix = s.rpartition(".")
    elif "-" in s:                    # Yahoo form: BRK-B / AAC-WT
        stem, _, suffix = s.rpartition("-")
    else:
        stem, suffix = s, ""
    if not stem or not stem.replace(".", "").replace("-", "").isalnum():
        return False
    if suffix:
        if suffix in _NON_COMMON_SUFFIXES:
            return False
        # class-share suffixes (single letters like B, or digits) are common
        return True
    # bare symbol: trailing-W warrants are >= 5 chars (AESPW); shorter names
    # ending in W (LOW, ARW, BBW, CW) are ordinary common tickers
    if len(stem) >= 5 and stem.endswith("W"):
        return False
    return True
