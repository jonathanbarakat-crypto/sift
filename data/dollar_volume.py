"""Split-consistent, PIT-invariant dollar volume and per-bar data-quality screens.

EODHD EOD API convention (https://eodhd.com/help/methodology):
  * ``open, high, low, close`` are RAW traded prices (unadjusted for splits or
    dividends).
  * ``adjusted_close`` is adjusted for BOTH splits and dividends (CRSP method).
  * ``volume`` is adjusted for SPLITS ONLY.

EODHD explicitly warns: *"Multiply volume, which is split-adjusted, by a
split-adjusted price. Do not multiply it by raw close or by adjusted_close."*

  * ``close * volume`` mixes raw price with split-adjusted volume, inflating
    historical dollar volume by the cumulative forward-split factor (AAPL 28x
    pre-2014) and leaking post-asof splits into historical rankings.
  * ``adjusted_close * volume`` mixes dividend-deflated price with volume that
    is not dividend-adjusted — deflating dividend payers 1.5x–3x over deep
    history (WMT, T, MO, KO, VZ, XOM), leaking post-asof dividends into PIT
    rankings, AND blowing up reverse-split names whose EODHD volume series was
    not restated consistently (e.g. SOS.US ranked #2 at $13.8B/day).

Canonical basis implemented here:
  1. Fetch historical split events from ``GET /api/splits/{SYMBOL}`` (included
     in the ``EOD Historical Data — All World`` paid plan at 1 call/ticker,
     cached under ``warehouse/splits/<EX>/<SYMBOL>.json``).
  2. For each bar date ``t``, compute the cumulative split factor from ``t`` to
     the vintage end: ``S(t) = prod_{split_date > t} (new / old)``.
  3. Compute ``split_adjusted_close(t) = close(t) / S(t)`` and
     ``dollar_volume(t) = split_adjusted_close(t) * volume(t)``.
     When EODHD's ``volume`` is split-adjusted by ``S(t)``, ``S(t)`` cancels
     exactly and recovers true historical traded dollar volume
     ``P_raw(t) * V_raw(t)`` — dividend-unbiased and strictly PIT-invariant.
  4. Verify split-vs-volume continuity empirically across every meaningful split
     event (catching EODHD vendor bugs like SOS.US where reverse splits adjusted
     price but left volume unadjusted) and run per-bar data-quality screens
     (catching single-bar ``999999.9999`` sentinels, ``PROSY``-style unsplit
     scale breaks, and missing split tables) without falsely quarantining high
     nominal prices (``BRK-A``) or heavily forward-split historical winners
     (early ``NVDA``).
"""
from __future__ import annotations

import math
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

DOLLAR_VOLUME_BASIS = "split_adjusted_close * volume"

# Per-bar and split-consistency screen thresholds (pre-registered).
EODHD_SENTINEL_PRICE = 999_999.9999
EODHD_SENTINEL_TOL = 0.01
MAX_SINGLE_DAY_PRICE_MULT = 10.0
MAX_SINGLE_BAR_DV_PRE_2010_USD = 25_000_000_000.0   # $25B cap (2000-2009)
MAX_SINGLE_BAR_DV_2010_2017_USD = 45_000_000_000.0  # $45B cap (2010-2017)
MAX_SINGLE_BAR_DOLLAR_VOLUME_USD = 250_000_000_000.0  # $250B cap (2018+)
MIN_MEDIAN_RAW_CLOSE_USD = 0.10

# Split-vs-volume & split-price continuity check thresholds.
SPLIT_CHECK_MIN_RATIO = 1.5
SPLIT_CHECK_WINDOW_BARS = 20
SPLIT_VOLUME_MIN_JUMP_FWD = 2.5
SPLIT_VOLUME_MIN_JUMP_REV = 2.5
SPLIT_VOLUME_SCORE_TOL = 0.40
SPLIT_VOLUME_REV_MIN_SCORE = 0.50
SPLIT_PARTIAL_REV_MAX_RATIO = 0.10
SPLIT_PARTIAL_REV_MIN_PRE_BARS = 40
SPLIT_PARTIAL_REV_MIN_POST_BARS = 10
SPLIT_PARTIAL_REV_MIN_DV_MULT = 15.0
MAX_SPLIT_EXDATE_PRICE_MULT = 2.5

# Foreign-exchange calendar contamination thresholds (US trading days that are
# major non-US market holidays: Dec 26 Boxing Day, May 1 Labour Day).
FOREIGN_CAL_MIN_HOLIDAYS = 3
FOREIGN_CAL_MIN_MISS_FRAC = 0.80

# Consistency between /api/splits/ and adjusted_close (dividend factor D(t)).
MAX_ADJ_TO_SPLIT_RATIO = 1.05
MAX_DAILY_DIVIDEND_STEP = 1.75
MISSING_SPLITS_STEP_RATIO = 1.25


def parse_split_ratio(raw: Any) -> float | None:
    """Parse an EODHD split ratio string ``"<new>/<old>"`` to ``new / old``.

    Per EODHD docs, ratios can be arbitrary floats (including spin-offs and
    stock dividends such as ``"1281.000000/1000.000000"``); always split on
    ``/`` and divide as floats. Returns ``None`` for invalid or non-positive
    inputs.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        val = float(raw)
        return val if math.isfinite(val) and val > 0.0 else None
    text = str(raw).strip()
    if not text:
        return None
    sep = "/" if "/" in text else (":" if ":" in text else None)
    try:
        if sep is None:
            val = float(text)
            return val if math.isfinite(val) and val > 0.0 else None
        parts = text.split(sep)
        if len(parts) != 2:
            return None
        num = float(parts[0].strip())
        den = float(parts[1].strip())
    except ValueError:
        return None
    if not (math.isfinite(num) and math.isfinite(den) and num > 0.0 and den > 0.0):
        return None
    val = num / den
    return val if math.isfinite(val) and val > 0.0 else None


def _to_date(val: Any) -> date | None:
    if val is None:
        return None
    if isinstance(val, date) and not hasattr(val, "hour"):
        return val
    try:
        return date.fromisoformat(str(val)[:10])
    except ValueError:
        return None


def parse_split_events(splits: list[dict[str, Any]] | None) -> list[tuple[date, float]]:
    """Normalize raw EODHD ``/api/splits/`` records into sorted ``(date, ratio)`` pairs.

    Multiple events sharing the same ex-date are multiplied together. Identity
    ratios (``ratio == 1``) and malformed records are dropped.
    """
    if not splits:
        return []
    by_date: dict[date, float] = {}
    for rec in splits:
        if not isinstance(rec, dict):
            continue
        d = _to_date(rec.get("date"))
        ratio = parse_split_ratio(rec.get("split"))
        if d is None or ratio is None:
            continue
        by_date[d] = by_date.get(d, 1.0) * ratio
    events = [(d, r) for d, r in sorted(by_date.items()) if abs(r - 1.0) > 1e-12]
    return events


def cumulative_split_factors(dates: np.ndarray,
                             events: list[tuple[date, float]]) -> np.ndarray:
    """Compute cumulative future split factor ``S(t) = prod_{d_k > t} r_k``.

    For a bar on date ``t``, dividing raw ``close(t)`` by ``S(t)`` yields the
    split-only-adjusted price restated to the same post-split share basis as
    EODHD's ``volume(t)``. On the ex-date itself (``d_k == t``), raw ``close``
    is already post-split, so only splits strictly AFTER ``t`` enter ``S(t)``.
    """
    n = len(dates)
    if n == 0 or not events:
        return np.ones(n, dtype=np.float64)
    date_ords = np.fromiter(
        (d.toordinal() if isinstance(d, date) else _to_date(d).toordinal() for d in dates),
        dtype=np.int64,
        count=n,
    )
    split_ords = np.array([d.toordinal() for d, _ in events], dtype=np.int64)
    ratios = np.array([r for _, r in events], dtype=np.float64)
    # suffix[i] = prod_{j >= i} ratios[j]; suffix[K] = 1.0
    suffix = np.ones(len(ratios) + 1, dtype=np.float64)
    suffix[:-1] = np.cumprod(ratios[::-1])[::-1]
    idx = np.searchsorted(split_ords, date_ords, side="right")
    return suffix[idx]


def _as_date_array(series: pd.Series) -> np.ndarray:
    arr = series.to_numpy()
    if arr.size == 0:
        return arr
    first = arr[0]
    if isinstance(first, date) and not hasattr(first, "hour"):
        return arr
    return pd.to_datetime(series).dt.date.to_numpy()


def _filter_executed_splits(
    events: list[tuple[date, float]],
    vintage_end: date | None,
) -> list[tuple[date, float]]:
    """Drop announced future splits (``split_date > vintage_end``) that have not
    yet been applied to the vintage's bars or volume."""
    if not events or vintage_end is None:
        return events
    return [(d, r) for d, r in events if d <= vintage_end]


def _group_split_events(
    dates: np.ndarray,
    events: list[tuple[date, float]],
) -> list[tuple[date, date, float]]:
    """Merge consecutive splits separated by 0 bars into ``(d_start, d_end, prod_ratio)``.

    Prevents phantom/duplicate vendor split entries with no bars between them
    (e.g. ``JDSU`` recording 1:8 on both ``2006-10-17`` and ``2006-11-14`` with
    0 bars in between) from silently bypassing continuity checks.
    """
    if not events or len(dates) == 0:
        return []
    first_d = dates[0]
    groups: list[list[Any]] = []
    for d, r in events:
        if not groups:
            groups.append([d, d, r])
            continue
        prev_end = groups[-1][1]
        if prev_end >= first_d and int(np.sum((dates >= prev_end) & (dates < d))) == 0:
            groups[-1][1] = d
            groups[-1][2] *= r
        else:
            groups.append([d, d, r])
    return [(g[0], g[1], float(g[2])) for g in groups if abs(g[2] - 1.0) > 1e-12]


def compute_split_adjusted_dollar_volume(
    df: pd.DataFrame,
    splits: list[dict[str, Any]] | None = None,
    *,
    vintage_end: date | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ``(split_factors, split_adjusted_close, dollar_volume)`` for ``df``.

    Always computes from the primitives ``close``, ``volume``, and ``splits``
    (never reads a stale stored ``dollar_volume`` column). Announced future
    splits strictly after ``vintage_end`` (default: ``max(df['date'])``) are
    ignored because the vintage's bars/volume have not yet been restated.
    """
    if df.empty:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty
    dates = _as_date_array(df["date"])
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=np.float64)
    volume = pd.to_numeric(df["volume"], errors="coerce").to_numpy(dtype=np.float64)
    end_dt = vintage_end if vintage_end is not None else max(dates)
    events = _filter_executed_splits(parse_split_events(splits), end_dt)
    factors = cumulative_split_factors(dates, events)
    split_adj_close = close / factors
    dollar_volume = split_adj_close * volume
    return factors, split_adj_close, dollar_volume


def check_split_price_continuity(
    dates: np.ndarray,
    close: np.ndarray,
    events: list[tuple[date, float]],
) -> list[str]:
    """Verify that raw ``close`` jumps by roughly ``1 / ratio`` across each
    split (equivalently, ``split_adjusted_close`` is continuous across the ex-date).

    Catches duplicate/phantom splits (e.g. ``JDSU`` applying 1:8 twice -> 8.2x
    cliff) and inverted/mis-dated splits (e.g. ``RHT`` 2000-01-10 -> 4.3x jump,
    ``AMCC`` 2007-12-11 -> 3.8x jump).
    """
    if len(dates) < 2 or not events:
        return []
    groups = _group_split_events(dates, events)
    if not groups:
        return []
    valid = np.isfinite(close) & (close > 0.0)
    if int(valid.sum()) < 2:
        return []
    factors = cumulative_split_factors(dates, events)
    split_c = close / factors
    mismatches: list[str] = []
    n_groups = len(groups)
    for i, (d_start, d_end, ratio) in enumerate(groups):
        if max(ratio, 1.0 / ratio) < SPLIT_CHECK_MIN_RATIO:
            continue
        prev_bound = groups[i - 1][1] if i > 0 else date.min
        next_bound = groups[i + 1][0] if i + 1 < n_groups else date.max
        pre_idx = np.flatnonzero(valid & (dates > prev_bound) & (dates < d_start))
        post_idx = np.flatnonzero(valid & (dates >= d_end) & (dates < next_bound))
        if pre_idx.size == 0 or post_idx.size == 0:
            continue
        sc_pre = float(split_c[pre_idx[-1]])
        sc_post = float(split_c[post_idx[0]])
        if sc_pre <= 0.0 or sc_post <= 0.0:
            continue
        jump = max(sc_post / sc_pre, sc_pre / sc_post)
        if jump > MAX_SPLIT_EXDATE_PRICE_MULT:
            mismatches.append(
                f"{d_start.isoformat()} split (r={ratio:g}): split_adjusted_close "
                f"jumped {jump:.2g}x across ex-date (> {MAX_SPLIT_EXDATE_PRICE_MULT:g}x)"
            )
    return mismatches


def check_split_volume_consistency(
    dates: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    events: list[tuple[date, float]],
) -> list[str]:
    """Verify empirically whether EODHD restated ``volume`` across each split.

    Compares pre-split vs post-split median dollar volume around each split
    (merging zero-bar-gap duplicate splits). For reverse splits, flags both:
      1. Local boundary cliff where split-adjusted DV collapses >= 2.5x and
         raw ``close * volume`` matches ``dv_post`` better than split-adjusted DV
         (``unadj_score >= 0.50``, no upper cap).
      2. Partial historical restatement on large reverse splits (``r <= 0.10``)
         where EODHD restated only the final few weeks before the split while
         leaving earlier history unadjusted (e.g. ``CXRXF``, ``USU``, ``BTUUQ``).

    Inputs must be sorted ascending by ``dates``. Returns human-readable
    mismatch descriptions (empty list when consistent).
    """
    if len(dates) < 2 or not events:
        return []
    groups = _group_split_events(dates, events)
    if not groups:
        return []
    mismatches: list[str] = []
    valid = np.isfinite(close) & (close > 0.0) & np.isfinite(volume) & (volume > 0.0)
    if int(valid.sum()) < 2:
        return []

    factors = cumulative_split_factors(dates, events)
    dv_adj = (close / factors) * volume
    n_groups = len(groups)
    for i, (d_start, d_end, ratio) in enumerate(groups):
        mag = max(ratio, 1.0 / ratio)
        if mag < SPLIT_CHECK_MIN_RATIO:
            continue
        prev_dt = groups[i - 1][1] if i > 0 else date.min
        next_dt = groups[i + 1][0] if i + 1 < n_groups else date.max

        pre_all = np.flatnonzero(valid & (dates > prev_dt) & (dates < d_start))
        post_all = np.flatnonzero(valid & (dates >= d_end) & (dates < next_dt))
        if pre_all.size == 0 or post_all.size == 0:
            continue
        pre_idx = pre_all[-SPLIT_CHECK_WINDOW_BARS:]
        post_idx = post_all[:SPLIT_CHECK_WINDOW_BARS]

        dv_pre_raw = float(np.median(close[pre_idx] * volume[pre_idx]))
        dv_post = float(np.median(close[post_idx] * volume[post_idx]))
        flagged_local = False
        if dv_pre_raw > 0.0 and dv_post > 0.0:
            dv_pre_adj = dv_pre_raw / ratio
            log_r = math.log(ratio)
            log_jump_adj = math.log(dv_post / dv_pre_adj)
            unadj_score = log_jump_adj / log_r
            if ratio < 1.0:
                if (
                    log_jump_adj <= -math.log(SPLIT_VOLUME_MIN_JUMP_REV)
                    and unadj_score >= SPLIT_VOLUME_REV_MIN_SCORE
                ):
                    mismatches.append(
                        f"{d_start.isoformat()} reverse split (r={ratio:g}): volume appears "
                        f"unadjusted (score={unadj_score:.2f}, adj dv jump={math.exp(log_jump_adj):.2g}x)"
                    )
                    flagged_local = True
            else:
                if (
                    abs(log_jump_adj) >= math.log(SPLIT_VOLUME_MIN_JUMP_FWD)
                    and abs(unadj_score - 1.0) <= SPLIT_VOLUME_SCORE_TOL
                ):
                    mismatches.append(
                        f"{d_start.isoformat()} forward split (r={ratio:g}): volume appears "
                        f"unadjusted (score={unadj_score:.2f}, adj dv jump={math.exp(log_jump_adj):.2g}x)"
                    )
                    flagged_local = True

        # Partial historical volume restatement check for large reverse splits.
        if (
            not flagged_local
            and ratio <= SPLIT_PARTIAL_REV_MAX_RATIO
            and pre_all.size >= SPLIT_PARTIAL_REV_MIN_PRE_BARS
            and post_all.size >= SPLIT_PARTIAL_REV_MIN_POST_BARS
        ):
            dv_pre_med = float(np.median(dv_adj[pre_all]))
            dv_post_med = float(np.median(dv_adj[post_all]))
            dv_pre_raw_med = float(np.median(close[pre_all] * volume[pre_all]))
            if dv_pre_med > 0.0 and dv_post_med > 0.0 and dv_pre_raw_med > 0.0:
                if (
                    (dv_pre_med / dv_post_med) > SPLIT_PARTIAL_REV_MIN_DV_MULT
                    and abs(math.log(dv_pre_raw_med / dv_post_med))
                    < 0.5 * abs(math.log(dv_pre_med / dv_post_med))
                ):
                    mismatches.append(
                        f"{d_start.isoformat()} reverse split (r={ratio:g}): historical pre-split "
                        f"volume appears partially unadjusted ({dv_pre_med / dv_post_med:.2g}x median DV)"
                    )
    return mismatches


def _era_dollar_volume_caps(dates: np.ndarray) -> np.ndarray:
    """Era-aware per-bar dollar-volume caps reflecting US equity market scale."""
    d_2010 = date(2010, 1, 1)
    d_2018 = date(2018, 1, 1)
    return np.where(
        dates < d_2010,
        MAX_SINGLE_BAR_DV_PRE_2010_USD,
        np.where(dates < d_2018, MAX_SINGLE_BAR_DV_2010_2017_USD, MAX_SINGLE_BAR_DOLLAR_VOLUME_USD),
    )


def _check_us_foreign_calendar(dates: np.ndarray, volume: np.ndarray) -> list[str]:
    """Detect non-US exchange series (or multi-year ticker-reuse holes) leaked
    into EODHD's US symbol list by checking US trading days that are major
    non-US holidays:
      * Dec 26 when Tue–Fri (US open; UK/Commonwealth Boxing Day closed)
      * May 1 when Mon–Fri (US open; Europe/Russia/LatAm/Asia Labour Day closed)
    """
    if len(dates) < 20:
        return []
    lo, hi = dates[0], dates[-1]
    if hi.year - lo.year < FOREIGN_CAL_MIN_HOLIDAYS - 1:
        return []
    active_dates = set(dates[np.isfinite(volume) & (volume > 0.0)])
    dec26 = [
        date(y, 12, 26)
        for y in range(lo.year, hi.year + 1)
        if lo <= date(y, 12, 26) <= hi and date(y, 12, 26).weekday() in (1, 2, 3, 4)
    ]
    may1 = [
        date(y, 5, 1)
        for y in range(lo.year, hi.year + 1)
        if lo <= date(y, 5, 1) <= hi and date(y, 5, 1).weekday() < 5
    ]
    hits: list[str] = []
    if len(dec26) >= FOREIGN_CAL_MIN_HOLIDAYS:
        miss = sum(1 for d in dec26 if d not in active_dates)
        if miss / len(dec26) >= FOREIGN_CAL_MIN_MISS_FRAC:
            hits.append(f"missing/zero-vol on {miss}/{len(dec26)} US Dec-26 sessions")
    if len(may1) >= FOREIGN_CAL_MIN_HOLIDAYS:
        miss = sum(1 for d in may1 if d not in active_dates)
        if miss / len(may1) >= FOREIGN_CAL_MIN_MISS_FRAC:
            hits.append(f"missing/zero-vol on {miss}/{len(may1)} US May-1 sessions")
    return hits


def _has_sentinel(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    return bool(np.any(np.abs(finite - EODHD_SENTINEL_PRICE) <= EODHD_SENTINEL_TOL))


def evaluate_symbol_dollar_volume(
    asof_df: pd.DataFrame,
    *,
    full_df: pd.DataFrame | None = None,
    splits: list[dict[str, Any]] | None = None,
    check_us_calendar: bool = True,
) -> dict[str, Any]:
    """Compute split-adjusted dollar volume and data-quality diagnostics for one symbol.

    ``asof_df`` contains only bars with ``date <= asof`` (all PIT aggregates and
    per-bar anomaly checks run strictly on ``asof_df``). ``full_df`` (optional)
    is the full loaded vintage series for the symbol and is used ONLY to decode
    the vendor's retroactive volume-adjustment encoding (vintage end date for
    filtering unexecuted future splits, split-vs-volume/price continuity, and
    missing-splits detection).
    """
    dates_raw = _as_date_array(asof_df["date"])
    order = np.argsort(dates_raw, kind="mergesort")
    dates = dates_raw[order]
    close = pd.to_numeric(asof_df["close"], errors="coerce").to_numpy(dtype=np.float64)[order]
    adj_close = pd.to_numeric(asof_df["adjusted_close"], errors="coerce").to_numpy(dtype=np.float64)[order]
    volume = pd.to_numeric(asof_df["volume"], errors="coerce").to_numpy(dtype=np.float64)[order]

    # Full vintage arrays for vendor encoding checks (vintage end date, split continuity, missing splits).
    if full_df is not None and not full_df.empty:
        f_dates_raw = _as_date_array(full_df["date"])
        f_order = np.argsort(f_dates_raw, kind="mergesort")
        f_dates = f_dates_raw[f_order]
        f_close = pd.to_numeric(full_df["close"], errors="coerce").to_numpy(dtype=np.float64)[f_order]
        f_adj = pd.to_numeric(full_df["adjusted_close"], errors="coerce").to_numpy(dtype=np.float64)[f_order]
        f_vol = pd.to_numeric(full_df["volume"], errors="coerce").to_numpy(dtype=np.float64)[f_order]
    else:
        f_dates, f_close, f_adj, f_vol = dates, close, adj_close, volume

    vintage_end = f_dates[-1] if len(f_dates) else None
    events = _filter_executed_splits(parse_split_events(splits), vintage_end)
    factors = cumulative_split_factors(dates, events)
    split_adj_close = close / factors
    dollar_vol = split_adj_close * volume

    reason_codes: list[str] = []
    details: list[str] = []

    def _flag(code: str, detail: str) -> None:
        if code not in reason_codes:
            reason_codes.append(code)
        details.append(f"{code}: {detail}")

    # 1. Per-bar EODHD sentinel check (999999.9999) — catches even a single bar.
    sent_close = _has_sentinel(close)
    sent_adj = _has_sentinel(adj_close)
    if sent_close or sent_adj:
        which = "/".join(
            name for name, hit in (("close", sent_close), ("adjusted_close", sent_adj)) if hit
        )
        _flag("eodhd_sentinel_price", f"{which} contains vendor sentinel {EODHD_SENTINEL_PRICE}")

    # 2. Per-bar validity check (NaN/inf, <= 0 price, negative volume).
    bad_close = (~np.isfinite(close)) | (close <= 0.0)
    bad_adj = (~np.isfinite(adj_close)) | (adj_close <= 0.0)
    bad_vol = (~np.isfinite(volume)) | (volume < 0.0)
    if bool(np.any(bad_close) or np.any(bad_adj) or np.any(bad_vol)):
        _flag(
            "non_finite_or_non_positive_price",
            f"bad bars: close={int(bad_close.sum())}, "
            f"adjusted_close={int(bad_adj.sum())}, volume={int(bad_vol.sum())}",
        )

    # 3. Split-vs-price & split-vs-volume consistency checks (when splits table is present).
    if splits is not None and events:
        price_mismatches = check_split_price_continuity(f_dates, f_close, events)
        if price_mismatches:
            _flag("split_price_mismatch", "; ".join(price_mismatches))
        mismatches = check_split_volume_consistency(f_dates, f_close, f_vol, events)
        if mismatches:
            _flag("split_volume_mismatch", "; ".join(mismatches))

    # 4. Missing-splits fail-closed check (when no /api/splits/ file is cached).
    if splits is None:
        f_ok = (
            np.isfinite(f_close) & (f_close > 0.0)
            & np.isfinite(f_adj) & (f_adj > 0.0)
            & (np.abs(f_close - EODHD_SENTINEL_PRICE) > EODHD_SENTINEL_TOL)
            & (np.abs(f_adj - EODHD_SENTINEL_PRICE) > EODHD_SENTINEL_TOL)
        )
        if int(f_ok.sum()) >= 1:
            q = f_adj[f_ok] / f_close[f_ok]
            max_q = float(np.max(q))
            max_step = 1.0
            if q.size >= 2:
                steps = q[1:] / q[:-1]
                max_step = float(np.max(np.maximum(steps, 1.0 / steps)))
            if max_q > MAX_ADJ_TO_SPLIT_RATIO or max_step > MISSING_SPLITS_STEP_RATIO:
                _flag(
                    "missing_splits_table",
                    f"no cached /api/splits/ file but adjusted_close/close indicates "
                    f"splits (max_ratio={max_q:.3g}, max_daily_step={max_step:.3g}x); "
                    f"run ingest_eodhd --splits-only",
                )

    # 5. Foreign-calendar / multi-year-gap check (catches LSE/MOEX/Euronext/B3
    #    series leaked into EODHD's US symbol list, plus multi-year ticker reuse).
    if check_us_calendar:
        cal_hits = _check_us_foreign_calendar(dates, volume)
        if cal_hits:
            _flag("foreign_calendar_mismatch", "; ".join(cal_hits))

    # Remaining per-bar checks on valid as-of bars.
    ok_bars = (~bad_close) & (~bad_adj) & (~bad_vol)
    if bool(np.any(ok_bars)):
        ok_dates = dates[ok_bars]
        ok_close = close[ok_bars]
        ok_split = split_adj_close[ok_bars]
        ok_adj = adj_close[ok_bars]
        ok_dv = dollar_vol[ok_bars]

        # 6. Consistency between /api/splits/ and adjusted_close (dividend factor D(t)).
        if splits is not None and not (sent_close or sent_adj):
            div_factor = ok_adj / ok_split
            max_div = float(np.max(div_factor))
            max_div_step = 1.0
            if div_factor.size >= 2:
                d_steps = div_factor[1:] / div_factor[:-1]
                max_div_step = float(np.max(np.maximum(d_steps, 1.0 / d_steps)))
            if max_div > MAX_ADJ_TO_SPLIT_RATIO or max_div_step > MAX_DAILY_DIVIDEND_STEP:
                _flag(
                    "adj_split_divergence",
                    f"adjusted_close/split_adjusted_close violates dividend-only bounds "
                    f"(max={max_div:.3g}, max_daily_step={max_div_step:.3g}x)",
                )

        # 7. Unsplit single-day price scale break (catches PROSY $900k -> $7 cliff,
        #    currency-unit flips, and unrecorded splits without touching BRK-A).
        if ok_split.size >= 2:
            p_steps = ok_split[1:] / ok_split[:-1]
            max_p_step = float(np.max(np.maximum(p_steps, 1.0 / p_steps)))
            if max_p_step > MAX_SINGLE_DAY_PRICE_MULT:
                _flag(
                    "unsplit_price_scale_break",
                    f"single-day split_adjusted_close jump {max_p_step:.2g}x "
                    f"(> {MAX_SINGLE_DAY_PRICE_MULT:g}x)",
                )

        # 8. Era-aware single-bar dollar volume cap ($25B pre-2010, $45B 2010-2017, $250B 2018+).
        caps = _era_dollar_volume_caps(ok_dates)
        over = ok_dv > caps
        if bool(np.any(over)):
            worst_i = int(np.argmax(ok_dv / caps))
            _flag(
                "implausible_bar_dollar_volume",
                f"single-bar dollar volume ${float(ok_dv[worst_i]):,.0f} on "
                f"{ok_dates[worst_i].isoformat()} exceeds era cap ${float(caps[worst_i]):,.0f}",
            )

        # 9. Sub-dime raw close screen (evaluated on RAW close, never adjusted_close,
        #    so forward-split winners like NVDA/AAPL are never falsely dropped).
        med_close = float(np.median(ok_close))
        if med_close < MIN_MEDIAN_RAW_CLOSE_USD:
            _flag(
                "sub_dime_raw_price",
                f"median raw close ${med_close:.4f} < ${MIN_MEDIAN_RAW_CLOSE_USD:g}",
            )

    finite_dv = dollar_vol[np.isfinite(dollar_vol)]
    finite_close = close[np.isfinite(close)]
    finite_split = split_adj_close[np.isfinite(split_adj_close)]
    finite_adj = adj_close[np.isfinite(adj_close)]

    return {
        "avg_dollar_volume": float(np.mean(finite_dv)) if finite_dv.size else float("nan"),
        "median_dollar_volume": float(np.median(finite_dv)) if finite_dv.size else float("nan"),
        "max_bar_dollar_volume": float(np.max(finite_dv)) if finite_dv.size else float("nan"),
        "bars": int(len(dates)),
        "first_date": dates[0],
        "last_date": dates[-1],
        "mean_close": float(np.mean(finite_close)) if finite_close.size else float("nan"),
        "median_close": float(np.median(finite_close)) if finite_close.size else float("nan"),
        "mean_split_adjusted_close": float(np.mean(finite_split)) if finite_split.size else float("nan"),
        "mean_adjusted_close": float(np.mean(finite_adj)) if finite_adj.size else float("nan"),
        "splits_verified": splits is not None,
        "n_splits": len(events),
        "reason_codes": reason_codes,
        "reason": "; ".join(details),
    }
