#!/usr/bin/env python3
"""SIFT point-in-time (PIT) universe builder — top-N by average dollar volume.

v3 scope doc: free tier searches the "top-1,000 liquid" universe (section 7.2)
and "survivorship-bias-free data is mandatory at every tier" (section 4.5 /
12.8). This module builds that universe from the ingested warehouse:

  * ranks every ingested common-stock symbol by average split-consistent dollar
    volume (``split_adjusted_close * volume`` using ``/api/splits/``) over the
    available window — point-in-time: the ranking is computed as of ``--asof``
    using only bars with ``date <= asof``, so no future information leaks into
    the listing;
  * NO survivorship bias: symbols whose series ends before ``asof``
    (delisted within the window) stay in the ranked table with their
    ``last_date`` and an explicit ``delisted``/``ended_before_asof`` flag —
    they are simply ranked by the dollar volume they actually had, and the
    top-N cut is labeled with the as-of date;
  * writes the full ranked table plus the top-N PIT listing as parquet under
    ``data/warehouse/universe/`` and updates the warehouse data-vintage hash.
    Symbols tripping per-bar data-quality or split-volume-consistency screens
    (vendor ``999999.9999`` sentinels, unadjusted reverse-split volume bugs
    such as ``SOS.US``, unsplit scale breaks such as ``PROSY``, or missing
    split tables on split symbols) are QUARANTINED to
    ``quarantined_<asof>.parquet`` for review — never silently ranked.

This is server-side only (v3 section 12): the universe table is an internal
input to the mining engine; clients never receive raw bars.

Usage:
    python3 -m data.universe --warehouse data/warehouse --n 1000
    python3 -m data.universe --asof 2026-11-13 --n 1000
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import date, datetime, timezone

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data import warehouse as wh  # type: ignore
    from data.dollar_volume import (  # type: ignore
        DOLLAR_VOLUME_BASIS,
        EODHD_SENTINEL_PRICE,
        MAX_SINGLE_BAR_DOLLAR_VOLUME_USD,
        MAX_SINGLE_DAY_PRICE_MULT,
        MIN_MEDIAN_RAW_CLOSE_USD,
        evaluate_symbol_dollar_volume,
    )
    from data.security_type import is_common_stock_symbol  # type: ignore
else:
    from . import warehouse as wh
    from .dollar_volume import (
        DOLLAR_VOLUME_BASIS,
        EODHD_SENTINEL_PRICE,
        MAX_SINGLE_BAR_DOLLAR_VOLUME_USD,
        MAX_SINGLE_DAY_PRICE_MULT,
        MIN_MEDIAN_RAW_CLOSE_USD,
        evaluate_symbol_dollar_volume,
    )
    from .security_type import is_common_stock_symbol

DEFAULT_WAREHOUSE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "data", "warehouse")

RESEARCH_VENDORS = {"yfinance", "yf", "yahoo"}

# Pre-registered universe QUALITY FLOORS (hermes-tasks #182): 1-bar names and
# micro-caps must not be rankable into the top-N universe. Defaults assume a
# ~5y daily window (~1260 bars); scale --min-bars with a shorter window
# (escape hatch: explicit --min-bars 1 / --min-avg-dollar-volume 0 restores
# the old permissive behavior). Non-common symbols (warrants/units/rights/
# preferred) are ALWAYS excluded from ranking, regardless of floors.
DEFAULT_MIN_BARS = 20
DEFAULT_MIN_AVG_DOLLAR_VOLUME = 1_000_000.0  # $1M mean daily dollar volume

RANKED_COLUMNS = [
    "rank", "symbol", "exchange", "avg_dollar_volume", "bars", "first_date", "last_date",
    "asof", "ended_before_asof", "delisted", "delist_idx_available",
]
QUARANTINE_COLUMNS = [
    "symbol", "exchange", "bars", "avg_dollar_volume", "median_dollar_volume",
    "max_bar_dollar_volume", "mean_close", "median_close",
    "mean_split_adjusted_close", "mean_adjusted_close", "last_date",
    "passes_quality_floors", "reason_codes", "reason",
]


def log(msg: str) -> None:
    print(f"[universe {datetime.now(timezone.utc).strftime('%H:%M:%S')}Z] {msg}", flush=True)


def vintage_is_research(root: str) -> tuple[bool, dict]:
    """Read vintage.json and report whether this warehouse is a RESEARCH-grade
    vintage (e.g. yfinance/Yahoo — vendor 'yfinance', research_grade=true).

    Research vendors have NO delisted-name list, so the universe built from
    them is ACTIVE-ONLY and survivorship-biased by construction (v3 §4.5/§12.8).
    Returns (is_research, vintage_record).
    """
    rec = wh.read_json(os.path.join(root, wh.VINTAGE_JSON)) or {}
    vendor = str(rec.get("vendor", "")).lower()
    is_research = bool(rec.get("research_grade")) or vendor in RESEARCH_VENDORS
    return is_research, rec


def load_all_bars(root: str) -> pd.DataFrame:
    """Concatenate every eod parquet under the warehouse. Returns empty frame if none."""
    eod_root = os.path.join(root, "eod")
    frames = []
    if os.path.isdir(eod_root):
        for dirpath, _dirnames, filenames in os.walk(eod_root):
            for name in sorted(filenames):
                if name.endswith(".parquet"):
                    df = wh.read_parquet(os.path.join(dirpath, name))
                    if df is not None and not df.empty:
                        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=wh.EOD_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def load_all_splits(root: str) -> dict[str, list]:
    """Load cached ``/api/splits/`` files from ``<root>/splits/<EX>/<SYM>.json``.

    Returns a dict keyed by ``"<SYM>.<EX>"`` (e.g. ``"AAPL.US"``). Missing
    directory returns ``{}``.
    """
    splits_root = os.path.join(root, "splits")
    out: dict[str, list] = {}
    if not os.path.isdir(splits_root):
        return out
    for dirpath, _dirnames, filenames in os.walk(splits_root):
        ex = os.path.basename(dirpath).upper()
        for name in sorted(filenames):
            if not name.endswith(".json"):
                continue
            stem = name[:-len(".json")]
            data = wh.read_json(os.path.join(dirpath, name))
            if not isinstance(data, list):
                continue
            key = stem if "." in stem else f"{stem}.{ex}"
            out[key] = data
    return out


def _empty_tables(n_non_common: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = pd.DataFrame(columns=RANKED_COLUMNS)
    ranked.attrs["non_common_excluded"] = n_non_common
    ranked.attrs["splits_verified_symbols"] = 0
    ranked.attrs["splits_missing_symbols"] = 0
    ranked.attrs["quarantined_by_reason"] = {}
    quarantine = pd.DataFrame(columns=QUARANTINE_COLUMNS)
    return ranked, quarantine


def build_ranked_table(
    bars: pd.DataFrame,
    asof: str,
    research_grade: bool = False,
    min_bars: int = 0,
    min_avg_dollar_volume: float = 0.0,
    splits_by_symbol: dict[str, list] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Point-in-time ranking as of ``asof`` (YYYY-MM-DD).

    Returns ``(ranked, quarantine)``:
      * ``ranked`` — clean, rankable common-stock symbols sorted by average
        split-consistent dollar volume (``split_adjusted_close * volume``),
        recomputed from the primitives ``close``, ``volume``, and ``splits``
        (never trusting a stale stored ``dollar_volume`` column).
      * ``quarantine`` — common-stock symbols excluded by the per-bar quality
        and split-volume-consistency screens, with ``passes_quality_floors``,
        machine-readable ``reason_codes``, and human-readable ``reason``.

    Uses only bars with ``date <= asof`` for all PIT aggregates and per-bar
    checks. Symbols whose last bar is before ``asof`` stay in the ranking (no
    survivorship bias) and are flagged.

    Quality floors (hermes-tasks #182): non-common-stock symbols (warrants,
    units, rights, preferred — classified by data/security_type.py) are
    ALWAYS excluded; rows with ``bars < min_bars`` or
    ``avg_dollar_volume < min_avg_dollar_volume`` are excluded when the
    caller passes floors > 0.
    """
    if bars.empty:
        return _empty_tables()

    asof_date = date.fromisoformat(asof)
    df_all = bars.copy()
    df_all["date"] = pd.to_datetime(df_all["date"]).dt.date

    # Restrict to bars <= asof for PIT membership; filter non-common stocks
    # before running vendor-artifact screens so warrants/units aren't counted
    # as quarantined common stocks.
    df_asof = df_all[df_all["date"] <= asof_date]
    if df_asof.empty:
        return _empty_tables()

    unique_syms = df_asof["symbol"].drop_duplicates()
    common_syms = {
        sym for sym in unique_syms
        if is_common_stock_symbol(str(sym).split(".")[0])
    }
    n_non_common = int(len(unique_syms) - len(common_syms))
    if not common_syms:
        return _empty_tables(n_non_common=n_non_common)

    df_asof_common = df_asof[df_asof["symbol"].isin(common_syms)]
    df_all_common = df_all[df_all["symbol"].isin(common_syms)]
    full_groups = {
        key: grp for key, grp in df_all_common.groupby(["symbol", "exchange"], sort=False)
    }

    ranked_rows: list[dict] = []
    quarantine_rows: list[dict] = []
    n_splits_verified = 0
    n_splits_missing = 0
    reason_counts: dict[str, int] = {}

    for (sym, ex), g_asof in df_asof_common.groupby(["symbol", "exchange"], sort=True):
        sym_splits: list | None = None
        if splits_by_symbol is not None:
            if sym in splits_by_symbol:
                sym_splits = splits_by_symbol[sym]
            else:
                bare = str(sym).split(".")[0]
                if bare in splits_by_symbol:
                    sym_splits = splits_by_symbol[bare]

        ev = evaluate_symbol_dollar_volume(
            g_asof,
            full_df=full_groups.get((sym, ex)),
            splits=sym_splits,
        )
        if ev["splits_verified"]:
            n_splits_verified += 1
        else:
            n_splits_missing += 1

        adv = ev["avg_dollar_volume"]
        n_bars = ev["bars"]
        passes_floors = (
            (min_bars <= 0 or n_bars >= min_bars)
            and (
                min_avg_dollar_volume <= 0.0
                or (math.isfinite(adv) and adv >= min_avg_dollar_volume)
            )
        )

        if ev["reason_codes"]:
            for code in ev["reason_codes"]:
                reason_counts[code] = reason_counts.get(code, 0) + 1
            quarantine_rows.append({
                "symbol": sym,
                "exchange": ex,
                "bars": n_bars,
                "avg_dollar_volume": adv,
                "median_dollar_volume": ev["median_dollar_volume"],
                "max_bar_dollar_volume": ev["max_bar_dollar_volume"],
                "mean_close": ev["mean_close"],
                "median_close": ev["median_close"],
                "mean_split_adjusted_close": ev["mean_split_adjusted_close"],
                "mean_adjusted_close": ev["mean_adjusted_close"],
                "last_date": ev["last_date"],
                "passes_quality_floors": bool(passes_floors),
                "reason_codes": ",".join(ev["reason_codes"]),
                "reason": ev["reason"],
            })
            continue

        if not passes_floors:
            continue

        ended = ev["last_date"] < asof_date
        ranked_rows.append({
            "symbol": sym,
            "exchange": ex,
            "avg_dollar_volume": adv,
            "bars": n_bars,
            "first_date": ev["first_date"],
            "last_date": ev["last_date"],
            "asof": asof,
            "ended_before_asof": ended,
            # Within a free-tier 1y window we cannot distinguish "delisted"
            # from "series simply ends"; on paid vintages delisted=1 symbols
            # end before asof and stay ranked with their actual history.
            "delisted": ended,
            # v3 §4.5/§12.8: research-grade vintages have no delisted list, so
            # delist_idx is unavailable for S5 delist-mask logic.
            "delist_idx_available": not research_grade,
        })

    if ranked_rows:
        out = pd.DataFrame(ranked_rows)
        out = out.sort_values(
            ["avg_dollar_volume", "symbol"], ascending=[False, True]
        ).reset_index(drop=True)
        out.insert(0, "rank", range(1, len(out) + 1))
        out = out[RANKED_COLUMNS]
    else:
        out = pd.DataFrame(columns=RANKED_COLUMNS)

    if quarantine_rows:
        quarantine = pd.DataFrame(quarantine_rows)
        quarantine = quarantine.sort_values(
            ["passes_quality_floors", "avg_dollar_volume", "symbol"],
            ascending=[False, False, True],
            na_position="last",
        ).reset_index(drop=True)
        quarantine = quarantine[QUARANTINE_COLUMNS]
        n_post = int(quarantine["passes_quality_floors"].sum())
        shown = ", ".join(quarantine["symbol"].head(10))
        log(
            f"data-quality screen: quarantined {len(quarantine)} common symbol(s) "
            f"({n_post} would pass quality floors): "
            f"{shown}{' …' if len(quarantine) > 10 else ''}"
        )
    else:
        quarantine = pd.DataFrame(columns=QUARANTINE_COLUMNS)

    out.attrs["non_common_excluded"] = n_non_common
    out.attrs["splits_verified_symbols"] = n_splits_verified
    out.attrs["splits_missing_symbols"] = n_splits_missing
    out.attrs["quarantined_by_reason"] = reason_counts
    return out, quarantine


def build_pit_listing(ranked: pd.DataFrame, n: int) -> pd.DataFrame:
    """Top-N by rank. Delisted/ended names that made the cut stay in the listing."""
    top = ranked.head(n).copy()
    top["in_top_n"] = True
    return top


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--warehouse", default=DEFAULT_WAREHOUSE)
    ap.add_argument("--n", type=int, default=1000, help="universe size (default 1000, free tier)")
    ap.add_argument("--asof", default=None,
                    help="point-in-time as-of date YYYY-MM-DD (default: today)")
    ap.add_argument("--min-bars", type=int, default=DEFAULT_MIN_BARS,
                    help=f"minimum number of bars up to asof to be rankable "
                         f"(default {DEFAULT_MIN_BARS}; pre-registered quality floor, "
                         "hermes-tasks #182 — pass 1 to restore the old permissive mode)")
    ap.add_argument("--min-avg-dollar-volume", type=float, default=DEFAULT_MIN_AVG_DOLLAR_VOLUME,
                    help=f"minimum mean daily dollar volume ({DOLLAR_VOLUME_BASIS}) to be "
                         f"rankable, in USD (default {DEFAULT_MIN_AVG_DOLLAR_VOLUME:.0f} = $1M; "
                         "pre-registered floor — pass 0 to disable)")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.warehouse)
    asof = args.asof or date.today().isoformat()
    research, vrec = vintage_is_research(root)
    if research:
        log(f"RESEARCH-GRADE vintage detected (vendor={vrec.get('vendor')!r}, "
            f"research_grade={vrec.get('research_grade')}) — the source vendor has NO "
            f"delisted list: this universe is ACTIVE-ONLY and survivorship-biased "
            f"(v3 §4.5/§12.8). delist_idx is marked UNAVAILABLE for the S5 delist-mask.")
    log(f"loading bars from {root}/eod …")
    bars = load_all_bars(root)
    splits_by_symbol = load_all_splits(root)
    log(f"loaded {len(bars)} bars, {bars['symbol'].nunique() if not bars.empty else 0} symbols, "
        f"{len(splits_by_symbol)} cached splits file(s)")

    ranked, quarantine = build_ranked_table(
        bars,
        asof,
        research_grade=research,
        min_bars=args.min_bars,
        min_avg_dollar_volume=args.min_avg_dollar_volume,
        splits_by_symbol=splits_by_symbol,
    )
    n_non_common = int(ranked.attrs.get("non_common_excluded", 0))
    n_splits_verified = int(ranked.attrs.get("splits_verified_symbols", 0))
    n_splits_missing = int(ranked.attrs.get("splits_missing_symbols", 0))
    reason_counts = dict(ranked.attrs.get("quarantined_by_reason", {}))

    if n_non_common:
        log(f"quality filter: excluded {n_non_common} non-common-stock symbols "
            f"(warrants/units/rights/preferred) from ranking")
    n_quarantined_total = int(len(quarantine))
    n_quarantined_post = (
        int(quarantine["passes_quality_floors"].sum()) if n_quarantined_total else 0
    )
    if args.min_bars > 0 or args.min_avg_dollar_volume > 0:
        log(f"quality floors: min_bars>={args.min_bars}, "
            f"min_avg_dollar_volume>=${args.min_avg_dollar_volume:,.0f} "
            f"(hermes-tasks #182 pre-registered floors)")
    log(f"dollar-volume basis: {DOLLAR_VOLUME_BASIS} "
        f"(splits verified={n_splits_verified}, missing={n_splits_missing})")

    if "rank" in ranked.columns:
        ranked = ranked.drop(columns=["rank"]).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    top = build_pit_listing(ranked, args.n)

    uroot = os.path.join(root, "universe")
    ranked_path = os.path.join(uroot, f"ranked_{asof}.parquet")
    top_path = os.path.join(uroot, f"top{args.n}_pit_{asof}.parquet")
    wh.write_parquet_atomic(ranked, ranked_path)
    wh.write_parquet_atomic(top, top_path)

    candidate_q_path = os.path.join(uroot, f"quarantined_{asof}.parquet")
    quarantined_path: str | None = None
    if n_quarantined_total:
        quarantined_path = candidate_q_path
        wh.write_parquet_atomic(quarantine, quarantined_path)
        log(f"quarantine table written: {quarantined_path} "
            f"({n_quarantined_total} total, {n_quarantined_post} post-floors)")
    elif os.path.exists(candidate_q_path):
        # Remove stale quarantine file from an earlier run so vintage hash stays clean.
        os.remove(candidate_q_path)

    n_ended = int(ranked["ended_before_asof"].sum())
    n_ended_top = int(top["ended_before_asof"].sum())
    if research:
        survivorship = (
            "BIASED — research-grade vintage (vendor has no delisted list): the "
            "ingested universe is active-only, so names delisted inside the window "
            "are absent entirely (v3 §4.5/§12.8). delist_idx unavailable for the "
            "S5 delist-mask: research-vintage trials must NOT claim survivorship-free "
            "S5 registry depth."
        )
        window_note = (f"research-grade vintage vendor={vrec.get('vendor')!r} "
                       f"scope={vrec.get('scope_id')!r} period={vrec.get('period')!r}")
    else:
        survivorship = ("none — names whose series ended before asof stay in the "
                        "ranking with ended_before_asof/delisted flags and their last_date")
        if vrec.get("plan") == "paid":
            window_note = f"paid plan: deep-history EOD window {vrec.get('window')}"
        else:
            window_note = "free tier: past-1-year EOD only; deep-history ranking needs a paid plan"
    manifest = {
        "asof": asof,
        "n": args.n,
        "min_bars": args.min_bars,
        "min_avg_dollar_volume": args.min_avg_dollar_volume,
        "dollar_volume_basis": DOLLAR_VOLUME_BASIS,
        "splits_coverage": {
            "symbols_with_splits_file": n_splits_verified,
            "symbols_without_splits_file": n_splits_missing,
        },
        "non_common_excluded": n_non_common,
        "ranked_path": ranked_path,
        "top_path": top_path,
        "data_quality_screen": {
            "quarantined_total": n_quarantined_total,
            "quarantined_would_pass_floors": n_quarantined_post,
            "quarantined_by_reason": reason_counts,
            "quarantined_path": quarantined_path,
            "thresholds": {
                "eodhd_sentinel_price": EODHD_SENTINEL_PRICE,
                "max_single_day_price_mult": MAX_SINGLE_DAY_PRICE_MULT,
                "max_single_bar_dollar_volume_usd": MAX_SINGLE_BAR_DOLLAR_VOLUME_USD,
                "min_median_raw_close_usd": MIN_MEDIAN_RAW_CLOSE_USD,
            },
        },
        "symbols_ranked": int(len(ranked)),
        "symbols_ended_before_asof": n_ended,
        "top_n_contains_ended_names": n_ended_top,
        "research_grade": research,
        "vendor": vrec.get("vendor"),
        "delist_idx_available": not research,
        "survivorship_bias": survivorship,
        "window_note": window_note,
        "built_utc": datetime.now(timezone.utc).isoformat(),
    }
    wh.write_json_atomic(manifest, os.path.join(uroot, "manifest.json"))
    rec = wh.write_vintage(root, extra={"built_by": "universe", "universe": manifest,
                                       **({"vendor": vrec.get("vendor"),
                                           "research_grade": True} if research else {})})

    log(f"ranked {len(ranked)} symbols as of {asof} ({n_ended} ended before asof — kept, flagged); "
        f"top {len(top)} written ({n_ended_top} ended names inside top-N)")
    if not top.empty:
        head = top.head(5)[["rank", "symbol", "avg_dollar_volume", "last_date", "ended_before_asof"]]
        log("top-5:\n" + head.to_string(index=False))
    log(f"vintage {rec['vintage'][:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
