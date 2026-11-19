#!/usr/bin/env python3
"""SIFT point-in-time (PIT) universe builder — top-N by average dollar volume.

v3 scope doc: free tier searches the "top-1,000 liquid" universe (section 7.2)
and "survivorship-bias-free data is mandatory at every tier" (section 4.5 /
12.8). This module builds that universe from the ingested warehouse:

  * ranks every ingested symbol by average dollar volume
    (close * volume) over the available window — point-in-time: the ranking
    is computed as of ``--asof`` using only bars with date <= asof, so no
    future information leaks into the listing;
  * NO survivorship bias: symbols whose series ends before ``asof``
    (delisted within the window) stay in the ranked table with their
    ``last_date`` and an explicit ``delisted``/``ended_before_asof`` flag —
    they are simply ranked by the dollar volume they actually had, and the
    top-N cut is labeled with the as-of date;
  * writes the full ranked table plus the top-N PIT listing as parquet under
    ``data/warehouse/universe/`` and updates the warehouse data-vintage hash.

This is server-side only (v3 section 12): the universe table is an internal
input to the mining engine; clients never receive raw bars.

Usage:
    python3 -m data.universe --warehouse data/warehouse --n 1000
    python3 -m data.universe --asof 2026-11-13 --n 1000
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timezone

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data import warehouse as wh  # type: ignore
    from data.security_type import is_common_stock_symbol  # type: ignore
else:
    from . import warehouse as wh
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


def build_ranked_table(bars: pd.DataFrame, asof: str,
                       research_grade: bool = False,
                       min_bars: int = 0,
                       min_avg_dollar_volume: float = 0.0) -> pd.DataFrame:
    """Point-in-time ranking as of ``asof`` (YYYY-MM-DD).

    Uses only bars with date <= asof. Average dollar volume = mean(close*volume)
    over the available window up to asof. Symbols whose last bar is before
    asof are kept (no survivorship bias) and flagged.

    Quality floors (hermes-tasks #182): non-common-stock symbols (warrants,
    units, rights, preferred — classified by data/security_type.py) are
    ALWAYS excluded; rows with ``bars < min_bars`` or
    ``avg_dollar_volume < min_avg_dollar_volume`` are excluded when the
    caller passes floors > 0 (the CLI defaults are DEFAULT_MIN_BARS /
    DEFAULT_MIN_AVG_DOLLAR_VOLUME).

    ``research_grade=True`` (yfinance/Yahoo vintage): the vendor has NO
    delisted list, so the ingested universe is ACTIVE-ONLY — every ranked
    row gets ``delist_idx_available=False`` and the manifest must say the
    ranking is survivorship-biased (v3 §4.5/§12.8). The ``delisted`` column
    then means only "series ended before asof", NOT "known delisting".
    """
    cols = [
        "symbol", "exchange", "avg_dollar_volume", "bars", "first_date", "last_date",
        "asof", "ended_before_asof", "delisted", "delist_idx_available", "rank",
    ]
    if bars.empty:
        return pd.DataFrame(columns=cols)
    df = bars.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[df["date"] <= date.fromisoformat(asof)]
    if df.empty:
        return pd.DataFrame(columns=cols)
    grp = df.groupby(["symbol", "exchange"], sort=True)
    out = grp.agg(
        avg_dollar_volume=("dollar_volume", "mean"),
        bars=("date", "count"),
        first_date=("date", "min"),
        last_date=("date", "max"),
    ).reset_index()
    out["asof"] = asof
    out["ended_before_asof"] = out["last_date"] < date.fromisoformat(asof)
    # within the free-tier 1-year window we cannot distinguish "delisted" from
    # "series simply ends" — flag honestly rather than inventing a delist date;
    # delisted names from a paid delisted=1 list would populate delisted_at.
    out["delisted"] = out["ended_before_asof"]
    # v3 §4.5/§12.8: on a research-grade (yfinance) vintage there is NO delisted
    # list at all, so a delist index for the S5 delist-mask logic is UNAVAILABLE —
    # record that explicitly instead of silently treating the data as
    # survivorship-free. On production vintages with a delisted=1 list this is True.
    out["delist_idx_available"] = not research_grade
    # quality floors: non-common symbols always out; bars / ADV floors when set
    bare = out["symbol"].str.split(".").str[0]
    common_mask = bare.map(is_common_stock_symbol)
    n_non_common = int((~common_mask).sum())
    out = out.loc[common_mask].copy()
    if min_bars > 0:
        out = out[out["bars"] >= min_bars]
    if min_avg_dollar_volume > 0:
        out = out[out["avg_dollar_volume"] >= min_avg_dollar_volume]
    out = out.sort_values("avg_dollar_volume", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    out.attrs["non_common_excluded"] = n_non_common
    return out


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
                    help=f"minimum mean daily dollar volume (close*volume) to be rankable, "
                         f"in USD (default {DEFAULT_MIN_AVG_DOLLAR_VOLUME:.0f} = $1M; "
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
    log(f"loaded {len(bars)} bars, {bars['symbol'].nunique() if not bars.empty else 0} symbols")

    ranked = build_ranked_table(bars, asof, research_grade=research,
                                min_bars=args.min_bars,
                                min_avg_dollar_volume=args.min_avg_dollar_volume)
    n_non_common = int(ranked.attrs.get("non_common_excluded", 0))
    if n_non_common:
        log(f"quality filter: excluded {n_non_common} non-common-stock symbols "
            f"(warrants/units/rights/preferred) from ranking")
    if args.min_bars > 0 or args.min_avg_dollar_volume > 0:
        log(f"quality floors: min_bars>={args.min_bars}, "
            f"min_avg_dollar_volume>=${args.min_avg_dollar_volume:,.0f} "
            f"(hermes-tasks #182 pre-registered floors)")
    if "rank" in ranked.columns:
        ranked = ranked.drop(columns=["rank"]).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    top = build_pit_listing(ranked, args.n)

    uroot = os.path.join(root, "universe")
    ranked_path = os.path.join(uroot, f"ranked_{asof}.parquet")
    top_path = os.path.join(uroot, f"top{args.n}_pit_{asof}.parquet")
    wh.write_parquet_atomic(ranked, ranked_path)
    wh.write_parquet_atomic(top, top_path)

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
        window_note = ("free tier: past-1-year EOD only; deep-history ranking needs a paid plan")
    manifest = {
        "asof": asof,
        "n": args.n,
        "min_bars": args.min_bars,
        "min_avg_dollar_volume": args.min_avg_dollar_volume,
        "non_common_excluded": n_non_common,
        "ranked_path": ranked_path,
        "top_path": top_path,
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
