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
else:
    from . import warehouse as wh

DEFAULT_WAREHOUSE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "data", "warehouse")


def log(msg: str) -> None:
    print(f"[universe {datetime.now(timezone.utc).strftime('%H:%M:%S')}Z] {msg}", flush=True)


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


def build_ranked_table(bars: pd.DataFrame, asof: str) -> pd.DataFrame:
    """Point-in-time ranking as of ``asof`` (YYYY-MM-DD).

    Uses only bars with date <= asof. Average dollar volume = mean(close*volume)
    over the available window up to asof. Symbols whose last bar is before
    asof are kept (no survivorship bias) and flagged.
    """
    if bars.empty:
        return pd.DataFrame(columns=[
            "symbol", "exchange", "avg_dollar_volume", "bars", "first_date", "last_date",
            "asof", "ended_before_asof", "delisted", "rank",
        ])
    df = bars.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[df["date"] <= date.fromisoformat(asof)]
    if df.empty:
        return pd.DataFrame(columns=[
            "symbol", "exchange", "avg_dollar_volume", "bars", "first_date", "last_date",
            "asof", "ended_before_asof", "delisted", "rank",
        ])
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
    out = out.sort_values("avg_dollar_volume", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
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
    ap.add_argument("--min-bars", type=int, default=1,
                    help="minimum number of bars up to asof to be rankable (default 1)")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.warehouse)
    asof = args.asof or date.today().isoformat()
    log(f"loading bars from {root}/eod …")
    bars = load_all_bars(root)
    log(f"loaded {len(bars)} bars, {bars['symbol'].nunique() if not bars.empty else 0} symbols")

    ranked = build_ranked_table(bars, asof)
    ranked = ranked[ranked["bars"] >= args.min_bars].copy()
    if "rank" in ranked.columns:
        ranked = ranked.drop(columns=["rank"])
    ranked = ranked.reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    top = build_pit_listing(ranked, args.n)

    uroot = os.path.join(root, "universe")
    ranked_path = os.path.join(uroot, f"ranked_{asof}.parquet")
    top_path = os.path.join(uroot, f"top{args.n}_pit_{asof}.parquet")
    wh.write_parquet_atomic(ranked, ranked_path)
    wh.write_parquet_atomic(top, top_path)

    n_ended = int(ranked["ended_before_asof"].sum())
    n_ended_top = int(top["ended_before_asof"].sum())
    manifest = {
        "asof": asof,
        "n": args.n,
        "min_bars": args.min_bars,
        "ranked_path": ranked_path,
        "top_path": top_path,
        "symbols_ranked": int(len(ranked)),
        "symbols_ended_before_asof": n_ended,
        "top_n_contains_ended_names": n_ended_top,
        "survivorship_bias": "none — names whose series ended before asof stay in the "
                             "ranking with ended_before_asof/delisted flags and their last_date",
        "window_note": "free tier: past-1-year EOD only; deep-history ranking needs a paid plan",
        "built_utc": datetime.now(timezone.utc).isoformat(),
    }
    wh.write_json_atomic(manifest, os.path.join(uroot, "manifest.json"))
    rec = wh.write_vintage(root, extra={"built_by": "universe", "universe": manifest})

    log(f"ranked {len(ranked)} symbols as of {asof} ({n_ended} ended before asof — kept, flagged); "
        f"top {len(top)} written ({n_ended_top} ended names inside top-N)")
    if not top.empty:
        head = top.head(5)[["rank", "symbol", "avg_dollar_volume", "last_date", "ended_before_asof"]]
        log("top-5:\n" + head.to_string(index=False))
    log(f"vintage {rec['vintage'][:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
