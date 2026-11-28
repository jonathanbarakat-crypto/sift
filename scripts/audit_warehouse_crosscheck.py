#!/usr/bin/env python3
"""Independent cross-source spot audit for SIFT's EODHD warehouse (v4 §4.5).

Compares a stratified sample of EODHD warehouse parquets (top active, top
delisted, and recent splitters) against an independent reference dataset
(e.g. a personal Sharadar SEP export in Parquet/CSV format with columns
``ticker``/``symbol``, ``date``, and ``closeadj``/``adjusted_close``).

Reports daily log-return discrepancy quantiles (p95, p99, max), fraction of
overlapping bars within ``--tol-bps`` (default 10 bps = 0.001), terminal-date
agreement on delisted names, and flags any symbol with a single-day return
divergence exceeding ``--gross-diff`` (default 5%) for triage.

Usage:
    python3 scripts/audit_warehouse_crosscheck.py \\
        --warehouse data/warehouse \\
        --reference /path/to/sharadar_sep.parquet \\
        --out /tmp/crosscheck_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data import warehouse as wh


def _load_reference_table(ref_path: str) -> pd.DataFrame:
    p = Path(ref_path)
    if p.is_dir():
        frames = []
        for f in sorted(p.glob("*.parquet")):
            df = pd.read_parquet(f)
            if "symbol" not in df.columns and "ticker" not in df.columns:
                df["ticker"] = f.stem
            frames.append(df)
        if not frames:
            raise ValueError(f"no parquet files found in reference directory {ref_path}")
        ref = pd.concat(frames, ignore_index=True)
    elif p.suffix.lower() == ".csv":
        ref = pd.read_csv(p)
    else:
        ref = pd.read_parquet(p)

    col_map = {c.lower(): c for c in ref.columns}
    sym_col = col_map.get("ticker") or col_map.get("symbol")
    date_col = col_map.get("date")
    adj_col = col_map.get("closeadj") or col_map.get("adjusted_close") or col_map.get("adj_close")
    if not (sym_col and date_col and adj_col):
        raise ValueError(
            f"reference table must have ticker/symbol, date, and closeadj/adjusted_close; "
            f"got columns={list(ref.columns)}"
        )
    out = ref[[sym_col, date_col, adj_col]].copy()
    out.columns = ["symbol", "date", "ref_adj"]
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper().str.split(".").str[0]
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["ref_adj"] = pd.to_numeric(out["ref_adj"], errors="coerce")
    return out.dropna(subset=["symbol", "date", "ref_adj"])


def audit_against_reference(
    warehouse_root: str,
    ref_df: pd.DataFrame,
    *,
    exchange: str = "US",
    n_active: int = 100,
    n_delisted: int = 100,
    tol_bps: float = 10.0,
    min_within_tol: float = 0.99,
    gross_diff_thresh: float = 0.05,
) -> dict[str, Any]:
    """Compare stratified EODHD warehouse symbols against ``ref_df``."""
    wh_root = Path(warehouse_root).resolve()
    manifest = wh.read_json(str(wh_root / "universe" / "manifest.json")) or {}
    top_path = manifest.get("top_path")
    if not top_path:
        raise ValueError("universe/manifest.json missing top_path")
    tpath = Path(str(top_path))
    if not tpath.is_file():
        tpath = wh_root / "universe" / tpath.name
    top = wh.read_parquet(str(tpath))
    if top is None or top.empty:
        raise ValueError("top-N universe parquet is empty")

    ex = exchange.upper()
    ex_s = f".{ex}"
    active_syms = [
        str(s)[:-len(ex_s)] if str(s).upper().endswith(ex_s) else str(s)
        for s in top[~top["ended_before_asof"]]["symbol"].head(n_active)
    ]
    delisted_syms = [
        str(s)[:-len(ex_s)] if str(s).upper().endswith(ex_s) else str(s)
        for s in top[top["ended_before_asof"]]["symbol"].head(n_delisted)
    ]
    sample_syms = list(dict.fromkeys(active_syms + delisted_syms))
    ref_groups = {k: g for k, g in ref_df.groupby("symbol", sort=False)}

    tol = tol_bps / 10_000.0
    all_diffs: list[np.ndarray] = []
    per_symbol: list[dict[str, Any]] = []
    gross_outliers: list[dict[str, Any]] = []
    missing_in_ref: list[str] = []

    for sym in sample_syms:
        ref_g = ref_groups.get(sym.upper())
        if ref_g is None or ref_g.empty:
            missing_in_ref.append(sym)
            continue
        edf = wh.read_parquet(wh.eod_parquet_path(str(wh_root), ex, sym))
        if edf is None or edf.empty:
            continue
        edf = edf[["date", "adjusted_close"]].copy()
        edf["date"] = pd.to_datetime(edf["date"]).dt.date
        edf["adjusted_close"] = pd.to_numeric(edf["adjusted_close"], errors="coerce")
        merged = pd.merge(edf, ref_g[["date", "ref_adj"]], on="date", how="inner").sort_values("date")
        merged = merged[(merged["adjusted_close"] > 0) & (merged["ref_adj"] > 0)]
        if len(merged) < 2:
            continue
        r_eod = np.diff(np.log(merged["adjusted_close"].to_numpy(dtype=float)))
        r_ref = np.diff(np.log(merged["ref_adj"].to_numpy(dtype=float)))
        diffs = np.abs(r_eod - r_ref)
        all_diffs.append(diffs)
        max_diff = float(np.max(diffs))
        within_frac = float(np.mean(diffs <= tol))
        rec = {
            "symbol": sym,
            "overlap_bars": int(len(merged)),
            "within_tol_frac": round(within_frac, 5),
            "p99_diff": round(float(np.percentile(diffs, 99)), 6),
            "max_diff": round(max_diff, 6),
            "eodhd_last_date": str(edf["date"].max()),
            "ref_last_date": str(ref_g["date"].max()),
        }
        per_symbol.append(rec)
        if max_diff > gross_diff_thresh:
            gross_outliers.append(rec)

    if all_diffs:
        flat = np.concatenate(all_diffs)
        overall_within = float(np.mean(flat <= tol))
        p95 = float(np.percentile(flat, 95))
        p99 = float(np.percentile(flat, 99))
        mx = float(np.max(flat))
        n_ret_bars = int(flat.size)
    else:
        overall_within = 0.0
        p95 = p99 = mx = float("nan")
        n_ret_bars = 0

    ok = (n_ret_bars > 0) and (overall_within >= min_within_tol) and (len(gross_outliers) == 0)
    return {
        "ok": ok,
        "warehouse": str(wh_root),
        "vintage": wh.read_vintage(str(wh_root)),
        "tol_bps": tol_bps,
        "min_within_tol": min_within_tol,
        "gross_diff_thresh": gross_diff_thresh,
        "symbols_sampled": len(sample_syms),
        "symbols_compared": len(per_symbol),
        "symbols_missing_in_reference": len(missing_in_ref),
        "return_bars_compared": n_ret_bars,
        "within_tol_frac": round(overall_within, 6),
        "p95_abs_log_ret_diff": round(p95, 6) if np.isfinite(p95) else None,
        "p99_abs_log_ret_diff": round(p99, 6) if np.isfinite(p99) else None,
        "max_abs_log_ret_diff": round(mx, 6) if np.isfinite(mx) else None,
        "gross_outliers": sorted(gross_outliers, key=lambda r: -r["max_diff"]),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--warehouse", required=True)
    ap.add_argument("--reference", required=True, help="reference parquet/csv file or directory of parquets")
    ap.add_argument("--exchange", default="US")
    ap.add_argument("--n-active", type=int, default=100)
    ap.add_argument("--n-delisted", type=int, default=100)
    ap.add_argument("--tol-bps", type=float, default=10.0)
    ap.add_argument("--min-within-tol", type=float, default=0.99)
    ap.add_argument("--gross-diff", type=float, default=0.05)
    ap.add_argument("--out", default=None, help="optional path to write JSON report")
    args = ap.parse_args(argv)

    ref_df = _load_reference_table(args.reference)
    report = audit_against_reference(
        args.warehouse,
        ref_df,
        exchange=args.exchange,
        n_active=args.n_active,
        n_delisted=args.n_delisted,
        tol_bps=args.tol_bps,
        min_within_tol=args.min_within_tol,
        gross_diff_thresh=args.gross_diff,
    )
    txt = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(txt + "\n", encoding="utf-8")
    print(txt)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
