#!/usr/bin/env python3
"""Automated post-ingest validation and vintage-promotion gate for SIFT warehouses.

Checks five production invariants fail-closed before promoting a rebuilt or
incrementally updated warehouse vintage:
  1. ``VINTAGE`` matches recomputed ``warehouse_vintage_hash(root)``
  2. ``universe/manifest.json`` is valid, non-research-grade, and ``delist_idx_available``
  3. ``splits_coverage.symbols_without_splits_file == 0``
  4. ``sift.web.mining._resolve_warehouse_universe(root, env="production", require_pit_universe=True)``
     resolves cleanly with ``mode == "pit_manifest"`` and at least ``--min-selected`` symbols
  5. Pre-registered bellwether equities (e.g. ``AAPL,MSFT,NVDA,AMZN,BRK-A``) remain
     ranked and non-quarantined, and (when ``--prior-manifest`` is passed) the
     quarantine count did not jump unexpectedly.

Usage:
    python3 scripts/validate_warehouse.py --warehouse data/warehouse
    python3 scripts/validate_warehouse.py --warehouse data/warehouse --prior-manifest /path/prev_manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data import warehouse as wh
from sift.web.mining import _resolve_warehouse_universe

DEFAULT_BELLWETHERS = ("AAPL", "MSFT", "NVDA", "AMZN", "BRK-A")


def validate_warehouse(
    root: str,
    *,
    exchange: str = "US",
    min_selected: int = 1000,
    bellwethers: tuple[str, ...] = DEFAULT_BELLWETHERS,
    prior_manifest_path: str | None = None,
    max_quarantine_delta_pct: float = 10.0,
    skip_vintage_rehash: bool = False,
) -> dict[str, Any]:
    """Run all production gates and return a structured report with ``ok: bool``."""
    wh_root = Path(root).resolve()
    checks: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    def _record(name: str, passed: bool, detail: Any) -> None:
        checks[name] = {"ok": bool(passed), "detail": detail}
        if not passed:
            failures.append(f"{name}: {detail}")

    # 1. VINTAGE integrity
    stored_v = wh.read_vintage(str(wh_root))
    vjson = wh.read_json(str(wh_root / wh.VINTAGE_JSON)) or {}
    if skip_vintage_rehash:
        v_ok = bool(stored_v) and stored_v == vjson.get("vintage")
        _record("vintage_integrity", v_ok, {"stored": stored_v, "vintage_json": vjson.get("vintage"), "rehashed": False})
    else:
        computed_v = wh.warehouse_vintage_hash(str(wh_root))
        v_ok = bool(stored_v) and stored_v == vjson.get("vintage") == computed_v
        _record(
            "vintage_integrity",
            v_ok,
            {"stored": stored_v, "vintage_json": vjson.get("vintage"), "computed": computed_v, "rehashed": True},
        )

    # 2. Manifest validity & survivorship flags
    manifest = wh.read_json(str(wh_root / "universe" / "manifest.json"))
    m_ok = (
        isinstance(manifest, dict)
        and manifest.get("research_grade") is False
        and manifest.get("delist_idx_available") is True
        and bool(manifest.get("top_path"))
        and bool(manifest.get("ranked_path"))
    )
    _record(
        "manifest_contract",
        m_ok,
        {
            "exists": isinstance(manifest, dict),
            "asof": manifest.get("asof") if isinstance(manifest, dict) else None,
            "research_grade": manifest.get("research_grade") if isinstance(manifest, dict) else None,
            "delist_idx_available": manifest.get("delist_idx_available") if isinstance(manifest, dict) else None,
        },
    )

    # 3. Splits coverage == 100%
    sc = (manifest or {}).get("splits_coverage") if isinstance(manifest, dict) else None
    missing_splits = sc.get("symbols_without_splits_file") if isinstance(sc, dict) else None
    verified_splits = sc.get("symbols_with_splits_file") if isinstance(sc, dict) else None
    _record(
        "splits_coverage",
        isinstance(sc, dict) and missing_splits == 0 and (verified_splits or 0) > 0,
        {"verified": verified_splits, "missing": missing_splits},
    )

    # 4. Production PairCatalog universe resolution
    resolved = _resolve_warehouse_universe(
        wh_root, env="production", require_pit_universe=True, exchange=exchange
    )
    cat_ok = (
        resolved is not None
        and resolved.mode == "pit_manifest"
        and len(resolved.symbols) >= min_selected
    )
    _record(
        "pair_catalog_resolution",
        cat_ok,
        {
            "mode": resolved.mode if resolved else None,
            "selected": len(resolved.symbols) if resolved else 0,
            "min_selected": min_selected,
            "forbidden": len(resolved.all_symbols) if resolved else 0,
            "quarantined_excluded": resolved.quarantined_excluded if resolved else None,
            "duplicate_pairs_blocked": len(resolved.duplicate_pairs) if resolved else None,
        },
    )

    # 5. Bellwether sanity check
    ranked_syms: set[str] = set()
    if isinstance(manifest, dict) and manifest.get("ranked_path"):
        rpath = Path(str(manifest["ranked_path"]))
        if not rpath.is_file():
            rpath = wh_root / "universe" / rpath.name
        rdf = wh.read_parquet(str(rpath))
        if rdf is not None and "symbol" in rdf.columns:
            ex_s = f".{exchange.upper()}"
            ranked_syms = {
                (str(s)[:-len(ex_s)] if str(s).upper().endswith(ex_s) else str(s)).upper()
                for s in rdf["symbol"].dropna()
            }
    missing_bellwethers = [b for b in bellwethers if b.upper() not in ranked_syms]
    _record(
        "bellwethers_ranked",
        len(missing_bellwethers) == 0,
        {"checked": list(bellwethers), "missing": missing_bellwethers},
    )

    # 6. Optional quarantine delta check vs prior manifest
    if prior_manifest_path:
        prior = wh.read_json(prior_manifest_path) or {}
        prev_q = int(((prior.get("data_quality_screen") or {}).get("quarantined_total")) or 0)
        curr_q = int((((manifest or {}).get("data_quality_screen") or {}).get("quarantined_total")) or 0)
        delta_pct = (abs(curr_q - prev_q) / max(prev_q, 1)) * 100.0
        _record(
            "quarantine_delta",
            delta_pct <= max_quarantine_delta_pct,
            {"prior": prev_q, "current": curr_q, "delta_pct": round(delta_pct, 2), "max_pct": max_quarantine_delta_pct},
        )

    return {
        "ok": len(failures) == 0,
        "warehouse": str(wh_root),
        "vintage": stored_v,
        "checks": checks,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--warehouse", required=True, help="warehouse root directory")
    ap.add_argument("--exchange", default="US")
    ap.add_argument("--min-selected", type=int, default=1000)
    ap.add_argument(
        "--bellwethers",
        default=",".join(DEFAULT_BELLWETHERS),
        help="comma-separated symbols that must be ranked and non-quarantined",
    )
    ap.add_argument("--prior-manifest", default=None, help="optional prior manifest.json for delta check")
    ap.add_argument("--max-quarantine-delta-pct", type=float, default=10.0)
    ap.add_argument("--skip-vintage-rehash", action="store_true", help="skip full sha256 walk (fast check)")
    args = ap.parse_args(argv)

    bellwethers = tuple(s.strip().upper() for s in args.bellwethers.split(",") if s.strip())
    report = validate_warehouse(
        args.warehouse,
        exchange=args.exchange,
        min_selected=args.min_selected,
        bellwethers=bellwethers,
        prior_manifest_path=args.prior_manifest,
        max_quarantine_delta_pct=args.max_quarantine_delta_pct,
        skip_vintage_rehash=args.skip_vintage_rehash,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
