#!/usr/bin/env python3
"""Server-side US equity sector metadata pipeline (SEC EDGAR SIC -> canonical buckets).

Maps public-domain SEC EDGAR 4-digit SIC codes (and optional GICS / vendor sector
strings) onto SIFT's 10 canonical aggregate sector buckets:
    technology, financials, healthcare, energy, industrials,
    consumer, utilities, materials, real_estate, communication

Unmapped or non-US issuers without an SEC SIC default honestly to ``"unclassified"``
(eligible in ``Random (All)``, excluded from sector-specific subspaces — never
fabricated).

Warehouse output (server-side only, gitignored):
    <warehouse>/meta/sectors/<EX>.parquet
    columns: symbol, exchange, sector, sic, source, updated_at

Usage:
    # Bulk SEC EDGAR archive (covers active AND delisted US issuers in one download):
    python3 -m data.sectors --warehouse data/warehouse --submissions-zip /path/to/submissions.zip

    # Live SEC EDGAR API for symbols currently in the warehouse:
    python3 -m data.sectors --warehouse data/warehouse --source sec-edgar \\
        --user-agent "SIFT Research contact@yourdomain.com"

    # Offline CSV seed (columns: symbol, sector or sic):
    python3 -m data.sectors --warehouse data/warehouse --from-csv sectors.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from datetime import datetime, timezone
from typing import Iterable, Optional

import pandas as pd
import requests

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data import warehouse as wh  # type: ignore
else:
    from . import warehouse as wh

DEFAULT_WAREHOUSE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "warehouse",
)

# Canonical aggregate sector buckets (matching sift/cards/face.py::SECTOR_BUCKETS).
SECTOR_BUCKETS: tuple[str, ...] = (
    "technology",
    "financials",
    "healthcare",
    "energy",
    "industrials",
    "consumer",
    "utilities",
    "materials",
    "real_estate",
    "communication",
)

SECTOR_LABELS: dict[str, str] = {
    "technology": "Technology",
    "financials": "Financials",
    "healthcare": "Healthcare",
    "energy": "Energy",
    "industrials": "Industrials",
    "consumer": "Consumer",
    "utilities": "Utilities",
    "materials": "Materials",
    "real_estate": "Real Estate",
    "communication": "Communication",
}

UNCLASSIFIED_SECTOR = "unclassified"

SECTOR_COLUMNS = ["symbol", "exchange", "sector", "sic", "source", "updated_at"]

_SECTOR_ALIASES: dict[str, str] = {
    "tech": "technology",
    "technology": "technology",
    "information technology": "technology",
    "it": "technology",
    "software": "technology",
    "semiconductors": "technology",
    "financial": "financials",
    "financials": "financials",
    "financial services": "financials",
    "banks": "financials",
    "insurance": "financials",
    "health": "healthcare",
    "healthcare": "healthcare",
    "health care": "healthcare",
    "biotech": "healthcare",
    "pharmaceuticals": "healthcare",
    "energy": "energy",
    "oil & gas": "energy",
    "oil and gas": "energy",
    "industrial": "industrials",
    "industrials": "industrials",
    "capital goods": "industrials",
    "transportation": "industrials",
    "consumer": "consumer",
    "consumer discretionary": "consumer",
    "consumer cyclical": "consumer",
    "consumer staples": "consumer",
    "consumer defensive": "consumer",
    "retail": "consumer",
    "utilities": "utilities",
    "utility": "utilities",
    "materials": "materials",
    "basic materials": "materials",
    "chemicals": "materials",
    "mining": "materials",
    "real estate": "real_estate",
    "real_estate": "real_estate",
    "reit": "real_estate",
    "reits": "real_estate",
    "communication": "communication",
    "communications": "communication",
    "communication services": "communication",
    "telecommunications": "communication",
    "telecom": "communication",
    "media": "communication",
}

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSION_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"


def normalize_symbol_stem(symbol: str, exchange: str = "US") -> str:
    """Normalize a symbol (e.g. 'AAPL.US' or 'aapl') to uppercase EOD stem."""
    s = str(symbol or "").strip().upper()
    ex_suffix = f".{exchange.upper()}"
    if s.endswith(ex_suffix):
        s = s[: -len(ex_suffix)]
    return s


def normalize_sector_bucket(raw: object) -> str:
    """Map canonical or GICS/vendor sector strings onto one of SECTOR_BUCKETS,
    or UNCLASSIFIED_SECTOR if unknown/missing."""
    if raw is None:
        return UNCLASSIFIED_SECTOR
    text = " ".join(str(raw).strip().lower().replace("-", " ").split())
    if not text or text in ("nan", "none", "null", "n/a", "unknown", UNCLASSIFIED_SECTOR):
        return UNCLASSIFIED_SECTOR
    if text in SECTOR_BUCKETS:
        return text
    if text in _SECTOR_ALIASES:
        return _SECTOR_ALIASES[text]
    return UNCLASSIFIED_SECTOR


def sic_to_sector_bucket(sic: object) -> str:
    """Deterministic mapping from US SEC 4-digit SIC code to SIFT's 10 canonical
    sector buckets (or UNCLASSIFIED_SECTOR when missing/unmapped)."""
    if sic is None:
        return UNCLASSIFIED_SECTOR
    try:
        s = str(sic).strip()
        if not s:
            return UNCLASSIFIED_SECTOR
        code = int(float(s))
    except (TypeError, ValueError):
        return UNCLASSIFIED_SECTOR
    if code <= 0 or code > 9999:
        return UNCLASSIFIED_SECTOR

    # Energy: Oil & gas extraction, petroleum refining, pipelines, petroleum wholesale
    if (
        1300 <= code <= 1399
        or 2900 <= code <= 2999
        or 4600 <= code <= 4699
        or 5170 <= code <= 5179
    ):
        return "energy"

    # Healthcare: Pharma, biotech, medical instruments, health services
    if (
        2830 <= code <= 2836
        or 3840 <= code <= 3851
        or 8000 <= code <= 8099
    ):
        return "healthcare"

    # Technology: Computers, semiconductors, telecom/electronic equipment, software
    if (
        3570 <= code <= 3579
        or 3660 <= code <= 3699
        or 3810 <= code <= 3829
        or 7370 <= code <= 7379
    ):
        return "technology"

    # Communication: Telecom, broadcasting, publishing/media, motion pictures
    if (
        4800 <= code <= 4899
        or 2700 <= code <= 2799
        or 7800 <= code <= 7841
    ):
        return "communication"

    # Utilities: Electric, gas, water, sanitary services
    if 4900 <= code <= 4999:
        return "utilities"

    # Real Estate: Real estate operators/developers & REITs (SIC 6798)
    if 6500 <= code <= 6553 or code == 6798:
        return "real_estate"

    # Financials: Banking, credit, brokers, insurance, investment offices
    if 6000 <= code <= 6499 or 6700 <= code <= 6799:
        return "financials"

    # Materials: Mining (non-oil), lumber, paper, basic chemicals, metals, stone/glass
    if (
        1000 <= code <= 1299
        or 1400 <= code <= 1499
        or 2400 <= code <= 2499
        or 2600 <= code <= 2699
        or 2800 <= code <= 2899
        or 3200 <= code <= 3399
    ):
        return "materials"

    # Consumer: Agriculture, food/apparel/furniture, autos, retail, hospitality, leisure
    if (
        100 <= code <= 999
        or 2000 <= code <= 2399
        or 2500 <= code <= 2599
        or 3100 <= code <= 3199
        or 3710 <= code <= 3716
        or 5200 <= code <= 5999
        or 7000 <= code <= 7299
        or 7900 <= code <= 7999
    ):
        return "consumer"

    # Industrials: Construction, machinery, aerospace/defense, transport, business/eng services
    if (
        1500 <= code <= 1799
        or 3000 <= code <= 3099
        or 3400 <= code <= 3599
        or 3600 <= code <= 3659
        or 3700 <= code <= 3799
        or 3800 <= code <= 3999
        or 4000 <= code <= 4599
        or 4700 <= code <= 4799
        or 5000 <= code <= 5199
        or 7300 <= code <= 7699
        or 8100 <= code <= 8999
    ):
        return "industrials"

    return UNCLASSIFIED_SECTOR


def sectors_parquet_path(root: str, exchange: str = "US") -> str:
    return wh.sectors_parquet_path(root, exchange)


def load_sector_map(root: str, exchange: str = "US") -> dict[str, str]:
    """Load `{SYMBOL_STEM: canonical_sector}` from `<root>/meta/sectors/<EX>.parquet`.

    Returns `{}` when the file does not exist or cannot be read.
    """
    path = sectors_parquet_path(root, exchange)
    df = wh.read_parquet(path)
    if df is None or df.empty or "symbol" not in df.columns:
        return {}
    out: dict[str, str] = {}
    has_sector = "sector" in df.columns
    has_sic = "sic" in df.columns
    for _, row in df.iterrows():
        stem = normalize_symbol_stem(row["symbol"], exchange=exchange)
        if not stem:
            continue
        sec = normalize_sector_bucket(row["sector"]) if has_sector else UNCLASSIFIED_SECTOR
        if sec == UNCLASSIFIED_SECTOR and has_sic:
            sec = sic_to_sector_bucket(row["sic"])
        out[stem] = sec
    return out


def build_sectors_from_submissions_zip(
    zip_path: str,
    *,
    exchange: str = "US",
    symbols_filter: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Parse SEC EDGAR bulk `submissions.zip` into canonical sector rows.

    Covers both active and delisted US issuers in one offline archive pass.
    """
    allow = (
        {normalize_symbol_stem(s, exchange=exchange) for s in symbols_filter}
        if symbols_filter is not None
        else None
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    by_sym: dict[str, dict] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(".json") or "-submissions-" in name:
                continue
            try:
                doc = json.loads(zf.read(name).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError):
                continue
            sic_raw = doc.get("sic")
            tickers = doc.get("tickers") or []
            if not tickers:
                continue
            sector = sic_to_sector_bucket(sic_raw)
            sic_str = str(sic_raw).strip() if sic_raw is not None else ""
            for tk in tickers:
                stem = normalize_symbol_stem(tk, exchange=exchange)
                if not stem:
                    continue
                if allow is not None and stem not in allow:
                    continue
                prev = by_sym.get(stem)
                # Prefer classified over unclassified when multiple filings share a ticker.
                if prev is None or (
                    prev["sector"] == UNCLASSIFIED_SECTOR and sector != UNCLASSIFIED_SECTOR
                ):
                    by_sym[stem] = {
                        "symbol": f"{stem}.{exchange.upper()}",
                        "exchange": exchange.upper(),
                        "sector": sector,
                        "sic": sic_str,
                        "source": "sec_edgar_submissions_zip",
                        "updated_at": now_iso,
                    }
    if not by_sym:
        return pd.DataFrame(columns=SECTOR_COLUMNS)
    df = pd.DataFrame(by_sym.values()).sort_values("symbol").reset_index(drop=True)
    return df[SECTOR_COLUMNS]


def build_sectors_from_csv(
    csv_path: str,
    *,
    exchange: str = "US",
    source: str = "csv_seed",
) -> pd.DataFrame:
    """Build canonical sector parquet frame from a CSV with `symbol` + (`sector` or `sic`)."""
    raw = pd.read_csv(csv_path)
    if "symbol" not in raw.columns:
        raise ValueError("CSV must contain a 'symbol' column")
    now_iso = datetime.now(timezone.utc).isoformat()
    rows: list[dict] = []
    seen: set[str] = set()
    for _, r in raw.iterrows():
        stem = normalize_symbol_stem(r["symbol"], exchange=exchange)
        if not stem or stem in seen:
            continue
        seen.add(stem)
        sic_val = r["sic"] if "sic" in raw.columns and pd.notna(r["sic"]) else ""
        sec = (
            normalize_sector_bucket(r["sector"])
            if "sector" in raw.columns and pd.notna(r["sector"])
            else UNCLASSIFIED_SECTOR
        )
        if sec == UNCLASSIFIED_SECTOR and sic_val != "":
            sec = sic_to_sector_bucket(sic_val)
        rows.append(
            {
                "symbol": f"{stem}.{exchange.upper()}",
                "exchange": exchange.upper(),
                "sector": sec,
                "sic": str(sic_val).strip(),
                "source": source,
                "updated_at": now_iso,
            }
        )
    if not rows:
        return pd.DataFrame(columns=SECTOR_COLUMNS)
    return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)[SECTOR_COLUMNS]


def fetch_sec_edgar_sectors(
    symbols: Optional[Iterable[str]],
    *,
    user_agent: str,
    exchange: str = "US",
    rate_limit_s: float = 0.12,
    timeout_s: float = 15.0,
) -> pd.DataFrame:
    """Fetch official SEC EDGAR SIC codes for `symbols` (or all SEC tickers if None).

    Requires a descriptive SEC-compliant `User-Agent` (e.g. 'SIFT Research email@domain').
    """
    ua = str(user_agent or "").strip()
    if not ua or "@" not in ua:
        raise ValueError("SEC EDGAR requires a descriptive --user-agent including contact email")

    headers = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}
    resp = requests.get(SEC_TICKERS_URL, headers=headers, timeout=timeout_s)
    resp.raise_for_status()
    ticker_map = resp.json()

    sym_to_cik: dict[str, int] = {}
    for rec in ticker_map.values():
        tk = normalize_symbol_stem(rec.get("ticker", ""), exchange=exchange)
        cik = rec.get("cik_str")
        if tk and cik is not None:
            sym_to_cik[tk] = int(cik)

    targets = (
        sorted({normalize_symbol_stem(s, exchange=exchange) for s in symbols if s})
        if symbols is not None
        else sorted(sym_to_cik.keys())
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    rows: list[dict] = []
    cik_cache: dict[int, tuple[str, str]] = {}

    for stem in targets:
        cik = sym_to_cik.get(stem)
        if cik is None:
            rows.append(
                {
                    "symbol": f"{stem}.{exchange.upper()}",
                    "exchange": exchange.upper(),
                    "sector": UNCLASSIFIED_SECTOR,
                    "sic": "",
                    "source": "sec_edgar_api",
                    "updated_at": now_iso,
                }
            )
            continue
        if cik not in cik_cache:
            if rate_limit_s > 0:
                time.sleep(rate_limit_s)
            r_sub = requests.get(
                SEC_SUBMISSION_URL.format(cik=cik), headers=headers, timeout=timeout_s
            )
            if r_sub.status_code == 200:
                doc = r_sub.json()
                sic_str = str(doc.get("sic") or "").strip()
                cik_cache[cik] = (sic_to_sector_bucket(sic_str), sic_str)
            else:
                cik_cache[cik] = (UNCLASSIFIED_SECTOR, "")
        sector, sic_str = cik_cache[cik]
        rows.append(
            {
                "symbol": f"{stem}.{exchange.upper()}",
                "exchange": exchange.upper(),
                "sector": sector,
                "sic": sic_str,
                "source": "sec_edgar_api",
                "updated_at": now_iso,
            }
        )
    if not rows:
        return pd.DataFrame(columns=SECTOR_COLUMNS)
    return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)[SECTOR_COLUMNS]


def _discover_warehouse_symbols(root: str, exchange: str = "US") -> list[str]:
    eod_dir = os.path.join(root, "eod", exchange.upper())
    if not os.path.isdir(eod_dir):
        return []
    return sorted(
        f[: -len(".parquet")]
        for f in os.listdir(eod_dir)
        if f.endswith(".parquet") and not f.startswith(".")
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build server-side sector metadata parquet.")
    ap.add_argument("--warehouse", default=DEFAULT_WAREHOUSE)
    ap.add_argument("--exchange", default="US")
    ap.add_argument(
        "--source",
        choices=("sec-edgar", "submissions-zip", "csv"),
        default="sec-edgar",
    )
    ap.add_argument("--submissions-zip", default=None, help="Path to SEC EDGAR submissions.zip")
    ap.add_argument("--from-csv", default=None, help="Path to CSV with symbol + (sector|sic)")
    ap.add_argument(
        "--user-agent",
        default=os.environ.get("SEC_EDGAR_USER_AGENT", ""),
        help="SEC-compliant User-Agent header (or set SEC_EDGAR_USER_AGENT)",
    )
    ap.add_argument("--rate-limit", type=float, default=0.12)
    args = ap.parse_args(argv)

    root = os.path.abspath(args.warehouse)
    ex = args.exchange.upper()
    wh_syms = _discover_warehouse_symbols(root, exchange=ex)

    if args.submissions_zip or args.source == "submissions-zip":
        if not args.submissions_zip:
            ap.error("--submissions-zip is required when --source=submissions-zip")
        df = build_sectors_from_submissions_zip(
            args.submissions_zip,
            exchange=ex,
            symbols_filter=wh_syms or None,
        )
    elif args.from_csv or args.source == "csv":
        if not args.from_csv:
            ap.error("--from-csv is required when --source=csv")
        df = build_sectors_from_csv(args.from_csv, exchange=ex)
    else:
        df = fetch_sec_edgar_sectors(
            wh_syms or None,
            user_agent=args.user_agent,
            exchange=ex,
            rate_limit_s=args.rate_limit,
        )

    out_path = sectors_parquet_path(root, exchange=ex)
    wh.write_parquet_atomic(df, out_path)
    counts = df["sector"].value_counts().to_dict() if not df.empty else {}
    print(f"[sectors] wrote {len(df)} rows to {out_path} ({counts})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
