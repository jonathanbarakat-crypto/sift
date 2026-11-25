"""Warehouse IO + data-vintage content hash for the SIFT data pipeline.

Layout under the warehouse root (default ``data/warehouse`` in the repo,
gitignored — raw vendor data never leaves the server):

    meta/exchanges-list.json            cached /exchanges-list response
    meta/symbol-list/<EX>-active.json   cached /exchange-symbol-list/<EX>
    meta/symbol-list/<EX>-delisted.json cached ...?delisted=1
    eod/<EX>/<SYMBOL>.parquet           one file per ticker, daily bars
    splits/<EX>/<SYMBOL>.json           cached /splits/<SYMBOL>.<EX> events
    live/<SYMBOL>.parquet               real-time quote snapshots
    universe/ranked_<asof>.parquet      full ranked candidate table
    universe/top<N>_pit_<asof>.parquet  point-in-time top-N listing
    universe/quarantined_<asof>.parquet symbols quarantined by quality screens
    universe/manifest.json              latest build metadata
    state/calls.json                    API call accounting (not vintage data)
    VINTAGE                             the data-vintage content hash
    vintage.json                        hash + build details

The *data vintage* (v3 scope doc section 3.2 / 5: "data vintage = content
hash") is a sha256 over every data file under the warehouse (meta/, eod/,
splits/, live/, universe/ — excluding the call-accounting state and the vintage
files themselves), so any byte change in the warehouse changes the hash.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import pandas as pd

VINTAGE_FILE = "VINTAGE"
VINTAGE_JSON = "vintage.json"
_EXCLUDED_TOP = {VINTAGE_FILE, VINTAGE_JSON, "state"}

EOD_COLUMNS = [
    "date", "open", "high", "low", "close", "adjusted_close",
    "volume", "dollar_volume", "symbol", "exchange",
]


def write_json_atomic(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_parquet_atomic(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def read_parquet(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def eod_parquet_path(root: str, exchange: str, symbol: str) -> str:
    safe = symbol.replace("/", "_").replace("\\", "_")
    return os.path.join(root, "eod", exchange.upper(), f"{safe}.parquet")


def splits_json_path(root: str, exchange: str, symbol: str) -> str:
    safe = symbol.replace("/", "_").replace("\\", "_")
    return os.path.join(root, "splits", exchange.upper(), f"{safe}.json")


def live_parquet_path(root: str, symbol: str) -> str:
    safe = symbol.replace("/", "_").replace("\\", "_")
    return os.path.join(root, "live", f"{safe}.parquet")


def sectors_parquet_path(root: str, exchange: str = "US") -> str:
    return os.path.join(root, "meta", "sectors", f"{exchange.upper()}.parquet")


def warehouse_vintage_hash(root: str) -> str:
    """sha256 over (relpath, sha256(bytes)) of every data file under root.

    Deterministic: files are processed in sorted relpath order. The call
    accounting under ``state/`` and the vintage files themselves are
    excluded — they are not market data.
    """
    h = hashlib.sha256()
    if not os.path.isdir(root):
        return h.hexdigest()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        rel_dir = os.path.relpath(dirpath, root)
        top = rel_dir.split(os.sep)[0]
        if top in _EXCLUDED_TOP and rel_dir != ".":
            dirnames[:] = []
            continue
        for name in sorted(filenames):
            if rel_dir == "." and name in _EXCLUDED_TOP:
                continue
            if name.endswith(".tmp"):
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            with open(full, "rb") as f:
                digest = hashlib.sha256(f.read()).digest()
            h.update(rel.encode("utf-8") + b"\0" + digest + b"\n")
    return h.hexdigest()


def write_vintage(root: str, extra: dict | None = None) -> dict:
    """Compute and persist the data-vintage content hash. Returns the record."""
    record = {
        "vintage": warehouse_vintage_hash(root),
        "algorithm": "sha256 over sorted (relpath, sha256(file)) of warehouse data files",
        "excluded": ["state/", VINTAGE_FILE, VINTAGE_JSON],
        **(extra or {}),
    }
    write_json_atomic(record, os.path.join(root, VINTAGE_JSON))
    with open(os.path.join(root, VINTAGE_FILE), "w", encoding="utf-8") as f:
        f.write(record["vintage"] + "\n")
    return record


def read_vintage(root: str) -> Optional[str]:
    p = os.path.join(root, VINTAGE_FILE)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return f.read().strip() or None
