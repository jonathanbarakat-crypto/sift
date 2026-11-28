"""Server-side pairs mining catalog, gauntlet coordinator, and SSE event hub.

Hard rules enforced here:
- EODHD warehouse stays strictly server-side (`data/warehouse.py`). Clients
  see ONLY opaque `candidate_id`s and coarse labels — NEVER tickers, symbol
  lists, raw price arrays, or S0 price-level hedge ratios.
- Delisted names are retained in the warehouse panel (union of dates with
  trailing NaN post-delist) and per-leg `delist_idx` is passed to S5 so
  F-DELIST masking runs genuinely.
- Soundness #10: locks a 252-bar frozen holdout out of screening & discovery
  whenever sufficient warehouse history exists, and evaluates `holdout_baseline`
  once on the frozen holdout.
- sift #59: pre-registers / freezes admission epoch 1 with `panel_corr` and
  `m_trials = N*(N-1)//2` (all screened pairs) BEFORE running any trial, so
  every trial faces an identical correlation-adjusted A.3 admission bar.
- Honest scope & S5 depth: refuses silently shrinking a scope below
  `cfg.scope_T(scope_id)` and runs S5 ONLY when full-depth history meets
  `cfg.scope_T("registry_depth")` (4032 bars); otherwise S5 stays honestly
  pending (`on-watch-amber`).
- Concurrent duplicate-config runs coalesce onto one in-flight execution
  instead of invoking `run_gauntlet` twice on the same `trial_id`.
- Channels and live |z| caches are LRU-bounded; SSE streams always terminate
  with `done` (including failed/interrupted runs).
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import math
import os
import queue
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data import universe as pit_universe
from data import warehouse as wh
from sift import flags
from sift.cards import CardError
from sift.cards import copy as card_copy
from sift.cards.face import FaceError
from sift.cards.service import CardLayer
from sift.cards.vault import VaultError
from sift.engine import first_run
from sift.gauntlet import config as cfg
from sift.gauntlet import stats
from sift.gauntlet.runner import GauntletRunner, StageResult
from sift.registry import Ledger, config_hash_of
from sift.web.compliance import ComplianceError, guard_api_payload
from sift.web.telemetry import TelemetryStore

ENGINE_VERSION = "gate1-beta-api-v1"
ALLOWED_SCOPES = ("free_5y_top1k", "free_1y_top1k")
ALLOWED_FORMATIONS = (40, 60, 80)
ALLOWED_OPEN_Z = (1.5, 2.0, 2.5)
ALLOWED_CLOSE_Z = (0.3, 0.5, 0.8)
HOLDOUT_BARS = cfg.BARS_PER_YEAR  # 252-bar frozen holdout (soundness #10)
REGISTRY_DEPTH_BARS = cfg.scope_T("registry_depth")  # 4032 bars

MAX_CACHED_CHANNELS = 128
MAX_CACHED_LIVE_Z = 512

# Three-segment ceremony mapping (v4 §3.1, beta-plan §3 step 2).
CEREMONY_SEGMENT = {
    "s0": "flicker",
    "s1": "flicker",
    "s2": "flicker",
    "s4": "amber_control",
    "s3": "amber_control",
    "s5": "s5_validation",
}

RUNS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS beta_run (
    run_id         TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL,
    trial_id       TEXT NOT NULL,
    candidate_id   TEXT NOT NULL,
    scope_id       TEXT NOT NULL,
    cached         INTEGER NOT NULL DEFAULT 0,
    status         TEXT NOT NULL,          -- running | completed | failed
    verdict        TEXT NOT NULL,          -- pending | killed | survivor
    survivor_class TEXT NOT NULL,
    killed_at      TEXT,
    card_id        TEXT,
    error_message  TEXT,
    created_at     REAL NOT NULL,
    completed_at   REAL
);
CREATE INDEX IF NOT EXISTS idx_beta_run_user ON beta_run(user_id, created_at DESC);
"""


class MiningError(Exception):
    """Fail-closed mining API error with HTTP status code."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = int(status_code)


@dataclass(frozen=True)
class CandidateMeta:
    """Public metadata for one server-side pair candidate (zero tickers/prices)."""

    candidate_id: str
    label: str
    description: str
    family: str
    default_scope: str
    sector_bucket: str
    is_example: bool
    synthetic_example: bool
    data_source: str


@dataclass
class _PairInput:
    """Internal server-side gauntlet input arrays — NEVER serialized to clients."""

    params: dict[str, Any]
    vintage: str
    fixture_x: np.ndarray
    fixture_y: np.ndarray
    beta_true: float
    half_life_true: float
    fixture_spread: np.ndarray | None
    excess_returns: np.ndarray
    perm_series: np.ndarray
    prices: np.ndarray
    full_prices: np.ndarray | None
    full_spread_returns: np.ndarray | None
    delist_idx: np.ndarray | None = None
    holdout_baseline: dict[str, Any] | None = None
    panel_corr: np.ndarray | None = None
    m_screened: int = 1
    current_abs_z: float | None = None


def _spread_returns(prices: np.ndarray) -> np.ndarray:
    rx = np.diff(np.log(prices[:, 0]))
    ry = np.diff(np.log(prices[:, 1]))
    return ry - rx


def _latest_abs_z(spread: np.ndarray, window: int = 60) -> float | None:
    s = np.asarray(spread, float)
    s = s[np.isfinite(s)]
    if s.size < max(window, 10):
        return None
    tail = s[-window:]
    sd = float(np.std(tail, ddof=1))
    if sd <= 1e-12:
        return 0.0
    return round(abs(float((tail[-1] - float(np.mean(tail))) / sd)), 4)


def _pairwise_corr_matrix(log_rets: np.ndarray) -> np.ndarray:
    """Pairwise-complete Pearson correlation matrix tolerating post-delist NaNs."""
    df = pd.DataFrame(log_rets)
    corr = df.corr().to_numpy(dtype=float, copy=True)
    np.fill_diagonal(corr, 1.0)
    return np.nan_to_num(corr, nan=0.0)


@dataclass(frozen=True)
class _WarehouseUniverse:
    """Resolved server-side symbol universe for a warehouse root.

    `symbols` are the screened EOD stems eligible for pair discovery and
    correlation-panel estimation; `all_symbols` includes every symbol stem on
    disk or in universe tables (including quarantined/non-top-N names) so the
    compliance guard blocks all of them from ever appearing in responses.
    """

    symbols: list[str]
    all_symbols: frozenset[str]
    asof: str | None
    mode: str  # "pit_manifest" | "pit_parquet" | "eod_fallback"
    quarantined_excluded: int
    pit_listed_count: int
    duplicate_pairs: frozenset[tuple[str, str]] = frozenset()


def _read_duplicate_cluster_pairs(
    path: Path, exchange: str = "US"
) -> frozenset[tuple[str, str]] | None:
    """Read duplicate/copied-history clusters JSON and return normalized forbidden pair tuples."""
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    clusters = payload.get("clusters")
    if not isinstance(clusters, list):
        return None
    pairs: set[tuple[str, str]] = set()
    for cluster in clusters:
        if not isinstance(cluster, list):
            continue
        stems = sorted({
            _symbol_stem(str(s), exchange).upper()
            for s in cluster
            if str(s).strip()
        })
        for a in range(len(stems)):
            for b in range(a + 1, len(stems)):
                pairs.add((stems[a], stems[b]))
    return frozenset(pairs)


def _symbol_stem(symbol: str, exchange: str = "US") -> str:
    """Normalize a warehouse symbol (e.g. 'AAPL.US' or 'AAPL') to its EOD file stem."""
    s = str(symbol).strip()
    ex_suffix = f".{exchange.upper()}"
    if s.upper().endswith(ex_suffix):
        s = s[: -len(ex_suffix)]
    return s.replace("/", "_").replace("\\", "_")


def _resolve_universe_file(root: Path, raw_path: str | None) -> Path | None:
    """Resolve a manifest-referenced parquet path strictly inside `<root>/universe/`.

    Handles both in-place absolute paths and relocated warehouse directories
    (falling back to `<root>/universe/<basename>`), while refusing any path
    outside `<root>/universe/`.
    """
    if not raw_path or not isinstance(raw_path, str):
        return None
    uroot = (root / "universe").resolve()
    if not uroot.is_dir():
        return None
    cand = Path(raw_path)
    if cand.is_absolute():
        with contextlib.suppress(OSError, ValueError):
            resolved = cand.resolve()
            if resolved.is_file() and resolved.is_relative_to(uroot):
                return resolved
    # Fallback for relocated warehouse mounts: match filename inside <root>/universe/.
    name = cand.name
    if not name or name in (".", ".."):
        return None
    with contextlib.suppress(OSError, ValueError):
        rel = (uroot / name).resolve()
        if rel.is_file() and rel.is_relative_to(uroot):
            return rel
    return None


def _read_universe_symbols(path: Path, exchange: str = "US") -> list[str] | None:
    """Read ordered normalized symbol stems from a PIT or quarantine parquet.

    Returns None if the parquet cannot be read or lacks a `symbol` column
    (allowing callers to fail closed).
    """
    try:
        df = wh.read_parquet(str(path))
    except Exception:  # noqa: BLE001
        return None
    if df is None:
        return None
    if df.empty:
        return []
    if "symbol" not in df.columns:
        return None
    if "exchange" in df.columns:
        ex_u = exchange.upper()
        ex_col = df["exchange"].fillna("").astype(str).str.upper()
        df = df[(ex_col == "") | (ex_col == ex_u)]
    seen: set[str] = set()
    out: list[str] = []
    for raw in df["symbol"].dropna().astype(str):
        stem = _symbol_stem(raw, exchange)
        key = stem.upper()
        if stem and key not in seen:
            seen.add(key)
            out.append(stem)
    return out


def _resolve_warehouse_universe(
    root: Path,
    *,
    env: str,
    require_pit_universe: bool,
    exchange: str = "US",
) -> _WarehouseUniverse | None:
    """Resolve the server-side screening universe for `root` (fail-closed).

    Precedence:
    1. Fail closed in production if `vintage.json` or `universe/manifest.json`
       marks the warehouse `research_grade=True` (survivorship-biased feed).
    2. When `universe/manifest.json` exists: require its `top_path` (PIT
       top-N listing) and subtract any symbols in `quarantined_path` (fail
       closed if a referenced file is missing or unreadable).
    3. When `manifest.json` is absent:
       - If `require_pit_universe` (default in production), fail closed.
       - Otherwise, if `universe/top*_pit_*.parquet` or
         `universe/quarantined_*.parquet` exist, honor the latest PIT/quarantine
         files; if no universe files exist at all, fall back to `eod/<EX>/`.
    """
    eod_dir = root / "eod" / exchange.upper()
    if not eod_dir.is_dir():
        return None
    disk_stems = sorted(p.stem for p in eod_dir.glob("*.parquet"))
    if len(disk_stems) < 2:
        return None
    disk_by_upper = {s.upper(): s for s in disk_stems}
    all_symbols: set[str] = set(disk_stems)

    research_vintage, _ = pit_universe.vintage_is_research(str(root))
    if env == "production" and research_vintage:
        return None

    uroot = root / "universe"
    manifest_path = uroot / "manifest.json"
    if manifest_path.is_file():
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError, ValueError):
            return None
        if not isinstance(manifest, dict):
            return None
        if env == "production" and bool(manifest.get("research_grade")):
            return None

        asof_raw = manifest.get("asof")
        asof = str(asof_raw)[:10] if isinstance(asof_raw, str) and asof_raw.strip() else None

        # Quarantine screen: fail closed if manifest declares quarantined entries
        # or a quarantined_path that cannot be read.
        dq = manifest.get("data_quality_screen")
        dq_dict = dq if isinstance(dq, dict) else {}
        q_raw_path = dq_dict.get("quarantined_path")
        q_total = int(dq_dict.get("quarantined_total") or 0)
        quarantined_upper: set[str] = set()
        if q_raw_path or q_total > 0:
            q_file = _resolve_universe_file(root, q_raw_path if isinstance(q_raw_path, str) else None)
            if q_file is None:
                return None
            q_syms = _read_universe_symbols(q_file, exchange=exchange)
            if q_syms is None:
                return None
            all_symbols.update(q_syms)
            quarantined_upper = {s.upper() for s in q_syms}

        top_file = _resolve_universe_file(
            root, manifest.get("top_path") if isinstance(manifest.get("top_path"), str) else None
        )
        if top_file is None:
            return None
        top_syms = _read_universe_symbols(top_file, exchange=exchange)
        if top_syms is None:
            return None
        all_symbols.update(top_syms)

        selected: list[str] = []
        seen_sel: set[str] = set()
        for stem in top_syms:
            key = stem.upper()
            if key in quarantined_upper or key in seen_sel:
                continue
            disk_stem = disk_by_upper.get(key)
            if disk_stem is not None:
                seen_sel.add(key)
                selected.append(disk_stem)
        if len(selected) < 2:
            return None
        dup_pairs: frozenset[tuple[str, str]] = frozenset()
        dup_meta = manifest.get("duplicate_history_clusters")
        if isinstance(dup_meta, dict):
            dup_raw_path = dup_meta.get("path")
            dup_count = int(dup_meta.get("clusters_count") or 0)
            if dup_raw_path or dup_count > 0:
                dup_file = _resolve_universe_file(
                    root, dup_raw_path if isinstance(dup_raw_path, str) else None
                )
                if dup_file is None:
                    return None
                loaded_dup = _read_duplicate_cluster_pairs(dup_file, exchange=exchange)
                if loaded_dup is None:
                    return None
                dup_pairs = loaded_dup
        return _WarehouseUniverse(
            symbols=selected,
            all_symbols=frozenset(all_symbols),
            asof=asof,
            mode="pit_manifest",
            quarantined_excluded=len(quarantined_upper),
            pit_listed_count=len(top_syms),
            duplicate_pairs=dup_pairs,
        )

    if require_pit_universe:
        return None

    # Non-production fallback when manifest.json is absent: still honor any
    # standalone quarantine / PIT parquet files under <root>/universe/.
    quarantined_upper = set()
    if uroot.is_dir():
        q_files = sorted(uroot.glob("quarantined_*.parquet"))
        if q_files:
            q_syms = _read_universe_symbols(q_files[-1], exchange=exchange)
            if q_syms is None:
                return None
            all_symbols.update(q_syms)
            quarantined_upper = {s.upper() for s in q_syms}

        top_files = sorted(uroot.glob("top*_pit_*.parquet"))
        if top_files:
            top_syms = _read_universe_symbols(top_files[-1], exchange=exchange)
            if top_syms is None:
                return None
            all_symbols.update(top_syms)
            selected = [
                disk_by_upper[s.upper()]
                for s in top_syms
                if s.upper() not in quarantined_upper and s.upper() in disk_by_upper
            ]
            if len(selected) < 2:
                return None
            return _WarehouseUniverse(
                symbols=selected,
                all_symbols=frozenset(all_symbols),
                asof=None,
                mode="pit_parquet",
                quarantined_excluded=len(quarantined_upper),
                pit_listed_count=len(top_syms),
            )

    selected = [s for s in disk_stems if s.upper() not in quarantined_upper]
    if len(selected) < 2:
        return None
    return _WarehouseUniverse(
        symbols=selected,
        all_symbols=frozenset(all_symbols),
        asof=None,
        mode="eod_fallback",
        quarantined_excluded=len(quarantined_upper),
        pit_listed_count=len(selected),
    )


def _load_delist_preserving_panel(
    warehouse_root: str,
    symbols: list[str],
    exchange: str = "US",
    *,
    asof: str | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Load adjusted_close panel retaining delisted names (trailing NaNs kept).

    Unlike `first_run.load_price_matrix`, this does NOT drop names with <50%
    coverage: delisted stocks keep their pre-delist prices and trailing NaNs,
    and `delist_idx[j]` records the first trailing-NaN bar index (or T if the
    symbol stayed listed through the end of the panel). When `asof` is given,
    bars with date > `asof` are truncated for point-in-time discipline.
    """
    frames: dict[str, pd.Series] = {}
    ex_suffix = f".{exchange.upper()}"
    asof_cut = str(asof)[:10] if asof else None
    for sym in symbols:
        base = sym[: -len(ex_suffix)] if sym.upper().endswith(ex_suffix) else sym
        df = wh.read_parquet(wh.eod_parquet_path(warehouse_root, exchange, base))
        if (df is None or df.empty) and base != sym:
            df = wh.read_parquet(wh.eod_parquet_path(warehouse_root, exchange, sym))
        if df is None or df.empty:
            continue
        s = df.set_index("date")["adjusted_close"].astype(float)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        if asof_cut is not None:
            s = s[s.index.astype(str).str.slice(0, 10) <= asof_cut]
        if s.notna().sum() >= 2:
            frames[sym] = s
    if len(frames) < 2:
        return np.zeros((0, 0)), [], np.zeros((0,), dtype=int)
    panel = pd.DataFrame(frames).sort_index()
    arr = panel.to_numpy(dtype=float)
    t_len, n_sym = arr.shape
    delist_idx = np.full(n_sym, t_len, dtype=int)
    for j in range(n_sym):
        finite = np.flatnonzero(np.isfinite(arr[:, j]))
        if finite.size == 0:
            delist_idx[j] = 0
        else:
            last_valid = int(finite[-1])
            if last_valid < t_len - 1:
                delist_idx[j] = last_valid + 1
    return arr, list(panel.columns), delist_idx


def _synth_cointegrated(
    scope_id: str, formation: int, open_z: float, close_z: float
) -> _PairInput:
    """Planted persistent AR(1) pair (synthetic example preset — survives S0-S5)."""
    t_disc = REGISTRY_DEPTH_BARS
    t_total = t_disc + HOLDOUT_BARS
    t_scope = cfg.scope_T(scope_id)
    rng = np.random.Generator(np.random.PCG64(7))
    lx = np.cumsum(rng.normal(0.0, 0.01, t_total))
    e = rng.normal(0.0, 1.0, t_total)
    phi, sig_in = 0.7, 12.0  # matches tests/fixtures/planted.py::f_pair
    s = np.zeros(t_total)
    for t in range(1, t_total):
        s[t] = phi * s[t - 1] + e[t] * sig_in
    px = 100.0 * np.exp(lx)
    py = px * np.exp(s / 100.0)
    prices_all = np.column_stack([px, py])
    # Lock trailing 252 bars as frozen holdout (never enters S0-S5 discovery/validation)
    prices_full = prices_all[:t_disc]
    holdout_prices = prices_all[t_disc:]
    prices_scope = prices_full[-t_scope:]
    x, y = prices_scope[:, 0], prices_scope[:, 1]
    xc, yc = x - x.mean(), y - y.mean()
    beta_true = float(np.dot(xc, yc) / np.dot(xc, xc))
    excess = stats.pairs_strategy_returns(
        prices_scope, open_z=open_z, close_z=close_z
    )["gross"][formation:]
    hb = first_run._holdout_baseline(
        holdout_prices,
        0,
        1,
        formation,
        holdout_bars=HOLDOUT_BARS,
        holdout_start_bar=t_disc,
    )
    return _PairInput(
        params={
            "preset": "example-pair-cointegrated",
            "formation": formation,
            "open_z": open_z,
            "close_z": close_z,
            "holdout_bars": HOLDOUT_BARS,
            "family": "pairs",
            "scope": scope_id,
        },
        vintage="synthetic-example-v1",
        fixture_x=x,
        fixture_y=y,
        beta_true=beta_true,
        half_life_true=-math.log(2.0) / math.log(phi),
        fixture_spread=s[t_disc - t_scope: t_disc],
        excess_returns=excess,
        perm_series=_spread_returns(prices_scope),
        prices=prices_scope,
        full_prices=prices_full,
        full_spread_returns=_spread_returns(prices_full),
        delist_idx=np.array([t_disc, t_disc], dtype=int),
        holdout_baseline=hb,
        panel_corr=np.eye(3, dtype=float),
        m_screened=3,
        current_abs_z=_latest_abs_z(
            s[t_disc - t_scope: t_disc], window=formation
        ),
    )


def _synth_noise(
    scope_id: str, formation: int, open_z: float, close_z: float
) -> _PairInput:
    """Planted random-walk noise pair (synthetic example preset — killed at S1/S2)."""
    t_scope = cfg.scope_T(scope_id)
    rng = np.random.Generator(np.random.PCG64(99))
    lx = np.cumsum(rng.normal(0.0, 0.01, t_scope))
    ly = np.cumsum(rng.normal(0.0, 0.01, t_scope))
    px = 100.0 * np.exp(lx)
    py = 100.0 * np.exp(ly)
    prices_scope = np.column_stack([px, py])
    xc, yc = px - px.mean(), py - py.mean()
    beta_true = float(np.dot(xc, yc) / np.dot(xc, xc))
    spread = ly - lx
    _, hl = stats.estimate_ar1(spread)
    excess = stats.pairs_strategy_returns(
        prices_scope, open_z=open_z, close_z=close_z
    )["gross"][formation:]
    return _PairInput(
        params={
            "preset": "example-pair-noise",
            "formation": formation,
            "open_z": open_z,
            "close_z": close_z,
            "family": "pairs",
            "scope": scope_id,
        },
        vintage="synthetic-example-v1",
        fixture_x=px,
        fixture_y=py,
        beta_true=beta_true,
        half_life_true=hl if math.isfinite(hl) else 10.0,
        fixture_spread=spread,
        excess_returns=excess,
        perm_series=_spread_returns(prices_scope),
        prices=prices_scope,
        full_prices=None,
        full_spread_returns=None,
        panel_corr=np.eye(3, dtype=float),
        m_screened=3,
        current_abs_z=_latest_abs_z(spread, window=formation),
    )


def _synth_regime(
    scope_id: str, formation: int, open_z: float, close_z: float
) -> _PairInput:
    """Planted regime-break pair (passes 5y window, automatic S5 catches decay -> amber)."""
    t_full = REGISTRY_DEPTH_BARS
    t_edge = 7 * 252
    t_scope = min(cfg.scope_T(scope_id), t_edge)
    rng = np.random.Generator(np.random.PCG64(7))
    lx = np.cumsum(rng.normal(0.0, 0.01, t_full))
    e = rng.normal(0.0, 1.0, t_full)
    phi, sig_in, sig_out = 0.70, 6.0, 16.0
    s = np.zeros(t_full)
    for t in range(1, t_full):
        if t < t_edge:
            s[t] = phi * s[t - 1] + e[t] * sig_in
        else:
            s[t] = s[t - 1] + e[t] * sig_out
    px = 100.0 * np.exp(lx)
    py = px * np.exp(s / 100.0)
    prices_full = np.column_stack([px, py])
    prices_scope = prices_full[t_edge - t_scope: t_edge]
    x, y = prices_scope[:, 0], prices_scope[:, 1]
    xc, yc = x - x.mean(), y - y.mean()
    beta_true = float(np.dot(xc, yc) / np.dot(xc, xc))
    excess = stats.pairs_strategy_returns(
        prices_scope, open_z=open_z, close_z=close_z
    )["gross"][formation:]
    return _PairInput(
        params={
            "preset": "example-pair-regime",
            "formation": formation,
            "open_z": open_z,
            "close_z": close_z,
            "family": "pairs",
            "scope": scope_id,
        },
        vintage="synthetic-example-v1",
        fixture_x=x,
        fixture_y=y,
        beta_true=beta_true,
        half_life_true=-math.log(2.0) / math.log(phi),
        fixture_spread=s[t_edge - t_scope: t_edge],
        excess_returns=excess,
        perm_series=_spread_returns(prices_scope),
        prices=prices_scope,
        full_prices=prices_full,
        full_spread_returns=_spread_returns(prices_full),
        delist_idx=np.array([t_full, t_full], dtype=int),
        panel_corr=np.eye(3, dtype=float),
        m_screened=3,
        current_abs_z=_latest_abs_z(s[t_edge - t_scope: t_edge], window=formation),
    )


class PairCatalog:
    """Server-side pair catalog. Raw tickers and prices NEVER leave this class."""

    def __init__(
        self,
        *,
        warehouse_dir: str | Path | None = None,
        allow_synthetic_examples: bool | None = None,
        max_warehouse_pairs: int = 20,
        require_pit_universe: bool | None = None,
    ):
        self.warehouse_dir = Path(warehouse_dir) if warehouse_dir else None
        env = os.environ.get("SIFT_ENV", "dev").strip().lower()
        if allow_synthetic_examples is None:
            env_flag = os.environ.get("SIFT_ALLOW_SYNTHETIC_EXAMPLES", "")
            if env_flag != "":
                allow_synthetic_examples = env_flag not in ("0", "false", "False", "off")
            else:
                allow_synthetic_examples = env != "production"
        self.allow_synthetic_examples = bool(allow_synthetic_examples)
        self.require_pit_universe = (
            (env == "production") if require_pit_universe is None else bool(require_pit_universe)
        )
        self.max_warehouse_pairs = int(max_warehouse_pairs)

        self._meta: dict[str, CandidateMeta] = {}
        self._wh_prices: np.ndarray | None = None
        self._wh_delist_idx: np.ndarray | None = None
        self._wh_symbols: list[str] = []
        self._wh_all_symbols: frozenset[str] = frozenset()
        self._wh_pairs: dict[str, tuple[int, int]] = {}
        self._wh_panel_corr: np.ndarray | None = None
        self._wh_m_screened: int = 1
        self._wh_holdout_start: int | None = None
        self._wh_vintage: str = "no-vintage"
        self._wh_universe_mode: str = "none"
        self._wh_universe_asof: str | None = None
        self._wh_quarantined_excluded: int = 0
        self._wh_pit_listed_count: int = 0
        self._load()

    @property
    def forbidden_symbols(self) -> frozenset[str]:
        """All warehouse ticker stems (including quarantined/non-top-N names),
        blocked by the compliance guard from ever appearing in HTTP responses."""
        return self._wh_all_symbols | frozenset(self._wh_symbols)

    def universe_summary(self) -> dict[str, Any]:
        """Aggregate-only universe metadata for `/api/mining/catalog` (never
        exposes ticker symbols or raw counts tied to individual names)."""
        return {
            "mode": self._wh_universe_mode,
            "asof": self._wh_universe_asof,
            "quarantined_excluded": self._wh_quarantined_excluded,
            "pit_listed_count": self._wh_pit_listed_count,
        }

    def _load(self) -> None:
        env = os.environ.get("SIFT_ENV", "dev").strip().lower()
        wh_loaded = False
        if self.warehouse_dir and self.warehouse_dir.is_dir():
            if env == "production" and os.environ.get(
                "SIFT_EODHD_BETA_LICENSE_CONFIRMED", "0"
            ) not in ("1", "true", "True", "yes"):
                wh_loaded = False
            else:
                wh_loaded = self._load_warehouse(self.warehouse_dir, env=env)

        if not wh_loaded and self.allow_synthetic_examples:
            self._wh_universe_mode = "synthetic"
            self._register_synthetic_presets()

    def _load_warehouse(self, root: Path, *, env: str) -> bool:
        resolved = _resolve_warehouse_universe(
            root,
            env=env,
            require_pit_universe=self.require_pit_universe,
            exchange="US",
        )
        if resolved is None:
            return False
        self._wh_all_symbols = resolved.all_symbols
        prices, kept, delist_idx = _load_delist_preserving_panel(
            str(root), resolved.symbols, exchange="US", asof=resolved.asof
        )
        if prices.shape[1] < 2 or prices.shape[0] < cfg.scope_T("free_1y_top1k"):
            return False
        self._wh_prices = prices
        self._wh_delist_idx = delist_idx
        self._wh_symbols = kept
        self._wh_vintage = wh.read_vintage(str(root)) or wh.warehouse_vintage_hash(str(root))
        self._wh_universe_mode = resolved.mode
        self._wh_universe_asof = resolved.asof
        self._wh_quarantined_excluded = resolved.quarantined_excluded
        self._wh_pit_listed_count = resolved.pit_listed_count

        # Lock the trailing 252 bars out of pair screening (soundness #10:
        # screening correlation NEVER peeks into the frozen holdout).
        if prices.shape[0] >= cfg.scope_T("free_1y_top1k") + HOLDOUT_BARS:
            self._wh_holdout_start = prices.shape[0] - HOLDOUT_BARS
            screen_prices = prices[: self._wh_holdout_start]
        else:
            self._wh_holdout_start = None
            screen_prices = prices

        with np.errstate(divide="ignore", invalid="ignore"):
            log_p = np.where(screen_prices > 0.0, np.log(screen_prices), np.nan)
        rets = np.diff(log_p, axis=0)
        corr = _pairwise_corr_matrix(rets)
        self._wh_panel_corr = corr
        n_sym = corr.shape[0]
        iu = np.triu_indices(n_sym, k=1)
        self._wh_m_screened = max(int(iu[0].size), 1)
        order = np.argsort(-np.abs(np.nan_to_num(corr[iu])))
        dup_pairs = resolved.duplicate_pairs
        rank_idx = 0
        for k_raw in order:
            if rank_idx >= self.max_warehouse_pairs:
                break
            k = int(k_raw)
            i, j = int(iu[0][k]), int(iu[1][k])
            if dup_pairs:
                a_key = _symbol_stem(kept[i], "US").upper()
                b_key = _symbol_stem(kept[j], "US").upper()
                if (a_key, b_key) in dup_pairs or (b_key, a_key) in dup_pairs:
                    continue
            cid = f"pair-{rank_idx + 1:03d}"
            self._wh_pairs[cid] = (i, j)
            self._meta[cid] = CandidateMeta(
                candidate_id=cid,
                label=f"Warehouse pair #{rank_idx + 1} (cross-sectional co-movement)",
                description=(
                    "Pre-registered server-side pair candidate from the production "
                    "warehouse. Symbols stay server-side; gauntlet stats only."
                ),
                family="pairs",
                default_scope="free_5y_top1k",
                sector_bucket="cross_sectional_pairs_aggregate",
                is_example=(rank_idx == 0),
                synthetic_example=False,
                data_source="eodhd_warehouse",
            )
            rank_idx += 1
        return len(self._wh_pairs) > 0

    def _register_synthetic_presets(self) -> None:
        presets = (
            CandidateMeta(
                candidate_id="example-pair-cointegrated",
                label="Guided example — persistent co-movement pair (synthetic)",
                description=(
                    "Pre-loaded guided first-run example on synthetic series "
                    "designed to exercise S0-S4 plus automatic S5 full-depth validation."
                ),
                family="pairs",
                default_scope="free_5y_top1k",
                sector_bucket="cross_sectional_pairs_aggregate",
                is_example=True,
                synthetic_example=True,
                data_source="synthetic_example",
            ),
            CandidateMeta(
                candidate_id="example-pair-noise",
                label="Honest kill example — uncorrelated noise pair (synthetic)",
                description=(
                    "Independent synthetic series illustrating an honest gauntlet "
                    "kill at the discovery screen or permutation null."
                ),
                family="pairs",
                default_scope="free_5y_top1k",
                sector_bucket="cross_sectional_pairs_aggregate",
                is_example=False,
                synthetic_example=True,
                data_source="synthetic_example",
            ),
            CandidateMeta(
                candidate_id="example-pair-regime",
                label="Regime-decay example — passes 5y, flagged by automatic S5 (synthetic)",
                description=(
                    "Synthetic pair whose coupling holds in a recent window but "
                    "decays over full history, illustrating automatic S5 amber review."
                ),
                family="pairs",
                default_scope="free_5y_top1k",
                sector_bucket="cross_sectional_pairs_aggregate",
                is_example=False,
                synthetic_example=True,
                data_source="synthetic_example",
            ),
        )
        for m in presets:
            self._meta[m.candidate_id] = m

    def list_public(self) -> list[dict[str, Any]]:
        return [
            {
                "candidate_id": m.candidate_id,
                "label": m.label,
                "description": m.description,
                "family": m.family,
                "default_scope": m.default_scope,
                "sector_bucket": m.sector_bucket,
                "is_example": m.is_example,
                "synthetic_example": m.synthetic_example,
                "data_source": m.data_source,
            }
            for m in self._meta.values()
        ]

    def default_example_id(self) -> str:
        for m in self._meta.values():
            if m.is_example:
                return m.candidate_id
        if self._meta:
            return next(iter(self._meta))
        raise MiningError(
            "No server-side warehouse or beta candidate catalog is configured.",
            503,
        )

    def build_input(
        self,
        candidate_id: str,
        *,
        scope_id: str,
        formation: int,
        open_z: float,
        close_z: float,
    ) -> tuple[CandidateMeta, _PairInput]:
        meta = self._meta.get(candidate_id)
        if meta is None:
            # Never echo user-supplied candidate_id back in error text.
            raise MiningError("Unknown candidate_id.", 404)

        if candidate_id == "example-pair-cointegrated":
            return meta, _synth_cointegrated(scope_id, formation, open_z, close_z)
        if candidate_id == "example-pair-noise":
            return meta, _synth_noise(scope_id, formation, open_z, close_z)
        if candidate_id == "example-pair-regime":
            return meta, _synth_regime(scope_id, formation, open_z, close_z)

        assert self._wh_prices is not None
        assert self._wh_delist_idx is not None
        i, j = self._wh_pairs[candidate_id]
        si, sj = self._wh_symbols[i], self._wh_symbols[j]
        full_pair = self._wh_prices[:, [i, j]]
        pair_delist = np.array([self._wh_delist_idx[i], self._wh_delist_idx[j]], dtype=int)

        required_bars = cfg.scope_T(scope_id)
        # Lock 252-bar frozen holdout if enough history exists beyond required_bars.
        holdout_baseline: dict[str, Any] | None = None
        if self._wh_holdout_start is not None and self._wh_holdout_start >= required_bars:
            disc_full = full_pair[: self._wh_holdout_start]
            holdout_slice = full_pair[self._wh_holdout_start:]
            finite_h = holdout_slice[np.all(np.isfinite(holdout_slice), axis=1)]
            if finite_h.shape[0] >= 4:
                holdout_baseline = first_run._holdout_baseline(
                    finite_h,
                    0,
                    1,
                    formation,
                    holdout_bars=HOLDOUT_BARS,
                    holdout_start_bar=int(self._wh_holdout_start),
                )
            holdout_used = HOLDOUT_BARS
        else:
            disc_full = full_pair
            holdout_used = 0

        # Leading valid pre-delist discovery bars for the pair (no post-delist leak).
        valid_mask = np.all(np.isfinite(disc_full), axis=1)
        valid_disc = disc_full[valid_mask]
        if valid_disc.shape[0] < required_bars:
            # Fail closed: NEVER silently shrink a scope window under a false label.
            raise MiningError(
                "Insufficient warehouse history for the requested scope.", 400
            )
        scope_pair = valid_disc[-required_bars:]

        ps = first_run.pair_series(scope_pair, 0, 1, formation)
        x, y = scope_pair[:, 0], scope_pair[:, 1]
        _, hl = stats.estimate_ar1(ps["spread"])
        excess = stats.pairs_strategy_returns(
            scope_pair, open_z=open_z, close_z=close_z
        )["gross"][formation:]

        # S5 requires genuine full registry_depth history (>= 4032 bars).
        # Never run S5's 3.0 hurdle on a 5y+1-bar window!
        has_full_depth = full_pair.shape[0] >= REGISTRY_DEPTH_BARS
        params: dict[str, Any] = {
            "pair": [si, sj],
            "formation": formation,
            "open_z": open_z,
            "close_z": close_z,
            "family": "pairs",
            "scope": scope_id,
        }
        if holdout_used > 0:
            params["holdout_bars"] = holdout_used

        return meta, _PairInput(
            params=params,
            vintage=self._wh_vintage,
            fixture_x=x,
            fixture_y=y,
            beta_true=ps["beta_levels"],
            half_life_true=hl,
            fixture_spread=ps["spread"],
            excess_returns=excess,
            perm_series=ps["spread_returns"],
            prices=scope_pair,
            full_prices=full_pair if has_full_depth else None,
            full_spread_returns=_spread_returns(full_pair) if has_full_depth else None,
            delist_idx=pair_delist if has_full_depth else None,
            holdout_baseline=holdout_baseline,
            panel_corr=self._wh_panel_corr,
            m_screened=self._wh_m_screened,
            current_abs_z=_latest_abs_z(ps["spread"], window=formation),
        )


def _stage_to_public(sr: StageResult | sqlite3.Row) -> dict[str, Any]:
    if isinstance(sr, StageResult):
        stage = sr.stage
        status = sr.status
        statistic = sr.statistic
        threshold = sr.threshold
        p_value = sr.p_value
        dsr = sr.dsr
        q_bh = sr.q_bh
        n_stratum = sr.n_stratum
        n_eff = sr.n_eff
    else:
        stage = str(sr["stage"])
        status = str(sr["status"])
        statistic = sr["statistic"]
        threshold = sr["threshold"]
        p_value = sr["p_value"]
        dsr = sr["dsr"]
        q_bh = sr["q_bh"]
        n_stratum = sr["n_stratum"]
        cols = set(sr.keys())
        n_eff = sr["n_eff"] if "n_eff" in cols else None

    # Security: NEVER expose S0 raw price-level hedge ratio (`beta_levels`),
    # which fingerprints the hidden pair's price levels. S0 is a pass/fail gate.
    if stage == "s0":
        statistic = None
        threshold = None

    return {
        "stage": stage,
        "ceremony_segment": CEREMONY_SEGMENT.get(stage, "flicker"),
        "status": status,
        "statistic": None
        if statistic is None or not math.isfinite(float(statistic))
        else round(float(statistic), 4),
        "threshold": None
        if threshold is None or not math.isfinite(float(threshold))
        else round(float(threshold), 4),
        "p_value": None
        if p_value is None or not math.isfinite(float(p_value))
        else round(float(p_value), 6),
        "dsr": None
        if dsr is None or not math.isfinite(float(dsr))
        else round(float(dsr), 4),
        "q_bh": None
        if q_bh is None or not math.isfinite(float(q_bh))
        else round(float(q_bh), 4),
        "n_stratum": n_stratum,
        "n_eff": None
        if n_eff is None or not math.isfinite(float(n_eff))
        else round(float(n_eff), 2),
    }


def _safe_verdict_copy(
    verdict: str,
    *,
    killed_at: str | None = None,
    survivor_class: str | None = None,
    handle: str | None = None,
) -> str:
    try:
        return card_copy.verdict_framing(
            verdict, killed_at=killed_at, survivor_class=survivor_class, handle=handle
        )
    except card_copy.CopyError:
        if verdict == "killed":
            stage = (killed_at or "screen").upper()
            return (
                f"Killed at stage {stage} — this is the product working. "
                "Most candidates do not survive the pre-registered stress test."
            )
        if verdict == "survivor":
            return (
                "Passed the pre-registered stress test screen. "
                "SIFT shows evidence; you make the calls."
            )
        return "Trial registered and awaiting stage completion."


class _RunChannel:
    """Thread-safe event buffer + subscriber broadcast for SSE streaming."""

    def __init__(self, *, extra_symbols: frozenset[str] = frozenset()) -> None:
        self._lock = threading.Lock()
        self._extra_symbols = extra_symbols
        self._events: list[tuple[str, dict[str, Any]]] = []
        self._done = False
        self._subscribers: list[queue.Queue[tuple[str, dict[str, Any]] | None]] = []

    def publish(self, event_name: str, data: dict[str, Any]) -> bool:
        """Compliance-check and broadcast an event. Never raises ComplianceError
        out to workers (converts into a safe non-leaking error event instead)."""
        try:
            safe_data = guard_api_payload(
                data, extra_symbols=self._extra_symbols
            )
        except ComplianceError:
            event_name = "error"
            safe_data = {"error": ComplianceError.public_message}
        with self._lock:
            if self._done:
                return False
            self._events.append((event_name, safe_data))
            for q in list(self._subscribers):
                q.put((event_name, safe_data))
            if event_name == "done":
                self._done = True
                for q in list(self._subscribers):
                    q.put(None)
        return True

    def finish(self, run_id: str, status: str) -> None:
        """Guarantee terminal `done` event is published once so streams never hang."""
        with self._lock:
            if self._done:
                return
            item = ("done", {"run_id": run_id, "status": status})
            self._events.append(item)
            self._done = True
            for q in list(self._subscribers):
                q.put(item)
                q.put(None)

    def subscribe(self) -> queue.Queue[tuple[str, dict[str, Any]] | None]:
        q: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue()
        with self._lock:
            for item in self._events:
                q.put(item)
            if self._done:
                q.put(None)
            else:
                self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[tuple[str, dict[str, Any]] | None]) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)


@dataclass
class _InFlightTrial:
    """Tracks all concurrent beta_run waiters attached to one executing trial_id."""

    trial_id: str
    waiters: list[tuple[str, str]] = field(default_factory=list)  # (run_id, user_id)


class MiningStore:
    """SQLite persistence for beta mining runs (`beta_run` table in app.db)."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        con: sqlite3.Connection | None = None,
    ):
        self.db_path = Path(db_path)
        self._owns_con = con is None
        if con is not None:
            self.con = con
        else:
            if str(self.db_path.parent) not in ("", "."):
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.con = sqlite3.connect(str(self.db_path), timeout=10.0)
            self.con.row_factory = sqlite3.Row
            self.con.execute("PRAGMA journal_mode = WAL")
            self.con.execute("PRAGMA busy_timeout = 10000")
        self.con.executescript(RUNS_SCHEMA_SQL)
        self.con.commit()

    def close(self) -> None:
        if self._owns_con:
            self.con.close()

    def fail_orphaned_runs(self, *, now: float | None = None) -> int:
        """On startup, mark any orphaned `running` rows from prior crashes failed."""
        ts = time.time() if now is None else float(now)
        try:
            cur = self.con.execute(
                "UPDATE beta_run SET status='failed',"
                " error_message='Run interrupted before completion.',"
                " completed_at=?"
                " WHERE status='running'",
                (ts,),
            )
            self.con.commit()
            return int(cur.rowcount)
        except Exception:
            self.con.rollback()
            raise

    def insert_run(
        self,
        *,
        run_id: str,
        user_id: str,
        trial_id: str,
        candidate_id: str,
        scope_id: str,
        cached: bool,
        status: str,
        verdict: str,
        survivor_class: str,
        killed_at: str | None = None,
        card_id: str | None = None,
        error_message: str | None = None,
        now: float | None = None,
    ) -> None:
        ts = time.time() if now is None else float(now)
        done_ts = ts if status in ("completed", "failed") else None
        try:
            self.con.execute(
                "INSERT INTO beta_run"
                " (run_id, user_id, trial_id, candidate_id, scope_id, cached,"
                "  status, verdict, survivor_class, killed_at, card_id, error_message,"
                "  created_at, completed_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    user_id,
                    trial_id,
                    candidate_id,
                    scope_id,
                    1 if cached else 0,
                    status,
                    verdict,
                    survivor_class,
                    killed_at,
                    card_id,
                    error_message,
                    ts,
                    done_ts,
                ),
            )
            self.con.commit()
        except Exception:
            self.con.rollback()
            raise

    def complete_run(
        self,
        run_id: str,
        *,
        status: str,
        verdict: str,
        survivor_class: str,
        killed_at: str | None = None,
        card_id: str | None = None,
        error_message: str | None = None,
        now: float | None = None,
    ) -> None:
        ts = time.time() if now is None else float(now)
        try:
            self.con.execute(
                "UPDATE beta_run SET status=?, verdict=?, survivor_class=?,"
                " killed_at=?, card_id=?, error_message=?, completed_at=?"
                " WHERE run_id=?",
                (
                    status,
                    verdict,
                    survivor_class,
                    killed_at,
                    card_id,
                    error_message,
                    ts,
                    run_id,
                ),
            )
            self.con.commit()
        except Exception:
            self.con.rollback()
            raise

    def get_run(self, run_id: str, user_id: str) -> sqlite3.Row | None:
        return self.con.execute(
            "SELECT * FROM beta_run WHERE run_id=? AND user_id=?",
            (run_id, user_id),
        ).fetchone()

    def list_runs_for_user(self, user_id: str, limit: int = 100) -> list[sqlite3.Row]:
        return self.con.execute(
            "SELECT * FROM beta_run WHERE user_id=?"
            " ORDER BY created_at DESC LIMIT ?",
            (user_id, int(limit)),
        ).fetchall()

    def user_has_trial(self, user_id: str, trial_id: str) -> bool:
        row = self.con.execute(
            "SELECT 1 FROM beta_run WHERE user_id=? AND trial_id=? LIMIT 1",
            (user_id, trial_id),
        ).fetchone()
        return row is not None

    def active_run_count(self, user_id: str) -> int:
        row = self.con.execute(
            "SELECT COUNT(*) FROM beta_run WHERE user_id=? AND status='running'",
            (user_id,),
        ).fetchone()
        return int(row[0])


class MiningManager:
    """Orchestrates pre-registered pairs gauntlet runs + bounded SSE channels."""

    def __init__(
        self,
        *,
        ledger: Ledger,
        card_layer: CardLayer,
        telemetry: TelemetryStore,
        runs: MiningStore,
        catalog: PairCatalog,
        commitment_secret: bytes,
        s2_B: int = cfg.S2_B,
        fast_call=None,
    ):
        self.ledger = ledger
        self.card_layer = card_layer
        self.telemetry = telemetry
        self.runs = runs
        self.catalog = catalog
        self._commitment_secret = bytes(commitment_secret)
        self.s2_B = int(s2_B)
        # Optional dispatcher to run `runs` / `telemetry` writes on the fast DB thread
        # so MiningManager (running on the engine thread) never touches fast-DB connections.
        self._fast_call = fast_call if fast_call is not None else (lambda fn, *a, **kw: fn(*a, **kw))

        self._lock = threading.Lock()
        self._channels: OrderedDict[str, _RunChannel] = OrderedDict()
        self._live_z: OrderedDict[str, float] = OrderedDict()
        self._in_flight: dict[str, _InFlightTrial] = {}
        self._frozen_epochs: set[str] = set()

    def _config_commitment(self, config_hash: str) -> str:
        """Keyed HMAC commitment of config_hash for telemetry (prevents ticker
        dictionary attacks against exported telemetry events)."""
        return hmac.new(
            self._commitment_secret,
            config_hash.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _put_channel(self, run_id: str, ch: _RunChannel) -> None:
        with self._lock:
            self._channels[run_id] = ch
            self._channels.move_to_end(run_id)
            while len(self._channels) > MAX_CACHED_CHANNELS:
                self._channels.popitem(last=False)

    def _get_channel_mem(self, run_id: str) -> _RunChannel | None:
        with self._lock:
            ch = self._channels.get(run_id)
            if ch is not None:
                self._channels.move_to_end(run_id)
            return ch

    def _put_live_z(self, card_id: str, val: float) -> None:
        with self._lock:
            self._live_z[card_id] = val
            self._live_z.move_to_end(card_id)
            while len(self._live_z) > MAX_CACHED_LIVE_Z:
                self._live_z.popitem(last=False)

    def current_abs_z(self, card_id: str) -> float | None:
        with self._lock:
            return self._live_z.get(card_id)

    def validate_config(
        self,
        *,
        candidate_id: str | None,
        example: bool,
        scope_id: str,
        formation: int,
        open_z: float,
        close_z: float,
    ) -> tuple[str, str, int, float, float]:
        cid = (
            self.catalog.default_example_id()
            if (example or not candidate_id)
            else str(candidate_id)
        )
        if scope_id not in ALLOWED_SCOPES:
            raise MiningError("Unsupported scope_id.", 400)
        if formation not in ALLOWED_FORMATIONS:
            raise MiningError("Unsupported formation window.", 400)
        if open_z not in ALLOWED_OPEN_Z:
            raise MiningError("Unsupported open_z threshold.", 400)
        if close_z not in ALLOWED_CLOSE_Z:
            raise MiningError("Unsupported close_z threshold.", 400)
        return cid, scope_id, int(formation), float(open_z), float(close_z)

    def _ensure_admission_epoch(
        self, runner: GauntletRunner, scope_id: str, pair_in: _PairInput
    ) -> None:
        """sift #59: pre-register/freeze epoch 1 with panel_corr + m_screened
        before any trial evaluates S3 so identical trials face identical bars."""
        if scope_id in self._frozen_epochs:
            return
        existing = self.ledger.stratum_epoch_stats(scope_id, epoch=1)
        if existing is not None:
            self._frozen_epochs.add(scope_id)
            return
        corr = pair_in.panel_corr
        if corr is not None and corr.shape[0] >= 2:
            runner.set_admission_epoch(
                scope_id,
                epoch=1,
                panel_corr=corr,
                m_trials=max(pair_in.m_screened, 1),
                detail={"vintage": pair_in.vintage[:16], "m_screened": pair_in.m_screened},
            )
        else:
            runner.set_admission_epoch(
                scope_id,
                epoch=1,
                n_eff=float(max(pair_in.m_screened, 1)),
                rho_bar=0.0,
                m_trials=max(pair_in.m_screened, 1),
                method="a3_avg_corr",
                detail={"vintage": pair_in.vintage[:16], "m_screened": pair_in.m_screened},
            )
        self._frozen_epochs.add(scope_id)

    def prepare_run(
        self,
        *,
        user_id: str,
        candidate_id: str,
        scope_id: str,
        formation: int,
        open_z: float,
        close_z: float,
    ) -> tuple[dict[str, Any], _PairInput | None]:
        """Pre-register trial BEFORE any gauntlet computation (v3 §5).

        Handles four cases cleanly without double-execution or race errors:
        1. New unique trial -> register in ledger, log `trial_registered` with
           HMAC `config_commitment`, mark in-flight, return `pair_in` to run.
        2. Already decided in ledger -> replay cached stages & verdict immediately
           (without leaking cross-user cache info or another user's card_id).
        3. Currently in-flight (`_in_flight[trial_id]`) -> coalesce this run
           as a waiter on the active execution (NEVER calls `run_gauntlet` twice!).
        4. Orphaned partially-decided pending trial (prior crash mid-gauntlet) ->
           fail cleanly instead of crashing on append-only stage constraints.
        """
        _, pair_in = self.catalog.build_input(
            candidate_id,
            scope_id=scope_id,
            formation=formation,
            open_z=open_z,
            close_z=close_z,
        )
        cfg_hash = config_hash_of(pair_in.params)
        existing = self.ledger.con.execute(
            "SELECT trial_id, verdict, survivor_class FROM trial"
            " WHERE family='pairs' AND scope_id=? AND config_hash=? AND data_vintage_hash=?",
            (scope_id, cfg_hash, pair_in.vintage),
        ).fetchone()
        ledger_cached = existing is not None

        trial = self.ledger.register(
            "pairs",
            scope_id,
            pair_in.params,
            pair_in.vintage,
            provenance="user_initiated",
            engine_version=ENGINE_VERSION,
            discoverer_handle=user_id,
            beta=True,
        )
        user_already_ran = self._fast_call(
            self.runs.user_has_trial, user_id, trial.trial_id
        )
        # Only expose `cached: true` when THIS user already ran the config
        # (prevents cross-user oracle revealing another user's research).
        cached_for_user = bool(ledger_cached and user_already_ran)

        run_id = f"run-{uuid.uuid4().hex[:12]}"
        ch = _RunChannel(extra_symbols=self.catalog.forbidden_symbols)
        self._put_channel(run_id, ch)

        if not ledger_cached:
            self._fast_call(
                self.telemetry.record,
                "trial_registered",
                {
                    "trial_id": trial.trial_id,
                    "scope_id": scope_id,
                    "provenance": "user_initiated",
                    # Keyed HMAC commitment — never raw SHA-256 of pair symbols.
                    "config_commitment": self._config_commitment(trial.config_hash),
                    "beta": True,
                },
                user_id=user_id,
            )

        # Case 2: Trial already decided in ledger -> instant cached replay.
        if ledger_cached and trial.verdict != "pending":
            stages = [_stage_to_public(r) for r in self.ledger.stages(trial.trial_id)]
            killed_at = next(
                (
                    s["stage"]
                    for s in stages
                    if s["status"] == "fail" and s["stage"] in ("s0", "s1", "s2")
                ),
                None,
            )
            owned_card_id = self._find_owned_card_for_trial(trial.trial_id, user_id)
            self._fast_call(
                self.runs.insert_run,
                run_id=run_id,
                user_id=user_id,
                trial_id=trial.trial_id,
                candidate_id=candidate_id,
                scope_id=scope_id,
                cached=cached_for_user,
                status="completed",
                verdict=trial.verdict,
                survivor_class=trial.survivor_class,
                killed_at=killed_at,
                card_id=owned_card_id,
            )
            try:
                ch.publish(
                    "registered",
                    {
                        "run_id": run_id,
                        "trial_id": trial.trial_id,
                        "scope_id": scope_id,
                        "cached": cached_for_user,
                    },
                )
                for s in stages:
                    if s["status"] != "pending":
                        ch.publish("stage", s)
                ch.publish(
                    "verdict",
                    {
                        "run_id": run_id,
                        "trial_id": trial.trial_id,
                        "verdict": trial.verdict,
                        "survivor_class": trial.survivor_class,
                        "killed_at": killed_at,
                        "card_id": owned_card_id,
                        "cached": cached_for_user,
                        "verdict_copy": _safe_verdict_copy(
                            trial.verdict,
                            killed_at=killed_at,
                            survivor_class=trial.survivor_class,
                            handle=user_id,
                        ),
                    },
                )
            finally:
                ch.finish(run_id, "completed")
            return (
                {
                    "run_id": run_id,
                    "trial_id": trial.trial_id,
                    "scope_id": scope_id,
                    "cached": cached_for_user,
                    "status": "completed",
                    "stream_url": f"/api/mining/runs/{run_id}/stream",
                    "poll_url": f"/api/mining/runs/{run_id}",
                },
                None,
            )

        # Case 3: Trial is ALREADY executing on another request -> coalesce!
        with self._lock:
            inflight = self._in_flight.get(trial.trial_id)
            if inflight is not None:
                inflight.waiters.append((run_id, user_id))
                coalesced = True
            else:
                coalesced = False

        if coalesced:
            self._fast_call(
                self.runs.insert_run,
                run_id=run_id,
                user_id=user_id,
                trial_id=trial.trial_id,
                candidate_id=candidate_id,
                scope_id=scope_id,
                cached=cached_for_user,
                status="running",
                verdict="pending",
                survivor_class="none",
            )
            ch.publish(
                "registered",
                {
                    "run_id": run_id,
                    "trial_id": trial.trial_id,
                    "scope_id": scope_id,
                    "cached": cached_for_user,
                },
            )
            return (
                {
                    "run_id": run_id,
                    "trial_id": trial.trial_id,
                    "scope_id": scope_id,
                    "cached": cached_for_user,
                    "status": "running",
                    "stream_url": f"/api/mining/runs/{run_id}/stream",
                    "poll_url": f"/api/mining/runs/{run_id}",
                },
                None,
            )

        # Case 4: Orphaned partially-decided pending trial (prior crash mid-run).
        stages_now = self.ledger.stages(trial.trial_id)
        if any(str(s["status"]) != "pending" for s in stages_now):
            self._fast_call(
                self.runs.insert_run,
                run_id=run_id,
                user_id=user_id,
                trial_id=trial.trial_id,
                candidate_id=candidate_id,
                scope_id=scope_id,
                cached=False,
                status="failed",
                verdict="pending",
                survivor_class="none",
                error_message="Prior trial was interrupted before completion.",
            )
            try:
                ch.publish(
                    "error",
                    {
                        "run_id": run_id,
                        "error": "Prior trial was interrupted before completion.",
                    },
                )
            finally:
                ch.finish(run_id, "failed")
            return (
                {
                    "run_id": run_id,
                    "trial_id": trial.trial_id,
                    "scope_id": scope_id,
                    "cached": False,
                    "status": "failed",
                    "stream_url": f"/api/mining/runs/{run_id}/stream",
                    "poll_url": f"/api/mining/runs/{run_id}",
                },
                None,
            )

        # Case 1: Fresh execution -> register in-flight and return pair_in.
        with self._lock:
            self._in_flight[trial.trial_id] = _InFlightTrial(
                trial_id=trial.trial_id, waiters=[(run_id, user_id)]
            )
        self._fast_call(
            self.runs.insert_run,
            run_id=run_id,
            user_id=user_id,
            trial_id=trial.trial_id,
            candidate_id=candidate_id,
            scope_id=scope_id,
            cached=False,
            status="running",
            verdict="pending",
            survivor_class="none",
        )
        ch.publish(
            "registered",
            {
                "run_id": run_id,
                "trial_id": trial.trial_id,
                "scope_id": scope_id,
                "cached": False,
            },
        )
        return (
            {
                "run_id": run_id,
                "trial_id": trial.trial_id,
                "scope_id": scope_id,
                "cached": False,
                "status": "running",
                "stream_url": f"/api/mining/runs/{run_id}/stream",
                "poll_url": f"/api/mining/runs/{run_id}",
            },
            pair_in,
        )

    def execute_run(
        self,
        *,
        run_id: str,
        user_id: str,
        trial_id: str,
        scope_id: str,
        pair_in: _PairInput,
    ) -> None:
        """Execute GauntletRunner (S0-S4 + automatic S5) on the engine worker thread."""
        completed_db = False
        final_status = "failed"
        try:
            runner = GauntletRunner(
                self.ledger,
                seed=7,
                card_minter=self.card_layer.minter,
                admission_epoch=1,
                panel_corr=pair_in.panel_corr,
            )
            self._ensure_admission_epoch(runner, scope_id, pair_in)

            def _on_stage(sr: StageResult) -> None:
                pub = _stage_to_public(sr)
                for wid, _ in self._snapshot_waiters(trial_id, run_id, user_id):
                    wch = self._get_channel_mem(wid)
                    if wch is not None:
                        wch.publish("stage", pub)

            verdict_dict = runner.run_gauntlet(
                trial_id,
                fixture_x=pair_in.fixture_x,
                fixture_y=pair_in.fixture_y,
                beta_true=pair_in.beta_true,
                half_life_true=pair_in.half_life_true,
                fixture_spread=pair_in.fixture_spread,
                excess_returns=pair_in.excess_returns,
                perm_series=pair_in.perm_series,
                perm_statistic_fn=stats.mean_reversion_stat,
                prices=pair_in.prices,
                scope_id=scope_id,
                s2_B=self.s2_B,
                full_prices=pair_in.full_prices,
                full_spread_returns=pair_in.full_spread_returns,
                delist_idx=pair_in.delist_idx,
                holdout_baseline=pair_in.holdout_baseline,
                on_stage=_on_stage,
            )
            verdict = str(verdict_dict["verdict"])
            survivor_class = str(verdict_dict["survivor_class"])
            killed_at = verdict_dict.get("killed_at")
            card_id = verdict_dict.get("card_id")

            last_stage = (
                verdict_dict["stages"][-1].stage
                if verdict_dict.get("stages")
                else killed_at
            )
            self._fast_call(
                self.telemetry.record,
                "trial_verdict",
                {
                    "trial_id": trial_id,
                    "verdict": verdict,
                    "survivor_class": survivor_class,
                    "killed_at": killed_at,
                    "stage": last_stage,
                },
                user_id=user_id,
            )

            vault_admitted = False
            card_owner: str | None = None
            if card_id:
                if pair_in.current_abs_z is not None:
                    self._put_live_z(card_id, pair_in.current_abs_z)
                card = self.card_layer.cards.get_card(card_id)
                card_owner = card.owner_handle
                self._fast_call(
                    self.telemetry.record,
                    "card_minted",
                    {
                        "card_id": card.card_id,
                        "trial_id": trial_id,
                        "kind": card.kind,
                        "tier": card.tier,
                        "survivor_class": card.survivor_class,
                    },
                    user_id=card_owner,
                )
                if flags.cards_v4_1_enabled(card_owner):
                    try:
                        self.card_layer.vault.admit(card.card_id, card_owner)
                        vault_admitted = True
                    except (VaultError, CardError):
                        vault_admitted = False

            # Pop in-flight entry and finalize EVERY coalesced waiter atomically.
            waiters = self._pop_waiters(trial_id, run_id, user_id)
            for wid, wuser in waiters:
                w_card = card_id if (card_id and card_owner == wuser) else None
                self._fast_call(
                    self.runs.complete_run,
                    wid,
                    status="completed",
                    verdict=verdict,
                    survivor_class=survivor_class,
                    killed_at=killed_at,
                    card_id=w_card,
                )
            completed_db = True
            final_status = "completed"

            # Publish verdict + done AFTER DB commit so SSE issues can never
            # overwrite a completed run row as failed.
            for wid, wuser in waiters:
                wch = self._get_channel_mem(wid)
                if wch is None:
                    continue
                w_card = card_id if (card_id and card_owner == wuser) else None
                try:
                    wch.publish(
                        "verdict",
                        {
                            "run_id": wid,
                            "trial_id": trial_id,
                            "verdict": verdict,
                            "survivor_class": survivor_class,
                            "killed_at": killed_at,
                            "card_id": w_card,
                            "vault_admitted": vault_admitted if w_card else False,
                            "cached": False,
                            "verdict_copy": _safe_verdict_copy(
                                verdict,
                                killed_at=killed_at,
                                survivor_class=survivor_class,
                                handle=wuser,
                            ),
                        },
                    )
                finally:
                    wch.finish(wid, "completed")
        except Exception as exc:  # noqa: BLE001 - worker boundary must mark failed & close SSE
            if not completed_db:
                waiters = self._pop_waiters(trial_id, run_id, user_id)
                for wid, _ in waiters:
                    with contextlib.suppress(Exception):
                        self._fast_call(
                            self.runs.complete_run,
                            wid,
                            status="failed",
                            verdict="pending",
                            survivor_class="none",
                            error_message="Mining run failed.",
                        )
                    wch = self._get_channel_mem(wid)
                    if wch is not None:
                        try:
                            wch.publish(
                                "error",
                                {
                                    "run_id": wid,
                                    "error": f"Mining run failed ({type(exc).__name__}).",
                                },
                            )
                        finally:
                            wch.finish(wid, "failed")
        finally:
            # Final safety net: guarantee primary channel is terminated.
            ch = self._get_channel_mem(run_id)
            if ch is not None:
                ch.finish(run_id, final_status)

    def _snapshot_waiters(
        self, trial_id: str, fallback_run_id: str, fallback_user_id: str
    ) -> list[tuple[str, str]]:
        with self._lock:
            inf = self._in_flight.get(trial_id)
            if inf is not None:
                return list(inf.waiters)
        return [(fallback_run_id, fallback_user_id)]

    def _pop_waiters(
        self, trial_id: str, fallback_run_id: str, fallback_user_id: str
    ) -> list[tuple[str, str]]:
        with self._lock:
            inf = self._in_flight.pop(trial_id, None)
            if inf is not None:
                return list(inf.waiters)
        return [(fallback_run_id, fallback_user_id)]

    def build_run_snapshot(
        self,
        row: sqlite3.Row,
        stages: list[dict[str, Any]],
        user_id: str,
        card_face: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "run_id": str(row["run_id"]),
            "trial_id": str(row["trial_id"]),
            "candidate_id": str(row["candidate_id"]),
            "scope_id": str(row["scope_id"]),
            "cached": bool(row["cached"]),
            "status": str(row["status"]),
            "verdict": str(row["verdict"]),
            "survivor_class": str(row["survivor_class"]),
            "killed_at": row["killed_at"],
            "card_id": row["card_id"],
            "card_face": card_face,
            "verdict_copy": _safe_verdict_copy(
                str(row["verdict"]),
                killed_at=row["killed_at"],
                survivor_class=str(row["survivor_class"]),
                handle=user_id,
            ),
            "stages": stages,
            "created_at": float(row["created_at"]),
            "completed_at": None
            if row["completed_at"] is None
            else float(row["completed_at"]),
        }

    def render_owned_face_or_none(
        self, card_id: str | None, user_id: str
    ) -> dict[str, Any] | None:
        """Verify ownership via assert_owned BEFORE rendering any card face."""
        if not card_id or not flags.cards_v4_1_enabled(user_id):
            return None
        try:
            self.card_layer.cards.assert_owned(str(card_id), user_id)
            return self.card_layer.face.render(
                str(card_id),
                viewer_handle=user_id,
                current_abs_z=self.current_abs_z(str(card_id)),
            )
        except (CardError, FaceError):
            return None

    def get_channel(
        self,
        run_id: str,
        user_id: str,
        *,
        row: sqlite3.Row | None = None,
        stages: list[dict[str, Any]] | None = None,
    ) -> _RunChannel:
        if row is None:
            raise MiningError("Mining run not found.", 404)
        ch = self._get_channel_mem(run_id)
        if ch is not None:
            return ch
        # Reconstruct replay channel from persisted DB rows (handles completed,
        # failed, or restart-interrupted runs cleanly so streams NEVER hang).
        replay = _RunChannel(extra_symbols=self.catalog.forbidden_symbols)
        status = str(row["status"])
        try:
            replay.publish(
                "registered",
                {
                    "run_id": str(row["run_id"]),
                    "trial_id": str(row["trial_id"]),
                    "scope_id": str(row["scope_id"]),
                    "cached": bool(row["cached"]),
                },
            )
            for s in stages or []:
                if s["status"] != "pending":
                    replay.publish("stage", s)
            if status == "completed":
                replay.publish(
                    "verdict",
                    {
                        "run_id": str(row["run_id"]),
                        "trial_id": str(row["trial_id"]),
                        "verdict": str(row["verdict"]),
                        "survivor_class": str(row["survivor_class"]),
                        "killed_at": row["killed_at"],
                        "card_id": row["card_id"],
                        "cached": bool(row["cached"]),
                        "verdict_copy": _safe_verdict_copy(
                            str(row["verdict"]),
                            killed_at=row["killed_at"],
                            survivor_class=str(row["survivor_class"]),
                            handle=user_id,
                        ),
                    },
                )
            elif status == "failed":
                replay.publish(
                    "error",
                    {
                        "run_id": str(row["run_id"]),
                        "error": str(row["error_message"] or "Mining run failed."),
                    },
                )
        finally:
            if status in ("completed", "failed"):
                replay.finish(str(row["run_id"]), status)
            else:
                # Running row with no live channel (e.g. post-restart): terminate cleanly.
                replay.publish(
                    "error",
                    {
                        "run_id": str(row["run_id"]),
                        "error": "Run interrupted before completion.",
                    },
                )
                replay.finish(str(row["run_id"]), "failed")
        return replay

    def _find_owned_card_for_trial(self, trial_id: str, user_id: str) -> str | None:
        row = self.card_layer.cards.con.execute(
            "SELECT card_id FROM card WHERE trial_id=? AND owner_handle=? LIMIT 1",
            (trial_id, user_id),
        ).fetchone()
        return None if row is None else str(row["card_id"])


__all__ = [
    "ALLOWED_CLOSE_Z",
    "ALLOWED_FORMATIONS",
    "ALLOWED_OPEN_Z",
    "ALLOWED_SCOPES",
    "CEREMONY_SEGMENT",
    "HOLDOUT_BARS",
    "MAX_CACHED_CHANNELS",
    "MAX_CACHED_LIVE_Z",
    "REGISTRY_DEPTH_BARS",
    "CandidateMeta",
    "MiningError",
    "MiningManager",
    "MiningStore",
    "PairCatalog",
]
