"""Server-side pairs mining catalog, execution coordinator, and SSE event hub.

Hard rules enforced here:
- EODHD warehouse stays strictly server-side (`data/warehouse.py`). Clients
  see ONLY opaque `candidate_id`s and coarse labels — NEVER tickers, symbol
  lists, or raw price arrays.
- In `SIFT_ENV=production`, serving EODHD-derived artifacts requires both a
  mounted warehouse AND `SIFT_EODHD_BETA_LICENSE_CONFIRMED=1`
  (docs/vendor-licensing.md §4.4 beta gate). Synthetic example presets are
  allowed only in `dev`/`test` (or explicit `allow_synthetic_examples=True`)
  and are labeled `synthetic_example: true`.
- S5 registry-depth validation runs AUTOMATICALLY whenever full-window data
  is present — never opt-in, never skippable.
- Skip-list semantics are preserved via `Ledger.register(...)`: identical
  `(family, scope_id, config_hash, data_vintage_hash)` reuses the `trial_id`,
  sets `cached=True`, and does NOT emit a duplicate `trial_registered` event.
- Every SSE frame and run snapshot passes `guard_api_payload` fail-closed.
"""

from __future__ import annotations

import math
import os
import queue
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data import warehouse as wh
from sift import flags
from sift.cards import copy as card_copy
from sift.cards.service import CardLayer
from sift.cards.vault import VaultError
from sift.engine import first_run
from sift.gauntlet import config as cfg
from sift.gauntlet import stats
from sift.gauntlet.runner import GauntletRunner, StageResult
from sift.registry import Ledger, config_hash_of
from sift.web.compliance import guard_api_payload
from sift.web.telemetry import TelemetryStore

ENGINE_VERSION = "gate1-beta-api-v1"
ALLOWED_SCOPES = ("free_5y_top1k", "free_1y_top1k")
ALLOWED_FORMATIONS = (40, 60, 80)
ALLOWED_OPEN_Z = (1.5, 2.0, 2.5)
ALLOWED_CLOSE_Z = (0.3, 0.5, 0.8)

# Three-segment ceremony mapping (v4 §3.1, beta-plan §3 step 2):
# S0-S2 = discovery/null flicker; S4/S3 = amber control segment; S5 = full-depth validation.
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


def _synth_cointegrated(scope_id: str, formation: int, open_z: float, close_z: float) -> _PairInput:
    """Planted persistent AR(1) pair (synthetic example preset — survives S0-S5)."""
    t_full = 4032
    t_scope = min(cfg.scope_T(scope_id), t_full)
    rng = np.random.Generator(np.random.PCG64(7))
    lx = np.cumsum(rng.normal(0.0, 0.01, t_full))
    e = rng.normal(0.0, 1.0, t_full)
    phi, sig_in = 0.7, 10.0
    s = np.zeros(t_full)
    for t in range(1, t_full):
        s[t] = phi * s[t - 1] + e[t] * sig_in
    px = 100.0 * np.exp(lx)
    py = px * np.exp(s / 100.0)
    prices_full = np.column_stack([px, py])
    prices_scope = prices_full[-t_scope:]
    x, y = prices_scope[:, 0], prices_scope[:, 1]
    xc, yc = x - x.mean(), y - y.mean()
    beta_true = float(np.dot(xc, yc) / np.dot(xc, xc))
    excess = stats.pairs_strategy_returns(
        prices_scope, open_z=open_z, close_z=close_z
    )["gross"][formation:]
    return _PairInput(
        params={
            "preset": "example-pair-cointegrated",
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
        fixture_spread=s[-t_scope:],
        excess_returns=excess,
        perm_series=_spread_returns(prices_scope),
        prices=prices_scope,
        full_prices=prices_full,
        full_spread_returns=_spread_returns(prices_full),
        current_abs_z=_latest_abs_z(s[-t_scope:], window=formation),
    )


def _synth_noise(scope_id: str, formation: int, open_z: float, close_z: float) -> _PairInput:
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
    # Give S0 a finite target so the pair reaches S1/S2 and dies honestly on the screen.
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
        current_abs_z=_latest_abs_z(spread, window=formation),
    )


def _synth_regime(scope_id: str, formation: int, open_z: float, close_z: float) -> _PairInput:
    """Planted regime-break pair (passes 5y window, automatic S5 catches decay -> amber)."""
    t_full = 4032
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
        self.max_warehouse_pairs = int(max_warehouse_pairs)

        self._meta: dict[str, CandidateMeta] = {}
        self._wh_prices: np.ndarray | None = None
        self._wh_symbols: list[str] = []
        self._wh_pairs: dict[str, tuple[int, int]] = {}
        self._wh_vintage: str = "no-vintage"
        self._load()

    def _load(self) -> None:
        env = os.environ.get("SIFT_ENV", "dev").strip().lower()
        wh_loaded = False
        if self.warehouse_dir and self.warehouse_dir.is_dir():
            if env == "production" and os.environ.get(
                "SIFT_EODHD_BETA_LICENSE_CONFIRMED", "0"
            ) not in ("1", "true", "True", "yes"):
                # docs/vendor-licensing.md §4.4 beta gate: fail closed in
                # production until written EODHD confirmation is recorded.
                wh_loaded = False
            else:
                wh_loaded = self._load_warehouse(self.warehouse_dir)

        if not wh_loaded and self.allow_synthetic_examples:
            self._register_synthetic_presets()

    def _load_warehouse(self, root: Path) -> bool:
        eod_dir = root / "eod" / "US"
        if not eod_dir.is_dir():
            return False
        symbols = sorted(p.stem for p in eod_dir.glob("*.parquet"))
        if len(symbols) < 2:
            return False
        prices, kept, _ = first_run.load_price_matrix(str(root), symbols, exchange="US")
        if prices.shape[1] < 2 or prices.shape[0] < 80:
            return False
        self._wh_prices = prices
        self._wh_symbols = kept
        self._wh_vintage = wh.read_vintage(str(root)) or wh.warehouse_vintage_hash(str(root))

        rets = np.zeros_like(prices)
        rets[1:] = np.diff(np.log(prices), axis=0)
        corr = first_run.corr_matrix(rets[1:])
        iu = np.triu_indices(corr.shape[0], k=1)
        order = np.argsort(-np.abs(np.nan_to_num(corr[iu])))
        count = min(len(order), self.max_warehouse_pairs)
        for rank_idx in range(count):
            k = int(order[rank_idx])
            i, j = int(iu[0][k]), int(iu[1][k])
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
        out = []
        for m in self._meta.values():
            out.append({
                "candidate_id": m.candidate_id,
                "label": m.label,
                "description": m.description,
                "family": m.family,
                "default_scope": m.default_scope,
                "sector_bucket": m.sector_bucket,
                "is_example": m.is_example,
                "synthetic_example": m.synthetic_example,
                "data_source": m.data_source,
            })
        return out

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
            raise MiningError(f"Unknown candidate_id {candidate_id!r}.", 404)

        if candidate_id == "example-pair-cointegrated":
            return meta, _synth_cointegrated(scope_id, formation, open_z, close_z)
        if candidate_id == "example-pair-noise":
            return meta, _synth_noise(scope_id, formation, open_z, close_z)
        if candidate_id == "example-pair-regime":
            return meta, _synth_regime(scope_id, formation, open_z, close_z)

        assert self._wh_prices is not None
        i, j = self._wh_pairs[candidate_id]
        si, sj = self._wh_symbols[i], self._wh_symbols[j]
        full_pair = self._wh_prices[:, [i, j]]
        t_scope = min(cfg.scope_T(scope_id), full_pair.shape[0])
        scope_pair = full_pair[-t_scope:]
        ps = first_run.pair_series(scope_pair, 0, 1, formation)
        x, y = scope_pair[:, 0], scope_pair[:, 1]
        _, hl = stats.estimate_ar1(ps["spread"])
        excess = stats.pairs_strategy_returns(
            scope_pair, open_z=open_z, close_z=close_z
        )["gross"][formation:]
        # S5 automatic whenever full-window depth exceeds the scope window
        has_full = full_pair.shape[0] > t_scope
        return meta, _PairInput(
            params={
                "pair": [si, sj],
                "formation": formation,
                "open_z": open_z,
                "close_z": close_z,
                "family": "pairs",
                "scope": scope_id,
            },
            vintage=self._wh_vintage,
            fixture_x=x,
            fixture_y=y,
            beta_true=ps["beta_levels"],
            half_life_true=hl,
            fixture_spread=ps["spread"],
            excess_returns=excess,
            perm_series=ps["spread_returns"],
            prices=scope_pair,
            full_prices=full_pair if has_full else None,
            full_spread_returns=_spread_returns(full_pair) if has_full else None,
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
    return {
        "stage": stage,
        "ceremony_segment": CEREMONY_SEGMENT.get(stage, "flicker"),
        "status": status,
        "statistic": None if statistic is None or not math.isfinite(float(statistic)) else round(float(statistic), 4),
        "threshold": None if threshold is None or not math.isfinite(float(threshold)) else round(float(threshold), 4),
        "p_value": None if p_value is None or not math.isfinite(float(p_value)) else round(float(p_value), 6),
        "dsr": None if dsr is None or not math.isfinite(float(dsr)) else round(float(dsr), 4),
        "q_bh": None if q_bh is None or not math.isfinite(float(q_bh)) else round(float(q_bh), 4),
        "n_stratum": n_stratum,
        "n_eff": None if n_eff is None or not math.isfinite(float(n_eff)) else round(float(n_eff), 2),
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

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[tuple[str, dict[str, Any]]] = []
        self._done = False
        self._subscribers: list[queue.Queue[tuple[str, dict[str, Any]] | None]] = []

    def publish(self, event_name: str, data: dict[str, Any]) -> None:
        safe_data = guard_api_payload(data)
        with self._lock:
            self._events.append((event_name, safe_data))
            for q in list(self._subscribers):
                q.put((event_name, safe_data))
            if event_name == "done":
                self._done = True
                for q in list(self._subscribers):
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


class MiningStore:
    """SQLite persistence for beta mining runs (`beta_run` table in beta.db)."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        if str(self.db_path.parent) not in ("", "."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.db_path))
        self.con.row_factory = sqlite3.Row
        self.con.executescript(RUNS_SCHEMA_SQL)
        self.con.commit()

    def close(self) -> None:
        self.con.close()

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
        now: float | None = None,
    ) -> None:
        ts = time.time() if now is None else float(now)
        done_ts = ts if status == "completed" else None
        self.con.execute(
            "INSERT INTO beta_run"
            " (run_id, user_id, trial_id, candidate_id, scope_id, cached,"
            "  status, verdict, survivor_class, killed_at, card_id, created_at, completed_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
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
                ts,
                done_ts,
            ),
        )
        self.con.commit()

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


class MiningManager:
    """Orchestrates pre-registered pairs gauntlet runs + SSE channels."""

    def __init__(
        self,
        *,
        ledger: Ledger,
        card_layer: CardLayer,
        telemetry: TelemetryStore,
        runs: MiningStore,
        catalog: PairCatalog,
        s2_B: int = cfg.S2_B,
    ):
        self.ledger = ledger
        self.card_layer = card_layer
        self.telemetry = telemetry
        self.runs = runs
        self.catalog = catalog
        self.s2_B = int(s2_B)
        self._channels_lock = threading.Lock()
        self._channels: dict[str, _RunChannel] = {}
        self._live_z: dict[str, float] = {}

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
        cid = self.catalog.default_example_id() if (example or not candidate_id) else str(candidate_id)
        if scope_id not in ALLOWED_SCOPES:
            raise MiningError(
                f"Unsupported scope_id {scope_id!r}; allowed: {list(ALLOWED_SCOPES)}",
                400,
            )
        if formation not in ALLOWED_FORMATIONS:
            raise MiningError(
                f"formation must be one of {list(ALLOWED_FORMATIONS)}", 400
            )
        if open_z not in ALLOWED_OPEN_Z:
            raise MiningError(
                f"open_z must be one of {list(ALLOWED_OPEN_Z)}", 400
            )
        if close_z not in ALLOWED_CLOSE_Z:
            raise MiningError(
                f"close_z must be one of {list(ALLOWED_CLOSE_Z)}", 400
            )
        return cid, scope_id, int(formation), float(open_z), float(close_z)

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

        Returns `(run_meta, pair_input_or_None_if_cached)`.
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
        cached = existing is not None

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
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        ch = _RunChannel()
        with self._channels_lock:
            self._channels[run_id] = ch

        if not cached:
            # Emit trial_registered ONLY for genuinely new unique trial_ids
            # (beta-plan.md §4 anti-gaming: skip-list hits never inflate activity).
            self.telemetry.record(
                "trial_registered",
                {
                    "trial_id": trial.trial_id,
                    "scope_id": scope_id,
                    "provenance": "user_initiated",
                    "config_hash": trial.config_hash,
                    "beta": True,
                },
                user_id=user_id,
            )

        if cached and trial.verdict != "pending":
            stages = [_stage_to_public(r) for r in self.ledger.stages(trial.trial_id)]
            killed_at = next(
                (s["stage"] for s in stages if s["status"] == "fail" and s["stage"] in ("s0", "s1", "s2")),
                None,
            )
            owned_card_id = self._find_owned_card_for_trial(trial.trial_id, user_id)
            self.runs.insert_run(
                run_id=run_id,
                user_id=user_id,
                trial_id=trial.trial_id,
                candidate_id=candidate_id,
                scope_id=scope_id,
                cached=True,
                status="completed",
                verdict=trial.verdict,
                survivor_class=trial.survivor_class,
                killed_at=killed_at,
                card_id=owned_card_id,
            )
            ch.publish(
                "registered",
                {
                    "run_id": run_id,
                    "trial_id": trial.trial_id,
                    "scope_id": scope_id,
                    "cached": True,
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
                    "cached": True,
                    "verdict_copy": _safe_verdict_copy(
                        trial.verdict,
                        killed_at=killed_at,
                        survivor_class=trial.survivor_class,
                        handle=user_id,
                    ),
                },
            )
            ch.publish("done", {"run_id": run_id, "status": "completed"})
            return (
                {
                    "run_id": run_id,
                    "trial_id": trial.trial_id,
                    "scope_id": scope_id,
                    "cached": True,
                    "status": "completed",
                    "stream_url": f"/api/mining/runs/{run_id}/stream",
                    "poll_url": f"/api/mining/runs/{run_id}",
                },
                None,
            )

        self.runs.insert_run(
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
        """Execute GauntletRunner (S0-S4 + automatic S5) on the DB worker thread."""
        ch = self._get_or_create_channel(run_id)
        try:
            runner = GauntletRunner(
                self.ledger,
                seed=7,
                card_minter=self.card_layer.minter,
            )

            def _on_stage(sr: StageResult) -> None:
                ch.publish("stage", _stage_to_public(sr))

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
                on_stage=_on_stage,
            )
            verdict = str(verdict_dict["verdict"])
            survivor_class = str(verdict_dict["survivor_class"])
            killed_at = verdict_dict.get("killed_at")
            card_id = verdict_dict.get("card_id")

            # Record trial_verdict telemetry
            last_stage = (
                verdict_dict["stages"][-1].stage
                if verdict_dict.get("stages")
                else killed_at
            )
            self.telemetry.record(
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

            # If a card was minted, record card_minted telemetry and admit to vault
            # if capacity allows (vault-full leaves it ready for keep/discard).
            vault_admitted = False
            if card_id:
                if pair_in.current_abs_z is not None:
                    self._live_z[card_id] = pair_in.current_abs_z
                card = self.card_layer.cards.get_card(card_id)
                self.telemetry.record(
                    "card_minted",
                    {
                        "card_id": card.card_id,
                        "trial_id": trial_id,
                        "kind": card.kind,
                        "tier": card.tier,
                        "survivor_class": card.survivor_class,
                    },
                    user_id=user_id,
                )
                if flags.cards_v4_1_enabled(user_id):
                    try:
                        self.card_layer.vault.admit(card.card_id, user_id)
                        vault_admitted = True
                    except VaultError:
                        vault_admitted = False

            self.runs.complete_run(
                run_id,
                status="completed",
                verdict=verdict,
                survivor_class=survivor_class,
                killed_at=killed_at,
                card_id=card_id,
            )
            ch.publish(
                "verdict",
                {
                    "run_id": run_id,
                    "trial_id": trial_id,
                    "verdict": verdict,
                    "survivor_class": survivor_class,
                    "killed_at": killed_at,
                    "card_id": card_id,
                    "vault_admitted": vault_admitted,
                    "cached": False,
                    "verdict_copy": _safe_verdict_copy(
                        verdict,
                        killed_at=killed_at,
                        survivor_class=survivor_class,
                        handle=user_id,
                    ),
                },
            )
            ch.publish("done", {"run_id": run_id, "status": "completed"})
        except Exception as exc:  # noqa: BLE001 - worker boundary must mark run failed & close SSE
            self.runs.complete_run(
                run_id,
                status="failed",
                verdict="pending",
                survivor_class="none",
                error_message="gauntlet execution failed",
            )
            ch.publish(
                "error",
                {"run_id": run_id, "error": f"Mining run failed: {type(exc).__name__}"},
            )
            ch.publish("done", {"run_id": run_id, "status": "failed"})

    def get_run_snapshot(self, run_id: str, user_id: str) -> dict[str, Any]:
        row = self.runs.get_run(run_id, user_id)
        if row is None:
            raise MiningError("Mining run not found.", 404)
        trial_id = str(row["trial_id"])
        stages = [_stage_to_public(r) for r in self.ledger.stages(trial_id)]
        card_id = row["card_id"]
        card_face = None
        if card_id and flags.cards_v4_1_enabled(user_id):
            try:
                card_face = self.card_layer.face.render(
                    str(card_id),
                    viewer_handle=user_id,
                    current_abs_z=self._live_z.get(str(card_id)),
                )
            except (ValueError, RuntimeError, Exception):  # noqa: BLE001 - optional preview face
                card_face = None
        return {
            "run_id": str(row["run_id"]),
            "trial_id": trial_id,
            "candidate_id": str(row["candidate_id"]),
            "scope_id": str(row["scope_id"]),
            "cached": bool(row["cached"]),
            "status": str(row["status"]),
            "verdict": str(row["verdict"]),
            "survivor_class": str(row["survivor_class"]),
            "killed_at": row["killed_at"],
            "card_id": card_id,
            "card_face": card_face,
            "verdict_copy": _safe_verdict_copy(
                str(row["verdict"]),
                killed_at=row["killed_at"],
                survivor_class=str(row["survivor_class"]),
                handle=user_id,
            ),
            "stages": stages,
            "created_at": float(row["created_at"]),
            "completed_at": None if row["completed_at"] is None else float(row["completed_at"]),
        }

    def get_channel(self, run_id: str, user_id: str) -> _RunChannel:
        row = self.runs.get_run(run_id, user_id)
        if row is None:
            raise MiningError("Mining run not found.", 404)
        with self._channels_lock:
            ch = self._channels.get(run_id)
        if ch is not None:
            return ch
        # Reconstruct replay channel from persisted DB rows (e.g., after restart).
        snap = self.get_run_snapshot(run_id, user_id)
        replay = _RunChannel()
        replay.publish(
            "registered",
            {
                "run_id": snap["run_id"],
                "trial_id": snap["trial_id"],
                "scope_id": snap["scope_id"],
                "cached": snap["cached"],
            },
        )
        for s in snap["stages"]:
            if s["status"] != "pending":
                replay.publish("stage", s)
        if snap["status"] == "completed":
            replay.publish(
                "verdict",
                {
                    "run_id": snap["run_id"],
                    "trial_id": snap["trial_id"],
                    "verdict": snap["verdict"],
                    "survivor_class": snap["survivor_class"],
                    "killed_at": snap["killed_at"],
                    "card_id": snap["card_id"],
                    "cached": snap["cached"],
                    "verdict_copy": snap["verdict_copy"],
                },
            )
            replay.publish("done", {"run_id": snap["run_id"], "status": "completed"})
        return replay

    def current_abs_z(self, card_id: str) -> float | None:
        return self._live_z.get(card_id)

    def _get_or_create_channel(self, run_id: str) -> _RunChannel:
        with self._channels_lock:
            ch = self._channels.get(run_id)
            if ch is None:
                ch = _RunChannel()
                self._channels[run_id] = ch
            return ch

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
    "CandidateMeta",
    "MiningError",
    "MiningManager",
    "MiningStore",
    "PairCatalog",
]
