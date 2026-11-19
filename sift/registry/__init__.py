"""Append-only trial registry ledger (scope doc v3 §5).

The implementation lives in this module; `sift.registry.ledger` re-exports
the public names for backwards compatibility. register() mints a trial ID
before any data access, the canonical-JSON params + config hash +
data-vintage hash form the skip-list key, stage rows s0..s5 are
append-only, and stratum stats are scope-stratified and never pooled.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

STAGES = ("s0", "s1", "s2", "s3", "s4", "s5")
VERDICTS = ("pending", "killed", "survivor")
# v4.2 §3: S0-S2 passers that miss the S4/S3 bar are survivors at the bronze
# evidence tier (not killed) — their classes are named after the missed bar.
SURVIVOR_CLASSES = (
    "none",
    "on-watch-amber",
    "validated-green",
    "on-watch-amber-s4-evidence",
    "on-watch-amber-s3-evidence",
)
PROVENANCES = ("operator_sweep", "user_initiated", "peer_retest")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trial (
    trial_id           TEXT PRIMARY KEY,
    family             TEXT NOT NULL,
    scope_id           TEXT NOT NULL,
    params_json        TEXT NOT NULL,
    config_hash        TEXT NOT NULL,
    data_vintage_hash  TEXT NOT NULL,
    provenance         TEXT NOT NULL,
    registered_at      REAL NOT NULL,
    verdict            TEXT NOT NULL DEFAULT 'pending',
    survivor_class     TEXT NOT NULL DEFAULT 'none',
    discoverer_handle  TEXT,
    parent_trial_id    TEXT,
    engine_version     TEXT NOT NULL,
    budget_consumed    INTEGER NOT NULL DEFAULT 1,
    UNIQUE (family, scope_id, config_hash, data_vintage_hash)
);

CREATE TABLE IF NOT EXISTS stage_result (
    trial_id    TEXT NOT NULL REFERENCES trial(trial_id),
    stage       TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',   -- pending|pass|fail
    statistic   REAL,
    threshold   REAL,
    p_value     REAL,
    n_stratum   INTEGER,
    v_stratum   REAL,
    dsr         REAL,
    q_bh        REAL,
    seed        INTEGER,
    computed_at REAL NOT NULL,
    PRIMARY KEY (trial_id, stage)
);

CREATE TABLE IF NOT EXISTS stratum_stats (
    scope_id    TEXT PRIMARY KEY,
    n_trials    INTEGER NOT NULL,
    var_stat    REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_stage_stage ON stage_result(stage);
CREATE INDEX IF NOT EXISTS idx_trial_scope ON trial(scope_id);

CREATE TABLE IF NOT EXISTS ledger_event (
    event_id     INTEGER PRIMARY KEY AUTOINCREMENT,   -- append-only; rows never deleted
    trial_id     TEXT REFERENCES trial(trial_id),
    event_type   TEXT NOT NULL,                        -- e.g. 'card_decay'
    payload_json TEXT NOT NULL,                        -- canonical JSON, immutable
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ledger_event_type ON ledger_event(event_type);
"""


class LedgerError(Exception):
    """Raised on ledger misuse (unknown trial, illegal transition, ...)."""


def canonical_json(params: dict) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)


def config_hash_of(params: dict) -> str:
    return hashlib.sha256(canonical_json(params).encode()).hexdigest()


@dataclass(frozen=True)
class Trial:
    trial_id: str
    family: str
    scope_id: str
    config_hash: str
    data_vintage_hash: str
    provenance: str
    registered_at: float
    verdict: str
    survivor_class: str


class Ledger:
    """Append-only trial registry. The only writer is the server-side engine."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        if self.db_path.parent and str(self.db_path.parent) not in ("", "."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.db_path))
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA foreign_keys = ON")
        self.con.executescript(SCHEMA_SQL)
        self.con.commit()

    def __enter__(self) -> "Ledger":
        return self
    def __exit__(self, *exc) -> None:
        self.close()

    # -- registration ------------------------------------------------------

    def close(self) -> None:
        self.con.close()

    # -- registration -----------------------------------------------------
    def register(
        self,
        family: str,
        scope_id: str,
        params: dict,
        data_vintage_hash: str,
        provenance: str = "user_initiated",
        engine_version: str = "dev",
        discoverer_handle: str | None = None,
        parent_trial_id: str | None = None,
    ) -> Trial:
        """Mint a trial_id before any data access. Re-registering the identical
        (family, scope_id, config_hash, data_vintage_hash) returns the existing
        trial (skip-list semantics; the ledger still counts it exactly once)."""
        if provenance not in PROVENANCES:
            raise LedgerError(f"invalid provenance {provenance!r}")
        cfg_hash = config_hash_of(params)
        row = self.con.execute(
            "SELECT * FROM trial WHERE family=? AND scope_id=? AND config_hash=?"
            " AND data_vintage_hash=?",
            (family, scope_id, cfg_hash, data_vintage_hash),
        ).fetchone()
        if row is not None:
            return self._to_trial(row)
        n = self.con.execute("SELECT COUNT(*) FROM trial").fetchone()[0]
        trial_id = f"trial-{n + 1:06d}"
        now = time.time()
        self.con.execute(
            "INSERT INTO trial (trial_id, family, scope_id, params_json, config_hash,"
            " data_vintage_hash, provenance, registered_at, discoverer_handle,"
            " parent_trial_id, engine_version) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (trial_id, family, scope_id, canonical_json(params), cfg_hash,
             data_vintage_hash, provenance, now, discoverer_handle,
             parent_trial_id, engine_version),
        )
        for s in STAGES:
            self.con.execute(
                "INSERT INTO stage_result (trial_id, stage, status, computed_at)"
                " VALUES (?,?, 'pending', ?)",
                (trial_id, s, now),
            )
        self._bump_stratum(scope_id, now)
        self.con.commit()
        return self.get(trial_id)

    def get(self, trial_id: str) -> Trial:
        row = self.con.execute(
            "SELECT * FROM trial WHERE trial_id=?", (trial_id,)
        ).fetchone()
        if row is None:
            raise LedgerError(f"unknown trial_id {trial_id!r} — register it first")
        return self._to_trial(row)

    def _to_trial(self, row: sqlite3.Row) -> Trial:
        return Trial(
            trial_id=row["trial_id"], family=row["family"], scope_id=row["scope_id"],
            config_hash=row["config_hash"], data_vintage_hash=row["data_vintage_hash"],
            provenance=row["provenance"], registered_at=row["registered_at"],
            verdict=row["verdict"], survivor_class=row["survivor_class"],
        )

    # -- stage rows --------------------------------------------------------
    def record_stage(
        self,
        trial_id: str,
        stage: str,
        status: str,
        *,
        statistic: float | None = None,
        threshold: float | None = None,
        p_value: float | None = None,
        n_stratum: int | None = None,
        v_stratum: float | None = None,
        dsr: float | None = None,
        q_bh: float | None = None,
        seed: int | None = None,
    ) -> None:
        """Append the result of one gauntlet stage. A stage row may transition
        pending -> pass|fail exactly once; a stage that is not run cannot pass."""
        if stage not in STAGES:
            raise LedgerError(f"invalid stage {stage!r}")
        if status not in ("pass", "fail"):
            raise LedgerError(f"status must be pass|fail, got {status!r}")
        row = self.con.execute(
            "SELECT status FROM stage_result WHERE trial_id=? AND stage=?",
            (trial_id, stage),
        ).fetchone()
        if row is None:
            raise LedgerError(f"unknown trial_id {trial_id!r}")
        if row["status"] != "pending":
            raise LedgerError(
                f"stage {stage} of {trial_id} already decided ({row['status']}) —"
                " append-only"
            )
        self.con.execute(
            "UPDATE stage_result SET status=?, statistic=?, threshold=?, p_value=?,"
            " n_stratum=?, v_stratum=?, dsr=?, q_bh=?, seed=?, computed_at=?"
            " WHERE trial_id=? AND stage=?",
            (status, statistic, threshold, p_value, n_stratum, v_stratum, dsr,
             q_bh, seed, time.time(), trial_id, stage),
        )
        self.con.commit()

    def stages(self, trial_id: str) -> list[sqlite3.Row]:
        return self.con.execute(
            "SELECT * FROM stage_result WHERE trial_id=? ORDER BY stage",
            (trial_id,),
        ).fetchall()

    # -- verdict --------------------------------------------------------------

    def stage(self, trial_id: str, stage: str) -> sqlite3.Row:
        row = self.con.execute(
            "SELECT * FROM stage_result WHERE trial_id=? AND stage=?",
            (trial_id, stage),
        ).fetchone()
        if row is None:
            raise LedgerError(f"unknown trial/stage {trial_id}/{stage}")
        return row

    # -- verdict -----------------------------------------------------------
    def decide(self, trial_id: str, verdict: str, survivor_class: str = "none") -> None:
        """Set the verdict once, append-only."""
        if verdict not in ("killed", "survivor"):
            raise LedgerError(f"verdict must be killed|survivor, got {verdict!r}")
        if survivor_class not in SURVIVOR_CLASSES:
            raise LedgerError(f"invalid survivor_class {survivor_class!r}")
        if verdict == "killed" and survivor_class != "none":
            raise LedgerError("a killed trial cannot carry a survivor class")
        if verdict == "survivor" and survivor_class == "none":
            raise LedgerError("a survivor verdict must carry a survivor class")
        row = self.con.execute(
            "SELECT verdict FROM trial WHERE trial_id=?", (trial_id,)
        ).fetchone()
        if row is None:
            raise LedgerError(f"unknown trial_id {trial_id!r}")
        if row["verdict"] != "pending":
            raise LedgerError(f"verdict of {trial_id} already set ({row['verdict']})")
        self.con.execute(
            "UPDATE trial SET verdict=?, survivor_class=? WHERE trial_id=?",
            (verdict, survivor_class, trial_id),
        )
        self.con.commit()

    # -- scope-stratified denominators -------------------------------------
    def _bump_stratum(self, scope_id: str, now: float) -> None:
        """N counts every registered trial in the stratum, including kills.
        var_stat is the sample variance of the stratum's recorded S4 statistics
        (floored at the theoretical 1/T by the caller via update_var_stat)."""
        n = self.con.execute(
            "SELECT COUNT(*) FROM trial WHERE scope_id=?", (scope_id,)
        ).fetchone()[0]
        row = self.con.execute(
            "SELECT var_stat FROM stratum_stats WHERE scope_id=?", (scope_id,)
        ).fetchone()
        v = row["var_stat"] if row else 0.0
        self.con.execute(
            "INSERT INTO stratum_stats (scope_id, n_trials, var_stat, updated_at)"
            " VALUES (?,?,?,?)"
            " ON CONFLICT(scope_id) DO UPDATE SET n_trials=excluded.n_trials,"
            " var_stat=excluded.var_stat, updated_at=excluded.updated_at",
            (scope_id, n, v, now),
        )

    def update_var_stat(self, scope_id: str) -> None:
        """Recompute the stratum's variance of S4 statistics from the ledger.

        V is the sample variance of the stratum's recorded S4 `statistic`
        values (per-period units matching the candidate's). Computed strictly
        within the scope — never pooled across scopes.
        """
        vals = [
            r[0]
            for r in self.con.execute(
                "SELECT sr.statistic FROM stage_result sr JOIN trial t"
                " ON sr.trial_id = t.trial_id"
                " WHERE t.scope_id=? AND sr.stage='s4' AND sr.statistic IS NOT NULL",
                (scope_id,),
            ).fetchall()
        ]
        v = sample_variance(vals)
        n = self.con.execute(
            "SELECT COUNT(*) FROM trial WHERE scope_id=?", (scope_id,)
        ).fetchone()[0]
        self._write_stratum(scope_id, n, v, time.time())
        self.con.commit()

    def _write_stratum(self, scope_id: str, n: int, v: float, now: float) -> None:
        self.con.execute(
            "INSERT INTO stratum_stats (scope_id, n_trials, var_stat, updated_at)"
            " VALUES (?,?,?,?)"
            " ON CONFLICT(scope_id) DO UPDATE SET n_trials=excluded.n_trials,"
            " var_stat=excluded.var_stat, updated_at=excluded.updated_at",
            (scope_id, n, v, now),
        )

    def stratum_stats(self, scope_id: str) -> tuple[int, float]:
        """(N, V̄) for the scope stratum — the DSR denominator. N includes kills."""
        row = self.con.execute(
            "SELECT n_trials, var_stat FROM stratum_stats WHERE scope_id=?",
            (scope_id,),
        ).fetchone()
        if row is None:
            return (0, 0.0)
        return (row["n_trials"], row["var_stat"])

    def trial_count(self) -> int:
        """Global registered-trial counter (read by multiplicity corrections)."""
        return self.con.execute("SELECT COUNT(*) FROM trial").fetchone()[0]

    # -- append-only registry events -------------------------------------------

    def record_event(self, event_type: str, payload: dict,
                     trial_id: str | None = None, now: float | None = None) -> int:
        """Append a registry event (card decay, audit, ...) to the ledger.
        Append-only: the row is never updated or deleted. Returns event_id."""
        now = time.time() if now is None else now
        cur = self.con.execute(
            "INSERT INTO ledger_event (trial_id, event_type, payload_json, created_at)"
            " VALUES (?,?,?,?)",
            (trial_id, event_type, canonical_json(payload), now),
        )
        self.con.commit()
        assert cur.lastrowid is not None
        return int(cur.lastrowid)

    def events(self, event_type: str | None = None) -> list[sqlite3.Row]:
        """Read append-only registry events, oldest first."""
        if event_type is None:
            return self.con.execute(
                "SELECT * FROM ledger_event ORDER BY event_id"
            ).fetchall()
        return self.con.execute(
            "SELECT * FROM ledger_event WHERE event_type=? ORDER BY event_id",
            (event_type,),
        ).fetchall()

    def trials_in_scope(self, scope_id: str) -> list[Trial]:
        rows = self.con.execute(
            "SELECT * FROM trial WHERE scope_id=? ORDER BY registered_at", (scope_id,)
        ).fetchall()
        return [self._to_trial(r) for r in rows]


def skip_list_key(family: str, scope_id: str, params: dict, data_vintage_hash: str) -> str:
    """Full skip-list key: canonical(family, scope, params, vintage).

    `config_hash` (params only) is stored on the row for inspection; the
    uniqueness constraint uses (family, scope_id, config_hash,
    data_vintage_hash) which is exactly this key.
    """
    blob = canonical_json(
        {
            "family": family,
            "scope_id": scope_id,
            "config_hash": config_hash_of(params),
            "data_vintage_hash": data_vintage_hash,
        }
    )
    return hashlib.sha256(blob.encode()).hexdigest()

def sample_variance(vals: list[float]) -> float:
    """Sample (unbiased) variance; 0.0 for fewer than 2 values.

    Pure-Python so the ledger layer has no numpy dependency at import time.
    """
    n = len(vals)
    if n < 2:
        return 0.0
    m = sum(vals) / n
    return sum((v - m) ** 2 for v in vals) / (n - 1)


__all__ = [
    "PROVENANCES",
    "STAGES",
    "SURVIVOR_CLASSES",
    "VERDICTS",
    "Ledger",
    "LedgerError",
    "Trial",
    "canonical_json",
    "config_hash_of",
    "sample_variance",
    "skip_list_key",
]
