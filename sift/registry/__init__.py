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

# Soundness #1: the s4 stage statistic is stored ANNUALIZED (sr_ann, see
# gauntlet/runner.py::run_s4), while stats.dsr consumes a PER-BAR sr_hat.
# update_var_stat therefore divides each stored s4 statistic by
# sqrt(BARS_PER_YEAR) before the sample variance, so stratum var_stat is in
# per-bar SR^2 units. BARS_PER_YEAR mirrors sift.gauntlet.config (252) —
# duplicated here as a constant so the ledger layer keeps its no-import-of-
# the-gauntlet layering (pure stdlib at import time).
BARS_PER_YEAR = 252
_BARS_PER_YEAR_SQRT = BARS_PER_YEAR ** 0.5

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
    beta               INTEGER NOT NULL DEFAULT 0,
    UNIQUE (family, scope_id, config_hash, data_vintage_hash)
);

CREATE TABLE IF NOT EXISTS stage_result (
    trial_id    TEXT NOT NULL REFERENCES trial(trial_id),
    stage       TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',   -- pending|pass|fail
    statistic   REAL,   -- UNITS PER STAGE (#1): s0=hedge ratio beta_hat (unitless),
                        -- s1=t-stat (unitless), s2=observed mean-reversion stat (unitless),
                        -- s3=DSR in [0,1] (probability), s4=ANNUALIZED Sharpe sr_ann
                        -- (per-bar would be sr_ann/sqrt(252); see update_var_stat),
                        -- s5=full-depth t_net (unitless)
    threshold   REAL,   -- units match the stage's statistic column
    p_value     REAL,
    n_stratum   INTEGER,
    v_stratum   REAL,   -- PER-BAR SR^2 units (#1): update_var_stat stores the sample
                        -- variance of the s4 statistics AFTER dividing by
                        -- sqrt(BARS_PER_YEAR), matching the per-bar sr_hat fed to dsr
    dsr         REAL,
    q_bh        REAL,
    seed        INTEGER,
    n_eff       REAL,     -- sift #59: A.3 implied INDEPENDENT trials used for the
                          -- s3/s5 DSR benchmark (raw n_stratum stays in n_stratum)
    rho_bar     REAL,     -- average inter-trial correlation behind n_eff
    epoch       INTEGER,  -- frozen evaluation epoch the admission N came from
    computed_at REAL NOT NULL,
    PRIMARY KEY (trial_id, stage)
);

CREATE TABLE IF NOT EXISTS stratum_stats (
    scope_id    TEXT PRIMARY KEY,
    n_trials    INTEGER NOT NULL,
    var_stat    REAL NOT NULL,  -- PER-BAR SR^2 units (#1): sample variance of the
                                -- stratum's s4 statistics converted to per-bar
                                -- (statistic / sqrt(BARS_PER_YEAR)) BEFORE the variance
    updated_at  REAL NOT NULL
);

-- sift #59: epoch-frozen admission denominators. At the pre-registered
-- evaluation epoch boundary the stratum's RAW N and the correlation-
-- adjusted independent N_eff (DSR Appendix A.3) are frozen and cached
-- here; run_s3/run_s5 read the FROZEN rows so identical trials face an
-- identical bar within the epoch regardless of registration order.
-- n_eff_raw counts registered trials; n_eff is the A.3-adjusted count
-- (float); both are recorded for audit. A new epoch = new row (epoch is
-- part of the key); strata never re-freeze retroactively.
CREATE TABLE IF NOT EXISTS stratum_epoch (
    scope_id    TEXT NOT NULL,
    epoch       INTEGER NOT NULL,
    n_eff_raw   INTEGER NOT NULL,   -- raw registered-trial count at freeze
    n_eff       REAL NOT NULL,       -- A.3 implied independent trials (float)
    rho_bar     REAL NOT NULL,       -- average inter-trial correlation used
    method      TEXT NOT NULL,       -- 'a3_avg_corr' | 'asset_cluster_pairs' | 'raw_fallback'
    detail_json TEXT NOT NULL,       -- canonical JSON: eigenspectrum / cluster detail
    frozen_at   REAL NOT NULL,
    PRIMARY KEY (scope_id, epoch)
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
        self._migrate()
        self.con.commit()

    def _migrate(self) -> None:
        """Additive-only migrations for append-only ledgers (sift #59 + beta-plan §5):
        the s3/s5 stage rows gain the admission-N audit columns n_eff /
        rho_bar / epoch, and trial gains the beta flag column. Existing DBs
        get the columns via ALTER TABLE; CREATE TABLE IF NOT EXISTS above
        already has them for new DBs."""
        existing = {r[1] for r in self.con.execute(
            "PRAGMA table_info(stage_result)").fetchall()}
        for col, decl in (("n_eff", "REAL"), ("rho_bar", "REAL"),
                          ("epoch", "INTEGER")):
            if col not in existing:
                self.con.execute(
                    f"ALTER TABLE stage_result ADD COLUMN {col} {decl}")
        trial_cols = {r[1] for r in self.con.execute(
            "PRAGMA table_info(trial)").fetchall()}
        if "beta" not in trial_cols:
            self.con.execute(
                "ALTER TABLE trial ADD COLUMN beta INTEGER NOT NULL DEFAULT 0")

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
        beta: bool = False,
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
            " parent_trial_id, engine_version, beta) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (trial_id, family, scope_id, canonical_json(params), cfg_hash,
             data_vintage_hash, provenance, now, discoverer_handle,
             parent_trial_id, engine_version, 1 if beta else 0),
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

    def trial_owner(self, trial_id: str) -> str | None:
        """Return discoverer_handle for a trial (never params_json)."""
        row = self.con.execute(
            "SELECT discoverer_handle FROM trial WHERE trial_id=?", (trial_id,)
        ).fetchone()
        if row is None:
            raise LedgerError(f"unknown trial_id {trial_id!r} — register it first")
        return row["discoverer_handle"]

    def trials_for_handle(self, handle: str, limit: int = 100) -> list[Trial]:
        """Return trials registered by `handle`, newest first (no params_json)."""
        rows = self.con.execute(
            "SELECT * FROM trial WHERE discoverer_handle=?"
            " ORDER BY registered_at DESC, trial_id DESC LIMIT ?",
            (handle, int(limit)),
        ).fetchall()
        return [self._to_trial(r) for r in rows]

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
        n_eff: float | None = None,
        rho_bar: float | None = None,
        epoch: int | None = None,
    ) -> None:
        """Append the result of one gauntlet stage. A stage row may transition
        pending -> pass|fail exactly once; a stage that is not run cannot pass.

        sift #59: n_eff / rho_bar / epoch record the admission denominator
        actually used (A.3 independent N + the frozen epoch) alongside the
        raw n_stratum for audit."""
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
            " n_stratum=?, v_stratum=?, dsr=?, q_bh=?, seed=?, n_eff=?, rho_bar=?,"
            " epoch=?, computed_at=?"
            " WHERE trial_id=? AND stage=?",
            (status, statistic, threshold, p_value, n_stratum, v_stratum, dsr,
             q_bh, seed, n_eff, rho_bar, epoch, time.time(), trial_id, stage),
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
        var_stat is the sample variance of the stratum's recorded S4
        statistics CONVERTED TO PER-BAR units (divided by sqrt(BARS_PER_YEAR))
        before the variance — #1: the s4 row stores ANNUALIZED sr_ann, and
        dsr() consumes a per-bar sr_hat, so V must be per-bar SR^2 (the deck
        path does the same in decks/service.py::_deck_stratum_variance)."""
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
        values AFTER converting them from annualized to PER-BAR units
        (each statistic is divided by sqrt(BARS_PER_YEAR) first), so V comes
        out in per-bar SR^2 units matching the per-bar sr_hat that
        runner.run_s3/run_s5 feed into stats.dsr (#1). The s4 statistic
        column itself stays ANNUALIZED (sr_ann) — every other reader
        (minting.py tier evidence, decay.py re-eval) consumes status/sr_ann
        semantics, so the conversion happens here, at the single choke
        point, exactly like the deck path (decks/service.py::
        _deck_stratum_variance). Computed strictly within the scope —
        never pooled across scopes.
        """
        vals = [
            r[0] / _BARS_PER_YEAR_SQRT
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

    # -- sift #59: epoch-frozen correlation-adjusted independent N ------------

    def freeze_stratum_epoch(self, scope_id: str, epoch: int, *,
                             n_eff: float, rho_bar: float, method: str,
                             detail: dict | None = None,
                             n_eff_raw: int | None = None,
                             now: float | None = None) -> None:
        """Freeze the stratum's admission denominator at a pre-registered
        evaluation epoch (sift #59): store the raw registration count AND
        the DSR-A.3 implied independent N_eff, cached per (scope, epoch).

        Idempotent per (scope_id, epoch): re-freezing the SAME epoch with
        different values raises (the epoch is pre-registered; silently
        moving the bar mid-epoch is exactly the defect #59 reports —
        identical trials must face identical bars). Freezing a NEW epoch
        appends a row; old epochs stay for audit.
        """
        now = time.time() if now is None else now
        n_raw = (int(n_eff_raw) if n_eff_raw is not None else
                 self.con.execute(
                     "SELECT COUNT(*) FROM trial WHERE scope_id=?",
                     (scope_id,)).fetchone()[0])
        row = self.con.execute(
            "SELECT n_eff_raw, n_eff, rho_bar, method FROM stratum_epoch"
            " WHERE scope_id=? AND epoch=?", (scope_id, epoch),
        ).fetchone()
        if row is not None:
            if (row["n_eff_raw"] != n_raw or
                    abs(row["n_eff"] - float(n_eff)) > 1e-9 or
                    abs(row["rho_bar"] - float(rho_bar)) > 1e-12 or
                    row["method"] != method):
                raise LedgerError(
                    f"stratum epoch {scope_id}/e{epoch} already frozen with "
                    f"different values (n_eff_raw={row['n_eff_raw']}, "
                    f"n_eff={row['n_eff']}, rho_bar={row['rho_bar']}, "
                    f"method={row['method']!r}) — epochs are immutable; "
                    "open a new epoch instead (sift #59)")
            return
        self.con.execute(
            "INSERT INTO stratum_epoch (scope_id, epoch, n_eff_raw, n_eff,"
            " rho_bar, method, detail_json, frozen_at) VALUES (?,?,?,?,?,?,?,?)",
            (scope_id, int(epoch), int(n_raw), float(n_eff), float(rho_bar),
             method, canonical_json(detail or {}), now),
        )
        self.con.commit()

    def stratum_epoch_stats(self, scope_id: str,
                            epoch: int | None = None) -> dict | None:
        """The frozen admission denominator for (scope, epoch). With
        `epoch=None` the LATEST frozen epoch is returned. Returns
        {"epoch", "n_eff_raw", "n_eff", "rho_bar", "method", "detail",
        "frozen_at"} or None when the stratum has no frozen epoch (callers
        then fall back to the live stratum_stats — raw N, rho_bar=0 — and
        record method='raw_fallback')."""
        if epoch is None:
            row = self.con.execute(
                "SELECT * FROM stratum_epoch WHERE scope_id=?"
                " ORDER BY epoch DESC LIMIT 1", (scope_id,),
            ).fetchone()
        else:
            row = self.con.execute(
                "SELECT * FROM stratum_epoch WHERE scope_id=? AND epoch=?",
                (scope_id, epoch),
            ).fetchone()
        if row is None:
            return None
        return {"epoch": int(row["epoch"]), "n_eff_raw": int(row["n_eff_raw"]),
                "n_eff": float(row["n_eff"]), "rho_bar": float(row["rho_bar"]),
                "method": row["method"],
                "detail": json.loads(row["detail_json"]),
                "frozen_at": float(row["frozen_at"])}

    def latest_epoch(self, scope_id: str) -> int | None:
        row = self.con.execute(
            "SELECT MAX(epoch) FROM stratum_epoch WHERE scope_id=?",
            (scope_id,),
        ).fetchone()
        return None if row is None or row[0] is None else int(row[0])

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
