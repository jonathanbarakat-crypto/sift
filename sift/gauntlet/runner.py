"""Gauntlet stage runner S0-S4: wires the pure stats into pre-registered
trials and records every stage row into the trial ledger.

Dependency note (task 2/8): the ledger from task 1/8 (hermes-tasks#81, PR to
jonathanbarakat-crypto/sift) was NOT merged at implementation time — the sift
repo's main is empty. This module therefore codes against the ledger schema
specified in that task and in the FL design docs
(research/design/2026-11-15-sift-gauntlet-s5-design.md SS7,
research/notes/2026-11-15-006-...-design-spec.md SS1.2):

    trial(trial_id PK, family, scope_id, window, params, config_hash,
          data_vintage_hash, provenance, registered_at, verdict,
          survivor_class, ...)
    stage_result(trial_id FK, stage s0..s5, status pending|pass|fail,
                 statistic, threshold, p_value, detail JSON, computed_at)
    stratum_stats(scope_id PK, n_trials, var_sharpe, updated_at)

`Ledger` is a Protocol with exactly the methods the runner needs; when task
1/8's registry lands, its class should satisfy this protocol (or a thin
adapter is added) and `SQLiteLedger` below serves as the reference
implementation of that schema (SQLite now, Postgres-compatible DDL).

Server-side only: nothing here serves raw data to clients — stage rows carry
statistics/thresholds/verdicts, never price series (v3 SS12 rule 2).
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from sift.gauntlet import stats

STAGES = ("s0", "s1", "s2", "s3", "s4")

# Pre-registered cost model (FL configs/backtest.yaml; 100 bps half-spread is
# the flagged placeholder, open item #6 — recorded in the cost params hash).
DEFAULT_COST_PARAMS = {
    "half_spread": 0.01,          # 100 bps placeholder
    "max_participation": 0.10,
    "risk_per_trade": 0.01,
    "max_positions": 20,
    "execution": "next_open_no_same_day_fill",
}


# ---------------------------------------------------------------------------
# Ledger protocol + reference SQLite implementation (task 1/8 schema)
# ---------------------------------------------------------------------------

class Ledger(Protocol):
    """The surface the gauntlet runner needs from the trial registry."""

    def register_trial(
        self,
        family: str,
        scope_id: str,
        window: dict,
        params: dict,
        data_vintage_hash: str,
        provenance: str = "operator_sweep",
    ) -> str: ...

    def record_stage(
        self,
        trial_id: str,
        stage: str,
        status: str,
        statistic: float | None,
        threshold: float | None,
        p_value: float | None,
        detail: dict,
    ) -> None: ...

    def set_verdict(self, trial_id: str, verdict: str, survivor_class: str | None) -> None: ...

    def stratum_stats(self, scope_id: str) -> tuple[int, float]:
        """(N, Vbar) for the scope stratum: N = all registered trials incl.
        kills; Vbar = sample variance of the stratum's S4 net Sharpe stats,
        floored at 1/T. Denominators are never pooled across scopes (v3 SS5.1).
        """
        ...

    def stratum_pvalues(self, family: str, scope_id: str) -> list[float]:
        """S2 p-values of all trials in the (family x scope) stratum, for BH-FDR."""
        ...


@dataclass
class StageRow:
    stage: str
    status: str  # pending | pass | fail
    statistic: float | None
    threshold: float | None
    p_value: float | None
    detail: dict = field(default_factory=dict)


@dataclass
class Trial:
    trial_id: str
    family: str
    scope_id: str
    window: dict
    params: dict
    config_hash: str
    data_vintage_hash: str
    provenance: str
    stages: dict[str, StageRow] = field(default_factory=dict)
    verdict: str = "pending"
    survivor_class: str | None = None


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trial (
    trial_id            TEXT PRIMARY KEY,
    family              TEXT NOT NULL,
    scope_id            TEXT NOT NULL,
    window              TEXT NOT NULL,          -- canonical JSON
    params              TEXT NOT NULL,          -- canonical JSON
    config_hash         TEXT NOT NULL UNIQUE,   -- skip-list key (spec+config)
    data_vintage_hash   TEXT NOT NULL,
    provenance          TEXT NOT NULL,
    registered_at       REAL NOT NULL,
    verdict             TEXT NOT NULL DEFAULT 'pending',
    survivor_class      TEXT
);
CREATE TABLE IF NOT EXISTS stage_result (
    trial_id    TEXT NOT NULL REFERENCES trial(trial_id),
    stage       TEXT NOT NULL,
    status      TEXT NOT NULL,                  -- pending | pass | fail
    statistic   REAL,
    threshold   REAL,
    p_value     REAL,
    detail      TEXT NOT NULL DEFAULT '{}',     -- JSON
    computed_at REAL NOT NULL,
    PRIMARY KEY (trial_id, stage)
);
CREATE TABLE IF NOT EXISTS stratum_stats (
    scope_id    TEXT PRIMARY KEY,
    n_trials    INTEGER NOT NULL,
    var_sharpe  REAL NOT NULL,
    t_bars      INTEGER NOT NULL,
    updated_at  REAL NOT NULL
);
"""


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


class SQLiteLedger:
    """Reference implementation of the task-1/8 ledger schema (SQLite now,
    Postgres-compatible DDL). Append-only for recorded facts: stage_result rows
    and the trial verdict are written exactly once (guarded UPDATEs from
    'pending'; rewrites raise). `stratum_stats` is a derived aggregate counter
    (N and Vbar of the scope stratum), not a recorded fact, and may be
    recomputed — it is excluded from the append-only guarantee. Corrections
    to verdicts are new trials, never edits.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.path))
        self.con.row_factory = sqlite3.Row
        self.con.executescript(SCHEMA_SQL)
        self.con.commit()
        self._counter = self._next_number()

    def _next_number(self) -> int:
        row = self.con.execute("SELECT COUNT(*) AS c FROM trial").fetchone()
        return row["c"] + 1

    def register_trial(
        self,
        family: str,
        scope_id: str,
        window: dict,
        params: dict,
        data_vintage_hash: str,
        provenance: str = "operator_sweep",
    ) -> str:
        config_hash = hashlib.sha256(
            _canonical({"family": family, "scope_id": scope_id, "window": window,
                        "params": params, "data_vintage_hash": data_vintage_hash}).encode()
        ).hexdigest()
        # Skip-list: identical (spec, config, vintage) re-registers to the same id.
        row = self.con.execute(
            "SELECT trial_id FROM trial WHERE config_hash = ?", (config_hash,)
        ).fetchone()
        if row is not None:
            return row["trial_id"]
        trial_id = f"trial-{self._counter:06d}"
        self._counter += 1
        self.con.execute(
            "INSERT INTO trial (trial_id, family, scope_id, window, params, config_hash,"
            " data_vintage_hash, provenance, registered_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (trial_id, family, scope_id, _canonical(window), _canonical(params),
             config_hash, data_vintage_hash, provenance, time.time()),
        )
        for s in ("s0", "s1", "s2", "s3", "s4", "s5"):
            self.con.execute(
                "INSERT INTO stage_result (trial_id, stage, status, detail, computed_at)"
                " VALUES (?,?, 'pending', '{}', ?)",
                (trial_id, s, time.time()),
            )
        self.con.execute(
            "INSERT INTO stratum_stats (scope_id, n_trials, var_sharpe, t_bars, updated_at)"
            " VALUES (?, 1, ?, ?, ?)"
            " ON CONFLICT(scope_id) DO UPDATE SET n_trials = stratum_stats.n_trials + 1,"
            " updated_at = excluded.updated_at",
            (scope_id, 1.0 / max(1, int(window.get("bars", 1260))), int(window.get("bars", 1260)), time.time()),
        )
        self.con.commit()
        return trial_id

    def record_stage(
        self,
        trial_id: str,
        stage: str,
        status: str,
        statistic: float | None,
        threshold: float | None,
        p_value: float | None,
        detail: dict,
    ) -> None:
        # Append-only: a stage row may be written exactly once — from pending.
        # Re-running a deduped trial (same config_hash -> same trial_id) must
        # not rewrite recorded statistics; corrections are new trials.
        cur = self.con.execute(
            "UPDATE stage_result SET status=?, statistic=?, threshold=?, p_value=?, detail=?,"
            " computed_at=? WHERE trial_id=? AND stage=? AND status='pending'",
            (status, statistic, threshold, p_value, _canonical(detail), time.time(), trial_id, stage),
        )
        if cur.rowcount != 1:
            raise RuntimeError(
                f"append-only violation: stage row {trial_id}/{stage} already recorded"
            )
        if stage == "s4" and statistic is not None:
            # Vbar = sample variance of the stratum's S4 net Sharpe estimates.
            scope = self.con.execute(
                "SELECT scope_id FROM trial WHERE trial_id=?", (trial_id,)
            ).fetchone()["scope_id"]
            srs = [r["statistic"] for r in self.con.execute(
                "SELECT sr.statistic FROM stage_result sr JOIN trial t ON t.trial_id = sr.trial_id"
                " WHERE t.scope_id=? AND sr.stage='s4' AND sr.statistic IS NOT NULL", (scope,)
            )]
            n = len(srs)
            if n >= 2:
                m = sum(srs) / n
                v = sum((x - m) ** 2 for x in srs) / (n - 1)
            else:
                v = 1.0 / 1260.0
            t_bars = self.con.execute(
                "SELECT t_bars FROM stratum_stats WHERE scope_id=?", (scope,)
            ).fetchone()["t_bars"]
            v = max(v, 1.0 / t_bars)
            self.con.execute(
                "UPDATE stratum_stats SET var_sharpe=?, updated_at=? WHERE scope_id=?",
                (v, time.time(), scope),
            )
        self.con.commit()

    def set_verdict(self, trial_id: str, verdict: str, survivor_class: str | None) -> None:
        # Append-only: verdict is set exactly once (pending -> decided).
        cur = self.con.execute(
            "UPDATE trial SET verdict=?, survivor_class=? WHERE trial_id=? AND verdict='pending'",
            (verdict, survivor_class, trial_id),
        )
        if cur.rowcount != 1:
            raise RuntimeError(f"append-only violation: verdict for {trial_id} already set")
        self.con.commit()

    def stratum_stats(self, scope_id: str) -> tuple[int, float]:
        row = self.con.execute(
            "SELECT n_trials, var_sharpe FROM stratum_stats WHERE scope_id=?", (scope_id,)
        ).fetchone()
        if row is None:
            return 0, 1.0 / 1260.0
        return row["n_trials"], row["var_sharpe"]

    def stratum_pvalues(self, family: str, scope_id: str) -> list[float]:
        rows = self.con.execute(
            "SELECT sr.p_value FROM stage_result sr JOIN trial t ON t.trial_id = sr.trial_id"
            " WHERE t.family=? AND t.scope_id=? AND sr.stage='s2' AND sr.p_value IS NOT NULL",
            (family, scope_id),
        ).fetchall()
        return [r["p_value"] for r in rows]

    def load_trial(self, trial_id: str) -> Trial:
        row = self.con.execute("SELECT * FROM trial WHERE trial_id=?", (trial_id,)).fetchone()
        if row is None:
            raise KeyError(trial_id)
        stages = {
            r["stage"]: StageRow(r["stage"], r["status"], r["statistic"], r["threshold"],
                                 r["p_value"], json.loads(r["detail"]))
            for r in self.con.execute(
                "SELECT * FROM stage_result WHERE trial_id=?", (trial_id,)
            ).fetchall()
        }
        return Trial(
            trial_id=row["trial_id"], family=row["family"], scope_id=row["scope_id"],
            window=json.loads(row["window"]), params=json.loads(row["params"]),
            config_hash=row["config_hash"], data_vintage_hash=row["data_vintage_hash"],
            provenance=row["provenance"], stages=stages,
            verdict=row["verdict"], survivor_class=row["survivor_class"],
        )


# ---------------------------------------------------------------------------
# Gauntlet config + stages S0-S4
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GauntletConfig:
    """Pre-registered stage thresholds. The config hash covers them."""

    scope_id: str = "free_5y_top1k"
    screen_t: float = stats.DEFAULT_SCREEN_T          # S1
    perm_B: int = stats.DEFAULT_PERM_B                # S2
    perm_p: float = stats.DEFAULT_PERM_P              # S2
    dsr_min: float = stats.DEFAULT_DSR                # S3
    bh_q: float = stats.DEFAULT_BH_Q                  # S3
    tau_cal: float = 3.5                              # S4 scope-matched hurdle (5y free scope)
    cost_params: dict = field(default_factory=lambda: dict(DEFAULT_COST_PARAMS))

    @classmethod
    def for_scope(cls, scope_id: str, **overrides) -> "GauntletConfig":
        tau = stats.SCOPE_TAU_CAL.get(scope_id, 3.0)
        return cls(scope_id=scope_id, tau_cal=tau, **overrides)


def _window_bars(window: dict) -> int:
    return int(window.get("bars", 1260))


def stage_s0_calibrate(
    x: list[float],
    y: list[float],
    beta_star: float,
    phi_star: float,
    beta_tol: float = 0.05,
    hl_rel_tol: float = 0.20,
) -> StageRow:
    """S0 harness calibration on a golden AR(1) fixture: the engine must
    reproduce the planted hedge ratio and half-life before any trial runs.
    Verdicts: reproduced | recorded_synthetic (synthetic-only is an honest
    limitation, never a pass)."""
    beta_hat = stats.ols_hedge_ratio(x, y)
    spread = [y[i] - beta_hat * x[i] for i in range(len(x))]
    hl_hat = stats.half_life(spread)
    hl_star = math.log(0.5) / math.log(phi_star)
    beta_ok = abs(beta_hat - beta_star) <= beta_tol
    hl_ok = math.isfinite(hl_hat) and abs(hl_hat - hl_star) / hl_star <= hl_rel_tol
    passed = beta_ok and hl_ok
    return StageRow(
        stage="s0",
        status="pass" if passed else "fail",
        statistic=abs(beta_hat - beta_star),
        threshold=beta_tol,
        p_value=None,
        detail={
            "verdict": "reproduced" if passed else "recorded_synthetic",
            "beta_hat": beta_hat, "beta_star": beta_star,
            "half_life_hat": hl_hat, "half_life_star": hl_star,
            "hl_rel_tol": hl_rel_tol,
        },
    )


def stage_s1_screen(returns: list[float], screen_t: float = stats.DEFAULT_SCREEN_T) -> StageRow:
    """S1 discovery screen: |t| of mean excess return >= screen_t (2.0)."""
    t = stats.t_statistic(returns)
    passed = abs(t) >= screen_t
    return StageRow("s1", "pass" if passed else "fail", statistic=t,
                    threshold=screen_t, p_value=None, detail={"abs_t": abs(t)})


def stage_s2_permutation(
    x: list[float],
    y: list[float],
    B: int = stats.DEFAULT_PERM_B,
    seed: int = 0,
    perm_p: float = stats.DEFAULT_PERM_P,
    half_spread: float = 0.0,
) -> StageRow:
    """S2 block-permutation null on the pairs effect.

    T_obs = t-stat of the strategy on the observed window; null replicates
    block-shuffle one leg (block b = ceil(T^(1/3)), min 5) and recompute the
    t-stat; p = (1 + #{|T_b| >= |T_obs|}) / (B + 1). PCG64-seeded, seed
    recorded to the ledger.
    """
    obs = stats.pair_spread_strategy_returns(x, y, half_spread=half_spread)
    t_obs = obs["t_stat"]

    def effect_fn(xs: list[float], ys: list[float]) -> float:
        r = stats.pair_spread_strategy_returns(xs, ys, half_spread=half_spread)
        return r["t_stat"]

    res = stats.block_permutation_stats(effect_fn, x, y, B=B, seed=seed, T_obs=t_obs)
    passed = res["p_perm"] <= perm_p
    return StageRow(
        stage="s2",
        status="pass" if passed else "fail",
        statistic=t_obs,
        threshold=perm_p,
        p_value=res["p_perm"],
        detail={"B": B, "block_len": res["block_len"], "seed": seed,
                "t_perm_mean": sum(res["t_perm"]) / len(res["t_perm"])},
    )


def stage_s3_multiplicity(
    ledger: Ledger,
    trial_id: str,
    family: str,
    scope_id: str,
    sr_hat_per_bar: float,
    T: int,
    returns: list[float],
    dsr_min: float = stats.DEFAULT_DSR,
    bh_q: float = stats.DEFAULT_BH_Q,
) -> StageRow:
    """S3 multiplicity correction: DSR >= dsr_min with scope-stratified
    denominators from the ledger (N and Vbar of this scope stratum only —
    never pooled), plus BH-FDR q-values within the (family x scope) stratum.
    """
    N, Vbar = ledger.stratum_stats(scope_id)
    if N < 1:
        N, Vbar = 1, 1.0 / T
    Vbar = max(Vbar, 1.0 / T)
    g3, g4 = stats.mean_skew_kurtosis(returns)
    sr_star = stats.expected_max_sharpe(N, Vbar)
    dsr = stats.deflated_sharpe_ratio(sr_hat_per_bar, N, Vbar, T, g3, g4)
    passed = dsr >= dsr_min
    # BH-FDR over the stratum's S2 p-values (family x scope).
    pvals = ledger.stratum_pvalues(family, scope_id)
    qvals = stats.benjamini_hochberg(pvals, q=bh_q) if pvals else []
    return StageRow(
        stage="s3",
        status="pass" if passed else "fail",
        statistic=dsr,
        threshold=dsr_min,
        p_value=None,
        detail={
            "N_stratum": N, "Vbar_stratum": Vbar, "sr_star": sr_star,
            "g3": g3, "g4": g4, "T": T, "sr_hat": sr_hat_per_bar,
            "bh_q": bh_q, "bh_qvalues": qvals, "bh_m": len(pvals),
        },
    )


def stage_s4_walk_forward(
    x: list[float],
    y: list[float],
    T: int,
    tau_cal: float,
    cost_params: dict,
) -> StageRow:
    """S4 cost-aware walk-forward: net SR > 0, |t_net| >= tau_cal (scope-
    matched; ~3.5 pre-registered for the 5y free scope), MinTRL <= T_scope."""
    wf = stats.walk_forward_net_returns(
        x, y, half_spread=cost_params.get("half_spread", 0.01), n_blocks=8
    )
    returns = wf["returns"]
    t_net = wf["t_net"]
    sr = wf["sharpe_per_bar"]
    g3, g4 = stats.mean_skew_kurtosis(returns)
    trl = stats.min_trl(sr, g3, g4) if sr > 0 else math.inf
    passed = (sr > 0.0) and (abs(t_net) >= tau_cal) and (trl <= T)
    return StageRow(
        stage="s4",
        status="pass" if passed else "fail",
        statistic=sr,
        threshold=0.0,
        p_value=None,
        detail={
            "t_net": t_net, "tau_cal": tau_cal, "min_trl": trl, "T": T,
            "net_sr_per_bar": sr, "gross_sr_per_bar": wf["gross_sharpe_per_bar"],
            "g3": g3, "g4": g4, "n_blocks": wf["n_blocks"], "turnover": wf["turnover"],
            "cost_params_hash": hashlib.sha256(_canonical(cost_params).encode()).hexdigest()[:16],
        },
    )


def run_gauntlet_s0_s4(
    ledger: Ledger,
    x: list[float],
    y: list[float],
    *,
    family: str = "pairs",
    scope_id: str = "free_5y_top1k",
    window: dict | None = None,
    params: dict | None = None,
    data_vintage_hash: str = "synthetic-v0",
    provenance: str = "operator_sweep",
    config: GauntletConfig | None = None,
    perm_seed: int = 20261115,
    s0_fixture: tuple[list[float], list[float], float, float] | None = None,
) -> Trial:
    """Run gauntlet stages S0-S4 for one candidate and record every stage row.

    Stops at the first failing stage (a stage that is not run cannot pass);
    verdict `killed` on failure, `survivor` (class on-watch-amber, S5 queued
    automatically by task 3/8 — never opt-in) on full pass. Returns the Trial.
    """
    config = config or GauntletConfig.for_scope(scope_id)
    window = window or {"bars": len(x), "basis": "registered_scope"}
    params = params or {}
    T = _window_bars(window)
    trial_id = ledger.register_trial(family, scope_id, window, params,
                                     data_vintage_hash, provenance)

    # S0 calibration (certification gate). The runner requires the golden
    # fixture: without it the harness is not certified and the trial is
    # killed at S0 — a stage that is not run cannot pass, and synthetic-only
    # calibration is an honest limitation, never a pass.
    if s0_fixture is None:
        s0 = StageRow("s0", "fail", statistic=None, threshold=None,
                      p_value=None,
                      detail={"verdict": "not_certified",
                              "reason": "S0 requires the golden AR(1) fixture; "
                                        "no harness certification on record"})
    else:
        fx, fy, beta_star, phi_star = s0_fixture
        s0 = stage_s0_calibrate(fx, fy, beta_star, phi_star)
    ledger.record_stage(trial_id, s0.stage, s0.status, s0.statistic, s0.threshold,
                        s0.p_value, s0.detail)
    if s0.status != "pass":
        ledger.set_verdict(trial_id, "killed", None)
        return _reload(ledger, trial_id)

    # S1 discovery screen on the observed window.
    obs = stats.pair_spread_strategy_returns(x, y, half_spread=0.0)
    s1 = stage_s1_screen(obs["returns"], config.screen_t)
    ledger.record_stage(trial_id, s1.stage, s1.status, s1.statistic, s1.threshold,
                        s1.p_value, s1.detail)
    if s1.status != "pass":
        ledger.set_verdict(trial_id, "killed", None)
        return _reload(ledger, trial_id)

    # S2 block-permutation null (PCG64 seed recorded).
    s2 = stage_s2_permutation(x, y, B=config.perm_B, seed=perm_seed,
                              perm_p=config.perm_p)
    ledger.record_stage(trial_id, s2.stage, s2.status, s2.statistic, s2.threshold,
                        s2.p_value, s2.detail)
    if s2.status != "pass":
        ledger.set_verdict(trial_id, "killed", None)
        return _reload(ledger, trial_id)

    # S3 DSR + BH-FDR, scope-stratified denominators from the ledger.
    s3 = stage_s3_multiplicity(ledger, trial_id, family, scope_id,
                               obs["sharpe_per_bar"], T, obs["returns"],
                               config.dsr_min, config.bh_q)
    ledger.record_stage(trial_id, s3.stage, s3.status, s3.statistic, s3.threshold,
                        s3.p_value, s3.detail)
    if s3.status != "pass":
        ledger.set_verdict(trial_id, "killed", None)
        return _reload(ledger, trial_id)

    # S4 cost-aware walk-forward at the registered scope.
    s4 = stage_s4_walk_forward(x, y, T, config.tau_cal, config.cost_params)
    ledger.record_stage(trial_id, s4.stage, s4.status, s4.statistic, s4.threshold,
                        s4.p_value, s4.detail)
    if s4.status != "pass":
        ledger.set_verdict(trial_id, "killed", None)
        return _reload(ledger, trial_id)

    # Survived S0-S4 at its registered scope -> on-watch amber; S5 registry-depth
    # validation is queued automatically (task 3/8), never opt-in (v3 SS8.3).
    ledger.set_verdict(trial_id, "survivor", "on-watch-amber")
    return _reload(ledger, trial_id)


def _reload(ledger: Ledger, trial_id: str) -> Trial:
    if isinstance(ledger, SQLiteLedger):
        return ledger.load_trial(trial_id)
    raise TypeError("run_gauntlet_s0_s4 needs a SQLiteLedger-compatible ledger to reload")
