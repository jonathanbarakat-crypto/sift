"""Gauntlet stage runner: executes S0-S4 for a registered trial and records
every stage result into the ledger (append-only, schema of task 1/8).

A stage that is not run cannot pass: the verdict is computed only from
recorded stage rows. S5 is out of scope for this task (wired by task 3/8) and
is never opt-in (v3 §8.3, §12.6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from sift.gauntlet import config as cfg
from sift.gauntlet import stats
from sift.registry import Ledger


@dataclass
class StageResult:
    stage: str
    status: str                 # pass | fail
    statistic: float | None
    threshold: float | None
    p_value: float | None = None
    n_stratum: int | None = None
    v_stratum: float | None = None
    dsr: float | None = None
    q_bh: float | None = None
    seed: int | None = None
    detail: dict = field(default_factory=dict)


class GauntletRunner:
    """Runs stages S0-S4 against a Ledger. Pure orchestration; all statistics
    live in sift.gauntlet.stats."""

    def __init__(self, ledger: Ledger, seed: int = 0):
        self.ledger = ledger
        self.seed = seed

    # -- individual stages -------------------------------------------------
    def run_s0(self, trial_id: str, x: np.ndarray, y: np.ndarray,
               beta_true: float, half_life_true: float) -> StageResult:
        """S0 calibration: AR(1) cointegrated fixture must reproduce the
        analytic hedge ratio and half-life within pre-registered tolerances."""
        r = stats.calibrate_ar1_fixture(
            x, y, beta_true, half_life_true,
            beta_tol=cfg.S0_BETA_TOL, hl_rel_tol=cfg.S0_HL_REL_TOL,
        )
        passed = r.passed
        return StageResult(
            "s0", "pass" if passed else "fail",
            statistic=r.beta_hat, threshold=beta_true,
            detail={"beta_hat": r.beta_hat, "beta_true": r.beta_true,
                    "half_life_hat": r.half_life_hat,
                    "half_life_true": r.half_life_true,
                    "beta_ok": r.beta_ok, "half_life_ok": r.half_life_ok},
        )

    def run_s1(self, trial_id: str, excess_returns: np.ndarray) -> StageResult:
        """S1 discovery screen: |t| >= 2.0 (screen only, not admission)."""
        r = stats.discovery_screen(excess_returns, threshold=cfg.S1_THRESHOLD)
        return StageResult("s1", "pass" if r["pass"] else "fail",
                           statistic=r["t"], threshold=cfg.S1_THRESHOLD)

    def run_s2(self, trial_id: str, series: np.ndarray, statistic_fn,
               B: int = cfg.S2_B, seed: int | None = None) -> StageResult:
        """S2 block-permutation null: p_perm <= 0.05, PCG64-seeded, block
        length ceil(T^(1/3)) (min 5), p = (1+#extreme)/(B+1).

        `statistic_fn(series)` must compute the candidate's predictive
        statistic on a (possibly block-permuted) spread-return series — e.g.
        -corr(cumsum(s)[:-1], s[1:]), the mean-reversion content. Block
        permutation preserves within-block dependence, so the null is valid
        under serial correlation (plain day-shuffle would be anticonservative).
        """
        seed = self.seed if seed is None else seed
        r = stats.block_permutation_null(series, statistic_fn, B=B, seed=seed)
        passed = r["p_perm"] <= cfg.S2_P_MAX
        return StageResult("s2", "pass" if passed else "fail",
                           statistic=r["observed"], threshold=cfg.S2_P_MAX,
                           p_value=r["p_perm"], seed=seed,
                           detail={"B": B, "block_len": r["block_len"]})

    def run_s3(self, trial_id: str, sr_hat_per_bar: float, T: int,
               p_value: float, scope_id: str) -> StageResult:
        """S3 scope-stratified DSR >= 0.95 with N, V from the ledger stratum;
        BH-FDR q=0.10 recorded at family level."""
        n_stratum, v_stratum = self.ledger.stratum_stats(scope_id)
        # the candidate's own S4 statistic enters the stratum variance
        returns = getattr(self, "_last_s4_returns", None)
        skew, kurt = stats.skew_kurt(returns) if returns is not None else (0.0, 3.0)
        d = stats.dsr(sr_hat_per_bar, T, max(n_stratum, 1), v_stratum,
                      skew=skew, kurt=kurt)
        # BH-FDR over the stratum's S2 p-values (family level, same scope)
        rows = self.ledger.con.execute(
            "SELECT sr.p_value FROM stage_result sr JOIN trial t"
            " ON sr.trial_id = t.trial_id WHERE t.scope_id=? AND sr.stage='s2'"
            " AND sr.p_value IS NOT NULL",
            (scope_id,),
        ).fetchall()
        pvals = [r[0] for r in rows] + [p_value]
        bh = stats.bh_fdr(np.array(pvals), q=cfg.S3_BH_Q)
        q_bh = float(bh["q_values"][-1])
        passed = d["dsr"] >= cfg.S3_DSR_MIN
        return StageResult("s3", "pass" if passed else "fail",
                           statistic=d["dsr"], threshold=cfg.S3_DSR_MIN,
                           p_value=p_value, n_stratum=n_stratum,
                           v_stratum=d["v_stratum"], dsr=d["dsr"], q_bh=q_bh,
                           detail={"sr_star": d["sr_star"], "sr_hat": d["sr_hat"],
                                   "skew": skew, "kurt": kurt})

    def run_s4(self, trial_id: str, net_returns: np.ndarray, scope_id: str,
               p_perm: float | None = None) -> StageResult:
        """S4 cost-aware walk-forward: net SR > 0, |t_net| >= tau_cal(scope),
        MinTRL <= T_scope."""
        r = np.asarray(net_returns, float)
        r = r[np.isfinite(r)]
        T = r.size
        sr_ann = stats.sharpe_ratio(r)
        t_net = stats.t_stat_sharpe(r)
        skew, kurt = stats.skew_kurt(r)
        sr_bar = sr_ann / np.sqrt(cfg.BARS_PER_YEAR)
        min_trl = stats.min_trl(sr_bar, skew, kurt, alpha=0.05)
        tau = cfg.tau_cal(scope_id)
        passed = (sr_ann > cfg.S4_NET_SR_MIN) and (abs(t_net) >= tau) and (min_trl <= T)
        self._last_s4_returns = r
        return StageResult("s4", "pass" if passed else "fail",
                           statistic=sr_ann, threshold=tau,
                           p_value=p_perm,
                           detail={"t_net": t_net, "sr_ann": sr_ann,
                                   "min_trl": min_trl, "T": T,
                                   "skew": skew, "kurt": kurt})

    # -- wiring --------------------------------------------------------------
    def record(self, trial_id: str, res: StageResult) -> None:
        self.ledger.record_stage(
            trial_id, res.stage, res.status,
            statistic=res.statistic, threshold=res.threshold,
            p_value=res.p_value, n_stratum=res.n_stratum,
            v_stratum=res.v_stratum, dsr=res.dsr, q_bh=res.q_bh,
            seed=res.seed,
        )

    def run_gauntlet(
        self,
        trial_id: str,
        *,
        fixture_x: np.ndarray | None = None,
        fixture_y: np.ndarray | None = None,
        beta_true: float = 1.0,
        half_life_true: float = math.inf,
        excess_returns: np.ndarray | None = None,
        perm_series: np.ndarray | None = None,
        perm_statistic_fn=None,
        prices: np.ndarray | None = None,
        scope_id: str = "free_5y_top1k",
        s2_B: int = cfg.S2_B,
    ) -> dict:
        """Run S0-S4 in order, recording each stage row. Short-circuits on the
        first failing stage (later stages stay pending — a stage not run
        cannot pass). Returns the verdict dict."""
        trial = self.ledger.get(trial_id)
        if fixture_x is None or fixture_y is None:
            raise ValueError("S0 requires fixture_x/fixture_y")
        if excess_returns is None:
            raise ValueError("S1 requires excess_returns")
        if perm_series is None or perm_statistic_fn is None:
            raise ValueError("S2 requires perm_series and perm_statistic_fn")
        if prices is None:
            raise ValueError("S4 requires prices")
        results: list[StageResult] = []

        s0 = self.run_s0(trial_id, fixture_x, fixture_y, beta_true, half_life_true)
        self.record(trial_id, s0); results.append(s0)
        if s0.status == "fail":
            return self._verdict(trial_id, results, killed_at="s0")

        s1 = self.run_s1(trial_id, excess_returns)
        self.record(trial_id, s1); results.append(s1)
        if s1.status == "fail":
            return self._verdict(trial_id, results, killed_at="s1")

        s2 = self.run_s2(trial_id, perm_series, perm_statistic_fn, B=s2_B)
        self.record(trial_id, s2); results.append(s2)
        if s2.status == "fail":
            return self._verdict(trial_id, results, killed_at="s2")

        # S4 first (its net returns feed S3's skew/kurt and stratum variance)
        wf = stats.walk_forward_returns(prices)
        net = wf["net"]
        s4 = self.run_s4(trial_id, net, scope_id, p_perm=s2.p_value)
        self.record(trial_id, s4); results.append(s4)
        self.ledger.update_var_stat(scope_id, 0.0)  # refresh N and V from ledger
        if s4.status == "fail":
            return self._verdict(trial_id, results, killed_at="s4")

        sr_bar = (s4.detail["sr_ann"] / np.sqrt(cfg.BARS_PER_YEAR))
        s3 = self.run_s3(trial_id, sr_bar, s4.detail["T"], s2.p_value, scope_id)
        self.record(trial_id, s3); results.append(s3)
        if s3.status == "fail":
            return self._verdict(trial_id, results, killed_at="s3")

        # all of S0-S4 passed at the registered scope: survivor, on-watch until
        # S5 (task 3/8) validates at registry depth — never opt-in.
        self.ledger.decide(trial_id, "survivor", "on-watch-amber")
        return {"trial_id": trial_id, "verdict": "survivor",
                "survivor_class": "on-watch-amber", "stages": results,
                "killed_at": None}

    def _verdict(self, trial_id: str, results: list[StageResult], killed_at: str) -> dict:
        self.ledger.decide(trial_id, "killed", "none")
        return {"trial_id": trial_id, "verdict": "killed",
                "survivor_class": "none", "stages": results, "killed_at": killed_at}
