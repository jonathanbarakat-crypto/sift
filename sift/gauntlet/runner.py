"""Gauntlet stage runner: executes S0-S5 for a registered trial and records
every stage result into the ledger (append-only, schema of task 1/8).

A stage that is not run cannot pass: the verdict is computed only from
recorded stage rows. S5 registry-depth validation runs automatically on every
S0-S4 survivor — never opt-in, never skippable (v3 §8.3): survivors that fail
full-depth become on-watch-amber, not validated-green (v3 §8.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from sift.gauntlet import config as cfg
from sift.gauntlet import s5 as s5stats
from sift.gauntlet import stats
from sift.gauntlet import v42
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

    def __init__(self, ledger: Ledger, seed: int = 0, card_minter=None):
        self.ledger = ledger
        self.seed = seed
        # v4.1: optional CardMinter (sift.cards.minting). When set, a
        # survivor mints exactly one card (flag-gated) — v4.2: EVERY S0-S2
        # passer mints at bronze+; killed trials (S0-S2 failures) mint
        # nothing. None keeps v3 behavior.
        self.card_minter = card_minter

    # -- individual stages -------------------------------------------------
    def run_s0(self, trial_id: str, x: np.ndarray, y: np.ndarray,
               beta_true: float, half_life_true: float,
               spread: np.ndarray | None = None) -> StageResult:
        """S0 calibration: AR(1) cointegrated fixture must reproduce the
        analytic hedge ratio and half-life within pre-registered tolerances.

        `spread` (optional) is the PLANTED spread series; when given, the
        half-life assertion runs on it directly (log-space fixtures plant the
        spread there), otherwise on y - beta_true*x as before.

        REAL-DATA ROLE (finding #14, doc-only — no behavior change): on real
        data (first_run) beta_true is the pair's own levels-OLS hedge and
        half_life_true the measured half-life, so the beta assertion is
        tautological (beta_hat - beta_true = 0.0 exactly) and S0's genuine
        real-data gate is the finite-half-life (mean-reversion) check — an
        explosive spread (phi_hat > 1, half-life = inf) fails half_life_ok
        and therefore S0. The planted AR(1) fixture (F-PAIR) remains the
        real calibration of both assertions."""
        r = stats.calibrate_ar1_fixture(
            x, y, beta_true, half_life_true,
            beta_tol=cfg.S0_BETA_TOL, hl_rel_tol=cfg.S0_HL_REL_TOL,
            spread=spread,
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
        """S3 scope-stratified DSR >= 0.95 AND BH-FDR enforced at family level:
        the candidate must be in the BH rejection set at q = S3_BH_Q over the
        stratum's S2 p-values (soundness #6, path A ENFORCE — the README
        advertises 'DSR>=0.95 + BH-FDR', so both gate).

        UNITS (#1/#16): `sr_hat_per_bar` is PER-BAR (callers pass
        sr_ann/sqrt(BARS_PER_YEAR)) and the ledger's stratum V is per-bar
        SR^2 (update_var_stat converts the stored ANNUALIZED s4 statistics
        before the variance), so sr_star inside stats.dsr shares units with
        sr_hat; stats.dsr runs the mint-time guard assert_sr_units_consistent
        before returning.

        What BH adds beyond S2's per-trial p<=0.05: S2 is a per-trial screen;
        BH-FDR controls the family-wise false discovery rate across the whole
        stratum (all trials sharing this scope), so a candidate that is only
        marginally significant in a large family (q_bh > q) is NOT admitted.

        Dependence caveat (documented, not silently assumed): BH assumes
        independence/PRDS on the p-values. Within-stratum trials share the
        universe and overlapping windows, so their p-values are positively
        correlated. Under PRDS BH still controls FDR; the conservative
        alternative under arbitrary dependence is Benjamini-Yekutieli
        (q/m -> q/(m * sum 1/i)). We state the PRDS assumption here rather
        than pretending independence; if the stratum is ever found to
        violate PRDS, switch to BY — do not weaken the gate.

        v4.2 §2-§3 (#170): S3's DSR is the MINT-TIME ADMISSION GATE only. The
        stage row records dsr, sr_star, n_stratum, v_stratum, q_bh as
        "admission evidence at mint-time N" context — the card tier never
        re-deflates by a later N."""
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
        cand_idx = len(pvals) - 1                 # the candidate is appended last
        q_bh = float(bh["q_values"][cand_idx])
        in_rejection_set = cand_idx in set(int(i) for i in bh["rejected"])
        # path A: BOTH gates — DSR threshold AND BH membership at q=S3_BH_Q
        dsr_ok = d["dsr"] >= cfg.S3_DSR_MIN
        bh_ok = in_rejection_set and q_bh <= cfg.S3_BH_Q
        passed = bool(dsr_ok and bh_ok)
        return StageResult("s3", "pass" if passed else "fail",
                           statistic=d["dsr"], threshold=cfg.S3_DSR_MIN,
                           p_value=p_value, n_stratum=n_stratum,
                           v_stratum=d["v_stratum"], dsr=d["dsr"], q_bh=q_bh,
                           detail={"sr_star": d["sr_star"], "sr_hat": d["sr_hat"],
                                   "skew": skew, "kurt": kurt,
                                   "admission_label": (
                                       "admission evidence at mint-time N "
                                       "(DSR gate only — never a tier input, "
                                       "never re-deflated by a later N)"),
                                   "dsr_ok": bool(dsr_ok), "bh_ok": bool(bh_ok),
                                   "bh_rejected": bool(in_rejection_set),
                                   "bh_m": int(bh["m"]),
                                   "bh_q": cfg.S3_BH_Q,
                                   "bh_prds_assumption": (
                                       "BH assumes independence/PRDS; "
                                       "stratum p-values are correlated "
                                       "(shared universe, overlapping "
                                       "windows) — PRDS stated, BY is the "
                                       "conservative fallback under "
                                       "arbitrary dependence")})

    def run_s4(self, trial_id: str, net_returns: np.ndarray, scope_id: str,
               p_perm: float | None = None) -> StageResult:
        """S4 cost-aware walk-forward — v4.2 §2: a SCORED stage, not a kill.
        The stage row records pass|fail against the pre-registered bar
        (net SR > S4_NET_SR_MIN, |t_net| >= tau_cal(scope), MinTRL <= T) AND
        the full score (net SR, t_net, MinTRL, T) in the stage detail either
        way; in v4.2 the bar defines the gold/platinum evidence tiers, not a
        kill line — S0-S2 passers survive S4 regardless.

        UNITS (#1): the s4 stage row stores statistic=sr_ann (ANNUALIZED);
        it stays annualized because every reader of the column (minting tier
        evidence, decay re-eval) expects sr_ann — the per-bar conversion for
        the stratum variance happens in Ledger.update_var_stat, not here."""
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
                                   "skew": skew, "kurt": kurt,
                                   "tau_cal": tau,
                                   "s4_bar_met": bool(passed)})

    # -- S5: registry-depth validation (automatic, never opt-in) ------------
    def run_s5(
        self,
        trial_id: str,
        *,
        full_prices: np.ndarray,
        full_spread_returns: np.ndarray,
        statistic_fn,
        delist_idx: np.ndarray | None = None,
        s2_B: int = cfg.S2_B,
        seed: int | None = None,
        scope_id: str = "registry_depth",
    ) -> StageResult:
        """S5 registry-depth validation: re-runs S2-S4 statistics at the
        `registry_depth` scope (full window) on every S0-S4 survivor.

        Thresholds are at FULL STRENGTH and never weakened (design doc §3.2
        rule 2): p_perm <= 0.05 (block b=ceil(T_F^(1/3))), DSR >= 0.95 with the
        registry_depth stratum denominator, net SR > 0, |t_net| >= 3.0
        (full-depth hurdle — vs tau_cal=3.5 at the 5-yr free scope), MinTRL
        <= T_F, PBO <= 0.5. Post-delist prices are excluded via the delist
        mask (F-DELIST): no statistic reads a post-delist cell.

        PBO is COMPUTED, never hardcoded (soundness #2): CSCV (S=16) over
        the pre-registered variant grid (config.S5_VARIANT_GRID) of
        walk-forward configs around the registered one, run on the same
        delist-safe leading window; the grid digest is recorded in the
        stage detail. If the window cannot support the CSCV design,
        cscv_pbo returns None and the pbo check FAILS CLOSED (soundness
        #15) — an uncomputed PBO can never pass as 0.0.

        Pass -> validated-green; fail -> on-watch-amber (v3 §8.4). Both are
        recorded as the trial's s5 stage row with scope registry_depth.
        """
        seed = self.seed if seed is None else seed
        prices = np.asarray(full_prices, float)
        mask = s5stats.delist_mask(prices, delist_idx)
        # S2 at full depth on the delist-safe spread returns: a return from
        # bar k to k+1 is used ONLY if both endpoint bars are valid in both
        # legs (post-delist returns are NaN, never kept — F-DELIST).
        spread_ret = np.asarray(full_spread_returns, float)
        sr = s5stats.valid_spread_returns(spread_ret, mask)
        sr = sr[np.isfinite(sr)]
        s2 = stats.block_permutation_null(sr, statistic_fn, B=s2_B, seed=seed)
        p_perm = s2["p_perm"]
        # S4 walk-forward at full depth runs ONLY on the leading pre-delist
        # window (leading_valid_window) — never on a window that contains a
        # post-delist row in any leg.
        T_use = min(prices.shape[0], max(s5stats.leading_valid_window(mask), 2))
        window = prices[:T_use]
        wf = stats.walk_forward_returns(window)
        net = wf["net"]
        net = net[np.isfinite(net)]
        # PBO via CSCV (S=16) over the PRE-REGISTERED variant grid around
        # the registered config (config.S5_VARIANT_GRID) — a genuine
        # multi-configuration returns matrix. CSCV on a single config is
        # meaningless, so the old hardcoded pbo=0.0 shortcut is gone
        # (soundness #2). Each variant runs the same delist-safe leading
        # window walk-forward with its own formation/open_z/close_z/step;
        # the grid digest is recorded in the stage detail so the PBO input
        # is auditable / pre-registered. If the window is too short for
        # S blocks, cscv_pbo returns None and the verdict FAILS CLOSED on
        # the pbo check (soundness #15) — never a fabricated 0.0.
        grid = cfg.s5_variant_grid()
        cols = []
        for v in grid:
            wv = stats.walk_forward_returns(
                window,
                formation=int(v["formation"]), step=int(v["step"]),
                open_z=float(v["open_z"]), close_z=float(v["close_z"]),
            )
            nv = wv["net"]
            cols.append(nv[np.isfinite(nv)] if nv.size == T_use else
                        np.full(T_use, np.nan))
        # align columns on the window length (pad/truncate to T_use)
        R = np.full((T_use, len(cols)), np.nan)
        for j, c in enumerate(cols):
            k = min(T_use, c.size)
            R[:k, j] = c[:k]
        pbo = stats.cscv_pbo(R, S=cfg.S5_CSCV_S, seed=seed)
        pbo_n_configs = len(grid)
        pbo_grid_digest = cfg.s5_variant_grid_digest()
        # S3 at registry_depth with the stratum denominator from the ledger
        # (UNITS #1: sr_bar is per-bar; the ledger V is per-bar SR^2 —
        # stats.dsr guards the units at mint time)
        sr_ann = stats.sharpe_ratio(net) if net.size >= 2 else 0.0
        t_net = stats.t_stat_sharpe(net) if net.size >= 2 else 0.0
        T = int(net.size)
        sr_bar = sr_ann / math.sqrt(cfg.BARS_PER_YEAR) if net.size >= 2 else 0.0
        n_stratum, v_stratum = self.ledger.stratum_stats(scope_id)
        skew, kurt = stats.skew_kurt(net) if net.size >= 4 else (0.0, 3.0)
        d = stats.dsr(sr_bar, max(T, 2), max(n_stratum, 1), v_stratum,
                      skew=skew, kurt=kurt)
        verdict = s5stats.s5_full_depth_verdict(
            net,
            t_hurdle=cfg.S4_T_NET_FULL,          # 3.0 full-depth (never 3.5-hurdle-waived)
            net_sr_min=cfg.S4_NET_SR_MIN,
            min_trl_max=cfg.scope_T(scope_id),
            p_perm=p_perm,
            p_perm_max=cfg.S2_P_MAX,
            dsr=d["dsr"],
            dsr_min=cfg.S3_DSR_MIN,
            pbo=pbo,
            pbo_max=cfg.S4_PBO_MAX,
        )
        passed = bool(verdict["pass"])
        return StageResult(
            "s5", "pass" if passed else "fail",
            statistic=t_net, threshold=cfg.S4_T_NET_FULL,
            p_value=p_perm, n_stratum=n_stratum,
            v_stratum=d["v_stratum"], dsr=d["dsr"], seed=seed,
            detail={"t_net": t_net, "sr_ann": sr_ann, "T": T,
                    "min_trl": verdict["min_trl"], "pbo": pbo,
                    "pbo_n_configs": pbo_n_configs,
                    "pbo_grid_digest": pbo_grid_digest,
                    "pbo_S": cfg.S5_CSCV_S,
                    "p_perm": p_perm, "block_len": s2["block_len"],
                    "T_window_used": T_use,
                    "n_s2_returns": int(sr.size),
                    "checks": verdict["checks"],
                    "scope": scope_id, "survivor_class": (
                        "validated-green" if passed else "on-watch-amber")},
        )


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
        fixture_spread: np.ndarray | None = None,
        excess_returns: np.ndarray | None = None,
        perm_series: np.ndarray | None = None,
        perm_statistic_fn=None,
        prices: np.ndarray | None = None,
        scope_id: str = "free_5y_top1k",
        s2_B: int = cfg.S2_B,
        full_prices: np.ndarray | None = None,
        full_spread_returns: np.ndarray | None = None,
        delist_idx: np.ndarray | None = None,
    ) -> dict:
        """Run S0-S4 in order, recording each stage row. Short-circuits on the
        first failing stage (later stages stay pending — a stage not run
        cannot pass). If `full_prices` + `full_spread_returns` (the
        registry_depth full window) are supplied, S5 runs automatically after
        an S0-S4 pass and decides the survivor class (validated-green vs
        on-watch-amber). `delist_idx` masks post-delist bars everywhere.
        Returns the verdict dict."""
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

        s0 = self.run_s0(trial_id, fixture_x, fixture_y, beta_true,
                         half_life_true, spread=fixture_spread)
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

        # S4 first (its net returns feed S3's skew/kurt and stratum variance).
        # v4.2 §2: S4 is SCORED, not a kill — a fail here does NOT short-circuit.
        wf = stats.walk_forward_returns(prices)
        net = wf["net"]
        s4 = self.run_s4(trial_id, net, scope_id, p_perm=s2.p_value)
        self.record(trial_id, s4); results.append(s4)
        self.ledger.update_var_stat(scope_id)  # refresh N and V from ledger
        s4_bar_met = v42.s4_passed(s4.detail, scope_id)

        sr_bar = (s4.detail["sr_ann"] / np.sqrt(cfg.BARS_PER_YEAR))
        s3 = self.run_s3(trial_id, sr_bar, s4.detail["T"], s2.p_value, scope_id)
        self.record(trial_id, s3); results.append(s3)
        s3_pass = s3.status == "pass"
        # v4.2 §2-§3: an S0-S2 passer that misses the S4 or S3 bar is NOT
        # killed — it is a survivor at the bronze evidence tier.
        if not s4_bar_met or not s3_pass:
            cls = v42.survivor_class_for(
                s4_pass=s4_bar_met, s3_pass=s3_pass, s5_pass=None)
            self.ledger.decide(trial_id, "survivor", cls)
            card_id = self._maybe_mint_card(trial_id, "survivor", cls)
            return {"trial_id": trial_id, "verdict": "survivor",
                    "survivor_class": cls, "stages": results,
                    "killed_at": None, "card_id": card_id,
                    "s4_score": {k: s4.detail.get(k) for k in
                                 ("sr_ann", "t_net", "min_trl", "T", "tau_cal",
                                  "s4_bar_met")}}

        # All of S0-S4 passed at the registered scope. S5 registry-depth
        # validation is AUTOMATIC and never opt-in (v3 §8.3): when the
        # full-window (registry_depth) data is supplied, S5 runs now and the
        # survivor class is decided by its outcome — pass -> validated-green,
        # fail -> on-watch-amber (v3 §8.4). Without full-window data the trial
        # stays on-watch-amber with the s5 row pending (a stage not run cannot
        # pass).
        s5_pass: bool | None = None
        if full_prices is not None and full_spread_returns is not None:
            s5 = self.run_s5(
                trial_id,
                full_prices=full_prices,
                full_spread_returns=full_spread_returns,
                statistic_fn=perm_statistic_fn,
                delist_idx=delist_idx,
                s2_B=s2_B,
                seed=self.seed,
            )
            self.record(trial_id, s5); results.append(s5)
            self.ledger.update_var_stat("registry_depth")
            s5_pass = s5.status == "pass"
            cls = s5.detail["survivor_class"]
            self.ledger.decide(trial_id, "survivor", cls)
            card_id = self._maybe_mint_card(trial_id, "survivor", cls)
            return {"trial_id": trial_id, "verdict": "survivor",
                    "survivor_class": cls, "stages": results,
                    "killed_at": None, "s5": s5.detail, "card_id": card_id,
                    "s4_score": {k: s4.detail.get(k) for k in
                                 ("sr_ann", "t_net", "min_trl", "T", "tau_cal",
                                  "s4_bar_met")}}

        self.ledger.decide(trial_id, "survivor", "on-watch-amber")
        card_id = self._maybe_mint_card(trial_id, "survivor", "on-watch-amber")
        return {"trial_id": trial_id, "verdict": "survivor",
                "survivor_class": "on-watch-amber", "stages": results,
                "killed_at": None, "card_id": card_id,
                "s4_score": {k: s4.detail.get(k) for k in
                             ("sr_ann", "t_net", "min_trl", "T", "tau_cal",
                              "s4_bar_met")}}

    def _maybe_mint_card(self, trial_id: str, verdict: str, survivor_class: str) -> str | None:
        """v4.2 mechanics: mint a card on EVERY S0-S2 passer (bronze+),
        behind the cards_v4_1 flag. Killed trials (S0-S2 failures,
        survivor_class='none') mint nothing. Owner handle: the trial's
        discoverer; operator sweeps use 'operator'. Returns the
        read-back-verified card id or None."""
        if self.card_minter is None:
            return None
        handle = self.ledger.con.execute(
            "SELECT discoverer_handle FROM trial WHERE trial_id=?", (trial_id,)
        ).fetchone()["discoverer_handle"] or "operator"
        return self.card_minter.maybe_mint(trial_id, handle, verdict, survivor_class)

    def _verdict(self, trial_id: str, results: list[StageResult], killed_at: str) -> dict:
        self.ledger.decide(trial_id, "killed", "none")
        card_id = self._maybe_mint_card(trial_id, "killed", "none")
        return {"trial_id": trial_id, "verdict": "killed",
                "survivor_class": "none", "stages": results, "killed_at": killed_at,
                "card_id": card_id}
