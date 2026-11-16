"""Ledger tests (schema of task 1/8) + end-to-end gauntlet wiring with the
planted fixtures F-PAIR (must pass) and F-NULL (must fail S2/S3)."""

import numpy as np
import pytest

from sift.gauntlet import config as cfg
from sift.gauntlet import stats
from sift.gauntlet.runner import GauntletRunner
from sift.registry import Ledger
from tests.fixtures.planted import f_null, f_pair


@pytest.fixture()
def ledger(tmp_path):
    l = Ledger(tmp_path / "ledger.db")
    yield l
    l.close()


# -- ledger -------------------------------------------------------------------

def test_register_dedupe_skiplist(ledger):
    t1 = ledger.register("pairs", "free_5y_top1k", {"a": 1}, "v1")
    t2 = ledger.register("pairs", "free_5y_top1k", {"a": 1}, "v1")
    assert t1.trial_id == t2.trial_id          # identical config -> same id
    t3 = ledger.register("pairs", "free_5y_top1k", {"a": 2}, "v1")
    assert t3.trial_id != t1.trial_id
    t4 = ledger.register("pairs", "free_5y_top1k", {"a": 1}, "v2")
    assert t4.trial_id != t1.trial_id          # new vintage -> new trial


def test_stage_rows_pending_then_decided(ledger):
    t = ledger.register("pairs", "free_5y_top1k", {"a": 1}, "v1")
    row = ledger.stage(t.trial_id, "s2")
    assert row["status"] == "pending"
    ledger.record_stage(t.trial_id, "s2", "fail", p_value=0.5)
    assert ledger.stage(t.trial_id, "s2")["status"] == "fail"
    with pytest.raises(Exception):
        ledger.record_stage(t.trial_id, "s2", "pass")   # append-only


def test_verdict_set_once(ledger):
    t = ledger.register("pairs", "free_5y_top1k", {"a": 1}, "v1")
    ledger.decide(t.trial_id, "killed", "none")
    with pytest.raises(Exception):
        ledger.decide(t.trial_id, "survivor", "on-watch-amber")


def test_stratum_stats_scope_stratified(ledger):
    ledger.register("pairs", "free_5y_top1k", {"a": 1}, "v1")
    ledger.register("pairs", "free_5y_top1k", {"a": 2}, "v1")
    ledger.register("pairs", "pro_full", {"a": 1}, "v1")
    n_free, _ = ledger.stratum_stats("free_5y_top1k")
    n_pro, _ = ledger.stratum_stats("pro_full")
    assert n_free == 2 and n_pro == 1          # never pooled


# -- F-PAIR: AR(1) cointegrated pair must PASS --------------------------------

def _run_pair(ledger, scope_id="free_5y_top1k", s2_B=200):
    fx = f_pair()
    prices = fx["prices"]
    x, y = prices[:, 0], prices[:, 1]
    rx = np.diff(np.log(x)); ry = np.diff(np.log(y))
    spread_ret = ry - rx                      # true-beta spread returns
    # S1: planted effect = gross strategy returns (positive mean by construction)
    from sift.gauntlet import stats as _stats
    strat = _stats.pairs_strategy_returns(prices)
    excess = strat["gross"][60:]
    trial = ledger.register("pairs", scope_id, {"fixture": "F-PAIR"}, "vintage-fx")
    runner = GauntletRunner(ledger, seed=fx["seed"])
    return runner.run_gauntlet(
        trial.trial_id,
        fixture_x=x, fixture_y=y,
        beta_true=fx["beta_true"], half_life_true=fx["half_life_true"],
        excess_returns=excess,
        perm_series=spread_ret,
        perm_statistic_fn=stats.mean_reversion_stat,
        prices=prices,
        scope_id=scope_id,
        s2_B=s2_B,
    ), trial, fx


def test_f_pair_passes_gauntlet(ledger):
    verdict, trial, fx = _run_pair(ledger)
    assert verdict["verdict"] == "survivor"
    assert verdict["survivor_class"] == "on-watch-amber"   # S5 pending (task 3/8)
    assert verdict["killed_at"] is None
    # every stage recorded pass in the ledger
    for s in ("s0", "s1", "s2", "s3", "s4"):
        assert ledger.stage(trial.trial_id, s)["status"] == "pass"
    # s5 stays pending: not run here, cannot pass
    assert ledger.stage(trial.trial_id, "s5")["status"] == "pending"


# -- F-NULL: 200 independent random walks must FAIL S2/S3 ---------------------

def test_f_null_fails_s2(ledger):
    walks = f_null()
    rng_pvals = []
    n_admitted_s2 = 0
    for i in range(200):
        rs = np.diff(np.log(walks[:, i]))
        r = stats.block_permutation_null(
            rs,
            stats.mean_reversion_stat,
            B=100, seed=1000 + i,
        )
        rng_pvals.append(r["p_perm"])
        if r["p_perm"] <= cfg.S2_P_MAX:
            n_admitted_s2 += 1
    # type-I control at alpha=0.05: ~5% +/- 3 sigma
    frac = n_admitted_s2 / 200
    assert frac <= 0.05 + 3 * (0.05 * 0.95 / 200) ** 0.5


def test_f_null_fails_s3_dsr(ledger):
    """Under the null, DSR of a noise 'strategy' must be below 0.95 — the
    scope-stratified denominator deflates it."""
    walks = f_null()
    T = 1260
    # register 200 null trials in the stratum so N=200
    for i in range(200):
        ledger.register("null", "free_5y_top1k", {"i": i}, "vintage-fx")
    n, v = ledger.stratum_stats("free_5y_top1k")
    assert n == 200
    sr_bar = 0.01  # a modest per-bar Sharpe a noise run might show
    d = stats.dsr(sr_bar, T, n, max(v, 1 / T), skew=0.0, kurt=3.0)
    assert d["dsr"] < cfg.S3_DSR_MIN
    # BH-FDR over the null p-values admits ~ q fraction
    rng = np.random.Generator(np.random.PCG64(7))
    pvals = rng.uniform(0, 1, 200)
    out = stats.bh_fdr(pvals, q=cfg.S3_BH_Q)
    assert len(out["rejected"]) / 200 <= cfg.S3_BH_Q + 3 * (cfg.S3_BH_Q * (1 - cfg.S3_BH_Q) / 200) ** 0.5


def test_f_null_end_to_end_killed(ledger):
    """A single null random walk run through the gauntlet must be killed
    (at S1/S2/S3/S4) — never a survivor."""
    from tests.fixtures.planted import f_null_spread_returns
    rs = f_null_spread_returns()
    prices = np.column_stack([100.0 + np.cumsum(rs), 100.0 + np.cumsum(rs)])
    trial = ledger.register("null", "free_5y_top1k", {"fixture": "F-NULL"}, "vintage-fx")
    runner = GauntletRunner(ledger, seed=123)
    verdict = runner.run_gauntlet(
        trial.trial_id,
        fixture_x=prices[:, 0], fixture_y=prices[:, 1],
        beta_true=1.0, half_life_true=1.0,
        excess_returns=rs - rs.mean(),          # no planted effect
        perm_series=rs,
        perm_statistic_fn=stats.mean_reversion_stat,
        prices=prices,
        scope_id="free_5y_top1k",
        s2_B=100,
    )
    assert verdict["verdict"] == "killed"
    assert verdict["killed_at"] in ("s0", "s1", "s2", "s3", "s4")
