"""S5 registry-depth validation acceptance tests (gate-1 build 3/8).

F-REGIME (load-bearing): a pair with a real edge ONLY in years 0-7 of a
16-year window (AR(1) spread inside, random walk outside). The 5-yr in-edge
window must show t > 2; the full window must attenuate (full-depth SR well
below scope SR, |t_full| < 3.0) so S5 kills it -> on-watch-amber.

F-DELIST: 20% of a universe delists mid-window; the pipeline must not crash
and must not use post-delist prices.

Plus: S5 never opt-in (survivor without S5 stays amber / s5 pending), S5 pass
-> validated-green, ledger rows carry scope registry_depth, thresholds full
strength (3.0 full-depth vs 3.5 free-scope tau_cal).
"""

import numpy as np
import pytest

from sift.gauntlet import config as cfg
from sift.gauntlet import s5 as s5stats
from sift.gauntlet import stats
from sift.gauntlet.runner import GauntletRunner
from sift.registry import Ledger
from tests.fixtures.planted import (
    T_5Y,
    T_FULL,
    f_delist,
    f_pair,
    f_regime,
)


@pytest.fixture()
def ledger(tmp_path):
    led = Ledger(tmp_path / "ledger.db")
    yield led
    led.close()


def _spread_returns(prices):
    rx = np.diff(np.log(prices[:, 0]))
    ry = np.diff(np.log(prices[:, 1]))
    return ry - rx


def _run_regime(ledger, s2_B=200):
    """F-REGIME through the full gauntlet: S0-S4 at the free scope (the 5y
    in-edge window), then automatic S5 at registry_depth (full 16y window)."""
    fx = f_regime()
    prices_scope = fx["prices_scope"]
    prices_full = fx["prices_full"]
    x, y = prices_scope[:, 0], prices_scope[:, 1]
    excess = stats.pairs_strategy_returns(prices_scope)["gross"][60:]
    spread_scope = _spread_returns(prices_scope)
    spread_full = _spread_returns(prices_full)
    trial = ledger.register("pairs", "free_5y_top1k",
                            {"fixture": "F-REGIME"}, "vintage-fx")
    runner = GauntletRunner(ledger, seed=fx["seed"])
    verdict = runner.run_gauntlet(
        trial.trial_id,
        fixture_x=x, fixture_y=y,
        beta_true=fx["beta_true"], half_life_true=fx["half_life_true"],
        fixture_spread=fx["spread"][max(0, fx["t_edge"] - T_5Y): fx["t_edge"]],
        excess_returns=excess,
        perm_series=spread_scope,
        perm_statistic_fn=stats.mean_reversion_stat,
        prices=prices_scope,
        scope_id="free_5y_top1k",
        s2_B=s2_B,
        full_prices=prices_full,
        full_spread_returns=spread_full,
    )
    return verdict, trial, fx


# -- F-REGIME: the load-bearing acceptance test -------------------------------

def test_f_regime_scope_window_shows_edge(ledger):
    """The 5y in-edge window must show t > 2 (S1 screen) — the planted edge
    is real inside its window."""
    fx = f_regime()
    excess = stats.pairs_strategy_returns(fx["prices_scope"])["gross"][60:]
    t_s1 = stats.t_stat_mean(excess)
    assert t_s1 > 2.0
    # and the S2 null on the in-edge spread returns is rejected
    r = stats.block_permutation_null(
        _spread_returns(fx["prices_scope"]), stats.mean_reversion_stat,
        B=200, seed=fx["seed"])
    assert r["p_perm"] <= cfg.S2_P_MAX


def test_f_regime_full_window_attenuates(ledger):
    """The full 16y window must attenuate: full-depth SR well below scope SR
    and |t_full| < 3.0 (the S5 hurdle), so S5 kills the regime-localized edge."""
    fx = f_regime()
    net_scope = stats.walk_forward_returns(fx["prices_scope"])["net"]
    net_scope = net_scope[np.isfinite(net_scope)]
    net_full = stats.walk_forward_returns(fx["prices_full"])["net"]
    net_full = net_full[np.isfinite(net_full)]
    sr_scope = stats.sharpe_ratio(net_scope)
    sr_full = stats.sharpe_ratio(net_full)
    t_scope = stats.t_stat_sharpe(net_scope)
    t_full = stats.t_stat_sharpe(net_full)
    assert abs(t_full) < cfg.S5_T_FULL          # |t_full| < 3.0
    assert sr_full < 0.5 * sr_scope             # full-depth SR well below scope SR
    # theory (design doc §3.1): t_full ≈ f · sqrt(T_F/T_scope) · t_scope with
    # f = 7/16 -> ratio ≈ 0.78; observed must be in the attenuated regime
    ratio = s5stats.attenuation_ratio(t_scope, t_full, T_5Y, T_FULL)
    assert ratio < 0.9


def test_f_regime_s5_kills_it_on_watch_amber(ledger):
    """End-to-end: F-REGIME passes S0-S4 at the free scope (edge is real in
    its window) but S5 at registry_depth fails -> survivor_class =
    on-watch-amber, NOT validated-green. The s5 ledger row is recorded with
    the registry_depth stratum."""
    verdict, trial, fx = _run_regime(ledger)
    assert verdict["verdict"] == "survivor"
    assert verdict["survivor_class"] == "on-watch-amber"
    row = ledger.stage(trial.trial_id, "s5")
    assert row["status"] == "fail"
    assert row["threshold"] == pytest.approx(cfg.S5_T_FULL)   # 3.0 full-depth bar
    assert row["n_stratum"] is not None                        # registry_depth stratum
    # the scope stages passed
    for s in ("s0", "s1", "s2", "s3", "s4"):
        assert ledger.stage(trial.trial_id, s)["status"] == "pass"
    # detail carries the attenuation evidence
    det = verdict["s5"]
    assert abs(det["t_net"]) < cfg.S5_T_FULL
    assert det["checks"]["t_full_depth"] is False


def test_f_regime_s5_thresholds_full_strength(ledger):
    """S5's hurdle is 3.0 at full depth vs tau_cal 3.5 at the free scope —
    S5 is never weakened to the scope's calibrated value, and never opt-in."""
    assert cfg.S5_T_FULL == 3.0
    assert cfg.tau_cal("free_5y_top1k") == 3.5
    assert cfg.tau_cal("registry_depth") == 3.0
    verdict, trial, fx = _run_regime(ledger)
    row = ledger.stage(trial.trial_id, "s5")
    assert row["threshold"] == pytest.approx(3.0)   # not 3.5, not 2.0


# -- S5 never opt-in ------------------------------------------------------------

def test_s5_never_opt_in_pending_means_not_validated(ledger):
    """A survivor run WITHOUT full-window data keeps s5 pending and stays
    on-watch-amber: a stage not run cannot pass (F-S5-LEDGER rule)."""
    fx = f_pair()
    prices = fx["prices"]
    x, y = prices[:, 0], prices[:, 1]
    excess = stats.pairs_strategy_returns(prices)["gross"][60:]
    trial = ledger.register("pairs", "free_5y_top1k", {"fixture": "F-PAIR"}, "v1")
    runner = GauntletRunner(ledger, seed=fx["seed"])
    verdict = runner.run_gauntlet(
        trial.trial_id,
        fixture_x=x, fixture_y=y,
        beta_true=fx["beta_true"], half_life_true=fx["half_life_true"],
        excess_returns=excess,
        perm_series=_spread_returns(prices),
        perm_statistic_fn=stats.mean_reversion_stat,
        prices=prices,
        scope_id="free_5y_top1k",
        s2_B=200,
        # no full_prices -> S5 not run
    )
    assert verdict["verdict"] == "survivor"
    assert verdict["survivor_class"] == "on-watch-amber"
    assert ledger.stage(trial.trial_id, "s5")["status"] == "pending"


def test_s5_pass_gives_validated_green(ledger):
    """A persistent edge (AR(1) over the FULL window) must pass S5 ->
    validated-green. Uses a full-window AR(1) pair for both the scope window
    and the registry_depth re-evaluation. Strong in-edge vol so the 5y scope
    also clears tau_cal=3.5."""
    T = T_FULL
    rng = np.random.Generator(np.random.PCG64(7))
    lx = np.cumsum(rng.normal(0, 0.01, T))
    e = rng.normal(0, 1.0, T)
    phi, sig_in = 0.7, 10.0
    s = np.zeros(T)
    for t in range(1, T):
        s[t] = phi * s[t - 1] + e[t] * sig_in
    px = 100.0 * np.exp(lx)
    py = px * np.exp(s / 100.0)
    prices_full = np.column_stack([px, py])
    prices_scope = prices_full[-T_5Y:]
    x, y = prices_scope[:, 0], prices_scope[:, 1]
    excess = stats.pairs_strategy_returns(prices_scope)["gross"][60:]
    trial = ledger.register("pairs", "free_5y_top1k", {"fixture": "F-PAIR-FULL"}, "v1")
    runner = GauntletRunner(ledger, seed=7)
    beta_true = float(np.dot(x - x.mean(), y - y.mean()) / np.dot(x - x.mean(), x - x.mean()))
    verdict = runner.run_gauntlet(
        trial.trial_id,
        fixture_x=x, fixture_y=y,
        beta_true=beta_true, half_life_true=-np.log(2.0) / np.log(phi),
        fixture_spread=s[-T_5Y:],
        excess_returns=excess,
        perm_series=_spread_returns(prices_scope),
        perm_statistic_fn=stats.mean_reversion_stat,
        prices=prices_scope,
        scope_id="free_5y_top1k",
        s2_B=200,
        full_prices=prices_full,
        full_spread_returns=_spread_returns(prices_full),
    )
    assert verdict["verdict"] == "survivor"
    assert verdict["survivor_class"] == "validated-green"
    assert ledger.stage(trial.trial_id, "s5")["status"] == "pass"


# -- F-DELIST -------------------------------------------------------------------

def test_f_delist_mask_excludes_post_delist():
    """The delist mask marks exactly the post-delist cells invalid; panel
    stats count zero valid bars after each delist point."""
    fx = f_delist(T=T_FULL, n=50, delist_frac=0.20)
    assert fx["n_delisted"] == 10                      # 20% of 50
    mask = s5stats.delist_mask(fx["prices"], fx["delist_idx"])
    ps = s5stats.panel_stats_valid(fx["prices"], fx["delist_idx"])
    for j in range(fx["n"]):
        d = fx["delist_idx"][j]
        assert mask[d:, j].sum() == 0                   # nothing post-delist
        assert ps["valid_bars"][j] == d                # exactly d valid bars
        assert ps["valid_returns"][j] == max(d - 1, 0)


def test_f_delist_valid_returns_nan_not_zero():
    """Post-delist returns are NaN, never a fabricated 0% return."""
    fx = f_delist(T=T_FULL, n=50, delist_frac=0.20)
    r = s5stats.valid_returns(fx["prices"], fx["delist_idx"])
    s5stats.delist_mask(fx["prices"], fx["delist_idx"])
    for j in range(fx["n"]):
        d = fx["delist_idx"][j]
        if d < fx["T"]:
            assert np.all(~np.isfinite(r[d:, j]))      # NaN after delist


def test_f_delist_pipeline_does_not_crash_and_ignores_post_delist(ledger):
    """Run S5 on a delist-heavy panel: the pipeline must not crash, and the
    S5 statistics must be computed only from pre-delist cells. We build a
    two-leg panel where leg 1 delists at T//2; S5 runs on the masked window
    and produces a finite verdict whose recorded window is the leading
    pre-delist prefix (T_use <= T//2), with S2 fed only the pre-delist
    spread returns."""
    fx = f_delist(T=T_FULL, n=50, delist_frac=0.20, seed=4242)
    prices = fx["prices"].copy()
    delist_idx = fx["delist_idx"].copy()
    # force leg 1 to delist exactly at the midpoint
    delist_idx[1] = T_FULL // 2
    prices[T_FULL // 2:, 1] = np.nan
    # honest spread returns (NaN past the delist, NOT nan_to_num)
    raw_spread = np.diff(np.log(prices[:, 1])) - np.diff(np.log(prices[:, 0]))
    runner = GauntletRunner(ledger, seed=1)
    trial = ledger.register("pairs", "free_5y_top1k", {"fixture": "F-DELIST"}, "v1")
    s5 = runner.run_s5(
        trial.trial_id,
        full_prices=prices,
        full_spread_returns=raw_spread,
        statistic_fn=stats.mean_reversion_stat,
        delist_idx=delist_idx,
        s2_B=50,
        seed=1,
    )
    # no crash, finite statistics, verdict decided
    assert s5.status in ("pass", "fail")
    assert np.isfinite(s5.detail["t_net"])
    # the walk-forward window is the leading pre-delist prefix: leg 1 delists
    # at T//2, so no bar >= T//2 may be used
    assert s5.detail["T_window_used"] <= T_FULL // 2
    # S2 saw only returns whose BOTH endpoints are pre-delist: at most T//2-1
    assert s5.detail["n_s2_returns"] <= T_FULL // 2 - 1
    # record into the ledger works and marks the registry_depth stratum
    runner.record(trial.trial_id, s5)
    row = ledger.stage(trial.trial_id, "s5")
    assert row["status"] == s5.status
    assert row["n_stratum"] is not None


def test_s2_spread_returns_alignment_both_endpoints():
    """valid_spread_returns keeps return k (bar k -> k+1) ONLY if both bars
    are valid: a mid-panel NaN in one leg must kill the returns on BOTH sides
    of the hole, not just after it."""
    T = 12
    prices = np.full((T, 2), 100.0)
    prices[5, 0] = np.nan               # hole at bar 5 in leg 0
    mask = s5stats.delist_mask(prices)
    sr = np.arange(T - 1, dtype=float)  # return k = k
    out = s5stats.valid_spread_returns(sr, mask)
    # returns 4 (4->5) and 5 (5->6) touch the invalid bar 5 -> NaN
    assert not np.isfinite(out[4]) and not np.isfinite(out[5])
    # all others survive
    for k in range(T - 1):
        if k not in (4, 5):
            assert out[k] == k


def test_leading_valid_window_is_prefix_not_count():
    """leading_valid_window returns the leading all-valid prefix length, not
    the count of all-valid rows scattered through the panel."""
    T = 10
    mask = np.ones((T, 2), bool)
    mask[3, 1] = False     # hole at bar 3
    mask[7, 0] = False     # another hole later — count would be 8, prefix is 3
    assert s5stats.leading_valid_window(mask) == 3


def test_s5_full_depth_verdict_pure():
    """Pure-function check of the S5 verdict: a strong full-window edge
    passes; a weak/attenuated one fails the t_full_depth check."""
    rng = np.random.Generator(np.random.PCG64(5))
    strong = rng.normal(0.002, 0.01, T_FULL)     # ~ t >> 3
    weak = rng.normal(0.0002, 0.01, T_FULL)      # |t| < 3
    v_pass = s5stats.s5_full_depth_verdict(strong, t_hurdle=3.0, p_perm=0.001, dsr=0.99, pbo=0.1)
    v_fail = s5stats.s5_full_depth_verdict(weak, t_hurdle=3.0, p_perm=0.9, dsr=0.5, pbo=0.9)
    assert v_pass["pass"] is True
    assert v_fail["pass"] is False
    assert v_fail["checks"]["t_full_depth"] is False
    assert v_fail["checks"]["p_perm"] is False
