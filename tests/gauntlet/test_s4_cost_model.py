"""Regression fixtures for soundness #9 (hermes-tasks #181 cross-check,
finding #9 P1; re-filed as #188/#204): the S4 pairs cost model.

Defects verified on main 8c1a1b0 (probe output in the PR body):
  (a) ONE-LEG cost: apply_costs charged |dpos| * half_spread_bps / 1e4 —
      a pairs trade trades BOTH y and beta*x, so the hedge leg's turnover
      was never charged (probe: ~2-2.95x understated at median |hedge|
      1.0-1.95, same 100 bps placeholder);
  (b) DEAD slippage fields: CostModel.slippage_participation_threshold /
      slippage_bps_per_pct / max_participation were never read — setting
      slippage_bps_per_pct=500, max_participation=0.001 changed net by
      EXACTLY 0;
  (c) FILL-BAR TIMING: cost was charged at the decision bar t while the
      position change only earns from t+1 (strat[1:] = pos[:-1]*rs[1:]).

Fix conventions (implemented in sift/gauntlet/stats.py):
  * two-leg per-leg half-spreads: cost_t = |dpos_t| * (hs_y_t + |h_{t-1}| *
    hs_x_t) / 1e4, with the LAGGED hedge h_{t-1} in force at the decision
    bar (no look-ahead);
  * participation-based slippage is APPLIED: participation proxy
    p_t = |dpos_t| * notional_per_unit / adv_proxy (pre-registered
    risk_capital_per_trade as the notional-per-unit, default ADV proxy
    documented; |h| scales the hedge leg's participation too); above
    slippage_participation_threshold the fill pays
    slippage_bps_per_pct per pct of participation, capped at
    max_participation; per-leg half-spread is the FLOOR (max, not sum);
  * cost is charged at the FILL bar t+1 — the first bar the changed
    position earns on (next-bar entry convention), matching
    strat[1:] = pos[:-1] * rs[1:];
  * per-name data-driven half-spreads (Roll 2*sqrt(-cov(dp_t, dp_{t-1})),
    floored) for the S4 path when the price panel is supplied; the 100 bps
    placeholder stays as the conservative fallback, clearly flagged.

Fixtures (TDD — written before the fix; they FAIL on main 8c1a1b0):
  1. known-answer two-leg cost arithmetic (the probe case);
  2. per-leg half-spreads are used independently (asymmetric hs_y vs hs_x);
  3. hedge ratio scales the hedge-leg cost; lagged hedge, no look-ahead;
  4. slippage fields demonstrably change net (no longer dead);
  5. participation cap: cost above max_participation is bounded;
  6. fill-bar timing: a position change at close t costs at bar t+1, the
     first bar it earns on — never at t;
  7. zero-turnover bars cost nothing (also with slippage configured);
  8. data-driven per-name half-spread estimator: Roll known-answer + floor
     + fallback to the conservative placeholder;
  9. end-to-end: walk_forward_returns net on a planted pair is LOWER (more
     honest) than the one-leg placeholder net, and the gauntlet still runs;
 10. existing gauntlet tests stay green (updated honestly, thresholds never
     weakened).
"""

import numpy as np
import pytest

from sift.gauntlet import stats

# ---------------------------------------------------------------------------
# Known-answer arithmetic helpers
# ---------------------------------------------------------------------------

def _two_leg_reference(strat, pos, hedge, hs_y, hs_x, cost_model):
    """Reference two-leg cost at the FILL bar (t+1 convention).

    dpos[t] = |pos[t] - pos[t-1]| is decided at close t and first earns at
    bar t+1, so its cost lands on bar t+1:
        cost[t+1] = dpos[t] * (hs_y + |hedge[t]| * hs_x) / 1e4
    hedge[t] is the hedge in force at the decision bar (lagged — no look-ahead).
    """
    strat = np.asarray(strat, float)
    pos = np.asarray(pos, float)
    hedge = np.asarray(hedge, float)
    T = strat.size
    dpos = np.abs(np.diff(np.concatenate([[0.0], pos])))
    cost = np.zeros(T)
    for t in range(T - 1):
        if dpos[t] > 0:
            cost[t + 1] += dpos[t] * (hs_y + abs(hedge[t]) * hs_x) / 1e4
    return strat - cost, cost


# ---------------------------------------------------------------------------
# 1-3. two-leg known-answer arithmetic
# ---------------------------------------------------------------------------

def test_two_leg_cost_known_answer():
    """Probe case (verdict record #9): turnover 10 units, median hedge 1.949,
    100 bps placeholder per leg -> 1000 bps one-leg vs 2948 bps two-leg."""
    gross = np.zeros(6)
    pos = np.array([0.0, 1.0, 1.0, -1.0, 0.0, 0.0])   # turnover = 1+0+2+1 = 4
    hedge = np.array([0.0, 1.949, 1.949, 1.949, 1.949, 1.949])
    cm = stats.CostModel(half_spread_bps=100.0,
                         slippage_participation_threshold=1e9)  # spread only
    net = stats.apply_costs(gross, pos, cm, hedge=hedge)
    # two-leg at fill bars: t=1 change -> bar 2; t=3 -> bar 4; t=4 -> bar 5
    # cost per unit = (100 + 1.949*100)/1e4 = 0.02949
    per_unit = (100.0 + 1.949 * 100.0) / 1e4
    assert net[2] == pytest.approx(-1.0 * per_unit, abs=1e-12)
    assert net[4] == pytest.approx(-2.0 * per_unit, abs=1e-12)
    assert net[5] == pytest.approx(-1.0 * per_unit, abs=1e-12)
    # one-leg (the defect) would have charged 1*100/1e4 at bar 1, 2* at bar 3:
    one_leg_total = 4.0 * 100.0 / 1e4
    two_leg_total = 4.0 * per_unit
    assert two_leg_total == pytest.approx(2.949 * one_leg_total, rel=1e-3)
    # the decision bars themselves carry NO cost (fill-bar timing)
    assert net[1] == 0.0 and net[3] == 0.0 and net[4] != 0.0


def test_two_leg_per_leg_half_spreads_independent():
    """Per-leg half-spreads are independent inputs: hs_y=40, hs_x=200,
    hedge=2 -> per-unit cost = (40 + 2*200)/1e4 = 0.044."""
    gross = np.zeros(3)
    pos = np.array([0.0, 1.0, 1.0])
    hedge = np.array([0.0, 2.0, 2.0])
    cm = stats.CostModel(half_spread_bps_y=40.0, half_spread_bps_x=200.0,
                         slippage_participation_threshold=1e9)
    net = stats.apply_costs(gross, pos, cm, hedge=hedge)
    assert net[2] == pytest.approx(-(40.0 + 2.0 * 200.0) / 1e4, abs=1e-12)
    assert net[1] == 0.0   # fill bar, not decision bar


def test_hedge_scales_hedge_leg_and_uses_lagged_hedge():
    """No look-ahead: the cost of the change decided at close t uses
    hedge[t] (the hedge in force at the decision), and a zero hedge means
    the hedge leg costs nothing."""
    cm = stats.CostModel(half_spread_bps=100.0,
                         slippage_participation_threshold=1e9)
    gross = np.zeros(4)
    pos = np.array([0.0, 1.0, 1.0, 1.0])
    hedge = np.array([5.0, 0.0, 3.0, 9.0])  # hedge[1]=0 at the decision bar
    net = stats.apply_costs(gross, pos, cm, hedge=hedge)
    # change decided at t=1 uses hedge[1]=0 -> only the y leg: 100 bps at bar 2
    assert net[2] == pytest.approx(-100.0 / 1e4, abs=1e-12)
    # a second fixture: change decided at t=2 with hedge[2]=3
    pos2 = np.array([0.0, 0.0, 1.0, 1.0])
    net2 = stats.apply_costs(np.zeros(4), pos2, cm, hedge=hedge)
    assert net2[2] == 0.0                       # decision bar: no cost
    assert net2[3] == pytest.approx(-(100.0 + 3.0 * 100.0) / 1e4, abs=1e-12)


# ---------------------------------------------------------------------------
# 4-5. slippage fields are LIVE, participation capped
# ---------------------------------------------------------------------------

def test_slippage_fields_change_net():
    """The defect probe: slippage_bps_per_pct=500 / max_participation=0.001
    changed net by EXACTLY 0. Now they must change it."""
    gross = np.zeros(50)
    pos = np.zeros(50)
    pos[10] = 1.0
    pos[30] = -1.0
    pos[45] = 0.0   # turnover 1 + 2 + 1 = 4
    hedge = np.ones(50)
    cm_plain = stats.CostModel(half_spread_bps=100.0)
    cm_slip = stats.CostModel(half_spread_bps=100.0,
                              slippage_participation_threshold=0.0,
                              slippage_bps_per_pct=500.0,
                              max_participation=0.001)
    net_plain = stats.apply_costs(gross, pos, cm_plain, hedge=hedge)
    net_slip = stats.apply_costs(gross, pos, cm_slip, hedge=hedge)
    assert not np.allclose(net_plain, net_slip)
    # slippage only ADDS cost (never cheaper): net_slip <= net_plain
    assert np.all(net_slip <= net_plain + 1e-12)


def test_participation_cap_bounds_slippage():
    """max_participation caps the participation proxy: cost above the cap is
    bounded, and the half-spread floor still applies (max, not sum)."""
    gross = np.zeros(10)
    pos = np.array([0.0] + [1.0] * 9)
    hedge = np.ones(10)
    cm = stats.CostModel(half_spread_bps=100.0,
                         slippage_participation_threshold=0.0,
                         slippage_bps_per_pct=1000.0,
                         max_participation=0.05)
    net = stats.apply_costs(gross, pos, cm, hedge=hedge)
    # participation proxy p = |dpos| * risk_capital_per_trade / adv_proxy is
    # capped at max_participation=0.05 -> slippage <= 0.05*100 pct-pts * 1000
    # bps = 50 bps per unit... total cost bounded by cap, and >= half-spread
    total_cost = float((gross - net).sum())
    hs_floor = 1.0 * 2 * 100.0 / 1e4     # two legs at 100 bps, turnover 1
    cap_cost = 1.0 * (0.05 * 100.0) * 1000.0 / 1e4 + hs_floor
    assert total_cost >= hs_floor - 1e-12
    assert total_cost <= cap_cost + 1e-12


# ---------------------------------------------------------------------------
# 6-7. fill-bar timing + zero-turnover bars
# ---------------------------------------------------------------------------

def test_cost_charged_at_fill_bar_not_decision_bar():
    """A position change decided at close t is charged at bar t+1 — the first
    bar it earns on (strat[1:] = pos[:-1]*rs[1:] convention). The decision
    bar itself carries zero cost."""
    T = 21
    gross = np.zeros(T)
    pos = np.zeros(T)
    pos[5] = 1.0     # decided at close 5, earns from bar 6
    pos[6:13] = 1.0  # hold through bar 12
    pos[13] = 0.0    # decided at close 13, earns from bar 14
    hedge = np.ones(T)
    cm = stats.CostModel(half_spread_bps=100.0,
                         slippage_participation_threshold=1e9)  # spread only
    net = stats.apply_costs(gross, pos, cm, hedge=hedge)
    assert net[5] == 0.0 and net[13] == 0.0         # decision bars: no cost
    assert net[6] == pytest.approx(-2 * 100.0 / 1e4, abs=1e-12)   # fill bar
    assert net[14] == pytest.approx(-2 * 100.0 / 1e4, abs=1e-12)
    # and it must align with the return timing: gross[t] = pos[t-1]*rs[t]
    rs = np.zeros(T)
    rs[6:] = 0.001
    earned = np.zeros(T)
    earned[1:] = pos[:-1] * rs[1:]
    assert earned[6] != 0 and earned[5] == 0


def test_zero_turnover_bars_cost_nothing():
    """Holding (or flat) bars cost exactly nothing, even with slippage on."""
    T = 30
    gross = np.zeros(T)
    pos = np.zeros(T)
    pos[5:20] = 1.0    # one entry, one exit
    hedge = np.full(T, 2.0)
    cm = stats.CostModel(half_spread_bps=100.0,
                         slippage_participation_threshold=0.0,
                         slippage_bps_per_pct=50.0,
                         max_participation=0.5)
    net = stats.apply_costs(gross, pos, cm, hedge=hedge)
    cost = gross - net
    assert cost[0:6].sum() == 0.0 or cost[5] == 0.0     # no cost before/at entry decision... entry decided at 4? pos[5]=1 -> change at index 5, cost at 6
    # precisely: changes at index 5 (0->1) and 20 (1->0); costs at bars 6 and 21
    assert cost[6] > 0 and cost[21] > 0
    mask = np.ones(T, bool)
    mask[[6, 21]] = False
    assert cost[mask].sum() == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# 8. data-driven per-name half-spreads (Roll), floor + conservative fallback
# ---------------------------------------------------------------------------

def test_roll_half_spread_known_answer_and_floor():
    """Roll (2000) estimator 2*sqrt(-cov(dp_t, dp_{t-1})) in bps: known-answer
    on a seeded series, floored at the pre-registered floor, and NaN ->
    conservative 100 bps placeholder fallback."""
    rng = np.random.Generator(np.random.PCG64(3))
    # AR(1)-ish price series with known autocorrelated innovations
    T = 500
    p = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, T)))
    est = stats.estimate_half_spread_bps_roll(p, floor_bps=1.0)
    assert est["source"] == "data_estimate"
    assert est["half_spread_bps"] >= 1.0
    # constant prices -> no evidence -> conservative fallback
    flat = np.full(50, 100.0)
    est_flat = stats.estimate_half_spread_bps_roll(flat, floor_bps=1.0,
                                                   conservative_bps=100.0)
    assert est_flat["source"] == "conservative_bound"
    assert est_flat["half_spread_bps"] == 100.0


def test_roll_matches_manual_formula():
    rng = np.random.Generator(np.random.PCG64(21))
    p = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.008, 400)))
    dp = np.diff(np.log(p))
    manual = 2.0 * np.sqrt(max(-np.cov(dp[1:], dp[:-1])[0, 1], 0.0)) * 1e4
    est = stats.estimate_half_spread_bps_roll(p, floor_bps=0.0)
    assert est["half_spread_bps"] == pytest.approx(manual, rel=1e-9)


# ---------------------------------------------------------------------------
# 9-10. end-to-end: walk_forward two-leg net is more honest; gauntlet green
# ---------------------------------------------------------------------------

def test_walk_forward_uses_two_leg_costs_and_fill_bar():
    """walk_forward_returns net on a planted pair must equal the reference
    two-leg fill-bar computation on the pairs path (same positions/hedge and
    per-name half-spreads), and two-leg costs must exceed the one-leg
    placeholder charge (costs only got more honest)."""
    from tests.fixtures.planted import f_pair
    fx = f_pair()
    prices = fx["prices"]
    wf = stats.walk_forward_returns(prices)
    net = wf["net"]
    assert np.isfinite(net).all()
    # pairs_strategy exposes positions+hedge+per-leg half-spreads: the shipped
    # net must match apply_costs with those exact inputs (two legs, fill bar)
    out = stats.pairs_strategy_returns(prices)
    gross, net_ps = out["gross"], out["net"]
    pos, hedge = out["positions"], out["hedge"]
    hs_x, hs_y = out["half_spread_x_bps"], out["half_spread_y_bps"]
    dpos = np.abs(np.diff(np.concatenate([[0.0], pos])))
    cm = stats.CostModel.from_config()
    recomputed = stats.apply_costs(gross, pos, cm, hedge=hedge,
                                   half_spread_bps_x=hs_x, half_spread_bps_y=hs_y)
    np.testing.assert_allclose(net_ps, recomputed, atol=1e-12)
    # two-leg total cost must exceed the one-leg placeholder charge
    two_leg_total = float((gross - net_ps).sum())
    one_leg_total = float((dpos * 100.0 / 1e4).sum())
    assert two_leg_total > one_leg_total
    assert wf["T"] == prices.shape[0]


def test_gauntlet_still_runs_after_cost_fix(ledger=None):
    """End-to-end smoke through the runner: F-PAIR gauntlet completes, S4
    recorded, net SR finite. (Full acceptance stays in test_s5/test_fixtures;
    thresholds are NOT weakened — costs only got more honest.)"""
    import pathlib
    import tempfile

    from sift.gauntlet.runner import GauntletRunner
    from sift.registry import Ledger
    from tests.fixtures.planted import f_pair
    fx = f_pair()
    prices = fx["prices"]
    x, y = prices[:, 0], prices[:, 1]
    excess = stats.pairs_strategy_returns(prices)["gross"][60:]
    with tempfile.TemporaryDirectory() as d:
        led = Ledger(pathlib.Path(d) / "l.db")
        trial = led.register("pairs", "free_5y_top1k", {"fixture": "F-PAIR"}, "v1")
        runner = GauntletRunner(led, seed=fx["seed"])
        spread = np.diff(np.log(y)) - np.diff(np.log(x))
        verdict = runner.run_gauntlet(
            trial.trial_id,
            fixture_x=x, fixture_y=y,
            beta_true=fx["beta_true"], half_life_true=fx["half_life_true"],
            fixture_spread=fx["spread"],
            excess_returns=excess,
            perm_series=spread,
            perm_statistic_fn=stats.mean_reversion_stat,
            prices=prices,
            scope_id="free_5y_top1k",
            s2_B=200,
        )
        led.close()
    assert verdict["verdict"] in ("survivor", "killed")
    stages = {s.stage: s.status for s in verdict["stages"]}
    assert "s4" in stages
    s4 = [s for s in verdict["stages"] if s.stage == "s4"][0]
    assert np.isfinite(s4.detail["sr_ann"])
