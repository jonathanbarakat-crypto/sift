"""v4.2 scored-S4 gauntlet wiring — tests (spec v4.2 §2-§3, issue #165).

Covers:
- run_s4 records a SCORED stage: detail carries net SR, t_net, MinTRL, T and
  the verdict is never "killed" at S4;
- an S0-S2 passer that fails the S4 bar (strong score, t_net below tau_cal)
  is verdict=survivor with survivor_class on-watch-amber-s4-evidence and
  mints a BRONZE card (S4-fail-with-strong-score mints bronze);
- S3-bar miss after S4 pass -> on-watch-amber-s3-evidence, bronze card;
- S5 pass -> validated-green; S5 fail after S4+S3 -> on-watch-amber (mints);
- `killed` only for S0-S2 failures;
- tier boundaries: bronze/silver/gold/platinum from the full evidence vector;
- decay upgrade bars extended to platinum (gold -> platinum);
- flag off -> no card minted (fail closed).
"""

import numpy as np
import pytest

from sift import flags
from sift.cards import CardRegistry, tier_from_stats
from sift.cards.decay import (
    GOLD_DSR,
    GOLD_T_NET,
    GOLD_P_PERM,
    PLATINUM_T_NET,
    SILVER_DSR,
    SILVER_T_NET,
    WindowStats,
    meets_tier_bar,
)
from sift.cards.minting import CardMinter
from sift.gauntlet import stats
from sift.gauntlet import v42
from sift.gauntlet.runner import GauntletRunner
from sift.registry import Ledger
from tests.fixtures.planted import f_regime

SECRET = b"test-hmac-secret"


@pytest.fixture(autouse=True)
def _flags_reset():
    flags.reset()
    yield
    flags.reset()


@pytest.fixture()
def ledger(tmp_path):
    l = Ledger(tmp_path / "ledger.db")
    yield l
    l.close()


@pytest.fixture()
def cards(tmp_path):
    c = CardRegistry(tmp_path / "cards.db", secret=SECRET)
    yield c
    c.close()


def _spread_returns(prices):
    rx = np.diff(np.log(prices[:, 0]))
    ry = np.diff(np.log(prices[:, 1]))
    return ry - rx


def _pair_prices(T, seed, phi=0.7, sig_in=4.0, sig_x=0.01):
    """Log-space AR(1) cointegrated pair (the pull/31 F-PAIR-FULL shape)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    lx = np.cumsum(rng.normal(0, sig_x, T))
    e = rng.normal(0, 1.0, T)
    s = np.zeros(T)
    for t in range(1, T):
        s[t] = phi * s[t - 1] + e[t] * sig_in
    px = 100.0 * np.exp(lx)
    py = px * np.exp(s / 100.0)
    return np.column_stack([px, py]), s, phi


def _run(ledger, cards, *, prices_scope, prices_full=None, handle="alice",
         seed=7, s2_B=200, scope_id="free_5y_top1k", n_null=0,
         half_life_true=1.9, with_minter=True):
    for i in range(n_null):
        ledger.register("null", scope_id, {"i": i, "h": handle, "s": seed}, "v1")
    x, y = prices_scope[:, 0], prices_scope[:, 1]
    sr = _spread_returns(prices_scope)
    excess = stats.pairs_strategy_returns(prices_scope)["gross"][60:]
    trial = ledger.register("pairs", scope_id,
                            {"cfg": f"v42-{seed}-{handle}-{n_null}"}, "v1",
                            discoverer_handle=handle)
    beta_true = float(np.dot(x - x.mean(), y - y.mean())
                      / np.dot(x - x.mean(), x - x.mean()))
    runner = GauntletRunner(
        ledger, seed=seed,
        card_minter=CardMinter(ledger, cards) if with_minter else None)
    kw = dict(
        fixture_x=x, fixture_y=y,
        beta_true=beta_true, half_life_true=half_life_true,
        fixture_spread=sr.cumsum(),
        excess_returns=excess,
        perm_series=sr,
        perm_statistic_fn=stats.mean_reversion_stat,
        prices=prices_scope,
        scope_id=scope_id,
        s2_B=s2_B,
    )
    if prices_full is not None:
        kw["full_prices"] = prices_full
        kw["full_spread_returns"] = _spread_returns(prices_full)
    verdict = runner.run_gauntlet(trial.trial_id, **kw)
    return verdict, trial


# ---------------------------------------------------------------------------
# S4 is scored, not a kill
# ---------------------------------------------------------------------------

def test_s4_detail_always_carries_score(ledger):
    """run_s4 records net SR, t_net, MinTRL, T in the stage detail whether or
    not the bar is met — a scored stage, not a binary kill."""
    runner = GauntletRunner(ledger, seed=1)
    t = ledger.register("pairs", "free_5y_top1k", {"cfg": "s4detail"}, "v1")
    rng = np.random.Generator(np.random.PCG64(3))
    net = rng.normal(0.0, 0.01, 300)     # noise: S4 bar not met
    r = runner.run_s4(t.trial_id, net, "free_5y_top1k", p_perm=0.5)
    assert r.status == "fail"                       # the bar is still recorded fail
    for k in ("sr_ann", "t_net", "min_trl", "T", "skew", "kurt", "tau_cal",
              "s4_bar_met"):
        assert k in r.detail, f"S4 detail missing {k}"
    assert r.detail["T"] == 300
    assert np.isfinite(r.detail["sr_ann"])
    assert np.isfinite(r.detail["t_net"])
    assert np.isfinite(r.detail["min_trl"])
    assert r.detail["s4_bar_met"] is False
    runner.record(t.trial_id, r)
    row = ledger.stage(t.trial_id, "s4")
    assert row["status"] == "fail"
    assert row["statistic"] == pytest.approx(r.detail["sr_ann"])


def test_s4_fail_with_strong_score_mints_bronze_card(ledger, cards):
    """THE load-bearing v4.2 test: a candidate that passes S0-S3 but fails the
    S4 bar (net SR 1.18 > 0, strong score, but t_net 2.63 < tau_cal 3.5) is
    NOT killed — it is a survivor at the s4-evidence class and mints a
    BRONZE card."""
    prices, s, phi = _pair_prices(1260, seed=11, phi=0.7, sig_in=4.0)
    verdict, trial = _run(ledger, cards, prices_scope=prices, seed=11)
    assert verdict["verdict"] == "survivor", verdict
    assert verdict["survivor_class"] == v42.SURVIVOR_CLASS_S4_EVIDENCE
    assert verdict["killed_at"] is None
    # S4 row recorded fail, and the verdict carries the full S4 score
    s4row = ledger.stage(trial.trial_id, "s4")
    assert s4row["status"] == "fail"
    sc = verdict["s4_score"]
    assert sc["sr_ann"] > 0                      # strong score...
    assert sc["s4_bar_met"] is False             # ...but below the tau_cal bar
    assert sc["tau_cal"] == 3.5
    # a bronze card was minted (read-back)
    card_id = verdict["card_id"]
    assert card_id is not None and card_id.startswith("card-")
    card = cards.get_card(card_id)
    assert card.tier == "bronze"
    assert card.kind == "regular"
    assert card.owner_handle == "alice"
    assert card.survivor_class == v42.SURVIVOR_CLASS_S4_EVIDENCE


def test_s3_bar_miss_after_s4_pass_mints_bronze(ledger, cards):
    """S0-S2+S4 pass but S3 DSR < 0.95 (big null stratum deflates DSR) ->
    on-watch-amber-s3-evidence, bronze card (not killed)."""
    prices_full, s, phi = _pair_prices(4032, seed=13, phi=0.7, sig_in=8.0)
    prices_scope = prices_full[-1260:]
    verdict, trial = _run(ledger, cards, prices_scope=prices_scope,
                          prices_full=prices_full, seed=13, n_null=800)
    assert verdict["verdict"] == "survivor"
    assert verdict["survivor_class"] == v42.SURVIVOR_CLASS_S3_EVIDENCE
    assert ledger.stage(trial.trial_id, "s4")["status"] == "pass"
    assert ledger.stage(trial.trial_id, "s3")["status"] == "fail"
    assert verdict["card_id"] is not None
    assert cards.get_card(verdict["card_id"]).tier == "bronze"


def test_s5_pass_still_validated_green(ledger, cards):
    """Unchanged: a persistent full-window edge passes S5 -> validated-green
    and mints (tier from the full evidence vector)."""
    prices_full, s, phi = _pair_prices(4032, seed=13, phi=0.7, sig_in=6.0)
    prices_scope = prices_full[-1260:]
    verdict, trial = _run(ledger, cards, prices_scope=prices_scope,
                          prices_full=prices_full, seed=13)
    assert verdict["survivor_class"] == "validated-green"
    assert ledger.stage(trial.trial_id, "s5")["status"] == "pass"
    assert verdict["card_id"] is not None
    card = cards.get_card(verdict["card_id"])
    assert card.tier in ("bronze", "silver", "gold", "platinum")
    assert card.survivor_class == "validated-green"


def test_s5_fail_after_s4_s3_on_watch_amber(ledger, cards):
    """F-REGIME: S0-S4 pass at the scope but S5 fails at full depth ->
    on-watch-amber (unchanged v3 §8.4), still mints a card at bronze+ (v4.2:
    S0-S2 pass)."""
    fx = f_regime()
    verdict, trial = _run(ledger, cards, prices_scope=fx["prices_scope"],
                          prices_full=fx["prices_full"], seed=fx["seed"],
                          handle="bob",
                          half_life_true=fx["half_life_true"])
    assert verdict["verdict"] == "survivor"
    assert verdict["survivor_class"] == "on-watch-amber"
    assert ledger.stage(trial.trial_id, "s5")["status"] == "fail"
    assert verdict["card_id"] is not None
    card = cards.get_card(verdict["card_id"])
    # tier from the full evidence vector: F-REGIME carries DSR ~0.999, S4 bar
    # met, p_perm ~0.005 -> gold (S5 failed, so not platinum)
    assert card.tier in ("bronze", "silver", "gold")
    assert card.owner_handle == "bob"


def test_killed_only_for_s0_s2_failures(ledger, cards):
    """A null candidate killed at S2 keeps verdict=killed and mints nothing."""
    from tests.fixtures.planted import f_null_spread_returns
    rs = f_null_spread_returns()
    prices = np.column_stack([100.0 + np.cumsum(rs), 100.0 + np.cumsum(rs)])
    trial = ledger.register("null", "free_5y_top1k", {"fixture": "F-NULL"}, "v1",
                            discoverer_handle="carol")
    runner = GauntletRunner(ledger, seed=123, card_minter=CardMinter(ledger, cards))
    verdict = runner.run_gauntlet(
        trial.trial_id,
        fixture_x=prices[:, 0], fixture_y=prices[:, 1],
        beta_true=1.0, half_life_true=1.0,
        excess_returns=rs - rs.mean(),
        perm_series=rs,
        perm_statistic_fn=stats.mean_reversion_stat,
        prices=prices,
        scope_id="free_5y_top1k",
        s2_B=100,
    )
    assert verdict["verdict"] == "killed"
    assert verdict["killed_at"] in ("s0", "s1", "s2")
    assert verdict["card_id"] is None
    assert cards.cards_for_owner("carol") == []


def test_flag_off_s4_fail_passer_mints_nothing(ledger, cards):
    """Kill-switch fail-closed: with cards_v4_1 off, an S0-S2 passer that
    fails S4 is still a survivor (gauntlet unaffected) but mints NO card."""
    flags.set_flag("cards_v4_1", False)
    prices, s, phi = _pair_prices(1260, seed=11, phi=0.7, sig_in=4.0)
    verdict, trial = _run(ledger, cards, prices_scope=prices, seed=11)
    assert verdict["verdict"] == "survivor"
    assert verdict["survivor_class"] == v42.SURVIVOR_CLASS_S4_EVIDENCE
    assert verdict["card_id"] is None
    assert cards.all_cards() == []


def test_research_scope_stays_amber(ledger, cards):
    """A research_ scope S0-S2 passer (S4 bar not met on the research window)
    is on-watch-amber — amber on research data, never a production class."""
    prices, s, phi = _pair_prices(1260, seed=11, phi=0.7, sig_in=4.0)
    verdict, trial = _run(ledger, cards, prices_scope=prices,
                          handle="rex", seed=11, scope_id="research_5y_top1k")
    assert verdict["verdict"] == "survivor"
    assert verdict["survivor_class"] == v42.SURVIVOR_CLASS_S4_EVIDENCE
    assert verdict["card_id"] is not None
    card = cards.get_card(verdict["card_id"])
    assert card.scope_id == "research_5y_top1k"
    assert card.tier == "bronze"


# ---------------------------------------------------------------------------
# tier boundaries (full evidence vector)
# ---------------------------------------------------------------------------

def test_tier_boundaries_v42():
    # platinum: S4 bar + S5 pass + DSR >= 0.99 + full-depth |t_net| >= 4.0 + p <= 0.01
    assert tier_from_stats(dsr=0.995, t_net_full=4.5, p_perm=0.005,
                           s4_pass=True, s5_pass=True) == "platinum"
    # gold: S4 bar met + DSR >= 0.99 + p_perm <= 0.01, S5 not passed
    assert tier_from_stats(dsr=0.995, t_net_full=3.2, p_perm=0.005,
                           s4_pass=True, s5_pass=False) == "gold"
    # S4 bar met but DSR below 0.99 -> silver if the S3/S5 line is met
    assert tier_from_stats(dsr=0.96, t_net_full=3.2, p_perm=0.04,
                           s4_pass=True, s5_pass=False) == "silver"
    # silver: S3 admission line (DSR >= 0.95, full-depth |t| >= 3.0)
    assert tier_from_stats(dsr=0.96, t_net_full=3.0, p_perm=0.04,
                           s4_pass=False, s5_pass=False) == "silver"
    # bronze: S0-S2 pass only
    assert tier_from_stats(dsr=0.90, t_net_full=2.0, p_perm=0.5,
                           s4_pass=False, s5_pass=False) == "bronze"
    # missing stats never inflate
    assert tier_from_stats(dsr=None, t_net_full=9.0, p_perm=0.001,
                           s4_pass=True, s5_pass=True) == "bronze"
    # boundary exactness
    assert tier_from_stats(dsr=0.99, t_net_full=4.0, p_perm=0.01,
                           s4_pass=True, s5_pass=True) == "platinum"
    assert tier_from_stats(dsr=0.989, t_net_full=4.0, p_perm=0.01,
                           s4_pass=True, s5_pass=False) == "silver"
    assert tier_from_stats(dsr=0.99, t_net_full=4.0, p_perm=0.011,
                           s4_pass=True, s5_pass=False) == "silver"
    # S4 bar NOT met caps gold/platinum even with stellar S3/S5 stats
    assert tier_from_stats(dsr=0.995, t_net_full=4.5, p_perm=0.005,
                           s4_pass=False, s5_pass=True) == "silver"
    # legacy callers (no s4/s5 flags) never see gold/platinum
    assert tier_from_stats(dsr=0.995, t_net_full=4.5, p_perm=0.005) == "silver"


def test_s4_passed_helper():
    d = {"sr_ann": 1.0, "t_net": 4.0, "min_trl": 100, "T": 1260}
    assert v42.s4_passed(d, "free_5y_top1k") is True       # tau_cal=3.5
    assert v42.s4_passed(d, "pro_full") is True             # tau=3.0
    d2 = {"sr_ann": 1.0, "t_net": 3.2, "min_trl": 100, "T": 1260}
    assert v42.s4_passed(d2, "free_5y_top1k") is False      # 3.2 < 3.5
    assert v42.s4_passed(d2, "pro_full") is True            # 3.2 >= 3.0
    d3 = {"sr_ann": 1.0, "t_net": 4.0, "min_trl": 5000, "T": 1260}
    assert v42.s4_passed(d3, "free_5y_top1k") is False      # MinTRL > T
    d4 = {"sr_ann": -0.1, "t_net": 4.0, "min_trl": 10, "T": 1260}
    assert v42.s4_passed(d4, "free_5y_top1k") is False      # SR not > 0
    assert v42.s4_passed({}, "free_5y_top1k") is False      # missing -> no


def test_survivor_class_for_mapping():
    assert v42.survivor_class_for(s4_pass=True, s3_pass=True, s5_pass=True) == "validated-green"
    assert v42.survivor_class_for(s4_pass=True, s3_pass=True, s5_pass=False) == "on-watch-amber"
    assert v42.survivor_class_for(s4_pass=True, s3_pass=True, s5_pass=None) == "on-watch-amber"
    assert v42.survivor_class_for(s4_pass=False, s3_pass=False, s5_pass=None) == v42.SURVIVOR_CLASS_S4_EVIDENCE
    assert v42.survivor_class_for(s4_pass=True, s3_pass=False, s5_pass=None) == v42.SURVIVOR_CLASS_S3_EVIDENCE


# ---------------------------------------------------------------------------
# decay bars extended to platinum (gold -> platinum upgrade path)
# ---------------------------------------------------------------------------

def test_meets_tier_bar_platinum():
    assert meets_tier_bar(WindowStats(0, 252, 0.99, 4.0, 0.01), "platinum")
    assert not meets_tier_bar(WindowStats(0, 252, 0.99, 3.99, 0.01), "platinum")
    assert not meets_tier_bar(WindowStats(0, 252, 0.989, 4.0, 0.01), "platinum")
    assert not meets_tier_bar(WindowStats(0, 252, 0.99, 4.0, 0.011), "platinum")
    # gold bar unchanged (same numbers as pull/31)
    assert meets_tier_bar(WindowStats(0, 252, 0.99, 4.0, 0.01), "gold")
    assert not meets_tier_bar(WindowStats(0, 252, 0.95, 3.0, 0.5), "gold")
    assert meets_tier_bar(WindowStats(0, 252, 0.95, 3.0, 0.5), "silver")


def test_platinum_constants_match_pull31_gold_bar():
    # the platinum bar is exactly pull/31's gold bar: DSR 0.99, |t| 4.0, p 0.01
    assert GOLD_DSR == 0.99 and PLATINUM_T_NET == 4.0 and GOLD_P_PERM == 0.01
    assert SILVER_DSR == 0.95 and SILVER_T_NET == 3.0
    assert GOLD_T_NET == 4.0
