"""v4.1 mechanics 5 — anchored run mode.

Fixtures: an anchored run with one locked anchor (a decided survivor trial)
+ random sampling of partner slots; the §12.11 scope record carries
anchor + subspace + N_eff + seed; the kill report shows PER-STAGE kills.
Flag off -> AnchoredRunError. Pending anchor refused (anchors must be
pre-registered/decided — no post-hoc anchoring).
"""

import numpy as np
import pytest

from sift import flags
from sift.cards import CardRegistry
from sift.cards.anchored import (
    AnchoredRunError, plan_anchored_run, run_anchored, sample_partners,
)
from sift.cards.minting import CardMinter
from sift.gauntlet import stats
from sift.gauntlet.runner import GauntletRunner
from sift.registry import Ledger
from tests.cards.test_minting import SECRET, _run_validated_green, _spread_returns
from tests.fixtures.planted import T_5Y, f_null_spread_returns


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


def _anchor_survivor(ledger, cards):
    """A decided (survivor) trial to lock as the anchor slot."""
    verdict, trial = _run_validated_green(ledger, cards, handle="alice")
    return trial.trial_id


def _null_sampler(ledger):
    """Sampler producing F-NULL-like candidates (killed at some stage)."""
    def sampler(i, rng):
        rs = f_null_spread_returns(T=T_5Y, seed=5000 + i)
        prices = np.column_stack([100.0 + np.cumsum(rs), 100.0 + np.cumsum(rs)])
        return {
            "family": "null",
            "params": {"fixture": "F-NULL", "i": i},
            "gauntlet": dict(
                fixture_x=prices[:, 0], fixture_y=prices[:, 1],
                beta_true=1.0, half_life_true=1.0,
                excess_returns=rs - rs.mean(),
                perm_series=rs,
                perm_statistic_fn=stats.mean_reversion_stat,
                prices=prices, scope_id="free_5y_top1k", s2_B=100,
            ),
        }
    return sampler


def test_scope_record_carries_anchor_subspace_n_eff(ledger, cards):
    anchor = _anchor_survivor(ledger, cards)
    scope = plan_anchored_run(
        ledger, anchors=[anchor],
        subspace={"partners": "top1k_pairs", "families": ["pairs"], "params": {"formation": [60]}},
        n_eff=25, scope_id="free_5y_top1k", seed=424242, handle="alice",
    )
    assert scope.anchors == [anchor]
    assert scope.n_eff == 25
    assert scope.seed == 424242
    assert scope.scope_id == "free_5y_top1k"
    blob = scope.to_json()
    assert anchor in blob and "top1k_pairs" in blob and '"n_eff":25' in blob


def test_pending_anchor_refused(ledger, cards):
    t = ledger.register("pairs", "free_5y_top1k", {"x": 1}, "v")  # still pending
    with pytest.raises(AnchoredRunError):
        plan_anchored_run(ledger, anchors=[t.trial_id], subspace={}, n_eff=5,
                          scope_id="free_5y_top1k", seed=1, handle="alice")


def test_kill_report_shows_per_stage_kills(ledger, cards):
    anchor = _anchor_survivor(ledger, cards)
    scope = plan_anchored_run(
        ledger, anchors=[anchor],
        subspace={"partners": "null_walks"}, n_eff=12,
        scope_id="free_5y_top1k", seed=99, handle="alice",
    )
    runner = GauntletRunner(ledger, seed=99, card_minter=CardMinter(ledger, cards))
    report = run_anchored(runner, ledger, scope, _null_sampler(ledger), "alice")
    d = report.to_dict()
    assert d["run_id"] == scope.run_id
    assert d["anchor_verdicts"] == {anchor: "survivor"}
    # every sampled null candidate was killed at SOME stage; per-stage counts sum to n_eff
    assert sum(d["per_stage_kills"].values()) == scope.n_eff
    assert set(d["per_stage_kills"]).issubset({"s0", "s1", "s2", "s3", "s4"})
    assert d["n_survivors"] == 0


def test_anchored_survivor_candidate_mints_card(ledger, cards):
    """A sampled candidate that survives the gauntlet is counted as a
    survivor in the kill report; without an S5 full window it is
    on-watch-amber and mints nothing (mechanics 1: only validated-green
    mints). With the full window it becomes validated-green and mints
    exactly one card."""
    anchor = _anchor_survivor(ledger, cards)
    scope = plan_anchored_run(
        ledger, anchors=[anchor], subspace={"partners": "pair_full"}, n_eff=1,
        scope_id="free_5y_top1k", seed=7, handle="alice",
    )
    from tests.fixtures.planted import T_FULL
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
    beta_true = float(np.dot(x - x.mean(), y - y.mean()) / np.dot(x - x.mean(), x - x.mean()))

    def sampler(i, rng):
        return {
            "family": "pairs",
            "params": {"fixture": "F-PAIR-FULL-ANCHORED", "i": i},
            "gauntlet": dict(
                fixture_x=x, fixture_y=y,
                beta_true=beta_true, half_life_true=-np.log(2.0) / np.log(phi),
                fixture_spread=s[-T_5Y:],
                excess_returns=excess,
                perm_series=_spread_returns(prices_scope),
                perm_statistic_fn=stats.mean_reversion_stat,
                prices=prices_scope, scope_id="free_5y_top1k", s2_B=200,
                full_prices=prices_full,
                full_spread_returns=_spread_returns(prices_full),
            ),
        }

    runner = GauntletRunner(ledger, seed=7, card_minter=CardMinter(ledger, cards))
    report = run_anchored(runner, ledger, scope, sampler, "alice")
    assert report.to_dict()["n_survivors"] == 1
    # validated-green sampled candidate -> exactly one NEW card minted
    owned = cards.cards_for_owner("alice")
    assert len(owned) == 2
    assert any(c.trial_id != anchor for c in owned)


def test_sample_partners_without_replacement():
    rng = np.random.Generator(np.random.PCG64(3))
    uni = list(range(100))
    picked = sample_partners(rng, uni, 10)
    assert len(picked) == 10 and len(set(picked)) == 10


def test_flag_off_refuses(ledger, cards):
    anchor = _anchor_survivor(ledger, cards)
    flags.disable_for_user("cards_v4_1", "alice")
    with pytest.raises(AnchoredRunError):
        plan_anchored_run(ledger, anchors=[anchor], subspace={}, n_eff=5,
                          scope_id="free_5y_top1k", seed=1, handle="alice")
