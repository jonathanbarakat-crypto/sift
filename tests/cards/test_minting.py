"""v4.1 mechanics 1 — card minting on gauntlet survivors, behind cards_v4_1.

Fixtures:
- F-PAIR-FULL (persistent full-window AR(1) edge) -> validated-green ->
  mints EXACTLY ONE regular card, id = HMAC commitment, read-back verified.
- duplicate spec (re-mint of the same canonical spec) -> ghost, FIFO no=1.
- F-REGIME (edge only in years 0-7) -> on-watch-amber -> mints NOTHING.
- F-NULL / killed trial -> mints NOTHING.
- flag OFF -> validated-green mints nothing (kill-switch), no rows written.
- tier from pre-registered stats only; decay() down-only primitive; cards
  non-transferable (assert_owned raises on a foreign handle).
"""

import numpy as np
import pytest

from sift import flags
from sift.cards import CardError, CardRegistry, decay_tier, spec_hash_of, tier_from_stats
from sift.cards.minting import CardMinter
from sift.gauntlet import stats
from sift.gauntlet.runner import GauntletRunner
from sift.registry import Ledger
from tests.fixtures.planted import T_5Y, T_FULL, f_null_spread_returns, f_regime

SECRET = b"test-hmac-secret"


@pytest.fixture(autouse=True)
def _flags_reset():
    flags.reset()
    yield
    flags.reset()


@pytest.fixture()
def ledger(tmp_path):
    led = Ledger(tmp_path / "ledger.db")
    yield led
    led.close()


@pytest.fixture()
def cards(tmp_path):
    c = CardRegistry(tmp_path / "cards.db", secret=SECRET)
    yield c
    c.close()


def _spread_returns(prices):
    rx = np.diff(np.log(prices[:, 0]))
    ry = np.diff(np.log(prices[:, 1]))
    return ry - rx


def _run_validated_green(ledger, cards, handle="alice", s2_B=200, fixture="F-PAIR-FULL"):
    """Persistent full-window AR(1) edge -> validated-green (S5 pass)."""
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
    trial = ledger.register("pairs", "free_5y_top1k", {"fixture": fixture}, "v1",
                            discoverer_handle=handle)
    runner = GauntletRunner(ledger, seed=7, card_minter=CardMinter(ledger, cards))
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
        s2_B=s2_B,
        full_prices=prices_full,
        full_spread_returns=_spread_returns(prices_full),
    )
    return verdict, trial


def _run_amber(ledger, cards, s2_B=200):
    """F-REGIME -> on-watch-amber (S5 kill at full depth)."""
    fx = f_regime()
    prices_scope, prices_full = fx["prices_scope"], fx["prices_full"]
    x, y = prices_scope[:, 0], prices_scope[:, 1]
    excess = stats.pairs_strategy_returns(prices_scope)["gross"][60:]
    trial = ledger.register("pairs", "free_5y_top1k", {"fixture": "F-REGIME"}, "v1",
                            discoverer_handle="bob")
    runner = GauntletRunner(ledger, seed=fx["seed"], card_minter=CardMinter(ledger, cards))
    verdict = runner.run_gauntlet(
        trial.trial_id,
        fixture_x=x, fixture_y=y,
        beta_true=fx["beta_true"], half_life_true=fx["half_life_true"],
        fixture_spread=fx["spread"][max(0, fx["t_edge"] - T_5Y): fx["t_edge"]],
        excess_returns=excess,
        perm_series=_spread_returns(prices_scope),
        perm_statistic_fn=stats.mean_reversion_stat,
        prices=prices_scope,
        scope_id="free_5y_top1k",
        s2_B=s2_B,
        full_prices=prices_full,
        full_spread_returns=_spread_returns(prices_full),
    )
    return verdict, trial


# -- minting on validated-green survivors -------------------------------------

def test_validated_green_mints_exactly_one_card(ledger, cards):
    verdict, trial = _run_validated_green(ledger, cards)
    assert verdict["survivor_class"] == "validated-green"
    card_id = verdict["card_id"]
    assert card_id is not None and card_id.startswith("card-")
    # read-back: card GET after mint (hard rule)
    card = cards.get_card(card_id)
    assert card.kind == "regular"
    assert card.trial_id == trial.trial_id
    assert card.owner_handle == "alice"
    assert card.tier in ("bronze", "silver", "gold", "platinum")
    # exactly one card in the registry
    assert len(cards.cards_for_owner("alice")) == 1


def test_card_id_is_hmac_commitment_of_canonical_spec(ledger, cards):
    verdict, trial = _run_validated_green(ledger, cards)
    trial_row = ledger.get(trial.trial_id)
    spec = {
        "family": trial_row.family, "scope_id": trial_row.scope_id,
        "config_hash": trial_row.config_hash,
        "data_vintage_hash": trial_row.data_vintage_hash,
        "trial_id": trial_row.trial_id, "survivor_class": "validated-green",
    }
    expected = cards.card_id_for(spec)
    assert verdict["card_id"] == expected
    # different spec -> different id (uniqueness)
    spec2 = dict(spec, trial_id="trial-999999")
    assert cards.card_id_for(spec2) != expected
    assert spec_hash_of(spec) != spec_hash_of(spec2)


def test_duplicate_spec_mints_ghost_fifo(ledger, cards):
    v1, t1 = _run_validated_green(ledger, cards, fixture="F-PAIR-FULL")
    # mint the SAME canonical spec again directly (skip-list re-survive path)
    CardMinter(ledger, cards)
    trial = ledger.get(t1.trial_id)
    s2 = ledger.stage(t1.trial_id, "s2")
    s3 = ledger.stage(t1.trial_id, "s3")
    s5 = ledger.stage(t1.trial_id, "s5")
    g1 = cards.mint(
        trial_id=trial.trial_id, family=trial.family, scope_id=trial.scope_id,
        config_hash=trial.config_hash, data_vintage_hash=trial.data_vintage_hash,
        survivor_class="validated-green", owner_handle="alice",
        dsr=s3["dsr"], t_net_full=s5["statistic"], p_perm=s2["p_value"],
    )
    g2 = cards.mint(
        trial_id=trial.trial_id, family=trial.family, scope_id=trial.scope_id,
        config_hash=trial.config_hash, data_vintage_hash=trial.data_vintage_hash,
        survivor_class="validated-green", owner_handle="alice",
        dsr=s3["dsr"], t_net_full=s5["statistic"], p_perm=s2["p_value"],
    )
    assert g1.kind == "ghost" and g1.ghost_no == 1
    assert g2.kind == "ghost" and g2.ghost_no == 2
    assert g1.card_id != g2.card_id != v1["card_id"]
    # FIFO order per spec
    ghosts = cards.ghosts_for_spec(g1.spec_hash)
    assert [g.ghost_no for g in ghosts] == [1, 2]
    # the regular card is untouched
    assert cards.get_card(v1["card_id"]).kind == "regular"


# -- F-REGIME / killed mint nothing --------------------------------------------

def test_f_regime_amber_mints_a_bronze_card(ledger, cards):
    """v4.2 §2: F-REGIME (S5 fail at full depth) is on-watch-amber but it
    PASSED S0-S2 — it mints a card at the tier its evidence supports."""
    verdict, _ = _run_amber(ledger, cards)
    assert verdict["survivor_class"] == "on-watch-amber"
    assert verdict["card_id"] is not None
    card = cards.get_card(verdict["card_id"])
    assert card.owner_handle == "bob"
    assert card.tier in ("bronze", "silver", "gold")   # not platinum: S5 failed


def test_killed_trial_mints_nothing(ledger, cards):
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
    assert verdict["card_id"] is None
    assert cards.cards_for_owner("carol") == []


def test_flag_off_mints_nothing(ledger, cards):
    flags.set_flag("cards_v4_1", False)
    verdict, _ = _run_validated_green(ledger, cards)
    assert verdict["survivor_class"] == "validated-green"   # gauntlet unaffected
    assert verdict["card_id"] is None
    assert cards.cards_for_owner("alice") == []


def test_flag_off_per_user(ledger, cards):
    flags.disable_for_user("cards_v4_1", "mallory")
    verdict, _ = _run_validated_green(ledger, cards, handle="mallory")
    assert verdict["survivor_class"] == "validated-green"
    assert verdict["card_id"] is None


# -- tier + decay + non-transferability -----------------------------------------

def test_tier_from_preregistered_stats():
    # v4.2 §3 (#170): tier from RAW, N-independent stats only — DSR is NOT an
    # input. platinum = S4 bar + S5 pass + full-depth |t| >= 4.0 + p <= 0.01
    # (the pull/31 gold bar minus DSR, plus S5 pass).
    assert tier_from_stats(dsr=0.995, t_net_full=4.5, p_perm=0.005,
                           s4_pass=True, s5_pass=True) == "platinum"
    # DSR changes with N but the tier does NOT (N-independence)
    assert tier_from_stats(dsr=0.10, t_net_full=4.5, p_perm=0.005,
                           s4_pass=True, s5_pass=True) == "platinum"
    assert tier_from_stats(dsr=None, t_net_full=4.5, p_perm=0.005,
                           s4_pass=True, s5_pass=True) == "platinum"
    # gold: S4 bar met + p_perm <= 0.01 + full-depth |t| >= 3.0 (S5 not passed)
    assert tier_from_stats(dsr=0.995, t_net_full=3.2, p_perm=0.005,
                           s4_pass=True, s5_pass=False) == "gold"
    assert tier_from_stats(dsr=0.96, t_net_full=3.2, p_perm=0.04,
                           s4_pass=True, s5_pass=False) == "silver"
    assert tier_from_stats(dsr=0.90, t_net_full=3.2, p_perm=0.04,
                           s4_pass=False, s5_pass=False) == "silver"
    # missing stats never inflate
    assert tier_from_stats(dsr=0.995, t_net_full=None, p_perm=0.005,
                           s4_pass=True, s5_pass=True) == "bronze"
    # legacy 3-arg call (no s4/s5 flags) never reaches gold/platinum
    assert tier_from_stats(dsr=0.995, t_net_full=4.5, p_perm=0.005) == "silver"
    # p_perm missing -> bronze (never inflate)
    assert tier_from_stats(dsr=None, t_net_full=9.0, p_perm=None,
                           s4_pass=True, s5_pass=True) == "bronze"


def test_decay_downgrade_only(ledger, cards):
    verdict, _ = _run_validated_green(ledger, cards)
    card_id = verdict["card_id"]
    card = cards.get_card(card_id)
    # downgrade ok
    down = cards.decay(card_id, "bronze")
    assert down.tier == "bronze"
    assert down.decayed_from == card.tier
    # read-back after decay
    assert cards.get_card(card_id).tier == "bronze"
    # upgrade refused
    with pytest.raises(CardError):
        cards.decay(card_id, "gold")
    with pytest.raises(CardError):
        decay_tier("bronze", "silver")


def test_cards_non_transferable(ledger, cards):
    verdict, _ = _run_validated_green(ledger, cards, handle="alice")
    card_id = verdict["card_id"]
    cards.assert_owned(card_id, "alice")
    with pytest.raises(CardError):
        cards.assert_owned(card_id, "eve")   # foreign handle -> refuse


def test_mint_refuses_killed_trial(cards):
    with pytest.raises(CardError):
        cards.mint(trial_id="trial-000001", family="pairs", scope_id="free_5y_top1k",
                   config_hash="x", data_vintage_hash="v", survivor_class="none",
                   owner_handle="alice", dsr=0.99, t_net_full=3.5, p_perm=0.01)


def test_mint_accepts_v42_evidence_classes(cards):
    """v4.2: the S4/S3-evidence survivor classes mint (bronze+); killed
    (survivor_class='none') refuses."""
    for cls in ("on-watch-amber", "on-watch-amber-s4-evidence",
                "on-watch-amber-s3-evidence", "validated-green"):
        card = cards.mint(
            trial_id=f"trial-{cls}", family="pairs", scope_id="free_5y_top1k",
            config_hash=f"h-{cls}", data_vintage_hash="v",
            survivor_class=cls, owner_handle="alice",
            dsr=0.90, t_net_full=2.0, p_perm=0.5,
        )
        assert card.tier == "bronze" and card.survivor_class == cls
