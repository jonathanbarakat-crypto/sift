"""v4.2 N-independent tiers + N-sensitive health — tests (hermes-tasks #170).

Owner decision 2026-11-18:
1. Tier is N-INDEPENDENT: tier_from_stats ignores DSR (raw stats only);
   same raw stats + different N => identical tier. DSR is the MINT-TIME
   ADMISSION GATE only: DSR < 0.95 at mint-time N => max bronze even with
   strong raw stats. dsr / sr_star / n_stratum recorded on the card row as
   "admission evidence at mint-time N" context.
2. Health is deliberately N-SENSITIVE: health_ratio = live-window Sharpe /
   E[max Sharpe of N_current random strategies] — N grows => ratio drops for
   unchanged performance; unchanged N => stable. Display-only, never a tier.
3. decay.py::meets_tier_bar uses RAW bars only (no DSR); decay paths
   unchanged otherwise (#157 rules on raw stats over fully OOS windows).
4. Flag `cards_v4_1` + kill-switch behavior unchanged.
"""

import numpy as np
import pytest

from sift import flags
from sift.cards import CARD_TIERS, CardRegistry, tier_from_stats
from sift.cards.decay import (
    WindowStats,
    health_ratio,
    meets_tier_bar,
)
from sift.gauntlet import v42
from sift.gauntlet.v42 import mint_admission_gate
from sift.registry import Ledger

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


# ---------------------------------------------------------------------------
# 1. tier_from_stats is N-independent (DSR is not an input)
# ---------------------------------------------------------------------------

def test_tier_ignores_dsr_same_raw_stats_different_n():
    """Same raw stats, wildly different DSR (i.e. different stratum N):
    identical tier. DSR changes with N but the tier does not."""
    raw = dict(t_net_full=4.5, p_perm=0.005, s4_pass=True, s5_pass=True)
    tiers = {
        tier_from_stats(dsr=dsr, **raw)
        for dsr in (None, 0.10, 0.50, 0.90, 0.95, 0.99, 0.999)
    }
    assert tiers == {"platinum"}
    raw_gold = dict(t_net_full=3.2, p_perm=0.005, s4_pass=True, s5_pass=False)
    assert {tier_from_stats(dsr=d, **raw_gold)
            for d in (None, 0.10, 0.99, 0.999)} == {"gold"}
    raw_silver = dict(t_net_full=3.2, p_perm=0.04, s4_pass=True, s5_pass=False)
    assert {tier_from_stats(dsr=d, **raw_silver)
            for d in (None, 0.10, 0.99)} == {"silver"}


def test_tier_bars_raw_only():
    # platinum: s4 + s5 + |t_net_full| >= 4.0 + p_perm <= 0.01
    assert tier_from_stats(dsr=None, t_net_full=4.0, p_perm=0.01,
                           s4_pass=True, s5_pass=True) == "platinum"
    assert tier_from_stats(dsr=None, t_net_full=3.99, p_perm=0.01,
                           s4_pass=True, s5_pass=True) == "gold"
    assert tier_from_stats(dsr=None, t_net_full=4.5, p_perm=0.011,
                           s4_pass=True, s5_pass=True) == "silver"
    # gold: s4 bar + p_perm <= 0.01 + full-depth |t_net_full| >= 3.0
    assert tier_from_stats(dsr=None, t_net_full=3.0, p_perm=0.01,
                           s4_pass=True, s5_pass=False) == "gold"
    assert tier_from_stats(dsr=None, t_net_full=2.99, p_perm=0.01,
                           s4_pass=True, s5_pass=False) == "bronze"
    assert tier_from_stats(dsr=None, t_net_full=9.0, p_perm=0.02,
                           s4_pass=True, s5_pass=False) == "silver"
    # silver: |t_net_full| >= 3.0 + p_perm <= 0.05
    assert tier_from_stats(dsr=None, t_net_full=3.0, p_perm=0.05,
                           s4_pass=False, s5_pass=False) == "silver"
    assert tier_from_stats(dsr=None, t_net_full=3.0, p_perm=0.051,
                           s4_pass=False, s5_pass=False) == "bronze"
    assert tier_from_stats(dsr=None, t_net_full=2.99, p_perm=0.01,
                           s4_pass=False, s5_pass=False) == "bronze"
    # bronze: S0-S2 pass (below silver); missing stats never inflate
    assert tier_from_stats(dsr=None, t_net_full=None, p_perm=0.001,
                           s4_pass=True, s5_pass=True) == "bronze"
    assert tier_from_stats(dsr=None, t_net_full=9.0, p_perm=None,
                           s4_pass=True, s5_pass=True) == "bronze"


# ---------------------------------------------------------------------------
# 2. mint-time admission gate: DSR < 0.95 at mint-time N => max bronze
# ---------------------------------------------------------------------------

def test_mint_admission_gate():
    assert mint_admission_gate(0.95) is True
    assert mint_admission_gate(0.99) is True
    assert mint_admission_gate(0.949) is False
    assert mint_admission_gate(0.0) is False
    assert mint_admission_gate(None) is False      # fail closed


def test_mint_gate_caps_at_bronze_even_with_strong_raw_stats(cards):
    """DSR < 0.95 at mint-time N => max bronze, even with platinum-grade raw
    stats (s4+s5 pass, |t| 4.5, p 0.005)."""
    card = cards.mint(
        trial_id="trial-gate", family="pairs", scope_id="free_5y_top1k",
        config_hash="h-gate", data_vintage_hash="v",
        survivor_class="validated-green", owner_handle="alice",
        dsr=0.90, t_net_full=4.5, p_perm=0.005,   # strong raw stats, gate missed
        s4_pass=True, s5_pass=True,
    )
    assert card.tier == "bronze"
    # same raw stats WITH the gate cleared => platinum
    card2 = cards.mint(
        trial_id="trial-gate2", family="pairs", scope_id="free_5y_top1k",
        config_hash="h-gate2", data_vintage_hash="v",
        survivor_class="validated-green", owner_handle="alice",
        dsr=0.97, t_net_full=4.5, p_perm=0.005,
        s4_pass=True, s5_pass=True,
    )
    assert card2.tier == "platinum"


def test_mint_records_admission_evidence_context(cards):
    """dsr / sr_star / n_stratum are recorded on the card row as mint-time
    context (label: 'admission evidence at mint-time N')."""
    card = cards.mint(
        trial_id="trial-ctx", family="pairs", scope_id="free_5y_top1k",
        config_hash="h-ctx", data_vintage_hash="v",
        survivor_class="validated-green", owner_handle="alice",
        dsr=0.97, t_net_full=3.2, p_perm=0.04,
        s4_pass=True, s5_pass=False,
        sr_star=0.31, n_stratum=500,
    )
    rb = cards.get_card(card.card_id)          # read-back
    assert rb.mint_dsr == 0.97
    assert rb.mint_sr_star == 0.31
    assert rb.mint_n_stratum == 500
    # the tier itself is from raw stats only: |t| 3.2 + p 0.04 + s4 pass => silver
    # (gold needs p_perm <= 0.01; s5 failed so not platinum)
    assert rb.tier == "silver"


def test_mint_none_dsr_fails_gate_closed(cards):
    card = cards.mint(
        trial_id="trial-nod", family="pairs", scope_id="free_5y_top1k",
        config_hash="h-nod", data_vintage_hash="v",
        survivor_class="validated-green", owner_handle="alice",
        dsr=None, t_net_full=4.5, p_perm=0.005, s4_pass=True, s5_pass=True,
    )
    assert card.tier == "bronze"     # None DSR -> no admission above bronze


# ---------------------------------------------------------------------------
# 3. meets_tier_bar uses RAW bars only (no DSR)
# ---------------------------------------------------------------------------

def test_meets_tier_bar_raw_only():
    # platinum: |t| >= 4.0 AND p <= 0.01 — dsr value irrelevant
    assert meets_tier_bar(WindowStats(0, 252, 0.0, 4.0, 0.01), "platinum")
    assert meets_tier_bar(WindowStats(0, 252, 0.999, 4.0, 0.01), "platinum")
    assert not meets_tier_bar(WindowStats(0, 252, 0.999, 3.99, 0.01), "platinum")
    assert not meets_tier_bar(WindowStats(0, 252, 0.999, 4.0, 0.011), "platinum")
    # gold: p_perm <= 0.01 (raw); dsr irrelevant
    assert meets_tier_bar(WindowStats(0, 252, 0.0, 9.0, 0.01), "gold")
    assert not meets_tier_bar(WindowStats(0, 252, 0.999, 9.0, 0.02), "gold")
    # silver: |t| >= 3.0 AND p <= 0.05
    assert meets_tier_bar(WindowStats(0, 252, 0.0, 3.0, 0.05), "silver")
    assert not meets_tier_bar(WindowStats(0, 252, 0.999, 2.99, 0.05), "silver")
    assert not meets_tier_bar(WindowStats(0, 252, 0.999, 3.0, 0.051), "silver")
    # missing stats never qualify
    assert not meets_tier_bar(WindowStats(0, 252, 0.999, None, 0.001), "platinum")
    assert not meets_tier_bar(WindowStats(0, 252, 0.999, 9.0, None), "gold")
    # unknown tier never qualifies
    assert not meets_tier_bar(WindowStats(0, 252, 0.999, 9.0, 0.001), "bronze")


# ---------------------------------------------------------------------------
# 4. health_ratio is N-SENSITIVE (the deliberate asymmetry)
# ---------------------------------------------------------------------------

def _live_series(n=2 * 252 + 40, seed=5):
    """A fixed, strongly mean-reverting live series (same card performance
    in every scenario — only N changes between scenarios)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    e = rng.normal(0, 0.01, n)
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = s[t - 1] + e[t]
    r = 0.002 - 0.8 * np.roll(s, 1) + e
    r[0] = 0.002 + e[0]
    return r


def test_health_ratio_drops_when_n_grows():
    """Synthetic stratum where N grows: unchanged card performance => the
    health ratio DROPS (the deflation responds to N)."""
    rets = _live_series()
    start, end = 252, 504
    h_small = health_ratio(rets, start=start, end=end,
                           n_stratum_current=10, v_stratum_current=0.01)
    h_large = health_ratio(rets, start=start, end=end,
                           n_stratum_current=10_000, v_stratum_current=0.01)
    assert h_small is not None and h_large is not None
    assert h_large < h_small          # E[max SR] grows with N -> ratio drops
    # monotone in N
    h_mid = health_ratio(rets, start=start, end=end,
                         n_stratum_current=1_000, v_stratum_current=0.01)
    assert h_small > h_mid > h_large


def test_health_ratio_stable_when_n_unchanged():
    """Unchanged N => health stable (deterministic, same inputs -> same
    ratio)."""
    rets = _live_series()
    kw = dict(start=252, end=504, n_stratum_current=500, v_stratum_current=0.01)
    h1 = health_ratio(rets, **kw)
    h2 = health_ratio(rets, **kw)
    assert h1 == h2
    assert h1 is not None and h1 > 0


def test_health_ratio_none_on_degenerate_window():
    assert health_ratio([0.001], start=0, end=1,
                        n_stratum_current=10, v_stratum_current=0.01) is None
    assert health_ratio([], start=0, end=0,
                        n_stratum_current=10, v_stratum_current=0.01) is None


def test_health_ratio_is_display_only_never_a_tier(cards, ledger):
    """health_ratio output never feeds tier_from_stats / meets_tier_bar —
    the tier functions take no health input (structural check)."""
    import inspect
    sig = inspect.signature(tier_from_stats)
    assert "health" not in sig.parameters
    sig2 = inspect.signature(meets_tier_bar)
    assert "health" not in sig2.parameters
    # and a huge N (deflated health) does not change the tier of raw stats
    assert tier_from_stats(dsr=0.10, t_net_full=4.5, p_perm=0.005,
                           s4_pass=True, s5_pass=True) == "platinum"


# ---------------------------------------------------------------------------
# 5. decay paths unchanged on raw stats (#157 rules) — evaluator ignores DSR
# ---------------------------------------------------------------------------

def test_default_re_evaluator_ignores_dsr(ledger, cards):
    """The default re-evaluator re-derives the tier from RAW ledger stats;
    changing the recorded S3 DSR (as if N grew) must NOT change the
    re-evaluated tier."""
    from sift.cards.decay import default_re_evaluator
    t = ledger.register("pairs", "free_5y_top1k", {"cfg": "reev"}, "v1",
                        discoverer_handle="alice")
    ledger.record_stage(t.trial_id, "s2", "pass", p_value=0.005)
    ledger.record_stage(t.trial_id, "s3", "pass", dsr=0.99)
    ledger.record_stage(t.trial_id, "s4", "pass", statistic=1.5)
    ledger.record_stage(t.trial_id, "s5", "pass", statistic=4.5)
    card = cards.mint(
        trial_id=t.trial_id, family=t.family, scope_id=t.scope_id,
        config_hash=t.config_hash, data_vintage_hash=t.data_vintage_hash,
        survivor_class="validated-green", owner_handle="alice",
        dsr=0.99, t_net_full=4.5, p_perm=0.005, s4_pass=True, s5_pass=True,
    )
    assert card.tier == "platinum"
    assert default_re_evaluator(card, ledger) == "platinum"
    # simulate the pool growing: the recorded DSR drops (deflated by a bigger
    # N) — the re-evaluated tier must NOT move
    ledger.con.execute(
        "UPDATE stage_result SET dsr=0.20 WHERE trial_id=? AND stage='s3'",
        (t.trial_id,),
    )
    ledger.con.commit()
    assert default_re_evaluator(cards.get_card(card.card_id), ledger) == "platinum"


def test_flag_off_kill_switch_unchanged(cards, monkeypatch):
    """flag cards_v4_1 off / SIFT_CARDS_V4_1=0 -> minting fails closed (no
    card written) — behavior unchanged from v4.1."""
    from sift.cards.minting import CardMinter
    ledger = Ledger(cards.db_path.parent / "ledger2.db")
    try:
        t = ledger.register("pairs", "free_5y_top1k", {"cfg": "k"}, "v1",
                            discoverer_handle="alice")
        ledger.record_stage(t.trial_id, "s2", "pass", p_value=0.005)
        ledger.record_stage(t.trial_id, "s3", "pass", dsr=0.99)
        ledger.record_stage(t.trial_id, "s4", "pass", statistic=1.5)
        ledger.record_stage(t.trial_id, "s5", "pass", statistic=4.5)
        minter = CardMinter(ledger, cards)
        flags.set_flag("cards_v4_1", False)
        assert minter.maybe_mint(t.trial_id, "alice", "survivor",
                                "validated-green") is None
        assert cards.all_cards() == []
        monkeypatch.setenv("SIFT_CARDS_V4_1", "0")
        flags.reset()
        assert minter.maybe_mint(t.trial_id, "alice", "survivor",
                                "validated-green") is None
        assert cards.all_cards() == []
    finally:
        ledger.close()


def test_card_tiers_ordering_unchanged():
    assert CARD_TIERS == ("bronze", "silver", "gold", "platinum")
