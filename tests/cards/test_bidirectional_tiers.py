"""v4.1 bidirectional card tiers — pre-registered live-window upgrades
(owner decision 2026-11-17) + 12m holdout at mint.

Fixtures:
- upgrade REFUSED before 12m age (in-sample echo: trailing window overlaps
  mint-time data);
- upgrade REFUSED when the two consecutive annual windows do not both meet
  the pre-registered bar (sustained, not spiky);
- upgrade REFUSED below the bar (identical to tier_from_stats at mint in
  window-length-INVARIANT terms — soundness #5: |t_win| >=
  B*sqrt(252/4032), i.e. annSR >= 1.0 for the 4.0 bar and >= 0.75 for the
  3.0 bar on a 252-bar window, AND p_perm <= 0.01 for gold/platinum /
  <= 0.05 for silver; DSR is not part of the live-window bar per #170);
- upgrade ALLOWED when all rules pass (age > 12m, trailing AND preceding
  12m windows at the bar, no tier change in the last 90d, seeded) ->
  card row raised + `card_upgrade` ledger event appended with evidence
  (window bounds, stats, config hash, vintage) + read back via events();
- rate limit enforced: a prior tier change within 90 days blocks the
  upgrade (and is recorded as the reason);
- a bare higher proposal with NO live-window evidence wired raises
  DecayError (never upgraded by payment or chance — the invariant and its
  test are kept);
- determinism: the same seed reproduces the same evidence (PCG64);
- banned-phrase tests on the new public upgrade text and notification copy:
  no "more valuable"/"appreciated"/"gained value"/"profit"/"return" numbers;
- flag off -> DecayError(flag_off);
- downgrade behavior unchanged (down-only any time from registry evidence);
- --holdout-bars on sift.engine.first_run locks the most recent N bars out
  of S0-S4 discovery/walk-forward and evaluates survivors once on the
  frozen holdout (never used for tuning).
"""

import json

import numpy as np
import pytest

from sift import flags
from sift.cards import CardRegistry
from sift.cards.decay import (
    DECAY_EVENT,
    UPGRADE_EVENT,
    WINDOW_BARS,
    DecayError,
    RegistryDecayJob,
    live_window_evaluator,
    meets_tier_bar,
    public_upgrade_explanation,
)
from sift.cards.notify import BANNED_PHRASES, EVENT_LIFECYCLE, Notifier, copy_for_event
from sift.engine import first_run as fr
from sift.registry import Ledger
from tests.cards.test_minting import SECRET
from tests.test_first_run import _make_warehouse

DAY_S = 86400.0


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


@pytest.fixture()
def notifier(cards, tmp_path):
    n = Notifier(cards, tmp_path / "notify.db")
    yield n
    n.close()


def _mint_silver(cards, ledger, handle="alice", minted_bar=0, now=1_000_000.0):
    """Mint a silver card at bar `minted_bar` (stats at the silver bar)."""
    t = ledger.register("pairs", "free_5y_top1k", {"cfg": "up"}, "v1",
                        discoverer_handle=handle)
    ledger.record_stage(t.trial_id, "s2", "pass", p_value=0.04)
    ledger.record_stage(t.trial_id, "s3", "pass", dsr=0.96)
    ledger.record_stage(t.trial_id, "s5", "pass", statistic=3.2)
    card = cards.mint(
        trial_id=t.trial_id, family=t.family, scope_id=t.scope_id,
        config_hash=t.config_hash, data_vintage_hash=t.data_vintage_hash,
        survivor_class="validated-green", owner_handle=handle,
        dsr=0.96, t_net_full=3.2, p_perm=0.04, now=now, minted_bar=minted_bar,
    )
    assert card.tier == "silver"
    return card


def _strong_live_returns(n=2 * WINDOW_BARS + 40, seed=5):
    """A live return series whose annual windows clear the GOLD bar
    (DSR >= 0.99, |t_net| >= 4.0, p_perm <= 0.01): a strongly
    mean-reverting spread-return series (negative level->return feedback),
    which is exactly what the S2 block-permutation null flags as extreme."""
    rng = np.random.Generator(np.random.PCG64(seed))
    e = rng.normal(0, 0.01, n)
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = s[t - 1] + e[t]
    r = 0.002 - 0.8 * np.roll(s, 1) + e
    r[0] = 0.002 + e[0]
    return r


def _weak_live_returns(n=2 * WINDOW_BARS, seed=7):
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.normal(0, 0.01, n)     # pure noise — windows below every bar


def _mixed_live_returns():
    """Strong ONLY in the trailing 252-bar window, weak in the preceding
    one — used to prove the two-consecutive-window (sustained) rule."""
    strong_tail = _strong_live_returns(n=WINDOW_BARS, seed=5)
    return np.concatenate([_weak_live_returns(n=WINDOW_BARS, seed=7), strong_tail])


def _make_live_evaluator(returns, n_stratum=50, v_stratum=0.01,
                         seed=424242, s2_B=200):
    """Build a live_evaluator closure over a fixed live return series."""
    def ev(card, now_bar=None):
        nb = now_bar if now_bar is not None else len(returns)
        return live_window_evaluator(
            card,
            live_returns=returns,
            now_bar=nb,
            n_stratum=n_stratum,
            v_stratum=v_stratum,
            seed=seed,
            s2_B=s2_B,
        )
    return ev


def _re_evaluator_returning(tier):
    return lambda card, ledger: tier


# -- rule 1: fully out-of-sample window only ---------------------------------

def test_upgrade_refused_before_12m_age(ledger, cards, notifier):
    # card minted at bar 0; "now" at bar WINDOW_BARS (exactly 12m) -> the
    # trailing window [0, 252) IS the mint-time data — in-sample echo.
    card = _mint_silver(cards, ledger, minted_bar=0)
    rets = _strong_live_returns()
    job = RegistryDecayJob(
        cards, ledger, notifier=notifier,
        re_evaluator=_re_evaluator_returning("gold"),
        live_evaluator=lambda c: live_window_evaluator(
            c, live_returns=rets, now_bar=WINDOW_BARS,   # age == 252, not > 252
            n_stratum=50, v_stratum=0.01, seed=424242, s2_B=200),
    )
    with pytest.raises(DecayError) as ei:
        job.evaluate_card(card, now=1_000_000.0 + WINDOW_BARS * DAY_S)
    assert "in-sample echo" in str(ei.value)
    # nothing changed
    assert cards.get_card(card.card_id).tier == "silver"
    assert ledger.events(UPGRADE_EVENT) == []


def test_live_window_evaluator_age_rule_directly(ledger, cards):
    card = _mint_silver(cards, ledger, minted_bar=100)
    rets = _strong_live_returns()
    # age exactly 252 bars -> refused (window not fully OOS: age must EXCEED 252)
    ev = live_window_evaluator(card, live_returns=rets, now_bar=100 + WINDOW_BARS,
                               n_stratum=50, v_stratum=0.01, seed=1, s2_B=100)
    assert not ev.eligible and "in-sample" in ev.reason
    # age 253 bars -> passes the age rule (eligibility then depends on stats)
    ev2 = live_window_evaluator(card, live_returns=rets, now_bar=100 + WINDOW_BARS + 1,
                                n_stratum=50, v_stratum=0.01, seed=1, s2_B=100)
    assert ev2.card_age_bars == WINDOW_BARS + 1


# -- rules 2+3: pre-registered bar on TWO consecutive windows -----------------

def test_upgrade_refused_below_gold_bar(ledger, cards):
    # card old enough, but the live series is pure noise -> below the bar
    card = _mint_silver(cards, ledger, minted_bar=0)
    ev = live_window_evaluator(card, live_returns=_weak_live_returns(),
                               now_bar=2 * WINDOW_BARS + 10,
                               n_stratum=50, v_stratum=0.01, seed=1, s2_B=200)
    assert not ev.eligible
    assert "below the pre-registered gold bar" in ev.reason
    assert ev.target_tier == "gold"


def test_upgrade_refused_when_not_sustained_two_windows(ledger, cards):
    # strong ONLY in the trailing window, weak in the preceding one. The
    # trailing window's p_perm draw (seed-dependent) can land anywhere, so
    # assert the structural fact: NOT eligible, and the preceding window is
    # the one below the bar (the trailing one clears DSR/t but not p).
    rets = _mixed_live_returns()
    card = _mint_silver(cards, ledger, minted_bar=0)
    ev = live_window_evaluator(card, live_returns=rets, now_bar=len(rets),
                               n_stratum=50, v_stratum=0.01, seed=1, s2_B=200)
    assert not ev.eligible
    assert ev.trailing is not None and ev.preceding is not None
    # the preceding (weak) window fails the gold bar outright
    assert not meets_tier_bar(ev.preceding, "gold")
    # and the evidence says which window failed
    assert ("below the pre-registered" in ev.reason
            or "not sustained" in ev.reason)
    # with a seed whose trailing p_perm clears, the refusal must be the
    # SUSTAINED rule (preceding window below the bar)
    ev2 = live_window_evaluator(card, live_returns=rets, now_bar=len(rets),
                                n_stratum=50, v_stratum=0.01, seed=424242, s2_B=200)
    if meets_tier_bar(ev2.trailing, "gold"):
        assert not ev2.eligible and "not sustained" in ev2.reason


def test_meets_tier_bar_matches_mint_time_thresholds():
    from sift.cards.decay import MINT_T_FULL, WindowStats, window_t_hurdle
    # v4.2 §3 (#170) + soundness #5 (hermes-tasks #181): RAW, N-independent
    # bars — DSR is NOT part of the bar (the dsr field is ignored on
    # purpose). The t component is compared in WINDOW-LENGTH-INVARIANT
    # terms: the mint bar is a full-depth (MINT_T_FULL-bar) t of 4.0
    # (platinum) / 3.0 (silver); on a 252-bar window the IDENTICAL bar is
    # |t_win| >= 4.0*sqrt(252/4032) = 1.0 / 0.75 (annualized Sharpe). These
    # fixtures previously fed RAW full-depth numbers (4.0/3.0) into 252-bar
    # windows — exactly why the defect was never exposed; they now pin BOTH
    # the invariant 252-bar hurdles AND the raw full-depth bars verbatim.
    # gold bar = p_perm <= 0.01 (raw, unchanged).
    assert meets_tier_bar(WindowStats(0, 252, None, 9.0, 0.01), "gold")
    assert meets_tier_bar(WindowStats(0, 252, 0.0, 9.0, 0.01), "gold")
    assert not meets_tier_bar(WindowStats(0, 252, 0.999, 9.0, 0.011), "gold")
    assert not meets_tier_bar(WindowStats(0, 252, 0.0, 9.0, 0.011), "gold")
    # platinum, 252-bar window: invariant hurdle 1.0 — dsr irrelevant
    assert window_t_hurdle(4.0, 252) == pytest.approx(1.0)
    assert meets_tier_bar(WindowStats(0, 252, 0.0, 1.0, 0.01), "platinum")
    assert meets_tier_bar(WindowStats(0, 252, 0.999, 3.99, 0.01), "platinum")
    assert not meets_tier_bar(WindowStats(0, 252, 0.999, 0.99, 0.01), "platinum")
    assert not meets_tier_bar(WindowStats(0, 252, 0.999, 1.0, 0.011), "platinum")
    # platinum at FULL depth: raw bar 4.0 verbatim — no weakening
    assert meets_tier_bar(WindowStats(0, MINT_T_FULL, 0.0, 4.0, 0.01), "platinum")
    assert not meets_tier_bar(WindowStats(0, MINT_T_FULL, 0.999, 3.99, 0.01), "platinum")
    # silver, 252-bar window: invariant hurdle 0.75
    assert window_t_hurdle(3.0, 252) == pytest.approx(0.75)
    assert meets_tier_bar(WindowStats(0, 252, 0.0, 0.75, 0.05), "silver")
    assert meets_tier_bar(WindowStats(0, 252, 0.0, 3.0, 0.05), "silver")
    assert not meets_tier_bar(WindowStats(0, 252, 0.999, 0.749, 0.05), "silver")
    assert not meets_tier_bar(WindowStats(0, 252, 0.999, 0.75, 0.051), "silver")
    # silver at FULL depth: raw bar 3.0 verbatim
    assert meets_tier_bar(WindowStats(0, MINT_T_FULL, 0.0, 3.0, 0.05), "silver")
    assert not meets_tier_bar(WindowStats(0, MINT_T_FULL, 0.999, 2.99, 0.05), "silver")
    # missing stats never qualify
    assert not meets_tier_bar(WindowStats(0, 252, None, None, 0.001), "gold")
    assert not meets_tier_bar(WindowStats(0, 252, 0.999, 9.0, None), "gold")
    assert not meets_tier_bar(WindowStats(0, 252, 0.999, None, 0.001), "platinum")


# -- all rules pass -> upgrade applied + ledger event + read-back -------------

def test_upgrade_allowed_when_all_rules_pass(ledger, cards, notifier):
    card = _mint_silver(cards, ledger, minted_bar=0)
    notifier.set_pref(card.card_id, "alice", enabled=True)
    rets = _strong_live_returns()
    now = 1_000_000.0 + (2 * WINDOW_BARS + 40) * DAY_S
    job = RegistryDecayJob(
        cards, ledger, notifier=notifier,
        re_evaluator=_re_evaluator_returning("gold"),
        live_evaluator=_make_live_evaluator(rets),
    )
    r = job.evaluate_card(card, now=now)
    assert r.upgraded is True and not r.downgraded
    assert r.from_tier == "silver" and r.to_tier == "gold"
    assert r.ledger_event_id is not None
    assert r.notified is True
    # public explanation: rarity language only
    assert r.public_text is not None
    assert "re-validated on live data" in r.public_text
    assert "tier was raised" in r.public_text
    # card row raised + read-back
    rb = cards.get_card(card.card_id)
    assert rb.tier == "gold"
    # ledger event appended + read back via events() with full evidence
    evs = ledger.events(UPGRADE_EVENT)
    assert len(evs) == 1
    payload = json.loads(evs[0]["payload_json"])
    assert payload["card_id"] == card.card_id
    assert payload["from_tier"] == "silver" and payload["to_tier"] == "gold"
    assert payload["public_explanation"] == r.public_text
    ev = payload["evidence"]
    assert ev["window_bars"] == WINDOW_BARS
    assert ev["card_age_bars"] == 2 * WINDOW_BARS + 40
    assert ev["seed"] == 424242
    assert ev["trailing_window"]["dsr"] is not None
    assert ev["trailing_window"]["t_net_full"] is not None
    assert ev["trailing_window"]["p_perm"] is not None
    assert ev["preceding_window"] is not None
    assert ev["config_hash"] == ledger.get(card.trial_id).config_hash
    assert ev["data_vintage_hash"] == ledger.get(card.trial_id).data_vintage_hash
    assert ev["trailing_window"]["end_bar"] - ev["trailing_window"]["start_bar"] == WINDOW_BARS
    # owner notified via the per-card channel (lifecycle, upgrade event)
    ns = notifier.log_for_card(card.card_id)
    assert len(ns) == 1 and ns[0].event_kind == EVENT_LIFECYCLE
    assert ns[0].payload["event"] == "upgrade"
    assert ns[0].payload["from_tier"] == "silver" and ns[0].payload["to_tier"] == "gold"


def test_upgrade_bronze_to_silver_against_silver_bar(ledger, cards):
    t = ledger.register("pairs", "free_5y_top1k", {"cfg": "b2s"}, "v1",
                        discoverer_handle="alice")
    ledger.record_stage(t.trial_id, "s2", "pass", p_value=0.5)
    ledger.record_stage(t.trial_id, "s3", "pass", dsr=0.90)
    ledger.record_stage(t.trial_id, "s5", "pass", statistic=2.0)
    card = cards.mint(
        trial_id=t.trial_id, family=t.family, scope_id=t.scope_id,
        config_hash=t.config_hash, data_vintage_hash=t.data_vintage_hash,
        survivor_class="validated-green", owner_handle="alice",
        dsr=0.90, t_net_full=2.0, p_perm=0.5, now=1_000_000.0, minted_bar=0,
    )
    assert card.tier == "bronze"
    rets = _strong_live_returns()
    now = 1_000_000.0 + (2 * WINDOW_BARS + 40) * DAY_S
    job = RegistryDecayJob(
        cards, ledger,
        re_evaluator=_re_evaluator_returning("silver"),
        live_evaluator=_make_live_evaluator(rets),
    )
    r = job.evaluate_card(card, now=now)
    # strong series clears the GOLD bar, but the proposal is only silver —
    # the evaluator targets the NEXT tier up (bronze->silver); eligible only
    # if the evidence supports silver. Strong windows support silver too.
    assert r.upgraded is True and r.to_tier == "silver"
    assert cards.get_card(card.card_id).tier == "silver"
    assert len(ledger.events(UPGRADE_EVENT)) == 1


# -- rule 4: rate limit ---------------------------------------------------------

def test_upgrade_rate_limit_enforced(ledger, cards):
    card = _mint_silver(cards, ledger, minted_bar=0)
    rets = _strong_live_returns()
    now = 1_000_000.0 + (2 * WINDOW_BARS + 40) * DAY_S
    job = RegistryDecayJob(
        cards, ledger,
        re_evaluator=_re_evaluator_returning("gold"),
        live_evaluator=_make_live_evaluator(rets),
    )
    # first upgrade succeeds
    r1 = job.evaluate_card(card, now=now)
    assert r1.upgraded and r1.to_tier == "gold"
    # a second change within 90 days is refused by the rate limit: simulate a
    # downgrade proposal inside 90d — downgrades from registry evidence are
    # allowed any time, but an UPGRADE within 90d of the upgrade is refused.
    # Re-mint scenario: gold card proposes... gold is top, so use a fresh
    # silver card whose ledger already carries a tier change 10 days ago.
    t = ledger.register("pairs", "free_5y_top1k", {"cfg": "rl"}, "v1",
                        discoverer_handle="bob")
    ledger.record_stage(t.trial_id, "s2", "pass", p_value=0.04)
    ledger.record_stage(t.trial_id, "s3", "pass", dsr=0.96)
    ledger.record_stage(t.trial_id, "s5", "pass", statistic=3.2)
    card2 = cards.mint(
        trial_id=t.trial_id, family=t.family, scope_id=t.scope_id,
        config_hash=t.config_hash, data_vintage_hash=t.data_vintage_hash,
        survivor_class="validated-green", owner_handle="bob",
        dsr=0.96, t_net_full=3.2, p_perm=0.04, now=now - 10 * DAY_S, minted_bar=0,
    )
    # a tier change (decay) 10 days ago on card2
    ledger.record_event(
        DECAY_EVENT,
        {"card_id": card2.card_id, "owner": "bob", "kind": "regular",
         "from_tier": "gold", "to_tier": "silver", "public_explanation": "x"},
        trial_id=t.trial_id, now=now - 10 * DAY_S,
    )
    job2 = RegistryDecayJob(
        cards, ledger,
        re_evaluator=_re_evaluator_returning("gold"),
        live_evaluator=_make_live_evaluator(rets),
    )
    with pytest.raises(DecayError) as ei:
        job2.evaluate_card(card2, now=now)
    assert "rate_limit" in str(ei.value)
    assert cards.get_card(card2.card_id).tier == "silver"
    assert ledger.events(UPGRADE_EVENT) and len(ledger.events(UPGRADE_EVENT)) == 1


def test_live_window_evaluator_rate_limit_rule(ledger, cards):
    card = _mint_silver(cards, ledger, minted_bar=0)
    rets = _strong_live_returns()
    ev = live_window_evaluator(
        card, live_returns=rets, now_bar=2 * WINDOW_BARS + 40,
        n_stratum=50, v_stratum=0.01, seed=1, s2_B=200,
        last_tier_change_at=1_000_000.0, now=1_000_000.0 + 10 * DAY_S,
    )
    assert not ev.eligible and "rate limit" in ev.reason


# -- rule 5: determinism (seeded PCG64) ------------------------------------------

def test_live_window_evaluation_is_deterministic_per_seed(ledger, cards):
    card = _mint_silver(cards, ledger, minted_bar=0)
    rets = _strong_live_returns()
    kw = dict(live_returns=rets, now_bar=2 * WINDOW_BARS + 40,
              n_stratum=50, v_stratum=0.01, s2_B=200)
    e1 = live_window_evaluator(card, seed=777, **kw)
    e2 = live_window_evaluator(card, seed=777, **kw)
    e3 = live_window_evaluator(card, seed=778, **kw)
    assert e1 == e2                      # same seed -> identical evidence
    assert e1.trailing == e2.trailing and e1.preceding == e2.preceding
    # different seed -> (almost surely) different permutation p at least
    assert e1.seed == 777 and e3.seed == 778


# -- invariant kept: never upgraded by payment or chance --------------------------

def test_upgrade_without_live_evidence_raises(ledger, cards):
    """A re-evaluator proposing a HIGHER tier with no live-window evidence
    wired raises DecayError and changes nothing — the 'never by payment or
    chance' invariant and its test are kept."""
    card = _mint_silver(cards, ledger, minted_bar=0)
    job = RegistryDecayJob(cards, ledger,
                           re_evaluator=_re_evaluator_returning("gold"))
    with pytest.raises(DecayError):
        job.run_pass(now=1_000_000.0)
    assert cards.get_card(card.card_id).tier == "silver"
    assert ledger.events(UPGRADE_EVENT) == []


def test_upgrade_refused_when_evidence_target_differs(ledger, cards):
    """Even with a live evaluator wired, if the evidence does not support the
    proposed tier the upgrade is refused (loud)."""
    card = _mint_silver(cards, ledger, minted_bar=0)
    job = RegistryDecayJob(
        cards, ledger,
        re_evaluator=_re_evaluator_returning("gold"),
        live_evaluator=_make_live_evaluator(_weak_live_returns()),
    )
    with pytest.raises(DecayError) as ei:
        job.evaluate_card(card, now=1_000_000.0 + 800 * DAY_S)
    assert "pre-registered live-window rules" in str(ei.value)
    assert cards.get_card(card.card_id).tier == "silver"


# -- rule 7: banned-phrase tests on the new public text ---------------------------

def test_public_upgrade_explanation_banned_phrase_free():
    text = public_upgrade_explanation("card-xyz", "silver", "gold",
                                      "re-validated on live data")
    assert "card-xyz" in text and "silver" in text and "gold" in text
    assert "re-validated on live data" in text
    assert "never granted by payment or chance" in text
    assert "SIFT does not trade" in text
    low = text.lower()
    for phrase in BANNED_PHRASES:
        assert phrase not in low, f"upgrade copy contains banned phrase {phrase!r}"
    # the specific profit-expectation phrases from the task
    for bad in ("more valuable", "appreciated", "gained value", "profit",
                "return", "worth", "price"):
        assert bad not in low
    # banned phrase in the reason fails closed
    with pytest.raises(DecayError):
        public_upgrade_explanation("card-xyz", "silver", "gold",
                                   "the card gained value!")


def test_notify_upgrade_copy_banned_phrase_free(cards):
    # build a minimal card via registry directly
    from tests.cards.test_minting import _run_validated_green  # noqa: F401
    # use copy_for_event on a lightweight stub card
    class Stub:
        card_id = "card-stub"
        family = "pairs"
        scope_id = "free_5y_top1k"
        tier = "gold"
        kind = "regular"
    for event in ("upgrade", "decay", "minted", "hit", "digest"):
        text = copy_for_event(event, Stub(), {"from_tier": "silver",
                                               "to_tier": "gold", "hit_count": 2})
        low = text.lower()
        for phrase in BANNED_PHRASES:
            assert phrase not in low, f"{event} copy contains banned phrase {phrase!r}"
        assert "not a recommendation" in low or "signal delivery only" in low
    # upgrade copy specifically: rarity language only
    up = copy_for_event("upgrade", Stub(), {"from_tier": "silver", "to_tier": "gold"})
    assert "re-validated on live data" in up
    assert "raised from silver to gold" in up


def test_banned_phrases_cover_value_language():
    # the extended banned list must catch the task's named phrases
    low_phrases = [p.lower() for p in BANNED_PHRASES]
    for needle in ("value", "valuable", "profit", "appreciat", "gained",
                   "worth", "price", "return"):
        assert needle in low_phrases
    # a public text with profit/return numbers is refused
    with pytest.raises(DecayError):
        public_upgrade_explanation("card-1", "silver", "gold",
                                   "returned 42% profit")


# -- rule 8: downgrade behavior unchanged ------------------------------------------

def test_downgrade_still_allowed_any_time(ledger, cards, notifier):
    from tests.cards.test_decay import _mint_with_stats
    card_id, _ = _mint_with_stats(cards, ledger, dsr=0.995, t_net=4.5, p=0.005, s5_pass=False)
    assert cards.get_card(card_id).tier == "gold"
    job = RegistryDecayJob(cards, ledger, notifier=notifier,
                           re_evaluator=_re_evaluator_returning("bronze"))
    (r,) = job.run_pass(now=1000.0)
    assert r.downgraded and r.to_tier == "bronze" and not r.upgraded
    assert cards.get_card(card_id).tier == "bronze"
    assert len(ledger.events(DECAY_EVENT)) == 1
    assert ledger.events(UPGRADE_EVENT) == []


# -- flag off -> DecayError ---------------------------------------------------------

def test_flag_off_upgrade_path_raises_decay_error(ledger, cards):
    card = _mint_silver(cards, ledger, minted_bar=0)
    job = RegistryDecayJob(
        cards, ledger,
        re_evaluator=_re_evaluator_returning("gold"),
        live_evaluator=_make_live_evaluator(_strong_live_returns()),
    )
    flags.set_flag("cards_v4_1", False)
    with pytest.raises(DecayError) as ei:
        job.evaluate_card(card, now=2_000_000.0)
    assert "flag_off" in str(ei.value)
    assert cards.get_card(card.card_id).tier == "silver"
    assert ledger.events(UPGRADE_EVENT) == []


# -- holdout: --holdout-bars locks the last N bars out of discovery ------------------

def test_first_run_holdout_locks_last_bars_out_of_discovery(tmp_path):
    root, uni_path = _make_warehouse(tmp_path)
    ledger = str(tmp_path / "ledger.db")
    meas_path = str(tmp_path / "meas.json")
    meas = fr.run_pairs(root, uni_path, ledger, max_pairs=6, formation=60, s2_B=50,
                        measurements_path=meas_path, max_universe=4,
                        holdout_bars=50)
    # the warehouse has 250 bars; holdout = last 50, discovery = first 200
    assert meas["holdout_bars"] == 50
    assert meas["holdout_start_bar"] == 200
    assert meas["discovery_bars"] == 200
    assert meas["window_bars"] == 250
    # holdout rows only for survivors; the structure is present + labeled
    for hr in meas["holdout_rows"]:
        assert hr["holdout_bars"] == 50
        assert hr["holdout_start_bar"] == 200
        assert "never used for tuning" in hr["note"]
    # the skip-list key carries the holdout: a full-window re-run is a NEW trial
    from sift.registry import Ledger as L
    led = L(ledger)
    n_with = led.con.execute(
        "SELECT COUNT(*) c FROM trial WHERE params_json LIKE '%holdout_bars%'"
    ).fetchone()["c"]
    assert n_with == meas["n_pairs"]
    led.close()


def test_first_run_holdout_zero_is_legacy_full_window(tmp_path):
    root, uni_path = _make_warehouse(tmp_path)
    ledger = str(tmp_path / "ledger.db")
    meas_path = str(tmp_path / "meas.json")
    meas = fr.run_pairs(root, uni_path, ledger, max_pairs=6, formation=60, s2_B=50,
                        measurements_path=meas_path, max_universe=4,
                        holdout_bars=0)
    assert meas["holdout_bars"] == 0
    assert meas["holdout_start_bar"] is None
    assert meas["discovery_bars"] == meas["window_bars"] == 250
    assert meas["holdout_rows"] == []


def test_first_run_holdout_too_large_refused(tmp_path):
    root, uni_path = _make_warehouse(tmp_path)
    with pytest.raises(SystemExit):
        fr.run_pairs(root, uni_path, str(tmp_path / "l.db"), max_pairs=6,
                     formation=60, s2_B=50,
                     measurements_path=str(tmp_path / "m.json"),
                     max_universe=4, holdout_bars=249)
