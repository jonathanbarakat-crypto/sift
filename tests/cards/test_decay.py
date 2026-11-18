"""v4.1 mechanics 7/8 — registry decay re-evaluation: amber downgrades
transitions, public explanation, ledger event, per-card notification hook,
flag-off refusal.

Fixtures:
- a registry pass re-evaluates every card; a card whose re-evaluated tier
  is LOWER is downgraded on the card row + a `card_decay` event is appended
  to the trial ledger + the owner is notified via the per-card channel.
- a re-evaluator proposing a
  HIGHER tier without pre-registered live-window evidence raises DecayError
  and changes nothing (never upgraded by payment or chance).
- unchanged tier -> no event, no notification, no row change.
- the amber transition is PUBLIC + EXPLAINED + LOGGED: public_explanation
  text is attached to the ledger event payload; banned-phrase scan fails
  closed; the event is read back from the ledger.
- the decay notification goes through the per-card opt-in channel: opted-
  out owners are NOT notified (opt-in respected), opted-in owners get a
  lifecycle notification capped at 1/week.
- ghost cards are re-evaluated too (full utility).
- flag off -> DecayError(flag_off) on run_pass / evaluate_card; card rows
  and ledger stay readable.
"""

import json

import pytest

from sift import flags
from sift.cards import CARD_TIERS, CardRegistry
from sift.cards.decay import (
    DECAY_EVENT,
    DecayError,
    RegistryDecayJob,
    default_re_evaluator,
    public_explanation,
)
from sift.cards.notify import (
    BANNED_PHRASES,
    EVENT_LIFECYCLE,
    Notifier,
)
from sift.registry import Ledger
from tests.cards.test_minting import SECRET, _run_validated_green


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


@pytest.fixture()
def notifier(cards, tmp_path):
    n = Notifier(cards, tmp_path / "notify.db")
    yield n
    n.close()


def _mint_with_stats(cards, ledger, handle="alice", trial="t1",
                     dsr=0.995, t_net=4.5, p=0.005):
    """Register a trial with stage rows so default_re_evaluator can read
    them, then mint a card at the given (mint-time) stats."""
    t = ledger.register("pairs", "free_5y_top1k", {"cfg": trial}, "v1",
                        discoverer_handle=handle)
    ledger.record_stage(t.trial_id, "s2", "pass", p_value=p)
    ledger.record_stage(t.trial_id, "s3", "pass", dsr=dsr)
    ledger.record_stage(t.trial_id, "s5", "pass", statistic=t_net)
    card = cards.mint(
        trial_id=t.trial_id, family=t.family, scope_id=t.scope_id,
        config_hash=t.config_hash, data_vintage_hash=t.data_vintage_hash,
        survivor_class="validated-green", owner_handle=handle,
        dsr=dsr, t_net_full=t_net, p_perm=p,
    )
    return card.card_id, t.trial_id


def _re_evaluator_returning(tier):
    """A re-evaluator stub that always proposes `tier`."""
    return lambda card, ledger: tier


# -- downgrade (amber) transitions ---------------------------------------------------------

def test_pass_downgrades_card_row_ledger_event_and_notifies(ledger, cards, notifier):
    # mint gold (dsr .995 / t 4.5 / p .005), then the registry's fresh
    # evidence re-evaluates to silver
    card_id, trial_id = _mint_with_stats(cards, ledger, dsr=0.995, t_net=4.5, p=0.005)
    assert cards.get_card(card_id).tier == "gold"
    notifier.set_pref(card_id, "alice", enabled=True)

    job = RegistryDecayJob(cards, ledger, notifier=notifier,
                           re_evaluator=_re_evaluator_returning("silver"))
    results = job.run_pass(now=1000.0)
    assert len(results) == 1
    r = results[0]
    assert r.downgraded is True
    assert r.from_tier == "gold" and r.to_tier == "silver"
    assert r.ledger_event_id is not None
    assert r.notified is True
    # public + explained
    assert r.public_text is not None and "amber" in r.public_text
    assert "gold" in r.public_text and "silver" in r.public_text
    # card row written + read-back
    rb = cards.get_card(card_id)
    assert rb.tier == "silver" and rb.decayed_from == "gold"
    # ledger event appended + read-back
    evs = ledger.events(DECAY_EVENT)
    assert len(evs) == 1
    payload = json.loads(evs[0]["payload_json"])
    assert payload["card_id"] == card_id and payload["from_tier"] == "gold"
    assert payload["to_tier"] == "silver"
    assert payload["public_explanation"] == r.public_text
    assert evs[0]["trial_id"] == trial_id
    # owner notified via the per-card channel (lifecycle, digest mode)
    ns = notifier.log_for_card(card_id)
    assert len(ns) == 1 and ns[0].event_kind == EVENT_LIFECYCLE
    assert ns[0].payload["event"] == "decay"
    assert ns[0].payload["from_tier"] == "gold" and ns[0].payload["to_tier"] == "silver"


def test_decay_never_upgrades(ledger, cards, notifier):
    # mint bronze; a re-evaluator proposing gold (e.g. "payment") must be
    # REFUSED — never upgraded by payment or chance
    card_id, _ = _mint_with_stats(cards, ledger, dsr=0.90, t_net=2.0, p=0.5)
    assert cards.get_card(card_id).tier == "bronze"
    job = RegistryDecayJob(cards, ledger, notifier=notifier,
                           re_evaluator=_re_evaluator_returning("gold"))
    with pytest.raises(DecayError):
        job.run_pass(now=1000.0)
    # nothing changed: card row, no ledger event
    assert cards.get_card(card_id).tier == "bronze"
    assert ledger.events(DECAY_EVENT) == []


def test_unchanged_tier_is_a_no_op(ledger, cards, notifier):
    card_id, _ = _mint_with_stats(cards, ledger, dsr=0.96, t_net=3.2, p=0.04)
    assert cards.get_card(card_id).tier == "silver"
    notifier.set_pref(card_id, "alice", enabled=True)
    job = RegistryDecayJob(cards, ledger, notifier=notifier,
                           re_evaluator=_re_evaluator_returning("silver"))
    (r,) = job.run_pass(now=1000.0)
    assert r.downgraded is False and r.to_tier == "silver"
    assert r.ledger_event_id is None and r.notified is False
    assert cards.get_card(card_id).tier == "silver"
    assert ledger.events(DECAY_EVENT) == []
    assert notifier.log_for_card(card_id) == []


def test_default_re_evaluator_uses_preregistered_ledger_stats(ledger, cards, notifier):
    """The default re-evaluator re-derives the tier from the card's own S2/S3/S5
    stage rows — a static function of gauntlet strength, not of payment."""
    card_id, trial_id = _mint_with_stats(cards, ledger, dsr=0.995, t_net=4.5, p=0.005)
    card = cards.get_card(card_id)
    assert default_re_evaluator(card, ledger) == "gold"
    # weaken the recorded S5 stat -> re-evaluation proposes silver -> downgrade
    ledger.con.execute(
        "UPDATE stage_result SET statistic=3.1 WHERE trial_id=? AND stage='s5'",
        (trial_id,),
    )
    ledger.con.commit()
    assert default_re_evaluator(cards.get_card(card_id), ledger) == "silver"
    job = RegistryDecayJob(cards, ledger, notifier=notifier)   # default evaluator
    (r,) = job.run_pass(now=1000.0)
    assert r.downgraded and r.to_tier == "silver"
    assert cards.get_card(card_id).tier == "silver"


def test_full_downgrade_ladder_gold_to_bronze(ledger, cards, notifier):
    card_id, _ = _mint_with_stats(cards, ledger, dsr=0.995, t_net=4.5, p=0.005)
    job = RegistryDecayJob(cards, ledger, notifier=notifier,
                           re_evaluator=_re_evaluator_returning("silver"))
    job.run_pass(now=1000.0)
    assert cards.get_card(card_id).tier == "silver"
    job2 = RegistryDecayJob(cards, ledger, notifier=notifier,
                            re_evaluator=_re_evaluator_returning("bronze"))
    job2.run_pass(now=2000.0)
    assert cards.get_card(card_id).tier == "bronze"
    # already bronze: proposing bronze again is a no-op, not an error
    job3 = RegistryDecayJob(cards, ledger, notifier=notifier,
                            re_evaluator=_re_evaluator_returning("bronze"))
    (r,) = job3.run_pass(now=3000.0)
    assert not r.downgraded
    # two decay events total, append-only
    assert len(ledger.events(DECAY_EVENT)) == 2


# -- amber transition is public + explained + logged -------------------------------------

def test_public_explanation_text_is_public_explained_and_banned_phrase_free():
    text = public_explanation("card-abc", "gold", "silver",
                              "registry re-evaluation reason")
    assert "card-abc" in text
    assert "gold" in text and "silver" in text and "amber" in text
    # bidirectional copy: down on re-evaluation, up only via pre-registered
    # live-window rules — never by payment or chance
    assert "move down" in text and "never by payment or chance" in text
    assert "SIFT does not trade" in text
    low = text.lower()
    for phrase in BANNED_PHRASES:
        assert phrase not in low
    # banned phrase in the reason would fail closed
    with pytest.raises(DecayError):
        public_explanation("card-abc", "gold", "silver", "value appreciates!")


def test_amber_transition_logged_in_ledger_and_visible_on_card(ledger, cards, notifier):
    card_id, trial_id = _mint_with_stats(cards, ledger, dsr=0.995, t_net=4.5, p=0.005)
    job = RegistryDecayJob(cards, ledger, notifier=notifier,
                           re_evaluator=_re_evaluator_returning("bronze"))
    (r,) = job.run_pass(now=1000.0)
    # logged: card_decay event with the public explanation in the payload
    (ev,) = ledger.events(DECAY_EVENT)
    payload = json.loads(ev["payload_json"])
    assert payload["public_explanation"] == r.public_text
    assert payload["public_explanation"] == public_explanation(
        card_id, "gold", "bronze", payload["reason"])
    # visible on the card row: decayed_from recorded, read-back
    assert cards.get_card(card_id).decayed_from == "gold"
    assert cards.get_card(card_id).tier == "bronze"


# -- notification hook respects opt-in + caps -------------------------------------------------

def test_decay_notification_respects_opt_in(ledger, cards, notifier):
    card_id, _ = _mint_with_stats(cards, ledger, dsr=0.995, t_net=4.5, p=0.005)
    # NOT opted in -> downgrade still happens, but no notification enqueued
    job = RegistryDecayJob(cards, ledger, notifier=notifier,
                           re_evaluator=_re_evaluator_returning("silver"))
    (r,) = job.run_pass(now=1000.0)
    assert r.downgraded is True and r.notified is False
    assert notifier.log_for_card(card_id) == []
    # card row + ledger still written (the downgrade is public regardless)
    assert cards.get_card(card_id).tier == "silver"
    assert len(ledger.events(DECAY_EVENT)) == 1


def test_decay_notification_lifecycle_capped_at_one_per_week(ledger, cards, notifier):
    card_id, _ = _mint_with_stats(cards, ledger, dsr=0.995, t_net=4.5, p=0.005)
    notifier.set_pref(card_id, "alice", enabled=True)
    job = RegistryDecayJob(cards, ledger, notifier=notifier,
                           re_evaluator=_re_evaluator_returning("silver"))
    job.run_pass(now=1000.0)
    # a second pass in the same week proposing another downgrade: the card
    # downgrades but the lifecycle notification is capped (1/week, v3 §7.2)
    job2 = RegistryDecayJob(cards, ledger, notifier=notifier,
                            re_evaluator=_re_evaluator_returning("bronze"))
    (r2,) = job2.run_pass(now=2000.0)
    assert r2.downgraded is True and r2.notified is False
    assert cards.get_card(card_id).tier == "bronze"
    ns = notifier.log_for_card(card_id)
    assert len(ns) == 1 and ns[0].event_kind == EVENT_LIFECYCLE


# -- ghost cards re-evaluated too --------------------------------------------------------------

def test_ghost_cards_are_re_evaluated(ledger, cards, notifier):
    card_id, trial_id = _mint_with_stats(cards, ledger, dsr=0.995, t_net=4.5, p=0.005)
    t = ledger.get(trial_id)
    ghost = cards.mint(
        trial_id=t.trial_id, family=t.family, scope_id=t.scope_id,
        config_hash=t.config_hash, data_vintage_hash=t.data_vintage_hash,
        survivor_class="validated-green", owner_handle="alice",
        dsr=0.995, t_net_full=4.5, p_perm=0.005,
    )
    assert ghost.kind == "ghost"
    notifier.set_pref(ghost.card_id, "alice", enabled=True)
    job = RegistryDecayJob(cards, ledger, notifier=notifier,
                           re_evaluator=_re_evaluator_returning("silver"))
    results = job.run_pass(now=1000.0)
    assert len(results) == 2                      # regular + ghost both evaluated
    assert all(r.downgraded for r in results)
    assert cards.get_card(ghost.card_id).tier == "silver"
    assert len(ledger.events(DECAY_EVENT)) == 2


# -- flag off -----------------------------------------------------------------------------------

def test_flag_off_raises_decay_error_and_leaves_data(ledger, cards, notifier):
    card_id, _ = _mint_with_stats(cards, ledger, dsr=0.995, t_net=4.5, p=0.005)
    job = RegistryDecayJob(cards, ledger, notifier=notifier,
                           re_evaluator=_re_evaluator_returning("bronze"))
    flags.set_flag("cards_v4_1", False)
    with pytest.raises(DecayError) as ei:
        job.run_pass(now=1000.0)
    assert "flag_off" in str(ei.value)
    card = cards.get_card(card_id)
    with pytest.raises(DecayError):
        job.evaluate_card(card, now=1000.0)
    # nothing changed; data stays readable
    assert cards.get_card(card_id).tier == "gold"
    assert ledger.events(DECAY_EVENT) == []
    flags.set_flag("cards_v4_1", True)
    (r,) = job.run_pass(now=2000.0)
    assert r.downgraded and r.to_tier == "bronze"


def test_flag_off_env_kill_switch(ledger, cards, monkeypatch):
    monkeypatch.setenv("SIFT_CARDS_V4_1", "0")
    flags.reset()
    card_id, _ = _mint_with_stats(cards, ledger)
    job = RegistryDecayJob(cards, ledger,
                           re_evaluator=_re_evaluator_returning("bronze"))
    with pytest.raises(DecayError):
        job.run_pass(now=1000.0)
    assert cards.get_card(card_id).tier == "gold"     # untouched
    monkeypatch.delenv("SIFT_CARDS_V4_1")
    flags.reset()


# -- invalid re-evaluator output ------------------------------------------------------------------

def test_invalid_tier_from_evaluator_raises(ledger, cards):
    card_id, _ = _mint_with_stats(cards, ledger)
    job = RegistryDecayJob(cards, ledger, re_evaluator=lambda c, l: "diamond")
    with pytest.raises(DecayError):
        job.run_pass(now=1000.0)
    assert cards.get_card(card_id).tier == "gold"


# -- gauntlet-integrated fixture (real survivor -> decay pass) -------------------------------------

def test_gauntlet_card_decay_pass_end_to_end(tmp_path):
    ledger = Ledger(tmp_path / "ledger.db")
    cards = CardRegistry(tmp_path / "cards.db", secret=SECRET)
    notifier = Notifier(cards, tmp_path / "notify.db")
    try:
        verdict, _ = _run_validated_green(ledger, cards)
        card_id = verdict["card_id"]
        notifier.set_pref(card_id, "alice", enabled=True)
        # registry evidence now says the edge weakened at full depth
        ledger.con.execute(
            "UPDATE stage_result SET statistic=1.0 WHERE trial_id=? AND stage='s5'",
            (cards.get_card(card_id).trial_id,),
        )
        ledger.con.commit()
        job = RegistryDecayJob(cards, ledger, notifier=notifier)  # default evaluator
        (r,) = job.run_pass(now=1000.0)
        assert r.downgraded is True
        assert r.to_tier in CARD_TIERS and CARD_TIERS.index(r.to_tier) < CARD_TIERS.index(r.from_tier)
        assert cards.get_card(card_id).tier == r.to_tier          # read-back
        assert len(ledger.events(DECAY_EVENT)) == 1               # read-back
        assert r.notified is True
    finally:
        notifier.close()
        cards.close()
        ledger.close()
