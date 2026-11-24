"""v4.1 mechanics 7/8 — per-card opt-in hit notifications: prefs, digest vs
live per tier, rate caps, append-only queue, ghost parity, flag-off refusal.

Fixtures:
- opt-in DEFAULT OFF: a fresh card has no pref / enabled=False -> notify()
  enqueues nothing (returns None).
- free tier defaults to DIGEST even if live is requested (no paid
  entitlement); paid entitlement -> live.
- digest batching: queued digest-mode hits are batched into ONE digest row
  per card per day; the hit rows are marked delivered, never deleted.
- rate cap: live hits capped at LIVE_CAP_PER_CARD_PER_DAY per card; digest
  summary capped at 1/day; lifecycle (mint/decay) capped at 1/WEEK (v3 §7.2).
- queue is append-only: delivered rows remain in the log (read-back).
- ghost cards get full notification utility (same path, kind='ghost').
- flag off -> NotifyError(flag_off) on mutations; reads stay available.
- copy: no banned (appreciation/urgency/loss-aversion) phrases in any
  generated notification copy (legal posture §6).
- signals-only: payload carries survival facts, no execution/betting fields.
"""


import pytest

from sift import flags
from sift.cards import CardRegistry
from sift.cards.notify import (
    BANNED_PHRASES,
    DAY,
    DIGEST_CAP_PER_CARD_PER_DAY,
    EVENT_DIGEST,
    EVENT_HIT,
    EVENT_LIFECYCLE,
    LIFECYCLE_CAP_PER_CARD_PER_WEEK,
    LIVE_CAP_PER_CARD_PER_DAY,
    LIVE_ENTITLEMENT,
    MODE_DIGEST,
    MODE_LIVE,
    WEEK,
    Notifier,
    NotifyError,
    copy_for_event,
)
from tests.cards.test_minting import SECRET, _run_validated_green


@pytest.fixture(autouse=True)
def _flags_reset():
    flags.reset()
    yield
    flags.reset()


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


def _mint(cards, handle="alice", tier_stats=(0.995, 4.5, 0.005), trial="t1"):
    """Direct mint (fast; no gauntlet run). tier_stats = (dsr, t_net, p_perm)."""
    dsr, t_net, p = tier_stats
    return cards.mint(
        trial_id=trial, family="pairs", scope_id="free_5y_top1k",
        config_hash=f"h-{trial}", data_vintage_hash="v1",
        survivor_class="validated-green", owner_handle=handle,
        dsr=dsr, t_net_full=t_net, p_perm=p,
    ).card_id


# -- opt-in default off -----------------------------------------------------------

def test_opt_in_default_off_free_tier(notifier, cards):
    card_id = _mint(cards)
    # no pref row -> default disabled, digest mode
    pref = notifier.get_pref(card_id, "alice")
    assert pref.enabled is False
    assert pref.mode == MODE_DIGEST
    # notify() without opt-in enqueues NOTHING
    assert notifier.notify(card_id, "alice", "hit") is None
    assert notifier.log_for_card(card_id) == []
    # opt-in explicitly
    pref = notifier.set_pref(card_id, "alice", enabled=True, mode=MODE_DIGEST)
    assert pref.enabled is True and pref.mode == MODE_DIGEST
    # read-back: pref persisted
    assert notifier.get_pref(card_id, "alice").enabled is True


def test_free_tier_defaults_to_digest_even_when_live_requested(notifier, cards):
    card_id = _mint(cards)
    # free tier (no entitlement): requesting live resolves to digest
    pref = notifier.set_pref(card_id, "alice", enabled=True, mode=MODE_LIVE,
                              entitlement=None)
    assert pref.mode == MODE_DIGEST
    # paid entitlement unlocks live
    pref = notifier.set_pref(card_id, "alice", enabled=True, mode=MODE_LIVE,
                              entitlement=LIVE_ENTITLEMENT)
    assert pref.mode == MODE_LIVE


def test_effective_mode_resolution(notifier, cards):
    assert notifier.effective_mode("alice", MODE_LIVE, entitlement=None) == MODE_DIGEST
    assert notifier.effective_mode("alice", MODE_LIVE, entitlement=LIVE_ENTITLEMENT) == MODE_LIVE
    assert notifier.effective_mode("alice", MODE_DIGEST, entitlement=None) == MODE_DIGEST
    with pytest.raises(NotifyError):
        notifier.effective_mode("alice", "carrier-pigeon")


# -- enqueue + read-back ------------------------------------------------------------

def test_live_hit_enqueues_and_reads_back(notifier, cards):
    card_id = _mint(cards)
    notifier.set_pref(card_id, "alice", enabled=True, mode=MODE_LIVE,
                      entitlement=LIVE_ENTITLEMENT)
    n = notifier.notify(card_id, "alice", "hit",
                        {"signal": "survival-fact-only"},
                        entitlement=LIVE_ENTITLEMENT, now=1000.0)
    assert n is not None and n.state == "queued" and n.mode == MODE_LIVE
    assert n.event_kind == EVENT_HIT
    # read-back from the DB (hard rule: GET after write)
    rb = notifier.get_notif(n.notif_id)
    assert rb.card_id == card_id and rb.owner == "alice"
    assert rb.payload["event"] == "hit"
    assert rb.payload["kind"] == "regular"
    assert rb.payload["tier"] == cards.get_card(card_id).tier
    # signals-only: no execution/betting fields in the payload
    for banned_key in ("side", "direction", "bet", "wager", "execution", "pnl", "p&l"):
        assert banned_key not in {k.lower() for k in rb.payload}
    # copy present, no banned phrases
    assert rb.payload["copy"]
    for phrase in BANNED_PHRASES:
        assert phrase not in rb.payload["copy"].lower()


def test_non_owner_cannot_notify(notifier, cards):
    card_id = _mint(cards, handle="alice")
    notifier.set_pref(card_id, "alice", enabled=True)
    from sift.cards import CardError
    with pytest.raises(CardError):
        notifier.notify(card_id, "eve", "hit")   # non-transferable


# -- rate caps -----------------------------------------------------------------------

def test_live_rate_cap_enforced(notifier, cards):
    card_id = _mint(cards)
    notifier.set_pref(card_id, "alice", enabled=True, mode=MODE_LIVE,
                      entitlement=LIVE_ENTITLEMENT)
    t0 = 1_000_000.0
    sent = 0
    for i in range(LIVE_CAP_PER_CARD_PER_DAY + 3):
        n = notifier.notify(card_id, "alice", "hit",
                            entitlement=LIVE_ENTITLEMENT, now=t0 + i)
        if n is not None:
            sent += 1
    assert sent == LIVE_CAP_PER_CARD_PER_DAY          # capped, never spammed
    # next day the cap resets
    n = notifier.notify(card_id, "alice", "hit",
                        entitlement=LIVE_ENTITLEMENT, now=t0 + DAY + 1)
    assert n is not None


def test_lifecycle_cap_one_per_week(notifier, cards):
    card_id = _mint(cards)
    notifier.set_pref(card_id, "alice", enabled=True, mode=MODE_DIGEST)
    t0 = 1_000_000.0
    n1 = notifier.notify(card_id, "alice", "decay",
                         {"from_tier": "gold", "to_tier": "silver"}, now=t0)
    assert n1 is not None and n1.event_kind == EVENT_LIFECYCLE
    # second lifecycle email within the week -> capped (v3 §7.2: 1/week)
    n2 = notifier.notify(card_id, "alice", "minted", now=t0 + DAY)
    assert n2 is None
    assert notifier.within_cap(card_id, EVENT_LIFECYCLE, MODE_DIGEST, now=t0 + DAY) is False
    # after a week, allowed again
    n3 = notifier.notify(card_id, "alice", "decay", now=t0 + WEEK + 1)
    assert n3 is not None
    assert LIFECYCLE_CAP_PER_CARD_PER_WEEK == 1


# -- digest batching -------------------------------------------------------------------

def test_digest_batching_one_summary_per_card_per_day(notifier, cards):
    card_id = _mint(cards)
    notifier.set_pref(card_id, "alice", enabled=True, mode=MODE_DIGEST)  # free default
    t0 = 1_000_000.0
    # several hits queue up (digest-mode hits are not capped at enqueue)
    for i in range(7):
        n = notifier.notify(card_id, "alice", "hit", now=t0 + i)
        assert n is not None and n.mode == MODE_DIGEST
    assert len(notifier.pending("alice")) == 7
    # digest batch: ONE digest row, hits marked delivered, rows kept
    digests = notifier.digest_batch("alice", now=t0 + 100)
    assert len(digests) == 1
    d = digests[0]
    assert d.event_kind == EVENT_DIGEST and d.mode == MODE_DIGEST
    assert d.payload["hit_count"] == 7 and len(d.payload["hit_ids"]) == 7
    # read-back: hit rows delivered but STILL in the append-only log
    log = notifier.log_for_card(card_id)
    assert len(log) == 8                                  # 7 hits + 1 digest
    hits = [x for x in log if x.event_kind == EVENT_HIT]
    assert all(x.state == "delivered" for x in hits)
    assert all(x.delivered_at is not None for x in hits)
    # second batch the same day -> capped (<=1 digest/day)
    more = notifier.notify(card_id, "alice", "hit", now=t0 + 200)
    assert more is not None
    digests2 = notifier.digest_batch("alice", now=t0 + 300)
    assert digests2 == []
    # next day a new digest is allowed
    digests3 = notifier.digest_batch("alice", now=t0 + DAY + 400)
    assert len(digests3) == 1 and digests3[0].payload["hit_count"] == 1
    assert DIGEST_CAP_PER_CARD_PER_DAY == 1


# -- delivery exactly once ---------------------------------------------------------------

def test_deliver_due_marks_exactly_once(notifier, cards):
    card_id = _mint(cards)
    notifier.set_pref(card_id, "alice", enabled=True, mode=MODE_LIVE,
                      entitlement=LIVE_ENTITLEMENT)
    n = notifier.notify(card_id, "alice", "hit",
                        entitlement=LIVE_ENTITLEMENT, now=1000.0)
    delivered = notifier.deliver_due("alice", now=2000.0)
    assert [x.notif_id for x in delivered] == [n.notif_id]
    assert notifier.get_notif(n.notif_id).state == "delivered"
    # idempotent: second pass delivers nothing new
    assert notifier.deliver_due("alice", now=3000.0) == []
    # append-only: the row is still there
    assert notifier.get_notif(n.notif_id).delivered_at == 2000.0


# -- ghost parity ------------------------------------------------------------------------

def test_ghost_cards_get_full_notification_utility(notifier, cards):
    _mint(cards, handle="alice", trial="t1")
    # same canonical spec again -> ghost
    ghost = cards.mint(
        trial_id="t1", family="pairs", scope_id="free_5y_top1k",
        config_hash="h-t1", data_vintage_hash="v1",
        survivor_class="validated-green", owner_handle="alice",
        dsr=0.995, t_net_full=4.5, p_perm=0.005,
    )
    assert ghost.kind == "ghost"
    notifier.set_pref(ghost.card_id, "alice", enabled=True, mode=MODE_LIVE,
                      entitlement=LIVE_ENTITLEMENT)
    n = notifier.notify(ghost.card_id, "alice", "hit", now=1000.0)
    assert n is not None
    assert n.payload["kind"] == "ghost"                   # full utility, marked
    assert notifier.get_notif(n.notif_id).card_id == ghost.card_id


# -- flag off -----------------------------------------------------------------------------

def test_flag_off_refuses_mutations_reads_stay(notifier, cards):
    card_id = _mint(cards)
    notifier.set_pref(card_id, "alice", enabled=True, mode=MODE_DIGEST)
    notifier.notify(card_id, "alice", "hit", now=1000.0)
    flags.set_flag("cards_v4_1", False)
    with pytest.raises(NotifyError) as ei:
        notifier.set_pref(card_id, "alice", enabled=False)
    assert "flag_off" in str(ei.value)
    with pytest.raises(NotifyError):
        notifier.notify(card_id, "alice", "hit")
    with pytest.raises(NotifyError):
        notifier.digest_batch("alice")
    with pytest.raises(NotifyError):
        notifier.deliver_due("alice")
    # reads stay available (kill-switch never hides data)
    assert len(notifier.log_for_card(card_id)) == 1
    assert notifier.get_pref(card_id, "alice").enabled is True
    flags.set_flag("cards_v4_1", True)
    assert notifier.notify(card_id, "alice", "hit", now=2000.0) is not None


def test_flag_off_env_kill_switch(notifier, cards, monkeypatch):
    monkeypatch.setenv("SIFT_CARDS_V4_1", "0")
    flags.reset()
    card_id = _mint(cards)
    with pytest.raises(NotifyError):
        notifier.set_pref(card_id, "alice", enabled=True)
    monkeypatch.delenv("SIFT_CARDS_V4_1")
    flags.reset()


# -- copy discipline (legal posture §6) ---------------------------------------------------

def test_copy_never_uses_appreciation_or_urgency_language(notifier, cards):
    card_id = _mint(cards)
    card = cards.get_card(card_id)
    for event in ("hit", "minted", "decay", "digest"):
        text = copy_for_event(event, card, {"from_tier": "gold", "to_tier": "silver",
                                            "hit_count": 3})
        low = text.lower()
        for phrase in BANNED_PHRASES:
            assert phrase not in low, f"{event} copy contains banned phrase {phrase!r}"
        # rarity/survival language present, no 'value appreciates'
        assert "appreciat" not in low
        assert "not a recommendation" in low or "signal delivery only" in low.lower()
    with pytest.raises(NotifyError):
        copy_for_event("warp-drive", card)


# -- gauntlet-integrated fixture (new path: real survivor -> notify) ------------------------

def test_gauntlet_minted_card_notifies_on_opt_in(tmp_path):
    from sift.registry import Ledger
    ledger = Ledger(tmp_path / "ledger.db")
    cards = CardRegistry(tmp_path / "cards.db", secret=SECRET)
    notifier = Notifier(cards, tmp_path / "notify.db")
    try:
        verdict, _ = _run_validated_green(ledger, cards)
        card_id = verdict["card_id"]
        assert card_id is not None
        # default off
        assert notifier.notify(card_id, "alice", "hit", now=1000.0) is None
        # opt in (free -> digest), hit queues
        notifier.set_pref(card_id, "alice", enabled=True, mode=MODE_DIGEST)
        n = notifier.notify(card_id, "alice", "hit", now=1001.0)
        assert n is not None and n.mode == MODE_DIGEST
        assert notifier.get_notif(n.notif_id).card_id == card_id   # read-back
    finally:
        notifier.close()
        cards.close()
        ledger.close()
