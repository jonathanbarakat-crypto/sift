"""v4.1 rich signal-event package (sift/cards/events.py) — tests.

Covers (per the task):
- versioned payload schema (v1): keys, ts UTC format, magnitude-only fields;
- no-sign / no-symbol invariant: negative statistic, forbidden keys, and
  ticker-shaped tokens all raise NotifyError and the event is NOT delivered;
  params_json symbols never leak into the payload (ledger stage rows only);
- banned-phrase grep over EVERY copy string built by copy_for_event (hit
  gloss incl. magnitude, minted, decay, digest) via contains_banned;
- digest vs live rate caps through emit_hit (free -> digest, live needs the
  paid entitlement, capped per card per day);
- quiet hours (incl. wrap-around) gating channel delivery;
- RSS auth token issue/resolve/revoke; webhook register + HMAC sign/verify
  (+ revoke);
- idempotent delivery read-back (deliver_due_channels twice -> nothing new);
- persistence/health facts from ledger events only (hits_trailing_12m,
  median_hold_days, half_life_days, health_ratio_trailing_12m, tier);
- cards_fired_today anonymized aggregate count;
- flag-off fail-closed (NotifyError(flag_off) on mutations, reads stay);
- non-owner cannot emit (cards non-transferable).
"""

import json

import pytest

from sift import flags
from sift.cards import CardError, CardRegistry
from sift.cards.events import (
    CHANNEL_API,
    CHANNEL_EMAIL,
    CHANNEL_RSS,
    CHANNELS,
    FORBIDDEN_PAYLOAD_KEYS,
    HIT_EVENT_TYPE,
    MINT_BASELINE_EVENT,
    PAYLOAD_SCHEMA_VERSION,
    SignalEventService,
    build_event_payload,
    cards_fired_today,
    cards_fired_today_ledger,
    health_facts,
    in_quiet_hours,
    persistence_facts,
    utc_ts,
    verify_webhook_signature,
    webhook_signature,
)
from sift.cards.notify import (
    BANNED_PHRASES,
    BANNED_WORDS,
    DAY,
    LIVE_CAP_PER_CARD_PER_DAY,
    LIVE_ENTITLEMENT,
    MODE_DIGEST,
    MODE_LIVE,
    Notifier,
    NotifyError,
    contains_banned,
    copy_for_event,
)
from sift.registry import Ledger
from tests.cards.test_minting import SECRET


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


@pytest.fixture()
def svc(notifier, ledger, tmp_path):
    s = SignalEventService(notifier, ledger, tmp_path / "signal.db")
    yield s
    s.close()


def _mint(cards, handle="alice", tier_stats=(0.995, 4.5, 0.005), trial="t1"):
    dsr, t_net, p = tier_stats
    return cards.mint(
        trial_id=trial, family="pairs", scope_id="free_5y_top1k",
        config_hash=f"h-{trial}", data_vintage_hash="v1",
        survivor_class="validated-green", owner_handle=handle,
        dsr=dsr, t_net_full=t_net, p_perm=p,
    ).card_id


def _register_trial(ledger, trial_id="t1", params=None):
    """Register a trial so ledger stage/event writes have a FK target.
    params deliberately contain a SYMBOL to prove it never leaks."""
    params = params or {"symbols": ["AAPL", "MSFT"], "side": "long"}
    return ledger.register("pairs", "free_5y_top1k", params, "v1",
                           discoverer_handle="alice").trial_id


T0 = 1_800_000_000.0  # fixed "now" for deterministic tests


# -- payload schema (versioned) ---------------------------------------------------

def test_payload_schema_v1_keys_and_ts(svc, cards, ledger):
    trial = _register_trial(ledger, "t1")
    card_id = _mint(cards, trial=trial)
    card = cards.get_card(card_id)
    p = build_event_payload(card=card, statistic=2.34, threshold=2.0,
                            now=T0, ledger=ledger, notifier=svc.notifier,
                            handle="alice")
    assert p["schema"] == PAYLOAD_SCHEMA_VERSION == "v1"
    assert p["event"] == "hit"
    assert p["card_id"] == card_id
    assert p["family"] == "pairs"
    assert p["scope_id"] == "free_5y_top1k"
    assert p["tier"] == card.tier
    assert p["ts"] == utc_ts(T0) and p["ts"].endswith("Z")
    assert p["statistic"] == 2.34 and p["threshold"] == 2.0
    # persistence + health + aggregate context present
    for k in ("hits_trailing_12m", "median_hold_days", "half_life_days",
              "health_ratio_trailing_12m", "cards_fired_today", "gloss"):
        assert k in p, k
    # JSON-serializable, stable versioned
    assert json.loads(json.dumps(p))["schema"] == "v1"


def test_payload_magnitude_only_no_sign_field(svc, cards, ledger):
    trial = _register_trial(ledger, "t1")
    card_id = _mint(cards, trial=trial)
    card = cards.get_card(card_id)
    p = build_event_payload(card=card, statistic=3.1, threshold=2.0,
                            now=T0, ledger=ledger)
    assert p["statistic"] >= 0
    low_keys = {k.lower() for k in p}
    for bad in FORBIDDEN_PAYLOAD_KEYS:
        assert bad not in low_keys
    # no direction-ish keys at all
    for k in low_keys:
        assert "sign" not in k and "side" not in k and "direction" not in k


# -- no-sign / no-symbol invariant (fail closed) ------------------------------------

def test_negative_statistic_refused_and_not_delivered(svc, cards, ledger):
    trial = _register_trial(ledger, "t1")
    card_id = _mint(cards, trial=trial)
    svc.notifier.set_pref(card_id, "alice", enabled=True, mode=MODE_DIGEST)
    with pytest.raises(NotifyError):
        svc.emit_hit(card_id, "alice", statistic=-2.5, threshold=2.0, now=T0)
    # fail closed: nothing enqueued, no ledger hit event
    assert svc.notifier.log_for_card(card_id) == []
    assert ledger.events(HIT_EVENT_TYPE) == []


def test_forbidden_key_refused(svc, cards, ledger):
    trial = _register_trial(ledger, "t1")
    card_id = _mint(cards, trial=trial)
    card = cards.get_card(card_id)
    for bad in ("symbol", "direction", "side", "pnl", "returns"):
        p = build_event_payload(card=card, statistic=2.0, threshold=2.0,
                                now=T0, ledger=ledger)
        p[bad] = "X"
        with pytest.raises(NotifyError):
            from sift.cards.events import _guard_payload
            _guard_payload(p)


def test_ticker_shaped_token_refused(svc, cards, ledger):
    trial = _register_trial(ledger, "t1")
    card_id = _mint(cards, trial=trial)
    card = cards.get_card(card_id)
    from sift.cards.events import _guard_payload
    p = build_event_payload(card=card, statistic=2.0, threshold=2.0,
                            now=T0, ledger=ledger)
    p["note"] = "watch AAPL now"          # ticker-shaped token
    with pytest.raises(NotifyError):
        _guard_payload(p)


def test_params_json_symbols_never_in_payload(svc, cards, ledger):
    """The trial's params_json contains symbols + a side; the payload must
    be built from ledger stage rows / events only and contain neither."""
    trial = _register_trial(ledger, "t1",
                            params={"symbols": ["AAPL", "MSFT"], "side": "long"})
    card_id = _mint(cards, trial=trial)
    card = cards.get_card(card_id)
    p = build_event_payload(card=card, statistic=2.0, threshold=2.0,
                            now=T0, ledger=ledger, notifier=svc.notifier,
                            handle="alice")
    blob = json.dumps(p)
    assert "AAPL" not in blob and "MSFT" not in blob
    assert "params" not in blob.lower()
    low_keys = {k.lower() for k in p}
    for bad in FORBIDDEN_PAYLOAD_KEYS:
        assert bad not in low_keys


# -- persistence / health facts from the ledger only ---------------------------------

def test_persistence_facts_from_ledger_events(svc, cards, ledger):
    trial = _register_trial(ledger, "t1")
    card_id = _mint(cards, trial=trial)
    card = cards.get_card(card_id)
    # mint baseline: half_life + baseline hit rate
    ledger.record_event(
        MINT_BASELINE_EVENT,
        {"card_id": card_id, "trial_id": trial,
         "half_life_days": 12.5, "baseline_hits_per_day": 0.01},
        trial_id=trial, now=T0 - 400 * DAY,
    )
    # three hits in the trailing 12m at known ages
    for age_days in (10, 30, 100):
        ledger.record_event(
            HIT_EVENT_TYPE,
            {"card_id": card_id, "trial_id": trial, "statistic": 2.1,
             "threshold": 2.0},
            trial_id=trial, now=T0 - age_days * DAY,
        )
    # one OLD hit outside the trailing 12m (must not count)
    ledger.record_event(HIT_EVENT_TYPE, {"card_id": card_id, "trial_id": trial},
                        trial_id=trial, now=T0 - 400 * DAY)
    pf = persistence_facts(ledger, card, now=T0)
    assert pf["hits_trailing_12m"] == 3
    assert pf["median_hold_days"] == 30.0          # median of (10, 30, 100)
    assert pf["half_life_days"] == 12.5
    hf = health_facts(ledger, card, now=T0)
    # baseline 0.01/day * 365d = 3.65 expected; 3 hits -> ratio 3/3.65
    assert hf["health_ratio_trailing_12m"] == round(3 / 3.65, 6)
    assert hf["tier"] == card.tier


def test_health_ratio_none_without_baseline(svc, cards, ledger):
    trial = _register_trial(ledger, "t1")
    card_id = _mint(cards, trial=trial)
    card = cards.get_card(card_id)
    hf = health_facts(ledger, card, now=T0)
    assert hf["health_ratio_trailing_12m"] is None   # never fabricated


# -- aggregate context ---------------------------------------------------------------

def test_cards_fired_today_anonymized_count(svc, cards, ledger):
    t1 = _register_trial(ledger, "t1")
    t2 = _register_trial(ledger, "t2", params={"x": 1})
    c1 = _mint(cards, trial=t1)
    c2 = _mint(cards, trial=t2, tier_stats=(0.96, 3.5, 0.02))
    svc.notifier.set_pref(c1, "alice", enabled=True)
    svc.notifier.set_pref(c2, "alice", enabled=True)
    svc.emit_hit(c1, "alice", statistic=2.1, threshold=2.0, now=T0)
    svc.emit_hit(c1, "alice", statistic=2.2, threshold=2.0, now=T0 + 60)
    svc.emit_hit(c2, "alice", statistic=2.3, threshold=2.0, now=T0 + 120)
    # count is per CARD (distinct), not per hit
    assert cards_fired_today_ledger(ledger, cards, "alice", now=T0 + 200) == 2
    # and it rides in the emitted payload
    n = svc.notifier.log_for_card(c2)[-1]
    assert n.payload["cards_fired_today"] == 2
    # a hit yesterday does not count today
    assert cards_fired_today_ledger(ledger, cards, "alice",
                                    now=T0 + DAY + 1) == 0
    # notifier-side variant agrees
    assert cards_fired_today(svc.notifier, "alice", now=T0 + 200) == 2


# -- emit_hit: opt-in, caps, digest vs live -------------------------------------------

def test_emit_hit_opt_in_and_free_tier_digest(svc, cards, ledger):
    trial = _register_trial(ledger, "t1")
    card_id = _mint(cards, trial=trial)
    # default off -> nothing
    assert svc.emit_hit(card_id, "alice", statistic=2.0, threshold=2.0,
                        now=T0) is None
    svc.notifier.set_pref(card_id, "alice", enabled=True, mode=MODE_LIVE,
                          entitlement=None)          # free -> digest
    n = svc.emit_hit(card_id, "alice", statistic=2.0, threshold=2.0, now=T0)
    assert n is not None and n.mode == MODE_DIGEST
    assert n.payload["schema"] == "v1"
    assert n.payload["gloss"]                      # built in copy_for_event only
    # read-back
    rb = svc.notifier.get_notif(n.notif_id)
    assert rb.payload["statistic"] == 2.0


def test_emit_hit_live_cap_enforced(svc, cards, ledger):
    trial = _register_trial(ledger, "t1")
    card_id = _mint(cards, trial=trial)
    svc.notifier.set_pref(card_id, "alice", enabled=True, mode=MODE_LIVE,
                          entitlement=LIVE_ENTITLEMENT)
    sent = 0
    for i in range(LIVE_CAP_PER_CARD_PER_DAY + 3):
        n = svc.emit_hit(card_id, "alice", statistic=2.0 + i * 0.1,
                         threshold=2.0, entitlement=LIVE_ENTITLEMENT,
                         now=T0 + i)
        sent += n is not None
    assert sent == LIVE_CAP_PER_CARD_PER_DAY
    # cap resets the next day
    n = svc.emit_hit(card_id, "alice", statistic=2.0, threshold=2.0,
                     entitlement=LIVE_ENTITLEMENT, now=T0 + DAY + 1)
    assert n is not None


def test_emit_hit_non_owner_refused(svc, cards, ledger):
    trial = _register_trial(ledger, "t1")
    card_id = _mint(cards, handle="alice", trial=trial)
    svc.notifier.set_pref(card_id, "alice", enabled=True)
    with pytest.raises(CardError):
        svc.emit_hit(card_id, "eve", statistic=2.0, threshold=2.0, now=T0)


# -- copy discipline: banned-phrase grep over EVERY copy string ------------------------

def test_banned_phrase_grep_covers_every_copy_string(svc, cards, ledger):
    trial = _register_trial(ledger, "t1")
    card_id = _mint(cards, trial=trial)
    card = cards.get_card(card_id)
    details = {"from_tier": "gold", "to_tier": "silver", "hit_count": 3,
               "statistic": 2.5, "threshold": 2.0}
    for event in ("hit", "minted", "decay", "digest"):
        text = copy_for_event(event, card, details)
        assert contains_banned(text) is None
        for phrase in BANNED_PHRASES:
            assert phrase not in text.lower()
        for w in BANNED_WORDS:
            import re as _re
            assert not _re.search(rf"\b{w}\b", text.lower())
        assert "not a recommendation" in text.lower()
    # the gloss strings actually used in payloads pass too
    p = build_event_payload(card=card, statistic=2.5, threshold=2.0,
                            now=T0, ledger=ledger)
    assert contains_banned(p["gloss"]) is None
    # fail-closed helper detects each banned class ...
    for probe in ("you should buy now", "SELL", "go long", "short it",
                  "guaranteed returns", "no p&l", "profit", "more value",
                  "act now", "countdown", "risk-free"):
        assert contains_banned(probe) is not None, probe
    # ... and does NOT false-positive on legitimate prose
    assert contains_banned("no longer supports the tier") is None
    assert contains_banned("the relationship along its range") is None


def test_copy_for_event_unknown_event_raises(svc, cards, ledger):
    trial = _register_trial(ledger, "t1")
    card_id = _mint(cards, trial=trial)
    card = cards.get_card(card_id)
    with pytest.raises(NotifyError):
        copy_for_event("warp-drive", card)


# -- channels: per-card pref, quiet hours ----------------------------------------------

def test_channel_pref_default_and_set(svc, cards, ledger):
    trial = _register_trial(ledger, "t1")
    card_id = _mint(cards, trial=trial)
    pref = svc.get_channel(card_id, "alice")
    assert pref.channel == CHANNEL_EMAIL and pref.quiet_start is None
    for ch in CHANNELS:
        pref = svc.set_channel(card_id, "alice", ch, now=T0)
        assert pref.channel == ch
        assert svc.get_channel(card_id, "alice").channel == ch   # read-back
    with pytest.raises(NotifyError):
        svc.set_channel(card_id, "alice", "carrier-pigeon", now=T0)
    with pytest.raises(NotifyError):
        svc.set_channel(card_id, "alice", CHANNEL_EMAIL,
                        quiet_start=DAY, now=T0)   # out of range


def test_quiet_hours_window_and_wraparound():
    # 22:00-07:00 UTC wraps midnight
    t_23 = T0 - (T0 % DAY) + 23 * 3600
    t_03 = T0 - (T0 % DAY) + 3 * 3600
    t_12 = T0 - (T0 % DAY) + 12 * 3600
    assert in_quiet_hours(t_23, 22 * 3600, 7 * 3600) is True
    assert in_quiet_hours(t_03, 22 * 3600, 7 * 3600) is True
    assert in_quiet_hours(t_12, 22 * 3600, 7 * 3600) is False
    # no quiet hours configured -> never quiet
    assert in_quiet_hours(t_23, None, None) is False
    # non-wrapping window 13:00-14:00
    assert in_quiet_hours(t_12 + 3600, 13 * 3600, 14 * 3600) is True
    assert in_quiet_hours(t_12, 13 * 3600, 14 * 3600) is False


def test_delivery_respects_quiet_hours_and_is_idempotent(svc, cards, ledger):
    trial = _register_trial(ledger, "t1")
    card_id = _mint(cards, trial=trial)
    svc.notifier.set_pref(card_id, "alice", enabled=True, mode=MODE_LIVE,
                          entitlement=LIVE_ENTITLEMENT)
    svc.set_channel(card_id, "alice", CHANNEL_EMAIL,
                    quiet_start=22 * 3600, quiet_end=7 * 3600, now=T0)
    n = svc.emit_hit(card_id, "alice", statistic=2.0, threshold=2.0,
                     entitlement=LIVE_ENTITLEMENT, now=T0)
    assert n is not None
    # 23:00 UTC -> inside quiet hours: nothing delivered
    t_quiet = T0 - (T0 % DAY) + 23 * 3600
    assert svc.due_now(card_id, "alice", now=t_quiet) is False
    assert svc.deliver_due_channels("alice", now=t_quiet) == []
    assert svc.notifier.get_notif(n.notif_id).state == "queued"
    # 12:00 UTC -> due: delivered once, read-back
    t_due = T0 - (T0 % DAY) + 12 * 3600
    assert svc.due_now(card_id, "alice", now=t_due) is True
    m = svc.deliver_due_channels("alice", now=t_due)
    assert len(m) == 1 and m[0]["notif_id"] == n.notif_id
    assert m[0]["state"] == "delivered" and m[0]["channel"] == CHANNEL_EMAIL
    assert svc.notifier.get_notif(n.notif_id).state == "delivered"
    # idempotent: second pass delivers nothing new
    assert svc.deliver_due_channels("alice", now=t_due + 60) == []


# -- RSS auth + webhook HMAC -------------------------------------------------------------

def test_rss_token_issue_resolve_revoke(svc):
    tok = svc.issue_rss_token("alice", now=T0)
    assert tok.token.startswith("rss_") and tok.revoked is False
    assert svc.owner_for_rss_token(tok.token) == "alice"      # auth ok
    assert svc.get_rss_token(tok.token).owner == "alice"       # read-back
    rb = svc.revoke_rss_token(tok.token, "alice", now=T0 + 1)
    assert rb.revoked is True
    assert svc.owner_for_rss_token(tok.token) is None          # revoked -> no auth
    with pytest.raises(NotifyError):
        svc.revoke_rss_token(tok.token, "eve")                 # not the owner
    assert len(svc.rss_tokens_for_owner("alice")) == 1


def test_webhook_register_sign_verify_revoke(svc):
    w = svc.register_webhook("alice", "https://hooks.example.com/sift",
                             now=T0)
    assert w.active and w.url.startswith("https://")
    assert svc.get_webhook(w.hook_id).owner == "alice"          # read-back
    secret = svc.webhook_secret(w.hook_id, "alice")
    assert secret and secret not in json.dumps(w.__dict__)
    body = json.dumps({"event": "hit", "card_id": "card-x"}, sort_keys=True)
    sig = webhook_signature(secret, body)
    assert sig.startswith("sha256=")
    assert verify_webhook_signature(secret, body, sig) is True
    assert verify_webhook_signature(secret, body + " ", sig) is False
    assert verify_webhook_signature(secret, body, "sha256=deadbeef") is False
    assert verify_webhook_signature("wrong", body, sig) is False
    # revoke
    rb = svc.revoke_webhook(w.hook_id, "alice", now=T0 + 5)
    assert rb.active is False
    assert svc.active_webhooks("alice") == []
    with pytest.raises(NotifyError):
        svc.revoke_webhook(w.hook_id, "eve")                    # not the owner
    with pytest.raises(NotifyError):
        svc.register_webhook("alice", "ftp://nope", now=T0)     # bad scheme


def test_api_channel_manifest_carries_signed_webhooks(svc, cards, ledger):
    trial = _register_trial(ledger, "t1")
    card_id = _mint(cards, trial=trial)
    svc.notifier.set_pref(card_id, "alice", enabled=True, mode=MODE_LIVE,
                          entitlement=LIVE_ENTITLEMENT)
    w = svc.register_webhook("alice", "https://hooks.example.com/sift", now=T0)
    svc.set_channel(card_id, "alice", CHANNEL_API, now=T0)
    svc.emit_hit(card_id, "alice", statistic=2.0, threshold=2.0,
                     entitlement=LIVE_ENTITLEMENT, now=T0)
    m = svc.deliver_due_channels("alice", now=T0 + 1)
    assert len(m) == 1
    assert m[0]["channel"] == CHANNEL_API
    hooks = m[0]["webhooks"]
    assert len(hooks) == 1 and hooks[0]["hook_id"] == w.hook_id
    body = json.dumps(m[0]["payload"], sort_keys=True, separators=(",", ":"),
                      default=str)
    assert verify_webhook_signature(
        svc.webhook_secret(w.hook_id, "alice"), body, hooks[0]["signature"])


def test_rss_channel_manifest_carries_token(svc, cards, ledger):
    trial = _register_trial(ledger, "t1")
    card_id = _mint(cards, trial=trial)
    svc.notifier.set_pref(card_id, "alice", enabled=True, mode=MODE_LIVE,
                          entitlement=LIVE_ENTITLEMENT)
    tok = svc.issue_rss_token("alice", now=T0)
    svc.set_channel(card_id, "alice", CHANNEL_RSS, now=T0)
    svc.emit_hit(card_id, "alice", statistic=2.0, threshold=2.0,
                 entitlement=LIVE_ENTITLEMENT, now=T0)
    m = svc.deliver_due_channels("alice", now=T0 + 1)
    assert m[0]["channel"] == CHANNEL_RSS and m[0]["rss_token"] == tok.token
    # revoked token -> feed auth fails
    svc.revoke_rss_token(tok.token, "alice", now=T0 + 2)
    assert svc.owner_for_rss_token(tok.token) is None


# -- flag off ----------------------------------------------------------------------------

def test_flag_off_fail_closed_reads_stay(svc, cards, ledger):
    trial = _register_trial(ledger, "t1")
    card_id = _mint(cards, trial=trial)
    svc.notifier.set_pref(card_id, "alice", enabled=True)
    svc.set_channel(card_id, "alice", CHANNEL_EMAIL, now=T0)
    tok = svc.issue_rss_token("alice", now=T0)
    n = svc.emit_hit(card_id, "alice", statistic=2.0, threshold=2.0, now=T0)
    assert n is not None
    flags.set_flag("cards_v4_1", False)
    with pytest.raises(NotifyError) as ei:
        svc.set_channel(card_id, "alice", CHANNEL_RSS)
    assert "flag_off" in str(ei.value)
    with pytest.raises(NotifyError):
        svc.emit_hit(card_id, "alice", statistic=2.0, threshold=2.0)
    with pytest.raises(NotifyError):
        svc.issue_rss_token("alice")
    with pytest.raises(NotifyError):
        svc.register_webhook("alice", "https://x.example.com")
    with pytest.raises(NotifyError):
        svc.deliver_due_channels("alice")
    # reads stay available (kill-switch never hides data)
    assert svc.get_channel(card_id, "alice").channel == CHANNEL_EMAIL
    assert svc.get_rss_token(tok.token).owner == "alice"
    assert len(svc.notifier.log_for_card(card_id)) == 1
    flags.set_flag("cards_v4_1", True)
    assert svc.emit_hit(card_id, "alice", statistic=2.1, threshold=2.0,
                        now=T0 + 10) is not None


def test_flag_off_env_kill_switch(svc, monkeypatch):
    monkeypatch.setenv("SIFT_CARDS_V4_1", "0")
    flags.reset()
    with pytest.raises(NotifyError):
        svc.issue_rss_token("alice")
    monkeypatch.delenv("SIFT_CARDS_V4_1")
    flags.reset()
