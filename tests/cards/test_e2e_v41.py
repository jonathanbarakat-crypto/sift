"""v4.1 cards (8/8) — end-to-end integration suite for the whole card layer
behind `cards_v4_1`.

Rebased onto current main (post pull/18 + pull/19): the four files that the
stale PR branch re-implemented in parallel (share.py / vault.py / notify.py /
decay.py) are RESOLVED TO MAIN here, and this suite imports MAIN's class
names (ShareService.request_share, vault.share_slots, Notifier,
RegistryDecayJob). What this suite adds on top of the merged per-mechanic
tests (test_share.py / test_notify.py / test_decay.py / test_vault.py) is the
END-TO-END wiring through the CardLayer composition root + the flag
fail-closed matrix + the env kill-switch.

Covers, in ONE path:
  mint (validated-green survivor via the gauntlet runner + CardMinter)
  -> vault admit (capacity by tier)
  -> face render (anonymized, watermarked, ticker-guarded)
  -> share grant via request_share (+1 permanent vault slot, verified post,
     append-only audit log, vault read-back)
  -> badge award (kill_rate etc. from the ledger, guard-checked payload)
  -> notification (per-card opt-in via set_pref, digest on free tier,
     rate-capped, digest_batch/deliver_due)
  -> decay downgrade (RegistryDecayJob.evaluate_card, amber down-only path;
     public explanation, ledger card_decay event, notification hook)

Flag OFF path: the same calls raise each module's fail-closed error type
(ShareError/BadgeError/NotifyError/DecayError/LayerError with flag_off) and
touch NO tables (row counts unchanged in every card-layer DB). The mint
path with the flag off writes no card (CardMinter returns None).

Kill-switch: SIFT_CARDS_V4_1=0 in the env disables the whole layer
(flags.py reads the env at import/reset); unknown flags fail closed
(is_enabled -> False, set_flag raises).

Beta-cohort default: cards_v4_1 defaults ON (no env set).
"""

import json
import os
import sqlite3

import numpy as np
import pytest

from sift import flags
from sift.cards import CardRegistry
from sift.cards.badges import BadgeError, BadgeService
from sift.cards.decay import DecayError, RegistryDecayJob
from sift.cards.face import CardFaceRenderer, FaceError
from sift.cards.metrics import MetricError, Metrics
from sift.cards.minting import CardMinter
from sift.cards.notify import Notifier, NotifyError
from sift.cards.service import CardLayer, LayerError
from sift.cards.share import MAX_SHARE_SLOTS, ShareError, ShareService
from sift.cards.vault import Vault, VaultError
from sift.gauntlet.runner import GauntletRunner
from sift.registry import Ledger
from tests.cards.test_minting import SECRET, _run_validated_green

HANDLE = "alice"
NOW = 1_800_000_000.0  # fixed clock for deterministic assertions
OLD = NOW - 30 * 24 * 3600          # 30d ago (old enough for the age rule)


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
def layer(ledger, tmp_path):
    """The whole v4.1 card layer wired end-to-end (composition root)."""
    lay = CardLayer(ledger, tmp_path / "cards", card_secret=SECRET)
    yield lay
    lay.close()


def _row_counts(db_path) -> dict:
    """Table -> row count snapshot, used to prove flag-OFF touches nothing."""
    con = sqlite3.connect(str(db_path))
    try:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        # table names come from sqlite_master (trusted), never from user input
        return {t: con.execute("SELECT COUNT(*) FROM " + t).fetchone()[0]
                for t in tables}
    finally:
        con.close()


def _layer_db_counts(layer, tmp_db_dir) -> dict:
    out = {}
    for name in ("cards.db", "vault.db", "shares.db", "badges.db",
                 "notify.db"):
        p = tmp_db_dir / name
        if p.exists():
            out[name] = _row_counts(p)
    return out


def _run_and_mint(layer, handle=HANDLE, fixture="F-PAIR-FULL"):
    """Full gauntlet run on the planted full-window pair -> validated-green
    -> exactly one card minted for `handle` (flag ON)."""
    runner = GauntletRunner(layer.ledger, seed=7, card_minter=layer.minter)
    # reuse the shared validated-green helper against this layer's stores
    verdict, trial = _run_validated_green(layer.ledger, layer.cards,
                                           handle=handle, fixture=fixture)
    assert verdict["survivor_class"] == "validated-green"
    assert verdict["card_id"] is not None
    return verdict, trial


def _ready_account(share, handle, now=NOW, runs=3, age_s=30 * 24 * 3600,
                   kyc=True):
    """Register the share-account prerequisites (email+phone OTP, age,
    completed runs, KYC-lite) — main's ShareService API."""
    share.register_account(
        handle, now=now, email_verified=True, phone_otp_verified=True,
        kyc_lite=kyc, created_at=now - age_s, completed_runs=runs,
    )


def _good_post(share, card_id, handle=HANDLE):
    """Inject a live-post fetcher returning a post that contains the
    anonymized card link + the attribution tag (#SIFT)."""
    share.post_fetcher = lambda url: (
        f"my post {url} — check https://sift.example/cards/{card_id}/face #SIFT"
    )


def _mint_card(layer, handle, tag, trial=None, stats=(0.995, 4.5, 0.005)):
    """Mint one distinct card for `handle` (distinct config hash per tag)."""
    if trial is None:
        trial = layer.ledger.get(layer.cards.cards_for_owner(handle)[0].trial_id)
    return layer.cards.mint(
        trial_id=trial.trial_id, family=trial.family, scope_id=trial.scope_id,
        config_hash=f"h-{tag}", data_vintage_hash=trial.data_vintage_hash,
        survivor_class="validated-green", owner_handle=handle,
        dsr=stats[0], t_net_full=stats[1], p_perm=stats[2],
    ).card_id


# ============================================================================
# FLAG ON — the full happy path, end to end
# ============================================================================

def test_full_path_mint_vault_face_share_badge_notify_decay(layer, tmp_path):
    # -- 1. mint (gauntlet -> validated-green -> card) -------------------------
    verdict, trial = _run_and_mint(layer)
    card_id = verdict["card_id"]
    card = layer.cards.get_card(card_id)                    # read-back
    assert card.owner_handle == HANDLE and card.kind == "regular"

    # -- 2. vault admit (capacity by tier, read-back) ---------------------------
    entry = layer.vault.admit(card_id, HANDLE, now=NOW)
    assert entry.state == "vault"
    assert [c.card_id for c in layer.vault.vault(HANDLE)] == [card_id]

    # -- 3. face render (anonymized, watermarked, no tickers) --------------------
    face = layer.face.render(card_id, viewer_handle=HANDLE, now=NOW)
    assert face["watermark"]["handle"] == HANDLE
    assert face["survivor_class"] == "validated-green"
    blob = json.dumps(face, default=str)
    assert "AAPL" not in blob and "MSFT" not in blob

    # -- 4. share grant (+1 permanent slot, verified post, audit) --------------
    _ready_account(layer.shares, HANDLE)
    _good_post(layer.shares, card_id)
    res = layer.shares.request_share(
        HANDLE, card_id, post_url="https://social.example/post/1",
        device_id="dev-1", ip="1.2.3.4", asn="AS1", now=NOW,
    )
    assert res["granted"] is True
    # vault read-back: capacity grew by exactly 1 via the authoritative
    # vault_owner_slots table
    base = layer.vault.base_capacity(card.tier)
    assert layer.vault.share_slots(HANDLE) == 1
    assert layer.vault.effective_capacity(HANDLE, card.tier) == base + 1
    # audit log recorded the grant (append-only share_grant)
    log = layer.shares.grants_for(HANDLE)
    assert any(r.outcome == "granted" and r.verified == 1 for r in log)

    # -- 5. badge award (kill_rate from the ledger, guard-checked) --------------
    badge = layer.badges.award(HANDLE, "kill_rate", opt_in=True, now=NOW)
    assert badge.payload["badge"] == "kill_rate"
    assert badge.payload["trials"] >= 1
    # opt-in respected on read
    assert [b.badge for b in layer.badges.badges_for(HANDLE)] == ["kill_rate"]

    # -- 6. notification (per-card opt-in; free tier -> digest) -----------------
    pref = layer.notify.set_pref(card_id, HANDLE, enabled=True, mode="live",
                                 entitlement="free", now=NOW)
    assert pref.mode == "digest"          # free can never go live
    assert layer.notify.effective_mode(HANDLE, "live", "free") == "digest"
    ev = layer.notify.notify(
        card_id, HANDLE, "hit",
        {"signal": "hit", "card_id": card_id}, entitlement="free", now=NOW,
    )
    assert ev is not None and ev.mode == "digest" and ev.event_kind == "hit"
    delivered = layer.notify.digest_batch(HANDLE, entitlement="free",
                                          now=NOW + 3600)
    assert len(delivered) == 1
    # the batched hit row is marked delivered (append-only log keeps the row)
    assert layer.notify.get_notif(ev.notif_id).state == "delivered"

    # -- 7. decay downgrade (registry re-evaluation; amber downgrade path) ---
    # force the ledger's recorded stats to weaken so re-derivation downgrades
    layer.ledger.con.execute(
        "UPDATE stage_result SET dsr=0.90 WHERE trial_id=? AND stage='s3'",
        (trial.trial_id,),
    )
    layer.ledger.con.execute(
        "UPDATE stage_result SET statistic=1.0 WHERE trial_id=? AND stage='s5'",
        (trial.trial_id,),
    )
    layer.ledger.con.commit()
    result = layer.decay.evaluate_card(card, now=NOW + 7200)
    assert result.downgraded is True
    assert result.to_tier == "bronze" and result.from_tier != "bronze"
    # read-back: card row shows the downgrade + decayed_from
    after = layer.cards.get_card(card_id)
    assert after.tier == "bronze" and after.decayed_from == result.from_tier
    # ledger card_decay event appended (pull/19 requirement)
    decay_events = layer.ledger.events("card_decay")
    assert any(e["payload_json"] and json.loads(e["payload_json"])["card_id"] == card_id
               for e in decay_events)
    # public explanation present, decay language only (no appreciation)
    assert result.public_text
    low = result.public_text.lower()
    assert "never up" in low or "down" in low
    assert "appreciates" not in low
    # notification hook fired through the per-card channel (opt-in pref on)
    decay_notifs = [n for n in layer.notify.log_for_owner(HANDLE)
                    if n.event_kind == "lifecycle"]
    assert len(decay_notifs) >= 1
    # a second pass is a no-op (no proposal; upgrades need live-window rules)
    again = layer.decay.evaluate_card(after, now=NOW + 9000)
    assert again.downgraded is False and again.to_tier == "bronze"

    # -- metrics hooks all compute ----------------------------------------------
    split = layer.metrics.d90_retention_split(HANDLE, now=NOW + 90 * 86400)
    assert {s.cohort for s in split} == {"card_holders", "non_card_holders"}
    assert layer.metrics.ghost_ratio(HANDLE).total_cards == 1
    assert layer.metrics.vault_turnover(handle=HANDLE,
                                        now=NOW).active_vault_cards == 1
    sgr = layer.metrics.share_grant_d30_retention(HANDLE, now=NOW + 30 * 86400)
    assert sgr.granted_users == 1
    ahr = layer.metrics.anchored_vs_unanchored_hit_rate(HANDLE)
    assert ahr.unanchored_trials >= 1


# ============================================================================
# FLAG OFF — fail closed everywhere, no tables touched
# ============================================================================

def test_flag_off_fails_closed_and_touches_no_tables(layer, tmp_path):
    # mint + admit first (flag ON) so we have a card to poke at
    verdict, _ = _run_and_mint(layer)
    card_id = verdict["card_id"]
    layer.vault.admit(card_id, HANDLE, now=NOW)
    layer.notify.set_pref(card_id, HANDLE, enabled=True, mode="digest", now=NOW)

    db_dir = tmp_path / "cards"
    before = _layer_db_counts(layer, db_dir)

    flags.set_flag("cards_v4_1", False)          # global kill-switch
    try:
        # facade guard
        with pytest.raises(LayerError):
            layer.require_layer(HANDLE)
        # vault mutation
        with pytest.raises(VaultError):
            layer.vault.discard(card_id, HANDLE)
        # face render
        with pytest.raises(FaceError):
            layer.face.render(card_id, viewer_handle=HANDLE)
        # share grant (main's request_share signature)
        with pytest.raises(ShareError):
            layer.shares.request_share(HANDLE, card_id,
                                       post_url="https://x.example/p",
                                       device_id="dev-1", now=NOW)
        # badge award
        with pytest.raises(BadgeError):
            layer.badges.award(HANDLE, "kill_rate", now=NOW)
        # notification
        with pytest.raises(NotifyError):
            layer.notify.set_pref(card_id, HANDLE, enabled=True, mode="live",
                                  now=NOW)
        with pytest.raises(NotifyError):
            layer.notify.notify(card_id, HANDLE, "hit", {}, now=NOW)
        # decay
        with pytest.raises(DecayError):
            layer.decay.run_pass(now=NOW)
        # metrics
        with pytest.raises(MetricError):
            layer.metrics.ghost_ratio(HANDLE)
        # mint path: flag off -> no card minted even on a green survivor.
        # Fresh handle + distinct params (skip-list safe): reuse the shared
        # validated-green helper, which builds its own full-window pair.
        v2, _ = _run_validated_green(layer.ledger, layer.cards, handle="bob",
                                     fixture="F-PAIR-FULL-BOB")
        assert v2["survivor_class"] == "validated-green"   # gauntlet unaffected
        assert v2["card_id"] is None                        # flag off -> no mint
        assert layer.cards.cards_for_owner("bob") == []
    finally:
        flags.reset()

    after = _layer_db_counts(layer, db_dir)
    assert before == after, (
        f"flag-OFF path touched tables: "
        f"{ {k: (before[k], after[k]) for k in before if before[k] != after[k]} }"
    )
    # reads still work (kill-switch never hides existing data)
    assert [c.card_id for c in layer.vault.vault(HANDLE)] == [card_id]
    assert layer.cards.get_card(card_id).card_id == card_id


def test_env_kill_switch_disables_whole_layer(monkeypatch):
    """SIFT_CARDS_V4_1=0 in the environment disables the layer at flag-read
    time (flags.py reads the env into the default)."""
    monkeypatch.setenv("SIFT_CARDS_V4_1", "0")
    flags.reset()
    assert flags.cards_v4_1_enabled("alice") is False
    assert flags.is_enabled("cards_v4_1") is False
    # and the layer facade fails closed
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        ledger = Ledger(os.path.join(d, "ledger.db"))
        try:
            lay = CardLayer(ledger, os.path.join(d, "cards"), card_secret=SECRET)
            try:
                with pytest.raises(LayerError):
                    lay.require_layer("alice")
                assert lay.flag_status("alice")["for_handle"] is False
            finally:
                lay.close()
        finally:
            ledger.close()
    monkeypatch.delenv("SIFT_CARDS_V4_1")
    flags.reset()
    assert flags.cards_v4_1_enabled("alice") is True   # beta default ON again


def test_unknown_flag_fails_closed():
    assert flags.is_enabled("no_such_flag") is False
    assert flags.is_enabled("cards_v4_1x") is False
    with pytest.raises(ValueError):
        flags.set_flag("no_such_flag", True)
    with pytest.raises(ValueError):
        flags.disable_for_user("no_such_flag", "alice")
    assert set(flags.known_flags()) == {"cards_v4_1"}


def test_beta_cohort_default_on():
    """No env var set -> cards_v4_1 defaults ON for the beta cohort."""
    os.environ.pop("SIFT_CARDS_V4_1", None)
    flags.reset()
    assert flags.cards_v4_1_enabled("any_beta_user") is True
    assert flags.cards_v4_1_enabled() is True


def test_per_user_kill_switch(layer):
    verdict, _ = _run_and_mint(layer)
    card_id = verdict["card_id"]
    layer.vault.admit(card_id, HANDLE, now=NOW)
    flags.disable_for_user("cards_v4_1", HANDLE)
    try:
        with pytest.raises(LayerError):
            layer.require_layer(HANDLE)
        with pytest.raises(VaultError):
            layer.vault.keep(card_id, HANDLE)
        # another user is unaffected
        assert flags.cards_v4_1_enabled("bob") is True
    finally:
        flags.enable_for_user("cards_v4_1", HANDLE)
        flags.reset()


# ============================================================================
# Share-for-capacity mechanics (pull/18 API) — unit-level through the layer
# ============================================================================

def test_share_grant_uniqueness_cap_and_prereqs(layer):
    verdict, trial = _run_and_mint(layer)
    card_id = verdict["card_id"]
    layer.vault.admit(card_id, HANDLE, now=NOW)
    _ready_account(layer.shares, HANDLE)
    _good_post(layer.shares, card_id)

    res = layer.shares.request_share(
        HANDLE, card_id, post_url="https://x.example/p/1",
        device_id="dev-1", ip="1.2.3.4", asn="AS1", now=NOW,
    )
    assert res["granted"] is True
    # duplicate (handle, card) -> refuse (absolute UNIQUE(owner, card_id))
    with pytest.raises(ShareError, match="one slot per"):
        layer.shares.request_share(
            HANDLE, card_id, post_url="https://x.example/p/2",
            device_id="dev-1", ip="1.2.3.4", asn="AS1", now=NOW + 10,
        )
    assert layer.shares.share_slots(HANDLE) == 1
    # prereqs: unverified email -> refuse
    layer.shares.register_account(
        HANDLE, now=NOW, email_verified=False, phone_otp_verified=True,
        kyc_lite=True, created_at=NOW - 30 * 86400, completed_runs=3,
    )
    cid2 = _mint_card(layer, HANDLE, "prereq")
    with pytest.raises(ShareError, match="email"):
        layer.shares.request_share(
            HANDLE, cid2, post_url="https://x.example/p/x",
            device_id="dev-2", ip="1.2.3.5", asn="AS2", now=NOW + 5,
        )
    _ready_account(layer.shares, HANDLE)   # restore verified state
    # unverifiable post (no card link + attribution) -> grants nothing
    layer.shares.post_fetcher = lambda url: "no link here"
    cid3 = _mint_card(layer, HANDLE, "unverifiable")
    res3 = layer.shares.request_share(
        HANDLE, cid3, post_url="https://x.example/no-link",
        device_id="dev-3", ip="1.2.3.6", asn="AS3", now=NOW + 6,
    )
    assert res3["granted"] is False and res3["reason"] == "unverifiable_post"
    assert layer.shares.share_slots(HANDLE) == 1   # nothing granted
    # cap at +5: fresh device per additional grant so the per-device 30d
    # rate limit (3/device) does not mask the +5 account cap we are testing.
    for i in range(4):
        cid = _mint_card(layer, HANDLE, f"cap{i}")
        _good_post(layer.shares, cid)
        r = layer.shares.request_share(
            HANDLE, cid, post_url=f"https://x.example/cap{i}",
            device_id=f"dev-cap-{i}", ip=f"10.0.0.{i}", asn=f"AS-{i}",
            now=NOW + 10 + i,
        )
        assert r["granted"] is True
    assert layer.vault.share_slots(HANDLE) == MAX_SHARE_SLOTS == 5
    # 6th distinct card grant -> cap refuse
    cid6 = _mint_card(layer, HANDLE, "cap6")
    _good_post(layer.shares, cid6)
    with pytest.raises(ShareError, match="cap"):
        layer.shares.request_share(
            HANDLE, cid6, post_url="https://x.example/cap6",
            device_id="dev-cap-99", ip="10.0.0.99", asn="AS-99", now=NOW + 99,
        )
    # vault capacity never exceeds base + 5 (authoritative read-back)
    card = layer.cards.get_card(card_id)
    assert layer.vault.effective_capacity(HANDLE, card.tier) == \
        layer.vault.base_capacity(card.tier) + MAX_SHARE_SLOTS
    # abuse revoke zeroes the vault permanent slots (read-back)
    layer.shares.revoke_for_abuse(HANDLE, now=NOW + 100)
    assert layer.vault.share_slots(HANDLE) == 0
    assert layer.vault.effective_capacity(HANDLE, card.tier) == \
        layer.vault.base_capacity(card.tier)
    assert layer.shares.fraud_queue_open()


def test_share_device_rate_limit_and_cluster_flag(layer):
    verdict, _ = _run_and_mint(layer)
    t = layer.ledger.get(layer.cards.get_card(verdict["card_id"]).trial_id)
    _ready_account(layer.shares, HANDLE)
    post = lambda cid, h: f"https://s.example/{cid} card={cid} @{h} #SIFT"  # noqa: E731

    def farm_post(cid):
        layer.shares.post_fetcher = lambda url: (
            f"https://sift.example/cards/{cid}/face #SIFT")

    # 3 grants/device/30d allowed, 4th refused
    for i in range(3):
        cid = _mint_card(layer, HANDLE, f"rl{i}")
        farm_post(cid)
        layer.shares.request_share(
            HANDLE, cid, post_url=f"https://x.example/rl{i}",
            device_id="dev-9", ip="9.9.9.9", asn="AS9", now=NOW + i,
        )
    cid4 = _mint_card(layer, HANDLE, "rl3")
    farm_post(cid4)
    res = layer.shares.request_share(
        HANDLE, cid4, post_url="https://x.example/rl3",
        device_id="dev-9", ip="9.9.9.9", asn="AS9", now=NOW + 3,
    )
    assert res["granted"] is False and res["reason"] == "device_ratelimit"
    # cluster flag: >=5 distinct accounts on one device/IP -> all FLAGGED.
    # Spread farming accounts over distinct device ids on the SAME ip so
    # the cluster check (distinct accounts per ip) trips before the
    # per-device rate limit (3 grants/device).
    for j in range(5):
        h = f"farm{j}"
        layer.ledger.register("null", "free_5y_top1k", {"f": j}, "v",
                              discoverer_handle=h)
        layer.ledger.decide(layer.ledger.con.execute(
            "SELECT trial_id FROM trial WHERE discoverer_handle=?", (h,)
        ).fetchone()[0], "killed")
        layer.shares.register_account(
            h, now=NOW, email_verified=True, phone_otp_verified=True,
            kyc_lite=True, created_at=NOW - 30 * 86400, completed_runs=1,
        )
        cid = layer.cards.mint(
            trial_id=f"t-{h}", family="pairs", scope_id="free_5y_top1k",
            config_hash=f"h-{h}", data_vintage_hash="v",
            survivor_class="validated-green", owner_handle=h,
            dsr=0.99, t_net_full=4.5, p_perm=0.005,
        ).card_id
        farm_post(cid)
        if j < 4:
            r = layer.shares.request_share(
                h, cid, post_url=f"https://x.example/{h}",
                device_id=f"dev-farm-{j}", ip="8.8.8.8", asn="AS-FARM",
                now=NOW + 10 + j,
            )
            assert r["granted"] is True
        else:
            # 5th account on the same ip -> velocity cluster FLAGGED
            r = layer.shares.request_share(
                h, cid, post_url=f"https://x.example/{h}",
                device_id=f"dev-farm-{j}", ip="8.8.8.8", asn="AS-FARM",
                now=NOW + 10 + j,
            )
            assert r["granted"] is False and r["reason"] == "velocity_cluster"
    # fraud queue holds the flagged attempts (SLA-tracked)
    assert layer.shares.fraud_queue_open()


# ============================================================================
# Badges + notify + decay mechanics (quoted from the 6/8 and 7/8 cards)
# ============================================================================

def test_badge_payload_guard_rejects_pnl_language(layer):
    verdict, _ = _run_and_mint(layer)
    from sift.cards.badges import _guard_payload
    # forbidden FIELD names are refused (fail closed)
    with pytest.raises(BadgeError):
        _guard_payload({"badge": "kill_rate", "win_rate": 0.9})
    with pytest.raises(BadgeError):
        _guard_payload({"badge": "kill_rate", "nested": {"pnl": 12.3}})
    # a bare forbidden term as a leaf VALUE is refused
    with pytest.raises(BadgeError):
        _guard_payload({"badge": "kill_rate", "note": "win_rate"})
    # ...but legal disclaimer prose mentioning the words is fine
    _guard_payload({"badge": "kill_rate",
                    "note": "validation survival only — never P&L (v3 §12.4)"})
    # ALL four badge kinds can be awarded (guard must not reject its own notes)
    for b in ("kill_rate", "pre_registration_streak", "peer_retest", "scope_honesty"):
        badge = layer.badges.award(HANDLE, b, opt_in=True, now=NOW)
        assert badge.payload["badge"] == b
    # opt-out respected: award without opt-in -> not visible
    layer.badges.set_opt_in(HANDLE, "kill_rate", False, now=NOW)
    assert "kill_rate" not in [b.badge for b in layer.badges.badges_for(HANDLE)]
    assert "kill_rate" in [b.badge for b in layer.badges.all_badges(HANDLE)]
    with pytest.raises(BadgeError):
        layer.badges.award(HANDLE, "no_such_badge", now=NOW)


def test_notify_rate_caps_and_digest_batching(layer):
    verdict, _ = _run_and_mint(layer)
    card_id = verdict["card_id"]
    layer.notify.set_pref(card_id, HANDLE, enabled=True, mode="live",
                          entitlement="free", now=NOW)   # pref live...
    # ...but free tier forces digest
    # digest-mode hits queue UNCAPPED at enqueue (delivered by the capped
    # daily digest batch) — so 5 free-tier hits all queue...
    for i in range(5):
        layer.notify.notify(card_id, HANDLE, "hit", {"i": i},
                            entitlement="free", now=NOW + i)
    events = [n for n in layer.notify.log_for_card(card_id)
              if n.event_kind == "hit"]
    assert len(events) == 5
    # ...and the digest batch collapses them into ONE digest row (cap 1/day)
    d1 = layer.notify.digest_batch(HANDLE, entitlement="free", now=NOW + 200)
    assert len(d1) == 1 and d1[0].event_kind == "digest"
    d2 = layer.notify.digest_batch(HANDLE, entitlement="free", now=NOW + 300)
    assert d2 == []                       # second batch same day: capped
    # paid tier gets live push, capped at LIVE_CAP (5) per card per day.
    # NOTE: the stored pref mode stays 'digest' (free resolution at set_pref
    # time), so live delivery is requested per-call via entitlement="live".
    layer.notify.set_pref(card_id, HANDLE, enabled=True, mode="live",
                          entitlement="live", now=NOW)
    for i in range(8):
        layer.notify.notify(card_id, HANDLE, "hit", {"i": i},
                            entitlement="live", now=NOW + 1000 + i)
    push = [n for n in layer.notify.log_for_card(card_id)
            if n.mode == "live" and n.event_kind == "hit"]
    assert len(push) == 5


def test_decay_never_upgrades_and_is_public(layer):
    verdict, trial = _run_and_mint(layer)
    card_id = verdict["card_id"]
    layer.vault.admit(card_id, HANDLE, now=NOW)
    card = layer.cards.get_card(card_id)
    # strengthen the ledger stats -> re-evaluation must NOT upgrade
    layer.ledger.con.execute(
        "UPDATE stage_result SET dsr=0.999 WHERE trial_id=? AND stage='s3'",
        (trial.trial_id,),
    )
    layer.ledger.con.execute(
        "UPDATE stage_result SET statistic=9.0 WHERE trial_id=? AND stage='s5'",
        (trial.trial_id,),
    )
    layer.ledger.con.commit()
    r = layer.decay.evaluate_card(card, now=NOW + 10)
    assert r.downgraded is False
    assert layer.cards.get_card(card_id).tier == card.tier   # unchanged
    # weaken -> downgrade to bronze, public + explained, ledger event appended
    layer.ledger.con.execute(
        "UPDATE stage_result SET dsr=0.80 WHERE trial_id=? AND stage='s3'",
        (trial.trial_id,),
    )
    layer.ledger.con.commit()
    ev = layer.decay.evaluate_card(layer.cards.get_card(card_id), now=NOW + 20)
    assert ev.downgraded and ev.to_tier == "bronze"
    assert ev.public_text and "down" in ev.public_text.lower()
    assert ev.ledger_event_id is not None
    # full pass over all cards also works (run_pass)
    results = layer.decay.run_pass(now=NOW + 30)
    assert len(results) == 1 and results[0].card_id == card_id


def test_production_hard_fail_on_ephemeral_hmac_secret(monkeypatch, tmp_path):
    """The PR's __init__.py hard-fail: SIFT_ENV=production with no secret
    configured refuses to mint instead of running on the ephemeral dev
    secret."""
    monkeypatch.delenv("SIFT_CARD_HMAC_SECRET", raising=False)
    monkeypatch.delenv("SIFT_CARD_HMAC_SECRET_FILE", raising=False)
    monkeypatch.setenv("SIFT_ENV", "production")
    monkeypatch.chdir(tmp_path)     # no .card-hmac-secret file here
    from sift.cards import _default_secret
    with pytest.raises(RuntimeError, match="ephemeral"):
        _default_secret()
    # dev default: warns loudly but still returns the dev secret
    monkeypatch.setenv("SIFT_ENV", "dev")
    import io, contextlib
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        s = _default_secret()
    assert s == b"sift-dev-ephemeral-secret-not-for-production"
    assert "ephemeral" in err.getvalue()
