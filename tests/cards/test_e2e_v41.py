"""v4.1 cards (8/8) — end-to-end integration suite for the whole card layer
behind `cards_v4_1`.

Covers, in ONE path:
  mint (validated-green survivor via the gauntlet runner + CardMinter)
  -> vault admit (capacity by tier)
  -> face render (anonymized, watermarked, ticker-guarded)
  -> share grant (+1 permanent vault slot, verified post, audit log)
  -> badge award (kill_rate etc. from the ledger, guard-checked payload)
  -> notification (per-card opt-in, digest on free tier, rate-capped)
  -> decay downgrade (registry re-evaluation, downgrade-only, public
     explanation, notification hook)

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
from sift.cards.decay import DecayError, DecayService
from sift.cards.face import CardFaceRenderer, FaceError
from sift.cards.metrics import MetricError, Metrics
from sift.cards.minting import CardMinter
from sift.cards.notify import NotifyError, NotifyService
from sift.cards.service import CardLayer, LayerError
from sift.cards.share import MAX_SHARE_SLOTS, ShareError, ShareService
from sift.cards.vault import Vault, VaultError
from sift.gauntlet.runner import GauntletRunner
from sift.registry import Ledger
from tests.cards.test_minting import SECRET, _run_validated_green

HANDLE = "alice"
NOW = 1_800_000_000.0  # fixed clock for deterministic assertions


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
                 "notify.db", "decay.db"):
        out[name] = _row_counts(tmp_db_dir / name)
    return out


def _run_and_mint(layer, handle=HANDLE):
    """Full gauntlet run on the planted full-window pair -> validated-green
    -> exactly one card minted for `handle` (flag ON)."""
    runner = GauntletRunner(layer.ledger, seed=7, card_minter=layer.minter)
    # reuse the shared validated-green helper against this layer's stores
    verdict, trial = _run_validated_green(layer.ledger, layer.cards, handle=handle)
    assert verdict["survivor_class"] == "validated-green"
    assert verdict["card_id"] is not None
    return verdict, trial


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
    post = f"https://social.example/post/1 card={card_id} by @{HANDLE}"
    grant = layer.shares.grant(
        handle=HANDLE, card_id=card_id, device_id="dev-1", ip_cluster="asn-1",
        post_ref=post, account_age_s=30 * 24 * 3600,
        email_verified=True, phone_otp_verified=True, now=NOW,
    )
    assert grant.verified is True
    # vault read-back: capacity grew by exactly 1
    base = layer.vault.base_capacity(card.tier)
    assert layer.vault.effective_capacity(HANDLE, card.tier) == base + 1
    # audit log recorded the grant
    assert any(a["action"] == "grant" and a["handle"] == HANDLE
               for a in layer.shares.audit_log())

    # -- 5. badge award (kill_rate from the ledger, guard-checked) --------------
    badge = layer.badges.award(HANDLE, "kill_rate", opt_in=True, now=NOW)
    assert badge.payload["badge"] == "kill_rate"
    assert badge.payload["trials"] >= 1
    # opt-in respected on read
    assert [b.badge for b in layer.badges.badges_for(HANDLE)] == ["kill_rate"]

    # -- 6. notification (per-card opt-in; free tier -> digest) -----------------
    mode = layer.notify.set_mode(HANDLE, card_id, "live", now=NOW)
    assert mode == "live"
    # free tier can never go live -> effective mode is digest
    assert layer.notify.effective_mode(HANDLE, card_id, tier="free") == "digest"
    ev = layer.notify.notify_hit(
        handle=HANDLE, card_id=card_id, tier="free",
        payload={"kind": "hit", "card_id": card_id}, now=NOW,
    )
    assert ev is not None and ev.channel == "digest"
    delivered = layer.notify.digest_batch(HANDLE, now=NOW + 3600)
    assert len(delivered) == 1 and delivered[0].delivered_at is not None

    # -- 7. decay downgrade (registry re-evaluation; downgrade-only) ------------
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
    ev_decay = layer.decay.reevaluate(card_id, now=NOW + 7200)
    assert ev_decay is not None
    assert ev_decay.new_tier == "bronze" and ev_decay.old_tier != "bronze"
    # read-back: card row shows the downgrade + decayed_from
    after = layer.cards.get_card(card_id)
    assert after.tier == "bronze" and after.decayed_from == ev_decay.old_tier
    # public explanation present, decay language only (no "appreciates")
    assert "never appreciate" in ev_decay.explanation.lower()
    assert "appreciates" not in ev_decay.explanation.lower()
    # notification hook fired through the per-card channel
    decay_events = [e for e in layer.notify.queue_for(HANDLE) if e.kind == "decay"]
    assert len(decay_events) == 1
    # a second pass is a no-op (downgrade-only; stronger re-derivation ignored)
    assert layer.decay.reevaluate(card_id, now=NOW + 9000) is None

    # -- metrics hooks all compute ----------------------------------------------
    split = layer.metrics.d90_retention_split(HANDLE, now=NOW + 90 * 86400)
    assert {s.cohort for s in split} == {"card_holders", "non_card_holders"}
    assert layer.metrics.ghost_ratio(HANDLE).total_cards == 1
    assert layer.metrics.vault_turnover(handle=HANDLE, now=NOW).active_vault_cards == 1
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
    layer.notify.set_mode(HANDLE, card_id, "digest", now=NOW)

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
        # share grant
        with pytest.raises(ShareError):
            layer.shares.grant(
                handle=HANDLE, card_id=card_id, device_id="dev-1",
                ip_cluster="asn-1", post_ref=f"post {card_id} @{HANDLE}",
                account_age_s=999 * 86400, email_verified=True,
                phone_otp_verified=True, now=NOW,
            )
        # badge award
        with pytest.raises(BadgeError):
            layer.badges.award(HANDLE, "kill_rate", now=NOW)
        # notification
        with pytest.raises(NotifyError):
            layer.notify.set_mode(HANDLE, card_id, "live", now=NOW)
        with pytest.raises(NotifyError):
            layer.notify.notify_hit(handle=HANDLE, card_id=card_id,
                                    payload={}, now=NOW)
        # decay
        with pytest.raises(DecayError):
            layer.decay.reevaluate(card_id, now=NOW)
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
# Share-for-capacity mechanics (quoted from the 5/8 card) — unit-level
# ============================================================================

def test_share_grant_uniqueness_cap_and_prereqs(layer):
    verdict, _ = _run_and_mint(layer)
    card_id = verdict["card_id"]
    layer.vault.admit(card_id, HANDLE, now=NOW)
    post = lambda cid: f"https://social.example/p/{cid} card={cid} @{HANDLE}"  # noqa: E731

    kw = dict(handle=HANDLE, card_id=card_id, device_id="dev-1",
              ip_cluster="asn-1", account_age_s=30 * 86400,
              email_verified=True, phone_otp_verified=True, now=NOW)
    g = layer.shares.grant(post_ref=post(card_id), **kw)
    assert g.verified
    # duplicate (handle, card) -> refuse
    with pytest.raises(ShareError):
        layer.shares.grant(post_ref=post(card_id), **kw)
    # prereqs: no verified email -> refuse
    with pytest.raises(ShareError):
        layer.shares.grant(handle=HANDLE, card_id=card_id, device_id="dev-1",
                           ip_cluster="asn-1", post_ref=post(card_id),
                           account_age_s=30 * 86400, email_verified=False,
                           phone_otp_verified=True, now=NOW)
    # unverifiable post (no card link + attribution) -> grants nothing.
    # Use a DIFFERENT owned card (not yet granted) with a post_ref that
    # carries neither the card link nor the handle tag: verification fails
    # closed before any grant row is written.
    t0 = layer.ledger.get(layer.cards.get_card(card_id).trial_id)
    s2, s3, s5 = (layer.ledger.stage(t0.trial_id, s) for s in ("s2", "s3", "s5"))
    cid_unv = layer.cards.mint(
        trial_id=t0.trial_id, family=t0.family, scope_id=t0.scope_id,
        config_hash=t0.config_hash, data_vintage_hash=t0.data_vintage_hash,
        survivor_class="validated-green", owner_handle=HANDLE,
        dsr=s3["dsr"], t_net_full=s5["statistic"], p_perm=s2["p_value"],
    ).card_id
    with pytest.raises(ShareError):
        layer.shares.grant(handle=HANDLE, card_id=cid_unv, device_id="dev-1",
                           ip_cluster="asn-1",
                           post_ref="https://social.example/no-link",
                           account_age_s=30 * 86400, email_verified=True,
                           phone_otp_verified=True, now=NOW)
    # and the audit log recorded the verify_fail with no grant row written
    assert any(a["action"] == "verify_fail" for a in layer.shares.audit_log())
    assert len(layer.shares.grants_for(HANDLE)) == 1
    # cap at +5: the first grant above already used 1 slot on dev-1; use a
    # fresh device per additional grant so the per-device 30d rate limit
    # (3/device) does not mask the +5 account cap we are testing.
    t = layer.ledger.get(layer.cards.get_card(card_id).trial_id)
    s2, s3, s5 = (layer.ledger.stage(t.trial_id, s) for s in ("s2", "s3", "s5"))
    for i in range(5):
        cid = layer.cards.mint(
            trial_id=t.trial_id, family=t.family, scope_id=t.scope_id,
            config_hash=t.config_hash, data_vintage_hash=t.data_vintage_hash,
            survivor_class="validated-green", owner_handle=HANDLE,
            dsr=s3["dsr"], t_net_full=s5["statistic"], p_perm=s2["p_value"],
        ).card_id
        layer.vault.admit(cid, HANDLE, now=NOW + i)
        if i < 4:
            layer.shares.grant(post_ref=post(cid),
                               handle=HANDLE, card_id=cid, device_id=f"dev-{i}",
                               ip_cluster="asn-1", account_age_s=30 * 86400,
                               email_verified=True, phone_otp_verified=True,
                               now=NOW + i)
    assert layer.shares.share_slots(HANDLE) == MAX_SHARE_SLOTS
    # 6th distinct card grant -> cap refuse
    cid6 = layer.cards.mint(
        trial_id=t.trial_id, family=t.family, scope_id=t.scope_id,
        config_hash=t.config_hash, data_vintage_hash=t.data_vintage_hash,
        survivor_class="validated-green", owner_handle=HANDLE,
        dsr=s3["dsr"], t_net_full=s5["statistic"], p_perm=s2["p_value"],
    ).card_id
    with pytest.raises(ShareError):
        layer.shares.grant(post_ref=post(cid6), handle=HANDLE, card_id=cid6,
                           device_id="dev-99", ip_cluster="asn-1",
                           account_age_s=30 * 86400, email_verified=True,
                           phone_otp_verified=True, now=NOW + 99)
    # revoke path zeroes vault permanent slots
    layer.shares.revoke(HANDLE, card_id, reason="abuse review", now=NOW + 100)
    card = layer.cards.get_card(card_id)
    assert layer.vault.effective_capacity(HANDLE, card.tier) == layer.vault.base_capacity(card.tier)
    assert layer.shares.fraud_queue("open")


def test_share_device_rate_limit_and_cluster_flag(layer):
    verdict, _ = _run_and_mint(layer)
    t = layer.ledger.get(layer.cards.get_card(verdict["card_id"]).trial_id)
    s2, s3, s5 = (layer.ledger.stage(t.trial_id, s) for s in ("s2", "s3", "s5"))
    post = lambda cid, h: f"https://s.example/{cid} card={cid} @{h}"  # noqa: E731
    # 3 grants/device/30d allowed, 4th refused
    for i in range(3):
        cid = layer.cards.mint(
            trial_id=t.trial_id, family=t.family, scope_id=t.scope_id,
            config_hash=t.config_hash, data_vintage_hash=t.data_vintage_hash,
            survivor_class="validated-green", owner_handle=HANDLE,
            dsr=s3["dsr"], t_net_full=s5["statistic"], p_perm=s2["p_value"],
        ).card_id
        layer.shares.grant(post_ref=post(cid, HANDLE), handle=HANDLE, card_id=cid,
                           device_id="dev-9", ip_cluster="asn-9",
                           account_age_s=30 * 86400, email_verified=True,
                           phone_otp_verified=True, now=NOW + i)
    cid4 = layer.cards.mint(
        trial_id=t.trial_id, family=t.family, scope_id=t.scope_id,
        config_hash=t.config_hash, data_vintage_hash=t.data_vintage_hash,
        survivor_class="validated-green", owner_handle=HANDLE,
        dsr=s3["dsr"], t_net_full=s5["statistic"], p_perm=s2["p_value"],
    ).card_id
    with pytest.raises(ShareError):
        layer.shares.grant(post_ref=post(cid4, HANDLE), handle=HANDLE, card_id=cid4,
                           device_id="dev-9", ip_cluster="asn-9",
                           account_age_s=30 * 86400, email_verified=True,
                           phone_otp_verified=True, now=NOW + 3)
    # cluster flag: 5 distinct accounts on one ip_cluster -> the 5th is
    # refused. The 3 grants alice already made were on asn-1/asn-9 clusters,
    # NOT asn-farm, so they don't count toward the farm cluster. Per-device
    # rate limit is 3 grants/device/30d, so spread the farming accounts over
    # distinct device ids on the SAME ip_cluster — the cluster check counts
    # distinct accounts per ip_cluster, the rate limit counts grants per
    # device.
    for j in range(4):
        h = f"farm{j}"
        layer.ledger.register("null", "free_5y_top1k", {"f": j}, "v",
                              discoverer_handle=h)
        layer.ledger.decide(layer.ledger.con.execute(
            "SELECT trial_id FROM trial WHERE discoverer_handle=?", (h,)
        ).fetchone()[0], "killed")
        cid = layer.cards.mint(
            trial_id=f"t-{h}", family="pairs", scope_id="free_5y_top1k",
            config_hash=f"h-{h}", data_vintage_hash="v",
            survivor_class="validated-green", owner_handle=h,
            dsr=0.99, t_net_full=4.5, p_perm=0.005,
        ).card_id
        layer.shares.grant(post_ref=post(cid, h), handle=h, card_id=cid,
                           device_id=f"dev-farm-{j}", ip_cluster="asn-farm",
                           account_age_s=30 * 86400, email_verified=True,
                           phone_otp_verified=True, now=NOW + 10 + j)
    # sanity: 4 accounts on asn-farm so far
    assert layer.shares.con.execute(
        "SELECT COUNT(DISTINCT handle) c FROM share_grant WHERE ip_cluster='asn-farm'"
    ).fetchone()["c"] == 4
    # 5th account on the same cluster -> cluster flagged, grant refused
    h5 = "farm4"
    layer.ledger.register("null", "free_5y_top1k", {"f": 99}, "v", discoverer_handle=h5)
    layer.ledger.decide(layer.ledger.con.execute(
        "SELECT trial_id FROM trial WHERE discoverer_handle=?", (h5,)
    ).fetchone()[0], "killed")
    cid5 = layer.cards.mint(
        trial_id="t-farm4", family="pairs", scope_id="free_5y_top1k",
        config_hash="h-farm4", data_vintage_hash="v",
        survivor_class="validated-green", owner_handle=h5,
        dsr=0.99, t_net_full=4.5, p_perm=0.005,
    ).card_id
    with pytest.raises(ShareError):
        layer.shares.grant(post_ref=post(cid5, h5), handle=h5, card_id=cid5,
                           device_id="dev-farm-4", ip_cluster="asn-farm",
                           account_age_s=30 * 86400, email_verified=True,
                           phone_otp_verified=True, now=NOW + 20)


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
    layer.notify.set_mode(HANDLE, card_id, "live", now=NOW)   # pref live...
    # ...but free tier forces digest
    for i in range(5):
        ev = layer.notify.notify_hit(handle=HANDLE, card_id=card_id, tier="free",
                                     payload={"i": i}, now=NOW + i)
    # digest channel caps at 1/day -> only the first enqueued
    events = [e for e in layer.notify.queue_for(HANDLE) if e.kind == "hit"]
    assert len(events) == 1
    # paid tier gets live push, capped at 3/day
    for i in range(5):
        layer.notify.notify_hit(handle=HANDLE, card_id=card_id, tier="paid",
                                payload={"i": i}, now=NOW + 100 + i)
    push = [e for e in layer.notify.queue_for(HANDLE) if e.channel == "push"]
    assert len(push) == 3
    # digest batch delivers at most once/day
    d1 = layer.notify.digest_batch(HANDLE, now=NOW + 200)
    d2 = layer.notify.digest_batch(HANDLE, now=NOW + 300)
    assert len(d1) == 1 and d2 == []


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
    assert layer.decay.reevaluate(card_id, now=NOW + 10) is None
    assert layer.cards.get_card(card_id).tier == card.tier   # unchanged
    # events are public + explained
    layer.ledger.con.execute(
        "UPDATE stage_result SET dsr=0.80 WHERE trial_id=? AND stage='s3'",
        (trial.trial_id,),
    )
    layer.ledger.con.commit()
    ev = layer.decay.reevaluate(card_id, now=NOW + 20)
    assert ev is not None and ev.new_tier == "bronze"
    assert ev.explanation and "down" in ev.explanation.lower()
    assert layer.decay.events_for(card_id)[0].new_tier == "bronze"
