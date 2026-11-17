"""v4.1 mechanics 6 — share-for-capacity: +1 permanent vault slot per
verified share (cap +5) with the §4.7 anti-abuse stack, behind cards_v4_1.

Fixtures/paths covered (one per new path):
- happy-path grant: +1 permanent slot, audit row, read-back of share_unique
  AND vault slot count / effective capacity;
- uniqueness per (account, card): second grant for the same card raises;
- cap enforcement at +5 (5 grants OK on 5 cards, 6th raises; vault cap too);
- prerequisites: email+phone OTP, minimum account age, >=1 completed mining
  run, KYC-lite for higher caps (beyond +3);
- device/IP clustering: velocity (>=5 accounts on one device -> flagged,
  not auto-granted, fraud queue) + per-device rate limit (3 grants/30d);
- live-post verification: unverifiable share grants NOTHING (and is queued
  for fraud review); a post with the anonymized card link + attribution tag
  verifies;
- revoke path: abuse -> slots revoked + account flagged, read-back; repeat
  offense -> share feature disabled (first offense = warning, not ban);
- planted honeypot: grant attempt on a canary account is flagged;
- flag off -> ShareError(flag_off) (global, per-user, and env kill-switch);
- audit log is append-only (UPDATE/DELETE blocked by trigger);
- vault capacity reads share slots (base tier + slots), read-back;
- foreign card cannot be shared (non-transferable -> CardError).
"""

import sqlite3
import time

import pytest

from sift import flags
from sift.cards import CardError, CardRegistry
from sift.cards.share import (
    FRAUD_QUEUE_SLA_S,
    MAX_SHARE_SLOTS,
    ShareError,
    ShareService,
)
from sift.cards.vault import Vault
from sift.registry import Ledger
from tests.cards.test_minting import SECRET, _run_validated_green

NOW = 1_800_000_000.0
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
def cards(tmp_path):
    c = CardRegistry(tmp_path / "cards.db", secret=SECRET)
    yield c
    c.close()


@pytest.fixture()
def vault(cards, tmp_path):
    v = Vault(cards, tmp_path / "vault.db")
    yield v
    v.close()


@pytest.fixture()
def share(cards, vault, tmp_path):
    s = ShareService(cards, vault, tmp_path / "share.db")
    yield s
    s.close()


def _mint_card(cards, handle, tag, tier_stats=(0.995, 4.5, 0.005)):
    """Mint one distinct card for `handle` (distinct config hash per tag)."""
    return cards.mint(
        trial_id=f"trial-{tag}", family="pairs", scope_id="free_5y_top1k",
        config_hash=f"h-{tag}", data_vintage_hash="v1",
        survivor_class="validated-green", owner_handle=handle,
        dsr=tier_stats[0], t_net_full=tier_stats[1], p_perm=tier_stats[2],
    ).card_id


def _ready_account(share, handle, runs=3, age_s=30 * 24 * 3600,
                   kyc=True, now=NOW):
    share.register_account(
        handle, now=now, email_verified=True, phone_otp_verified=True,
        kyc_lite=kyc, created_at=now - age_s, completed_runs=runs,
    )


def _good_post(share, card_id):
    """Inject a live-post fetcher returning a post that contains the
    anonymized card link + the attribution tag."""
    share.post_fetcher = lambda url: (
        f"my post {url} — love this card https://sift.example/cards/{card_id}/face #SIFT"
    )


# -- happy path ---------------------------------------------------------------


def test_grant_adds_one_permanent_slot_with_readback(ledger, cards, vault, share):
    verdict, _ = _run_validated_green(ledger, cards, handle="alice")
    card_id = verdict["card_id"]
    _ready_account(share, "alice")
    _good_post(share, card_id)
    tier = cards.get_card(card_id).tier
    base = vault.effective_capacity("alice", tier)

    res = share.request_share("alice", card_id, post_url="https://x.example/p/1",
                              device_id="dev-a", ip="1.2.3.4", asn="AS1", now=NOW)

    assert res["granted"] is True
    assert res["slots"] == 1                              # read-back from vault DB
    assert res["capacity"] == base + 1                    # base tier + share slot
    assert res["granted_at"] == NOW                       # read-back from share_unique
    # vault capacity reads the share-slot count
    assert vault.effective_capacity("alice", tier) == base + 1
    assert vault.share_slots("alice") == 1
    # audit log row
    log = share.grants_for("alice")
    assert len(log) == 1 and log[0].outcome == "granted" and log[0].verified == 1
    assert log[0].device_id == "dev-a" and log[0].ip == "1.2.3.4" and log[0].asn == "AS1"


# -- uniqueness per (account, card) --------------------------------------------


def test_second_grant_for_same_card_raises(ledger, cards, vault, share):
    verdict, _ = _run_validated_green(ledger, cards, handle="alice")
    card_id = verdict["card_id"]
    _ready_account(share, "alice")
    _good_post(share, card_id)
    share.request_share("alice", card_id, post_url="https://x.example/p/1",
                        device_id="dev-a", now=NOW)
    with pytest.raises(ShareError, match="one slot per"):
        share.request_share("alice", card_id, post_url="https://x.example/p/2",
                            device_id="dev-a", now=NOW + 10)
    assert share.share_slots("alice") == 1                # still exactly one


# -- cap at +5 ------------------------------------------------------------------


def test_cap_enforced_at_plus_5(ledger, cards, vault, share):
    _ready_account(share, "alice")
    ids = [_mint_card(cards, "alice", f"c{i}") for i in range(6)]
    for i, cid in enumerate(ids[:MAX_SHARE_SLOTS]):
        _good_post(share, cid)
        res = share.request_share("alice", cid, post_url=f"https://x.example/p/{i}",
                                  device_id=f"dev-{i}", now=NOW + i)
        assert res["granted"] is True
    assert share.share_slots("alice") == 5
    with pytest.raises(ShareError, match="cap"):
        _good_post(share, ids[5])
        share.request_share("alice", ids[5], post_url="https://x.example/p/5",
                            device_id="dev-9", now=NOW + 99)
    # vault never exceeds base + 5
    tier = cards.get_card(ids[0]).tier
    assert vault.effective_capacity("alice", tier) == vault.base_capacity(tier) + 5


# -- prerequisites ---------------------------------------------------------------


def test_email_and_phone_otp_required_before_first_grant(ledger, cards, vault, share):
    card_id = _mint_card(cards, "alice", "x")
    _ready_account(share, "alice")
    share.con.execute(
        "UPDATE account_state SET email_verified=0, phone_otp_verified=0 WHERE handle='alice'"
    )
    share.con.commit()
    _good_post(share, card_id)
    with pytest.raises(ShareError, match="OTP"):
        share.request_share("alice", card_id, post_url="https://x.example/p",
                            device_id="d", now=NOW)
    assert share.share_slots("alice") == 0
    # audit log recorded the denial
    assert share.grants_for("alice")[-1].outcome == "denied_prereq"


def test_minimum_account_age_required(ledger, cards, vault, share):
    card_id = _mint_card(cards, "alice", "x")
    _ready_account(share, "alice", age_s=3600)           # 1h old — too young
    _good_post(share, card_id)
    with pytest.raises(ShareError, match="age"):
        share.request_share("alice", card_id, post_url="https://x.example/p",
                            device_id="d", now=NOW)
    assert share.share_slots("alice") == 0


def test_completed_mining_run_required(ledger, cards, vault, share):
    card_id = _mint_card(cards, "alice", "x")
    _ready_account(share, "alice", runs=0)                # no completed run
    _good_post(share, card_id)
    with pytest.raises(ShareError, match="mining run"):
        share.request_share("alice", card_id, post_url="https://x.example/p",
                            device_id="d", now=NOW)
    assert share.share_slots("alice") == 0


def test_kyc_lite_required_for_higher_caps(ledger, cards, vault, share):
    """KYC-lite (name+DOB) is required once the account already has +3 slots
    (higher caps)."""
    _ready_account(share, "alice", kyc=False)
    ids = [_mint_card(cards, "alice", f"k{i}") for i in range(4)]
    for i, cid in enumerate(ids[:3]):
        _good_post(share, cid)
        res = share.request_share("alice", cid, post_url=f"https://x.example/p/{i}",
                                  device_id=f"dev-{i}", now=NOW + i)
        assert res["granted"] is True
    assert share.share_slots("alice") == 3
    _good_post(share, ids[3])
    with pytest.raises(ShareError, match="KYC-lite"):
        share.request_share("alice", ids[3], post_url="https://x.example/p/3",
                            device_id="dev-3", now=NOW + 3)
    assert share.share_slots("alice") == 3                # no 4th slot without KYC


# -- anti-abuse: clustering, rate limits, verification ---------------------------


def test_velocity_cluster_flags_instead_of_granting(ledger, cards, vault, share):
    """5 accounts from one IP/ASN cluster are all flagged, not auto-granted
    (velocity/pattern cap). Distinct devices per account so the per-DEVICE
    rate limit does not fire first; the IP cluster is what reaches 5."""
    ip = "9.9.9.9"
    handles = [f"u{i}" for i in range(5)]
    for h in handles:
        _ready_account(share, h)
    for i, h in enumerate(handles[:4]):
        cid = _mint_card(cards, h, f"v{i}")
        _good_post(share, cid)
        res = share.request_share(h, cid, post_url=f"https://x.example/{h}",
                                  device_id=f"dev-{h}", ip=ip, asn="AS9", now=NOW + i)
        assert res["granted"] is True
    # 5th account from the same IP/ASN: cluster size hits 5 -> flagged
    h5 = handles[4]
    cid5 = _mint_card(cards, h5, "v4")
    _good_post(share, cid5)
    res = share.request_share(h5, cid5, post_url="https://x.example/u4",
                              device_id="dev-u4", ip=ip, asn="AS9", now=NOW + 10)
    assert res["granted"] is False and res["reason"] == "velocity_cluster"
    assert share.share_slots(h5) == 0
    # fraud queue got the flagged account, with an SLA deadline
    q = [r for r in share.fraud_queue_open() if r["owner"] == h5]
    assert q and q[0]["signal"] == "velocity_cluster"
    assert q[0]["sla_deadline"] == q[0]["queued_at"] + FRAUD_QUEUE_SLA_S
    assert share.grants_for(h5)[-1].outcome == "flagged"


def test_per_device_rate_limit_3_grants_per_30d(ledger, cards, vault, share):
    _ready_account(share, "alice")
    ids = [_mint_card(cards, "alice", f"r{i}") for i in range(4)]
    for i, cid in enumerate(ids[:3]):
        _good_post(share, cid)
        res = share.request_share("alice", cid, post_url=f"https://x.example/p/{i}",
                                  device_id="dev-rl", now=NOW + i)
        assert res["granted"] is True
    _good_post(share, ids[3])
    res = share.request_share("alice", ids[3], post_url="https://x.example/p/3",
                              device_id="dev-rl", now=NOW + 3)
    assert res["granted"] is False and res["reason"] == "device_ratelimit"
    assert share.share_slots("alice") == 3
    # a different device is not rate-limited by the first device's grants
    cid4 = _mint_card(cards, "alice", "r4")
    _good_post(share, cid4)
    res = share.request_share("alice", cid4, post_url="https://x.example/p/4",
                              device_id="dev-other", now=NOW + 4)
    assert res["granted"] is True


def test_unverifiable_share_grants_nothing(ledger, cards, vault, share):
    """No public API check possible -> grants nothing, queued for review."""
    card_id = _mint_card(cards, "alice", "u")
    _ready_account(share, "alice")
    # no post_fetcher -> the post cannot be retrieved -> unverifiable
    res = share.request_share("alice", card_id, post_url="https://x.example/gone",
                              device_id="dev-u", now=NOW)
    assert res["granted"] is False and res["reason"] == "unverifiable_post"
    assert share.share_slots("alice") == 0                # nothing granted
    assert vault.effective_capacity("alice", cards.get_card(card_id).tier) == \
        vault.base_capacity(cards.get_card(card_id).tier)
    # audit + fraud queue recorded it
    assert share.grants_for("alice")[-1].outcome == "denied_unverifiable"
    assert any(r["signal"] == "unverifiable_post" for r in share.fraud_queue_open())


def test_post_without_card_link_or_tag_is_unverifiable(ledger, cards, vault, share):
    card_id = _mint_card(cards, "alice", "u2")
    _ready_account(share, "alice")
    other = _mint_card(cards, "alice", "other")
    # post mentions a DIFFERENT card's link and no attribution tag
    share.post_fetcher = lambda url: f"see https://sift.example/cards/{other}/face"
    res = share.request_share("alice", card_id, post_url="https://x.example/p",
                              device_id="dev-u2", now=NOW)
    assert res["granted"] is False and res["reason"] == "unverifiable_post"
    # post with the right link but missing the attribution tag
    share.post_fetcher = lambda url: f"see https://sift.example/cards/{card_id}/face"
    res = share.request_share("alice", card_id, post_url="https://x.example/p2",
                              device_id="dev-u2", now=NOW + 1)
    assert res["granted"] is False
    assert share.share_slots("alice") == 0


# -- abuse: revoke / flag / disable ------------------------------------------------


def test_revoke_path_revokes_slots_and_flags_account_with_readback(ledger, cards, vault, share):
    _ready_account(share, "alice")
    ids = [_mint_card(cards, "alice", f"a{i}") for i in range(2)]
    for i, cid in enumerate(ids):
        _good_post(share, cid)
        share.request_share("alice", cid, post_url=f"https://x.example/p/{i}",
                            device_id=f"dev-{i}", now=NOW + i)
    assert share.share_slots("alice") == 2

    res = share.revoke_for_abuse("alice", ids[0], reason="share_farming", now=NOW + 100)

    assert res["revoked_slots"] == 2
    assert res["slots_after"] == 0                        # read-back
    assert res["offense_count"] == 1                      # read-back
    assert res["share_disabled"] is False                 # first offense: warning, not ban
    assert res["warning"] is True
    assert vault.share_slots("alice") == 0                # vault read-back
    # audit log has the revocation row (append-only)
    assert share.grants_for("alice")[-1].outcome == "revoked"


def test_repeat_offense_disables_share_feature(ledger, cards, vault, share):
    _ready_account(share, "alice")
    cid = _mint_card(cards, "alice", "a0")
    _good_post(share, cid)
    share.request_share("alice", cid, post_url="https://x.example/p",
                        device_id="dev-0", now=NOW)
    share.revoke_for_abuse("alice", cid, now=NOW + 10)    # offense 1: warning
    cid2 = _mint_card(cards, "alice", "a1")
    _good_post(share, cid2)
    share.request_share("alice", cid2, post_url="https://x.example/p2",
                        device_id="dev-1", now=NOW + 20)
    res = share.revoke_for_abuse("alice", cid2, now=NOW + 30)   # offense 2
    assert res["offense_count"] == 2
    assert res["share_disabled"] is True
    # now the share feature is disabled for the account
    cid3 = _mint_card(cards, "alice", "a2")
    _good_post(share, cid3)
    with pytest.raises(ShareError, match="share_disabled"):
        share.request_share("alice", cid3, post_url="https://x.example/p3",
                            device_id="dev-2", now=NOW + 40)


def test_planted_honeypot_grant_is_flagged(ledger, cards, vault, share):
    """A grant attempt against a planted honeypot/canary account is flagged
    (share-farming service detection) and grants nothing."""
    share.plant_honeypot("canary-1", note="planted canary", now=NOW)
    card_id = _mint_card(cards, "canary-1", "hp")
    _ready_account(share, "canary-1")
    _good_post(share, card_id)
    res = share.request_share("canary-1", card_id, post_url="https://x.example/hp",
                              device_id="dev-hp", now=NOW)
    assert res["granted"] is False and res["reason"] == "honeypot_account"
    assert share.share_slots("canary-1") == 0
    log = share.grants_for("canary-1")
    assert log[-1].outcome == "flagged" and log[-1].reason == "honeypot_account"
    assert any(r["signal"] == "honeypot_account" for r in share.fraud_queue_open())


# -- flag off -> fail closed --------------------------------------------------------


def test_flag_off_raises_share_error(ledger, cards, vault, share):
    card_id = _mint_card(cards, "alice", "f")
    _ready_account(share, "alice")
    flags.set_flag("cards_v4_1", False)
    with pytest.raises(ShareError, match="flag_off"):
        share.request_share("alice", card_id, post_url="https://x.example/p",
                            device_id="d", now=NOW)
    with pytest.raises(ShareError, match="flag_off"):
        share.revoke_for_abuse("alice", card_id, now=NOW)
    flags.reset()
    flags.disable_for_user("cards_v4_1", "alice")
    with pytest.raises(ShareError, match="flag_off"):
        share.request_share("alice", card_id, post_url="https://x.example/p",
                            device_id="d", now=NOW)


def test_env_kill_switch_sift_cards_v4_1_0(ledger, cards, vault, share, monkeypatch):
    card_id = _mint_card(cards, "alice", "f2")
    _ready_account(share, "alice")
    monkeypatch.setenv("SIFT_CARDS_V4_1", "0")
    flags.reset()                                          # re-reads the env var
    assert flags.cards_v4_1_enabled("alice") is False
    with pytest.raises(ShareError, match="flag_off"):
        share.request_share("alice", card_id, post_url="https://x.example/p",
                            device_id="d", now=NOW)


# -- audit log append-only + vault integration --------------------------------------


def test_audit_log_is_append_only(ledger, cards, vault, share):
    _ready_account(share, "alice")
    cid = _mint_card(cards, "alice", "ap")
    _good_post(share, cid)
    share.request_share("alice", cid, post_url="https://x.example/p",
                        device_id="d", now=NOW)
    with pytest.raises(sqlite3.IntegrityError):
        share.con.execute("UPDATE share_grant SET outcome='granted' WHERE owner='alice'")
    with pytest.raises(sqlite3.IntegrityError):
        share.con.execute("DELETE FROM share_grant WHERE owner='alice'")
    share.con.rollback()
    assert len(share.audit_log()) == 1                    # row intact


def test_vault_capacity_reads_share_slots_and_readback_after_grant_and_revoke(
        ledger, cards, vault, share):
    _ready_account(share, "alice")
    cid = _mint_card(cards, "alice", "vi")
    card = cards.get_card(cid)
    base = vault.base_capacity(card.tier)
    assert vault.effective_capacity("alice", card.tier) == base   # no slots yet
    _good_post(share, cid)
    res = share.request_share("alice", cid, post_url="https://x.example/p",
                              device_id="d", now=NOW)
    assert res["granted"] is True
    # read-back after grant: capacity includes the share slot
    assert vault.effective_capacity("alice", card.tier) == base + 1
    assert vault.share_slots("alice") == 1
    # read-back after revoke: back to base
    share.revoke_for_abuse("alice", cid, now=NOW + 5)
    assert vault.effective_capacity("alice", card.tier) == base
    assert vault.share_slots("alice") == 0


def test_foreign_card_cannot_be_shared(ledger, cards, vault, share):
    """Cards are non-transferable: sharing someone else's card raises."""
    cid = _mint_card(cards, "alice", "own")
    _ready_account(share, "mallory")
    with pytest.raises(CardError):
        share.request_share("mallory", cid, post_url="https://x.example/p",
                            device_id="d", now=NOW)
    assert share.share_slots("mallory") == 0
