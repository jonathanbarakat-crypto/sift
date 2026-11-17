"""v4.1 mechanics 6 — share-for-capacity, behind cards_v4_1.

Every new path has a fixture:
- verified share grants +1 permanent slot (cap +5), read-back from the DB
- per-(account, card) uniqueness: second share of the same card -> ShareError
- cap +5: the 6th verified share -> outcome cap_reached, no grant
- revocation: -1 slot (floor 0), grant row gone, audit log keeps the events
- slots non-transferable: they live on the owner's vault row
- anti-abuse: OTP missing, KYC-lite failed, account too new, no completed
  run, device/IP/ASN cluster on another account, live-post verification
  failed (bad URL / wrong watermark / wrong card) -> fraud_queued + open
  fraud flag; while flagged, later shares are held_fraud_review
- honeypot: any share of a planted trap card -> fraud_queued, never grants
- append-only audit log: seqs strictly increasing, every attempt logged,
  revoke does not erase history
- flag off -> ShareError(flag_off) on record_share and revoke
- foreign card -> ShareError (cards non-transferable)
- share payload with a ticker-like token is refused by the face guard
"""

import pytest

from sift import flags
from sift.cards import CardError, CardRegistry
from sift.cards.face import CardFaceRenderer
from sift.cards.shares import (
    MAX_EXTRA_SLOTS,
    MIN_ACCOUNT_AGE_DAYS,
    MIN_COMPLETED_RUNS,
    ShareError,
    ShareManager,
)
from sift.cards.vault import Vault
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
def vault(cards, tmp_path):
    v = Vault(cards, tmp_path / "vault.db")
    yield v
    v.close()


@pytest.fixture()
def face(ledger, cards):
    return CardFaceRenderer(ledger, cards)


@pytest.fixture()
def shares(cards, vault, tmp_path, face):
    s = ShareManager(cards, vault, tmp_path / "shares.db", face=face)
    yield s
    s.close()


def _mint_owned(ledger, cards, handle="alice", fixture=None):
    # a per-handle fixture tag keeps the canonical spec distinct (the ledger
    # skip-list would otherwise return the same trial and its stages are
    # append-only), while the planted data is identical -> still validated-green
    verdict, _ = _run_validated_green(ledger, cards, handle=handle,
                                      fixture=fixture or f"F-PAIR-FULL-{handle}")
    return verdict["card_id"]


def _mint_extra_owned(cards, handle="alice", tag="x"):
    """A second card owned by the same handle (different spec)."""
    return cards.mint(
        trial_id=f"trial-{tag}", family="pairs", scope_id="free_5y_top1k",
        config_hash=f"h-{tag}", data_vintage_hash="v",
        survivor_class="validated-green", owner_handle=handle,
        dsr=0.96, t_net_full=3.2, p_perm=0.04,
    ).card_id


def _good_post(cards, handle, card_id, face, now=1_700_000_000.0):
    share_obj = face.render_share(card_id, handle, now=now)
    return {
        "url": "https://example.social/@%s/post/1" % handle,
        "payload": {"card_id": card_id, "watermark": share_obj["watermark"]},
    }


def _share(shares, handle, card_id, post, device_id="dev-1", ip="10.0.0.1",
           asn="AS123", age=90.0, runs=3, otp=True, kyc=True):
    return shares.record_share(
        handle, card_id, post_url=post["url"], post_payload=post["payload"],
        device_id=device_id, ip=ip, asn=asn,
        account_age_days=age, completed_runs=runs, otp_verified=otp, kyc_lite=kyc,
    )


# -- grant path ------------------------------------------------------------------

def test_verified_share_grants_plus_one_permanent_slot(ledger, cards, vault, shares, face):
    card_id = _mint_owned(ledger, cards)
    vault.admit(card_id, "alice")
    card = cards.get_card(card_id)
    base = vault.effective_capacity("alice", card.tier)

    post = _good_post(cards, "alice", card_id, face)
    res = _share(shares, "alice", card_id, post)

    assert res["outcome"] == "granted" and res["granted"] is True
    # read-back: grant row + vault extra_slots re-read from the DB
    assert res["grant"]["card_id"] == card_id
    assert res["slots_after"] == 1
    assert shares.slots_granted("alice") == 1
    assert vault.effective_capacity("alice", card.tier) == base + 1
    assert res["audit_seq"] == shares.audit_log()[-1]["seq"]


def test_cap_plus_five(ledger, cards, vault, shares, face):
    card_id = _mint_owned(ledger, cards)
    vault.admit(card_id, "alice")
    # 5 different cards (per-(account,card) uniqueness) -> 5 grants
    others = [_mint_extra_owned(cards, tag=f"cap{i}") for i in range(5)]
    for i, cid in enumerate([card_id] + others[:4]):
        post = _good_post(cards, "alice", cid, face, now=1_700_000_000.0 + i)
        res = _share(shares, "alice", cid, post, device_id=f"dev-{i}", ip=f"10.0.0.{i+1}", asn=f"AS{i}")
        assert res["granted"] is True
    assert shares.slots_granted("alice") == 5
    assert vault.add_permanent_slots("alice", 0) == MAX_EXTRA_SLOTS
    # 6th verified share -> cap_reached, no new grant
    post = _good_post(cards, "alice", others[4], face)
    res = _share(shares, "alice", others[4], post)
    assert res["outcome"] == "cap_reached" and res["granted"] is False
    assert res["slots_after"] == MAX_EXTRA_SLOTS
    assert shares.slots_granted("alice") == 5


def test_duplicate_account_card_share_raises(ledger, cards, vault, shares, face):
    card_id = _mint_owned(ledger, cards)
    vault.admit(card_id, "alice")
    post = _good_post(cards, "alice", card_id, face)
    assert _share(shares, "alice", card_id, post)["granted"] is True
    with pytest.raises(ShareError):
        _share(shares, "alice", card_id, post)     # same (account, card) again
    assert shares.slots_granted("alice") == 1
    assert shares.audit_log()[-1]["outcome"] == "duplicate"


def test_revoke_subtracts_one_slot_and_keeps_audit(ledger, cards, vault, shares, face):
    card_id = _mint_owned(ledger, cards)
    other = _mint_extra_owned(cards, tag="rev")
    vault.admit(card_id, "alice")
    p1 = _good_post(cards, "alice", card_id, face)
    p2 = _good_post(cards, "alice", other, face)
    _share(shares, "alice", card_id, p1)
    _share(shares, "alice", other, p2, device_id="dev-2", ip="10.0.0.2", asn="AS9")
    assert shares.slots_granted("alice") == 2

    res = shares.revoke("alice", card_id)
    assert res["outcome"] == "revoked"
    assert res["grant_removed"] is True            # read-back
    assert res["slots_after"] == 1
    assert shares.slots_granted("alice") == 1
    # audit log is append-only: the grant and the revoke are both there
    outcomes = [e["outcome"] for e in shares.audit_log() if e["action"] in ("share", "revoke")]
    assert "granted" in outcomes and "revoked" in outcomes
    # revoking a non-grant raises
    with pytest.raises(ShareError):
        shares.revoke("alice", card_id)


def test_slots_are_non_transferable(ledger, cards, vault, shares, face):
    card_id = _mint_owned(ledger, cards)
    vault.admit(card_id, "alice")
    post = _good_post(cards, "alice", card_id, face)
    _share(shares, "alice", card_id, post)
    # bob's capacity has no extra slots: slots live on the owner's row only
    bob_card = _mint_owned(ledger, cards, handle="bob")
    vault.admit(bob_card, "bob")
    card = cards.get_card(bob_card)
    assert vault.effective_capacity("bob", card.tier) == vault.base_capacity(card.tier)


def test_foreign_card_share_refused(ledger, cards, vault, shares, face):
    card_id = _mint_owned(ledger, cards, handle="alice")
    # build the post as the OWNER (mallory cannot render a foreign card), then
    # have mallory try to record the share — ownership check must refuse
    post = _good_post(cards, "alice", card_id, face)
    with pytest.raises((ShareError, CardError)):
        _share(shares, "mallory", card_id, post)


# -- anti-abuse stack --------------------------------------------------------------

def test_otp_and_kyc_lite_required(ledger, cards, vault, shares, face):
    card_id = _mint_owned(ledger, cards)
    vault.admit(card_id, "alice")
    post = _good_post(cards, "alice", card_id, face)
    r1 = _share(shares, "alice", card_id, post, otp=False)
    assert r1["outcome"] == "fraud_queued" and r1["reason"] == "otp_not_verified"
    assert shares.has_open_fraud("alice")
    # while flagged: held, no grant even with OTP
    r2 = _share(shares, "alice", card_id, post, otp=True, kyc=True)
    assert r2["outcome"] == "held_fraud_review"
    assert shares.slots_granted("alice") == 0


def test_kyc_lite_failed_queues_fraud(ledger, cards, vault, shares, face):
    card_id = _mint_owned(ledger, cards)
    post = _good_post(cards, "alice", card_id, face)
    r = _share(shares, "alice", card_id, post, kyc=False)
    assert r["outcome"] == "fraud_queued" and r["reason"] == "kyc_lite_failed"


def test_account_age_and_completed_runs_required(ledger, cards, vault, shares, face):
    card_id = _mint_owned(ledger, cards)
    post = _good_post(cards, "alice", card_id, face)
    r1 = _share(shares, "alice", card_id, post, age=MIN_ACCOUNT_AGE_DAYS - 1)
    assert r1["reason"] == "account_too_new"
    shares2_reason = None
    # clear the fraud flag server-side, then try the runs check
    shares.con.execute("UPDATE fraud_queue SET status='cleared'")
    shares.con.commit()
    r2 = _share(shares, "alice", card_id, post, age=MIN_ACCOUNT_AGE_DAYS, runs=MIN_COMPLETED_RUNS - 1)
    assert r2["reason"] == "no_completed_run"
    assert shares2_reason is None


def test_device_ip_asn_cluster_flags_sybil(ledger, cards, vault, shares, face):
    card_id = _mint_owned(ledger, cards)
    post = _good_post(cards, "alice", card_id, face)
    # alice shares from dev-1 / 10.0.0.1 / AS123 -> granted
    assert _share(shares, "alice", card_id, post)["granted"] is True
    # bob reuses the same device key on a card he owns
    bob_card = _mint_owned(ledger, cards, handle="bob")
    post_bob = _good_post(cards, "bob", bob_card, face)
    r = _share(shares, "bob", bob_card, post_bob, device_id="dev-1", ip="10.0.0.1", asn="AS123")
    assert r["outcome"] == "fraud_queued" and r["reason"] == "device_ip_asn_cluster"
    assert shares.has_open_fraud("bob")


def test_live_post_verification(ledger, cards, vault, shares, face):
    card_id = _mint_owned(ledger, cards)
    good = _good_post(cards, "alice", card_id, face)
    # bad URL
    bad_url = dict(good, url="ftp://nope")
    r = _share(shares, "alice", card_id, bad_url)
    assert r["reason"] == "live_post_verification_failed"
    shares.con.execute("UPDATE fraud_queue SET status='cleared'"); shares.con.commit()
    # wrong watermark handle
    bad_wm = {"url": good["url"], "payload": {"card_id": card_id,
               "watermark": {"handle": "mallory", "ts": 1.0}}}
    r = _share(shares, "alice", card_id, bad_wm)
    assert r["reason"] == "live_post_verification_failed"
    shares.con.execute("UPDATE fraud_queue SET status='cleared'"); shares.con.commit()
    # wrong card id in the payload
    bad_card = {"url": good["url"], "payload": {"card_id": "card-forged",
                 "watermark": good["payload"]["watermark"]}}
    r = _share(shares, "alice", card_id, bad_card)
    assert r["reason"] == "live_post_verification_failed"


def test_honeypot_share_never_grants(ledger, cards, vault, shares, face):
    card_id = _mint_owned(ledger, cards)
    shares.plant_honeypot(card_id)
    post = _good_post(cards, "alice", card_id, face)
    r = _share(shares, "alice", card_id, post)
    assert r["outcome"] == "fraud_queued" and r["reason"] == "honeypot_share"
    assert shares.slots_granted("alice") == 0
    assert shares.has_open_fraud("alice")


def test_audit_log_append_only_monotonic(ledger, cards, vault, shares, face):
    card_id = _mint_owned(ledger, cards)
    vault.admit(card_id, "alice")
    other = _mint_extra_owned(cards, tag="aud")
    vault.admit(other, "alice")
    p1 = _good_post(cards, "alice", card_id, face)
    p2 = _good_post(cards, "alice", other, face)
    _share(shares, "alice", card_id, p1)
    _share(shares, "alice", other, p2, device_id="dev-2", ip="10.0.0.2", asn="AS9")
    shares.revoke("alice", card_id)
    log = shares.audit_log()
    seqs = [e["seq"] for e in log]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)   # strictly increasing
    actions = [e["action"] for e in log]
    assert "share" in actions and "revoke" in actions
    # there is no delete/update path: revoke kept the grant's audit events
    assert any(e["outcome"] == "granted" for e in log)
    assert any(e["outcome"] == "revoked" for e in log)


# -- flag + guard ------------------------------------------------------------------

def test_flag_off_refuses_share_and_revoke(ledger, cards, vault, shares, face):
    card_id = _mint_owned(ledger, cards)
    vault.admit(card_id, "alice")
    post = _good_post(cards, "alice", card_id, face)
    flags.set_flag("cards_v4_1", False)
    with pytest.raises(ShareError):
        _share(shares, "alice", card_id, post)
    with pytest.raises(ShareError):
        shares.revoke("alice", card_id)
    # reads still work; nothing was granted
    assert shares.grants_for("alice") == []
    assert shares.slots_granted("alice") == 0


def test_share_payload_with_ticker_like_token_refused(ledger, cards, vault, shares, face):
    card_id = _mint_owned(ledger, cards)
    vault.admit(card_id, "alice")
    good = _good_post(cards, "alice", card_id, face)
    # payload carrying a raw ticker-like token is not a valid watermarked share
    poisoned = {"url": good["url"],
                "payload": {"card_id": card_id, "watermark": good["payload"]["watermark"],
                            "holdings": "long AAPL"}}
    r = _share(shares, "alice", card_id, poisoned)
    # the face renderer's own guard must refuse a ticker-bearing share object
    import json
    from sift.cards.face import _guard_no_tickers, FaceError
    with pytest.raises(FaceError):
        _guard_no_tickers(json.dumps(poisoned["payload"], default=str))
    # and the live-post check still requires the exact watermarked artifact
    assert r["outcome"] in ("fraud_queued",)
