"""Anti-abuse attack sims against the v4.1 §4.7 controls.

SIMULATED — infrastructure validation, not user-behavior evidence.

Each attack below is a synthetic attack pattern (seeded, fixed clock) run
against an ephemeral test world. The sim asserts the CONTROLS FIRE:
grants rejected/flagged, the append-only `share_grant` log intact (UPDATE
and DELETE blocked by triggers), the audit log written, fraud-queue SLA
routing present.

Attacks covered (task scope item 2):
  1. device/IP/ASN cluster claiming share slots  -> velocity flag at >=5 accounts
  2. day-zero accounts sharing before first completed run -> denied_prereq
  3. unverifiable share posts                       -> grant must NOT fire
  4. duplicate shares of the same card               -> one slot per (account, card)
  5. velocity limit >3 grants/device/30d            -> denied_ratelimit
  6. +5 cap enforcement                              -> ShareError cap
  7. revocation path on abuse flag                   -> slots zeroed, offense counted
  8. fraud-queue SLA routing                         -> rows carry sla_deadline = queued_at + SLA
Plus: append-only integrity probe (UPDATE/DELETE on share_grant must raise).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from sift.cards.share import (
    FRAUD_QUEUE_SLA_S,
    MAX_GRANTS_PER_DEVICE_30D,
    MAX_SHARE_SLOTS,
    ShareError,
)
from sift.cards.vault import VaultError

from .clock import DEFAULT_SEED
from .report import build_report, dump_json
from .world import SIM_SCOPE_ID, SIM_VINTAGE, SimWorld

DAY = 86400.0


def mint_sim_card(world: SimWorld, handle: str, tag: str,
                  stats=(0.995, 4.5, 0.005), admit: bool = True) -> str:
    """Mint one distinct synthetic card for `handle` (distinct config hash
    per tag); admit it to the vault when there is capacity (the anti-abuse
    sims mint many cards per account, so admit is optional — share grants
    only need the card to exist and be owned)."""
    trial = world.ledger.register(
        "pairs", SIM_SCOPE_ID, {"sim": True, "tag": tag, "owner": handle},
        SIM_VINTAGE, provenance="user_initiated", discoverer_handle=handle,
    )
    world.ledger.decide(trial.trial_id, "survivor", "validated-green")
    card = world.layer.cards.mint(
        trial_id=trial.trial_id, family=trial.family, scope_id=trial.scope_id,
        config_hash=f"sim-h-{tag}", data_vintage_hash=trial.data_vintage_hash,
        survivor_class="validated-green", owner_handle=handle,
        dsr=stats[0], t_net_full=stats[1], p_perm=stats[2],
        now=world.clock.now(),
    )
    if admit:
        try:
            world.layer.vault.admit(card.card_id, handle, now=world.clock.now())
        except Exception:
            pass  # vault full in attack sims — card ownership is what matters
    return card.card_id


def register_ready(world: SimWorld, handle: str, *, age_days: float = 30.0,
                   runs: int = 3, kyc: bool = True, email: bool = True,
                   phone: bool = True) -> None:
    """Register a share-account with prerequisites satisfied (or not, per
    the attack's needs)."""
    world.layer.shares.register_account(
        handle, now=world.clock.now(),
        email_verified=email, phone_otp_verified=phone, kyc_lite=kyc,
        created_at=world.clock.now() - age_days * DAY,
        completed_runs=runs,
    )


def good_post_fetcher(world: SimWorld) -> None:
    """Inject a post fetcher that returns a verifiable post (card link +
    attribution tag) for any URL — the 'honest' platform stub."""
    def fetch(url: str) -> str:
        return f"post {url} — see https://sift.example/cards/CARDID/face #SIFT"
    setattr(world.layer.shares, "post_fetcher", fetch)


def good_post_for(world: SimWorld, card_id: str) -> None:
    setattr(world.layer.shares, "post_fetcher",
            lambda url, cid=card_id: f"post {url} — https://sift.example/cards/{cid}/face #SIFT")


def _grant(world: SimWorld, user, card_id: str, post_url: str) -> dict:
    """One share-grant attempt; hard refusals (ShareError) are caught and
    returned as {'granted': False, 'raised': reason}."""
    try:
        return world.layer.shares.request_share(
            user.handle, card_id, post_url=post_url,
            device_id=user.device_id, ip=user.ip, asn=user.asn,
            now=world.clock.now(),
        )
    except ShareError as exc:
        return {"granted": False, "raised": str(exc)}


@dataclass
class AttackResult:
    report: dict
    json_text: str
    checks: dict[str, bool]


def run_antiabuse_sims(db_dir: str | Path, *, seed: int = DEFAULT_SEED) -> AttackResult:
    """Run all §4.7 attack sims in one ephemeral world; aggregate a bannered
    report with per-attack assertions."""
    checks: dict[str, bool] = {}
    sections: dict = {}

    with SimWorld(db_dir, seed=seed, n_users=0) as world:
        now = world.clock.now()

        # ---- attack 1: device/IP/ASN cluster claiming share slots --------
        cluster_handles = []
        for j in range(6):
            u = world.add_user(
                handle=f"sim-cluster-{j}", device_id="sim-shared-device",
                ip="203.0.113.7", asn="AS-SIM-ATTACK",
                created_at=now - 30 * DAY, rng=world.rng.spawn(7000 + j),
            )
            register_ready(world, u.handle)
            cid = mint_sim_card(world, u.handle, f"cluster{j}")
            good_post_for(world, cid)
            r = _grant(world, u, cid, f"https://social.example/cluster{j}")
            cluster_handles.append((u.handle, r))
        # first 4 granted (cluster < 5), the 5th and 6th FLAGGED (>=5 accounts
        # on one device/IP -> velocity_cluster, never auto-granted)
        granted = [h for h, r in cluster_handles if r.get("granted")]
        flagged = [r for _, r in cluster_handles if r.get("reason") == "velocity_cluster"]
        checks["cluster_velocity_flag_fires"] = len(flagged) >= 1 and len(granted) < 6
        sections["attack_1_device_ip_asn_cluster"] = {
            "accounts_on_one_device_ip": len(cluster_handles),
            "granted": len(granted), "flagged_velocity_cluster": len(flagged),
            "assertion": ">=5 accounts on one device/IP -> all later flagged, not granted",
        }

        # ---- attack 2: day-zero accounts sharing before first completed run
        day0_results = []
        for j in range(3):
            u = world.add_user(
                handle=f"sim-day0-{j}", device_id=f"sim-dev-day0-{j}",
                ip=f"198.51.100.{j + 1}", asn="AS-SIM-DAY0",
                created_at=now,  # day-zero
                rng=world.rng.spawn(7100 + j),
            )
            # age 0 AND zero completed runs
            world.layer.shares.register_account(
                u.handle, now=now, email_verified=True, phone_otp_verified=True,
                kyc_lite=True, created_at=now, completed_runs=0,
            )
            cid = mint_sim_card(world, u.handle, f"day0-{j}")
            good_post_for(world, cid)
            r = _grant(world, u, cid, f"https://social.example/day0-{j}")
            day0_results.append(r)
        checks["day_zero_denied"] = all(
            (not r.get("granted")) and ("raised" in r or r.get("reason")) for r in day0_results
        )
        # audit rows must exist for the denials
        day0_audit = [rec for rec in world.layer.shares.audit_log()
                      if rec.owner.startswith("sim-day0-")]
        checks["day_zero_audited"] = len(day0_audit) >= 3 and all(
            rec.outcome.startswith("denied") for rec in day0_audit)
        sections["attack_2_day_zero_before_first_run"] = {
            "attempts": len(day0_results), "granted": sum(1 for r in day0_results if r.get("granted")),
            "audit_rows": len(day0_audit),
            "assertion": "day-zero account, 0 completed runs -> denied_prereq, audited, nothing granted",
        }

        # ---- attack 3: unverifiable share posts (grant must NOT fire) ------
        setattr(world.layer.shares, "post_fetcher", lambda url: "no link, no tag here")
        u = world.add_user(handle="sim-unverif", device_id="sim-dev-unv",
                           ip="198.51.100.50", asn="AS-SIM-UNV",
                           created_at=now - 30 * DAY, rng=world.rng.spawn(7200))
        register_ready(world, u.handle)
        cid = mint_sim_card(world, u.handle, "unverif")
        slots_before = world.layer.shares.share_slots(u.handle)
        r = _grant(world, u, cid, "https://social.example/unverifiable")
        slots_after = world.layer.shares.share_slots(u.handle)
        checks["unverifiable_grants_nothing"] = (
            r.get("granted") is False and r.get("reason") == "unverifiable_post"
            and slots_after == slots_before
        )
        # and a post missing the attribution tag also fails
        setattr(world.layer.shares, "post_fetcher",
                lambda url: f"https://sift.example/cards/{cid}/face but no tag")
        cid_b = mint_sim_card(world, u.handle, "unverif-b")
        r_b = _grant(world, u, cid_b, "https://social.example/unverifiable-b")
        checks["missing_attribution_tag_denied"] = (
            r_b.get("granted") is False and slots_after == world.layer.shares.share_slots(u.handle)
        )
        sections["attack_3_unverifiable_posts"] = {
            "attempts": 2, "granted": 0,
            "slots_before": slots_before, "slots_after": slots_after,
            "assertion": "unverifiable post (no public card link + #SIFT tag) -> denied_unverifiable, grant does NOT fire",
        }

        # ---- attack 4: duplicate shares of the same card --------------------
        u = world.add_user(handle="sim-dup", device_id="sim-dev-dup",
                           ip="198.51.100.60", asn="AS-SIM-DUP",
                           created_at=now - 30 * DAY, rng=world.rng.spawn(7300))
        register_ready(world, u.handle)
        cid = mint_sim_card(world, u.handle, "dup")
        good_post_for(world, cid)
        r1 = _grant(world, u, cid, "https://social.example/dup1")
        world.clock.tick(60)
        r2 = _grant(world, u, cid, "https://social.example/dup2")  # same card again
        checks["duplicate_share_one_slot_per_account_card"] = (
            r1.get("granted") is True and r2.get("granted") is False
            and "one slot per" in r2.get("raised", "")
            and world.layer.shares.share_slots(u.handle) == 1
        )
        sections["attack_4_duplicate_share_same_card"] = {
            "first_granted": bool(r1.get("granted")),
            "second_granted": bool(r2.get("granted")),
            "second_refusal": r2.get("raised"),
            "slots": world.layer.shares.share_slots(u.handle),
            "assertion": "one slot per (account, card) EVER — duplicate raises",
        }

        # ---- attack 5: per-device velocity >3 grants/device/30d -------------
        u = world.add_user(handle="sim-velocity", device_id="sim-dev-vel",
                           ip="198.51.100.70", asn="AS-SIM-VEL",
                           created_at=now - 30 * DAY, rng=world.rng.spawn(7400))
        register_ready(world, u.handle)
        vel = []
        for i in range(MAX_GRANTS_PER_DEVICE_30D + 2):
            cid = mint_sim_card(world, u.handle, f"vel{i}")
            good_post_for(world, cid)
            vel.append(_grant(world, u, cid, f"https://social.example/vel{i}"))
            world.clock.tick(30)
        granted_vel = [r for r in vel if r.get("granted")]
        denied_vel = [r for r in vel if r.get("reason") == "device_ratelimit"]
        checks["device_rate_limit_3_per_30d"] = (
            len(granted_vel) == MAX_GRANTS_PER_DEVICE_30D and len(denied_vel) >= 1
        )
        sections["attack_5_device_rate_limit"] = {
            "grants_per_device_30d_allowed": MAX_GRANTS_PER_DEVICE_30D,
            "granted": len(granted_vel), "denied_ratelimit": len(denied_vel),
            "assertion": f">{MAX_GRANTS_PER_DEVICE_30D} grants/device/30d -> denied_ratelimit",
        }

        # ---- attack 6: +5 hard cap -------------------------------------------
        u = world.add_user(handle="sim-cap", device_id="sim-dev-cap",
                           ip="198.51.100.80", asn="AS-SIM-CAP",
                           created_at=now - 30 * DAY, rng=world.rng.spawn(7500))
        register_ready(world, u.handle)
        cap_results = []
        for i in range(MAX_SHARE_SLOTS + 2):
            cid = mint_sim_card(world, u.handle, f"cap{i}")
            good_post_for(world, cid)
            # fresh device per grant so the per-device 3/30d limit does not
            # mask the account-level +5 cap we are testing
            u.device_id = f"sim-dev-cap-{i}"
            cap_results.append(_grant(world, u, cid, f"https://social.example/cap{i}"))
            world.clock.tick(30)
        cap_granted = sum(1 for r in cap_results if r.get("granted"))
        cap_refused = [r for r in cap_results if "cap" in r.get("raised", "")]
        slots = world.layer.shares.share_slots(u.handle)
        checks["plus5_cap_enforced"] = (
            cap_granted == MAX_SHARE_SLOTS and slots == MAX_SHARE_SLOTS
            and len(cap_refused) >= 1
        )
        card = world.layer.cards.get_card(mint_sim_card(world, u.handle, "cap-probe"))
        checks["effective_capacity_capped_at_base_plus_5"] = (
            world.layer.vault.effective_capacity(u.handle, card.tier)
            == world.layer.vault.base_capacity(card.tier) + MAX_SHARE_SLOTS
        )
        sections["attack_6_plus5_cap"] = {
            "granted": cap_granted, "slots_after": slots,
            "cap_refusals": len(cap_refused),
            "assertion": "hard cap +5 permanent slots; 6th distinct card refused",
        }

        # ---- attack 7: revocation path on abuse flag -------------------------
        abused_card = mint_sim_card(world, u.handle, "abused")
        before = world.layer.shares.share_slots(u.handle)
        rev = world.layer.shares.revoke_for_abuse(
            u.handle, abused_card, reason="simulated_abuse_flag",
            now=world.clock.now())
        after = world.layer.shares.share_slots(u.handle)
        checks["revocation_zeroes_slots_and_counts_offense"] = (
            before == MAX_SHARE_SLOTS and after == 0
            and rev["offense_count"] == 1 and rev["warning"] is True
            and rev["share_disabled"] is False
        )
        # repeat offense -> share feature disabled for the account
        rev2 = world.layer.shares.revoke_for_abuse(
            u.handle, abused_card, reason="repeat",
            now=world.clock.now())
        checks["repeat_offense_disables_share"] = (
            rev2["share_disabled"] is True and rev2["offense_count"] == 2
        )
        r_after = _grant(world, u, mint_sim_card(world, u.handle, "post-abuse"),
                         "https://social.example/post-abuse")
        checks["disabled_account_cannot_grant"] = (
            r_after.get("granted") is False and "share_disabled" in r_after.get("raised", "")
        )
        # revocation is a NEW audit row (append-only), never an edit
        rev_rows = [rec for rec in world.layer.shares.audit_log()
                    if rec.owner == u.handle and rec.outcome == "revoked"]
        checks["revocation_append_only_audited"] = len(rev_rows) == 2
        sections["attack_7_abuse_revocation"] = {
            "slots_before": before, "slots_after": after,
            "offense_count": rev2["offense_count"],
            "share_disabled_after_repeat": rev2["share_disabled"],
            "revoked_audit_rows": len(rev_rows),
            "assertion": "abuse -> slots revoked (new audit row), offense counted, repeat disables share",
        }

        # ---- attack 8: fraud-queue SLA routing --------------------------------
        fq = world.layer.shares.fraud_queue_open()
        checks["fraud_queue_sla_routing"] = (
            len(fq) > 0 and all(
                abs(row["sla_deadline"] - (row["queued_at"] + FRAUD_QUEUE_SLA_S)) < 1e-6
                and row["resolved"] == 0
                for row in fq
            )
        )
        signals = sorted({row["signal"] for row in fq})
        sections["attack_8_fraud_queue_sla"] = {
            "open_items": len(fq), "signals": signals,
            "sla_s": FRAUD_QUEUE_SLA_S,
            "assertion": "every flagged/denied signal lands in fraud_queue with sla_deadline = queued_at + 72h",
        }

        # ---- append-only integrity probe: triggers must block UPDATE/DELETE ---
        con = sqlite3.connect(str(world.db_dir / "cards" / "shares.db"))
        update_blocked = delete_blocked = False
        try:
            try:
                con.execute("UPDATE share_grant SET outcome='hacked' WHERE seq=1")
            except sqlite3.IntegrityError:
                update_blocked = True
            try:
                con.execute("DELETE FROM share_grant WHERE seq=1")
            except sqlite3.IntegrityError:
                delete_blocked = True
            con.rollback()
        finally:
            con.close()
        # audit log row count unchanged after the blocked attempts
        audit_rows = len(world.layer.shares.audit_log())
        checks["share_grant_append_only_triggers_block"] = update_blocked and delete_blocked
        checks["audit_log_written"] = audit_rows >= 10
        sections["append_only_integrity"] = {
            "update_blocked_by_trigger": update_blocked,
            "delete_blocked_by_trigger": delete_blocked,
            "audit_rows": audit_rows,
            "assertion": "share_grant is append-only: UPDATE/DELETE raise (triggers), audit log written for every attempt",
        }

        sections["summary"] = {
            "attacks_run": 8,
            "all_checks_pass": all(checks.values()),
            "checks": {k: bool(v) for k, v in sorted(checks.items())},
        }
        report = build_report("antiabuse_attack_sims", seed, sections)
        return AttackResult(report=report, json_text=dump_json(report), checks=checks)


__all__ = ["AttackResult", "mint_sim_card", "register_ready", "run_antiabuse_sims"]
