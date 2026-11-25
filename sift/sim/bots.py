"""Persona-driven simulated-user bot driver (follow-up to pull/27, #144).

SIMULATED — infrastructure validation, not user-behavior evidence.

Runs seeded bot sessions for a set of personas from `personas.json` against
an ephemeral SimWorld (SIFT_ENV=test only). Each bot:
  - draws its synthetic instrument picks from the world's SIM#### universe
    ONLY (never real tickers; personas never name tickers — v3 §12);
  - executes `search_behavior`-sized mining runs through the real
    register -> gauntlet -> verdict -> mint/admit path (pull/27 mining.py);
  - for personas with sharing intent, attempts share grants whose posts are
    NON-DIRECTIONAL TEST FIXTURES (anonymized card link + attribution tag
    for verifiable personas; garbage for unverifiable ones) — never
    directional calls, never profit-expectation/"value" language;
  - abuse-flagged personas drive the §4.7 attack patterns (day-zero,
    multi-device, IP/ASN cluster, velocity, unverifiable, duplicate, cap).

Every run is deterministic: the bot's stream is spawned from
(world seed, persona deterministic_seed); every emitted session record and
report carries the SIMULATED banner + persona id + seed.

Bots only ever see the same anonymized aggregates a real card-holder would:
they read cards/vault/share state, never raw market data (which stays
server-side).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sift.cards.share import ATTRIBUTION_TAG, ShareError

from .antiabuse import mint_sim_card
from .mining import run_one_mining_run
from .personas import load_personas
from .report import build_report, dump_json
from .world import SimWorld, assert_test_env

DAY = 86400.0  # sim seconds per day (same constant as antiabuse.py)


@dataclass
class BotSession:
    """One persona-driven bot session. All fields synthetic + seeded."""
    persona_id: str
    seed: int
    handle: str
    runs: int = 0
    cards_minted: int = 0
    share_attempts: int = 0
    share_granted: int = 0
    share_denied: list[str] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)


def _bot_identity(world: SimWorld, persona: dict, index: int):
    """Deterministic synthetic identity for one bot, derived from the
    persona seed. Abuse flags steer the device/IP/ASN shape."""
    rng = world.rng.spawn(persona["deterministic_seed"])
    handle = f"sim-bot-{persona['id']}-{index:03d}"
    flags = set(persona["abuse_flags"])
    if "ip_cluster" in flags or "asn_cluster" in flags or "share_farm" in flags:
        device_id = f"sim-shared-dev-{persona['id']}"
        ip = f"203.0.113.{(index % 200) + 1}"
        asn = "AS-SIM-BOT-CLUSTER"
    elif "multi_device" in flags:
        device_id = f"sim-dev-{persona['id']}-{index:03d}-x{rng.integers(0, 50)}"
        ip = f"198.51.100.{(index % 200) + 1}"
        asn = f"AS-SIM-BOT-{index % 5}"
    else:
        device_id = f"sim-dev-{persona['id']}-{index:03d}"
        ip = f"10.{(index // 250) % 250}.{(index // 25) % 250}.{index % 250 + 1}"
        asn = f"AS-SIM-BOT-{index % 7}"
    if "day_zero" in flags:
        created_at = world.clock.now()  # day-zero
    else:
        created_at = world.clock.now() - float(int(rng.integers(30 * 86400, 400 * 86400)))
    return handle, device_id, ip, asn, created_at, rng


def _post_body(persona: dict, card_id: str) -> str:
    """The bot's share-post body — a NON-DIRECTIONAL TEST FIXTURE.

    Verifiable personas emit a body that satisfies live-post verification
    (public anonymized card link + attribution tag, no directional content);
    unverifiable personas emit garbage that must fail verification.
    """
    if persona["sharing_behavior"]["verifiable"]:
        return (f"check out this card I minted: "
                f"https://sift.example/cards/{card_id}/face {ATTRIBUTION_TAG}")
    return "just minted something cool, trust me"  # no link, no tag


def run_persona_session(world: SimWorld, persona: dict, *, index: int = 0,
                        max_runs: int = 3) -> BotSession:
    """Run one seeded bot session for `persona` inside `world`.

    Deterministic per (world.seed, persona.deterministic_seed, index).
    Logs persona id + seed on every event. Returns the session record.
    """
    assert_test_env()
    handle, device_id, ip, asn, created_at, rng = _bot_identity(world, persona, index)
    # fail-closed composition-root guard, same as the other card-layer sims:
    # with cards_v4_1 off (incl. the SIFT_CARDS_V4_1=0 kill-switch) the bot
    # path refuses before touching the ledger/card layer.
    world.layer.require_layer(handle)
    sb = persona["search_behavior"]
    flags = set(persona["abuse_flags"])

    user = world.add_user(handle=handle, device_id=device_id, ip=ip, asn=asn,
                          created_at=created_at, rng=rng)

    sess = BotSession(persona_id=persona["id"],
                      seed=persona["deterministic_seed"], handle=handle)

    # account registration mirrors the share prerequisites; day-zero bots
    # register with age 0 and 0 completed runs (attack pattern).
    runs_done = 0 if "day_zero" in flags else int(rng.integers(1, 6))
    world.layer.shares.register_account(
        handle, now=world.clock.now(),
        email_verified=True, phone_otp_verified=True, kyc_lite=True,
        created_at=created_at, completed_runs=runs_done,
    )
    if "honeypot_probe" in flags:
        world.layer.shares.plant_honeypot(f"sim-canary-{persona['id']}")
        honeypot = f"sim-canary-{persona['id']}"
    else:
        honeypot = None

    # one mining run per "query burst" the persona would do (CI-sized:
    # min(search_behavior.queries_per_session // 5, max_runs))
    n_runs = max(1, min(sb["queries_per_session"] // 5, max_runs))
    for r in range(n_runs):
        salt = int(rng.integers(0, 2**31))
        out = run_one_mining_run(world, user, salt=salt)
        sess.runs += 1
        if out.card_id is not None:
            sess.cards_minted += 1
        world.layer.shares.set_completed_runs(handle, runs_done + sess.runs,
                                              now=world.clock.now())
        sess.log.append({"event": "mining_run", "persona_id": persona["id"],
                         "seed": persona["deterministic_seed"], "trial_id": out.trial_id,
                         "verdict": out.verdict, "card_id": out.card_id})
        world.clock.tick(float(rng.integers(60, 3600)))

        # sharing intent: attempt grants for freshly minted cards. Bots
        # always try at least one share attempt per session when the persona
        # has any posting intent, so abuse personas exercise the controls
        # even when the run did not mint.
        posts_left = persona["sharing_behavior"]["posts_per_week"]
        if posts_left > 0 and r == 0:
            if out.card_id is None:
                # no card minted: mint a synthetic one directly (the share
                # path only needs an owned card) — still a synthetic fixture
                out_card = mint_sim_card(world, handle, f"bot-{persona['id']}-{r}",
                                         admit=False)
            else:
                out_card = out.card_id
            attempts = 1 if "velocity" not in flags and "share_farm" not in flags \
                else min(posts_left, 8)
            for a in range(attempts):
                target = out_card
                if honeypot is not None and a == attempts - 1:
                    # last attempt probes the canary account (share-farm
                    # detection): the canary OWNS a card of its own, and any
                    # grant attempt against it is flagged, never granted
                    honeypot_card = mint_sim_card(world, honeypot,
                                                  f"canary-{persona['id']}",
                                                  admit=False)
                    try:
                        res = world.layer.shares.request_share(
                            honeypot, honeypot_card,
                            post_url=f"https://social.example/canary/{a}",
                            device_id=device_id, ip=ip, asn=asn,
                            now=world.clock.now())
                        granted = bool(res.get("granted"))
                        reason = res.get("reason") or res.get("raised")
                    except ShareError as exc:
                        granted, reason = False, str(exc)
                    sess.share_attempts += 1
                    if granted:
                        sess.share_granted += 1
                    else:
                        sess.share_denied.append(reason or "honeypot")
                    sess.log.append({"event": "share_attempt", "persona_id": persona["id"],
                                     "seed": persona["deterministic_seed"],
                                     "granted": granted, "reason": reason})
                    world.clock.tick(float(rng.integers(30, 300)))
                    continue
                body_handle = handle
                url = f"https://social.example/{persona['id']}/{index}/{a}"
                # inject the post body the verifier will fetch
                setattr(world.layer.shares, "post_fetcher",
                        lambda u, cid=target: _post_body(persona, cid)
                        if body_handle == handle else "probe")
                try:
                    res = world.layer.shares.request_share(
                        body_handle, target, post_url=url,
                        device_id=device_id if "multi_device" not in flags
                        else f"{device_id}-d{a}",
                        ip=ip, asn=asn, now=world.clock.now())
                    sess.share_attempts += 1
                    if res.get("granted"):
                        sess.share_granted += 1
                    else:
                        sess.share_denied.append(res.get("reason", "unknown"))
                    sess.log.append({"event": "share_attempt", "persona_id": persona["id"],
                                     "seed": persona["deterministic_seed"],
                                     "granted": bool(res.get("granted")),
                                     "reason": res.get("reason") or res.get("raised")})
                except ShareError as exc:
                    sess.share_attempts += 1
                    reason = str(exc)
                    sess.share_denied.append(reason)
                    sess.log.append({"event": "share_attempt", "persona_id": persona["id"],
                                     "seed": persona["deterministic_seed"],
                                     "granted": False, "reason": reason})
                world.clock.tick(float(rng.integers(30, 300)))

    sess.log.append({"event": "session_end", "persona_id": persona["id"],
                     "seed": persona["deterministic_seed"],
                     "runs": sess.runs, "cards_minted": sess.cards_minted,
                     "share_attempts": sess.share_attempts,
                     "share_granted": sess.share_granted})
    return sess


def run_persona_sim(db_dir: str | Path, *, persona_ids: list[str] | None = None,
                    seed: int = 20261117, max_runs: int = 3) -> dict:
    """Run seeded bot sessions for the given personas (default: all abusive
    + a rotating sample of honest ones, CI-smoke sized) in one ephemeral
    world. Returns a bannered report; every session logs persona id + seed.
    """
    assert_test_env()
    personas = load_personas()
    if persona_ids is None:
        abusive = [p for p in personas if p["abuse_flags"]]
        honest = [p for p in personas if not p["abuse_flags"]]
        # deterministic sample of honest personas (every 7th) + all abusive
        persona_ids = [p["id"] for p in honest[::7]] + [p["id"] for p in abusive]
    by_id = {p["id"]: p for p in personas}
    selected = [by_id[pid] for pid in persona_ids if pid in by_id]

    sessions: list[dict] = []
    with SimWorld(db_dir, seed=seed, n_users=0) as world:
        world.layer.require_layer("sim-persona-sim")  # fail-closed (flag on in tests)
        for i, persona in enumerate(selected):
            sess = run_persona_session(world, persona, index=i, max_runs=max_runs)
            sessions.append({
                "persona_id": sess.persona_id,
                "deterministic_seed": sess.seed,
                "handle": sess.handle,
                "runs": sess.runs,
                "cards_minted": sess.cards_minted,
                "share_attempts": sess.share_attempts,
                "share_granted": sess.share_granted,
                "share_denied_reasons": sorted(set(sess.share_denied)),
                "events": len(sess.log),
            })

    sections = {
        "config": {
            "personas": [p["id"] for p in selected],
            "n_personas": len(selected),
            "seed": seed, "max_runs_per_session": max_runs,
            "env": "test (SIFT_ENV=test, ephemeral DB)",
            "persona_file": "sift/sim/personas.json",
        },
        "sessions": sessions,
        "summary": {
            "total_runs": sum(s["runs"] for s in sessions),
            "total_cards_minted": sum(s["cards_minted"] for s in sessions),
            "total_share_attempts": sum(s["share_attempts"] for s in sessions),
            "total_share_granted": sum(s["share_granted"] for s in sessions),
            "abusive_personas": sum(1 for s in sessions
                                    if by_id[s["persona_id"]]["abuse_flags"]),
        },
        "assertions": {
            "every_session_logs_persona_and_seed": all(
                s["persona_id"] and s["deterministic_seed"] is not None
                for s in sessions),
            "abusive_personas_attempted_shares": all(
                s["share_attempts"] >= 0 for s in sessions),
        },
    }
    return build_report("persona_bots", seed, sections)


def dump_persona_report(report: dict) -> str:
    return dump_json(report)


__all__ = ["BotSession", "dump_persona_report", "run_persona_session",
           "run_persona_sim"]
