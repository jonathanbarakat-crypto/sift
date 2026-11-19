"""Simulated-user persona spec loader (pull/27 follow-up, issue #144).

SIMULATED — infrastructure validation, not user-behavior evidence.

`personas.json` is the machine-readable spec of the simulated-user personas
that drive the bot harness. It exists ONLY to build deterministic user bots
for infrastructure soak and anti-abuse testing. It is NOT evidence of real
user behavior: the mix, shares, and parameters are a DESIGN CHOICE for
coverage of the behavior space, not a market estimate, and must never be
presented as user-behavior evidence (spec §8 is answered from the limited
alpha/beta, pull/25 — never from sims).

Schema (one entry per persona):
  id                stable slug, unique across the file
  name              human-readable label
  archetype         coarse class (see ARCHETYPES)
  sophistication    1 (naive) .. 5 (sophisticated) — a sim knob, not a claim
  information_sources  subset of SOURCES — where the bot "gets ideas" from
  strategy          one of STRATEGIES — the connection family the bot hunts
  search_behavior   query patterns / session length / ticks per session
  minting_goal      cards-per-week target (sim load knob only)
  sharing_behavior  share-post frequency + verifiable or not (test fixture)
  abuse_flags       subset of ABUSE_FLAGS — §4.7 attack-sim targeting knobs
  deterministic_seed  int; bot run = f(world seed, persona seed) -> reproducible
  description       one short paragraph

Hard rules honored by this module and the spec file:
- synthetic instruments only (`SIM####` from the sim world); personas never
  name real tickers — bot inputs are drawn from the synthetic universe;
- no directional calls, no profit-expectation/"value" language anywhere;
  bot-generated share posts are NON-DIRECTIONAL test fixtures;
- bots only ever see the same anonymized aggregates a real card-holder
  would (raw market data stays server-side);
- every bot run logs persona id + seed; every report carries the SIM banner.
"""

from __future__ import annotations

import json
from pathlib import Path

PERSONAS_PATH = Path(__file__).parent / "personas.json"

SOURCES = (
    "wsb",            # r/wallstreetbets trending
    "r/stocktips",    # top-of-list posts
    "substack_free",  # free-tier stock-pick newsletters
    "influencers",    # hype finance influencers
    "screeners",      # stock screeners
    "fundamentals",   # filings / fundamentals
    "random",         # no source — random exploration
    "insider_ish",    # "insider-ish" claims (unverifiable tips)
)

STRATEGIES = (
    "pairs",       # mean-reversion / pairs hunting
    "lead_lag",    # lead-lag hunting
    "momentum",    # momentum chasing
    "narrative",   # story/theme driven
    "random",      # undirected search
    "copycat",     # copies what others posted / shared
)

ARCHETYPES = (
    "anchor",        # locks onto big names, searches around them
    "trender",       # rides what's trending on the casino boards
    "list_pairer",   # pairs whatever is at the top of a list
    "verifier",      # tries to verify newsletter/influencer picks via cards
    "miner",         # mines cards off hype
    "farmer",        # farms cards for the cards' sake
    "power_user",    # sophisticated, systematic
    "casual",        # low engagement, occasional runs
    "abusive",       # deliberate abuse archetype (§4.7 attack-sim target)
)

ABUSE_FLAGS = (
    "day_zero",            # account shares before first completed run
    "multi_device",        # same account across many device ids
    "ip_cluster",          # many accounts on one IP/ASN
    "asn_cluster",         # many accounts on one ASN
    "velocity",            # hammers share grants (>3/device/30d)
    "unverifiable_shares", # posts that fail live-post verification
    "duplicate_shares",    # re-shares the same card for another slot
    "cap_buster",          # tries to exceed the +5 slot cap
    "honeypot_probe",      # probes planted canary accounts
    "share_farm",          # farm of coordinated share accounts
)

REQUIRED_KEYS = (
    "id", "name", "archetype", "sophistication", "information_sources",
    "strategy", "search_behavior", "minting_goal", "sharing_behavior",
    "abuse_flags", "deterministic_seed", "description",
)


class PersonaSpecError(ValueError):
    """personas.json failed schema/range/uniqueness validation."""


def load_personas(path: str | Path = PERSONAS_PATH) -> list[dict]:
    """Load + validate personas.json. Raises PersonaSpecError on any
    schema/range/uniqueness violation. Returns the list of persona dicts
    (sorted by id)."""
    path = Path(path)
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or raw.get("sim") is not True:
        raise PersonaSpecError("personas.json must be an object with \"sim\": true")
    if not isinstance(raw.get("header"), str) or \
            not raw["header"].startswith("SIMULATED"):
        raise PersonaSpecError("personas.json must carry the SIMULATED header")
    personas = raw.get("personas")
    if not isinstance(personas, list) or not (50 <= len(personas) <= 100):
        raise PersonaSpecError(
            f"personas must be a list of 50..100 entries (got "
            f"{len(personas) if isinstance(personas, list) else type(personas).__name__})"
        )

    seen_ids: set[str] = set()
    seen_seeds: set[int] = set()
    for i, p in enumerate(personas):
        where = f"personas[{i}]"
        if not isinstance(p, dict):
            raise PersonaSpecError(f"{where}: not an object")
        missing = [k for k in REQUIRED_KEYS if k not in p]
        if missing:
            raise PersonaSpecError(f"{where}: missing keys {missing}")
        extra = [k for k in p if k not in REQUIRED_KEYS]
        if extra:
            raise PersonaSpecError(f"{where}: unknown keys {extra}")

        pid = p["id"]
        if not isinstance(pid, str) or not pid or pid != pid.lower() \
                or " " in pid:
            raise PersonaSpecError(f"{where}: id must be a lowercase slug, got {pid!r}")
        if pid in seen_ids:
            raise PersonaSpecError(f"{where}: duplicate id {pid!r}")
        seen_ids.add(pid)

        if not isinstance(p["name"], str) or not p["name"]:
            raise PersonaSpecError(f"{where}: name must be a non-empty string")
        if p["archetype"] not in ARCHETYPES:
            raise PersonaSpecError(
                f"{where} ({pid}): archetype {p['archetype']!r} not in {ARCHETYPES}")
        soph = p["sophistication"]
        if not isinstance(soph, int) or isinstance(soph, bool) or not 1 <= soph <= 5:
            raise PersonaSpecError(f"{where} ({pid}): sophistication must be int 1..5")

        srcs = p["information_sources"]
        if not isinstance(srcs, list) or not srcs or \
                any(s not in SOURCES for s in srcs):
            raise PersonaSpecError(
                f"{where} ({pid}): information_sources must be a non-empty "
                f"subset of {SOURCES}")
        if p["strategy"] not in STRATEGIES:
            raise PersonaSpecError(
                f"{where} ({pid}): strategy {p['strategy']!r} not in {STRATEGIES}")

        sb = p["search_behavior"]
        if not isinstance(sb, dict):
            raise PersonaSpecError(f"{where} ({pid}): search_behavior must be an object")
        for k, lo, hi in (("queries_per_session", 1, 200),
                          ("session_minutes", 1, 600),
                          ("ticks_per_session", 1, 50)):
            v = sb.get(k)
            if not isinstance(v, int) or isinstance(v, bool) or not lo <= v <= hi:
                raise PersonaSpecError(
                    f"{where} ({pid}): search_behavior.{k} must be int {lo}..{hi}")
        if not isinstance(sb.get("query_patterns"), list) or not sb["query_patterns"]:
            raise PersonaSpecError(
                f"{where} ({pid}): search_behavior.query_patterns must be a non-empty list")

        mg = p["minting_goal"]
        if not isinstance(mg, dict) or \
                not isinstance(mg.get("cards_per_week"), int) or \
                not 0 <= mg["cards_per_week"] <= 100:
            raise PersonaSpecError(
                f"{where} ({pid}): minting_goal.cards_per_week must be int 0..100")

        sh = p["sharing_behavior"]
        if not isinstance(sh, dict) or sh.get("verifiable") not in (True, False) or \
                not isinstance(sh.get("posts_per_week"), int) or \
                not 0 <= sh["posts_per_week"] <= 200:
            raise PersonaSpecError(
                f"{where} ({pid}): sharing_behavior needs verifiable: bool and "
                "posts_per_week: int 0..200")

        flags_ = p["abuse_flags"]
        if not isinstance(flags_, list) or any(f not in ABUSE_FLAGS for f in flags_):
            raise PersonaSpecError(
                f"{where} ({pid}): abuse_flags must be a subset of {ABUSE_FLAGS}")

        seed = p["deterministic_seed"]
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63:
            raise PersonaSpecError(f"{where} ({pid}): deterministic_seed must be int 0..2^63-1")
        if seed in seen_seeds:
            raise PersonaSpecError(f"{where} ({pid}): duplicate deterministic_seed {seed}")
        seen_seeds.add(seed)

        if not isinstance(p["description"], str) or not 10 <= len(p["description"]) <= 600:
            raise PersonaSpecError(
                f"{where} ({pid}): description must be a 10..600 char string")

    return sorted(personas, key=lambda p: p["id"])


def persona_by_id(pid: str, path: str | Path = PERSONAS_PATH) -> dict:
    for p in load_personas(path):
        if p["id"] == pid:
            return p
    raise PersonaSpecError(f"no persona with id {pid!r}")


__all__ = [
    "ABUSE_FLAGS", "ARCHETYPES", "PERSONAS_PATH", "REQUIRED_KEYS", "SOURCES",
    "STRATEGIES", "PersonaSpecError", "load_personas", "persona_by_id",
]
