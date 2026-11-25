"""Persona spec + seeded bot-session tests (SIMULATED — infrastructure
validation, not user-behavior evidence).

Covers the definition of done for the personas spec (issue #144):
1. personas.json loads and validates (schema / ranges / unique ids / seeds,
   50..100 entries, SIMULATED header on the file itself).
2. Three seeded bot sessions run end-to-end against the sim harness
   (pull/27) and every output carries the SIMULATED header; every session
   logs persona id + deterministic seed.
3. Determinism: same world seed + persona seed -> identical session log.
4. Abuse-flag targeting: an abusive persona's unverifiable posts are never
   granted (the §4.7 control fires through the persona-driven path too).
"""

import json

import pytest

from sift import flags
from sift.cards.service import LayerError
from sift.sim.bots import run_persona_session, run_persona_sim
from sift.sim.header import SIM_HEADER
from sift.sim.personas import PERSONAS_PATH, PersonaSpecError, load_personas, persona_by_id
from sift.sim.report import dump_json, render_text
from sift.sim.world import SimWorld


def test_personas_json_validates(sim_env):
    """50..100 entries, schema/ranges/unique ids+seeds, SIMULATED header."""
    personas = load_personas()
    assert 50 <= len(personas) <= 100
    ids = [p["id"] for p in personas]
    seeds = [p["deterministic_seed"] for p in personas]
    assert len(set(ids)) == len(ids)
    assert len(set(seeds)) == len(seeds)
    raw = json.loads(PERSONAS_PATH.read_text())
    assert raw["sim"] is True
    assert raw["header"].startswith("SIMULATED")
    # the six owner seed archetypes are present as named archetypes
    names = {p["name"] for p in personas}
    for n in ("Big-name anchor", "WSB trender", "Top-of-list pairer",
              "Substack verifier", "Influencer miner", "Card farmer"):
        assert n in names, n
    # deliberate abuse archetypes exist for the §4.7 sims
    flagged = {f for p in personas for f in p["abuse_flags"]}
    for f in ("day_zero", "multi_device", "ip_cluster", "velocity",
              "unverifiable_shares"):
        assert f in flagged, f


def test_personas_json_rejects_bad_specs(sim_env, tmp_path):
    """Schema violations raise PersonaSpecError (negative cases)."""
    raw = json.loads(PERSONAS_PATH.read_text())
    raw["personas"][0]

    def write(mutate):
        doc = json.loads(json.dumps(raw))
        mutate(doc)
        f = tmp_path / "bad.json"
        f.write_text(json.dumps(doc))
        return f

    # duplicate id
    def dup_id(doc):
        doc["personas"][1]["id"] = doc["personas"][0]["id"]
    with pytest.raises(PersonaSpecError, match="duplicate id"):
        load_personas(write(dup_id))
    # sophistication out of range
    def bad_soph(doc):
        doc["personas"][0]["sophistication"] = 6
    with pytest.raises(PersonaSpecError, match="sophistication"):
        load_personas(write(bad_soph))

    # unknown source
    def bad_src(doc):
        doc["personas"][0]["information_sources"] = ["carrier_pigeon"]
    with pytest.raises(PersonaSpecError, match="information_sources"):
        load_personas(write(bad_src))

    # too few personas
    def too_few(doc):
        doc["personas"] = doc["personas"][:10]
    with pytest.raises(PersonaSpecError, match="50..100"):
        load_personas(write(too_few))

    # missing SIMULATED header
    def no_header(doc):
        del doc["header"]
    with pytest.raises(PersonaSpecError, match="SIMULATED"):
        load_personas(write(no_header))


def _run_three_seeded_sessions(seed: int, tmp_path):
    """Three seeded bot sessions (one honest, one farmer, one abusive)
    against the pull/27 harness world."""
    personas = {p["id"]: p for p in load_personas()}
    picks = ["big-name-anchor", "card-farmer", "day-zero-sharer"]
    sessions = []
    with SimWorld(tmp_path, seed=seed, n_users=0) as world:
        world.layer.require_layer("sim-test")
        for i, pid in enumerate(picks):
            sessions.append(run_persona_session(world, personas[pid], index=i))
    return sessions


def test_three_seeded_bot_sessions_end_to_end(sim_env, tmp_path):
    """DoD: 3 seeded bot sessions end-to-end; each logs persona id + seed;
    the persona sim report carries the SIMULATED header on all outputs."""
    sessions = _run_three_seeded_sessions(20261117, tmp_path / "w")
    assert len(sessions) == 3
    for s in sessions:
        assert s.persona_id and s.seed is not None
        assert s.runs >= 1
        # every log event carries persona id + seed
        assert all(ev["persona_id"] == s.persona_id and ev["seed"] == s.seed
                   for ev in s.log)
    report = run_persona_sim(tmp_path / "sim", seed=20261117, max_runs=1)
    assert report["sim"] is True
    assert report["header"] == SIM_HEADER
    text = render_text(report)
    assert text.splitlines()[0] == SIM_HEADER
    payload = json.loads(dump_json(report))
    assert payload["header"] == SIM_HEADER
    for s in payload["sections"]["sessions"]:
        assert s["persona_id"] and isinstance(s["deterministic_seed"], int)


def test_bot_sessions_deterministic_same_seed(sim_env, tmp_path):
    """Same world seed -> identical session logs (seeded/deterministic)."""
    s1 = _run_three_seeded_sessions(777, tmp_path / "a")
    s2 = _run_three_seeded_sessions(777, tmp_path / "b")
    l1 = [(s.persona_id, s.seed, json.dumps(s.log, sort_keys=True, default=str))
          for s in s1]
    l2 = [(s.persona_id, s.seed, json.dumps(s.log, sort_keys=True, default=str))
          for s in s2]
    assert l1 == l2


def test_abusive_persona_unverifiable_shares_never_granted(sim_env, tmp_path):
    """A day-zero + unverifiable persona driving the bot path: the §4.7
    controls fire (denied_prereq / unverifiable), nothing is granted."""
    p = persona_by_id("day-zero-sharer")
    with SimWorld(tmp_path / "w", seed=99, n_users=0) as world:
        world.layer.require_layer("sim-test")
        sess = run_persona_session(world, p, index=0)
    assert sess.share_attempts >= 1
    assert sess.share_granted == 0
    assert sess.share_denied  # some denial reason recorded
    # day-zero: prereq denial (age / completed runs) must be among them
    assert any("age" in d or "prereq" in d or "unverifiable" in d or "run" in d
               for d in sess.share_denied)


def test_persona_bots_fail_closed_when_flag_off(sim_env, tmp_path):
    """cards_v4_1 off (incl. SIFT_CARDS_V4_1=0 kill-switch): the persona bot
    path refuses at the composition-root guard, exactly like the other
    card-layer sims (fail-closed)."""
    flags.set_flag("cards_v4_1", False)
    try:
        p = persona_by_id("card-farmer")
        with SimWorld(tmp_path / "w", seed=5, n_users=0) as world:
            with pytest.raises(LayerError):
                world.layer.require_layer("sim-test")
            with pytest.raises(LayerError):
                run_persona_session(world, p, index=0)
            with pytest.raises(LayerError):
                run_persona_sim(tmp_path / "sim2", seed=5, max_runs=1)
    finally:
        flags.reset()
