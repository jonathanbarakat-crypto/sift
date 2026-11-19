"""TDD (RED) tests for the custom-deck identity collapse fix (kanban t_1b86e0f3).

Bug: register_custom builds config = {size, sector, family_mix, custom}
WITHOUT membership, so the dedupe key (scope, config, vintage) collapses every
custom deck of the same size into ONE deck_id — stratum_n undercounts (S6
multiplicity broken) and the stored membership is the first registration's
(provenance broken).

Fix spec option A: include a canonical membership hash (sorted
member_card_ids) in the custom config -> distinct memberships get distinct
deck_ids; identical membership still dedupes; stratum_n counts each attempt.

Tests:
  (a) two custom decks, same size/scope/vintage, different members -> different ids
  (b) identical members -> dedupe to one id (order-insensitive)
  (c) stratum_n equals number of distinct registered custom decks
  (d) run_s6 results traceable to the registered membership (per-deck trials)
  (e) existing semantics unchanged: custom stays unranked/informational even
      on an S6 pass, and ranked dedupe is untouched.
"""

import json

import numpy as np
import pytest

from sift import flags
from sift.decks import DeckError, DeckRegistry, bucket_config, config_hash_of
from sift.decks.service import DeckService
from tests.decks.test_decks_v4_2 import (
    SCOPE,
    SECRET,
    VINTAGE,
    _mint_cards,
    _planted_basket,
)


@pytest.fixture(autouse=True)
def _flags_reset():
    flags.reset()
    yield
    flags.reset()


@pytest.fixture()
def cards(tmp_path):
    from sift.cards import CardRegistry
    c = CardRegistry(tmp_path / "cards.db", secret=SECRET)
    yield c
    c.close()


@pytest.fixture()
def decks(tmp_path):
    d = DeckRegistry(tmp_path / "decks.db", secret=SECRET)
    yield d
    d.close()


@pytest.fixture()
def svc(cards, decks):
    return DeckService(cards, decks)


# (a) distinct memberships -> distinct deck ids, same size/scope/vintage
def test_custom_decks_different_members_get_different_ids(svc, cards, decks):
    cs = _mint_cards(cards, n=10)
    ids = [c.card_id for c in cs]
    d1 = svc.register_custom(scope_id=SCOPE, member_card_ids=ids[:5],
                             owner_handle="alice", data_vintage_hash=VINTAGE, now=1.0)
    d2 = svc.register_custom(scope_id=SCOPE, member_card_ids=ids[5:],
                             owner_handle="alice", data_vintage_hash=VINTAGE, now=2.0)
    assert d1.size == d2.size == 5
    assert d1.scope_id == d2.scope_id == SCOPE
    assert d1.data_vintage_hash == d2.data_vintage_hash == VINTAGE
    assert d1.deck_id != d2.deck_id          # THE BUG: previously collapsed to one id
    assert len(decks.all_decks()) == 2
    # each deck row stores ITS OWN membership (provenance)
    assert json.loads(decks.get_deck(d1.deck_id).member_card_ids) == sorted(ids[:5])
    assert json.loads(decks.get_deck(d2.deck_id).member_card_ids) == sorted(ids[5:])


# (b) identical membership -> dedupe to one id (order-insensitive)
def test_custom_decks_identical_members_dedupe_to_one_id(svc, cards, decks):
    cs = _mint_cards(cards, n=8)
    ids = [c.card_id for c in cs[:5]]
    d1 = svc.register_custom(scope_id=SCOPE, member_card_ids=ids,
                             owner_handle="alice", data_vintage_hash=VINTAGE, now=1.0)
    d2 = svc.register_custom(scope_id=SCOPE, member_card_ids=list(reversed(ids)),
                             owner_handle="alice", data_vintage_hash=VINTAGE, now=2.0)
    assert d1.deck_id == d2.deck_id          # same membership -> same deck (cached)
    assert len(decks.all_decks()) == 1
    # a different vintage is still a distinct deck
    d3 = svc.register_custom(scope_id=SCOPE, member_card_ids=ids,
                             owner_handle="alice", data_vintage_hash="v-other", now=3.0)
    assert d3.deck_id != d1.deck_id


# (c) stratum_n equals the number of distinct registered custom decks
def test_stratum_n_counts_distinct_custom_decks(svc, cards, decks):
    cs = _mint_cards(cards, n=12)
    ids = [c.card_id for c in cs]
    seen = set()
    for i in range(5):
        members = ids[i:i + 5] if i + 5 <= len(ids) else ids[:5]
        if i == 4:
            members = ids[:5]                 # duplicate of the first membership
        d = svc.register_custom(scope_id=SCOPE, member_card_ids=members,
                                owner_handle="alice", data_vintage_hash=VINTAGE,
                                now=float(i))
        seen.add(d.deck_id)
    assert len(seen) == 4                     # 5 registrations, 4 distinct memberships
    assert decks.stratum_n(SCOPE) == 4        # THE BUG: previously 1


# (d) run_s6 results are traceable to the registered membership
def test_run_s6_trials_traceable_to_each_registered_membership(svc, cards, decks):
    cs = _mint_cards(cards, n=10)
    ids = [c.card_id for c in cs]
    d1 = svc.register_custom(scope_id=SCOPE, member_card_ids=ids[:5],
                             owner_handle="alice", data_vintage_hash=VINTAGE, now=1.0)
    d2 = svc.register_custom(scope_id=SCOPE, member_card_ids=ids[5:],
                             owner_handle="alice", data_vintage_hash=VINTAGE, now=2.0)
    R = _planted_basket(T=1260, K=5, seed=13)
    names = [{f"t{i}"} for i in range(5)]
    r1 = svc.run_s6(d1.deck_id, component_returns=R, component_names=names,
                    owner_handle="alice", now=10.0)
    r2 = svc.run_s6(d2.deck_id, component_returns=R, component_names=names,
                    owner_handle="alice", now=11.0)
    # each deck has its OWN deck_trial event (previously all 500 landed on one deck)
    ev1 = decks.events("deck_trial", deck_id=d1.deck_id)
    ev2 = decks.events("deck_trial", deck_id=d2.deck_id)
    assert len(ev1) == 1 and len(ev2) == 1
    assert ev1[0]["deck_id"] == d1.deck_id and ev2[0]["deck_id"] == d2.deck_id
    # the returned deck is the one whose membership was run (provenance read-back)
    assert json.loads(r1["deck"].member_card_ids) == sorted(ids[:5])
    assert json.loads(r2["deck"].member_card_ids) == sorted(ids[5:])
    assert r1["s6"]["K"] == 5 and r2["s6"]["K"] == 5
    # multiplicity now sees both attempts
    assert r2["s6"]["n_stratum"] == decks.stratum_n(SCOPE) == 2


# (e) semantics unchanged: custom stays unranked/informational; ranked dedupe intact
def test_custom_semantics_unchanged_after_fix(svc, cards, decks):
    cs = _mint_cards(cards, n=10)
    ids = [c.card_id for c in cs]
    d = svc.register_custom(scope_id=SCOPE, member_card_ids=ids[:5],
                            owner_handle="alice", data_vintage_hash=VINTAGE, now=1.0)
    assert d.kind == "custom" and d.rank is None
    R = _planted_basket(T=1260, K=5, seed=17)
    res = svc.run_s6(d.deck_id, component_returns=R,
                     component_names=[{f"u{i}"} for i in range(5)],
                     owner_handle="alice", now=2.0)
    assert res["deck"].rank is None                       # never ranked
    assert res["s6"]["informational_only"] is True
    ev = decks.events("deck_trial", deck_id=d.deck_id)
    payload = json.loads(ev[-1]["payload_json"])
    assert payload["informational_only"] is True and payload["rank"] is None
    # ranked dedupe on (scope, bucket config, vintage) is untouched
    bucket = bucket_config(size=5, sector="all", family_mix=("pairs",))
    r1 = svc.register_ranked(scope_id=SCOPE, bucket=bucket, owner_handle="alice",
                             data_vintage_hash=VINTAGE, now=3.0)
    r2 = svc.register_ranked(scope_id=SCOPE, bucket=bucket, owner_handle="bob",
                             data_vintage_hash=VINTAGE, now=4.0)
    assert r1.deck_id == r2.deck_id
    # ... and a ranked deck never collides with a custom deck of the same size
    assert r1.deck_id != d.deck_id
    assert config_hash_of(bucket) != decks.get_deck(d.deck_id).config_hash


# membership hash is canonical: order-insensitive, duplicates-insensitive
def test_custom_membership_hash_canonical(svc, cards, decks):
    cs = _mint_cards(cards, n=6)
    ids = [c.card_id for c in cs[:4]]
    d1 = svc.register_custom(scope_id=SCOPE, member_card_ids=ids,
                             owner_handle="alice", data_vintage_hash=VINTAGE, now=1.0)
    d2 = svc.register_custom(scope_id=SCOPE, member_card_ids=ids[::-1] + ids[:1],
                             owner_handle="alice", data_vintage_hash=VINTAGE, now=2.0)
    assert d1.deck_id == d2.deck_id          # sorted+deduped membership -> same id
    assert len(decks.all_decks()) == 1


def test_custom_deck_still_requires_nonempty_and_owned(svc, cards):
    with pytest.raises(DeckError):
        svc.decks.register_custom(scope_id=SCOPE, member_card_ids=[],
                                  owner_handle="alice", data_vintage_hash=VINTAGE)
    cs = _mint_cards(cards, n=3, owner="alice")
    with pytest.raises(Exception):
        svc.register_custom(scope_id=SCOPE, member_card_ids=[cs[0].card_id],
                            owner_handle="mallory", data_vintage_hash=VINTAGE)
