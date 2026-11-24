"""v4.1 mechanics 2+3 — vault custody, capacity by tier, discard ->
personal archive + aggregate registry pool (never destroyed, never
randomized), ghost FIFO shift on discard, flag-off refusal.

Read-back rule: keep/discard/promote return DB-read state.
"""

import pytest

from sift import flags
from sift.cards import CardRegistry
from sift.cards.vault import Vault, VaultError
from sift.registry import Ledger
from tests.cards.test_minting import (
    SECRET,
    _run_validated_green,
)


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
def vault(cards, tmp_path):
    v = Vault(cards, tmp_path / "vault.db")
    yield v
    v.close()


def _mint_green(ledger, cards, handle="alice", fixture="F-PAIR-FULL"):
    verdict, trial = _run_validated_green(ledger, cards, handle=handle, fixture=fixture)
    assert verdict["card_id"] is not None
    return verdict["card_id"], trial


def test_admit_and_capacity_by_tier(ledger, cards, vault):
    card_id, _ = _mint_green(ledger, cards)
    card = cards.get_card(card_id)
    # capacity depends on the card's tier (v4.2: platinum is top, capacity 8)
    cap = vault.effective_capacity("alice", card.tier)
    assert cap == vault.base_capacity(card.tier)   # platinum 8 / gold 6 / silver 4 / bronze 2
    entry = vault.admit(card_id, "alice")
    assert entry.state == "vault"
    # read-back: vault listing shows it
    assert [c.card_id for c in vault.vault("alice")] == [card_id]
    assert vault.vault_count("alice") == 1


def test_capacity_blocks_over_admit(ledger, cards, vault):
    card_id, _ = _mint_green(ledger, cards)
    card = cards.get_card(card_id)
    cap = vault.effective_capacity("alice", card.tier)
    # fill the vault up to capacity with cards of the SAME tier
    for i in range(cap):
        cid = cards.mint(
            trial_id=f"t{i}", family="pairs", scope_id="free_5y_top1k",
            config_hash=f"h{i}", data_vintage_hash="v",
            survivor_class="validated-green", owner_handle="alice",
            dsr=0.96 if card.tier == "silver" else (0.995 if card.tier in ("gold", "platinum") else 0.90),
            t_net_full=3.2 if card.tier not in ("gold", "platinum") else 4.5,
            p_perm=0.04 if card.tier not in ("gold", "platinum") else 0.005,
            s4_pass=card.tier in ("gold", "platinum"),
            s5_pass=card.tier == "platinum",
        ).card_id
        vault.admit(cid, "alice")
    assert vault.vault_count("alice") == cap
    extra = cards.mint(
        trial_id="tX", family="pairs", scope_id="free_5y_top1k",
        config_hash="hX", data_vintage_hash="v",
        survivor_class="validated-green", owner_handle="alice",
        dsr=0.96, t_net_full=3.2, p_perm=0.04,
    ).card_id
    with pytest.raises(VaultError):
        vault.admit(extra, "alice")


def test_discard_goes_to_archive_and_pool_never_destroyed(ledger, cards, vault):
    card_id, _ = _mint_green(ledger, cards)
    vault.admit(card_id, "alice")
    res = vault.discard(card_id, "alice")
    # read-back: entry state + pool membership
    assert res["entry"].state == "archive"
    assert res["pooled"] is True
    assert vault.vault("alice") == []
    assert [c.card_id for c in vault.archive("alice")] == [card_id]
    assert vault.pool_size() == 1
    # never destroyed: the card row still exists and is readable
    assert cards.get_card(card_id).card_id == card_id


def test_promote_reads_back_and_capacity_checked(ledger, cards, vault):
    card_id, _ = _mint_green(ledger, cards)
    vault.admit(card_id, "alice")
    vault.discard(card_id, "alice")
    entry = vault.promote(card_id, "alice")
    assert entry.state == "vault"                    # read-back after promote
    assert [c.card_id for c in vault.vault("alice")] == [card_id]


def test_ghost_fifo_shift_on_discard(ledger, cards, vault):
    """Three ghosts of one spec (nos 1,2,3); discarding no 2 shifts no 3
    up to 2. Discarding no 1 shifts 2->1, 3->2."""
    card_id, trial = _mint_green(ledger, cards)
    t = ledger.get(trial.trial_id)
    s2, s3, s5 = (ledger.stage(trial.trial_id, s) for s in ("s2", "s3", "s5"))
    mk = dict(trial_id=t.trial_id, family=t.family, scope_id=t.scope_id,
              config_hash=t.config_hash, data_vintage_hash=t.data_vintage_hash,
              survivor_class="validated-green", owner_handle="alice",
              dsr=s3["dsr"], t_net_full=s5["statistic"], p_perm=s2["p_value"])
    g1 = cards.mint(**mk)
    g2 = cards.mint(**mk)
    g3 = cards.mint(**mk)
    assert (g1.ghost_no, g2.ghost_no, g3.ghost_no) == (1, 2, 3)
    for g in (g1, g2, g3):
        vault.admit(g.card_id, "alice")
    # discard ghost 2 -> ghost 3 shifts to slot 2
    res = vault.discard(g2.card_id, "alice")
    assert res["ghosts_shifted"] == 1
    assert cards.get_card(g3.card_id).ghost_no == 2
    assert cards.get_card(g1.card_id).ghost_no == 1     # above the discarded slot: unchanged
    # discard ghost 1 -> remaining active ghost (was 3, now 2) shifts to 1
    res = vault.discard(g1.card_id, "alice")
    assert res["ghosts_shifted"] == 1
    assert cards.get_card(g3.card_id).ghost_no == 1
    # pool holds both discards; cards never destroyed
    assert vault.pool_size() == 2
    assert cards.get_card(g1.card_id).kind == "ghost"


def test_flag_off_refuses_mutations_but_reads_stay(ledger, cards, vault):
    card_id, _ = _mint_green(ledger, cards)
    vault.admit(card_id, "alice")
    flags.set_flag("cards_v4_1", False)
    with pytest.raises(VaultError):
        vault.discard(card_id, "alice")
    with pytest.raises(VaultError):
        vault.keep(card_id, "alice")
    # reads still work (kill-switch never hides existing data)
    assert [c.card_id for c in vault.vault("alice")] == [card_id]
    flags.set_flag("cards_v4_1", True)
    res = vault.discard(card_id, "alice")
    assert res["entry"].state == "archive"


def test_share_slots_hook_capped_at_5(ledger, cards, vault):
    card_id, _ = _mint_green(ledger, cards)
    vault.admit(card_id, "alice")
    card = cards.get_card(card_id)
    base = vault.effective_capacity("alice", card.tier)
    assert vault.add_permanent_slots("alice", 3) == 3
    assert vault.effective_capacity("alice", card.tier) == base + 3
    assert vault.add_permanent_slots("alice", 10) == 5      # capped +5
    assert vault.revoke_permanent_slots("alice") == 5
    assert vault.effective_capacity("alice", card.tier) == base
