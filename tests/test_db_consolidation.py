"""Tests for the consolidated single-file SQLite layer (sift/db.py + CardLayer)."""

from __future__ import annotations

import pytest

from sift import db, flags
from sift.cards import CardRegistry
from sift.cards.events import SignalEventService
from sift.cards.notify import Notifier
from sift.cards.service import CardLayer
from sift.cards.share import ShareService
from sift.cards.vault import Vault
from sift.decks import DeckRegistry
from sift.registry import Ledger

SECRET = b"test-db-consolidation-secret"


@pytest.fixture(autouse=True)
def _flags_reset():
    flags.reset()
    yield
    flags.reset()


def test_db_connect_enables_wal_fk_and_busy_timeout(tmp_path):
    path = tmp_path / "sift.db"
    con = db.connect(path)
    try:
        mode = con.execute("PRAGMA journal_mode").fetchone()[0]
        fk = con.execute("PRAGMA foreign_keys").fetchone()[0]
        timeout = con.execute("PRAGMA busy_timeout").fetchone()[0]
        assert str(mode).lower() == "wal"
        assert fk == 1
        assert timeout == db.DEFAULT_BUSY_TIMEOUT_MS
        assert db.db_path_of(con) == path
    finally:
        con.close()


def test_card_layer_consolidates_all_seven_stores_into_one_db(tmp_path):
    db_path = tmp_path / "sift.db"
    ledger = Ledger(db_path)
    try:
        layer = CardLayer(ledger, tmp_path / "ignored_legacy_dir", card_secret=SECRET)
        try:
            # All 7 stores share the exact same SQLite connection & file
            for store in (
                layer.ledger,
                layer.cards,
                layer.vault,
                layer.shares,
                layer.notify,
                layer.events,
                layer.decks,
            ):
                assert store.con is ledger.con
                assert store.db_path == db_path

            # Only the single consolidated DB file exists (plus optional WAL/SHM sidecars)
            db_files = sorted(p.name for p in tmp_path.rglob("*.db"))
            assert db_files == ["sift.db"]

            # All 7 store schemas exist in the single SQLite file
            tables = {
                r[0]
                for r in ledger.con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            for expected in (
                "trial",
                "stage_result",
                "card",
                "vault_entry",
                "share_grant",
                "notify_pref",
                "notify_channel",
                "deck",
                "deck_event",
            ):
                assert expected in tables, f"missing table {expected!r}"
        finally:
            layer.close()

        # Closing CardLayer must NOT close the caller-owned ledger connection
        assert ledger.trial_count() == 0
    finally:
        ledger.close()


def test_card_layer_open_owns_and_closes_ledger(tmp_path):
    db_path = tmp_path / "owned.db"
    with CardLayer.open(db_path, card_secret=SECRET) as layer:
        assert layer.db_path == db_path
        assert layer.con is layer.ledger.con
        layer.ledger.register("pairs", "free_5y_top1k", {"a": 1}, "v1")
        assert layer.ledger.trial_count() == 1
    # Connection is closed after exiting CardLayer.open() context
    with pytest.raises(Exception):
        layer.ledger.trial_count()


def test_cross_store_transaction_rolls_back_atomically(tmp_path):
    with CardLayer.open(tmp_path / "tx.db", card_secret=SECRET) as layer:
        with pytest.raises(RuntimeError, match="boom"):
            with layer.transaction():
                trial = layer.ledger.register("pairs", "free_5y_top1k", {"p": 1}, "v1")
                card = layer.cards.mint(
                    trial_id=trial.trial_id,
                    family="pairs",
                    scope_id="free_5y_top1k",
                    config_hash=trial.config_hash,
                    data_vintage_hash="v1",
                    survivor_class="validated-green",
                    owner_handle="alice",
                    dsr=0.99,
                    t_net_full=4.5,
                    p_perm=0.005,
                    now=100.0,
                )
                layer.vault.admit(card.card_id, "alice", now=100.0)
                raise RuntimeError("boom")

        # All three stores rolled back cleanly
        assert layer.ledger.trial_count() == 0
        assert layer.cards.all_cards() == []
        assert layer.vault.vault("alice") == []


def test_nested_savepoint_partial_rollback(tmp_path):
    with CardLayer.open(tmp_path / "sp.db", card_secret=SECRET) as layer:
        with layer.transaction():
            t1 = layer.ledger.register("pairs", "free_5y_top1k", {"p": 1}, "v1")
            with pytest.raises(ValueError, match="inner"):
                with layer.transaction():
                    layer.ledger.register("pairs", "free_5y_top1k", {"p": 2}, "v1")
                    raise ValueError("inner")
            # Outer transaction still intact and commits t1 only
            assert layer.ledger.trial_count() == 1
        assert layer.ledger.trial_count() == 1
        assert layer.ledger.get(t1.trial_id).trial_id == t1.trial_id


def test_standalone_store_init_remains_backward_compatible(tmp_path):
    cards = CardRegistry(tmp_path / "cards.db", secret=SECRET)
    vault = Vault(cards, tmp_path / "vault.db")
    shares = ShareService(cards, vault, tmp_path / "shares.db")
    notify = Notifier(cards, tmp_path / "notify.db")
    events = SignalEventService(notify, None, tmp_path / "signal.db")
    decks = DeckRegistry(tmp_path / "decks.db", secret=SECRET)
    try:
        assert cards.con is not vault.con
        assert shares.con is not vault.con
        assert notify.con is not events.con
        assert decks.con is not cards.con
    finally:
        for s in (events, notify, shares, vault, cards, decks):
            s.close()
