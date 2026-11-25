"""v4.1 vault: per-user card custody (mechanics 2 + 3, behind `cards_v4_1`).

Mechanics:
2. Vault: per-user capacity by tier; keep/discard. A discard moves the card
   to the user's PERSONAL ARCHIVE and to the aggregate REGISTRY POOL — never
   destroyed, never randomized. Capacity by tier:
     gold: 6, silver: 4, bronze: 2 (base slots; share-for-capacity adds
     permanent slots in mechanics item 6 — see sift/cards/shares.py, which
     calls vault.effective_capacity()).
3. Ghost cards: numbered FIFO per canonical spec (assigned at mint). A
   discard of ghost number k shifts all ghosts below k up one slot
   (k+1 -> k, ...). Ghosts are usable for the owner's own investing +
   notifications but are NEVER transferable (same rule as regular cards).

Read-back discipline (hard rule): keep/discard/promote return state that is
re-read from the DB before being returned — never claim success from the
write alone.

Flag: every mutating path refuses with VaultError(flag_off) when
cards_v4_1 is off for the handle. Reads (list/archive) stay available —
existing data is never hidden or destroyed by the kill-switch.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from sift import db, flags
from sift.cards import Card, CardRegistry

VAULT_STATES = ("vault", "archive")     # custody states; pool membership is separate
BASE_CAPACITY = {"platinum": 8, "gold": 6, "silver": 4, "bronze": 2}

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS vault_entry (
    card_id     TEXT NOT NULL,
    owner       TEXT NOT NULL,
    state       TEXT NOT NULL DEFAULT 'vault',   -- vault|archive
    extra_slots INTEGER NOT NULL DEFAULT 0,       -- permanent share-for-capacity slots (mechanics 6)
    updated_at  REAL NOT NULL,
    PRIMARY KEY (card_id, owner)
);
CREATE TABLE IF NOT EXISTS registry_pool (
    card_id     TEXT NOT NULL,
    spec_hash   TEXT NOT NULL,
    discarded_at REAL NOT NULL,
    PRIMARY KEY (card_id)
);
CREATE TABLE IF NOT EXISTS vault_owner_slots (
    owner       TEXT PRIMARY KEY,          -- mechanics 6: authoritative per-owner share slots
    extra_slots INTEGER NOT NULL DEFAULT 0, -- capped at +5 (see set_permanent_slots)
    updated_at  REAL NOT NULL
);
"""


class VaultError(Exception):
    """Vault misuse: unknown card, over capacity, flag off, ..."""


@dataclass(frozen=True)
class VaultEntry:
    card_id: str
    owner: str
    state: str
    extra_slots: int
    updated_at: float


class Vault:
    """Per-user vault over the card registry. Server-side only."""

    def __init__(
        self,
        cards: CardRegistry,
        db_path: str | Path | None = None,
        *,
        con: sqlite3.Connection | None = None,
    ):
        self.cards = cards
        if con is not None or db_path is None:
            shared = con if con is not None else cards.con
            self.con = shared
            self._owns_con = False
            self.db_path = (
                Path(db_path) if db_path is not None else (db.db_path_of(shared) or cards.db_path)
            )
            self.con.row_factory = sqlite3.Row
            self.con.execute("PRAGMA foreign_keys = ON")
        else:
            self.db_path = Path(db_path)
            self.con = db.connect(self.db_path)
            self._owns_con = True
        self.con.executescript(_SCHEMA_SQL)
        db.maybe_commit(self.con)

    def __enter__(self) -> "Vault":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_con:
            self.con.close()

    # -- capacity -------------------------------------------------------------

    def base_capacity(self, tier: str) -> int:
        if tier not in BASE_CAPACITY:
            raise VaultError(f"unknown tier {tier!r}")
        return BASE_CAPACITY[tier]

    def share_slots(self, handle: str) -> int:
        """Permanent share-for-capacity slots for the account (mechanics 6).
        Authoritative source is vault_owner_slots (works even before the
        account owns a card); the vault_entry.extra_slots column is kept in
        sync as a mirror for per-card reads. Read-back from the DB."""
        row = self.con.execute(
            "SELECT extra_slots FROM vault_owner_slots WHERE owner=?", (handle,)
        ).fetchone()
        if row is not None:
            return row["extra_slots"]
        row = self.con.execute(
            "SELECT MAX(extra_slots) m FROM vault_entry WHERE owner=?", (handle,)
        ).fetchone()
        return row["m"] if row and row["m"] is not None else 0

    def effective_capacity(self, handle: str, tier: str) -> int:
        """Base capacity by tier + permanent share-for-capacity slots
        (mechanics 6, cap +5 — never combined with paid-tier purchases to
        exceed caps). Read-back from the DB."""
        return self.base_capacity(tier) + self.share_slots(handle)

    def _require_flag(self, handle: str) -> None:
        if not flags.cards_v4_1_enabled(handle):
            raise VaultError("flag_off: cards_v4_1 is disabled for this account")

    # -- custody ----------------------------------------------------------------

    def admit(self, card_id: str, handle: str, now: float | None = None) -> VaultEntry:
        """Place a freshly minted card into the owner's vault. Enforces
        capacity: over capacity -> VaultError (the mint path checks first
        via can_admit). Read-back before returning."""
        self._require_flag(handle)
        card = self.cards.assert_owned(card_id, handle)
        now = time.time() if now is None else now
        if not self.can_admit(handle, card.tier):
            raise VaultError(
                f"vault full for {handle!r} at tier {card.tier}"
                f" ({self.vault_count(handle)}/{self.effective_capacity(handle, card.tier)})"
            )
        self.con.execute(
            "INSERT INTO vault_entry (card_id, owner, state, extra_slots, updated_at)"
            " VALUES (?,?, 'vault', 0, ?)"
            " ON CONFLICT(card_id, owner) DO UPDATE SET state='vault', updated_at=excluded.updated_at",
            (card_id, handle, now),
        )
        db.maybe_commit(self.con)
        return self.get_entry(card_id, handle)      # read-back

    def can_admit(self, handle: str, tier: str) -> bool:
        return self.vault_count(handle) < self.effective_capacity(handle, tier)

    def vault_count(self, handle: str) -> int:
        row = self.con.execute(
            "SELECT COUNT(*) c FROM vault_entry WHERE owner=? AND state='vault'",
            (handle,),
        ).fetchone()
        return row["c"]

    def keep(self, card_id: str, handle: str, now: float | None = None) -> VaultEntry:
        """Explicit keep: (re-)assert vault state. Idempotent. Read-back."""
        self._require_flag(handle)
        self.cards.assert_owned(card_id, handle)
        now = time.time() if now is None else now
        self.con.execute(
            "INSERT INTO vault_entry (card_id, owner, state, extra_slots, updated_at)"
            " VALUES (?,?, 'vault', 0, ?)"
            " ON CONFLICT(card_id, owner) DO UPDATE SET state='vault', updated_at=excluded.updated_at",
            (card_id, handle, now),
        )
        db.maybe_commit(self.con)
        return self.get_entry(card_id, handle)

    def discard(self, card_id: str, handle: str, now: float | None = None) -> dict:
        """Discard (mechanics 2+3): card -> personal archive + aggregate
        registry pool (never destroyed, never randomized); if the card is a
        ghost, all ghosts below it shift up one slot (FIFO). Returns the
        read-back state."""
        self._require_flag(handle)
        card = self.cards.assert_owned(card_id, handle)
        now = time.time() if now is None else now
        self.con.execute(
            "INSERT INTO vault_entry (card_id, owner, state, extra_slots, updated_at)"
            " VALUES (?,?, 'archive', 0, ?)"
            " ON CONFLICT(card_id, owner) DO UPDATE SET state='archive', updated_at=excluded.updated_at",
            (card_id, handle, now),
        )
        self.con.execute(
            "INSERT OR IGNORE INTO registry_pool (card_id, spec_hash, discarded_at)"
            " VALUES (?,?,?)",
            (card_id, card.spec_hash, now),
        )
        shifted = 0
        if card.kind == "ghost":
            shifted = self._shift_ghosts_up(card.spec_hash, card.ghost_no or 0, card.card_id)
        db.maybe_commit(self.con)
        # read-back: vault state after discard + pool membership
        entry = self.get_entry(card_id, handle)
        pooled = self.con.execute(
            "SELECT 1 FROM registry_pool WHERE card_id=?", (card_id,)
        ).fetchone() is not None
        ghosts = self.cards.ghosts_for_spec(card.spec_hash) if card.kind == "ghost" else []
        return {
            "entry": entry,
            "pooled": pooled,
            "ghosts_shifted": shifted,
            "ghost_slots": [g.ghost_no for g in ghosts],
        }

    def _shift_ghosts_up(self, spec_hash: str, discarded_no: int, discarded_card_id: str) -> int:
        """Ghosts with ghost_no > discarded_no move up one slot (FIFO).
        Ghost numbering is per canonical spec; the discarded ghost's slot is
        freed and everyone below shifts. The discarded ghost itself is
        re-numbered to a negative slot (out of the active queue; it stays in
        the pool) so the active FIFO stays 1..k with no gaps and no UNIQUE
        collision with another discarded ghost. Returns the number of
        ghosts shifted."""
        self.cards.con.execute(
            "UPDATE card SET ghost_no=-(rowid) WHERE card_id=?", (discarded_card_id,)
        )
        rows = self.cards.con.execute(
            "SELECT card_id, ghost_no FROM card WHERE spec_hash=? AND kind='ghost'"
            " AND ghost_no>? AND card_id<>? ORDER BY ghost_no",
            (spec_hash, discarded_no, discarded_card_id),
        ).fetchall()
        for r in rows:
            self.cards.con.execute(
                "UPDATE card SET ghost_no=? WHERE card_id=?", (r["ghost_no"] - 1, r["card_id"])
            )
        db.maybe_commit(self.cards.con)
        return len(rows)

    def promote(self, card_id: str, handle: str, now: float | None = None) -> VaultEntry:
        """Promote an archived card back into the vault (capacity-checked).
        Read-back before returning."""
        self._require_flag(handle)
        card = self.cards.assert_owned(card_id, handle)
        now = time.time() if now is None else now
        if not self.can_admit(handle, card.tier):
            raise VaultError(
                f"vault full for {handle!r} at tier {card.tier} — cannot promote"
            )
        self.con.execute(
            "UPDATE vault_entry SET state='vault', updated_at=? WHERE card_id=? AND owner=?",
            (now, card_id, handle),
        )
        db.maybe_commit(self.con)
        return self.get_entry(card_id, handle)

    # -- reads ------------------------------------------------------------------

    def get_entry(self, card_id: str, handle: str) -> VaultEntry:
        row = self.con.execute(
            "SELECT * FROM vault_entry WHERE card_id=? AND owner=?", (card_id, handle)
        ).fetchone()
        if row is None:
            raise VaultError(f"no vault entry for {card_id!r}/{handle!r}")
        return VaultEntry(row["card_id"], row["owner"], row["state"],
                          row["extra_slots"], row["updated_at"])

    def vault(self, handle: str) -> list[Card]:
        """Cards currently in the vault (state='vault'), read-back joined
        with the card registry."""
        rows = self.con.execute(
            "SELECT card_id FROM vault_entry WHERE owner=? AND state='vault'"
            " ORDER BY updated_at", (handle,)
        ).fetchall()
        return [self.cards.get_card(r["card_id"]) for r in rows]

    def archive(self, handle: str) -> list[Card]:
        rows = self.con.execute(
            "SELECT card_id FROM vault_entry WHERE owner=? AND state='archive'"
            " ORDER BY updated_at", (handle,)
        ).fetchall()
        return [self.cards.get_card(r["card_id"]) for r in rows]

    def pool_size(self) -> int:
        """Aggregate registry pool size (discarded cards; never destroyed)."""
        return self.con.execute("SELECT COUNT(*) c FROM registry_pool").fetchone()["c"]

    def add_permanent_slots(self, handle: str, n: int) -> int:
        """Mechanics 6 hook: add permanent (non-transferable) vault slots.
        Capped at +5 total per account; returns the new extra_slots total.
        Writes the authoritative vault_owner_slots row AND mirrors onto the
        vault rows, then READS BACK before returning."""
        if n < 0:
            raise VaultError("n must be >= 0")
        cur = self.share_slots(handle)
        new = min(cur + n, 5)
        self.set_permanent_slots(handle, new)
        return self.share_slots(handle)                  # read-back

    def revoke_permanent_slots(self, handle: str) -> int:
        """Remove ALL permanent share slots (abuse revocation path, mechanics
        6/§4.7). Returns the previous total. Read-back before returning."""
        prev = self.share_slots(handle)
        self.set_permanent_slots(handle, 0)
        self.share_slots(handle)                          # read-back
        return prev

    def set_permanent_slots(self, handle: str, value: int) -> int:
        """Set the permanent share-slot count (clamped to 0..5) in the
        authoritative vault_owner_slots row and mirror it onto every
        vault_entry of the owner. Used by sift/cards/share.py after each
        grant/revoke. Read-back."""
        if value < 0:
            raise VaultError("slots must be >= 0")
        value = min(value, 5)
        now = time.time()
        self.con.execute(
            "INSERT INTO vault_owner_slots (owner, extra_slots, updated_at)"
            " VALUES (?,?,?) ON CONFLICT(owner) DO UPDATE SET"
            " extra_slots=excluded.extra_slots, updated_at=excluded.updated_at",
            (handle, value, now),
        )
        self.con.execute(
            "UPDATE vault_entry SET extra_slots=?, updated_at=? WHERE owner=?",
            (value, now, handle),
        )
        db.maybe_commit(self.con)
        return self.share_slots(handle)                  # read-back


__all__ = [
    "BASE_CAPACITY",
    "VAULT_STATES",
    "Vault",
    "VaultEntry",
    "VaultError",
]
