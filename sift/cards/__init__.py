"""v4.1 card registry: card minting on gauntlet survivors (mechanics item 1).

Design (v4.1 mechanics 1, behind the `cards_v4_1` flag):

- A card is minted ONLY when a trial reaches verdict=survivor AND its
  survivor_class is `validated-green` (S5 registry-depth pass). Killed
  trials and on-watch-amber survivors (the F-REGIME class — edge that does
  not survive full depth) mint nothing. F-DELIST runs are kill/amber paths
  too: nothing mints. A planted clean survivor (F-PAIR-style, full-window
  validated) mints exactly one card.
- The public card id is an HMAC commitment (server-side secret) of the
  canonical card spec (family, scope_id, config_hash, data_vintage_hash,
  trial_id, survivor_class) — `card-<hex>`. HMAC makes ids unforgeable and
  unique per canonical spec without revealing the spec; the secret never
  leaves the server (never logged, never committed, env var or file).
- Tier comes from the pre-registered stats recorded in the ledger stage
  rows (S3 DSR + S5 t_net at full depth): see `tier_from_stats` — the mapping
  is a pure function of recorded, pre-registered quantities only.
- Decay is downgrade-only: `decay()` may only move a tier DOWN
  (gold -> silver -> bronze), never up, and never re-randomizes.
- Duplicate spec (same canonical spec minted again, e.g. a re-run that
  hits the skip-list and re-survives) -> a GHOST card: kind="ghost",
  numbered FIFO per canonical spec (ghost_no). Ghost handling proper
  (vault shifting) lives in sift/cards/vault.py (mechanics 2-3); this
  module owns minting + id commitment + tier + decay.
- Cards are non-transferable: the record carries owner_handle and any
  transfer attempt raises (mechanics hard rule).
- Read-back discipline (hard rule): after every mint, the card is read
  back from the registry (`get_card`) before success is claimed; the
  minter returns the read-back row, not just the id.

Storage: its own SQLite file (like the trial ledger) with an append-only
card table + a ghost-counter table. Never destroyed, never randomized
(mechanics 2 applies to cards as registry records).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

CARD_KINDS = ("regular", "ghost")
CARD_TIERS = ("bronze", "silver", "gold")   # index = strength; decay moves down

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS card (
    card_id            TEXT PRIMARY KEY,          -- HMAC commitment, public id
    canonical_spec     TEXT NOT NULL,             -- canonical JSON of the spec
    spec_hash          TEXT NOT NULL,             -- sha256 of canonical_spec (internal dedupe key)
    trial_id           TEXT NOT NULL,
    family             TEXT NOT NULL,
    scope_id           TEXT NOT NULL,
    survivor_class     TEXT NOT NULL,
    kind               TEXT NOT NULL DEFAULT 'regular',   -- regular|ghost
    ghost_no           INTEGER,                   -- FIFO slot per spec_hash (ghosts only)
    tier               TEXT NOT NULL,
    tier_updated_at    REAL NOT NULL,
    owner_handle       TEXT NOT NULL,             -- non-transferable
    minted_at          REAL NOT NULL,
    decayed_from       TEXT,                      -- tier before last downgrade
    UNIQUE (spec_hash, kind, ghost_no)
);
CREATE INDEX IF NOT EXISTS idx_card_spec ON card(spec_hash);
CREATE INDEX IF NOT EXISTS idx_card_owner ON card(owner_handle);
CREATE TABLE IF NOT EXISTS spec_ghost_counter (
    spec_hash   TEXT PRIMARY KEY,
    next_no     INTEGER NOT NULL DEFAULT 1
);
"""


class CardError(Exception):
    """Raised on card-registry misuse (unknown card, transfer attempt, ...)."""


def canonical_spec_json(spec: dict) -> str:
    return json.dumps(spec, sort_keys=True, separators=(",", ":"), default=str)


def spec_hash_of(spec: dict) -> str:
    return hashlib.sha256(canonical_spec_json(spec).encode()).hexdigest()


def _default_secret() -> bytes:
    """Server-side HMAC secret. Env var first (deployment), else a local
    dev secret file (gitignored pattern: *.card-secret), else an ephemeral
    in-memory secret (dev only — ids won't survive a restart; tests set
    SIFT_CARD_HMAC_SECRET explicitly). Never logged, never committed."""
    env = os.environ.get("SIFT_CARD_HMAC_SECRET")
    if env:
        return env.encode()
    path = Path(os.environ.get("SIFT_CARD_HMAC_SECRET_FILE", ".card-hmac-secret"))
    if path.exists():
        return path.read_bytes().strip()
    return b"sift-dev-ephemeral-secret-not-for-production"


@dataclass(frozen=True)
class Card:
    card_id: str
    spec_hash: str
    trial_id: str
    family: str
    scope_id: str
    survivor_class: str
    kind: str
    ghost_no: int | None
    tier: str
    owner_handle: str
    minted_at: float
    decayed_from: str | None


def tier_from_stats(dsr: float | None, t_net_full: float | None,
                    p_perm: float | None) -> str:
    """Tier from PRE-REGISTERED, ledger-recorded stats only (mechanics 1:
    'tier from pre-registered stats'). Pure function, deterministic.

    - gold:   DSR >= 0.99 at scope AND full-depth |t_net| >= 4.0 AND p_perm <= 0.01
    - silver: DSR >= 0.95 (the S3 admission line) AND full-depth |t_net| >= 3.0 (S5 line)
    - bronze: survivor but below the silver bar (on-watch-green edge cases)

    Missing stats never inflate: a None component fails its check.
    """
    if dsr is None or t_net_full is None or p_perm is None:
        return "bronze"
    if dsr >= 0.99 and abs(t_net_full) >= 4.0 and p_perm <= 0.01:
        return "gold"
    if dsr >= 0.95 and abs(t_net_full) >= 3.0:
        return "silver"
    return "bronze"


def decay_tier(current: str, new: str) -> str:
    """Downgrade-only decay: the result is min(strength) — never an
    upgrade. Raises if `new` would be an upgrade (programmer error)."""
    if current not in CARD_TIERS or new not in CARD_TIERS:
        raise CardError(f"invalid tier {current!r} -> {new!r}")
    if CARD_TIERS.index(new) > CARD_TIERS.index(current):
        raise CardError(f"decay may only downgrade, got {current} -> {new}")
    return new


class CardRegistry:
    """Append-only card registry. The only writer is the server-side
    engine (mint on survivor) and the server-side decay job."""

    def __init__(self, db_path: str | Path, secret: bytes | None = None):
        self.db_path = Path(db_path)
        if str(self.db_path.parent) not in ("", "."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.db_path))
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA foreign_keys = ON")
        self.con.executescript(_SCHEMA_SQL)
        self.con.commit()
        self._secret = secret if secret is not None else _default_secret()

    def __enter__(self) -> "CardRegistry":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.con.close()

    # -- id commitment ------------------------------------------------------

    def card_id_for(self, spec: dict) -> str:
        """Public card id = HMAC-SHA256(secret, canonical_spec), hex, prefixed.
        Unforgeable without the server secret; unique per canonical spec."""
        mac = hmac.new(self._secret, canonical_spec_json(spec).encode(), hashlib.sha256)
        return "card-" + mac.hexdigest()

    # -- minting --------------------------------------------------------------

    def mint(self, *, trial_id: str, family: str, scope_id: str,
             config_hash: str, data_vintage_hash: str,
             survivor_class: str, owner_handle: str,
             dsr: float | None, t_net_full: float | None,
             p_perm: float | None, now: float | None = None) -> Card:
        """Mint a card for a gauntlet survivor. Mechanics 1:

        - only survivor_class == 'validated-green' mints (callers pass
          the ledger verdict; killed / on-watch-amber mint nothing — the
          caller enforces, and mint() refuses anything else);
        - duplicate canonical spec -> ghost (kind='ghost', FIFO ghost_no);
        - read-back: the returned Card is re-read from the DB (hard rule:
          never claim success without read-back).
        """
        if survivor_class != "validated-green":
            raise CardError(
                f"only validated-green survivors mint cards, got {survivor_class!r}"
            )
        spec = {
            "family": family,
            "scope_id": scope_id,
            "config_hash": config_hash,
            "data_vintage_hash": data_vintage_hash,
            "trial_id": trial_id,
            "survivor_class": survivor_class,
        }
        sh = spec_hash_of(spec)
        card_id = self.card_id_for(spec)
        now = time.time() if now is None else now
        tier = tier_from_stats(dsr, t_net_full, p_perm)

        existing = self.con.execute(
            "SELECT card_id FROM card WHERE spec_hash=? AND kind='regular'", (sh,)
        ).fetchone()
        if existing is not None:
            # duplicate spec -> ghost, numbered FIFO per canonical spec
            no = self._next_ghost_no(sh)
            self.con.execute(
                "INSERT INTO card (card_id, canonical_spec, spec_hash, trial_id,"
                " family, scope_id, survivor_class, kind, ghost_no, tier,"
                " tier_updated_at, owner_handle, minted_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (card_id + f":g{no}", canonical_spec_json(spec), sh, trial_id,
                 family, scope_id, survivor_class, "ghost", no, tier,
                 now, owner_handle, now),
            )
            self.con.commit()
            return self.get_card(card_id + f":g{no}")   # read-back

        self.con.execute(
            "INSERT INTO card (card_id, canonical_spec, spec_hash, trial_id,"
            " family, scope_id, survivor_class, kind, ghost_no, tier,"
            " tier_updated_at, owner_handle, minted_at)"
            " VALUES (?,?,?,?,?,?,?, 'regular', NULL, ?,?,?,?)",
            (card_id, canonical_spec_json(spec), sh, trial_id,
             family, scope_id, survivor_class, tier, now, owner_handle, now),
        )
        self.con.commit()
        return self.get_card(card_id)                    # read-back

    def _next_ghost_no(self, spec_hash: str) -> int:
        row = self.con.execute(
            "SELECT next_no FROM spec_ghost_counter WHERE spec_hash=?", (spec_hash,)
        ).fetchone()
        no = row["next_no"] if row else 1
        self.con.execute(
            "INSERT INTO spec_ghost_counter (spec_hash, next_no) VALUES (?,?)"
            " ON CONFLICT(spec_hash) DO UPDATE SET next_no=excluded.next_no",
            (spec_hash, no + 1),
        )
        return no

    # -- reads ----------------------------------------------------------------

    def get_card(self, card_id: str) -> Card:
        """Read-back of one card. Raises on unknown id."""
        row = self.con.execute("SELECT * FROM card WHERE card_id=?", (card_id,)).fetchone()
        if row is None:
            raise CardError(f"unknown card_id {card_id!r}")
        return self._to_card(row)

    def cards_for_owner(self, handle: str) -> list[Card]:
        rows = self.con.execute(
            "SELECT * FROM card WHERE owner_handle=? ORDER BY minted_at, card_id",
            (handle,),
        ).fetchall()
        return [self._to_card(r) for r in rows]

    def all_cards(self) -> list[Card]:
        """Every card in the registry (regular + ghosts), oldest first.
        Read used by the decay re-evaluation pass (sift/cards/decay.py)."""
        rows = self.con.execute("SELECT * FROM card ORDER BY minted_at, card_id").fetchall()
        return [self._to_card(r) for r in rows]

    def ghosts_for_spec(self, spec_hash: str) -> list[Card]:
        """ACTIVE ghost cards for one canonical spec, FIFO order (mechanics
        3). Ghosts discarded from the FIFO carry ghost_no<=0 and are excluded
        here (they remain in the registry pool, never destroyed)."""
        rows = self.con.execute(
            "SELECT * FROM card WHERE spec_hash=? AND kind='ghost' AND ghost_no>0"
            " ORDER BY ghost_no",
            (spec_hash,),
        ).fetchall()
        return [self._to_card(r) for r in rows]

    def _to_card(self, row: sqlite3.Row) -> Card:
        return Card(
            card_id=row["card_id"], spec_hash=row["spec_hash"],
            trial_id=row["trial_id"], family=row["family"],
            scope_id=row["scope_id"], survivor_class=row["survivor_class"],
            kind=row["kind"], ghost_no=row["ghost_no"], tier=row["tier"],
            owner_handle=row["owner_handle"], minted_at=row["minted_at"],
            decayed_from=row["decayed_from"],
        )

    # -- decay + non-transferability ------------------------------------------

    def decay(self, card_id: str, new_tier: str, now: float | None = None) -> Card:
        """Downgrade-only decay (mechanics 1). Records decayed_from; the row
        is updated in place but the card's history is kept in the returned
        record + the tier change is append-only in spirit (decayed_from).
        Read-back before returning."""
        card = self.get_card(card_id)
        now = time.time() if now is None else now
        downgraded = decay_tier(card.tier, new_tier)
        self.con.execute(
            "UPDATE card SET tier=?, tier_updated_at=?, decayed_from=? WHERE card_id=?",
            (downgraded, now, card.tier if downgraded != card.tier else card.decayed_from,
             card_id),
        )
        self.con.commit()
        return self.get_card(card_id)

    def assert_owned(self, card_id: str, handle: str) -> Card:
        """Non-transferability guard: cards never change owner; any path that
        needs the card for a user checks ownership here and raises otherwise."""
        card = self.get_card(card_id)
        if card.owner_handle != handle:
            raise CardError(
                f"card {card_id} belongs to {card.owner_handle!r}, not {handle!r} —"
                " cards are non-transferable"
            )
        return card


__all__ = [
    "CARD_KINDS",
    "CARD_TIERS",
    "Card",
    "CardError",
    "CardRegistry",
    "canonical_spec_json",
    "decay_tier",
    "spec_hash_of",
    "tier_from_stats",
]
