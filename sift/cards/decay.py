"""v4.1 decay re-evaluation (mechanics: registry re-evaluation, behind `cards_v4_1`).

Mechanics (authoritative, spec §2/§3/§4 + v3 §6.1):
- Tiers are a static function of gauntlet strength at mint (pull/13). On
  registry re-evaluation a card may DOWNGRADE ONLY (gold -> silver -> bronze,
  "amber on decay") — never upgrade, never by payment or chance (v3 §6.1:
  a card that can visibly die is the core honesty artifact).
- The downgrade is PUBLICLY visible with an explanation, written to the card
  row (decayed_from, via CardRegistry.decay) and surfaced through the
  per-card notification channel (NotifyService.enqueue_decay_notice).
- The re-evaluation job is a monthly-pass candidate (spec §8 open question 5);
  `reevaluate` re-derives the tier from the trial's CURRENT ledger stage rows
  (the same pre-registered stats the minter used) and applies decay_tier —
  which refuses any upgrade attempt (programmer error -> DecayError).
- No "value appreciates" language anywhere: the public explanation text is
  fixed rarity/decay language only (legal posture §6).

Flag: flag off -> DecayError(flag_off) on every mutating path; reads stay
available (kill-switch never hides existing data).
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from sift import flags
from sift.cards import CARD_TIERS, CardError, CardRegistry, decay_tier, tier_from_stats
from sift.cards.notify import NotifyService
from sift.registry import Ledger

# Public explanation text for a downgrade — fixed, rarity/decay language only
# (no "value appreciates", no market language; legal posture §6).
DOWNGRADE_EXPLANATION = (
    "This card was re-evaluated against the current registry and its tier moved "
    "down ({old} -> {new}). Tiers reflect pre-registered gauntlet strength only; "
    "they can decay on re-evaluation but never appreciate, never upgrade by "
    "payment or chance. Survival facts only — not investment advice (v3 §12)."
)


class DecayError(Exception):
    """Decay misuse: flag off, unknown card, upgrade attempt, ..."""


@dataclass(frozen=True)
class DecayEvent:
    card_id: str
    handle: str
    old_tier: str
    new_tier: str
    at: float
    explanation: str


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS decay_event (
    event_id    TEXT PRIMARY KEY,
    card_id     TEXT NOT NULL,
    handle      TEXT NOT NULL,
    old_tier    TEXT NOT NULL,
    new_tier    TEXT NOT NULL,
    at          REAL NOT NULL,
    explanation TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decay_card ON decay_event(card_id);
"""


class DecayService:
    """Registry re-evaluation job: downgrade-only tier transitions, public
    explanation, notification hook. Server-side only."""

    def __init__(self, ledger: Ledger, cards: CardRegistry, notify: NotifyService,
                 db_path: str | Path):
        self.ledger = ledger
        self.cards = cards
        self.notify = notify
        self.db_path = Path(db_path)
        if str(self.db_path.parent) not in ("", "."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.db_path))
        self.con.row_factory = sqlite3.Row
        self.con.executescript(_SCHEMA_SQL)
        self.con.commit()

    def __enter__(self) -> "DecayService":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.con.close()

    def _require_flag(self, handle: str | None) -> None:
        if not flags.cards_v4_1_enabled(handle):
            raise DecayError("flag_off: cards_v4_1 is disabled for this account")

    def current_tier_from_ledger(self, card_id: str) -> str:
        """Re-derive the tier from the trial's CURRENT ledger stage rows —
        the same pre-registered quantities the minter used (s2 p_perm, s3 dsr,
        s5 full-depth statistic). Pure read of the ledger."""
        card = self.cards.get_card(card_id)
        s2 = self.ledger.stage(card.trial_id, "s2")
        s3 = self.ledger.stage(card.trial_id, "s3")
        s5 = self.ledger.stage(card.trial_id, "s5")
        return tier_from_stats(s3["dsr"], s5["statistic"], s2["p_value"])

    def reevaluate(self, card_id: str, now: float | None = None) -> DecayEvent | None:
        """Re-evaluate one card against the current registry. Downgrade-only:
        if the re-derived tier is WEAKER than the stored tier, decay; if equal
        or stronger, NOTHING changes (decay_tier refuses upgrades — a stronger
        re-derivation is not an upgrade path, tiers only move down on decay).
        Returns the DecayEvent, or None when no downgrade applied. Read-back."""
        self._require_flag(None)  # layer-level fail-closed gate first (uniform ordering)
        card = self.cards.get_card(card_id)
        now = time.time() if now is None else now
        rederived = self.current_tier_from_ledger(card_id)
        if CARD_TIERS.index(rederived) >= CARD_TIERS.index(card.tier):
            return None                     # equal or stronger: NO transition (downgrade-only)
        try:
            downgraded = self.cards.decay(card_id, rederived, now=now)
        except CardError as e:
            raise DecayError(str(e)) from e
        explanation = DOWNGRADE_EXPLANATION.format(old=card.tier, new=downgraded.tier)
        event_id = f"decay-{card_id}-{int(now * 1000)}-{abs(hash((card_id, now))) % 100000:05d}"
        self.con.execute(
            "INSERT INTO decay_event (event_id, card_id, handle, old_tier, new_tier,"
            " at, explanation) VALUES (?,?,?,?,?,?,?)",
            (event_id, card_id, card.owner_handle, card.tier, downgraded.tier,
             now, explanation),
        )
        self.con.commit()
        # notification hook: the user is told via the per-card channel
        self.notify.enqueue_decay_notice(
            handle=card.owner_handle, card_id=card_id,
            payload={"event_id": event_id, "old_tier": card.tier,
                     "new_tier": downgraded.tier, "explanation": explanation},
            now=now,
        )
        return DecayEvent(card_id=card_id, handle=card.owner_handle,
                          old_tier=card.tier, new_tier=downgraded.tier,
                          at=now, explanation=explanation)

    def reevaluate_all(self, now: float | None = None) -> list[DecayEvent]:
        """The monthly-pass candidate (spec §8 open question 5): re-evaluate
        every card in the registry. Downgrade-only per card. Returns all
        applied DecayEvents (read-back verified)."""
        now = time.time() if now is None else now
        events: list[DecayEvent] = []
        rows = self.cards.con.execute("SELECT card_id FROM card").fetchall()
        for r in rows:
            ev = self.reevaluate(r["card_id"], now=now)
            if ev is not None:
                events.append(ev)
        return events

    def events_for(self, card_id: str) -> list[DecayEvent]:
        rows = self.con.execute(
            "SELECT * FROM decay_event WHERE card_id=? ORDER BY at", (card_id,)
        ).fetchall()
        return [DecayEvent(card_id=r["card_id"], handle=r["handle"],
                           old_tier=r["old_tier"], new_tier=r["new_tier"],
                           at=r["at"], explanation=r["explanation"]) for r in rows]

    def all_events(self) -> list[DecayEvent]:
        rows = self.con.execute("SELECT * FROM decay_event ORDER BY at").fetchall()
        return [DecayEvent(card_id=r["card_id"], handle=r["handle"],
                           old_tier=r["old_tier"], new_tier=r["new_tier"],
                           at=r["at"], explanation=r["explanation"]) for r in rows]


__all__ = ["DOWNGRADE_EXPLANATION", "DecayError", "DecayEvent", "DecayService"]
