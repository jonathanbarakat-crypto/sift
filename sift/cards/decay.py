"""v4.1 registry decay re-evaluation (mechanics 7/8 — downgrade-only, amber).

Decay re-evaluation (v4.1 spec §2/§4 + v3 §6.1):

- A card's tier is a STATIC function of gauntlet strength at mint
  (tier_from_stats). It is NEVER upgraded by payment or chance; on registry
  re-evaluation it can only move DOWN (gold -> silver -> bronze).
- The registry pass (`RegistryDecayJob.run_pass`) re-evaluates every card
  against fresh registry evidence (a `re_evaluator` callable supplied by
  the engine; default `default_re_evaluator` re-derives the tier from the
  card's own pre-registered ledger stats). If the re-evaluated tier is
  lower, the downgrade is written to the card row (decayed_from recorded)
  AND appended to the trial ledger as a `card_decay` event (append-only),
  and the owner is notified through the per-card notification channel
  (sift/cards/notify.py) — respecting opt-in + rate caps.
- The amber transition is PUBLIC and EXPLAINED: `public_explanation`
  returns the public text shown on the card face / registry page. A card
  that can visibly die is the core honesty artifact (v3 §6.1).
- Cadence: monthly registry pass is the candidate (v4.1 spec §8 open
  question 5) — the job is cadence-agnostic; the caller schedules it.

Flag: behind `cards_v4_1`; flag off -> DecayError(flag_off) on the pass.
Kill-switch env: SIFT_CARDS_V4_1=0 (sift/flags.py).

Copy discipline (legal posture §6): rarity language only — never "value
appreciates". The public explanation text lives in
`public_explanation` and is covered by a banned-phrase test.

Read-back discipline (hard rule): every transition returns the card
re-read from the DB, and the ledger event is read back via events().
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from sift import flags
from sift.cards import CARD_TIERS, Card, CardError, CardRegistry, decay_tier, tier_from_stats
from sift.cards.notify import BANNED_PHRASES, Notifier, contains_banned
from sift.registry import Ledger

# The public event type appended to the trial ledger on a downgrade.
DECAY_EVENT = "card_decay"

# Default re-evaluator: re-derive the tier from the card's own
# pre-registered ledger stats (S3 DSR + S5 full-depth t_net + S2 p_perm).
# The registry pass is the only thing allowed to move a tier, and only
# down (decay_tier enforces downgrade-only).


def default_re_evaluator(card: Card, ledger: Ledger) -> str:
    """Re-evaluate a card's tier from its pre-registered ledger stats.
    Pure read of stage rows; returns one of CARD_TIERS."""
    s2 = ledger.stage(card.trial_id, "s2")
    s3 = ledger.stage(card.trial_id, "s3")
    s5 = ledger.stage(card.trial_id, "s5")
    return tier_from_stats(
        dsr=s3["dsr"], t_net_full=s5["statistic"], p_perm=s2["p_value"]
    )


def public_explanation(card_id: str, from_tier: str, to_tier: str,
                       reason: str) -> str:
    """The PUBLIC explanation text attached to an amber downgrade (v3 §6.1:
    publicly visible with explanation). Rarity language only — fail closed
    on any banned (appreciation/urgency) phrase."""
    text = (
        f"Card {card_id} was re-evaluated by the registry pass and its rarity "
        f"tier moved from {from_tier} to {to_tier} (amber). Reason: {reason}. "
        f"Tiers are set by gauntlet strength at mint and only ever move down on "
        f"re-evaluation — never up, never by payment or chance. This is a "
        f"survival fact, not a recommendation; SIFT does not trade."
    )
    low = text.lower()
    banned = contains_banned(text)
    if banned is not None:
        raise DecayError(f"public explanation contains banned phrase {banned!r} — refused")
    return text


class DecayError(Exception):
    """Decay misuse: flag off, upgrade attempt, unknown card, ..."""


@dataclass(frozen=True)
class DecayResult:
    """One card's outcome of a registry re-evaluation pass."""
    card_id: str
    owner: str
    from_tier: str
    to_tier: str            # == from_tier when unchanged
    downgraded: bool
    ledger_event_id: int | None
    notified: bool          # whether a notification was enqueued for the owner
    public_text: str | None  # public explanation when downgraded


class RegistryDecayJob:
    """Monthly-cadence registry re-evaluation pass (spec §8 Q5 candidate).

    Downgrade-ONLY: for every card, the re-evaluator proposes a tier; if it
    is lower than the card's current tier, the downgrade is applied to the
    card row + appended to the ledger + notified via the per-card channel.
    If it is equal or higher, NOTHING changes (higher is refused by
    decay_tier — upgrades never happen, by payment or chance).
    """

    def __init__(self, cards: CardRegistry, ledger: Ledger,
                 notifier: Notifier | None = None,
                 re_evaluator: Callable[[Card, Ledger], str] | None = None):
        self.cards = cards
        self.ledger = ledger
        self.notifier = notifier
        self.re_evaluator = re_evaluator or default_re_evaluator

    def _require_flag(self) -> None:
        if not flags.cards_v4_1_enabled():
            raise DecayError("flag_off: cards_v4_1 is disabled (global kill-switch)")

    def evaluate_card(self, card: Card, now: float | None = None) -> DecayResult:
        """Re-evaluate ONE card. Downgrade-only. Read-back on every write."""
        self._require_flag()
        now = time.time() if now is None else now
        proposed = self.re_evaluator(card, self.ledger)
        if proposed not in CARD_TIERS:
            raise DecayError(f"re-evaluator returned invalid tier {proposed!r}")
        # decay_tier raises on upgrade attempts (programmer error guard):
        # a proposal HIGHER than current is refused, never applied —
        # never upgraded by payment or chance.
        try:
            target = decay_tier(card.tier, proposed)
        except CardError as exc:
            raise DecayError(
                f"decay may only downgrade, got {card.tier} -> {proposed}: {exc}"
            ) from exc
        downgraded = target != card.tier
        event_id: int | None = None
        notified = False
        public_text: str | None = None
        if downgraded:
            reason = (
                "registry re-evaluation against pre-registered gauntlet stats "
                f"(S3 DSR, S5 full-depth t_net, S2 permutation p) no longer "
                f"supports the {card.tier} rarity tier"
            )
            public_text = public_explanation(card.card_id, card.tier, target, reason)
            # write the downgrade to the card row (read-back inside decay())
            updated = self.cards.decay(card.card_id, target, now=now)
            # append the transition to the trial ledger (append-only event)
            event_id = self.ledger.record_event(
                DECAY_EVENT,
                {
                    "card_id": card.card_id,
                    "owner": card.owner_handle,
                    "kind": card.kind,
                    "from_tier": card.tier,
                    "to_tier": target,
                    "reason": reason,
                    "public_explanation": public_text,
                },
                trial_id=card.trial_id,
                now=now,
            )
            assert updated.tier == target and updated.decayed_from == card.tier
            # notify the owner via the per-card channel (opt-in + caps apply)
            if self.notifier is not None:
                n = self.notifier.notify(
                    card.card_id, card.owner_handle, "decay",
                    {"from_tier": card.tier, "to_tier": target,
                     "public_explanation": public_text},
                    now=now,
                )
                notified = n is not None
        return DecayResult(
            card_id=card.card_id, owner=card.owner_handle,
            from_tier=card.tier, to_tier=target, downgraded=downgraded,
            ledger_event_id=event_id, notified=notified, public_text=public_text,
        )

    def run_pass(self, now: float | None = None) -> list[DecayResult]:
        """The registry re-evaluation pass over ALL cards (regular + ghost).
        Returns one DecayResult per card, in mint order. Flag off ->
        DecayError(flag_off)."""
        self._require_flag()
        now = time.time() if now is None else now
        results: list[DecayResult] = []
        for card in self.cards.all_cards():
            results.append(self.evaluate_card(card, now=now))
        return results


__all__ = [
    "DECAY_EVENT",
    "DecayError",
    "DecayResult",
    "RegistryDecayJob",
    "default_re_evaluator",
    "public_explanation",
]
