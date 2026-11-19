"""v4.1 card layer — end-to-end wiring facade (behind `cards_v4_1`).

`CardLayer` is the single server-side composition root for the whole v4.1
card layer:

    trial registry (Ledger)
      -> CardMinter / CardRegistry   (mint on validated-green, pull/13)
      -> Vault                       (custody, capacity, ghosts, pull/14)
      -> CardFaceRenderer            (anonymized shareable face, pull/15)
      -> anchored run mode           (pull/16)
      -> ShareService                (share-for-capacity, cards 5/8, pull/18)
      -> Notifier                    (per-card opt-in notifications, cards 7/8, pull/19)
      -> SignalEventService          (rich v1 signal-event payload + channels:
                                      email/RSS/API webhook, quiet hours, pull/32)
      -> RegistryDecayJob            (bidirectional re-evaluation: amber
                                       downgrades any time + pre-registered
                                       live-window upgrades, cards 7/8, pull/31)
      -> Metrics                     (gate-1 metric hooks)
      -> DeckRegistry + DeckService  (v4.2 deck layer: bucket registry,
                                      S6 portfolio gauntlet, F-S ranking —
                                      behind decks_v4_2, hermes-tasks #166)

Every component independently enforces the flag (each mutating path raises
its own fail-closed error type with flag_off when `cards_v4_1` is off);
this facade adds ONE guard, `require_layer(handle)`, so an operator path
that touches several components fails closed once, up front, with
LayerError(flag_off). The env kill-switch SIFT_CARDS_V4_1=0 disables the
whole layer because every read goes through sift.flags. The v4.2 deck
layer additionally requires `decks_v4_2` (kill-switch SIFT_DECKS_V4_2=0)
and sits on top of the card layer: `decks_v4_2_enabled` ANDs the two.
"""

from __future__ import annotations

from pathlib import Path

from sift import flags
from sift.cards import CardRegistry
from sift.cards.anchored import AnchoredRunError  # noqa: F401  (re-export surface)
from sift.cards.copy import CopyError  # noqa: F401  (re-export surface)
from sift.cards.decay import RegistryDecayJob
from sift.cards.events import SignalEventService
from sift.cards.face import CardFaceRenderer
from sift.cards.metrics import Metrics
from sift.cards.minting import CardMinter
from sift.cards.notify import Notifier
from sift.cards.share import ShareService
from sift.cards.vault import Vault
from sift.decks import DeckRegistry
from sift.decks.service import DeckService
from sift.registry import Ledger


class LayerError(Exception):
    """Whole-layer fail-closed error (flag off / not wired)."""


class CardLayer:
    """Composition root for the v4.1 card layer. Server-side only."""

    def __init__(self, ledger: Ledger, db_dir: str | Path,
                 card_secret: bytes | None = None):
        db_dir = Path(db_dir)
        db_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = ledger
        self.cards = CardRegistry(db_dir / "cards.db", secret=card_secret)
        self.minter = CardMinter(ledger, self.cards)
        self.vault = Vault(self.cards, db_dir / "vault.db")
        self.face = CardFaceRenderer(ledger, self.cards)
        self.shares = ShareService(self.cards, self.vault, db_dir / "shares.db")
        self.notify = Notifier(self.cards, db_dir / "notify.db")
        self.events = SignalEventService(self.notify, ledger,
                                         db_dir / "signal.db")
        self.decay = RegistryDecayJob(self.cards, ledger, notifier=self.notify)
        self.metrics = Metrics(ledger, self.cards, self.vault, self.shares)
        # v4.2 deck layer (hermes-tasks #166): its own registry + service,
        # gated by decks_v4_2 (which also requires cards_v4_1).
        self.decks = DeckRegistry(db_dir / "decks.db", secret=card_secret)
        self.deck_service = DeckService(self.cards, self.decks)

    def __enter__(self) -> "CardLayer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        # RegistryDecayJob holds no DB of its own (it writes through the
        # ledger + registry), so there is nothing to close on it.
        for c in (self.events, self.notify, self.shares,
                  self.vault, self.cards, self.decks):
            c.close()

    # -- the one flag guard ---------------------------------------------------------

    def require_layer(self, handle: str | None = None) -> None:
        """Fail closed once for operator paths: raises LayerError(flag_off)
        when cards_v4_1 is off (global kill-switch SIFT_CARDS_V4_1=0,
        set_flag(False), or per-user disable_for_user)."""
        if not flags.cards_v4_1_enabled(handle):
            raise LayerError("flag_off: cards_v4_1 is disabled — the whole "
                             "v4.1 card layer is inert (kill-switch "
                             "SIFT_CARDS_V4_1=0)")

    def flag_status(self, handle: str | None = None) -> dict:
        """Operator read: the effective flag state (global + per-user)."""
        return {
            "flag": flags.FLAG_CARDS_V4_1,
            "global": flags.is_enabled(flags.FLAG_CARDS_V4_1),
            "for_handle": flags.cards_v4_1_enabled(handle),
            "decks_v4_2": {
                "global": flags.is_enabled(flags.FLAG_DECKS_V4_2),
                "for_handle": flags.decks_v4_2_enabled(handle),
            },
            "known_flags": list(flags.known_flags()),
        }


__all__ = ["CardLayer", "LayerError"]
