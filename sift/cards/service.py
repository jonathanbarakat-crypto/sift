"""v4.1 card layer — end-to-end wiring facade (behind `cards_v4_1`).

`CardLayer` is the single server-side composition root for the whole v4.1
card layer:

    trial registry (Ledger)
      -> CardMinter / CardRegistry   (mint on validated-green, pull/13)
      -> Vault                       (custody, capacity, ghosts, pull/14)
      -> CardFaceRenderer            (anonymized shareable face, pull/15)
      -> anchored run mode           (pull/16)
      -> ShareService                (share-for-capacity, cards 5/8, pull/18)
      -> BadgeService                (knowledge-reward badges, cards 6/8)
      -> Notifier                    (per-card opt-in notifications, cards 7/8, pull/19)
      -> RegistryDecayJob            (downgrade-only re-evaluation, cards 7/8, pull/19)
      -> Metrics                     (gate-1 metric hooks)

Every component independently enforces the flag (each mutating path raises
its own fail-closed error type with flag_off when `cards_v4_1` is off);
this facade adds ONE guard, `require_layer(handle)`, so an operator path
that touches several components fails closed once, up front, with
LayerError(flag_off). The env kill-switch SIFT_CARDS_V4_1=0 disables the
whole layer because every read goes through sift.flags.
"""

from __future__ import annotations

from pathlib import Path

from sift import flags
from sift.cards import CardRegistry
from sift.cards.anchored import AnchoredRunError  # noqa: F401  (re-export surface)
from sift.cards.badges import BadgeService
from sift.cards.decay import RegistryDecayJob
from sift.cards.face import CardFaceRenderer
from sift.cards.metrics import Metrics
from sift.cards.minting import CardMinter
from sift.cards.notify import Notifier
from sift.cards.share import ShareService
from sift.cards.vault import Vault
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
        self.badges = BadgeService(ledger, db_dir / "badges.db")
        self.notify = Notifier(self.cards, db_dir / "notify.db")
        self.decay = RegistryDecayJob(self.cards, ledger, notifier=self.notify)
        self.metrics = Metrics(ledger, self.cards, self.vault, self.shares)

    def __enter__(self) -> "CardLayer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        # RegistryDecayJob holds no DB of its own (it writes through the
        # ledger + registry), so there is nothing to close on it.
        for c in (self.notify, self.badges, self.shares,
                  self.vault, self.cards):
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
            "known_flags": list(flags.known_flags()),
        }


__all__ = ["CardLayer", "LayerError"]
