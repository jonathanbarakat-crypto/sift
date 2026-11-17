"""Card minting wiring (v4.1 mechanics 1, behind `cards_v4_1`).

Hook into the gauntlet runner: after a trial's verdict is decided, if the
flag is ON and the trial is a validated-green survivor, the runner asks the
CardMinter to mint exactly one card (duplicate spec -> ghost). Killed
trials and on-watch-amber survivors (F-REGIME class) mint nothing.
F-DELIST runs are amber/kill paths — nothing mints.

The minter pulls the pre-registered stats from the ledger stage rows
(s3.dsr, s5.statistic = full-depth t_net, s2.p_value) — tier is computed
from those recorded numbers only (mechanics 1: 'tier from pre-registered
stats').

Read-back discipline (hard rule): mint() returns a Card re-read from the
registry; the verdict dict carries `card_id` only when the read-back
succeeded. With the flag OFF, `card_id` is None and nothing is written.
"""

from __future__ import annotations

from sift import flags
from sift.cards import CardRegistry
from sift.registry import Ledger


class CardMinter:
    """Mints cards on gauntlet survivors. Server-side only."""

    def __init__(self, ledger: Ledger, cards: CardRegistry):
        self.ledger = ledger
        self.cards = cards

    def maybe_mint(self, trial_id: str, owner_handle: str,
                   verdict: str, survivor_class: str) -> str | None:
        """Mint on survivor if the flag is on. Returns the public card id
        (read-back verified) or None. F-REGIME/F-DELIST (amber) and killed
        trials mint nothing."""
        if not flags.cards_v4_1_enabled(owner_handle):
            return None
        if verdict != "survivor" or survivor_class != "validated-green":
            return None
        trial = self.ledger.get(trial_id)
        s2 = self.ledger.stage(trial_id, "s2")
        s3 = self.ledger.stage(trial_id, "s3")
        s5 = self.ledger.stage(trial_id, "s5")
        card = self.cards.mint(
            trial_id=trial_id,
            family=trial.family,
            scope_id=trial.scope_id,
            config_hash=trial.config_hash,
            data_vintage_hash=trial.data_vintage_hash,
            survivor_class=survivor_class,
            owner_handle=owner_handle,
            dsr=s3["dsr"],
            t_net_full=s5["statistic"],
            p_perm=s2["p_value"],
        )
        # read-back: confirm the card is retrievable before claiming success
        rb = self.cards.get_card(card.card_id)
        return rb.card_id


__all__ = ["CardMinter"]
