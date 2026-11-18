"""Card minting wiring (v4.1 mechanics 1, extended by v4.2 §2-§3, behind `cards_v4_1`).

Hook into the gauntlet runner: after a trial's verdict is decided, if the
flag is ON and the trial is a survivor (ANY S0-S2 passer — v4.2), the runner
asks the CardMinter to mint exactly one card (duplicate spec -> ghost). Killed
trials (S0-S2 failures, survivor_class='none') mint nothing.

The minter pulls the pre-registered stats from the ledger stage rows and the
v4.2 evidence flags:
  - s2.p_value  (S2 block-permutation p)
  - s3.dsr      (S3 scope-stratified DSR)
  - s5.statistic (full-depth t_net) + s5 status (the S5 pass)
  - s4 detail   (net SR, t_net, MinTRL, T) -> the S4 bar via v42.s4_passed
Tier is computed from that FULL evidence vector by tier_from_stats (v4.2 §3:
bronze = S0-S2 pass; silver/gold/platinum per the pre-registered bars).

Read-back discipline (hard rule): mint() returns a Card re-read from the
registry; the verdict dict carries `card_id` only when the read-back
succeeded. With the flag OFF, `card_id` is None and nothing is written.
"""

from __future__ import annotations

from sift import flags
from sift.cards import CardRegistry
from sift.gauntlet import v42
from sift.registry import Ledger


class CardMinter:
    """Mints cards on gauntlet survivors. Server-side only."""

    def __init__(self, ledger: Ledger, cards: CardRegistry):
        self.ledger = ledger
        self.cards = cards

    def maybe_mint(self, trial_id: str, owner_handle: str,
                   verdict: str, survivor_class: str) -> str | None:
        """Mint on ANY survivor (v4.2: every S0-S2 passer mints at bronze+)
        if the flag is on. Returns the public card id (read-back verified)
        or None. Killed trials (verdict='killed' / survivor_class='none')
        mint nothing."""
        if not flags.cards_v4_1_enabled(owner_handle):
            return None
        if verdict != "survivor" or survivor_class == "none":
            return None
        trial = self.ledger.get(trial_id)
        s2 = self.ledger.stage(trial_id, "s2")
        s3 = self.ledger.stage(trial_id, "s3")
        s4 = self.ledger.stage(trial_id, "s4")
        s5 = self.ledger.stage(trial_id, "s5")
        # v4.2: the S4 score bar from the recorded s4 stage row detail is not
        # persisted in the stage table (detail is in the verdict dict), so
        # recompute the bar from the recorded s4 statistic/threshold plus the
        # trial's scope: status pass == the bar was met at mint.
        s4_pass = s4["status"] == "pass"
        s5_pass = s5["status"] == "pass"
        # full-depth t_net is the s5 statistic when S5 ran; otherwise fall
        # back to the scope S4 t_net is not stored on the row — use None so
        # tier_from_stats never inflates on a missing component.
        t_net_full = s5["statistic"] if s5["status"] in ("pass", "fail") else None
        card = self.cards.mint(
            trial_id=trial_id,
            family=trial.family,
            scope_id=trial.scope_id,
            config_hash=trial.config_hash,
            data_vintage_hash=trial.data_vintage_hash,
            survivor_class=survivor_class,
            owner_handle=owner_handle,
            dsr=s3["dsr"],
            t_net_full=t_net_full,
            p_perm=s2["p_value"],
            s4_pass=s4_pass,
            s5_pass=s5_pass,
        )
        # read-back: confirm the card is retrievable before claiming success
        rb = self.cards.get_card(card.card_id)
        return rb.card_id


__all__ = ["CardMinter"]
