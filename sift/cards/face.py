"""v4.1 card face: server-rendered anonymized public artifact (mechanics 4).

Hard rules (v3 §12 + mechanics 4):
- The card face NEVER contains raw tickers, never directional calls, no
  P&L/win rates. It shows only: sector/industry aggregates, per-family
  specifics (family name, scope label, pre-registered gauntlet stats —
  survival facts), and the survivor class. Raw tickers never leave the
  server: the face renderer receives the ledger + card registry and emits
  aggregates only; any ticker-like string in the rendered face raises.
- The shareable object is watermarked with the viewer's handle + timestamp
  (share watermark), so a leaked card face is traceable.
- Rendering is server-side only; the function returns a dict/JSON-able
  artifact (a web layer can serialize it — nothing raw rides along).

Behind `cards_v4_1`; flag off -> FaceError(flag_off).
"""

from __future__ import annotations

import re
import time

from sift import flags
from sift.cards import Card, CardError, CardRegistry
from sift.registry import Ledger

# Ticker-like tokens (US equities): 2-5 UPPERCASE letters, optionally a
# class suffix. The face must never contain one (v3 §12.3 — no tickers in
# SIFT content). Single uppercase letters and face-internal ALL-CAPS words
# are not tickers and are whitelisted (SIFT, JSON, US, P&L wording, ...).
_TICKER_RE = re.compile(r"\b[A-Z]{2,5}(?:\.[A-Z]{1,2})?\b")
_FACE_WHITELIST = {"SIFT", "JSON", "US", "NO", "ID", "TS", "P&L".replace("&", ""), "L"}

# Sector/industry aggregate buckets the face may show (mechanics 4:
# "sector/industry aggregates"). Pure labels, no tickers.
SECTOR_BUCKETS = (
    "technology", "financials", "healthcare", "energy", "industrials",
    "consumer", "utilities", "materials", "real_estate", "communication",
)


class FaceError(Exception):
    """Card-face rendering misuse (flag off, unknown card, leak detected)."""


def _guard_no_tickers(text: str) -> None:
    """Fail closed: any ticker-shaped token in the rendered face aborts the
    render (raw tickers never leave the server). Face-internal ALL-CAPS
    labels are whitelisted."""
    for tok in _TICKER_RE.findall(text):
        if tok not in _FACE_WHITELIST:
            raise FaceError(f"rendered face contains a ticker-like token {tok!r} — refused")


class CardFaceRenderer:
    """Server-side renderer of the anonymized public card artifact."""

    def __init__(self, ledger: Ledger, cards: CardRegistry):
        self.ledger = ledger
        self.cards = cards

    def render(self, card_id: str, viewer_handle: str,
               now: float | None = None) -> dict:
        """Render the public card face for `viewer_handle`. The artifact is
        watermarked handle+timestamp. Raises FaceError(flag_off) when
        cards_v4_1 is off for the viewer; CardError on unknown/foreign card
        (cards are non-transferable — the face is readable by the owner; for
        a shareable object the same artifact is emitted with the *sharing*
        handle's watermark, see render_share)."""
        if not flags.cards_v4_1_enabled(viewer_handle):
            raise FaceError("flag_off: cards_v4_1 is disabled for this account")
        card = self.cards.get_card(card_id)
        now = time.time() if now is None else now
        face = self._build_face(card, watermark_handle=viewer_handle, now=now)
        # leak guard on the serialized artifact (hard rule: no raw tickers)
        import json
        _guard_no_tickers(json.dumps(face, default=str))
        return face

    def render_share(self, card_id: str, owner_handle: str,
                     now: float | None = None) -> dict:
        """The shareable object: same anonymized face, watermarked with the
        OWNER handle + timestamp (the handle that shared it), so a leaked
        share is traceable to the sharing account."""
        if not flags.cards_v4_1_enabled(owner_handle):
            raise FaceError("flag_off: cards_v4_1 is disabled for this account")
        card = self.cards.assert_owned(card_id, owner_handle)   # non-transferable
        now = time.time() if now is None else now
        face = self._build_face(card, watermark_handle=owner_handle, now=now)
        import json
        _guard_no_tickers(json.dumps(face, default=str))
        return face

    def _build_face(self, card: Card, watermark_handle: str, now: float) -> dict:
        trial_stages = {r["stage"]: dict(r) for r in self.ledger.stages(card.trial_id)}
        s5 = trial_stages.get("s5", {})
        s3 = trial_stages.get("s3", {})
        s2 = trial_stages.get("s2", {})
        # per-family specifics + gauntlet stats = SURVIVAL FACTS ONLY
        face = {
            "card_id": card.card_id,
            "kind": card.kind,
            "ghost_no": card.ghost_no if (card.ghost_no or 0) > 0 else None,
            "family": card.family,                       # e.g. "pairs" — not a ticker
            "scope_id": card.scope_id,                   # labeled scope, honest
            "tier": card.tier,
            "survivor_class": card.survivor_class,
            "sector_aggregate": self._sector_aggregate(card),
            "gauntlet_stats": {
                "stages_passed": [s for s, r in trial_stages.items() if r["status"] == "pass"],
                "s2_p_value": s2.get("p_value"),
                "s3_dsr": s3.get("dsr"),
                "s5_full_depth_t": s5.get("statistic"),
                "s5_status": s5.get("status"),
            },
            "watermark": {
                "handle": watermark_handle,
                "ts": now,
                "note": "shareable object watermarked handle+timestamp (v4.1 m4)",
            },
            "disclaimer": "SIFT does not trade. Survival facts only — not "
                          "investment advice; no P&L, no picks (v3 §12).",
        }
        return face

    def _sector_aggregate(self, card: Card) -> dict:
        """Sector/industry AGGREGATE only — derived from the family + scope,
        never from raw holdings. Deterministic bucket label, no tickers."""
        # family -> coarse sector aggregate label (aggregates, not names)
        fam = card.family.lower()
        bucket = "diversified_multi_sector"
        if "pairs" in fam:
            bucket = "cross_sectional_pairs_aggregate"
        elif "sector" in fam:
            bucket = "sector_neutral_aggregate"
        return {
            "bucket": bucket,
            "level": "aggregate",
            "scope_id": card.scope_id,
        }


__all__ = ["CardFaceRenderer", "FaceError", "SECTOR_BUCKETS"]
