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
from sift.cards import Card, CardRegistry
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


def _guard_face_artifact(face: dict) -> None:
    """Leak guard on the serialized face artifact: every string field is
    guarded. Base64 image payloads (keys ending in `_b64`) are not text —
    every string DRAWN into the PNG was already guarded at draw time by the
    renderer — so they are skipped here (base64 alphabet produces accidental
    ALL-CAPS runs that are not tickers)."""
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, str) and k.endswith("_b64"):
                    continue
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            _guard_no_tickers(o)
    walk(face)


class CardFaceRenderer:
    """Server-side renderer of the anonymized public card artifact."""

    def __init__(self, ledger: Ledger, cards: CardRegistry):
        self.ledger = ledger
        self.cards = cards

    def render(self, card_id: str, viewer_handle: str,
               now: float | None = None, *, with_image: bool = False,
               diagram: dict | None = None,
               current_abs_z: float | None = None,
               health_ratio: float | None = None) -> dict:
        """Render the public card face for `viewer_handle`. The artifact is
        watermarked handle+timestamp. Raises FaceError(flag_off) when
        cards_v4_1 is off for the viewer; CardError on unknown/foreign card
        (cards are non-transferable — the face is readable by the owner; for
        a shareable object the same artifact is emitted with the *sharing*
        handle's watermark, see render_share).

        The face now carries the newcomer-friendly blocks (all built ONLY in
        sift/cards/copy.py, all banned-phrase fail-closed):
        `explainer` (one-paragraph plain-language face explainer),
        `glossary` (jargon behind tooltips, not in the main view),
        `gauge` (stretched/normal/compressed band from |z| + the numeric
        |z| one click away) and `health` (plain ratio language vs the mint
        baseline). `current_abs_z` / `health_ratio` are OPTIONAL live
        readings supplied by the server (magnitude is always |z| — never a
        sign; health is always a ratio — never a level).

        With `with_image=True` the PIL renderer (sift/cards/render.py,
        server-side only) additionally attaches an optional `image` field
        (base64 2x + 1x PNG of the anonymized face) built from this face dict
        plus the per-family `diagram` inputs (anonymized aggregate series
        only). The JSON artifact is otherwise unchanged; every string drawn
        into the image passes the same _guard_no_tickers."""
        if not flags.cards_v4_1_enabled(viewer_handle):
            raise FaceError("flag_off: cards_v4_1 is disabled for this account")
        card = self.cards.get_card(card_id)
        now = time.time() if now is None else now
        face = self._build_face(card, watermark_handle=viewer_handle, now=now,
                                current_abs_z=current_abs_z,
                                health_ratio=health_ratio)
        # leak guard on the serialized artifact (hard rule: no raw tickers)
        _guard_face_artifact(face)
        if with_image:
            from sift.cards.render import attach_card_image
            face = attach_card_image(face, diagram=diagram)
            _guard_face_artifact(face)
        return face

    def render_share(self, card_id: str, owner_handle: str,
                     now: float | None = None, *, with_image: bool = False,
                     diagram: dict | None = None,
                     current_abs_z: float | None = None,
                     health_ratio: float | None = None) -> dict:
        """The shareable object: same anonymized face, watermarked with the
        OWNER handle + timestamp (the handle that shared it), so a leaked
        share is traceable to the sharing account. `with_image` attaches the
        code-rendered PNG face (server-side PIL, watermarked) as the optional
        `image` field — the shareable visual artifact. `current_abs_z` /
        `health_ratio` are the same optional live readings as in render()."""
        if not flags.cards_v4_1_enabled(owner_handle):
            raise FaceError("flag_off: cards_v4_1 is disabled for this account")
        card = self.cards.assert_owned(card_id, owner_handle)   # non-transferable
        now = time.time() if now is None else now
        face = self._build_face(card, watermark_handle=owner_handle, now=now,
                                current_abs_z=current_abs_z,
                                health_ratio=health_ratio)
        _guard_face_artifact(face)
        if with_image:
            from sift.cards.render import attach_card_image
            face = attach_card_image(face, diagram=diagram)
            _guard_face_artifact(face)
        return face

    def _build_face(self, card: Card, watermark_handle: str, now: float,
                    current_abs_z: float | None = None,
                    health_ratio: float | None = None) -> dict:
        from sift.cards import copy as card_copy
        trial_stages = {r["stage"]: dict(r) for r in self.ledger.stages(card.trial_id)}
        s5 = trial_stages.get("s5", {})
        s3 = trial_stages.get("s3", {})
        s2 = trial_stages.get("s2", {})
        s0 = trial_stages.get("s0", {})
        # per-family specifics + gauntlet stats = SURVIVAL FACTS ONLY.
        # Read strictly from the ledger STAGE ROWS — never from params_json —
        # so no server-side symbol can reach the face by construction.
        face = {
            "card_id": card.card_id,
            "kind": card.kind,
            "ghost_no": card.ghost_no if (card.ghost_no or 0) > 0 else None,
            "family": card.family,                       # e.g. "pairs" — not a ticker
            "scope_id": card.scope_id,                   # labeled scope, honest
            "tier": card.tier,
            "survivor_class": card.survivor_class,
            "sector_aggregate": self._sector_aggregate(card),
            "specifics": self._family_specifics(card, s0),
            "gauntlet_stats": {
                "stages_passed": [s for s, r in trial_stages.items() if r["status"] == "pass"],
                "s2_p_value": s2.get("p_value"),
                "s3_dsr": s3.get("dsr"),
                "s5_full_depth_t": s5.get("statistic"),
                "s5_status": s5.get("status"),
            },
            # soundness #10: the mint-time holdout baseline persisted on the
            # card at mint (frozen-holdout live-baseline). None when the mint
            # run had no holdout — never fabricated. Survival facts only; the
            # note string is built in copy.py and scanned fail-closed there.
            "mint_baseline": (
                {
                    "t_net": card.baseline_t_net,
                    "sr_ann": card.baseline_sr_ann,
                    "holdout_bars": card.baseline_holdout_bars,
                    "note": card_copy.mint_baseline_note(),
                }
                if card.baseline_t_net is not None
                or card.baseline_sr_ann is not None
                or card.baseline_holdout_bars is not None
                else None
            ),
            # -- newcomer-friendly layer (all strings built ONLY in
            # sift/cards/copy.py and banned-phrase fail-closed there) ----
            "explainer": card_copy.face_explainer(
                stages=len([s for s, r in trial_stages.items() if r["status"] == "pass"])
                or 5, handle=watermark_handle),
            "glossary": card_copy.glossary(handle=watermark_handle),
            "gauge": {
                # gauge-style magnitude: band from |z| ONLY (never the
                # sign); the numeric |z| rides along one click away.
                "band": card_copy.gauge_band(current_abs_z if current_abs_z is not None else 0.0),
                "label": card_copy.gauge_label(
                    card_copy.gauge_band(current_abs_z if current_abs_z is not None else 0.0)),
                "magnitude_z": (round(abs(float(current_abs_z)), 4)
                                if current_abs_z is not None else None),
                "note": "magnitude is |z| only — no sign, no direction",
            },
            "health": {
                # plain ratio language vs the mint baseline
                "ratio_vs_mint_baseline": (round(float(health_ratio), 4)
                                           if health_ratio is not None else None),
                "phrase": card_copy.health_phrase(health_ratio),
                "note": "a ratio vs the mint baseline — an observation about "
                        "the signal, not a recommendation",
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

    def _family_specifics(self, card: Card, s0: dict) -> dict:
        """Per-family anonymized specifics (mechanics 4): lead-lag lag /
        hedge ratio + half-life — numeric survival facts from the S0 stage
        row only. No symbols, no directions, no P&L."""
        fam = card.family.lower()
        spec: dict = {}
        if "lead" in fam or "lag" in fam:
            lag = s0.get("lag_days")
            if lag is not None:
                spec["lag_days"] = lag
            xcorr = s0.get("peak_xcorr")
            if xcorr is not None:
                spec["peak_xcorr"] = xcorr
        else:
            # pairs / hedge / mean-reversion: hedge ratio + half-life
            beta = s0.get("statistic")          # S0 hedge-ratio estimate
            if beta is not None:
                spec["hedge_ratio"] = round(float(beta), 4)
            hl = s0.get("half_life_days")
            if hl is not None:
                spec["half_life_days"] = round(float(hl), 2)
            if "hedge" not in fam and "duration" not in fam:
                spec["entry_z"] = 2.0           # pre-registered entry threshold
        return spec


__all__ = ["CardFaceRenderer", "FaceError", "SECTOR_BUCKETS",
           "_guard_no_tickers", "_guard_face_artifact"]
