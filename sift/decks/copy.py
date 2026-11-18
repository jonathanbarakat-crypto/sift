"""v4.2 deck-layer copy (spec v4.2 §5, #166) — the ONLY place deck strings
are built. All user-facing deck text goes through `assert_deck_copy_clean`
(the SAME fail-closed banned-phrase guard as cards — v3 §12 bans extended
to ALL deck strings): no buy/sell/long/short, no P&L/returns, no
profit-expectation or "value" language, no urgency/countdown. Any banned
phrase -> DeckCopyError: the render is refused, never ships.

Beta posture (hard): deck results are AMBER TELEMETRY ONLY, never
performance claims. Ranked decks show the F-S rank as an evidence label
with the telemetry caveat; custom decks are informational-only (amber,
unranked, unshareable).

No tickers, no sign/direction anywhere: builders take NO symbol and NO
sign inputs — membership is card ids, magnitude is always |z|-style
magnitude only, decks never carry direction. Flag off -> DeckCopyError
(flag_off); kill-switch SIFT_DECKS_V4_2=0 (and SIFT_CARDS_V4_1=0).
"""

from __future__ import annotations

from sift import flags
from sift.cards.copy import BANNED_PHRASES, BANNED_WORDS, CopyError, assert_copy_clean


class DeckCopyError(CopyError):
    """Deck-copy misuse: flag off, banned phrase, bad input."""


def _require_flag(handle: str | None = None) -> None:
    if not flags.decks_v4_2_enabled(handle):
        raise DeckCopyError(
            "flag_off: decks_v4_2 is disabled (kill-switch SIFT_DECKS_V4_2=0"
            " or SIFT_CARDS_V4_1=0)"
        )


def assert_deck_copy_clean(text: str) -> str:
    """Fail-closed guard for EVERY deck string: delegates to the shared
    card guard (substring BANNED_PHRASES + whole-word BANNED_WORDS) so the
    v3 §12 bans apply to ALL deck strings exactly as to card strings.
    Raises DeckCopyError on any hit."""
    try:
        return assert_copy_clean(text)
    except CopyError as e:
        raise DeckCopyError(str(e)) from e


# -- deck face / telemetry copy ------------------------------------------------

def deck_explainer(kind: str, handle: str | None = None) -> str:
    """One-paragraph explainer on every deck view. A deck is the TESTED
    unit: 'does this pre-registered set of cards, run as a portfolio with
    realistic costs, clear the walk-forward bar?' Never an instruction."""
    _require_flag(handle)
    if kind == "custom":
        text = (
            "This is your own hand-picked set, shown for information only. "
            "It is not ranked and cannot be shared as a claim: any "
            "hand-picked set can be chosen after the fact, so only "
            "pre-registered buckets are ranked. It says nothing about what "
            "to do with anything — SIFT shows evidence; you make the calls."
        )
    else:
        text = (
            "This deck is a pre-registered set of cards from a fixed bucket, "
            "run as an equal-weight portfolio with realistic costs through "
            "the walk-forward stress test. Its rank is an evidence label "
            "about the portfolio's durability in the research window — "
            "amber telemetry, not a performance claim, and nothing about "
            "what to do with it. SIFT shows evidence; you make the calls."
        )
    return assert_deck_copy_clean(text)


def deck_rank_label(rank: str | None, kind: str,
                    handle: str | None = None) -> str:
    """The F-S rank label with its mandatory beta caveat. Custom decks are
    always 'informational only' — no rank is ever shown for them."""
    _require_flag(handle)
    if kind == "custom" or rank is None:
        return assert_deck_copy_clean(
            "informational only — unranked (amber telemetry, not a claim)"
        )
    if rank not in ("F", "D", "C", "B", "A", "S"):
        raise DeckCopyError(f"unknown deck rank {rank!r}")
    return assert_deck_copy_clean(
        f"deck evidence rank {rank} — amber telemetry about durability in the "
        f"research window, not a performance claim"
    )


def deck_telemetry_caveat(handle: str | None = None) -> str:
    """The standing beta caveat shown with every deck result: amber
    telemetry only, never a performance claim."""
    _require_flag(handle)
    return assert_deck_copy_clean(
        "Beta: deck results are telemetry about how the pre-registered set "
        "held up under the stress test. They are not performance figures and "
        "not a claim about the future."
    )


def deck_custom_warning(handle: str | None = None) -> str:
    """The amber banner on custom (hand-picked) decks: informational only,
    unranked, unshareable."""
    _require_flag(handle)
    return assert_deck_copy_clean(
        "Hand-picked set: informational only. It is scored and shown back to "
        "you, but it is unranked and cannot be shared as a claim, because "
        "arbitrary picks can be chosen after the fact."
    )


DECK_GLOSSARY: dict[str, str] = {
    "deck": (
        "A pre-registered set of cards run as one equal-weight portfolio "
        "through the walk-forward stress test. The deck — not a single "
        "card — is the tested unit."
    ),
    "bucket": (
        "One cell of the fixed, enumerable deck grid: a card count from "
        "{5, 10, 20, 50, 100}, an optional sector split, and a family mix. "
        "Membership in a ranked deck is deterministic: the top cards by "
        "evidence tier in that bucket."
    ),
    "deck rank": (
        "An evidence label (F, D, C, B, A, S) for the PORTFOLIO claim, set "
        "only by pre-registered stress-test stats. Distinct from card "
        "tiers. Beta: amber telemetry only, never a performance claim."
    ),
    "effective n": (
        "How many independent bets a deck really contains. Cards that "
        "share names move together, so 50 mixed cards are not 50 "
        "independent bets — the count is corrected for overlap."
    ),
    "pbo": (
        "Probability that the stress test itself was overfit — the share "
        "of test configurations where the best in-sample choice ends up "
        "in the bottom half out of sample. A durability fact about the "
        "test, not a forecast."
    ),
    "rebalance cadence": (
        "How often the deck is reset to equal weights — fixed in advance, "
        "every 21 trading days. Fixed rules are what make the test "
        "pre-registered."
    ),
}


def deck_glossary(handle: str | None = None) -> dict[str, str]:
    """Deck glossary tooltips, every entry guard-checked. Flag off ->
    DeckCopyError(flag_off)."""
    _require_flag(handle)
    return {k: assert_deck_copy_clean(v) for k, v in DECK_GLOSSARY.items()}


__all__ = [
    "DECK_GLOSSARY",
    "DeckCopyError",
    "assert_deck_copy_clean",
    "deck_custom_warning",
    "deck_explainer",
    "deck_glossary",
    "deck_rank_label",
    "deck_telemetry_caveat",
]
