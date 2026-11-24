"""v4.2 deck-layer copy + artifact layer (spec v4.2 §5, #166).

This module is the ONLY place user-facing deck strings, deck event payloads,
and watermarked deck artifacts are built:

- `DECK_RANKS` / `deck_rank_label(...)` / `deck_rank_phrase(...)` — the
  F–S evidence-rank labels for TESTED PORTFOLIOS (decks), distinct from the
  card bronze/silver/gold/platinum evidence tiers. Rank is evidence strength
  of the portfolio claim only; in beta every deck result is amber telemetry
  and is never displayed as performance. Custom decks are informational only
  (amber, unranked, unshareable).
- `deck_explainer(...)` — the one-paragraph plain-language deck face: what a
  deck certifies (a pre-registered set of cards, tested as a portfolio with
  realistic costs, cleared the walk-forward bar) and what it is not.
- `deck_telemetry_caveat()` / `deck_custom_warning()` / `deck_glossary()` —
  standing beta telemetry caveat, custom-deck amber banner, and tooltip
  glossary.
- `cards_are_candidates_line()` — the explicit onboarding/first-card line:
  NO SINGLE CARD IS AN INSTANT WIN. Cards are evidence about connections
  (candidate tier + health); decks are the tested portfolios.
- `deck_event_copy(...)` — plain-language RSS/API/email gloss for deck
  events, built only here and covered by the banned-phrase grep test.
- `build_deck_event_payload(...)` — the versioned deck-event payload for
  RSS/API channels: magnitude / aggregate / rank ONLY. Same fail-closed
  guards as the card signal-event payload (sift/cards/events.py):
  FORBIDDEN_PAYLOAD_KEYS (no symbol/sign/direction/P&L keys), ticker-token
  scan, non-negative magnitude, banned-phrase scan of every gloss string.
- `build_deck_artifact(...)` — the watermarked deck artifact (like the card
  face): watermark handle+ts, rank label, aggregate stats from LEDGER STAGE
  ROWS only — never params_json — then `_guard_deck_artifact` re-scans the
  serialized artifact for ticker-shaped tokens and directional words.

Copy discipline (HARD, fail-closed — v4.2 §5, extending v3 §12 + v4.1 §6):
- BANNED_PHRASES_DECK extends the shared card/notify lists with deck-named
  bans: no "top deck", no "best portfolio", no "winner", no buy/sell/long/
  short, no P&L/returns, no profit-expectation/"value" language, no
  urgency/countdown. Any hit -> DeckCopyError / NotifyError: the render or
  delivery is refused, never shipped.
- No tickers, no sign/direction anywhere in deck artifacts or payloads
  (ledger stage rows only, never params_json). Magnitude is always absolute;
  rank is always a label from DECK_RANKS (or None for custom/unranked).
- Decks are non-transferable like cards; no marketplace/resale; no money in
  beta; ToS day one; SIFT does not trade; the user makes ALL trading
  decisions.

Flag: gated by `decks_v4_2` (which ANDs `decks_v4_2` and `cards_v4_1` in
sift/flags.py, so disabling either flag kills deck copy/artifact builders).
Flag off -> DeckCopyError(flag_off). Kill-switches: SIFT_DECKS_V4_2=0 and
SIFT_CARDS_V4_1=0.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from sift import flags
from sift.cards.copy import (
    BANNED_PHRASES as _CARD_BANNED_PHRASES,
)
from sift.cards.copy import (
    BANNED_WORDS as _CARD_BANNED_WORDS,
)
from sift.cards.copy import (
    CopyError,
    assert_copy_clean,
)
from sift.cards.events import (
    _PAYLOAD_WHITELIST,
    _TICKER_RE,
    FORBIDDEN_PAYLOAD_KEYS,
    _guard_no_tickers_payload,
)
from sift.cards.notify import NotifyError, contains_banned


class DeckCopyError(CopyError):
    """Deck copy/artifact misuse: flag off, banned phrase in a built
    string, bad input, leak detected. Fail closed."""


# -- banned phrases (extended with the deck-named bans) -----------------------
# Shares the card/notify guard discipline (substring BANNED_PHRASES +
# whole-word directional BANNED_WORDS) and adds the v4.2 §5 deck bans:
# no "top deck", no "best portfolio", no superlative/comparative framing of
# one deck over another.
_EXTRA_DECK_BANNED = (
    "top deck",
    "best deck",
    "top portfolio",
    "best portfolio",
    "top pick",
    "best pick",
    "winning deck",
    "winner",
    "champion",
    "crushes",
    "dominates",
    "guaranteed outcome",
    "sure thing",
)
BANNED_PHRASES_DECK = _CARD_BANNED_PHRASES + _EXTRA_DECK_BANNED
BANNED_WORDS_DECK = _CARD_BANNED_WORDS
_BANNED_WORD_RE_DECK = re.compile(
    r"\b(?:" + "|".join(BANNED_WORDS_DECK) + r")\b", re.IGNORECASE
)


def contains_banned_deck(text: str) -> str | None:
    """Return the first banned phrase/word found in `text`, or None when
    clean. Single fail-closed check used by every deck builder (substring
    scan over BANNED_PHRASES_DECK, word-boundary scan over
    BANNED_WORDS_DECK)."""
    low = text.lower()
    for phrase in BANNED_PHRASES_DECK:
        if phrase in low:
            return phrase
    m = _BANNED_WORD_RE_DECK.search(low)
    if m:
        return m.group(0)
    return None


def assert_deck_copy_clean(text: str) -> str:
    """Fail-closed guard for EVERY deck string: enforces both the shared
    card/notify guard (v3 §12 bans) and the v4.2 §5 deck-specific bans.
    Raises DeckCopyError on any hit. Returns the text unchanged."""
    try:
        assert_copy_clean(text)
    except CopyError as e:
        raise DeckCopyError(str(e)) from e
    banned = contains_banned(text)
    if banned is not None:
        raise DeckCopyError(f"deck copy contains banned phrase {banned!r} — refused")
    banned = contains_banned_deck(text)
    if banned is not None:
        raise DeckCopyError(f"deck copy contains banned phrase {banned!r} — refused")
    return text


def _require_flag(handle: str | None = None) -> None:
    if not flags.decks_v4_2_enabled(handle):
        raise DeckCopyError(
            "flag_off: decks_v4_2 is disabled (kill-switch SIFT_DECKS_V4_2=0"
            " or SIFT_CARDS_V4_1=0)"
        )


# -- deck evidence ranks F–S (tested portfolios; distinct from card tiers) ----
# Rank = evidence strength of the PORTFOLIO claim (S6 gauntlet + multiplicity-
# corrected over the deck stratum). In beta all deck results are amber
# telemetry — never displayed as performance.
DECK_RANKS = ("F", "D", "C", "B", "A", "S")
DECK_RANK_MIN = "F"
DECK_RANK_MAX = "S"

_RANK_LABELS = {
    "F": "F — weakest portfolio evidence in the pre-registered grid (amber telemetry, not a performance claim)",
    "D": "D — weak portfolio evidence (amber telemetry, not a performance claim)",
    "C": "C — moderate portfolio evidence (amber telemetry, not a performance claim)",
    "B": "B — solid portfolio evidence (amber telemetry, not a performance claim)",
    "A": "A — strong portfolio evidence (amber telemetry, not a performance claim)",
    "S": "S — strongest portfolio evidence in the pre-registered grid (amber telemetry, not a performance claim)",
}

_RANK_PHRASES = {
    "F": "the portfolio evidence is weak",
    "D": "the portfolio evidence is below the mid of the grid",
    "C": "the portfolio evidence is moderate",
    "B": "the portfolio evidence is solid",
    "A": "the portfolio evidence is strong",
    "S": "the portfolio evidence is the strongest in the grid",
}


def deck_rank_label(
    rank: str | None, kind: str = "ranked", handle: str | None = None
) -> str:
    """The F–S rank label drawn on deck views and artifacts, with its
    mandatory beta amber-telemetry caveat. Custom decks (or None rank) are
    always 'informational only — unranked'. Unknown rank -> refused (fail
    closed). Flag off -> refused."""
    _require_flag(handle)
    if kind == "custom" or rank is None:
        return assert_deck_copy_clean(
            "informational only — unranked (amber telemetry, not a claim)"
        )
    r = str(rank).strip().upper()
    if r not in DECK_RANKS:
        raise DeckCopyError(f"unknown deck rank {rank!r} — refused")
    return assert_deck_copy_clean(_RANK_LABELS[r])


def deck_rank_phrase(rank: str, handle: str | None = None) -> str:
    """The plain-language rank sentence for notifications/RSS/API gloss.
    Flag off -> refused."""
    _require_flag(handle)
    r = str(rank).strip().upper()
    if r not in DECK_RANKS:
        raise DeckCopyError(f"unknown deck rank {rank!r} — refused")
    return assert_deck_copy_clean(_RANK_PHRASES[r])


# -- the explicit line: no single card is an instant win ----------------------

def cards_are_candidates_line(handle: str | None = None) -> str:
    """The explicit v4.2 §5 onboarding/first-card line: NO single card is an
    instant win. Cards are evidence about connections (candidate tier +
    health); decks are the tested portfolios with F–S evidence ranks; the
    user makes all trading decisions; SIFT does not trade.

    This string is a LEGAL-adjacent clause: it contains "trade/trading" only
    inside the negated "SIFT does not trade / you make all the trading
    decisions" mandated wording, so it scans against the legal-clause guard
    (every ban except the words inside that mandated negation)."""
    _require_flag(handle)
    text = (
        "No single card is an instant win. A card is evidence about a "
        "connection — a candidate with an evidence tier and a health ratio, "
        "nothing more. Decks are the tested portfolios: pre-registered sets "
        "of cards that cleared the portfolio walk-forward bar, ranked F to S "
        "by evidence strength. You make all trading decisions; SIFT does "
        "not trade."
    )
    return _assert_candidates_clause_clean(text)


# The mandated clause uses "trading"/"trade" only inside "You make all
# trading decisions; SIFT does not trade". Those two words are the only
# relaxation; everything else scans full.
_CANDIDATES_SKIP_WORDS = ("trade", "trading")
_CANDIDATES_SKIP_PHRASES: tuple[str, ...] = ()
_CANDIDATES_BANNED = tuple(
    p for p in BANNED_PHRASES_DECK if p not in _CANDIDATES_SKIP_PHRASES
)


def _assert_candidates_clause_clean(text: str) -> str:
    low = text.lower()
    for phrase in _CANDIDATES_BANNED:
        if phrase in low:
            raise DeckCopyError(f"deck copy contains banned phrase {phrase!r} — refused")
    for w in BANNED_WORDS_DECK:
        if w in _CANDIDATES_SKIP_WORDS:
            continue
        if re.search(r"\b" + w + r"\b", low):
            raise DeckCopyError(f"deck copy contains banned word {w!r} — refused")
    # the directional/value words must appear ONLY inside the mandated
    # negated clause — strip that clause and re-scan everything else full.
    residue = low
    for mandated in ("you make all trading decisions", "sift does not trade"):
        residue = residue.replace(mandated, " ")
    banned = contains_banned(residue) or contains_banned_deck(residue)
    if banned is not None:
        raise DeckCopyError(f"deck copy contains banned phrase {banned!r} — refused")
    return text


# -- deck face / telemetry copy -----------------------------------------------

def deck_explainer(
    kind: str = "ranked",
    *,
    bucket: str | None = None,
    stages: int = 6,
    handle: str | None = None,
) -> str:
    """One-paragraph explainer on every deck view and artifact. A deck is
    the TESTED unit: 'does this pre-registered set of cards, run as a
    portfolio with realistic costs, clear the walk-forward bar?' Never an
    instruction."""
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
        b = f" in the {bucket} bucket" if bucket else " from a fixed bucket"
        text = (
            f"This deck is a pre-registered set of cards{b}, run as an "
            f"equal-weight portfolio with realistic costs through the "
            f"pre-registered {stages}-stage walk-forward stress test "
            f"(S0-S{stages - 1}), with multiplicity corrected over the whole "
            f"deck stratum. Its F-S rank is an evidence label about the "
            f"portfolio's durability in the research window — amber "
            f"telemetry, not a performance claim, and nothing about what to "
            f"do with it. No single card is an instant win; SIFT shows "
            f"evidence; you make the calls."
        )
    return assert_deck_copy_clean(text)


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


# -- deck event copy (RSS / API / email gloss) --------------------------------

def deck_event_copy(
    event: str,
    deck_id: str,
    rank: str | None = None,
    magnitude: float | None = None,
    aggregate: int | None = None,
    handle: str | None = None,
) -> str:
    """The ONLY user-facing deck-event text builder (RSS/API/email gloss).
    Carries magnitude (absolute), aggregate counts, and the F–S rank only —
    never a sign, never a symbol, never a direction, never performance."""
    _require_flag(handle)
    parts = [f"Deck event (deck {deck_id})"]
    if rank is not None:
        parts.append(
            f"evidence rank {deck_rank_label(rank, handle=handle)} — "
            f"{deck_rank_phrase(rank, handle=handle)}."
        )
    if magnitude is not None:
        mag = abs(float(magnitude))
        parts.append(
            f"Portfolio magnitude |z| = {mag:.2f} (magnitude only — "
            f"the event carries no sign and no component names)."
        )
    if aggregate is not None:
        parts.append(
            f"Decks evaluated in this stratum today: {int(aggregate)} "
            f"(anonymized aggregate count)."
        )
    parts.append(
        "This is a monitoring note about your own saved deck — evidence "
        "strength of the portfolio claim, nothing more. It is not a "
        "recommendation; SIFT does not trade and does not tell you what to "
        "do."
    )
    return assert_deck_copy_clean(" ".join(parts))


# -- deck event payload (RSS/API): magnitude / aggregate / rank ONLY ----------

DECK_PAYLOAD_SCHEMA_VERSION = "v1"

DECK_EVENT_TYPES = ("deck_hit", "deck_minted", "deck_ranked")


def build_deck_event_payload(
    *,
    deck_id: str,
    bucket: str,
    rank: str | None,
    event: str = "deck_hit",
    magnitude: float | None = None,
    aggregate: int | None = None,
    now: float = 0.0,
    handle: str | None = None,
) -> dict:
    """Versioned deck-event payload for RSS/API channels. Carries ONLY:
    schema, event, deck_id, bucket (a pre-registered grid label, not a
    ticker), rank (F–S label), magnitude (ABSOLUTE |z| — a negative value
    raises NotifyError because a sign would leak direction), aggregate
    count, ts (UTC), and the gloss built ONLY in `deck_event_copy`.

    Fail closed: forbidden keys (symbol/sign/direction/P&L), ticker-shaped
    tokens, signed magnitude, or any banned phrase in the gloss raise
    NotifyError and the event is not delivered."""
    _require_flag(handle)
    if event not in DECK_EVENT_TYPES:
        raise NotifyError(f"unknown deck event {event!r}")
    if magnitude is not None:
        # magnitude is ABSOLUTE: take |z|; the sign never ships. The
        # post-build guard still refuses any negative that slips through.
        magnitude = round(abs(float(magnitude)), 6)
    payload: dict = {
        "schema": DECK_PAYLOAD_SCHEMA_VERSION,
        "event": event,
        "deck_id": deck_id,
        "bucket": bucket,  # grid label e.g. "n10_all_pairs"
        "rank": (str(rank).strip().upper() if rank is not None else None),
        "magnitude_z": magnitude,
        "decks_in_stratum_today": (int(aggregate) if aggregate is not None else None),
        "ts": (
            datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if now
            else None
        ),
    }
    if payload["rank"] is not None and payload["rank"] not in DECK_RANKS:
        raise NotifyError(
            f"deck payload carries unknown rank {payload['rank']!r} — refused"
        )
    payload["gloss"] = deck_event_copy(
        event,
        deck_id,
        rank=payload["rank"],
        magnitude=magnitude,
        aggregate=aggregate,
        handle=handle,
    )
    _guard_deck_payload(payload)
    return payload


def _guard_deck_payload(payload: dict) -> None:
    """Full deck-payload discipline, fail closed (same rules as the card
    signal-event payload in events.py): no forbidden keys, no ticker-shaped
    tokens, non-negative magnitude, no banned phrase in any string."""
    low_keys = {k.lower() for k in payload}
    for bad in FORBIDDEN_PAYLOAD_KEYS:
        if bad in low_keys:
            raise NotifyError(
                f"deck payload carries forbidden key {bad!r} (sign/symbol/"
                f"direction/P&L) — event refused, not delivered"
            )
    for mag_key in ("magnitude_z", "statistic"):
        if mag_key in payload and payload[mag_key] is not None:
            m = payload[mag_key]
            if isinstance(m, (int, float)) and not isinstance(m, bool) and m < 0:
                raise NotifyError(
                    f"deck payload {mag_key} is negative — a signed statistic "
                    f"would leak direction; refused, not delivered"
                )
    _guard_no_tickers_payload(payload)

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            banned = contains_banned(o) or contains_banned_deck(o)
            if banned is not None:
                raise NotifyError(
                    f"deck payload string contains banned phrase {banned!r} — "
                    f"refused, not delivered"
                )

    walk(payload)


# -- watermarked deck artifact (like the card face) ---------------------------

def build_deck_artifact(
    *,
    deck_id: str,
    bucket: str,
    rank: str | None,
    stage_stats: dict | None = None,
    watermark_handle: str,
    now: float,
    handle: str | None = None,
) -> dict:
    """The shareable deck artifact: watermarked (viewer handle + ts) like
    the card face, carrying the rank label, the deck explainer, and
    aggregate stage stats. `stage_stats` must come from LEDGER STAGE ROWS
    only — never params_json — so no server-side symbol can reach the
    artifact by construction. `_guard_deck_artifact` re-scans the
    serialized artifact (fail closed)."""
    _require_flag(handle)
    rank_label = (
        deck_rank_label(rank, handle=handle) if rank is not None else None
    )
    artifact = {
        "deck_id": deck_id,
        "bucket": bucket,  # pre-registered grid label
        "rank": (str(rank).strip().upper() if rank is not None else None),
        "rank_label": rank_label,
        "explainer": deck_explainer(bucket=bucket, handle=handle),
        "candidates_line": cards_are_candidates_line(handle=handle),
        "stage_stats": dict(stage_stats or {}),  # ledger stage rows only
        "watermark": {
            "handle": watermark_handle,
            "ts": now,
            "note": "deck artifact watermarked handle+timestamp (v4.2 §5)",
        },
        "disclaimer": (
            "Decks are evidence about a tested portfolio, not instructions. "
            "No single card is an instant win. SIFT does not trade; you make "
            "all trading decisions. Survival facts only — not advice; no "
            "picks (v3 §12 + v4.2 §5)."
        ),
    }
    _guard_deck_artifact(artifact)
    return artifact


def _guard_deck_artifact(artifact: dict) -> None:
    """Leak guard on the serialized deck artifact: every string is scanned
    for ticker-shaped tokens (face discipline) and for banned/directional
    phrases (copy discipline). Raises DeckCopyError on any hit."""

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            for tok in _TICKER_RE.findall(o):
                if tok not in _PAYLOAD_WHITELIST:
                    raise DeckCopyError(
                        f"deck artifact contains a ticker-like token {tok!r} — refused"
                    )
            banned = contains_banned(o) or contains_banned_deck(o)
            if banned is not None:
                raise DeckCopyError(
                    f"deck artifact string contains banned phrase {banned!r} — refused"
                )

    walk(artifact)


__all__ = [
    "BANNED_PHRASES_DECK",
    "BANNED_WORDS_DECK",
    "DECK_EVENT_TYPES",
    "DECK_GLOSSARY",
    "DECK_PAYLOAD_SCHEMA_VERSION",
    "DECK_RANKS",
    "DECK_RANK_MAX",
    "DECK_RANK_MIN",
    "DeckCopyError",
    "assert_deck_copy_clean",
    "build_deck_artifact",
    "build_deck_event_payload",
    "cards_are_candidates_line",
    "contains_banned_deck",
    "deck_custom_warning",
    "deck_event_copy",
    "deck_explainer",
    "deck_glossary",
    "deck_rank_label",
    "deck_rank_phrase",
    "deck_telemetry_caveat",
]
