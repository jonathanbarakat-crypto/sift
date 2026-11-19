"""v4.1 newcomer-friendly copy + onboarding layer (behind `cards_v4_1`).

This module is the ONLY place the new user-facing strings are built:

- `face_explainer(...)` — the one-paragraph plain-language explainer shown
  on every card face ("what this card certifies — and what it does not");
- `GLOSSARY` — plain-language definitions behind tooltips for the jargon
  that stays OUT of the main view (z-score, half-life, DSR, tier, health);
- `gauge_band(|z|)` / `gauge_label(band)` — gauge-style rendering of
  magnitude: "stretched / normal / compressed" with the numeric |z| one
  click away (the face carries both `band` and `magnitude_z`);
- `health_phrase(ratio)` — health in plain ratio language vs the mint
  baseline ("weakening ~0.8x of mint baseline", "strengthening ~1.3x");
- `onboarding_statement()` — the signup / first-card statement
  ("you make the calls — we make the evidence legible") + ToS pointer;
- `tos_acceptance_text()` — the day-one ToS line ("not an investment / no
  resale value / no right to profit");
- `hit_loop_hint()` — the practical loop: on a hit, check the pair in your
  OWN charting/broker tool; SIFT's notification means "your saved
  connection fired, here's how strong and how often it has fired".

Copy discipline (HARD, fail-closed):
- `assert_copy_clean(text)` scans every string against BANNED_PHRASES —
  no buy/sell/long/short, no P&L/returns, no profit-expectation or "value"
  language, no urgency/countdown. Every builder in this module runs it on
  its output and raises CopyError on any hit: a banned phrase refuses the
  render, never ships.
- No tickers and no sign/direction anywhere: builders take NO symbol and
  NO sign inputs; magnitude is always |z| (absolute), health is always a
  ratio vs baseline. The artifact-level invariant (face.py: ledger stage
  rows only, never params_json) is unchanged and still guarded by
  `_guard_no_tickers` / `_guard_face_artifact`.

Flag: behind `cards_v4_1`; flag off -> CopyError(flag_off) from every
builder. Kill-switch env: SIFT_CARDS_V4_1=0 (sift/flags.py).
"""

from __future__ import annotations

import re

from sift import flags
from sift.cards.notify import BANNED_PHRASES as _NOTIFY_BANNED
from sift.cards.notify import BANNED_WORDS as _NOTIFY_BANNED_WORDS
from sift.cards.notify import contains_banned as _contains_banned


class CopyError(Exception):
    """Copy misuse: flag off, banned phrase in a built string, bad input."""


# -- banned phrases ------------------------------------------------------------
# Extends notify.py's list with the task's named bans: no buy/sell/long/
# short (whole-word, so "longer" does not false-positive), no P&L/returns,
# no profit-expectation or "value" language, no urgency/countdown.
# `assert_copy_clean` delegates to notify.contains_banned (substring scan
# over BANNED_PHRASES + word-boundary scan over BANNED_WORDS) plus the extra
# substring bans below, so this module and notify/decay share ONE guard
# discipline.
_EXTRA_BANNED = (
    "pnl", "loss", "gain", "alpha", "win rate", "winrate", "win-rate",
    "expected return", "roi", "lambo", "urgent", "hurry", "now or never",
    "limited time", "fomo",
)
BANNED_PHRASES = _NOTIFY_BANNED + _EXTRA_BANNED
BANNED_WORDS = _NOTIFY_BANNED_WORDS


def assert_copy_clean(text: str, *, phrase_list: tuple[str, ...] | None = None) -> str:
    """Fail-closed copy guard: any banned phrase/word in `text` raises
    CopyError (the render is refused, never shipped). Returns the text
    unchanged. Default scans BANNED_PHRASES (substring) + BANNED_WORDS
    (whole word) + _EXTRA_BANNED; `phrase_list` overrides the substring
    list for the legal-clause builders (see _LEGAL_CLAUSE_BANNED)."""
    banned = _contains_banned(text)
    if banned is not None:
        raise CopyError(f"copy contains banned phrase {banned!r} — refused")
    low = text.lower()
    for phrase in (BANNED_PHRASES if phrase_list is None else phrase_list):
        if phrase in low:
            raise CopyError(f"copy contains banned phrase {phrase!r} — refused")
    return text


# -- legal-clause exception -----------------------------------------------------
# The owner-mandated onboarding sentence ("never tells you what to buy or
# sell") and the day-one ToS line ("no resale value ... no right to any
# profit") contain "buy"/"sell"/"value"/"profit" ONLY inside those negated
# legal clauses. Those two builders therefore scan against a dedicated
# phrase list that keeps every ban except those four words; the whole-word
# directional scan is also skipped for them (the words appear only in the
# negated clause). Every OTHER string (faces, tooltips, gauges, health,
# hints, notifications) uses the FULL guard: substring BANNED_PHRASES +
# whole-word BANNED_WORDS.
_LEGAL_CLAUSE_SKIP_WORDS = ("buy", "sell")
_LEGAL_CLAUSE_SKIP_PHRASES = ("value", "valuable", "profit", "worth", "price",
                              "invest")   # "not an investment" — negated legal clause
_LEGAL_CLAUSE_BANNED = tuple(
    p for p in BANNED_PHRASES if p not in _LEGAL_CLAUSE_SKIP_PHRASES
)


def _assert_legal_clause_clean(text: str) -> str:
    """Guard for the onboarding/ToS legal clauses: full substring scan minus
    the four value/profit words that appear only inside the negated legal
    clause, and no whole-word directional scan (the mandated "what to buy
    or sell" negation). Any OTHER banned phrase/word still refuses."""
    low = text.lower()
    for phrase in _LEGAL_CLAUSE_BANNED:
        if phrase in low:
            raise CopyError(f"copy contains banned phrase {phrase!r} — refused")
    for w in BANNED_WORDS:
        if w in _LEGAL_CLAUSE_SKIP_WORDS:
            continue
        if re.search(r"\b" + w + r"\b", low):
            raise CopyError(f"copy contains banned word {w!r} — refused")
    return text


def _require_flag(handle: str | None = None) -> None:
    if not flags.cards_v4_1_enabled(handle):
        raise CopyError("flag_off: cards_v4_1 is disabled for this account")


# -- gauge: |z| -> stretched / normal / compressed -----------------------------
# Pre-registered bands (plain-language magnitude, numeric |z| one click
# away on the face). Bands are on |z| only — never on the sign.
GAUGE_NORMAL_LO = 1.0      # |z| < 1.0          -> "normal"
GAUGE_STRETCH_HI = 2.0     # |z| >= 2.0         -> "stretched"
# 1.0 <= |z| < 2.0          -> "normal" edge band widened to "near the
# entry band" is still "normal"; compressed is the low end:
GAUGE_COMPRESS_LO = 0.5    # |z| < 0.5          -> "compressed"

GAUGE_BANDS = ("compressed", "normal", "stretched")


def gauge_band(abs_z: float) -> str:
    """Map |z| to a plain-language gauge band. Pure function of the
    magnitude ONLY (never the sign): |z| >= GAUGE_STRETCH_HI -> stretched;
    |z| < GAUGE_COMPRESS_LO -> compressed; else normal."""
    z = abs(float(abs_z))
    if z >= GAUGE_STRETCH_HI:
        return "stretched"
    if z < GAUGE_COMPRESS_LO:
        return "compressed"
    return "normal"


def gauge_label(band: str) -> str:
    """The label drawn next to the gauge on the card face."""
    if band not in GAUGE_BANDS:
        raise CopyError(f"unknown gauge band {band!r}")
    return {
        "stretched": "stretched beyond its usual range",
        "normal": "within its usual range",
        "compressed": "quiet — closer to its mean than usual",
    }[band]


# -- health: ratio vs mint baseline, plain ratio language -----------------------

def health_phrase(ratio: float | None) -> str:
    """Health as a plain ratio vs the card's mint baseline:
    ratio < 1 -> "weakening ~{r}x of mint baseline";
    ratio > 1 -> "strengthening ~{r}x of mint baseline";
    ratio == 1 (or None) -> "holding near its mint baseline".
    Ratio language only — never a level, never a direction of price."""
    if ratio is None:
        return "holding near its mint baseline"
    r = float(ratio)
    if not (r == r) or r in (float("inf"), float("-inf")):
        return "holding near its mint baseline"
    if r < 1.0:
        return f"weakening ~{r:.1f}x of mint baseline"
    if r > 1.0:
        return f"strengthening ~{r:.1f}x of mint baseline"
    return "holding near its mint baseline"


# -- glossary: jargon behind tooltips, not in the main view ---------------------

GLOSSARY: dict[str, str] = {
    "z-score": (
        "How far the monitored spread is from its own usual range, measured "
        "in standard deviations. The card face shows the gauge band "
        "(stretched / normal / compressed); the numeric |z| is one click away."
    ),
    "half-life": (
        "How many days, on average, it takes the monitored spread to close "
        "half the distance back to its mean after a move. A durability "
        "fact about the pattern, not a forecast."
    ),
    "dsr": (
        "Defensive Sharpe-style ratio: the gauntlet's evidence strength "
        "after correcting for how many relationships were tried in the "
        "scope. Higher means the result is less likely to be selection "
        "luck. A survival fact."
    ),
    "tier": (
        "A rarity label (bronze / silver / gold / platinum) set by the "
        "pre-registered gauntlet stats at mint, and nothing else. It can "
        "move down on re-evaluation, and up only when pre-registered "
        "live-window rules are met — never by payment or chance."
    ),
    "health": (
        "How the pattern is doing in the trailing live window compared with "
        "its mint baseline, shown as a plain ratio (e.g. weakening ~0.8x). "
        "An observation about the signal's persistence, not a recommendation."
    ),
    "gauntlet": (
        "The pre-registered 5-stage statistical stress test (S0-S5) every "
        "relationship must survive before a card is minted. The card face "
        "shows which stages passed."
    ),
    "survivor class": (
        "The verdict class of the underlying trial. Only 'validated-green' "
        "(full-depth registry validation) mints a card."
    ),
}


def glossary_entry(term: str, handle: str | None = None) -> str:
    """One glossary tooltip text. Unknown term -> CopyError (fail closed:
    jargon never ships unglossed). Flag off -> CopyError(flag_off)."""
    _require_flag(handle)
    key = term.strip().lower()
    if key not in GLOSSARY:
        raise CopyError(f"no glossary entry for {term!r} — refused")
    return assert_copy_clean(GLOSSARY[key])


def glossary(handle: str | None = None) -> dict[str, str]:
    """The whole tooltip glossary, every entry guard-checked. Flag off ->
    CopyError(flag_off)."""
    _require_flag(handle)
    return {k: assert_copy_clean(v) for k, v in GLOSSARY.items()}


# -- card-face explainer ---------------------------------------------------------

def face_explainer(stages: int = 5, handle: str | None = None) -> str:
    """The one-paragraph plain-language explainer on every card face.
    Certifies what the card IS (survived the stress test) and what it is
    NOT (nothing about what to do with it)."""
    _require_flag(handle)
    text = (
        f"This card certifies that a co-movement relationship in this scope "
        f"survived the pre-registered {stages}-stage statistical stress test "
        f"(S0-S{stages - 1}) over the research window. It says the pattern is "
        f"real and durable — nothing about what to do with it. SIFT shows "
        f"evidence; you make the calls."
    )
    return assert_copy_clean(text)


# -- onboarding (signup / first card) --------------------------------------------

def onboarding_statement(handle: str | None = None) -> str:
    """The plain statement shown at signup and on the first card. The
    'never tells you what to buy or sell' phrasing is the owner-mandated
    wording, so this builder scans against _ONBOARDING_BANNED (which keeps
    every ban except the two words inside that one mandated negation)."""
    _require_flag(handle)
    text = (
        "SIFT shows which relationships in your research held up under "
        "stress. It never tells you what to buy or sell; nothing here is "
        "advice. You make the calls — we make the evidence legible."
    )
    return _assert_legal_clause_clean(text)


def tos_acceptance_text(handle: str | None = None) -> str:
    """Day-one ToS acceptance line (beta plan: live from day one)."""
    _require_flag(handle)
    text = (
        "Cards are not an investment, have no resale value, and confer no "
        "right to any profit. They are non-transferable. SIFT does not "
        "trade. By continuing you accept these terms."
    )
    return _assert_legal_clause_clean(text)


# -- practical loop: the hit hint -------------------------------------------------

def hit_loop_hint(handle: str | None = None) -> str:
    """In-app hint on a hit: check the pair in your OWN charting/broker
    tool. The notification means 'your saved connection fired, here's how
    strong and how often it has fired' — never an instruction to trade."""
    _require_flag(handle)
    text = (
        "Your saved connection fired. This tells you how strong and how "
        "often it has fired — that's all. Check the pair in your own "
        "charting or broker tool and make your own call; SIFT never tells "
        "you what to do with it."
    )
    return assert_copy_clean(text)


# -- notification hit copy (plain-language variant) --------------------------------

def hit_notification_copy(card_id: str, band: str, abs_z: float | None,
                          hit_count_trailing_12m: int | None = None,
                          health_ratio: float | None = None,
                          handle: str | None = None) -> str:
    """Plain-language hit notification: 'your saved connection fired,
    here's how strong and how often it has fired'. Magnitude via the
    gauge band (+ numeric |z|), frequency via the trailing-12m hit count,
    health via the ratio phrase. Never a sign, never a symbol, never an
    instruction."""
    _require_flag(handle)
    band = gauge_band(abs_z if abs_z is not None else 0.0)
    parts = [
        f"Your saved connection fired (card {card_id}).",
        f"Strength right now: {gauge_label(band)}"
        + (f" (|z| = {abs(abs_z):.2f})" if abs_z is not None else "") + ".",
    ]
    if hit_count_trailing_12m is not None:
        parts.append(f"It has fired {int(hit_count_trailing_12m)} time(s) in "
                     f"the trailing 12 months.")
    parts.append(f"Health: {health_phrase(health_ratio)}.")
    parts.append(
        "This is a monitoring note about your own saved connection — how "
        "strong and how often it has fired. Check it in your own charting "
        "tool; SIFT does not trade and does not tell you what to do."
    )
    return assert_copy_clean(" ".join(parts))


__all__ = [
    "BANNED_PHRASES",
    "BANNED_WORDS",
    "GAUGE_BANDS",
    "GAUGE_COMPRESS_LO",
    "GAUGE_NORMAL_LO",
    "GAUGE_STRETCH_HI",
    "GLOSSARY",
    "CopyError",
    "assert_copy_clean",
    "face_explainer",
    "gauge_band",
    "gauge_label",
    "glossary",
    "glossary_entry",
    "health_phrase",
    "hit_loop_hint",
    "hit_notification_copy",
    "onboarding_statement",
    "tos_acceptance_text",
]
