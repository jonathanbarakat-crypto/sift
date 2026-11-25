"""Fail-closed compliance guards for SIFT beta HTTP API & SSE responses.

Enforces v3 §12, v4.1 §6, and v4.2 §5 hard rules on every serialized
response payload leaving the server:

1. Zero raw market data (no prices, bars, OHLCV, volume, params_json).
2. Zero tickers (ticker-shaped tokens [A-Z]{2,5} fail closed; raw symbols
   never leave the server).
3. Zero directional calls or P&L/return/profit/value keys.
4. Copy discipline: non-legal user-facing strings pass `assert_copy_clean`
   from `sift/cards/copy.py`; legal-clause fields pass
   `_assert_legal_clause_clean` (the mandated negated clauses only).
5. Zero email PII outside auth inputs (telemetry and analytics exports
   refuse any email-shaped value or email key).
"""

from __future__ import annotations

import re
from typing import Any

from sift.cards import copy as card_copy
from sift.cards.events import FORBIDDEN_PAYLOAD_KEYS as _EVENT_FORBIDDEN_KEYS

EMAIL_RE = re.compile(r"[^\s@]+@[^\s@]+\.[^\s@]{2,}")
_TICKER_RE = re.compile(r"\b[A-Z]{2,5}(?:\.[A-Z]{1,2})?\b")

# Minimal protocol/domain whitelist matching sift/cards/face.py + events.py.
# Kept intentionally narrow so real US stock tickers (AAPL, MSFT, JPM, ON, IT,
# BE, AN, ...) are NEVER whitelisted.
_RESPONSE_WHITELIST = frozenset({
    "SIFT",
    "JSON",
    "UTC",
    "API",
    "RSS",
    "HMAC",
    "DSR",
    "PBO",
    "FIFO",
    "PNG",
    "SSE",
    "US",
    "NO",
    "ID",
    "TS",
    "PL",
})

# Keys that must NEVER appear in any beta API response payload.
FORBIDDEN_RESPONSE_KEYS = frozenset(
    set(_EVENT_FORBIDDEN_KEYS)
    | {
        "price",
        "prices",
        "ohlcv",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "dollar_volume",
        "bars",
        "raw_bars",
        "params_json",
        "password",
        "password_hash",
        "token_hash",
        "invite_code_hash",
        "secret",
    }
)

# Fields that carry mandated negated legal clauses (already guard-checked by
# sift/cards/copy.py::_assert_legal_clause_clean or face.py disclaimer).
_LEGAL_CLAUSE_FIELDS = frozenset({
    "statement",
    "tos_acceptance",
    "cards_are_candidates",
    "disclaimer",
    "privacy_note",
})

# User-facing prose fields that must pass strict `assert_copy_clean`.
_STRICT_COPY_FIELDS = frozenset({
    "verdict_copy",
    "explainer",
    "hit_loop_hint",
    "phrase",
    "label",
    "note",
})

# Keys holding base64 binary blobs (PNG card images already guarded at draw
# time inside sift/cards/render.py) or opaque hashes/URLs where base64/hex
# character runs are not English/ticker tokens.
_SKIP_TEXT_KEYS = frozenset({
    "png_2x_b64",
    "png_1x_b64",
})


class ComplianceError(Exception):
    """Fail-closed response or telemetry compliance violation."""


def assert_no_email_pii(obj: Any, *, path: str = "$") -> None:
    """Reject any email key or email-shaped string inside `obj`."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if "email" in kl:
                raise ComplianceError(
                    f"forbidden PII key {k!r} at {path}.{k} — refused"
                )
            assert_no_email_pii(v, path=f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            assert_no_email_pii(item, path=f"{path}[{idx}]")
    elif isinstance(obj, str) and EMAIL_RE.search(obj):
        raise ComplianceError(
            f"email-shaped PII string detected at {path} — refused"
        )


def assert_no_tickers(text: str, *, path: str = "$") -> None:
    """Fail closed if any unwhitelisted ticker-shaped token appears in `text`."""
    for tok in _TICKER_RE.findall(text):
        if tok not in _RESPONSE_WHITELIST:
            raise ComplianceError(
                f"ticker-like token {tok!r} detected at {path} — refused"
            )


def guard_api_payload(
    payload: Any,
    *,
    allow_email: bool = False,
    path: str = "$",
) -> Any:
    """Fail-closed compliance guard for every HTTP JSON / SSE payload.

    Checks:
    - No forbidden keys (tickers, raw prices, directions, P&L, secrets).
    - No ticker-shaped tokens in string values (skipping *_b64 image blobs).
    - Strict banned-phrase check on user-facing copy fields, and negated
      legal-clause check on mandated legal fields.
    - No email PII unless explicitly allowed (default False).
    """

    def _walk(obj: Any, cur_path: str, key_name: str | None = None) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                ks = str(k)
                kl = ks.lower()
                if kl in FORBIDDEN_RESPONSE_KEYS:
                    raise ComplianceError(
                        f"forbidden response key {ks!r} at {cur_path}.{ks} — refused"
                    )
                if not allow_email and "email" in kl:
                    raise ComplianceError(
                        f"forbidden email key {ks!r} at {cur_path}.{ks} — refused"
                    )
                if ks in _SKIP_TEXT_KEYS or ks.endswith("_b64"):
                    continue
                _walk(v, f"{cur_path}.{ks}", key_name=kl)
        elif isinstance(obj, (list, tuple)):
            for idx, item in enumerate(obj):
                _walk(item, f"{cur_path}[{idx}]", key_name=key_name)
        elif isinstance(obj, str):
            if not allow_email and EMAIL_RE.search(obj):
                raise ComplianceError(
                    f"email-shaped PII detected at {cur_path} — refused"
                )
            assert_no_tickers(obj, path=cur_path)
            if key_name in _STRICT_COPY_FIELDS:
                try:
                    card_copy.assert_copy_clean(obj)
                except card_copy.CopyError as exc:
                    raise ComplianceError(
                        f"banned phrase at {cur_path}: {exc}"
                    ) from exc
            elif key_name in _LEGAL_CLAUSE_FIELDS:
                # disclaimer from face.py contains negated "no P&L, no picks"
                if key_name == "disclaimer":
                    low = obj.lower()
                    if "does not trade" not in low:
                        raise ComplianceError(
                            f"invalid disclaimer at {cur_path} — refused"
                        )
                elif key_name == "privacy_note":
                    try:
                        card_copy.assert_copy_clean(obj)
                    except card_copy.CopyError as exc:
                        raise ComplianceError(
                            f"banned phrase at {cur_path}: {exc}"
                        ) from exc
                else:
                    try:
                        card_copy._assert_legal_clause_clean(obj)
                    except card_copy.CopyError as exc:
                        raise ComplianceError(
                            f"banned phrase in legal clause at {cur_path}: {exc}"
                        ) from exc

    _walk(payload, path)
    return payload


def sanitize_user_label(label: str) -> str:
    """Validate and clean a user-supplied saved-connection label.

    Refuses tickers, banned phrases, email strings, or excessive length so
    user text can never smuggle forbidden content into rendered views.
    """
    cleaned = " ".join(str(label or "").split()).strip()
    if not cleaned:
        raise ComplianceError("connection label must not be empty")
    if len(cleaned) > 80:
        raise ComplianceError("connection label must be <= 80 characters")
    assert_no_email_pii(cleaned, path="label")
    assert_no_tickers(cleaned, path="label")
    try:
        card_copy.assert_copy_clean(cleaned)
    except card_copy.CopyError as exc:
        raise ComplianceError(f"label refused: {exc}") from exc
    return cleaned


__all__ = [
    "EMAIL_RE",
    "FORBIDDEN_RESPONSE_KEYS",
    "ComplianceError",
    "assert_no_email_pii",
    "assert_no_tickers",
    "guard_api_payload",
    "sanitize_user_label",
]
