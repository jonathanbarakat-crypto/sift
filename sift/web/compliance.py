"""Fail-closed compliance guards for SIFT beta HTTP API & SSE responses.

Enforces v3 §12, v4.1 §6, and v4.2 §5 hard rules on every serialized
response payload leaving the server:

1. Zero raw market data (no prices, bars, OHLCV, volume, params_json).
2. Zero tickers (ticker-shaped tokens [A-Z]{2,5}, plus case-insensitive
   matching against any active warehouse symbols — fail closed).
3. Zero directional calls or P&L/return/profit/value keys.
4. Copy discipline: EVERY non-identifier string field (including `error`,
   `message`, `description`, `verdict_copy`, `explainer`, `label`, ...) is
   scanned by `assert_copy_clean`; mandated legal-clause fields pass
   `_assert_legal_clause_clean`.
5. Zero email PII outside auth inputs (telemetry and analytics exports
   refuse any email-shaped value or email key).
6. ComplianceError NEVER exposes raw offending tokens in public HTTP/log
   messages (prevents echoing blocked tickers/phrases back to callers).
"""

from __future__ import annotations

import re
from collections.abc import Collection
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

# Fields carrying mandated negated legal clauses (already guard-checked by
# sift/cards/copy.py::_assert_legal_clause_clean or face.py disclaimer).
_LEGAL_CLAUSE_FIELDS = frozenset({
    "statement",
    "tos_acceptance",
    "cards_are_candidates",
    "disclaimer",
    "privacy_note",
})

# Identifier / machine-token fields that are NOT English prose (still checked
# for tickers, forbidden keys, and email PII — only skipped for English
# banned-phrase substring scan so opaque hex/scope/URL tokens don't collide).
_OPAQUE_ID_FIELDS = frozenset({
    "user_id",
    "session_token",
    "run_id",
    "trial_id",
    "card_id",
    "connection_id",
    "candidate_id",
    "scope_id",
    "default_scope",
    "stream_url",
    "poll_url",
    "config_hash",
    "config_commitment",
    "data_source",
    "sector_bucket",
    "bucket",
    "level",
    "content_type",
    "encoding",
    "tos_version",
    "handle",
    "source",
    "service",
    "family",
    "kind",
    "tier",
    "decayed_from",
    "survivor_class",
    "verdict",
    "status",
    "stage",
    "killed_at",
    "ceremony_segment",
    "vault_state",
    "state",
    "band",
    "provenance",
    "removed",
})

# Exact keys allowed to carry base64 PNG strings (already guarded at draw time
# inside sift/cards/render.py). Never skip arbitrary *_b64 keys or nested dicts.
_PNG_B64_KEYS = frozenset({
    "png_2x_b64",
    "png_1x_b64",
})


class ComplianceError(Exception):
    """Fail-closed response or telemetry compliance violation.

    `public_message` is a static, non-leaking message safe for HTTP responses
    and logs (never reflects offending tickers, phrases, or PII).
    """

    public_message = "Response blocked by compliance guard."

    def __init__(self, reason: str, *, rule: str = "compliance_violation"):
        super().__init__(reason)
        self.rule = rule


def assert_no_email_pii(obj: Any, *, path: str = "$") -> None:
    """Reject any email key or email-shaped string inside `obj`."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if "email" in kl:
                raise ComplianceError(
                    f"forbidden PII key at {path}", rule="email_key"
                )
            assert_no_email_pii(v, path=f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            assert_no_email_pii(item, path=f"{path}[{idx}]")
    elif isinstance(obj, str) and EMAIL_RE.search(obj):
        raise ComplianceError(
            f"email-shaped PII detected at {path}", rule="email_value"
        )


def assert_no_tickers(
    text: str,
    *,
    path: str = "$",
    extra_symbols: Collection[str] | None = None,
) -> None:
    """Fail closed if any unwhitelisted ticker-shaped token or known warehouse
    symbol appears in `text`."""
    for tok in _TICKER_RE.findall(text):
        if tok not in _RESPONSE_WHITELIST:
            raise ComplianceError(
                f"ticker-like token detected at {path}", rule="ticker_token"
            )
    if extra_symbols:
        for sym in extra_symbols:
            clean = sym.split(".", 1)[0].strip()
            if not clean:
                continue
            if len(clean) == 1:
                # Single-letter tickers (e.g. F, T, C, V, X): check uppercase
                # standalone token so normal English prose isn't false-flagged.
                if re.search(rf"(?<![A-Za-z0-9_]){re.escape(clean.upper())}(?![A-Za-z0-9_])", text):
                    raise ComplianceError(
                        f"warehouse symbol detected at {path}", rule="warehouse_symbol"
                    )
            else:
                if re.search(rf"\b{re.escape(clean)}\b", text, re.IGNORECASE):
                    raise ComplianceError(
                        f"warehouse symbol detected at {path}", rule="warehouse_symbol"
                    )


def guard_api_payload(
    payload: Any,
    *,
    allow_email: bool = False,
    extra_symbols: Collection[str] | None = None,
    path: str = "$",
) -> Any:
    """Fail-closed compliance guard for every HTTP JSON / SSE payload.

    Checks:
    - No forbidden keys (tickers, raw prices, directions, P&L, secrets).
    - No ticker-shaped tokens or active warehouse symbols in string values
      (skipping only exact string blobs in `_PNG_B64_KEYS`).
    - Strict `assert_copy_clean` on EVERY non-identifier string field
      (including `error`, `message`, `description`, `detail`, `label`, ...),
      and `_assert_legal_clause_clean` on mandated legal clauses.
    - No email PII unless explicitly allowed (default False).
    """

    def _walk(obj: Any, cur_path: str, key_name: str | None = None) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                ks = str(k)
                kl = ks.lower()
                if kl in FORBIDDEN_RESPONSE_KEYS:
                    raise ComplianceError(
                        f"forbidden response key at {cur_path}", rule="forbidden_key"
                    )
                if not allow_email and "email" in kl:
                    raise ComplianceError(
                        f"forbidden email key at {cur_path}", rule="email_key"
                    )
                # Only skip exact PNG base64 string blobs (never nested dicts/lists).
                if ks in _PNG_B64_KEYS and isinstance(v, str):
                    continue
                _walk(v, f"{cur_path}.{ks}", key_name=kl)
        elif isinstance(obj, (list, tuple)):
            for idx, item in enumerate(obj):
                _walk(item, f"{cur_path}[{idx}]", key_name=key_name)
        elif isinstance(obj, str):
            if not allow_email and EMAIL_RE.search(obj):
                raise ComplianceError(
                    f"email-shaped PII detected at {cur_path}", rule="email_value"
                )
            assert_no_tickers(obj, path=cur_path, extra_symbols=extra_symbols)
            if key_name in _LEGAL_CLAUSE_FIELDS:
                if key_name == "disclaimer":
                    if "does not trade" not in obj.lower():
                        raise ComplianceError(
                            f"invalid disclaimer at {cur_path}", rule="disclaimer"
                        )
                elif key_name == "privacy_note":
                    try:
                        card_copy.assert_copy_clean(obj)
                    except card_copy.CopyError as exc:
                        raise ComplianceError(
                            f"banned phrase at {cur_path}", rule="banned_phrase"
                        ) from exc
                else:
                    try:
                        card_copy._assert_legal_clause_clean(obj)
                    except card_copy.CopyError as exc:
                        raise ComplianceError(
                            f"banned phrase in legal clause at {cur_path}",
                            rule="banned_phrase",
                        ) from exc
            elif key_name not in _OPAQUE_ID_FIELDS:
                # Scan EVERY other string field (error, message, description,
                # verdict_copy, explainer, glossary entries, label, note, ...)
                try:
                    card_copy.assert_copy_clean(obj)
                except card_copy.CopyError as exc:
                    raise ComplianceError(
                        f"banned phrase at {cur_path}", rule="banned_phrase"
                    ) from exc

    _walk(payload, path)
    return payload


def sanitize_user_label(
    label: str, *, extra_symbols: Collection[str] | None = None
) -> str:
    """Validate and clean a user-supplied saved-connection label.

    Refuses tickers, warehouse symbols, banned phrases, email strings, or
    excessive length so user text can never smuggle forbidden content.
    """
    cleaned = " ".join(str(label or "").split()).strip()
    if not cleaned:
        raise ComplianceError("connection label must not be empty", rule="empty_label")
    if len(cleaned) > 80:
        raise ComplianceError("connection label too long", rule="label_length")
    assert_no_email_pii(cleaned, path="label")
    assert_no_tickers(cleaned, path="label", extra_symbols=extra_symbols)
    try:
        card_copy.assert_copy_clean(cleaned)
    except card_copy.CopyError as exc:
        raise ComplianceError("banned phrase in label", rule="banned_phrase") from exc
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
