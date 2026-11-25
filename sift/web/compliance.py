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
        "gamma",
        "gamma_hat",
        "gamma_true",
        "phi_leader",
        "phi_follower",
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
    "mining_mode",
    "subspace_filter",
    "anchor_id_hmac",
    "anchor_role",
    "anchor_sector",
    "data_vintage_hash",
    "id",
    "asof",
    "mode",
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


# Fields used by mining mode / subspace transparency (still checked for
# tickers, forbidden keys, and email PII — never bypasses ticker guard).
_PRIVATE_EXTRA_FORBIDDEN_KEYS = frozenset({
    "slope",
    "beta",
    "beta_hat",
    "beta_levels",
    "beta_true",
    "hedge_ratio",
    "z_signed",
    "signed_z",
    "spread_series",
    "residuals",
    "series",
    "prices_x",
    "prices_y",
})

PRIVATE_FORBIDDEN_KEYS = frozenset(FORBIDDEN_RESPONSE_KEYS | _PRIVATE_EXTRA_FORBIDDEN_KEYS)

_VALID_INSTRUMENT_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,11}$")
_INSTRUMENT_SYMBOL_PATH_RE = re.compile(r"^\$\.private_spec\.instruments\[\d+\]\.symbol$")
_MAGNITUDE_KEYS = frozenset({"magnitude_z", "current_abs_z"})
_PRIVATE_ALLOWED_ROLES = frozenset({"leg", "leader", "follower", "anchor", "asset"})


def guard_owner_private_payload(
    payload: Any,
    *,
    allowed_symbols: Collection[str],
    extra_symbols: Collection[str] | None = None,
    max_instruments: int = 10,
    path: str = "$",
) -> Any:
    """Fail-closed compliance guard strictly for authenticated owner-only
    Private Spec responses (`/api/private/.../spec`, v4.2 §5.1).

    Preserves every red line except allowing the exact symbols of ONE owned
    trial/card inside `$.private_spec.instruments[i].symbol`:
    - Zero raw prices, OHLCV, volume, bars, or raw `params_json`.
    - Zero vendor-fitted slope / beta / hedge_ratio / spread_series.
    - Zero sign / direction / buy / sell / long / short / P&L / returns.
    - Zero negative `magnitude_z` / `current_abs_z` (magnitude only).
    - Zero email PII.
    - Zero banned phrases across every non-identifier string field.
    - Zero tickers or warehouse symbols anywhere OUTSIDE
      `$.private_spec.instruments[i].symbol`; inside that exact slot, every
      symbol MUST belong to `allowed_symbols` and `len(instruments) <= max_instruments`.
    """
    import math as _math

    allowed_set = frozenset(
        str(s).split(".", 1)[0].strip().upper()
        for s in allowed_symbols
        if str(s).strip()
    )

    if isinstance(payload, dict) and "private_spec" in payload:
        ps = payload["private_spec"]
        if not isinstance(ps, dict):
            raise ComplianceError("invalid private_spec container", rule="private_spec_shape")
        instruments = ps.get("instruments")
        if not isinstance(instruments, list) or len(instruments) > int(max_instruments):
            raise ComplianceError(
                "invalid or oversized private_spec.instruments",
                rule="private_instruments_limit",
            )
        wm = ps.get("watermark")
        if not isinstance(wm, dict) or wm.get("non_shareable") is not True:
            raise ComplianceError(
                "private_spec must carry watermark.non_shareable=True",
                rule="private_watermark",
            )
        disc = ps.get("disclaimer")
        if not isinstance(disc, str) or "does not trade" not in disc.lower() or "private" not in disc.lower():
            raise ComplianceError(
                "private_spec missing private non-trading disclaimer",
                rule="private_disclaimer",
            )

    def _walk(obj: Any, cur_path: str, key_name: str | None = None) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                ks = str(k)
                kl = ks.lower()
                child_path = f"{cur_path}.{ks}"
                if kl == "symbol" and _INSTRUMENT_SYMBOL_PATH_RE.match(child_path):
                    if not isinstance(v, str):
                        raise ComplianceError(
                            f"non-string instrument symbol at {child_path}",
                            rule="private_symbol_type",
                        )
                    norm = v.strip().upper()
                    if not _VALID_INSTRUMENT_SYMBOL_RE.match(norm) or norm not in allowed_set:
                        raise ComplianceError(
                            f"unauthorized instrument symbol at {child_path}",
                            rule="unauthorized_symbol",
                        )
                    continue
                if kl in PRIVATE_FORBIDDEN_KEYS:
                    raise ComplianceError(
                        f"forbidden private response key at {cur_path}",
                        rule="forbidden_key",
                    )
                if "email" in kl:
                    raise ComplianceError(
                        f"forbidden email key at {cur_path}", rule="email_key"
                    )
                if kl in _MAGNITUDE_KEYS and v is not None:
                    if (
                        isinstance(v, bool)
                        or not isinstance(v, (int, float))
                        or not _math.isfinite(float(v))
                        or float(v) < 0.0
                    ):
                        raise ComplianceError(
                            f"invalid or signed magnitude at {child_path}",
                            rule="signed_magnitude",
                        )
                if kl == "role" and cur_path.startswith("$.private_spec.instruments["):
                    if not isinstance(v, str) or v not in _PRIVATE_ALLOWED_ROLES:
                        raise ComplianceError(
                            f"invalid instrument role at {child_path}",
                            rule="invalid_role",
                        )
                    continue
                _walk(v, child_path, key_name=kl)
        elif isinstance(obj, (list, tuple)):
            for idx, item in enumerate(obj):
                _walk(item, f"{cur_path}[{idx}]", key_name=key_name)
        elif isinstance(obj, str):
            if EMAIL_RE.search(obj):
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
                    try:
                        card_copy._assert_legal_clause_clean(obj)
                    except card_copy.CopyError as exc:
                        raise ComplianceError(
                            f"banned phrase in disclaimer at {cur_path}",
                            rule="banned_phrase",
                        ) from exc
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
    "PRIVATE_FORBIDDEN_KEYS",
    "ComplianceError",
    "assert_no_email_pii",
    "assert_no_tickers",
    "guard_api_payload",
    "guard_owner_private_payload",
    "sanitize_user_label",
]
