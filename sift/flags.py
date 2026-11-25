"""Feature flags for SIFT (v4.1 card layer rollout).

`cards_v4_1` wraps the entire v4.1 cards/vault/engagement layer
(sift/cards/*, sift/flags usage in the gauntlet runner and engine):

  1. card minting on gauntlet survivors
  2. per-user vault (keep/discard)
  3. ghost cards (numbered FIFO per canonical spec)
  4. server-rendered anonymized card faces
  5. anchored run mode
  6. share-for-capacity
  8. per-card opt-in hit notifications (digest default on free, live on
     paid, rate-capped) + registry decay re-evaluation (amber downgrades
     any time + pre-registered live-window upgrades, both append-only
     ledger events, publicly explained)

Default ON for the beta cohort. The flag is read server-side only; it is
never exposed to clients as a toggle (v3 §12: no client-side feature
surface that could leak raw data or weaken validation).

Kill-switch (documented per v4.1 mechanics item 8): set the environment
variable `SIFT_CARDS_V4_1=0` (or call `set_flag("cards_v4_1", False)` /
`disable_for_user("cards_v4_1", handle)`) and every cards_v4_1 code path
is inert: the runner mints no cards, the vault API refuses with
`flag_off`, and already-minted cards remain readable (they are append-only
registry records) but no new engagement (share-for-capacity) is granted. Disabling the flag never destroys data (v4.1 mechanics item 2:
never destroyed, never randomized).

`decks_v4_2` wraps the v4.2 deck layer (sift/decks/*: bucket registry,
S6 portfolio gauntlet, F-S deck ranking, deck ledger) — see spec v4.2 §4
(hermes-tasks #166). The deck layer sits ON TOP of the card layer: it is
effective only when BOTH flags are on (`decks_v4_2_enabled` ANDs the two),
so the cards_v4_1 kill-switch also disables decks. Separate kill-switch:
`SIFT_DECKS_V4_2=0` (or `set_flag("decks_v4_2", False)`) disables only the
deck layer. Default ON for the beta cohort, same posture as cards_v4_1.
Fail-closed: unknown flags read False (see is_enabled).
"""

from __future__ import annotations

import os
import threading
import time

FLAG_CARDS_V4_1 = "cards_v4_1"
FLAG_DECKS_V4_2 = "decks_v4_2"
FLAG_OWNER_PRIVATE_SPEC = "owner_private_spec"

_KNOWN_FLAGS = (FLAG_CARDS_V4_1, FLAG_DECKS_V4_2, FLAG_OWNER_PRIVATE_SPEC)

# Beta cohort: ON by default. Override globally with SIFT_CARDS_V4_1=0 /
# SIFT_DECKS_V4_2=0 / SIFT_OWNER_PRIVATE_SPEC=0.
_DEFAULTS = {
    FLAG_CARDS_V4_1: os.environ.get("SIFT_CARDS_V4_1", "1") not in ("0", "false", "False", "off", "OFF"),
    FLAG_DECKS_V4_2: os.environ.get("SIFT_DECKS_V4_2", "1") not in ("0", "false", "False", "off", "OFF"),
    FLAG_OWNER_PRIVATE_SPEC: os.environ.get("SIFT_OWNER_PRIVATE_SPEC", "1") not in ("0", "false", "False", "off", "OFF"),
}

_lock = threading.Lock()
_overrides: dict[str, bool] = {}
_user_overrides: dict[tuple[str, str], tuple[bool, float]] = {}  # (flag, handle) -> (value, set_at)


def known_flags() -> tuple[str, ...]:
    return _KNOWN_FLAGS


def is_enabled(flag: str, handle: str | None = None) -> bool:
    """Server-side flag read. `handle` (the account handle) may be None for
    global/operator paths. Per-user overrides beat global overrides beat the
    default. Unknown flag -> False (fail closed)."""
    if flag not in _KNOWN_FLAGS:
        return False
    with _lock:
        if handle is not None:
            ov = _user_overrides.get((flag, handle))
            if ov is not None:
                return ov[0]
        if flag in _overrides:
            return _overrides[flag]
        return _DEFAULTS[flag]


def set_flag(flag: str, value: bool) -> None:
    """Global override (kill-switch). Raises on unknown flag."""
    if flag not in _KNOWN_FLAGS:
        raise ValueError(f"unknown flag {flag!r}")
    with _lock:
        _overrides[flag] = bool(value)


def disable_for_user(flag: str, handle: str) -> None:
    """Per-user kill-switch (e.g. a user under fraud review)."""
    if flag not in _KNOWN_FLAGS:
        raise ValueError(f"unknown flag {flag!r}")
    with _lock:
        _user_overrides[(flag, handle)] = (False, time.time())


def enable_for_user(flag: str, handle: str) -> None:
    if flag not in _KNOWN_FLAGS:
        raise ValueError(f"unknown flag {flag!r}")
    with _lock:
        _user_overrides[(flag, handle)] = (True, time.time())


def cards_v4_1_enabled(handle: str | None = None) -> bool:
    """The single read used by every v4.1 code path."""
    return is_enabled(FLAG_CARDS_V4_1, handle)


def decks_v4_2_enabled(handle: str | None = None) -> bool:
    """The single read used by every v4.2 deck code path. The deck layer
    sits on top of the card layer: it is effective only when BOTH
    `decks_v4_2` and `cards_v4_1` are on (the cards_v4_1 kill-switch
    therefore also disables decks); `SIFT_DECKS_V4_2=0` disables only
    the deck layer. Fail-closed on unknown flag (is_enabled)."""
    return is_enabled(FLAG_DECKS_V4_2, handle) and is_enabled(FLAG_CARDS_V4_1, handle)


def owner_private_spec_enabled(handle: str | None = None) -> bool:
    """The single read gating `/api/private/.../spec` endpoints (v4.2 §5.1).

    Kill-switch: `SIFT_OWNER_PRIVATE_SPEC=0` (or `set_flag("owner_private_spec", False)`).
    Always fail-closed when off; never weakens public/shared guards.
    """
    return is_enabled(FLAG_OWNER_PRIVATE_SPEC, handle)


def reset() -> None:
    """Test helper: restore defaults and clear overrides."""
    with _lock:
        _overrides.clear()
        _user_overrides.clear()
        _DEFAULTS[FLAG_CARDS_V4_1] = (
            os.environ.get("SIFT_CARDS_V4_1", "1") not in ("0", "false", "False", "off", "OFF")
        )
        _DEFAULTS[FLAG_DECKS_V4_2] = (
            os.environ.get("SIFT_DECKS_V4_2", "1") not in ("0", "false", "False", "off", "OFF")
        )
        _DEFAULTS[FLAG_OWNER_PRIVATE_SPEC] = (
            os.environ.get("SIFT_OWNER_PRIVATE_SPEC", "1")
            not in ("0", "false", "False", "off", "OFF")
        )


__all__ = [
    "FLAG_CARDS_V4_1",
    "FLAG_DECKS_V4_2",
    "FLAG_OWNER_PRIVATE_SPEC",
    "cards_v4_1_enabled",
    "decks_v4_2_enabled",
    "disable_for_user",
    "enable_for_user",
    "is_enabled",
    "known_flags",
    "owner_private_spec_enabled",
    "reset",
    "set_flag",
]
