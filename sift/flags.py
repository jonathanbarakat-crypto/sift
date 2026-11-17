"""Feature flags for SIFT (v4.1 card layer rollout).

`cards_v4_1` wraps the entire v4.1 cards/vault/engagement layer
(sift/cards/*, sift/flags usage in the gauntlet runner and engine):

  1. card minting on gauntlet survivors
  2. per-user vault (keep/discard)
  3. ghost cards (numbered FIFO per canonical spec)
  4. server-rendered anonymized card faces
  5. anchored run mode
  6. share-for-capacity
  7. badges
  8. per-card opt-in hit notifications (digest default on free, live on
     paid, rate-capped) + registry decay re-evaluation (downgrade-only
     amber transitions, publicly explained)

Default ON for the beta cohort. The flag is read server-side only; it is
never exposed to clients as a toggle (v3 §12: no client-side feature
surface that could leak raw data or weaken validation).

Kill-switch (documented per v4.1 mechanics item 8): set the environment
variable `SIFT_CARDS_V4_1=0` (or call `set_flag("cards_v4_1", False)` /
`disable_for_user("cards_v4_1", handle)`) and every cards_v4_1 code path
is inert: the runner mints no cards, the vault API refuses with
`flag_off`, and already-minted cards remain readable (they are append-only
registry records) but no new engagement (share-for-capacity, badges) is
granted. Disabling the flag never destroys data (v4.1 mechanics item 2:
never destroyed, never randomized).
"""

from __future__ import annotations

import os
import threading
import time

FLAG_CARDS_V4_1 = "cards_v4_1"

_KNOWN_FLAGS = (FLAG_CARDS_V4_1,)

# Beta cohort: ON by default. Override globally with SIFT_CARDS_V4_1=0.
_DEFAULTS = {
    FLAG_CARDS_V4_1: os.environ.get("SIFT_CARDS_V4_1", "1") not in ("0", "false", "False", "off", "OFF"),
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


def reset() -> None:
    """Test helper: restore defaults and clear overrides."""
    with _lock:
        _overrides.clear()
        _user_overrides.clear()
        _DEFAULTS[FLAG_CARDS_V4_1] = (
            os.environ.get("SIFT_CARDS_V4_1", "1") not in ("0", "false", "False", "off", "OFF")
        )


__all__ = [
    "FLAG_CARDS_V4_1",
    "cards_v4_1_enabled",
    "disable_for_user",
    "enable_for_user",
    "is_enabled",
    "known_flags",
    "reset",
    "set_flag",
]
