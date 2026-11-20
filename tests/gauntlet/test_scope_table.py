"""Regression tests for the gauntlet scope table (sift issue #58).

The engine's registered default production scope id is
``first_run.SCOPE_ID == "free_1y_top1k"`` (EODHD, ``SOURCES["eodhd"]
["default_scope"]``), but ``gauntlet.config.SCOPE_T`` historically lacked
that key, so ``cfg.scope_T("free_1y_top1k")`` raised a bare KeyError and
S6 ``deck_gauntlet`` (s6.py) crashed on any production-scope deck run.

These tests pin the durable minimum from issue #58:
  * ``scope_T()`` succeeds for EVERY scope id registered in
    ``first_run.SOURCES[*]["default_scope"]`` and ``first_run.SCOPE_ID``;
  * ``scope_T()`` fails CLOSED with a clear error naming the missing id
    (never a bare KeyError) for unknown production scope ids;
  * ``free_1y_top1k`` carries the honest trailing-1-year bar count
    (BARS_PER_YEAR = 252), matching the engine's default window/holdout.
"""
from __future__ import annotations

import pytest

from sift.engine import first_run as fr
from sift.gauntlet import config as cfg


def _registered_production_scope_ids() -> list[str]:
    """Every scope id the engine can register a production trial under."""
    ids = {fr.SCOPE_ID}
    for spec in fr.SOURCES.values():
        sid = spec.get("default_scope")
        if sid is not None:
            ids.add(sid)
    return sorted(ids)


def test_scope_T_covers_every_registered_production_scope_id():
    """scope_T() must succeed for every registered production scope id
    (first_run.SOURCES[*]["default_scope"] and first_run.SCOPE_ID)."""
    ids = _registered_production_scope_ids()
    assert "free_1y_top1k" in ids, "engine default production scope id missing from test fixture"
    for sid in ids:
        T = cfg.scope_T(sid)                      # must NOT raise
        assert isinstance(T, int) and T > 0, f"scope_T({sid!r}) = {T!r}, expected positive int"


def test_scope_T_free_1y_top1k_is_trailing_one_year():
    """free_1y_top1k is the free-tier trailing-1y window: one trading year
    of daily bars (BARS_PER_YEAR = 252) — the engine's default window is
    '1y' with a 252-bar holdout (first_run.py), NOT the paid full-history
    window (that stays mapped to pro_full / registry_depth at 4032)."""
    assert cfg.scope_T("free_1y_top1k") == cfg.BARS_PER_YEAR == 252
    # the paid full-history window keeps its own honest entries
    assert cfg.scope_T("pro_full") == 4032
    assert cfg.scope_T("registry_depth") == 4032


def test_scope_T_unknown_id_fails_closed_naming_the_id():
    """An unknown NON-research scope id must fail closed with a clear error
    that names the missing id and lists the known ids — never a bare
    KeyError (the original bug: KeyError: 'free_1y_top1k')."""
    with pytest.raises(ValueError) as excinfo:
        cfg.scope_T("free_99y_top1k")
    msg = str(excinfo.value)
    assert "free_99y_top1k" in msg, f"error must name the missing id, got: {msg}"
    for known in sorted(cfg.SCOPE_T):
        assert known in msg, f"error must list known scope ids, missing {known!r}: {msg}"
    # and it must NOT be a KeyError
    assert not isinstance(excinfo.value, KeyError)


def test_scope_T_research_fallback_unchanged():
    """research_* ids keep the 5y fallback (research_5y_top500 -> 1260)."""
    assert cfg.scope_T("research_5y_top1k") == 1260
    assert cfg.scope_T("research_5y_top500") == 1260


def test_scope_T_existing_entries_unchanged():
    """The pre-existing scope-table entries are not altered by the fix."""
    assert cfg.SCOPE_T["free_5y_top1k"] == 1260
    assert cfg.SCOPE_T["pro_full"] == 4032
    assert cfg.SCOPE_T["registry_depth"] == 4032
    assert cfg.SCOPE_T["research_5y_top1k"] == 1260
