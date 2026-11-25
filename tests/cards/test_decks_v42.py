"""v4.2 §5 copy layer for cards-as-candidates & decks (task t_6218276b).

Covers the deliverable:
- banned-phrase grep test over EVERY new deck string (F–S rank labels and
  phrases, deck explainer, candidates line, deck event gloss): no
  buy/sell/long/short, no P&L/returns, no profit-expectation/"value"
  language, no urgency/countdown, no "top deck"/"best portfolio"; a banned
  phrase -> the builder itself raises (render refused, fail closed).
- the explicit onboarding line: "no single card is an instant win"; cards =
  evidence about connections (candidate tier + health); decks = tested
  portfolios with F–S evidence ranks; user makes all trading decisions;
  SIFT does not trade. Served by GET /api/onboarding (waitlist payload)
  and built only in the copy builders.
- deck event payloads (RSS/API) carry magnitude/aggregate/rank ONLY:
  negative magnitude, forbidden keys (symbol/sign/direction/P&L),
  ticker-shaped tokens, unknown ranks, and banned phrases all raise
  NotifyError and the event is refused.
- watermarked deck artifacts: watermark handle+ts present; no tickers and
  no sign/direction anywhere (ledger stage rows only, never params_json);
  the serialized artifact passes the no-ticker + no-banned-phrase guard.
- flag-off behavior: every deck builder raises DeckCopyError(flag_off)
  with cards_v4_1 disabled (global + per-user + env kill-switch), and the
  onboarding endpoint falls back fail-closed.
"""

import importlib.util
import json
import re
from pathlib import Path

import pytest

from sift import flags
from sift.cards.notify import NotifyError, contains_banned
from sift.decks import copy as deck_copy
from sift.decks.copy import (
    BANNED_PHRASES_DECK,
    BANNED_WORDS_DECK,
    DECK_EVENT_TYPES,
    DECK_PAYLOAD_SCHEMA_VERSION,
    DECK_RANKS,
    DeckCopyError,
    assert_deck_copy_clean,
    build_deck_artifact,
    build_deck_event_payload,
    cards_are_candidates_line,
    contains_banned_deck,
    deck_event_copy,
    deck_explainer,
    deck_rank_label,
    deck_rank_phrase,
)


@pytest.fixture(autouse=True)
def _flags_reset():
    flags.reset()
    yield
    flags.reset()


# ============================================================================
# 1. Banned-phrase grep over EVERY new deck string (fail-closed)
# ============================================================================

def _all_deck_strings():
    strings = {
        "candidates_line": cards_are_candidates_line(),
        "deck_explainer": deck_explainer(bucket="n10_all_pairs"),
        "deck_explainer_plain": deck_explainer(),
    }
    for rank in DECK_RANKS:
        strings[f"rank_label:{rank}"] = deck_rank_label(rank)
        strings[f"rank_phrase:{rank}"] = deck_rank_phrase(rank)
    for event in DECK_EVENT_TYPES:
        strings[f"deck_event:{event}"] = deck_event_copy(
            event, "deck-abc", rank="B", magnitude=2.5, aggregate=3)
    return strings


def test_every_new_deck_string_passes_the_banned_phrase_grep():
    """Grep every new deck string for the banned phrases. The candidates
    line scans against its dedicated clause guard (the mandated
    'you make all trading decisions / SIFT does not trade' negation);
    everything else against the full deck guard."""
    for name, text in _all_deck_strings().items():
        if name == "candidates_line":
            deck_copy._assert_candidates_clause_clean(text)   # must not raise
        else:
            assert_deck_copy_clean(text)                       # must not raise
        low = text.lower()
        for w in BANNED_WORDS_DECK:
            assert not re.search(r"\b" + w + r"\b", low), \
                f"{name} contains directional word {w!r}"
        for phrase in ("top deck", "best deck", "top portfolio",
                       "best portfolio", "winner"):
            assert phrase not in low, f"{name} contains {phrase!r}"


def test_banned_phrase_list_covers_the_v42_named_bans():
    low = [p.lower() for p in BANNED_PHRASES_DECK]
    # deck-named bans (v4.2 §5)
    for required in ("top deck", "best deck", "top portfolio", "best portfolio",
                     "winner", "guaranteed outcome", "sure thing"):
        assert required in low, f"BANNED_PHRASES_DECK missing {required!r}"
    # inherited card/notify bans
    for required in ("pnl", "p&l", "profit", "value", "return", "yield",
                     "countdown", "act now", "guaranteed", "risk-free"):
        assert required in low, f"BANNED_PHRASES_DECK missing {required!r}"
    assert tuple(w.lower() for w in BANNED_WORDS_DECK) == (
        "buy", "sell", "long", "short")


def test_assert_deck_copy_clean_refuses_banned_phrases():
    for probe in ("you should buy now", "go long on this deck",
                  "the top deck of the week", "best portfolio returns",
                  "guaranteed outcome", "expected return 20%",
                  "this deck is more valuable than the last",
                  "act now — countdown ends soon"):
        with pytest.raises(DeckCopyError):
            assert_deck_copy_clean(probe)
    # clean prose passes through (and 'longer' must NOT trip the whole-word scan)
    assert assert_deck_copy_clean("survival facts only") == "survival facts only"
    assert assert_deck_copy_clean("the longer the window, the stronger the test")


def test_every_deck_builder_fails_closed_on_injected_banned_phrase(monkeypatch):
    """If a banned phrase ever slipped into a deck template, the builder
    itself refuses to return it (fail closed at build time)."""
    monkeypatch.setitem(deck_copy._RANK_PHRASES, "A",
                        "the best portfolio — guaranteed returns")
    with pytest.raises(DeckCopyError):
        deck_rank_phrase("A")
    monkeypatch.setitem(deck_copy._RANK_LABELS, "A",
                        "A — the best portfolio, guaranteed profit")
    with pytest.raises(DeckCopyError):
        deck_rank_label("A")
    # the candidates-line clause guard also refuses injected bans
    with pytest.raises(DeckCopyError):
        deck_copy._assert_candidates_clause_clean(
            "No single card is an instant win — but this is the top deck, "
            "guaranteed profit")
    # a clean deck explainer still passes after restore
    monkeypatch.setitem(deck_copy._RANK_PHRASES, "A",
                        "the portfolio evidence is strong")
    assert "portfolio" in deck_explainer(bucket="n10_all_pairs")


def test_unknown_deck_rank_refused():
    with pytest.raises(DeckCopyError):
        deck_rank_label("Z")
    with pytest.raises(DeckCopyError):
        deck_rank_phrase("platinum")


# ============================================================================
# 2. The explicit candidates line (onboarding / first card)
# ============================================================================

def test_candidates_line_says_no_single_card_is_an_instant_win():
    line = cards_are_candidates_line()
    low = line.lower()
    assert "no single card is an instant win" in low
    assert "candidate" in low and "tier" in low and "health" in low
    assert "tested portfolios" in low and "f to s" in low
    assert "you make all trading decisions" in low
    assert "sift does not trade" in low
    # full guard on the residue outside the mandated negated clause
    deck_copy._assert_candidates_clause_clean(line)


def test_candidates_line_fail_closed_on_injected_banned(monkeypatch):
    orig = cards_are_candidates_line
    import sift.decks.copy as D
    monkeypatch.setattr(D, "cards_are_candidates_line", lambda *a, **k: (
        "No single card is an instant win — buy the top deck now"))
    with pytest.raises(DeckCopyError):
        D._assert_candidates_clause_clean(
            "No single card is an instant win — buy the top deck now")
    monkeypatch.setattr(D, "cards_are_candidates_line", orig)


# ============================================================================
# 3. Onboarding at signup serves the candidates line (GET /api/onboarding)
# ============================================================================

def _load_waitlist():
    spec = importlib.util.spec_from_file_location(
        "waitlist_v42",
        Path(__file__).resolve().parents[2] / "web" / "api" / "waitlist.py")
    wl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wl)
    return wl


def test_onboarding_endpoint_serves_candidates_line(tmp_path, monkeypatch):
    wl = _load_waitlist()
    monkeypatch.setattr(wl, "DATA_DIR", tmp_path)
    monkeypatch.setattr(wl, "WAITLIST_CSV", tmp_path / "waitlist.csv")
    payload = wl.onboarding_payload()
    assert payload["cards_are_candidates"] == cards_are_candidates_line()
    assert "no single card is an instant win" in \
        payload["cards_are_candidates"].lower()
    # safe variant never raises and carries the line too
    safe = wl.onboarding_payload_safe()
    assert "no single card is an instant win" in safe["cards_are_candidates"].lower()


def test_onboarding_flag_off_falls_back_fail_closed(tmp_path, monkeypatch):
    wl = _load_waitlist()
    monkeypatch.setattr(wl, "DATA_DIR", tmp_path)
    monkeypatch.setattr(wl, "WAITLIST_CSV", tmp_path / "waitlist.csv")
    flags.set_flag("cards_v4_1", False)
    with pytest.raises(Exception):
        wl.onboarding_payload()          # builder raises (fail closed)
    safe = wl.onboarding_payload_safe()   # never raises
    assert "no single card is an instant win" in safe["cards_are_candidates"].lower()
    assert safe["source"].startswith("static fallback")


# ============================================================================
# 4. Deck event payload: magnitude / aggregate / rank ONLY (RSS/API)
# ============================================================================

def test_deck_payload_carries_magnitude_aggregate_rank_only():
    p = build_deck_event_payload(deck_id="deck-1", bucket="n10_all_pairs",
                                 rank="A", event="deck_ranked", magnitude=2.5,
                                 aggregate=7, now=1700000000.0)
    assert p["schema"] == DECK_PAYLOAD_SCHEMA_VERSION
    assert p["event"] == "deck_ranked"
    assert p["rank"] == "A"
    assert p["magnitude_z"] == 2.5          # absolute
    assert p["decks_in_stratum_today"] == 7  # aggregate only
    assert set(p) == {"schema", "event", "deck_id", "bucket", "rank",
                      "magnitude_z", "decks_in_stratum_today", "ts", "gloss"}
    # gloss built only in deck_event_copy and clean
    assert p["gloss"] == deck_event_copy("deck_ranked", "deck-1", rank="A",
                                         magnitude=2.5, aggregate=7)
    assert contains_banned(p["gloss"]) is None
    assert contains_banned_deck(p["gloss"]) is None


def test_deck_payload_magnitude_is_absolute():
    p = build_deck_event_payload(deck_id="deck-1", bucket="b", rank="F",
                                 magnitude=-3.2, now=1.0)
    assert p["magnitude_z"] == 3.2           # sign stripped, never shipped
    # unknown rank refused
    with pytest.raises(NotifyError):
        build_deck_event_payload(deck_id="d", bucket="b", rank="Z", now=1.0)
    # unknown event refused
    with pytest.raises(NotifyError):
        build_deck_event_payload(deck_id="d", bucket="b", rank="A",
                                 event="warp", now=1.0)
    # a payload that still carries a signed magnitude after build is
    # refused by the guard (defense in depth)
    bad = build_deck_event_payload(deck_id="d", bucket="b", rank="A", now=1.0)
    bad["magnitude_z"] = -1.0
    with pytest.raises(NotifyError):
        deck_copy._guard_deck_payload(bad)


def test_deck_payload_forbidden_keys_and_tickers_refused():
    # forbidden key injected post-build -> guard refuses
    p = build_deck_event_payload(deck_id="deck-1", bucket="b", rank="A", now=1.0)
    p["symbol"] = "AAPL"
    with pytest.raises(NotifyError):
        deck_copy._guard_deck_payload(p)
    p2 = build_deck_event_payload(deck_id="deck-1", bucket="b", rank="A", now=1.0)
    p2["gloss"] = p2["gloss"] + " AAPL"
    with pytest.raises(NotifyError):
        deck_copy._guard_deck_payload(p2)     # ticker-shaped token in string
    p3 = build_deck_event_payload(deck_id="deck-1", bucket="b", rank="A", now=1.0)
    p3["sign"] = -1
    with pytest.raises(NotifyError):
        deck_copy._guard_deck_payload(p3)
    p4 = build_deck_event_payload(deck_id="deck-1", bucket="b", rank="A", now=1.0)
    p4["gloss"] = p4["gloss"] + " — guaranteed profit"
    with pytest.raises(NotifyError):
        deck_copy._guard_deck_payload(p4)     # banned phrase in payload string


def test_deck_payload_serialized_has_no_ticker_no_direction():
    p = build_deck_event_payload(deck_id="deck-7", bucket="n50_all_pairs",
                                 rank="S", magnitude=4.1, aggregate=12,
                                 now=1700000000.0)
    blob = json.dumps(p, sort_keys=True)
    assert "AAPL" not in blob and "MSFT" not in blob
    low = blob.lower()
    for w in ("buy", "sell", "long", "short"):
        assert w not in low
    for phrase in ("top deck", "best portfolio", "profit", "return", "value"):
        assert phrase not in low
    # key names carry no direction/symbol surface either
    for k in p:
        kl = k.lower()
        assert kl not in ("symbol", "sign", "direction", "ticker", "side")


# ============================================================================
# 5. Watermarked deck artifact: no tickers / no sign / no direction
# ============================================================================

def test_deck_artifact_watermarked_and_clean():
    art = build_deck_artifact(
        deck_id="deck-1", bucket="n10_all_pairs", rank="B",
        stage_stats={"s6_net_sr_pass": True, "s6_mintrl": 100, "s6_pbo": 0.2},
        watermark_handle="alice", now=1700000000.0)
    assert art["watermark"]["handle"] == "alice"
    assert art["watermark"]["ts"] == 1700000000.0
    assert art["rank"] == "B"
    assert art["rank_label"] == deck_rank_label("B")
    assert "no single card is an instant win" in art["candidates_line"].lower()
    assert "sift does not trade" in art["disclaimer"].lower()
    blob = json.dumps(art, default=str)
    assert "AAPL" not in blob
    low = blob.lower()
    for w in ("buy", "sell", "long", "short"):
        assert w not in low
    deck_copy._guard_deck_artifact(art)     # must not raise


def test_deck_artifact_refuses_leaked_ticker_or_banned(monkeypatch):
    # stage_stats carrying a ticker-shaped token (e.g. if a caller ever fed
    # params_json in) -> the artifact guard refuses (fail closed)
    with pytest.raises(DeckCopyError):
        build_deck_artifact(deck_id="deck-1", bucket="b", rank="A",
                            stage_stats={"note": "AAPL vs MSFT"},
                            watermark_handle="alice", now=1.0)
    # banned phrase in stage stats -> refused
    with pytest.raises(DeckCopyError):
        build_deck_artifact(deck_id="deck-1", bucket="b", rank="A",
                            stage_stats={"note": "guaranteed profit"},
                            watermark_handle="alice", now=1.0)
    # unknown rank -> refused at build time
    with pytest.raises(DeckCopyError):
        build_deck_artifact(deck_id="deck-1", bucket="b", rank="Z",
                            watermark_handle="alice", now=1.0)


# ============================================================================
# 6. Flag-off behavior (fail closed) + kill-switch
# ============================================================================

def test_flag_off_refuses_every_deck_builder():
    flags.set_flag("cards_v4_1", False)
    with pytest.raises(DeckCopyError):
        cards_are_candidates_line()
    with pytest.raises(DeckCopyError):
        deck_explainer(bucket="b")
    with pytest.raises(DeckCopyError):
        deck_rank_label("A")
    with pytest.raises(DeckCopyError):
        deck_event_copy("deck_hit", "d", rank="A")
    with pytest.raises(DeckCopyError):
        build_deck_event_payload(deck_id="d", bucket="b", rank="A", now=1.0)
    with pytest.raises(DeckCopyError):
        build_deck_artifact(deck_id="d", bucket="b", rank="A",
                            watermark_handle="alice", now=1.0)


def test_per_user_flag_off_refuses_deck_copy():
    flags.disable_for_user("cards_v4_1", "alice")
    with pytest.raises(DeckCopyError):
        cards_are_candidates_line(handle="alice")
    # another user still works
    assert "instant win" in cards_are_candidates_line(handle="bob")


def test_kill_switch_env_zero_refuses_deck_copy(monkeypatch):
    monkeypatch.setenv("SIFT_CARDS_V4_1", "0")
    flags.reset()
    with pytest.raises(DeckCopyError):
        cards_are_candidates_line()
    with pytest.raises(DeckCopyError):
        deck_rank_label("S")
    monkeypatch.delenv("SIFT_CARDS_V4_1")
    flags.reset()
    assert "instant win" in cards_are_candidates_line()


def test_decks_v4_2_kill_switch_refuses_deck_copy(monkeypatch):
    monkeypatch.setenv("SIFT_DECKS_V4_2", "0")
    flags.reset()
    with pytest.raises(DeckCopyError):
        cards_are_candidates_line()
    with pytest.raises(DeckCopyError):
        deck_rank_label("S")
    with pytest.raises(DeckCopyError):
        build_deck_event_payload(deck_id="d", bucket="b", rank="A", now=1.0)
    with pytest.raises(DeckCopyError):
        build_deck_artifact(deck_id="d", bucket="b", rank="A",
                            watermark_handle="alice", now=1.0)
    monkeypatch.delenv("SIFT_DECKS_V4_2")
    flags.reset()
    assert "instant win" in cards_are_candidates_line()
