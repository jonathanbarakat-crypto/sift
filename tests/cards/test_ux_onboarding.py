"""v4.1 newcomer-friendly card UX + onboarding (task t_679cd083).

Covers the deliverable:
- copy grep test over EVERY new user-facing string (explainer, glossary,
  gauge labels, health phrases, onboarding, ToS, hit hint, hit
  notification copy): no buy/sell/long/short, no P&L/returns, no
  profit-expectation/"value" language, no urgency/countdown; a banned
  phrase -> the builder itself raises CopyError (render refused).
- gauge mapping unit tests: |z| vs band -> stretched/normal/compressed
  (boundaries included), sign-insensitive (|-2.5| == |2.5|).
- face renders with the explainer + glossary + gauge + health blocks;
  the PNG renderer draws them and every drawn string passes the
  no-ticker guard.
- onboarding shown at signup: GET /api/onboarding returns the plain
  statement + ToS; POST /api/waitlist refuses without tos_accepted and
  records it (fail closed).
- no-symbol/no-sign invariant on all artifacts: face + share artifacts
  carry no ticker-like token and no sign/direction (gauge magnitude is
  |z| only); params_json never read by the face.
- flag-off behavior: every copy builder raises CopyError(flag_off) with
  cards_v4_1 disabled (global + per-user + env kill-switch), and the
  face render still raises FaceError(flag_off).
- health ratio language: weakening/strengthening/holding phrases.
"""

import json
import re

import pytest

from sift import flags
from sift.cards import CardRegistry
from sift.cards import copy as card_copy
from sift.cards.copy import (
    BANNED_PHRASES,
    CopyError,
    assert_copy_clean,
    face_explainer,
    gauge_band,
    gauge_label,
    glossary,
    glossary_entry,
    health_phrase,
    hit_loop_hint,
    hit_notification_copy,
    onboarding_statement,
    tos_acceptance_text,
)
from sift.cards.face import CardFaceRenderer, FaceError, _guard_face_artifact
from sift.cards.render import CardImageRenderer
from sift.registry import Ledger
from tests.cards.test_minting import SECRET, _run_validated_green


@pytest.fixture(autouse=True)
def _flags_reset():
    flags.reset()
    yield
    flags.reset()


@pytest.fixture()
def ledger(tmp_path):
    led = Ledger(tmp_path / "ledger.db")
    yield led
    led.close()


@pytest.fixture()
def cards(tmp_path):
    c = CardRegistry(tmp_path / "cards.db", secret=SECRET)
    yield c
    c.close()


@pytest.fixture()
def renderer(ledger, cards):
    return CardFaceRenderer(ledger, cards)


# ============================================================================
# 1. Copy grep test over EVERY new string (fail-closed)
# ============================================================================

def _all_user_facing_strings():
    strings = {
        "face_explainer": face_explainer(),
        "face_explainer_5": face_explainer(stages=5),
        "onboarding_statement": onboarding_statement(),
        "tos_acceptance_text": tos_acceptance_text(),
        "hit_loop_hint": hit_loop_hint(),
        "hit_notification_copy": hit_notification_copy(
            "card-abc", "stretched", 2.31, hit_count_trailing_12m=7,
            health_ratio=1.3),
        "health_weak": health_phrase(0.8),
        "health_strong": health_phrase(1.3),
        "health_hold": health_phrase(1.0),
        "health_none": health_phrase(None),
        "gauge_stretched": gauge_label("stretched"),
        "gauge_normal": gauge_label("normal"),
        "gauge_compressed": gauge_label("compressed"),
    }
    for term, text in glossary().items():
        strings[f"glossary:{term}"] = text
    return strings


def test_every_new_string_passes_the_banned_phrase_grep():
    """Grep every new user-facing string for the banned phrases. The
    onboarding/ToS legal clauses are scanned against their dedicated
    legal-clause guard; everything else against the full guard (substring
    BANNED_PHRASES + whole-word directional BANNED_WORDS)."""
    for name, text in _all_user_facing_strings().items():
        if name in ("onboarding_statement", "tos_acceptance_text"):
            card_copy._assert_legal_clause_clean(text)   # must not raise
        else:
            assert_copy_clean(text)                     # must not raise
        # ...and no directional word anywhere except the mandated negated
        # onboarding clause
        low = text.lower()
        for w in card_copy.BANNED_WORDS:
            if name == "onboarding_statement" and w in ("buy", "sell"):
                continue
            assert not re.search(r"\b" + w + r"\b", low), \
                f"{name} contains directional word {w!r}"


def test_banned_phrase_list_covers_the_task_named_bans():
    low = [p.lower() for p in BANNED_PHRASES]
    for required in ("pnl", "p&l", "profit", "value", "valuable", "return",
                     "yield", "countdown", "act now", "last chance",
                     "guaranteed", "risk-free"):
        assert required in low, f"BANNED_PHRASES missing {required!r}"
    # directional words are whole-word bans (shared with notify)
    assert tuple(w.lower() for w in card_copy.BANNED_WORDS) == (
        "buy", "sell", "long", "short")


def test_assert_copy_clean_refuses_banned_phrase():
    with pytest.raises(CopyError):
        assert_copy_clean("you should buy now — guaranteed")
    with pytest.raises(CopyError):
        assert_copy_clean("expected return of 20%")
    with pytest.raises(CopyError):
        assert_copy_clean("this card is more valuable than gold")
    with pytest.raises(CopyError):
        assert_copy_clean("sell before the countdown ends")
    with pytest.raises(CopyError):
        assert_copy_clean("go long on this pair")      # whole-word directional
    # clean text passes through unchanged (and 'longer' must NOT trip the
    # whole-word scan)
    assert assert_copy_clean("survival facts only") == "survival facts only"
    assert assert_copy_clean("the longer the window, the stronger the test")


def test_every_builder_fails_closed_on_injected_banned_phrase(monkeypatch):
    """If a banned phrase ever slipped into a builder's template, the
    builder itself refuses to return it (fail closed at build time)."""
    monkeypatch.setitem(card_copy.GLOSSARY, "z-score",
                        "the z-score tells you when to buy")
    with pytest.raises(CopyError):
        glossary()
    with pytest.raises(CopyError):
        glossary_entry("z-score")


def test_glossary_unknown_term_refused():
    with pytest.raises(CopyError):
        glossary_entry("lambda-omega")


# ============================================================================
# 2. Gauge mapping unit tests (|z| vs band)
# ============================================================================

@pytest.mark.parametrize("z,band", [
    (0.0, "compressed"),
    (0.49, "compressed"),
    (0.5, "normal"),          # boundary: >= 0.5 is normal
    (1.0, "normal"),
    (1.99, "normal"),
    (2.0, "stretched"),       # boundary: >= 2.0 is stretched
    (2.31, "stretched"),
    (5.5, "stretched"),
])
def test_gauge_band_mapping(z, band):
    assert gauge_band(z) == band


def test_gauge_is_sign_insensitive():
    """Magnitude is |z| only — the band never depends on the sign."""
    for z in (0.3, 0.7, 1.5, 2.0, 2.7, 4.0):
        assert gauge_band(z) == gauge_band(-z)


def test_gauge_band_rejects_unknown_band_label():
    with pytest.raises(CopyError):
        gauge_label("mega")


def test_gauge_labels_are_plain_language_and_clean():
    for band in ("compressed", "normal", "stretched"):
        label = gauge_label(band)
        assert_copy_clean(label)
        assert label  # non-empty


# ============================================================================
# 3. Health ratio language
# ============================================================================

def test_health_phrase_ratio_language():
    assert health_phrase(0.8) == "weakening ~0.8x of mint baseline"
    assert health_phrase(1.3) == "strengthening ~1.3x of mint baseline"
    assert health_phrase(1.0) == "holding near its mint baseline"
    assert health_phrase(None) == "holding near its mint baseline"
    assert health_phrase(float("nan")) == "holding near its mint baseline"
    # clean of banned phrases
    for r in (0.5, 0.8, 1.0, 1.3, 2.0, None):
        assert_copy_clean(health_phrase(r))


# ============================================================================
# 4. Face renders with explainer + glossary + gauge + health
# ============================================================================

def test_face_renders_with_newcomer_blocks(ledger, cards, renderer):
    verdict, _ = _run_validated_green(ledger, cards, handle="alice")
    face = renderer.render(verdict["card_id"], viewer_handle="alice",
                           now=1700000000.0, current_abs_z=2.31,
                           health_ratio=0.8)
    # explainer: one paragraph, plain language, certifies + disclaims
    assert "certifies" in face["explainer"]
    assert "nothing about what to do with it" in face["explainer"]
    # glossary behind tooltips (z-score, half-life, DSR, tier, health)
    for term in ("z-score", "half-life", "dsr", "tier", "health"):
        assert term in face["glossary"]
    # gauge: band from |z| + numeric |z| one click away
    assert face["gauge"]["band"] == "stretched"
    assert face["gauge"]["magnitude_z"] == 2.31
    assert "stretched" in face["gauge"]["label"]
    # health: plain ratio language
    assert face["health"]["phrase"] == "weakening ~0.8x of mint baseline"
    assert face["health"]["ratio_vs_mint_baseline"] == 0.8
    # the serialized artifact still passes the no-ticker guard
    _guard_face_artifact(face)


def test_face_gauge_defaults_when_no_live_reading(ledger, cards, renderer):
    verdict, _ = _run_validated_green(ledger, cards, handle="alice")
    face = renderer.render(verdict["card_id"], viewer_handle="alice",
                           now=1700000000.0)
    assert face["gauge"]["band"] == "compressed"     # |z| defaults to 0
    assert face["gauge"]["magnitude_z"] is None
    assert face["health"]["phrase"] == "holding near its mint baseline"
    assert face["health"]["ratio_vs_mint_baseline"] is None


def test_face_share_carries_same_newcomer_blocks(ledger, cards, renderer):
    verdict, _ = _run_validated_green(ledger, cards, handle="alice")
    share = renderer.render_share(verdict["card_id"], owner_handle="alice",
                                  now=1700000001.0, current_abs_z=-2.7,
                                  health_ratio=1.3)
    # sign-insensitive: -2.7 -> |2.7| -> stretched, magnitude_z is 2.7
    assert share["gauge"]["band"] == "stretched"
    assert share["gauge"]["magnitude_z"] == 2.7
    assert share["health"]["phrase"] == "strengthening ~1.3x of mint baseline"
    assert "certifies" in share["explainer"]
    _guard_face_artifact(share)


def test_face_png_draws_explainer_gauge_health(ledger, cards, renderer):
    verdict, _ = _run_validated_green(ledger, cards, handle="alice")
    face = renderer.render(verdict["card_id"], viewer_handle="alice",
                           now=1700000000.0, current_abs_z=2.31,
                           health_ratio=0.8, with_image=True,
                           diagram=_pairs_diagram())
    assert "image" in face
    r = CardImageRenderer()
    r.render(face, _pairs_diagram())
    texts = " | ".join(r.canvas.drawn_texts)
    assert "certifies" in texts
    assert "magnitude: stretched beyond its usual range  (|z| = 2.31)" in texts
    assert "health: weakening ~0.8x of mint baseline" in texts
    # jargon is NOT drawn in the main view (glossary tooltips only)
    assert "DSR" not in texts
    # every drawn string passes the guard
    for t in r.canvas.drawn_texts:
        from sift.cards.face import _guard_no_tickers
        _guard_no_tickers(t)


def _pairs_diagram():
    import numpy as np
    rng = np.random.default_rng(42)
    z = np.zeros(120)
    for t in range(1, 120):
        z[t] = 0.82 * z[t - 1] + rng.normal(0, 0.7)
    z = (z / np.std(z)).tolist()
    return {"family": "pairs", "series": z, "entry_z": 2.0,
            "peak_idx": int(np.argmax(np.abs(z))), "half_life_days": 12}


# ============================================================================
# 5. No-symbol / no-sign invariant on all artifacts
# ============================================================================

def test_no_symbol_no_sign_invariant_on_all_artifacts(ledger, cards, renderer):
    """Plant raw tickers in the trial params (server-side only) and a
    SIGNED z; the artifacts must carry neither the symbols nor the sign."""
    verdict, trial = _run_validated_green(ledger, cards, handle="alice")
    ledger.con.execute(
        "UPDATE trial SET params_json=? WHERE trial_id=?",
        (json.dumps({"pair": ["AAPL", "MSFT"], "sign": -1}), trial.trial_id),
    )
    ledger.con.commit()
    face = renderer.render(verdict["card_id"], viewer_handle="alice",
                           now=1700000000.0, current_abs_z=-3.2,
                           health_ratio=0.7)
    share = renderer.render_share(verdict["card_id"], owner_handle="alice",
                                  now=1700000001.0, current_abs_z=-3.2,
                                  health_ratio=0.7)
    for art in (face, share):
        blob = json.dumps(art, default=str)
        assert "AAPL" not in blob and "MSFT" not in blob
        _guard_face_artifact(art)
        # magnitude is |z| — the sign never rides along
        assert art["gauge"]["magnitude_z"] == 3.2
        assert "signed" not in json.dumps(art["gauge"]).lower()
        assert set(art["gauge"]) == {"band", "label", "magnitude_z", "note"}
        # no directional words anywhere in the artifact
        low = blob.lower()
        for w in ("buy", "sell", "long", "short"):
            assert w not in low, f"artifact contains directional word {w!r}"


def test_rendered_png_artifact_carries_no_sign(ledger, cards, renderer):
    verdict, _ = _run_validated_green(ledger, cards, handle="alice")
    face = renderer.render(verdict["card_id"], viewer_handle="alice",
                           now=1700000000.0, current_abs_z=-2.5,
                           with_image=True, diagram=_pairs_diagram())
    r = CardImageRenderer()
    r.render(face, _pairs_diagram())
    for t in r.canvas.drawn_texts:
        assert "-2.5" not in t          # only the absolute |z| is drawn
        if "|z| = " in t:
            assert "|z| = 2.50" in t


# ============================================================================
# 6. Onboarding at signup (waitlist endpoint)
# ============================================================================

def _load_waitlist():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "waitlist_ux", Path(__file__).resolve().parents[2] / "web" / "api" / "waitlist.py")
    wl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wl)
    return wl


def test_onboarding_endpoint_returns_plain_statement_and_tos(tmp_path, monkeypatch):
    wl = _load_waitlist()
    monkeypatch.setattr(wl, "DATA_DIR", tmp_path)
    monkeypatch.setattr(wl, "WAITLIST_CSV", tmp_path / "waitlist.csv")
    payload = wl.onboarding_payload()
    assert payload["statement"] == onboarding_statement()
    assert payload["tos_acceptance"] == tos_acceptance_text()
    assert "you make the calls" in payload["statement"].lower()
    assert "no resale value" in payload["tos_acceptance"].lower()
    # safe variant never raises
    assert wl.onboarding_payload_safe()["statement"]


def test_onboarding_copy_flag_off_fail_closed():
    flags.set_flag("cards_v4_1", False)
    with pytest.raises(CopyError):
        onboarding_statement()
    with pytest.raises(CopyError):
        tos_acceptance_text()
    with pytest.raises(CopyError):
        hit_loop_hint()
    with pytest.raises(CopyError):
        face_explainer()
    with pytest.raises(CopyError):
        hit_notification_copy("card-x", "normal", 1.0)


def test_signup_requires_tos_and_records_it(tmp_path, monkeypatch):
    wl = _load_waitlist()
    monkeypatch.setattr(wl, "DATA_DIR", tmp_path)
    monkeypatch.setattr(wl, "WAITLIST_CSV", tmp_path / "waitlist.csv")
    # accepted -> recorded
    assert wl.save_email("a@b.io", tos_accepted=True) is True
    rows = wl.WAITLIST_CSV.read_text().strip().splitlines()
    assert rows[0].startswith("email,ts_utc,source,tos_accepted")
    assert rows[1].endswith(",yes")
    # not accepted -> recorded as no (endpoint refuses before this, but the
    # write path keeps the flag honest)
    assert wl.save_email("c@d.io", tos_accepted=False) is True
    assert wl.WAITLIST_CSV.read_text().strip().splitlines()[2].endswith(",no")


def test_waitlist_post_refuses_without_tos():
    """End-to-end handler check: POST /api/waitlist without tos_accepted ->
    400 and nothing written."""
    import tempfile
    from pathlib import Path
    wl = _load_waitlist()
    tmp = tempfile.TemporaryDirectory()
    wl.DATA_DIR = Path(tmp.name)
    wl.WAITLIST_CSV = wl.DATA_DIR / "waitlist.csv"

    class H(wl.Handler):
        def __init__(self):
            self.status = None
            self.body = b""
            self.headers = {}
        def send_response(self, code):
            self.status = code
        def send_header(self, k, v):
            self.headers[k] = v
        def end_headers(self):
            pass
        def wfile_write(self, b):
            self.body += b
        @property
        def wfile(self):
            class W:
                def write(_, b):
                    self.body = getattr(H, "_body", b"")
            return W()

    # drive do_POST with a fake request
    import io
    h = wl.Handler.__new__(wl.Handler)
    h.path = "/api/waitlist"
    h.headers = {"Content-Length": "38", "Content-Type": "application/json"}
    body = json.dumps({"email": "x@y.io"}).encode()   # no tos_accepted
    h.rfile = io.BytesIO(body)
    captured = {}
    def fake_send_json(code, payload):
        captured["code"] = code
        captured["payload"] = payload
    h._send_json = fake_send_json
    h.do_POST()
    assert captured["code"] == 400
    assert not wl.WAITLIST_CSV.exists()

    # with tos_accepted=True -> 200 and recorded
    body = json.dumps({"email": "x@y.io", "tos_accepted": True}).encode()
    h.headers = {"Content-Length": str(len(body)), "Content-Type": "application/json"}
    h.rfile = io.BytesIO(body)
    h.do_POST()
    assert captured["code"] == 200 and captured["payload"]["ok"] is True
    rows = wl.WAITLIST_CSV.read_text().strip().splitlines()
    assert rows[1].split(",")[0] == "x@y.io" and rows[1].endswith(",yes")
    tmp.cleanup()


# ============================================================================
# 7. Flag-off behavior
# ============================================================================

def test_flag_off_refuses_face_and_copy(ledger, cards, renderer):
    verdict, _ = _run_validated_green(ledger, cards, handle="alice")
    flags.disable_for_user("cards_v4_1", "alice")
    with pytest.raises(FaceError):
        renderer.render(verdict["card_id"], viewer_handle="alice",
                        current_abs_z=2.0, health_ratio=1.1)
    with pytest.raises(CopyError):
        face_explainer(handle="alice")
    with pytest.raises(CopyError):
        card_copy.glossary(handle="alice")
    with pytest.raises(CopyError):
        card_copy.glossary_entry("tier", handle="alice")


def test_kill_switch_env_zero_refuses_copy(monkeypatch):
    monkeypatch.setenv("SIFT_CARDS_V4_1", "0")
    flags.reset()
    with pytest.raises(CopyError):
        onboarding_statement()
    with pytest.raises(CopyError):
        card_copy.face_explainer()
    monkeypatch.delenv("SIFT_CARDS_V4_1")
    flags.reset()
    # flag back on -> builders work again
    assert "certifies" in face_explainer()


# ============================================================================
# 8. Hit-loop framing copy
# ============================================================================

def test_hit_loop_hint_never_instructs_to_trade():
    hint = hit_loop_hint()
    low = hint.lower()
    assert "your own charting or broker tool" in low
    assert "never tells you what to do" in low
    for w in ("buy", "sell", "long", "short"):
        assert w not in low
    assert_copy_clean(hint)


def test_hit_notification_copy_carries_strength_and_frequency():
    text = hit_notification_copy("card-abc", "stretched", 2.31,
                                 hit_count_trailing_12m=7, health_ratio=1.3)
    low = text.lower()
    assert "fired" in low
    assert "|z| = 2.31" in text
    assert "7 time(s) in the trailing 12 months" in text
    assert "strengthening ~1.3x" in text
    assert "your own charting" in low
    assert_copy_clean(text)
