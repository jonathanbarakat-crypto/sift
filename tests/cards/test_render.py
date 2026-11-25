"""v4.1 card face — code-rendered PNG (PIL) tests (mechanics 4, behind cards_v4_1).

Covers the deliverable requirements:
- no-ticker guard on every text string DRAWN into the image (a ticker-like
  token anywhere in the face/diagram aborts the render with FaceError);
- flag off -> FaceError(flag_off) even for with_image=True (kill-switch
  SIFT_CARDS_V4_1=0 path via flags);
- watermark handle+timestamp is drawn on the face (visible in the drawn-text
  log and in the face dict);
- both families render: pairs (z-score + ±entry_z bands + |z| peak +
  half-life marker) and lead-lag (two shifted aggregate series + shaded lead
  band + "peak xcorr @ +{k}d"); hedge/half-life renders too;
- fit-assert: every drawn text bbox sits inside the frame; pixel-scan finds
  no ink outside the frame; an over-long string is refused by the fit assert;
- 2x PNG + 50% downscale are returned and decode as valid PNGs of the right
  sizes;
- render/render_share wire the optional `image` field (base64 PNGs) and the
  JSON artifact is otherwise unchanged; the image field itself is
  guard-scanned.
"""

import base64
import io
import json

import numpy as np
import pytest
from PIL import Image

from sift import flags
from sift.cards import CardError, CardRegistry
from sift.cards.face import CardFaceRenderer, FaceError, _guard_face_artifact, _guard_no_tickers
from sift.cards.render import (
    ART_LABEL,
    CardImageRenderer,
    H,
    S,
    W,
    assert_no_ink_outside_frame,
    attach_card_image,
    inner_frame_rect,
    render_face_png,
)
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


def _face_dict(**over):
    face = {
        "card_id": "card-" + "ab" * 32,
        "kind": "regular",
        "ghost_no": None,
        "family": "pairs",
        "scope_id": "free-us-largecap-daily",
        "tier": "gold",
        "survivor_class": "S0-S5 full depth",
        "sector_aggregate": {"bucket": "cross_sectional_pairs_aggregate",
                             "level": "aggregate"},
        "specifics": {"hedge_ratio": 0.83, "half_life_days": 12, "entry_z": 2.0},
        "gauntlet_stats": {"stages_passed": ["S0", "S1", "S2", "S3", "S4", "S5"],
                           "s2_p_value": 0.0042, "s3_dsr": 2.18,
                           "s5_full_depth_t": 4.21, "s5_status": "pass",
                           "permutations": 10000},
        "watermark": {"handle": "alice", "ts": 1763404800.0, "note": "n"},
        "disclaimer": ("SIFT does not trade. Survival facts only — not "
                       "investment advice; no P&L, no picks (v3 §12)."),
    }
    face.update(over)
    return face


def _z_series(seed=42, T=120, phi=0.82):
    rng = np.random.default_rng(seed)
    z = np.zeros(T)
    for t in range(1, T):
        z[t] = phi * z[t - 1] + rng.normal(0, 0.7)
    return (z / np.std(z)).tolist()


def _pairs_diagram():
    z = _z_series()
    return {"family": "pairs", "series": z, "entry_z": 2.0,
            "peak_idx": int(np.argmax(np.abs(z))), "half_life_days": 12}


def _lead_lag_diagram():
    rng = np.random.default_rng(7)
    leader = np.cumsum(rng.normal(0, 1, 120)).tolist()
    return {"family": "lead-lag", "leader": leader, "lag": 5}


def _hedge_diagram():
    z = _z_series()
    return {"family": "hedge", "series": np.cumsum(z).tolist(),
            "peak_idx": int(np.argmax(np.cumsum(z))), "half_life_days": 18}


def _decode(png_bytes):
    return Image.open(io.BytesIO(png_bytes))


# -- both families render; 2x + 1x; sizes ---------------------------------------

def test_pairs_family_renders_2x_and_1x():
    png2x, png1x = render_face_png(_face_dict(), _pairs_diagram())
    im2, im1 = _decode(png2x), _decode(png1x)
    assert im2.format == "PNG" and im1.format == "PNG"
    assert im2.size == (W * S, H * S)          # 2x
    assert im1.size == (W, H)                  # 50% downscale
    assert len(png2x) > 10_000 and len(png1x) > 10_000


def test_lead_lag_family_renders():
    png2x, _ = render_face_png(_face_dict(family="lead-lag"), _lead_lag_diagram())
    assert _decode(png2x).size == (W * S, H * S)


def test_hedge_family_renders():
    png2x, _ = render_face_png(_face_dict(family="hedge"), _hedge_diagram())
    assert _decode(png2x).size == (W * S, H * S)


def test_all_tiers_render_with_same_layout():
    for tier in ("gold", "silver", "bronze"):
        png2x, _ = render_face_png(_face_dict(tier=tier), _pairs_diagram())
        assert _decode(png2x).size == (W * S, H * S)


def test_ghost_card_renders():
    png2x, _ = render_face_png(_face_dict(kind="ghost", ghost_no=3), _pairs_diagram())
    assert _decode(png2x).size == (W * S, H * S)


# -- watermark + labels drawn on the face -------------------------------------------

def test_watermark_handle_and_ts_drawn_on_face():
    r = CardImageRenderer()
    face = _face_dict()
    r.render(face, _pairs_diagram())
    texts = " | ".join(r.canvas.drawn_texts)
    assert "shared by @alice · 2025-11-17T18:40:00Z" in texts   # handle + ts
    assert ART_LABEL in texts                    # anonymized-aggregate label
    assert "gold tier" in texts                  # tier identity on the band
    assert any("half-life 12d" in t for t in r.canvas.drawn_texts)


def test_lead_lag_diagram_grammar_drawn():
    r = CardImageRenderer()
    r.render(_face_dict(family="lead-lag"), _lead_lag_diagram())
    texts = " | ".join(r.canvas.drawn_texts)
    assert "peak xcorr @ +5d" in texts
    assert "leader" in texts and "follower" in texts


def test_pairs_diagram_grammar_drawn():
    r = CardImageRenderer()
    r.render(_face_dict(), _pairs_diagram())
    texts = " | ".join(r.canvas.drawn_texts)
    assert "±2σ entry" in texts
    assert "|z| peak" in texts


# -- no-ticker guard on rendered text -------------------------------------------------

def test_guard_runs_on_every_drawn_string():
    """A ticker-like token in ANY face field that gets drawn must abort."""
    with pytest.raises(FaceError):
        render_face_png(_face_dict(scope_id="free-AAPL-daily"), _pairs_diagram())
    with pytest.raises(FaceError):
        render_face_png(_face_dict(watermark={"handle": "TSLA", "ts": 1.0}),
                        _pairs_diagram())
    with pytest.raises(FaceError):
        render_face_png(_face_dict(disclaimer="pairs AAPL vs MSFT only"),
                        _pairs_diagram())


def test_diagram_strings_are_guarded_when_drawn():
    """The renderer's text path guards EVERY string it draws (fail closed on
    ticker-like tokens) — exercised directly on the canvas text primitive."""
    from sift.cards import render as R
    canv = R._Canvas()
    with pytest.raises(FaceError):
        canv.text(10, 10, "MSFT spread z", R._font("mono", 20), (255, 255, 255))
    # and a clean string passes
    canv.text(10, 10, "anonymized aggregate series", R._font("mono", 20),
              (255, 255, 255))


def test_drawn_texts_all_pass_guard():
    r = CardImageRenderer()
    r.render(_face_dict(), _pairs_diagram())
    for t in r.canvas.drawn_texts:
        _guard_no_tickers(t)      # must not raise


def test_raw_tickers_planted_server_side_never_reach_image(ledger, cards, renderer):
    """Plant raw tickers in the trial params (server-side only) and prove the
    rendered PNG artifact carries none — the face reads stage rows + card row
    only, and the guard runs on the serialized image artifact too."""
    verdict, trial = _run_validated_green(ledger, cards, handle="alice")
    ledger.con.execute(
        "UPDATE trial SET params_json=? WHERE trial_id=?",
        (json.dumps({"pair": ["AAPL", "MSFT"]}), trial.trial_id))
    ledger.con.commit()
    face = renderer.render(verdict["card_id"], viewer_handle="alice",
                           now=1700000000.0, with_image=True,
                           diagram=_pairs_diagram())
    blob = json.dumps(face, default=str)
    assert "AAPL" not in blob and "MSFT" not in blob
    assert "image" in face and face["image"]["content_type"] == "image/png"
    # the guard also ran on the JSON artifact incl. the image field
    _guard_face_artifact(face)


# -- flag off -> FaceError(flag_off), incl. image path ---------------------------------

def test_flag_off_refuses_image_render(ledger, cards, renderer):
    verdict, _ = _run_validated_green(ledger, cards, handle="alice")
    flags.disable_for_user("cards_v4_1", "alice")
    with pytest.raises(FaceError):
        renderer.render(verdict["card_id"], viewer_handle="alice",
                        with_image=True, diagram=_pairs_diagram())
    with pytest.raises(FaceError):
        renderer.render_share(verdict["card_id"], owner_handle="alice",
                              with_image=True, diagram=_pairs_diagram())


def test_kill_switch_env_zero_refuses(monkeypatch, ledger, cards, renderer):
    verdict, _ = _run_validated_green(ledger, cards, handle="alice")
    monkeypatch.setenv("SIFT_CARDS_V4_1", "0")
    flags.reset()
    with pytest.raises(FaceError):
        renderer.render(verdict["card_id"], viewer_handle="alice", with_image=True,
                        diagram=_pairs_diagram())
    monkeypatch.delenv("SIFT_CARDS_V4_1")
    flags.reset()


# -- fit-assert + pixel-scan ------------------------------------------------------------

def test_fit_assert_and_pixel_scan_pass_on_normal_face():
    r = CardImageRenderer()
    r.render(_face_dict(), _pairs_diagram())     # would raise if any text overflowed
    # pixel-scan on a clean render: no ink outside the frame
    png2x, _ = render_face_png(_face_dict(), _pairs_diagram())
    assert_no_ink_outside_frame(_decode(png2x), S)   # must not raise


def test_fit_assert_catches_overflow():
    """A pathologically long chip must be refused by the fit assert, not
    silently drawn past the frame."""
    face = _face_dict(scope_id="x" * 400)
    with pytest.raises(FaceError):
        render_face_png(face, _pairs_diagram())


def test_pixel_scan_detects_ink_outside_frame():
    png2x, _ = render_face_png(_face_dict(), _pairs_diagram())
    im = _decode(png2x).copy()
    im.paste((255, 0, 0), (0, 0, 20, 20))       # red dot in the outer margin
    with pytest.raises(FaceError):
        assert_no_ink_outside_frame(im, S)


def test_inner_frame_rect_is_inside_outer():
    from sift.cards.render import frame_rect
    fo, fi = frame_rect(), inner_frame_rect()
    assert fi[0] >= fo[0] and fi[1] >= fo[1] and fi[2] <= fo[2] and fi[3] <= fo[3]


# -- wiring: optional image field, JSON artifact unchanged -------------------------------

def test_render_wires_optional_image_field(ledger, cards, renderer):
    verdict, _ = _run_validated_green(ledger, cards, handle="alice")
    face_json = renderer.render(verdict["card_id"], viewer_handle="alice",
                                now=1700000000.0)
    face_img = renderer.render(verdict["card_id"], viewer_handle="alice",
                               now=1700000000.0, with_image=True,
                               diagram=_pairs_diagram())
    # JSON artifact unchanged apart from the optional image field
    for k, v in face_json.items():
        assert face_img[k] == v
    assert set(face_img) - set(face_json) == {"image"}
    img = face_img["image"]
    assert img["encoding"] == "base64" and img["content_type"] == "image/png"
    assert img["scale"] == S and (img["width"], img["height"]) == (W, H)
    im2 = _decode(base64.b64decode(img["png_2x_b64"]))
    im1 = _decode(base64.b64decode(img["png_1x_b64"]))
    assert im2.size == (W * S, H * S) and im1.size == (W, H)
    assert img["bytes_2x"] == len(base64.b64decode(img["png_2x_b64"]))


def test_render_share_image_watermarked_with_owner(ledger, cards, renderer):
    verdict, _ = _run_validated_green(ledger, cards, handle="alice")
    share = renderer.render_share(verdict["card_id"], owner_handle="alice",
                                  now=1700000001.0, with_image=True,
                                  diagram=_pairs_diagram())
    assert share["watermark"]["handle"] == "alice"
    r = CardImageRenderer()
    r.render(share, _pairs_diagram())
    assert any("@alice" in t for t in r.canvas.drawn_texts)
    # foreign handle still cannot produce a share image
    with pytest.raises(CardError):
        renderer.render_share(verdict["card_id"], owner_handle="eve",
                              with_image=True, diagram=_pairs_diagram())


def test_attach_card_image_returns_jsonable_dict():
    att = attach_card_image(_face_dict(), _pairs_diagram())
    blob = json.dumps(att, default=str)
    assert "png_2x_b64" in blob
    _guard_face_artifact(att)


# -- diagram validation --------------------------------------------------------------------

def test_diagram_requires_min_series_length():
    with pytest.raises(FaceError):
        render_face_png(_face_dict(), {"series": [1.0, 2.0]})


def test_diagram_rejects_non_finite():
    s = _z_series()
    s[5] = float("nan")
    with pytest.raises(FaceError):
        render_face_png(_face_dict(), {"series": s})


def test_diagram_rejects_out_of_range_marker():
    with pytest.raises(FaceError):
        render_face_png(_face_dict(), {"series": _z_series(), "peak_idx": 9999})


def test_unknown_tier_refused():
    with pytest.raises(FaceError):
        render_face_png(_face_dict(tier="diamond"), _pairs_diagram())


def test_platinum_tier_renders():
    # v4.2: platinum is a first-class tier on the face
    out = render_face_png(_face_dict(tier="platinum"), _pairs_diagram())
    png = out[0] if isinstance(out, tuple) else out
    assert png.startswith(b"\x89PNG")
