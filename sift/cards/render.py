"""v4.1 card face — code-rendered PNG (PIL), server-side only (mechanics 4).

The shareable, watermarked, anonymized public artifact. No AI image
generation: the face is rendered server-side with PIL from the face dict
produced by ``CardFaceRenderer._build_face`` (sift/cards/face.py) plus
per-family diagram inputs (anonymized aggregate series only).

Hard rules (v3 §12 + face.py, unchanged):
- every text string drawn into the image passes ``_guard_no_tickers``
  (fail closed: any ticker-shaped token aborts the render) — raw tickers
  never leave the server;
- the art window shows anonymized aggregate series only and is labeled
  "anonymized aggregate series - no tickers";
- the face is watermarked with handle + timestamp (share traceability,
  v4.1 §4.7.5);
- tier identity via frame/title-band accent (gold/silver/bronze);
- rendered at 2x (S=2); ships the 2x PNG plus a 50% downscale; before
  saving, every drawn text bbox is asserted inside the frame and a
  pixel-scan asserts no ink outside the frame.

Diagram grammar per family (same layout/palette; only the chart differs):
- lead-lag: leader/follower aggregate series (follower = leader shifted
  +k), shaded band from x_k to the end, vertical gold marker
  "peak xcorr @ +{k}d";
- pairs / mean-reversion: spread z-score line, red horizontal bands at
  ±entry_z labeled "±2σ entry" (entry_z defaults to 2), shaded zone
  between the bands, zero line, gold markers at "|z| peak" and
  "half-life {n}d";
- hedge / half-life (duration): z-score or cumulative excursion with a
  peak -> half-decay bracket.

``render_face_png(face, diagram)`` returns ``(png_2x_bytes, png_1x_bytes)``.
``attach_card_image(face, diagram)`` returns the face dict plus an
``image`` field (base64 PNGs) so the JSON artifact stays JSON-able.
"""

from __future__ import annotations

import base64
import io
import math
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sift.cards.face import FaceError, _guard_no_tickers

# -- geometry / palette (dark fintech) ---------------------------------------

S = 2                    # supersample factor: render at 2x, ship 2x + 50%
W, H = 1200, 1500        # logical card size (px at 1x)
MARGIN = 28              # outer margin (pure BG — pixel-scanned)
PAD = 28                 # content pad inside the frame
BORDER = 6               # frame border width (tier accent)

PALETTE = {
    "BG":      "#0D1017",
    "PANEL":   "#151A26",
    "FRAME":   "#343E56",
    "INK":     "#E8EDF5",
    "INK_DIM": "#94A0B6",
    "ACCENT":  "#5EC8A0",
    "ACCENT2": "#7AA2F7",
    "GOLD":    "#E0BA5C",
    "GOLD_DK": "#B08A34",
    "GOLD_BG": "#3A2E14",
    "RED":     "#EB6E64",
    "GRID":    "#262E40",
    "ARTBG":   "#0A0D14",
}

# tier -> frame/title-band accent (gold identity per mechanics 4; silver and
# bronze keep the same layout with their accent)
TIER_ACCENTS = {
    "gold":   {"accent": PALETTE["GOLD"], "dk": PALETTE["GOLD_DK"], "bg": PALETTE["GOLD_BG"]},
    "silver": {"accent": "#C7CEDC",       "dk": "#8A93A6",           "bg": "#262B36"},
    "bronze": {"accent": "#D08A5C",       "dk": "#8C5A2B",           "bg": "#2E2318"},
}

ART_LABEL = "anonymized aggregate series - no tickers"
STAGES = ("S0", "S1", "S2", "S3", "S4", "S5")


def _rgb(hexstr: str) -> tuple[int, int, int]:
    hexstr = hexstr.lstrip("#")
    return (int(hexstr[0:2], 16), int(hexstr[2:4], 16), int(hexstr[4:6], 16))


# -- fonts (DejaVu Sans / DejaVu Sans Mono) ------------------------------------

_FONT_FILES = {
    "sans": "DejaVuSans.ttf",
    "sans_b": "DejaVuSans-Bold.ttf",
    "mono": "DejaVuSansMono.ttf",
    "mono_b": "DejaVuSansMono-Bold.ttf",
}
_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def _find_font(name: str) -> str | None:
    env_dir = os.environ.get("SIFT_CARD_FONT_DIR")
    cands: list[Path] = []
    if env_dir:
        cands.append(Path(env_dir) / name)
    for base in ("/usr/share/fonts", "/usr/local/share/fonts",
                 "/Library/Fonts", str(Path.home() / "Library/Fonts")):
        p = Path(base)
        if p.exists():
            cands.extend(sorted(p.rglob(name))[:1])
    for c in cands:
        if c.exists():
            return str(c)
    return None


def _font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    """DejaVu font at `size` logical px (scaled to canvas px). Falls back to
    PIL's default if DejaVu is not installed (still guarded, still fitted)."""
    key = (kind, size)
    if key in _font_cache:
        return _font_cache[key]
    path = _find_font(_FONT_FILES[kind])
    if path is not None:
        f = ImageFont.truetype(path, size * S)
    else:  # pragma: no cover - DejaVu ships on the CI/dev images
        f = ImageFont.load_default(size=size * S)  # type: ignore[assignment]
    _font_cache[key] = f
    return f


# -- canvas wrapper (all coords logical, scaled by S) ---------------------------

def _canvas_rect(cv: _Canvas, x0, y0, x1, y1, radius=0):
    r = [x0 * cv.s, y0 * cv.s, x1 * cv.s, y1 * cv.s]
    if radius:
        cv.d.rounded_rectangle(r, radius=radius * cv.s)
    else:
        cv.d.rectangle(r)


def frame_rect() -> tuple[int, int, int, int]:
    """OUTER (ink) edge of the frame — logical px. PIL strokes the outline
    INWARD from the given rect, so the ink edge is exactly this rect. The
    pixel-scan boundary: anything outside this rect must be pure BG."""
    return (MARGIN, MARGIN, W - MARGIN, H - MARGIN)


def inner_frame_rect() -> tuple[int, int, int, int]:
    """INNER content rect: inside the frame border — all text must fit here."""
    b = MARGIN + BORDER
    return (b, b, W - b, H - b)


def scan_boundary_rect() -> tuple[int, int, int, int]:
    """Pixel-scan boundary in SCALED px: pure BG is required strictly
    OUTSIDE the frame ink. The frame ink's outermost pixel column/row is
    W*S-1-MARGIN*S inclusive (PIL strokes inward), so BG-only is checked at
    x < MARGIN*S or x > W*S-1-MARGIN*S (exclusive upper bound), etc."""
    s = S
    return (MARGIN * s, MARGIN * s, W * s - MARGIN * s, H * s - MARGIN * s)


class _Canvas:
    def __init__(self, s: int = S):
        self.s = s
        self.img = Image.new("RGB", (W * s, H * s), _rgb(PALETTE["BG"]))
        self.d = ImageDraw.Draw(self.img)
        self.overlay = Image.new("RGBA", (W * s, H * s), (0, 0, 0, 0))
        self.od = ImageDraw.Draw(self.overlay)
        self.boxes: list[tuple[int, int, int, int]] = []   # text bboxes, scaled px
        self.drawn_texts: list[str] = []

    def rect(self, x0, y0, x1, y1, fill=None, outline=None, width=1, radius=0):
        r = [x0 * self.s, y0 * self.s, x1 * self.s, y1 * self.s]
        if radius:
            self.d.rounded_rectangle(r, radius=radius * self.s, fill=fill,
                                     outline=outline, width=max(1, width * self.s))
        else:
            self.d.rectangle(r, fill=fill, outline=outline, width=max(1, width * self.s))

    def tint_rect(self, x0, y0, x1, y1, rgba):
        """Translucent fill (composited at finalize)."""
        self.od.rectangle([x0 * self.s, y0 * self.s, x1 * self.s, y1 * self.s], fill=rgba)

    def line(self, pts, fill, width=1):
        self.d.line([(x * self.s, y * self.s) for x, y in pts],
                    fill=fill, width=max(1, width * self.s))

    def ellipse(self, cx, cy, r, fill=None, outline=None, width=1):
        c = [cx * self.s, cy * self.s]
        rr = r * self.s
        self.d.ellipse([c[0] - rr, c[1] - rr, c[0] + rr, c[1] + rr],
                       fill=fill, outline=outline, width=max(1, width * self.s))

    def text(self, x, y, s, font, fill, anchor=None):
        s = str(s)
        _guard_no_tickers(s)               # HARD RULE: guard every drawn string
        self.d.text((x * self.s, y * self.s), s, font=font, fill=fill, anchor=anchor)
        bb = self.d.textbbox((x * self.s, y * self.s), s, font=font, anchor=anchor)
        self.boxes.append(bb)
        self.drawn_texts.append(s)

    def text_w(self, s, font) -> float:
        return self.d.textlength(str(s), font=font)

    def wrap(self, s, font, max_w_logical) -> list[str]:
        max_w = max_w_logical * self.s
        words = str(s).split()
        lines, cur = [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            if self.d.textlength(trial, font=font) <= max_w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    def finalize(self) -> Image.Image:
        return Image.alpha_composite(self.img.convert("RGBA"), self.overlay).convert("RGB")


# -- fit assertions ---------------------------------------------------------------

def _assert_fit(cv: _Canvas) -> None:
    """Every drawn text bbox must sit inside the inner frame (frame border
    included in the scan margin, so text may touch but not cross the ink)."""
    fx0, fy0, fx1, fy1 = (v * cv.s for v in inner_frame_rect())
    for bb in cv.boxes:
        if bb[0] < fx0 or bb[1] < fy0 or bb[2] > fx1 or bb[3] > fy1:
            raise FaceError(
                f"text overflow past the card frame: bbox {bb} vs inner frame "
                f"{(fx0, fy0, fx1, fy1)} — refused"
            )


def assert_no_ink_outside_frame(img: Image.Image, s: int = S) -> None:
    """Pixel-scan: outside the frame ink the image must be pure BG.

    PIL strokes a `rectangle` outline INWARD from the given rect, so the
    outermost ink pixel is exactly at frame_rect() scaled — the BG-only zone
    is strictly outside [fx0, fx1] x [fy0, fy1] (end-exclusive)."""
    arr = np.asarray(img.convert("RGB"))
    fx0, fy0, fx1, fy1 = (int(v * s) for v in frame_rect())
    bg = np.array(_rgb(PALETTE["BG"]), dtype=np.uint8)
    mask = np.ones(arr.shape[:2], dtype=bool)
    mask[fy0:fy1 + 1, fx0:fx1 + 1] = False      # frame ink edge inclusive
    outside = arr[mask]
    if outside.size and not bool((outside == bg).all()):
        raise FaceError("pixel-scan: ink found outside the card frame — refused")


# -- diagram inputs (anonymized aggregates only) ----------------------------------

def _family_kind(face: dict, diagram: dict) -> str:
    fam = str((diagram or {}).get("family") or face.get("family") or "").lower()
    if "lead" in fam or "lag" in fam:
        return "lead_lag"
    if "hedge" in fam or "duration" in fam:
        return "hedge"
    return "pairs"          # pairs / mean-reversion / anything with a z series


def _series(d: dict, key: str) -> list[float]:
    vals = d.get(key) or []
    out = [float(v) for v in vals]
    if len(out) < 8:
        raise FaceError(f"diagram {key!r}: need >= 8 anonymized aggregate points, got {len(out)}")
    if not all(math.isfinite(v) for v in out):
        raise FaceError(f"diagram {key!r}: non-finite value in aggregate series — refused")
    return out


def _idx(d: dict, key: str, n: int, default: int | None = None) -> int:
    v = d.get(key, default)
    if v is None:
        raise FaceError(f"diagram {key!r}: missing marker index")
    v = int(v)
    if not 0 <= v < n:
        raise FaceError(f"diagram {key!r}: index {v} out of range [0, {n})")
    return v


# -- chart blocks -------------------------------------------------------------------

def _chart_geom(cv: _Canvas, art: tuple[int, int, int, int]):
    return (art[0] + 64, art[1] + 56, art[2] - 28, art[3] - 28)  # x0,y0,x1,y1


def _draw_pairs(cv: _Canvas, chart, d: dict) -> None:
    z = _series(d, "series")
    n = len(z)
    entry_z = float(d.get("entry_z", 2.0) or 2.0)
    if entry_z <= 0:
        raise FaceError("diagram entry_z must be > 0")
    half_life_days = d.get("half_life_days")
    peak_idx = _idx(d, "peak_idx", n, default=int(np.argmax(np.abs(z))))
    if d.get("half_idx") is not None:
        half_idx = _idx(d, "half_idx", n)
    elif half_life_days is not None:
        half_idx = min(n - 1, peak_idx + int(round(float(half_life_days))))
    else:
        half_idx = min(n - 1, peak_idx + max(1, n // 10))

    xs = np.linspace(chart[0], chart[2], n)
    lo = min(min(z), -entry_z) - 0.4
    hi = max(max(z), entry_z) + 0.4
    span = (hi - lo) or 1.0

    def Y(v: float) -> float:
        return chart[3] - (v - lo) / span * (chart[3] - chart[1])

    for g in np.linspace(lo, hi, 5):                      # grid
        cv.line([(chart[0], Y(float(g))), (chart[2], Y(float(g)))], _rgb(PALETTE["GRID"]), 1)
    cv.tint_rect(chart[0], Y(entry_z), chart[2], Y(-entry_z),
                 (*_rgb(PALETTE["RED"]), 38))             # shaded zone between bands
    for v in (entry_z, -entry_z):                          # ±entry_z entry bands
        cv.line([(chart[0], Y(v)), (chart[2], Y(v))], _rgb(PALETTE["RED"]), 3)
    cv.text(chart[2] - 8, Y(entry_z) - 10, f"±{entry_z:g}σ entry",
            _font("mono", 18), _rgb(PALETTE["RED"]), anchor="ra")
    cv.line([(chart[0], Y(0.0)), (chart[2], Y(0.0))], _rgb(PALETTE["INK_DIM"]), 2)  # zero line
    cv.line([(float(x), Y(float(v))) for x, v in zip(xs, z)], _rgb(PALETTE["ACCENT2"]), 3)
    cv.line([(xs[peak_idx], chart[1]), (xs[peak_idx], chart[3])], _rgb(PALETTE["GOLD"]), 2)
    cv.text(min(max(xs[peak_idx], chart[0] + 4), chart[2] - 60), chart[1] + 8,
            "|z| peak", _font("mono", 18), _rgb(PALETTE["GOLD"]))
    hl_lbl = (f"half-life {float(half_life_days):g}d" if half_life_days is not None
              else f"half-decay @{half_idx}")
    cv.text(min(max(xs[half_idx], chart[0] + 4), chart[2] - 120), chart[3] - 26,
            hl_lbl, _font("mono", 18), _rgb(PALETTE["GOLD"]))


def _draw_lead_lag(cv: _Canvas, chart, d: dict) -> None:
    leader = _series(d, "leader")
    n = len(leader)
    lag = int(d.get("lag", 0) or 0)
    if not 0 <= lag < n:
        raise FaceError(f"diagram lag {lag} out of range [0, {n})")
    follower = _series(d, "follower") if d.get("follower") else \
        [leader[max(0, t - lag)] for t in range(n)]      # follower = leader shifted +k
    if len(follower) != n:
        raise FaceError("diagram leader/follower length mismatch")

    xs = np.linspace(chart[0], chart[2], n)
    lo = min(min(leader), min(follower)) - 0.05 * (max(max(leader), max(follower)) - min(min(leader), min(follower)) + 1e-9) - 0.01
    hi = max(max(leader), max(follower)) + 0.05 * (max(max(leader), max(follower)) - min(min(leader), min(follower)) + 1e-9) + 0.01
    span = (hi - lo) or 1.0

    def Y(v: float) -> float:
        return chart[3] - (v - lo) / span * (chart[3] - chart[1])

    for g in np.linspace(lo, hi, 5):
        cv.line([(chart[0], Y(float(g))), (chart[2], Y(float(g)))], _rgb(PALETTE["GRID"]), 1)
    if lag > 0:                                            # shaded lead band x_k..end
        cv.tint_rect(xs[lag], chart[1], chart[2], chart[3], (*_rgb(PALETTE["GOLD"]), 34))
    cv.line([(float(x), Y(float(v))) for x, v in zip(xs, leader)], _rgb(PALETTE["ACCENT"]), 3)
    cv.line([(float(x), Y(float(v))) for x, v in zip(xs, follower)], _rgb(PALETTE["ACCENT2"]), 3)
    cv.text(chart[0] + 8, chart[1] + 8, "leader", _font("mono", 18), _rgb(PALETTE["ACCENT"]))
    cv.text(chart[0] + 8, chart[1] + 30, "follower", _font("mono", 18), _rgb(PALETTE["ACCENT2"]))
    if lag > 0:
        cv.line([(xs[lag], chart[1]), (xs[lag], chart[3])], _rgb(PALETTE["GOLD"]), 3)
        cv.text(min(max(xs[lag], chart[0] + 4), chart[2] - 190), chart[3] - 26,
                f"peak xcorr @ +{lag}d", _font("mono", 18), _rgb(PALETTE["GOLD"]))


def _draw_hedge(cv: _Canvas, chart, d: dict) -> None:
    s = _series(d, "series")
    n = len(s)
    half_life_days = d.get("half_life_days")
    peak_idx = _idx(d, "peak_idx", n, default=int(np.argmax(s)))
    if d.get("half_idx") is not None:
        half_idx = _idx(d, "half_idx", n)
    else:
        pk = s[peak_idx]
        half_idx = next((t for t in range(peak_idx + 1, n) if s[t] <= pk / 2.0), None)
        if half_idx is None:
            half_idx = n - 1

    xs = np.linspace(chart[0], chart[2], n)
    lo, hi = min(s), max(s)
    span = (hi - lo) or 1.0
    lo -= 0.05 * span
    hi += 0.05 * span
    span = hi - lo

    def Y(v: float) -> float:
        return chart[3] - (v - lo) / span * (chart[3] - chart[1])

    for g in np.linspace(lo, hi, 5):
        cv.line([(chart[0], Y(float(g))), (chart[2], Y(float(g)))], _rgb(PALETTE["GRID"]), 1)
    cv.line([(float(x), Y(float(v))) for x, v in zip(xs, s)], _rgb(PALETTE["ACCENT"]), 3)
    cv.line([(xs[peak_idx], chart[1]), (xs[peak_idx], chart[3])], _rgb(PALETTE["GOLD"]), 2)
    cv.text(min(max(xs[peak_idx], chart[0] + 4), chart[2] - 60), chart[1] + 8,
            "peak", _font("mono", 18), _rgb(PALETTE["GOLD"]))
    # peak -> half-decay bracket
    yb = chart[1] + 34
    cv.line([(xs[peak_idx], yb), (xs[half_idx], yb)], _rgb(PALETTE["GOLD"]), 2)
    cv.line([(xs[peak_idx], yb), (xs[peak_idx], yb + 8)], _rgb(PALETTE["GOLD"]), 2)
    cv.line([(xs[half_idx], yb), (xs[half_idx], yb + 8)], _rgb(PALETTE["GOLD"]), 2)
    lbl = (f"half-life {float(half_life_days):g}d" if half_life_days is not None
           else f"half-decay @{half_idx}")
    cv.text(min(max((xs[peak_idx] + xs[half_idx]) / 2, chart[0] + 4), chart[2] - 130),
            yb + 12, lbl, _font("mono", 18), _rgb(PALETTE["GOLD"]))


# -- layout -------------------------------------------------------------------------

def _short_card_id(card_id: str) -> str:
    cid = str(card_id)
    hexpart = cid.split("-", 1)[1] if "-" in cid else cid
    hexpart = hexpart.split(":")[0].lower()   # lower: guard-safe (no ALL-CAPS token)
    return "card-" + hexpart[:4] + "…" + hexpart[-4:]


def _chip(cv: _Canvas, x: int, y: int, label: str) -> tuple[int, int]:
    """One chip; returns (next_x, y) in logical px. Caller handles wrapping
    with the same width formula: int(text_w/s) + 48."""
    f = _font("mono", 20)
    w = int(cv.text_w(label, f) / cv.s) + 48
    cv.rect(x, y, x + w, y + 40, fill=_rgb(PALETTE["PANEL"]),
            outline=_rgb(PALETTE["FRAME"]), width=2, radius=9)
    cv.text(x + 18, y + 20, label, f, _rgb(PALETTE["INK"]), anchor="lm")
    return x + w + 12, y


class CardImageRenderer:
    """Renders the anonymized card face dict to PNG bytes (2x + 1x)."""

    def __init__(self, s: int = S):
        self.s = s

    def render(self, face: dict, diagram: dict | None = None) -> tuple[bytes, bytes]:
        """Returns (png_2x, png_1x). Raises FaceError on any guard/fit/pixel
        violation or on any ticker-like token in a drawn string."""
        tier = str(face.get("tier", "bronze")).lower()
        if tier not in TIER_ACCENTS:
            raise FaceError(f"unknown tier {tier!r}")
        tc = TIER_ACCENTS[tier]
        cv = _Canvas(self.s)
        self.canvas = cv
        F = frame_rect()
        x0, x1 = F[0] + PAD, F[2] - PAD
        y = F[1] + PAD

        # frame + panel + tier title band
        cv.rect(*F, fill=_rgb(PALETTE["PANEL"]), outline=None)
        _canvas_rect(cv, *F, radius=0)                      # panel body
        cv.d.rectangle([F[0] * cv.s, F[1] * cv.s, F[2] * cv.s, F[3] * cv.s],
                       outline=_rgb(tc["accent"]),
                       width=max(1, BORDER * cv.s))          # tier frame (inset 0)
        band_h = 92
        cv.rect(x0, y, x1, y + band_h, fill=_rgb(tc["bg"]),
                outline=_rgb(tc["dk"]), width=2, radius=10)
        cv.text(x0 + 22, y + band_h // 2, "SIFT survivor card", _font("sans_b", 34),
                _rgb(PALETTE["INK"]), anchor="lm")
        tier_label = f"{tier} tier"
        tf = _font("sans_b", 24)
        tw = cv.text_w(tier_label, tf) / self.s + 28
        cv.rect(x1 - tw, y + 20, x1 - 12, y + band_h - 20, fill=_rgb(tc["accent"]),
                radius=8)
        cv.text(x1 - tw + 14, y + band_h // 2, tier_label, tf,
                _rgb(PALETTE["BG"]), anchor="lm")
        y += band_h + 18

        # chips row (family, scope, sector aggregate, survivor, ghost)
        chips = [
            str(face.get("family", "?")),
            str(face.get("scope_id", "?")),
            str((face.get("sector_aggregate") or {}).get("bucket", "aggregate")),
            str(face.get("survivor_class", "?")),
        ]
        if face.get("ghost_no"):
            chips.append(f"ghost #{face['ghost_no']}")
        cx, cy = x0, y
        for c in chips:
            f = _font("mono", 20)
            w = int(cv.text_w(c, f) / self.s) + 48
            if cx + w > x1:                       # wrap to next row
                cx = x0
                cy += 52
            cx, cy = _chip(cv, cx, cy, c)
        y = cy + 58

        # art window
        art_h = 560
        art = (x0, y, x1, y + art_h)
        cv.rect(*art, fill=_rgb(PALETTE["ARTBG"]), outline=_rgb(PALETTE["FRAME"]),
                width=2, radius=8)
        cv.text(art[0] + 20, art[1] + 16, ART_LABEL, _font("mono", 19),
                _rgb(PALETTE["INK_DIM"]))
        chart = _chart_geom(cv, art)
        kind = _family_kind(face, diagram or {})
        d = dict(diagram or {})
        if kind == "lead_lag":
            _draw_lead_lag(cv, chart, d)
        elif kind == "hedge":
            _draw_hedge(cv, chart, d)
        else:
            _draw_pairs(cv, chart, d)
        y += art_h + 26

        # gauntlet block: pips + survival stats
        cv.text(x0, y, "gauntlet — survival stages", _font("sans_b", 24),
                _rgb(PALETTE["INK"]))
        y += 42
        passed = set(str(s).lower() for s in
                     (face.get("gauntlet_stats") or {}).get("stages_passed", []))
        px = x0 + 16
        for st in STAGES:
            ok = st.lower() in passed
            cv.ellipse(px, y + 14, 14,
                       fill=_rgb(PALETTE["ACCENT"]) if ok else _rgb(PALETTE["GRID"]),
                       outline=_rgb(PALETTE["FRAME"]), width=2)
            cv.text(px, y + 42, st, _font("mono", 18), _rgb(PALETTE["INK_DIM"]), anchor="ma")
            px += 76
        y += 78
        gs = face.get("gauntlet_stats") or {}
        lines: list[str] = []
        if gs.get("s2_p_value") is not None:
            lines.append(f"s2 perm p = {float(gs['s2_p_value']):.4f}")
        if gs.get("s3_dsr") is not None:
            lines.append(f"s3 dsr = {float(gs['s3_dsr']):.2f}")
        if gs.get("s5_full_depth_t") is not None:
            lines.append(f"s5 full-depth t = {float(gs['s5_full_depth_t']):.2f}"
                         f" ({gs.get('s5_status')})")
        spec = face.get("specifics") or {}
        if spec:
            lines.append(" · ".join(f"{str(k).replace('_', ' ')} {v}" for k, v in spec.items()))
        for ln in lines:
            cv.text(x0, y, ln, _font("mono", 22), _rgb(PALETTE["INK"]))
            y += 32

        # -- newcomer-friendly blocks: explainer + gauge + health ---------
        # All strings come from the face dict, i.e. from the copy builders in
        # sift/cards/copy.py (banned-phrase fail-closed there); every string
        # drawn here passes _guard_no_tickers via cv.text. Jargon (z-score,
        # half-life, DSR, tier, health) stays behind the glossary tooltips
        # (face["glossary"]) — not drawn in the main view.
        expl = face.get("explainer")
        if expl:
            ef = _font("sans", 21)
            for ln in cv.wrap(str(expl), ef, (x1 - x0)):
                cv.text(x0, y, ln, ef, _rgb(PALETTE["INK"]))
                y += 28
            y += 10
        gauge = face.get("gauge") or {}
        if gauge.get("band"):
            gtxt = f"magnitude: {gauge.get('label', gauge['band'])}"
            if gauge.get("magnitude_z") is not None:
                gtxt += f"  (|z| = {float(gauge['magnitude_z']):.2f})"
            cv.text(x0, y, gtxt, _font("mono", 21), _rgb(PALETTE["ACCENT2"]))
            y += 32
        health = face.get("health") or {}
        if health.get("phrase"):
            htxt = f"health: {health['phrase']}"
            cv.text(x0, y, htxt, _font("mono", 21), _rgb(PALETTE["GOLD"]))
            y += 32

        # footer: watermark (handle + ts) + disclaimer, anchored at frame bottom
        wm = face.get("watermark") or {}
        ts_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(wm.get("ts", 0))))
        wm_line = f"shared by @{wm.get('handle', '?')} · {ts_iso}"
        disc = str(face.get("disclaimer", ""))
        df = _font("mono", 18)
        cid_f = _font("mono", 18)
        cid_str = _short_card_id(str(face.get("card_id", "")))
        # the disclaimer and the short card id share the bottom line: wrap the
        # disclaimer to the width left of the card-id so nothing can overrun
        cid_w = int(cv.text_w(cid_str, cid_f) / self.s) + 24
        dlines = cv.wrap(disc, df, (x1 - x0 - cid_w))
        lh = 24
        y_disc = F[3] - PAD - lh * len(dlines)
        cv.text(x0, y_disc - 34, wm_line, _font("mono_b", 22), _rgb(PALETTE["GOLD"]))
        for i, dl in enumerate(dlines):
            cv.text(x0, y_disc + i * lh, dl, df, _rgb(PALETTE["INK_DIM"]))
        cv.text(x1, y_disc + (len(dlines) - 1) * lh + lh // 2, cid_str, cid_f,
                _rgb(PALETTE["INK_DIM"]), anchor="rm")

        # fit assert + pixel scan before saving
        img = cv.finalize()
        _assert_fit(cv)
        assert_no_ink_outside_frame(img, self.s)

        buf2 = io.BytesIO()
        img.save(buf2, format="PNG", optimize=True)
        png2x = buf2.getvalue()
        png1x_buf = io.BytesIO()
        img.resize((W, H), Image.LANCZOS).save(png1x_buf, format="PNG", optimize=True)
        return png2x, png1x_buf.getvalue()


def render_face_png(face: dict, diagram: dict | None = None, s: int = S) -> tuple[bytes, bytes]:
    """Render the card face dict (+ per-family diagram inputs) to PNG bytes.
    Returns (png_2x, png_1x)."""
    return CardImageRenderer(s=s).render(face, diagram)


def attach_card_image(face: dict, diagram: dict | None = None, s: int = S) -> dict:
    """Face dict + optional `image` field (base64 PNGs, JSON-serializable).
    The JSON artifact is otherwise unchanged."""
    png2x, png1x = render_face_png(face, diagram=diagram, s=s)
    out = dict(face)
    out["image"] = {
        "content_type": "image/png",
        "encoding": "base64",
        "scale": s,
        "width": W,
        "height": H,
        "png_2x_b64": base64.b64encode(png2x).decode("ascii"),
        "png_1x_b64": base64.b64encode(png1x).decode("ascii"),
        "bytes_2x": len(png2x),
        "bytes_1x": len(png1x),
    }
    return out


__all__ = [
    "S", "W", "H", "PALETTE", "TIER_ACCENTS", "ART_LABEL",
    "CardImageRenderer", "render_face_png", "attach_card_image",
    "assert_no_ink_outside_frame", "frame_rect", "inner_frame_rect",
]
