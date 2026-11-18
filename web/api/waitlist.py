#!/usr/bin/env python3
"""SIFT landing page server + beta waitlist endpoint stub.

Stdlib only — no framework, no dependencies.

Serves the static site in web/ and handles:

    POST /api/waitlist   JSON {"email": "..."} or form-encoded email=...
                         -> appends to data/waitlist.csv (created on first signup)

Run:

    python3 web/api/waitlist.py            # serves http://127.0.0.1:8080/
    PORT=9000 python3 web/api/waitlist.py  # custom port

On a pure static host (GitHub Pages, Netlify, Cloudflare Pages, ...) the page
works as-is; the waitlist form then needs any endpoint mapped to
POST /api/waitlist that accepts {"email": ...}. Until then, run this stub.
"""

import csv
import json
import os
import re
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
WEB_DIR = ROOT / "web"
DATA_DIR = Path(os.environ.get("SIFT_DATA_DIR", ROOT / "data"))
WAITLIST_CSV = DATA_DIR / "waitlist.csv"
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]{2,}$")
MAX_BODY = 16 * 1024  # 16 KiB is far more than an email needs
# spreadsheet formula prefixes — neutralize CSV/formula injection on export
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
_WRITE_LOCK = threading.Lock()

# -- onboarding copy (built ONLY in the copy builders, fail-closed) -----------
# The waitlist endpoint is stdlib-only and must stay dependency-free for
# static hosting, so it imports the copy builders lazily and only when the
# v4.1 flag is on; with the flag off (kill-switch SIFT_CARDS_V4_1=0) the
# endpoint serves the plain static strings below (fail-closed on copy,
# never on availability of the waitlist itself).
_ONBOARDING_FALLBACK = {
    "statement": ("SIFT shows which relationships in your research held up "
                  "under stress. It never tells you what to buy or sell; "
                  "nothing here is advice. You make the calls — we make the "
                  "evidence legible."),
    "tos": ("Cards are not an investment, have no resale value, and confer "
            "no right to any profit. They are non-transferable. SIFT does "
            "not trade. By continuing you accept these terms."),
}


def onboarding_payload() -> dict:
    """The onboarding payload shown at signup / first card: the plain
    statement + the day-one ToS acceptance line + the v4.2 §5 candidates &
    decks line, built ONLY by the copy builders in sift/cards/copy.py and
    sift/cards/decks.py when cards_v4_1 is on (banned-phrase fail-closed
    there). Flag off -> CopyError/DeckCopyError propagates and the caller
    falls back to the static strings above."""
    sys.path.insert(0, str(ROOT))  # repo root importable (no-op under pytest)
    from sift.cards import copy as card_copy
    from sift.cards import decks as deck_copy
    return {
        "statement": card_copy.onboarding_statement(),
        "tos_acceptance": card_copy.tos_acceptance_text(),
        "hit_loop_hint": card_copy.hit_loop_hint(),
        "cards_are_candidates": deck_copy.cards_are_candidates_line(),
        "source": "sift.cards.copy + sift.cards.decks (cards_v4_1)",
    }


def onboarding_payload_safe() -> dict:
    """onboarding_payload() with the fail-closed fallback: never raises."""
    try:
        return onboarding_payload()
    except Exception:
        return {
            "statement": _ONBOARDING_FALLBACK["statement"],
            "tos_acceptance": _ONBOARDING_FALLBACK["tos"],
            "hit_loop_hint": ("Your saved connection fired — check it in your "
                              "own charting tool; SIFT does not trade."),
            "cards_are_candidates": ("No single card is an instant win. Cards "
                                     "are evidence about connections; decks "
                                     "are the tested portfolios. You make all "
                                     "trading decisions; SIFT does not trade."),
            "source": "static fallback (cards_v4_1 off)",
        }

MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".json": "application/json; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
}


def save_email(email: str, tos_accepted: bool = False) -> bool:
    """Append email to waitlist CSV. Returns False if it was a duplicate.

    `tos_accepted` records day-one ToS acceptance (beta plan: "not an
    investment / no resale value / no right to profit" is part of account
    creation, live from day one). The waitlist form carries a mandatory
    checkbox; the endpoint refuses signups without it (fail closed).

    Serialized with a lock: the server is threaded, and the
    exists-check/read/append sequence must be atomic to avoid duplicated
    header rows or duplicate signups under concurrent POSTs.
    """
    # Neutralize CSV/formula injection: a leading =, +, -, @, tab or CR would
    # be interpreted as a formula when the CSV is opened in a spreadsheet.
    if email[:1] in FORMULA_PREFIXES:
        email = "'" + email
    with _WRITE_LOCK:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        exists = WAITLIST_CSV.exists() and WAITLIST_CSV.stat().st_size > 0
        # Duplicate check against existing rows (small file at beta scale).
        if exists:
            with WAITLIST_CSV.open(newline="", encoding="utf-8") as f:
                for row in csv.reader(f):
                    if len(row) >= 1 and row[0].lower() == email.lower():
                        return False
        with WAITLIST_CSV.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(["email", "ts_utc", "source", "tos_accepted"])
            w.writerow([email, datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "landing", "yes" if tos_accepted else "no"])
        return True


class Handler(BaseHTTPRequestHandler):
    server_version = "SIFTWaitlist/0.1"

    # ---- helpers ----
    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        try:
            body = path.read_bytes()
        except OSError:
            self.send_error(404, "Not found")
            return
        mime = MIME.get(path.suffix.lower(), "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # quieter logs
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # ---- routes ----
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send_file(WEB_DIR / "index.html")
            return
        # onboarding copy endpoint (first-run / account creation surface):
        # the plain statement + day-one ToS, built by the copy builders.
        if path == "/api/onboarding":
            self._send_json(200, {"ok": True, "onboarding": onboarding_payload_safe()})
            return
        # static files, confined to web/
        rel = Path(path.lstrip("/"))
        if rel.is_absolute() or ".." in rel.parts:
            self.send_error(400, "Bad path")
            return
        target = (WEB_DIR / rel).resolve()
        if WEB_DIR.resolve() not in target.parents and target != WEB_DIR.resolve():
            self.send_error(403, "Forbidden")
            return
        # never serve source/config files that live under web/ (e.g. the endpoint stub)
        if target.suffix.lower() in (".py", ".pyc", ".csv", ".json", ".md"):
            self.send_error(404, "Not found")
            return
        if target.is_file():
            self._send_file(target)
        else:
            self.send_error(404, "Not found")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/waitlist":
            self._send_json(404, {"ok": False, "error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY:
            self._send_json(400, {"ok": False, "error": "Invalid body"})
            return
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()

        email = None
        tos_accepted = False
        if ctype == "application/json":
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = None
            if not isinstance(data, dict):
                self._send_json(400, {"ok": False, "error": "Invalid body"})
                return
            raw_email = data.get("email", "")
            if not isinstance(raw_email, str):
                self._send_json(400, {"ok": False, "error": "Please provide a valid email address."})
                return
            email = raw_email.strip()
            # day-one ToS acceptance (fail closed): a signup without it is
            # refused — "not an investment / no resale value / no right to
            # profit" is part of account creation from day one.
            tos_accepted = data.get("tos_accepted") is True
        else:  # form-encoded fallback
            from urllib.parse import parse_qs
            qs = parse_qs(raw)
            email = qs.get("email", [""])[0].strip()
            tos_accepted = qs.get("tos_accepted", [""])[0].strip().lower() in (
                "1", "true", "yes", "on")

        if not EMAIL_RE.match(email):
            self._send_json(400, {"ok": False, "error": "Please provide a valid email address."})
            return
        if not tos_accepted:
            self._send_json(400, {"ok": False, "error":
                                  "Please accept the terms to join the waitlist."})
            return

        added = save_email(email, tos_accepted=tos_accepted)
        if added:
            self._send_json(200, {"ok": True, "message": "Added to the beta waitlist."})
        else:
            self._send_json(200, {"ok": True, "message": "You're already on the list."})


def main() -> None:
    port = int(os.environ.get("PORT", "8080"))
    host = os.environ.get("HOST", "127.0.0.1")
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"SIFT landing page:  http://{host}:{port}/  (web/ served statically)")
    print(f"Waitlist endpoint:  POST http://{host}:{port}/api/waitlist  -> {WAITLIST_CSV}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
