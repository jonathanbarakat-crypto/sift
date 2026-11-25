"""Thin FastAPI server for the SIFT Gate-1 invited beta application.

Hardened architecture wrapping `GauntletRunner` + `CardLayer` + `Ledger`:
- Physical DB isolation:
    * `auth.db`      — invites, sessions, `beta_user` (ONLY file with email)
    * `telemetry.db` — append-only `telemetry_event` (zero email PII)
    * `app.db`       — `beta_run` + `saved_connection` (shared WAL connection)
    * `ledger.db` & `cards/` — trial ledger & card layer (WAL mode)
- Non-blocking concurrency:
    * `_fast_pool`   — serves auth/session, waitlist, telemetry, run/trial
                       reads, and saved connections in milliseconds even while
                       gauntlet runs execute.
    * `_engine_pool` — exclusively owns writable `Ledger` + `CardLayer` and
                       runs `GauntletRunner` CPU work.
    * `scrypt` password KDF runs off both DB threads via `asyncio.to_thread`
      with constant-time verification for unknown emails.
- Fail-closed compliance:
    * Every JSON response, error handler, and SSE frame is checked by
      `guard_api_payload` (including active warehouse symbols). Blocked
      responses return a static 500 message — NEVER echoing the blocked token.
- Security:
    * Sliding-window `RateLimiter` on auth, waitlist, and mining endpoints,
      plus per-user concurrency cap on mining runs.
    * Strict `Content-Type: application/json` on JSON mutation endpoints (415).
    * Re-checks `tos_version` on every authenticated request.
    * `POST /api/auth/delete-account` permanently erases user email & sessions.

Run:
    uvicorn sift.web.app:app --host 127.0.0.1 --port 8080
    python3 -m sift.web.app
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import csv
import hashlib
import json
import math
import os
import queue
import secrets
import sqlite3
import threading
from collections.abc import Collection
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from sift.cards import CardError
from sift.cards import copy as card_copy
from sift.cards.face import FaceError
from sift.cards.notify import MODE_DIGEST, NotifyError
from sift.cards.service import CardLayer, LayerError
from sift.cards.vault import BASE_CAPACITY, VaultError
from sift.decks import copy as deck_copy
from sift.gauntlet import config as cfg
from sift.registry import Ledger
from sift.web.auth import (
    DUMMY_SCRYPT_HASH,
    EMAIL_RE,
    TOS_VERSION,
    AuthError,
    AuthStore,
    BetaUser,
    RateLimiter,
    hash_password,
    verify_password,
)
from sift.web.compliance import ComplianceError, guard_api_payload
from sift.web.connections import ConnectionError as ConnError
from sift.web.connections import ConnectionStore
from sift.web.mining import (
    ALLOWED_CLOSE_Z,
    ALLOWED_FORMATIONS,
    ALLOWED_OPEN_Z,
    ALLOWED_SCOPES,
    MiningError,
    MiningManager,
    MiningStore,
    PairCatalog,
    _safe_verdict_copy,
    _stage_to_public,
)
from sift.web.telemetry import TelemetryStore

ROOT = Path(__file__).resolve().parent.parent.parent
WEB_DIR = ROOT / "web"
SESSION_COOKIE = "sift_session"

# Waitlist CSV formula-injection neutralization (matching web/api/waitlist.py)
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
_WAITLIST_LOCK = threading.Lock()

# Static serving allowlist & denylist (matching web/api/waitlist.py)
_FORBIDDEN_STATIC_EXTS = frozenset({
    ".py", ".pyc", ".csv", ".json", ".md", ".db", ".sqlite", ".db-wal", ".db-shm"
})
_STATIC_MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
}

# Day-one privacy note from docs/beta-plan.md §5
PRIVACY_NOTE = (
    "We store your email and an anonymous ID; product events are stored under "
    "the anonymous ID; reports are aggregate-only, ranges and qualitative "
    "findings at this small n; we never publish per-user data; you can "
    "request deletion."
)

# Request fields forbidden on mining runs (S5 is automatic; raw symbols/prices forbidden)
_FORBIDDEN_RUN_INPUT_KEYS = frozenset({
    "skip_s5",
    "opt_out_s5",
    "disable_s5",
    "no_s5",
    "tickers",
    "ticker",
    "symbols",
    "symbol",
    "prices",
    "raw_bars",
    "legs",
})


def _onboarding_payload(handle: str | None = None) -> dict[str, Any]:
    """Build onboarding copy ONLY from sift/cards/copy.py + sift/decks/copy.py
    when cards_v4_1 / decks_v4_2 are enabled; fall back to the verified static
    clauses when the kill-switch is off (preserving GET /api/onboarding
    availability)."""
    try:
        return {
            "statement": card_copy.onboarding_statement(handle=handle),
            "tos_acceptance": card_copy.tos_acceptance_text(handle=handle),
            "hit_loop_hint": card_copy.hit_loop_hint(handle=handle),
            "cards_are_candidates": deck_copy.cards_are_candidates_line(handle=handle),
            "glossary": card_copy.glossary(handle=handle),
            "privacy_note": PRIVACY_NOTE,
            "source": "sift.cards.copy + sift.decks.copy (cards_v4_1 + decks_v4_2)",
        }
    except (card_copy.CopyError, deck_copy.DeckCopyError):
        return {
            "statement": (
                "SIFT shows which relationships in your research held up "
                "under stress. It never tells you what to buy or sell; "
                "nothing here is advice. You make the calls — we make the "
                "evidence legible."
            ),
            "tos_acceptance": (
                "Cards are not an investment, have no resale value, and confer "
                "no right to any profit. They are non-transferable. SIFT does "
                "not trade. By continuing you accept these terms."
            ),
            "hit_loop_hint": (
                "Your saved connection fired — check it in your own charting "
                "tool; SIFT does not trade."
            ),
            "cards_are_candidates": (
                "No single card is an instant win. Cards are evidence about "
                "connections; decks are the tested portfolios. You make all "
                "trading decisions; SIFT does not trade."
            ),
            "glossary": {},
            "privacy_note": PRIVACY_NOTE,
            "source": "static fallback (cards_v4_1 off)",
        }


def _save_waitlist_csv(data_dir: Path, email: str, tos_accepted: bool) -> bool:
    """Append email to data_dir/waitlist.csv (same schema & safety as web/api/waitlist.py)."""
    if email[:1] in _FORMULA_PREFIXES:
        email = "'" + email
    csv_path = data_dir / "waitlist.csv"
    with _WAITLIST_LOCK:
        data_dir.mkdir(parents=True, exist_ok=True)
        exists = csv_path.exists() and csv_path.stat().st_size > 0
        if exists:
            with csv_path.open(newline="", encoding="utf-8") as f:
                for row in csv.reader(f):
                    if len(row) >= 1 and row[0].lower() == email.lower():
                        return False
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(["email", "ts_utc", "source", "tos_accepted"])
            w.writerow([
                email,
                datetime.now(UTC).isoformat(timespec="seconds"),
                "landing",
                "yes" if tos_accepted else "no",
            ])
        return True


def _safe_json(
    payload: dict[str, Any],
    status_code: int = 200,
    *,
    extra_symbols: Collection[str] | None = None,
) -> JSONResponse:
    """Serialize JSON after fail-closed compliance inspection.

    NEVER echoes offending tokens, tickers, or phrases in the 500 body.
    """
    try:
        guard_api_payload(payload, extra_symbols=extra_symbols)
    except ComplianceError:
        return JSONResponse(
            content={"ok": False, "error": ComplianceError.public_message},
            status_code=500,
            headers={"Cache-Control": "no-store"},
        )
    return JSONResponse(
        content=payload,
        status_code=status_code,
        headers={"Cache-Control": "no-store"},
    )


def _enable_wal(con: sqlite3.Connection) -> None:
    con.execute("PRAGMA journal_mode = WAL")
    con.execute("PRAGMA busy_timeout = 10000")


class BetaRuntime:
    """Two-pool runtime decoupling fast auth/app/telemetry I/O from gauntlet compute.

    - `_fast_pool` (1 thread): owns `auth.db`, `telemetry.db`, shared WAL
      `app.db` (`beta_run` + `saved_connection`), and a read-only WAL
      connection to `ledger.db` for non-blocking stage reads.
    - `_engine_pool` (1 thread): owns writable `ledger.db` + `CardLayer` and
      executes CPU-bound `GauntletRunner` runs.
    """

    def __init__(
        self,
        *,
        data_dir: str | Path,
        warehouse_dir: str | Path | None = None,
        card_secret: bytes | None = None,
        allow_synthetic_examples: bool | None = None,
        s2_B: int = cfg.S2_B,
        rate_limit_enabled: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.warehouse_dir = Path(warehouse_dir) if warehouse_dir else None
        self.card_secret = (
            card_secret
            if card_secret is not None
            else (
                os.environ.get("SIFT_CARD_HMAC_SECRET", "").encode("utf-8")
                or secrets.token_bytes(32)
            )
        )
        self.allow_synthetic_examples = allow_synthetic_examples
        self.s2_B = int(s2_B)
        self.limiter = RateLimiter(enabled=rate_limit_enabled)

        self._fast_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sift-beta-fast"
        )
        self._engine_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sift-beta-engine"
        )
        self._closed = False

        # Owned by _fast_pool:
        self.auth: AuthStore
        self.telemetry: TelemetryStore
        self._app_con: sqlite3.Connection
        self.connections: ConnectionStore
        self.runs: MiningStore
        self._ledger_ro: sqlite3.Connection

        # Owned by _engine_pool:
        self.ledger: Ledger
        self.card_layer: CardLayer
        self.catalog: PairCatalog
        self.mining: MiningManager

        self.engine_call(self._init_engine)
        self.fast_call(self._init_fast)
        self.engine_call(self._wire_mining)

    def _init_engine(self) -> None:
        self.ledger = Ledger(self.data_dir / "ledger.db")
        _enable_wal(self.ledger.con)
        self.card_layer = CardLayer(
            self.ledger,
            self.data_dir / "cards",
            card_secret=self.card_secret,
        )
        for comp in (
            self.card_layer.cards,
            self.card_layer.vault,
            self.card_layer.shares,
            self.card_layer.notify,
            self.card_layer.events,
            self.card_layer.decks,
        ):
            _enable_wal(comp.con)
        self.catalog = PairCatalog(
            warehouse_dir=self.warehouse_dir,
            allow_synthetic_examples=self.allow_synthetic_examples,
        )

    def _init_fast(self) -> None:
        self.auth = AuthStore(self.data_dir / "auth.db")
        self.telemetry = TelemetryStore(self.data_dir / "telemetry.db")
        app_db = self.data_dir / "app.db"
        self._app_con = sqlite3.connect(str(app_db), timeout=10.0)
        self._app_con.row_factory = sqlite3.Row
        _enable_wal(self._app_con)
        self.connections = ConnectionStore(app_db, con=self._app_con)
        self.runs = MiningStore(app_db, con=self._app_con)
        # Mark any orphaned running rows from a previous crash as failed.
        self.runs.fail_orphaned_runs()
        # Fast read-only WAL handle onto ledger.db (never blocks behind writes).
        ledger_uri = f"file:{(self.data_dir / 'ledger.db').resolve()}?mode=ro"
        self._ledger_ro = sqlite3.connect(ledger_uri, uri=True, timeout=10.0)
        self._ledger_ro.row_factory = sqlite3.Row
        self._ledger_ro.execute("PRAGMA busy_timeout = 10000")

    def _wire_mining(self) -> None:
        self.mining = MiningManager(
            ledger=self.ledger,
            card_layer=self.card_layer,
            telemetry=self.telemetry,
            runs=self.runs,
            catalog=self.catalog,
            commitment_secret=self.card_secret,
            s2_B=self.s2_B,
            fast_call=self.fast_call,
        )

    # -- fast pool helpers (auth / telemetry / app.db / RO ledger) -----------

    def fast_submit(self, fn, *args, **kwargs) -> concurrent.futures.Future:
        return self._fast_pool.submit(fn, *args, **kwargs)

    def fast_call(self, fn, *args, **kwargs):
        return self.fast_submit(fn, *args, **kwargs).result()

    async def afast(self, fn, *args, **kwargs):
        return await asyncio.wrap_future(self.fast_submit(fn, *args, **kwargs))

    def read_stages_fast(self, trial_id: str) -> list[dict[str, Any]]:
        """Read trial stages on _fast_pool via WAL read-only connection."""
        rows = self._ledger_ro.execute(
            "SELECT * FROM stage_result WHERE trial_id=? ORDER BY stage",
            (trial_id,),
        ).fetchall()
        return [_stage_to_public(r) for r in rows]

    def read_trial_fast(self, trial_id: str) -> sqlite3.Row | None:
        return self._ledger_ro.execute(
            "SELECT trial_id, family, scope_id, provenance, registered_at,"
            " verdict, survivor_class, discoverer_handle"
            " FROM trial WHERE trial_id=?",
            (trial_id,),
        ).fetchone()

    # -- engine pool helpers (Ledger writes / CardLayer / GauntletRunner) ----

    def engine_submit(self, fn, *args, **kwargs) -> concurrent.futures.Future:
        return self._engine_pool.submit(fn, *args, **kwargs)

    def engine_call(self, fn, *args, **kwargs):
        return self.engine_submit(fn, *args, **kwargs).result()

    async def aengine(self, fn, *args, **kwargs):
        return await asyncio.wrap_future(self.engine_submit(fn, *args, **kwargs))

    # Backwards-compatible alias for tests calling rt.call(rt.auth.*) / rt.telemetry.*
    def call(self, fn, *args, **kwargs):
        fast_objs = (self.auth, self.telemetry, self.connections, self.runs)
        owner = getattr(fn, "__self__", None)
        if (
            owner in fast_objs
            or any(a in fast_objs for a in args)
            or any(v in fast_objs for v in kwargs.values())
        ):
            return self.fast_call(fn, *args, **kwargs)
        return self.engine_call(fn, *args, **kwargs)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        def _close_engine() -> None:
            for store in (self.card_layer, self.ledger):
                with contextlib.suppress(OSError, RuntimeError):
                    store.close()

        def _close_fast() -> None:
            for store in (
                self._ledger_ro,
                self._app_con,
                self.telemetry,
                self.auth,
            ):
                with contextlib.suppress(OSError, RuntimeError):
                    store.close()

        try:
            self._engine_pool.submit(_close_engine).result(timeout=15.0)
        finally:
            self._engine_pool.shutdown(wait=True)
        try:
            self._fast_pool.submit(_close_fast).result(timeout=10.0)
        finally:
            self._fast_pool.shutdown(wait=True)


def _extract_token(request: Request) -> str | None:
    auth_hdr = request.headers.get("authorization", "").strip()
    if auth_hdr.lower().startswith("bearer "):
        tok = auth_hdr[7:].strip()
        if tok:
            return tok
    cookie_tok = request.cookies.get(SESSION_COOKIE)
    return cookie_tok.strip() if cookie_tok else None


def _client_ip(request: Request) -> str:
    if request.client and request.client.host:
        return str(request.client.host)
    return "unknown"


def _cookie_secure(request: Request) -> bool:
    if os.environ.get("SIFT_ENV", "dev").strip().lower() == "production":
        return True
    if os.environ.get("SIFT_COOKIE_SECURE", "0") in ("1", "true", "True", "yes"):
        return True
    return request.url.scheme == "https"


def _require_json_content_type(request: Request) -> None:
    """Enforce `application/json` on JSON mutation routes (blocks simple
    cross-origin `text/plain` CSRF requests even from sibling subdomains)."""
    ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    if ctype != "application/json":
        raise HTTPException(
            status_code=415, detail="Content-Type must be application/json."
        )


def _user_dict(user: BetaUser) -> dict[str, Any]:
    return {
        "user_id": user.user_id,
        "beta": user.beta,
        "tos_accepted": user.tos_version == TOS_VERSION,
        "tos_version": user.tos_version,
        "kill_rate_opt_in": user.kill_rate_opt_in,
        "created_at": user.created_at,
    }


def _card_summary(card, vault_state: str | None) -> dict[str, Any]:
    return {
        "card_id": card.card_id,
        "trial_id": card.trial_id,
        "family": card.family,
        "scope_id": card.scope_id,
        "survivor_class": card.survivor_class,
        "kind": card.kind,
        "ghost_no": card.ghost_no if (card.ghost_no or 0) > 0 else None,
        "tier": card.tier,
        "decayed_from": card.decayed_from,
        "minted_at": card.minted_at,
        "vault_state": vault_state,
    }


def create_app(
    *,
    data_dir: str | Path | None = None,
    warehouse_dir: str | Path | None = None,
    card_secret: bytes | None = None,
    allow_synthetic_examples: bool | None = None,
    s2_B: int = cfg.S2_B,
    rate_limit_enabled: bool = True,
) -> FastAPI:
    """Create a configured FastAPI application for the SIFT Gate-1 beta."""
    resolved_data = Path(
        data_dir if data_dir is not None else os.environ.get("SIFT_DATA_DIR", ROOT / "data")
    )
    resolved_wh = (
        warehouse_dir
        if warehouse_dir is not None
        else os.environ.get("SIFT_WAREHOUSE_DIR")
    )
    runtime = BetaRuntime(
        data_dir=resolved_data,
        warehouse_dir=resolved_wh,
        card_secret=card_secret,
        allow_synthetic_examples=allow_synthetic_examples,
        s2_B=s2_B,
        rate_limit_enabled=rate_limit_enabled,
    )

    def _json(payload: dict[str, Any], status_code: int = 200) -> JSONResponse:
        return _safe_json(
            payload,
            status_code=status_code,
            extra_symbols=runtime.catalog.forbidden_symbols,
        )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            runtime.close()

    app = FastAPI(
        title="SIFT Gate-1 Invited Beta API",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.runtime = runtime

    # Centralized compliance-guarded exception handlers (zero unguarded detail leaks).
    @app.exception_handler(HTTPException)
    async def _http_exc_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        msg = exc.detail if isinstance(exc.detail, str) else "Request failed."
        return _json({"ok": False, "error": msg}, status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def _unhandled_exc_handler(_request: Request, _exc: Exception) -> JSONResponse:
        return JSONResponse(
            content={"ok": False, "error": "Internal server error."},
            status_code=500,
            headers={"Cache-Control": "no-store"},
        )

    async def _require_user(request: Request) -> BetaUser:
        token = _extract_token(request)
        user = await runtime.afast(runtime.auth.resolve_session, token)
        if user is None:
            raise HTTPException(status_code=401, detail="Authentication required.")
        if user.tos_version != TOS_VERSION:
            raise HTTPException(
                status_code=403, detail="Updated terms acceptance required."
            )
        return user

    # =========================================================================
    # Public routes: landing page, beta app HTML, onboarding, waitlist
    # =========================================================================

    @app.get("/api/health")
    async def api_health() -> JSONResponse:
        return _json({"ok": True, "service": "sift-beta", "signals_only": True})

    @app.get("/api/onboarding")
    async def api_onboarding(request: Request) -> JSONResponse:
        token = _extract_token(request)
        user = await runtime.afast(runtime.auth.resolve_session, token)
        handle = user.user_id if user is not None else None
        return _json({"ok": True, "onboarding": _onboarding_payload(handle=handle)})

    @app.post("/api/waitlist")
    async def api_waitlist(request: Request) -> JSONResponse:
        if not runtime.limiter.check("waitlist", _client_ip(request), limit=20, window_s=60.0):
            return _json({"ok": False, "error": "Too many requests. Please try later."}, status_code=429)
        raw_bytes = await request.body()
        if not raw_bytes or len(raw_bytes) > 16 * 1024:
            return _json({"ok": False, "error": "Invalid body"}, status_code=400)
        ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
        email = ""
        tos_accepted = False
        if ctype == "application/json":
            try:
                body = json.loads(raw_bytes.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                body = None
            if not isinstance(body, dict):
                return _json({"ok": False, "error": "Invalid body"}, status_code=400)
            raw_email = body.get("email", "")
            if not isinstance(raw_email, str):
                return _json(
                    {"ok": False, "error": "Please provide a valid email address."},
                    status_code=400,
                )
            email = raw_email.strip()
            tos_accepted = body.get("tos_accepted") is True
        elif ctype.startswith("multipart/form-data"):
            form = await request.form()
            email = str(form.get("email", "")).strip()
            tos_accepted = str(form.get("tos_accepted", "")).strip().lower() in (
                "1", "true", "yes", "on"
            )
        else:
            qs = parse_qs(raw_bytes.decode("utf-8", errors="replace"))
            email = qs.get("email", [""])[0].strip()
            tos_accepted = qs.get("tos_accepted", [""])[0].strip().lower() in (
                "1", "true", "yes", "on"
            )

        if not EMAIL_RE.match(email):
            return _json(
                {"ok": False, "error": "Please provide a valid email address."},
                status_code=400,
            )
        if not tos_accepted:
            return _json(
                {"ok": False, "error": "Please accept the terms to join the waitlist."},
                status_code=400,
            )

        added = _save_waitlist_csv(runtime.data_dir, email, tos_accepted=True)
        await runtime.afast(
            runtime.telemetry.record,
            "beta_signup",
            {"source": "landing", "tos_accepted": True, "duplicate": not added},
        )
        msg = "Added to the beta waitlist." if added else "You're already on the list."
        return _json({"ok": True, "message": msg})

    # =========================================================================
    # Auth routes (invite-only signup, constant-time off-thread login, delete)
    # =========================================================================

    @app.post("/api/auth/signup")
    async def api_signup(request: Request) -> JSONResponse:
        _require_json_content_type(request)
        if not runtime.limiter.check("signup", _client_ip(request), limit=15, window_s=60.0):
            return _json({"ok": False, "error": "Too many requests. Please try later."}, status_code=429)
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return _json({"ok": False, "error": "Invalid JSON body."}, status_code=400)
        if not isinstance(body, dict):
            return _json({"ok": False, "error": "Invalid JSON body."}, status_code=400)

        try:
            norm_email, norm_code = runtime.auth.validate_signup_inputs(
                email=body.get("email", ""),
                password=body.get("password", ""),
                invite_code=body.get("invite_code", ""),
                tos_accepted=body.get("tos_accepted") is True,
            )
        except AuthError as exc:
            return _json({"ok": False, "error": str(exc)}, status_code=exc.status_code)

        # Compute scrypt OFF the DB thread so auth floods never stall other users.
        pw_hash = await asyncio.to_thread(hash_password, str(body.get("password", "")))

        def _commit_signup():
            user, token, wave = runtime.auth.complete_signup(
                norm_email=norm_email,
                password_hash=pw_hash,
                norm_code=norm_code,
            )
            runtime.telemetry.record(
                "beta_accepted", {"wave": wave}, user_id=user.user_id
            )
            runtime.telemetry.record(
                "account_created",
                {"beta": True, "tos_version": user.tos_version, "wave": wave},
                user_id=user.user_id,
            )
            runtime.telemetry.record(
                "session_login", {"via": "signup"}, user_id=user.user_id
            )
            return user, token

        try:
            user, token = await runtime.afast(_commit_signup)
        except AuthError as exc:
            return _json({"ok": False, "error": str(exc)}, status_code=exc.status_code)

        resp = _json(
            {"ok": True, "user": _user_dict(user), "session_token": token},
            status_code=201,
        )
        resp.set_cookie(
            key=SESSION_COOKIE,
            value=token,
            httponly=True,
            samesite="lax",
            secure=_cookie_secure(request),
            max_age=14 * 86400,
        )
        return resp

    @app.post("/api/auth/login")
    async def api_login(request: Request) -> JSONResponse:
        _require_json_content_type(request)
        if not runtime.limiter.check("login", _client_ip(request), limit=20, window_s=60.0):
            return _json({"ok": False, "error": "Too many requests. Please try later."}, status_code=429)
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return _json({"ok": False, "error": "Invalid JSON body."}, status_code=400)
        if not isinstance(body, dict):
            return _json({"ok": False, "error": "Invalid JSON body."}, status_code=400)

        email = str(body.get("email", ""))
        password = str(body.get("password", ""))
        found = await runtime.afast(runtime.auth.lookup_for_login, email)
        # Constant-time scrypt off the DB thread even when email does not exist.
        stored_hash = found[1] if found is not None else DUMMY_SCRYPT_HASH
        verified = await asyncio.to_thread(verify_password, password, stored_hash)
        if found is None or not verified:
            return _json({"ok": False, "error": "Invalid email or password."}, status_code=401)

        user = found[0]

        def _mint_session():
            tok = runtime.auth.create_session(user.user_id)
            runtime.telemetry.record(
                "session_login", {"via": "password"}, user_id=user.user_id
            )
            return tok

        token = await runtime.afast(_mint_session)
        resp = _json({"ok": True, "user": _user_dict(user), "session_token": token})
        resp.set_cookie(
            key=SESSION_COOKIE,
            value=token,
            httponly=True,
            samesite="lax",
            secure=_cookie_secure(request),
            max_age=14 * 86400,
        )
        return resp

    @app.post("/api/auth/logout")
    async def api_logout(request: Request) -> JSONResponse:
        token = _extract_token(request)
        await runtime.afast(runtime.auth.logout, token)
        resp = _json({"ok": True})
        resp.delete_cookie(SESSION_COOKIE)
        return resp

    @app.post("/api/auth/delete-account")
    async def api_delete_account(request: Request) -> JSONResponse:
        """Fulfill the privacy-note deletion right: permanently erases `email`
        and sessions from `auth.db` and removes saved connections, leaving only
        anonymized opaque-ID telemetry in `telemetry.db`."""
        _require_json_content_type(request)
        user = await _require_user(request)
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            body = {}
        if not isinstance(body, dict) or body.get("confirm") is not True:
            return _json(
                {"ok": False, "error": "confirm: true is required to delete account."},
                status_code=400,
            )

        def _do_delete():
            runtime.connections.delete_for_user(user.user_id)
            return runtime.auth.delete_account(user.user_id)

        deleted = await runtime.afast(_do_delete)
        resp = _json({"ok": True, "deleted": bool(deleted)})
        resp.delete_cookie(SESSION_COOKIE)
        return resp

    @app.get("/api/auth/me")
    async def api_me(request: Request) -> JSONResponse:
        user = await _require_user(request)
        return _json({"ok": True, "user": _user_dict(user)})

    @app.patch("/api/auth/profile")
    async def api_profile(request: Request) -> JSONResponse:
        _require_json_content_type(request)
        user = await _require_user(request)
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return _json({"ok": False, "error": "Invalid JSON body."}, status_code=400)
        if not isinstance(body, dict) or not isinstance(body.get("kill_rate_opt_in"), bool):
            return _json(
                {"ok": False, "error": "kill_rate_opt_in boolean is required."},
                status_code=400,
            )
        updated = await runtime.afast(
            runtime.auth.set_kill_rate_opt_in, user.user_id, body["kill_rate_opt_in"]
        )
        return _json({"ok": True, "user": _user_dict(updated)})

    # =========================================================================
    # Mining endpoints (catalog, trigger run, polling, SSE stream)
    # =========================================================================

    @app.get("/api/mining/catalog")
    async def api_mining_catalog(request: Request) -> JSONResponse:
        await _require_user(request)
        items = runtime.catalog.list_public()
        return _json({
            "ok": True,
            "candidates": items,
            "allowed_scopes": list(ALLOWED_SCOPES),
            "allowed_formations": list(ALLOWED_FORMATIONS),
            "allowed_open_z": list(ALLOWED_OPEN_Z),
            "allowed_close_z": list(ALLOWED_CLOSE_Z),
            "s5_automatic": True,
        })

    @app.post("/api/mining/runs")
    async def api_start_mining_run(request: Request) -> JSONResponse:
        _require_json_content_type(request)
        user = await _require_user(request)
        if not runtime.limiter.check("mining", user.user_id, limit=15, window_s=60.0):
            return _json(
                {"ok": False, "error": "Mining rate limit reached. Please wait a moment."},
                status_code=429,
            )
        active_cnt = await runtime.afast(runtime.runs.active_run_count, user.user_id)
        if active_cnt >= 1:
            return _json(
                {"ok": False, "error": "A mining run is already active for this account."},
                status_code=429,
            )

        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return _json({"ok": False, "error": "Invalid JSON body."}, status_code=400)
        if not isinstance(body, dict):
            return _json({"ok": False, "error": "Invalid JSON body."}, status_code=400)

        # Fail closed on any attempt to disable S5 or inject raw symbols/prices.
        bad_keys = [k for k in body if str(k).lower() in _FORBIDDEN_RUN_INPUT_KEYS]
        if bad_keys:
            return _json(
                {
                    "ok": False,
                    "error": (
                        "Forbidden parameter: S5 registry-depth validation is "
                        "automatic and raw symbols are not accepted."
                    ),
                },
                status_code=400,
            )

        raw_example = body.get("example", False)
        if not isinstance(raw_example, bool):
            return _json(
                {"ok": False, "error": "example must be a boolean."}, status_code=400
            )

        def _prepare():
            cid, scope_id, formation, open_z, close_z = runtime.mining.validate_config(
                candidate_id=body.get("candidate_id"),
                example=raw_example,
                scope_id=str(body.get("scope_id", "free_5y_top1k")),
                formation=int(body.get("formation", 60)),
                open_z=float(body.get("open_z", 2.0)),
                close_z=float(body.get("close_z", 0.5)),
            )
            return runtime.mining.prepare_run(
                user_id=user.user_id,
                candidate_id=cid,
                scope_id=scope_id,
                formation=formation,
                open_z=open_z,
                close_z=close_z,
            )

        try:
            run_meta, pair_in = await runtime.aengine(_prepare)
        except (MiningError, ValueError, TypeError) as exc:
            code = exc.status_code if isinstance(exc, MiningError) else 400
            msg = str(exc) if isinstance(exc, MiningError) else "Invalid mining configuration."
            return _json({"ok": False, "error": msg}, status_code=code)

        if pair_in is not None:
            runtime.engine_submit(
                runtime.mining.execute_run,
                run_id=run_meta["run_id"],
                user_id=user.user_id,
                trial_id=run_meta["trial_id"],
                scope_id=run_meta["scope_id"],
                pair_in=pair_in,
            )

        return _json({"ok": True, "run": run_meta}, status_code=202)

    @app.get("/api/mining/runs/{run_id}")
    async def api_get_mining_run(run_id: str, request: Request) -> JSONResponse:
        user = await _require_user(request)

        def _read_fast():
            row = runtime.runs.get_run(run_id, user.user_id)
            if row is None:
                return None, []
            stages = runtime.read_stages_fast(str(row["trial_id"]))
            return row, stages

        row, stages = await runtime.afast(_read_fast)
        if row is None:
            return _json({"ok": False, "error": "Mining run not found."}, status_code=404)

        card_face = None
        if row["card_id"] and str(row["status"]) == "completed":
            card_face = await runtime.aengine(
                runtime.mining.render_owned_face_or_none,
                str(row["card_id"]),
                user.user_id,
            )
        snap = runtime.mining.build_run_snapshot(row, stages, user.user_id, card_face=card_face)
        return _json({"ok": True, "run": snap})

    @app.get("/api/mining/runs/{run_id}/stream")
    async def api_stream_mining_run(run_id: str, request: Request) -> Response:
        user = await _require_user(request)

        def _read_fast():
            row = runtime.runs.get_run(run_id, user.user_id)
            if row is None:
                return None, []
            return row, runtime.read_stages_fast(str(row["trial_id"]))

        row, stages = await runtime.afast(_read_fast)
        if row is None:
            return _json({"ok": False, "error": "Mining run not found."}, status_code=404)

        channel = runtime.mining.get_channel(
            run_id, user.user_id, row=row, stages=stages
        )
        sub_q = channel.subscribe()

        async def _event_gen():
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        item = await asyncio.to_thread(sub_q.get, True, 0.25)
                    except queue.Empty:
                        yield ": heartbeat\n\n"
                        continue
                    if item is None:
                        break
                    ev_name, ev_data = item
                    try:
                        guard_api_payload(
                            ev_data, extra_symbols=runtime.catalog.forbidden_symbols
                        )
                    except ComplianceError:
                        ev_name = "error"
                        ev_data = {"error": ComplianceError.public_message}
                    yield (
                        f"event: {ev_name}\n"
                        f"data: {json.dumps(ev_data, separators=(',', ':'))}\n\n"
                    )
            finally:
                channel.unsubscribe(sub_q)

        return StreamingResponse(
            _event_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-store",
                "X-Accel-Buffering": "no",
            },
        )

    # =========================================================================
    # Trial history, honest verdict, and minted cards
    # =========================================================================

    @app.get("/api/trials")
    async def api_list_trials(request: Request) -> JSONResponse:
        user = await _require_user(request)

        def _load():
            run_rows = runtime.runs.list_runs_for_user(user.user_id, limit=100)
            trials_out = []
            killed_n = 0
            survivor_n = 0
            for r in run_rows:
                tid = str(r["trial_id"])
                stages = runtime.read_stages_fast(tid)
                verdict = str(r["verdict"])
                if verdict == "killed":
                    killed_n += 1
                elif verdict == "survivor":
                    survivor_n += 1
                trials_out.append({
                    "run_id": str(r["run_id"]),
                    "trial_id": tid,
                    "candidate_id": str(r["candidate_id"]),
                    "family": "pairs",
                    "scope_id": str(r["scope_id"]),
                    "cached": bool(r["cached"]),
                    "status": str(r["status"]),
                    "verdict": verdict,
                    "survivor_class": str(r["survivor_class"]),
                    "killed_at": r["killed_at"],
                    "card_id": r["card_id"],
                    "verdict_copy": _safe_verdict_copy(
                        verdict,
                        killed_at=r["killed_at"],
                        survivor_class=str(r["survivor_class"]),
                        handle=user.user_id,
                    ),
                    "stages": stages,
                    "registered_at": float(r["created_at"]),
                })
            return {
                "ok": True,
                "trials": trials_out,
                "kill_rate_summary": {
                    "opt_in": user.kill_rate_opt_in,
                    "total_trials": len(trials_out),
                    "killed_count": killed_n,
                    "survivor_count": survivor_n,
                },
            }

        return _json(await runtime.afast(_load))

    @app.get("/api/trials/{trial_id}")
    async def api_get_trial(trial_id: str, request: Request) -> JSONResponse:
        user = await _require_user(request)

        def _load_fast():
            row = runtime.read_trial_fast(trial_id)
            if row is None:
                return None, []
            owner = row["discoverer_handle"]
            if owner != user.user_id and not runtime.runs.user_has_trial(
                user.user_id, trial_id
            ):
                # Uniform 404 (no unknown-vs-foreign trial_id oracle).
                return None, []
            return row, runtime.read_stages_fast(trial_id)

        row, stages = await runtime.afast(_load_fast)
        if row is None:
            return _json({"ok": False, "error": "Trial not found."}, status_code=404)

        killed_at = next(
            (
                s["stage"]
                for s in stages
                if s["status"] == "fail" and s["stage"] in ("s0", "s1", "s2")
            ),
            None,
        )
        owned_card_id = await runtime.aengine(
            runtime.mining._find_owned_card_for_trial, trial_id, user.user_id
        )
        return _json({
            "ok": True,
            "trial": {
                "trial_id": str(row["trial_id"]),
                "family": str(row["family"]),
                "scope_id": str(row["scope_id"]),
                "provenance": str(row["provenance"]),
                "registered_at": float(row["registered_at"]),
                "verdict": str(row["verdict"]),
                "survivor_class": str(row["survivor_class"]),
                "killed_at": killed_at,
                "card_id": owned_card_id,
                "verdict_copy": _safe_verdict_copy(
                    str(row["verdict"]),
                    killed_at=killed_at,
                    survivor_class=str(row["survivor_class"]),
                    handle=user.user_id,
                ),
                "stages": stages,
            },
        })

    @app.get("/api/cards")
    async def api_list_cards(request: Request) -> JSONResponse:
        user = await _require_user(request)

        def _load():
            cards = runtime.card_layer.cards.cards_for_owner(user.user_id)
            vault_ids = {c.card_id for c in runtime.card_layer.vault.vault(user.user_id)}
            archive_ids = {c.card_id for c in runtime.card_layer.vault.archive(user.user_id)}
            items = []
            for c in cards:
                vstate = (
                    "vault"
                    if c.card_id in vault_ids
                    else ("archive" if c.card_id in archive_ids else "unassigned")
                )
                item = _card_summary(c, vstate)
                item["face"] = runtime.mining.render_owned_face_or_none(
                    c.card_id, user.user_id
                )
                items.append(item)
            return {"ok": True, "cards": items}

        return _json(await runtime.aengine(_load))

    @app.get("/api/cards/{card_id}/face")
    async def api_card_face(
        card_id: str, request: Request, with_image: bool = False
    ) -> JSONResponse:
        user = await _require_user(request)

        def _render():
            runtime.card_layer.require_layer(user.user_id)
            card = runtime.card_layer.cards.assert_owned(card_id, user.user_id)
            diagram = None
            if with_image:
                seed_val = int(
                    hashlib.sha256(card.card_id.encode("utf-8")).hexdigest()[:8], 16
                )
                phase = (seed_val % 360) * math.pi / 180.0
                series = [
                    round(
                        2.15 * math.sin(phase + k * 0.24) * math.exp(-0.004 * k)
                        + 0.45 * math.cos(phase * 0.5 + k * 0.61),
                        4,
                    )
                    for k in range(64)
                ]
                diagram = {"family": card.family, "series": series, "entry_z": 2.0}
            return runtime.card_layer.face.render(
                card_id,
                viewer_handle=user.user_id,
                with_image=bool(with_image),
                diagram=diagram,
                current_abs_z=runtime.mining.current_abs_z(card_id),
            )

        try:
            face = await runtime.aengine(_render)
        except (LayerError, FaceError):
            return _json(
                {"ok": False, "error": "Card face rendering is unavailable."},
                status_code=503,
            )
        except CardError as exc:
            msg = str(exc)
            if "belongs to" in msg or "non-transferable" in msg:
                return _json(
                    {
                        "ok": False,
                        "error": "Card does not belong to this account (non-transferable).",
                    },
                    status_code=403,
                )
            return _json({"ok": False, "error": "Card not found."}, status_code=404)

        return _json({"ok": True, "face": face})

    # =========================================================================
    # Vault & Saved Connections (`connection_saved`)
    # =========================================================================

    @app.get("/api/vault")
    async def api_vault(request: Request) -> JSONResponse:
        user = await _require_user(request)

        def _load_engine():
            vault_cards = runtime.card_layer.vault.vault(user.user_id)
            archive_cards = runtime.card_layer.vault.archive(user.user_id)
            all_owned = runtime.card_layer.cards.cards_for_owner(user.user_id)
            known = {c.card_id for c in vault_cards} | {c.card_id for c in archive_cards}
            unassigned = [c for c in all_owned if c.card_id not in known]
            share_slots = runtime.card_layer.vault.share_slots(user.user_id)
            return vault_cards, archive_cards, unassigned, share_slots

        vault_cards, archive_cards, unassigned, share_slots = await runtime.aengine(
            _load_engine
        )
        conns = await runtime.afast(runtime.connections.list_for_user, user.user_id)
        hint = _onboarding_payload(handle=user.user_id)["hit_loop_hint"]
        return _json({
            "ok": True,
            "vault": [_card_summary(c, "vault") for c in vault_cards],
            "archive": [_card_summary(c, "archive") for c in archive_cards],
            "unassigned": [_card_summary(c, "unassigned") for c in unassigned],
            "capacity": {
                "used": len(vault_cards),
                "share_slots": share_slots,
                "by_tier": {
                    tier: base + share_slots for tier, base in BASE_CAPACITY.items()
                },
            },
            "connections": [
                {
                    "connection_id": cn.connection_id,
                    "card_id": cn.card_id,
                    "trial_id": cn.trial_id,
                    "label": cn.label,
                    "notify_enabled": cn.notify_enabled,
                    "created_at": cn.created_at,
                }
                for cn in conns
            ],
            "hit_loop_hint": hint,
        })

    @app.post("/api/vault/{card_id}/keep")
    async def api_vault_keep(card_id: str, request: Request) -> JSONResponse:
        user = await _require_user(request)

        def _do():
            entry = runtime.card_layer.vault.keep(card_id, user.user_id)
            return {
                "ok": True,
                "entry": {
                    "card_id": entry.card_id,
                    "state": entry.state,
                    "extra_slots": entry.extra_slots,
                    "updated_at": entry.updated_at,
                },
            }

        try:
            return _json(await runtime.aengine(_do))
        except (VaultError, CardError) as exc:
            msg = str(exc)
            if "flag_off" in msg:
                return _json({"ok": False, "error": "Card layer is disabled."}, status_code=503)
            if "belongs to" in msg or "non-transferable" in msg:
                return _json(
                    {
                        "ok": False,
                        "error": "Card does not belong to this account (non-transferable).",
                    },
                    status_code=403,
                )
            return _json({"ok": False, "error": "Vault operation refused."}, status_code=400)

    @app.post("/api/vault/{card_id}/discard")
    async def api_vault_discard(card_id: str, request: Request) -> JSONResponse:
        user = await _require_user(request)

        def _do():
            res = runtime.card_layer.vault.discard(card_id, user.user_id)
            runtime.fast_call(
                runtime.telemetry.record,
                "card_discarded",
                {
                    "card_id": card_id,
                    "pooled": bool(res["pooled"]),
                    "ghosts_shifted": int(res["ghosts_shifted"]),
                },
                user_id=user.user_id,
            )
            if int(res["ghosts_shifted"]) > 0:
                runtime.fast_call(
                    runtime.telemetry.record,
                    "ghost_promoted",
                    {
                        "card_id": card_id,
                        "ghosts_shifted": int(res["ghosts_shifted"]),
                    },
                    user_id=user.user_id,
                )
            entry = res["entry"]
            return {
                "ok": True,
                "entry": {
                    "card_id": entry.card_id,
                    "state": entry.state,
                    "extra_slots": entry.extra_slots,
                    "updated_at": entry.updated_at,
                },
                "pooled": bool(res["pooled"]),
                "ghosts_shifted": int(res["ghosts_shifted"]),
            }

        try:
            return _json(await runtime.aengine(_do))
        except (VaultError, CardError) as exc:
            msg = str(exc)
            if "flag_off" in msg:
                return _json({"ok": False, "error": "Card layer is disabled."}, status_code=503)
            if "belongs to" in msg or "non-transferable" in msg:
                return _json(
                    {
                        "ok": False,
                        "error": "Card does not belong to this account (non-transferable).",
                    },
                    status_code=403,
                )
            return _json({"ok": False, "error": "Vault discard refused."}, status_code=400)

    @app.post("/api/vault/{card_id}/promote")
    async def api_vault_promote(card_id: str, request: Request) -> JSONResponse:
        user = await _require_user(request)

        def _do():
            entry = runtime.card_layer.vault.promote(card_id, user.user_id)
            return {
                "ok": True,
                "entry": {
                    "card_id": entry.card_id,
                    "state": entry.state,
                    "extra_slots": entry.extra_slots,
                    "updated_at": entry.updated_at,
                },
            }

        try:
            return _json(await runtime.aengine(_do))
        except (VaultError, CardError) as exc:
            msg = str(exc)
            if "flag_off" in msg:
                return _json({"ok": False, "error": "Card layer is disabled."}, status_code=503)
            if "belongs to" in msg or "non-transferable" in msg:
                return _json(
                    {
                        "ok": False,
                        "error": "Card does not belong to this account (non-transferable).",
                    },
                    status_code=403,
                )
            return _json({"ok": False, "error": "Vault promote refused."}, status_code=400)

    @app.get("/api/connections")
    async def api_list_connections(request: Request) -> JSONResponse:
        user = await _require_user(request)
        conns = await runtime.afast(runtime.connections.list_for_user, user.user_id)
        hint = _onboarding_payload(handle=user.user_id)["hit_loop_hint"]

        def _render_faces():
            return {
                cn.card_id: runtime.mining.render_owned_face_or_none(
                    cn.card_id, user.user_id
                )
                for cn in conns
            }

        faces = await runtime.aengine(_render_faces) if conns else {}
        return _json({
            "ok": True,
            "connections": [
                {
                    "connection_id": cn.connection_id,
                    "card_id": cn.card_id,
                    "trial_id": cn.trial_id,
                    "label": cn.label,
                    "notify_enabled": cn.notify_enabled,
                    "created_at": cn.created_at,
                    "face": faces.get(cn.card_id),
                }
                for cn in conns
            ],
            "hit_loop_hint": hint,
        })

    @app.post("/api/connections")
    async def api_save_connection(request: Request) -> JSONResponse:
        _require_json_content_type(request)
        user = await _require_user(request)
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return _json({"ok": False, "error": "Invalid JSON body."}, status_code=400)
        if not isinstance(body, dict) or not body.get("card_id"):
            return _json({"ok": False, "error": "card_id is required."}, status_code=400)

        card_id = str(body["card_id"])
        notify_enabled = bool(body.get("notify_enabled", False))

        # Step 1: verify card ownership & optional digest pref on engine pool.
        def _verify_card():
            card = runtime.card_layer.cards.assert_owned(card_id, user.user_id)
            if notify_enabled:
                with contextlib.suppress(NotifyError):
                    runtime.card_layer.notify.set_pref(
                        card.card_id, user.user_id, True, mode=MODE_DIGEST
                    )
            return card

        try:
            card = await runtime.aengine(_verify_card)
        except CardError as exc:
            msg = str(exc)
            if "belongs to" in msg or "non-transferable" in msg:
                return _json(
                    {
                        "ok": False,
                        "error": "Card does not belong to this account (non-transferable).",
                    },
                    status_code=403,
                )
            return _json({"ok": False, "error": "Card not found."}, status_code=404)

        # Step 2: persist connection & telemetry on fast pool (using pre-verified card).
        def _save_fast():
            from sift.web.compliance import sanitize_user_label

            default_label = f"saved {card.family} connection ({card.scope_id})"
            clean_label = sanitize_user_label(
                body.get("label") if body.get("label") else default_label,
                extra_symbols=runtime.catalog.forbidden_symbols,
            )
            import time as _time
            import uuid as _uuid

            ts = _time.time()
            con = runtime.connections.con
            try:
                existing = con.execute(
                    "SELECT * FROM saved_connection"
                    " WHERE user_id=? AND card_id=? AND removed_at IS NULL",
                    (user.user_id, card.card_id),
                ).fetchone()
                if existing is not None:
                    con.execute(
                        "UPDATE saved_connection SET label=?, notify_enabled=?"
                        " WHERE connection_id=?",
                        (clean_label, 1 if notify_enabled else 0, existing["connection_id"]),
                    )
                    con.commit()
                    return runtime.connections.get(str(existing["connection_id"]), user.user_id), False

                conn_id = f"conn-{_uuid.uuid4().hex[:12]}"
                con.execute(
                    "INSERT INTO saved_connection"
                    " (connection_id, user_id, card_id, trial_id, label, notify_enabled, created_at)"
                    " VALUES (?,?,?,?,?,?,?)",
                    (
                        conn_id,
                        user.user_id,
                        card.card_id,
                        card.trial_id,
                        clean_label,
                        1 if notify_enabled else 0,
                        ts,
                    ),
                )
                con.commit()
            except Exception:
                con.rollback()
                raise
            conn = runtime.connections.get(conn_id, user.user_id)
            runtime.telemetry.record(
                "connection_saved",
                {
                    "connection_id": conn.connection_id,
                    "card_id": conn.card_id,
                    "trial_id": conn.trial_id,
                },
                user_id=user.user_id,
            )
            return conn, True

        try:
            conn, newly_created = await runtime.afast(_save_fast)
        except ComplianceError:
            return _json(
                {
                    "ok": False,
                    "error": (
                        "Connection label refused: labels may not contain "
                        "tickers, banned words, or email addresses."
                    ),
                },
                status_code=400,
            )
        except ConnError as exc:
            return _json({"ok": False, "error": str(exc)}, status_code=exc.status_code)

        return _json(
            {
                "ok": True,
                "created": newly_created,
                "connection": {
                    "connection_id": conn.connection_id,
                    "card_id": conn.card_id,
                    "trial_id": conn.trial_id,
                    "label": conn.label,
                    "notify_enabled": conn.notify_enabled,
                    "created_at": conn.created_at,
                },
                "hit_loop_hint": _onboarding_payload(handle=user.user_id)["hit_loop_hint"],
            },
            status_code=201 if newly_created else 200,
        )

    @app.delete("/api/connections/{connection_id}")
    async def api_delete_connection(
        connection_id: str, request: Request
    ) -> JSONResponse:
        user = await _require_user(request)
        try:
            await runtime.afast(
                runtime.connections.remove, connection_id, user.user_id
            )
        except ConnError as exc:
            return _json({"ok": False, "error": str(exc)}, status_code=exc.status_code)
        return _json({"ok": True, "removed": connection_id})

    # =========================================================================
    # Static web UI routes (confined to web/, blocking source/data files)
    # =========================================================================

    @app.get("/")
    async def serve_landing() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html", media_type="text/html; charset=utf-8")

    @app.get("/app")
    @app.get("/beta")
    async def serve_beta_app() -> FileResponse:
        return FileResponse(WEB_DIR / "beta.html", media_type="text/html; charset=utf-8")

    @app.get("/{rel_path:path}")
    async def serve_static(rel_path: str) -> Response:
        if rel_path.startswith("api/") or rel_path == "api":
            return _json({"ok": False, "error": "Not found"}, status_code=404)
        rel = Path(rel_path)
        if rel.is_absolute() or ".." in rel.parts:
            return _json({"ok": False, "error": "Bad path"}, status_code=400)
        target = (WEB_DIR / rel).resolve()
        web_root = WEB_DIR.resolve()
        if web_root not in target.parents and target != web_root:
            return _json({"ok": False, "error": "Forbidden"}, status_code=403)
        if target.suffix.lower() in _FORBIDDEN_STATIC_EXTS or target.name.startswith("."):
            return _json({"ok": False, "error": "Not found"}, status_code=404)
        if not target.is_file():
            return _json({"ok": False, "error": "Not found"}, status_code=404)
        mime = _STATIC_MIME.get(target.suffix.lower(), "application/octet-stream")
        return FileResponse(target, media_type=mime)

    return app


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    host = os.environ.get("HOST", "127.0.0.1")
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()


__all__ = ["PRIVACY_NOTE", "BetaRuntime", "create_app", "main"]
