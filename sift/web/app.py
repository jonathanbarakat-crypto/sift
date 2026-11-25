"""Thin FastAPI server for the SIFT Gate-1 invited beta application.

Wraps the existing server-side engine (`GauntletRunner` + `CardLayer` +
`Ledger`) without altering any quant, card, deck, or sim-harness mechanics:
- Preserves existing static landing page (`/`), `GET /api/onboarding`, and
  `POST /api/waitlist`.
- Invite-code-only account creation (`POST /api/auth/signup`) with mandatory
  day-one ToS acceptance (`tos_accepted: true`) and email-only PII isolation.
- Authenticated pairs mining (`POST /api/mining/runs`, polling, and SSE
  ceremony streaming) with automatic S5 full-depth validation (never opt-in).
- Trial history, honest kill/survivor verdict framing, server-rendered card
  faces (`sift/cards/face.py` + `sift/cards/copy.py`), vault keep/discard/
  promote, and saved connections (`connection_saved`).
- Centralized fail-closed `guard_api_payload` on every API JSON/SSE response.

Run:
    uvicorn sift.web.app:app --host 127.0.0.1 --port 8080
    python3 -m sift.web.app
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import csv
import json
import os
import queue
import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from sift import flags
from sift.cards import CardError
from sift.cards import copy as card_copy
from sift.cards import decks as deck_copy
from sift.cards.face import FaceError
from sift.cards.service import CardLayer, LayerError
from sift.cards.vault import BASE_CAPACITY, VaultError
from sift.gauntlet import config as cfg
from sift.registry import Ledger, LedgerError
from sift.web.auth import EMAIL_RE, AuthError, AuthStore, BetaUser
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
_FORBIDDEN_STATIC_EXTS = frozenset({".py", ".pyc", ".csv", ".json", ".md", ".db", ".sqlite"})
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
    """Build onboarding copy ONLY from sift/cards/copy.py + sift/cards/decks.py
    when cards_v4_1 is enabled; fall back to the verified static clauses when
    the kill-switch is off (preserving GET /api/onboarding availability)."""
    try:
        return {
            "statement": card_copy.onboarding_statement(handle=handle),
            "tos_acceptance": card_copy.tos_acceptance_text(handle=handle),
            "hit_loop_hint": card_copy.hit_loop_hint(handle=handle),
            "cards_are_candidates": deck_copy.cards_are_candidates_line(handle=handle),
            "glossary": card_copy.glossary(handle=handle),
            "privacy_note": PRIVACY_NOTE,
            "source": "sift.cards.copy + sift.cards.decks (cards_v4_1)",
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


def _safe_json(payload: dict[str, Any], status_code: int = 200) -> JSONResponse:
    """Serialize JSON after fail-closed compliance inspection."""
    try:
        guard_api_payload(payload)
    except ComplianceError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"compliance guard blocked response: {exc}",
        ) from exc
    return JSONResponse(
        content=payload,
        status_code=status_code,
        headers={"Cache-Control": "no-store"},
    )


class BetaRuntime:
    """Single-thread DB/engine runtime owning Ledger, CardLayer, and beta.db.

    Every SQLite connection is created and accessed strictly on one dedicated
    worker thread, eliminating SQLite thread-affinity errors and write races
    without modifying `Ledger` or `CardLayer` internals.
    """

    def __init__(
        self,
        *,
        data_dir: str | Path,
        warehouse_dir: str | Path | None = None,
        card_secret: bytes | None = None,
        allow_synthetic_examples: bool | None = None,
        s2_B: int = cfg.S2_B,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.warehouse_dir = Path(warehouse_dir) if warehouse_dir else None
        self.card_secret = card_secret
        self.allow_synthetic_examples = allow_synthetic_examples
        self.s2_B = int(s2_B)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sift-beta-db"
        )
        self._closed = False

        # Initialized inside the dedicated DB thread:
        self.ledger: Ledger
        self.card_layer: CardLayer
        self.auth: AuthStore
        self.telemetry: TelemetryStore
        self.connections: ConnectionStore
        self.runs: MiningStore
        self.catalog: PairCatalog
        self.mining: MiningManager
        self.call(self._init_stores)

    def _init_stores(self) -> None:
        beta_db = self.data_dir / "beta.db"
        self.ledger = Ledger(self.data_dir / "ledger.db")
        self.card_layer = CardLayer(
            self.ledger,
            self.data_dir / "cards",
            card_secret=self.card_secret,
        )
        self.auth = AuthStore(beta_db)
        self.telemetry = TelemetryStore(beta_db)
        self.connections = ConnectionStore(beta_db)
        self.runs = MiningStore(beta_db)
        self.catalog = PairCatalog(
            warehouse_dir=self.warehouse_dir,
            allow_synthetic_examples=self.allow_synthetic_examples,
        )
        self.mining = MiningManager(
            ledger=self.ledger,
            card_layer=self.card_layer,
            telemetry=self.telemetry,
            runs=self.runs,
            catalog=self.catalog,
            s2_B=self.s2_B,
        )

    def submit(self, fn, *args, **kwargs) -> concurrent.futures.Future:
        return self._executor.submit(fn, *args, **kwargs)

    def call(self, fn, *args, **kwargs):
        return self.submit(fn, *args, **kwargs).result()

    async def acall(self, fn, *args, **kwargs):
        fut = self.submit(fn, *args, **kwargs)
        return await asyncio.wrap_future(fut)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        def _close_all() -> None:
            for store in (
                self.runs,
                self.connections,
                self.telemetry,
                self.auth,
                self.card_layer,
                self.ledger,
            ):
                with contextlib.suppress(OSError, RuntimeError):
                    store.close()

        try:
            self._executor.submit(_close_all).result(timeout=10.0)
        finally:
            self._executor.shutdown(wait=True)


def _extract_token(request: Request) -> str | None:
    auth_hdr = request.headers.get("authorization", "").strip()
    if auth_hdr.lower().startswith("bearer "):
        tok = auth_hdr[7:].strip()
        if tok:
            return tok
    cookie_tok = request.cookies.get(SESSION_COOKIE)
    return cookie_tok.strip() if cookie_tok else None


def _user_dict(user: BetaUser) -> dict[str, Any]:
    return {
        "user_id": user.user_id,
        "beta": user.beta,
        "tos_accepted": True,
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

    async def _require_user(request: Request) -> BetaUser:
        token = _extract_token(request)
        user = await runtime.acall(runtime.auth.resolve_session, token)
        if user is None:
            raise HTTPException(status_code=401, detail="Authentication required.")
        return user

    # =========================================================================
    # Public routes: landing page, beta app HTML, onboarding, waitlist
    # =========================================================================

    @app.get("/api/health")
    async def api_health() -> JSONResponse:
        return _safe_json({"ok": True, "service": "sift-beta", "signals_only": True})

    @app.get("/api/onboarding")
    async def api_onboarding(request: Request) -> JSONResponse:
        token = _extract_token(request)
        user = await runtime.acall(runtime.auth.resolve_session, token)
        handle = user.user_id if user is not None else None
        return _safe_json({"ok": True, "onboarding": _onboarding_payload(handle=handle)})

    @app.post("/api/waitlist")
    async def api_waitlist(request: Request) -> JSONResponse:
        raw_bytes = await request.body()
        if not raw_bytes or len(raw_bytes) > 16 * 1024:
            return _safe_json({"ok": False, "error": "Invalid body"}, status_code=400)
        ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
        email = ""
        tos_accepted = False
        if ctype == "application/json":
            try:
                body = json.loads(raw_bytes.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                body = None
            if not isinstance(body, dict):
                return _safe_json({"ok": False, "error": "Invalid body"}, status_code=400)
            raw_email = body.get("email", "")
            if not isinstance(raw_email, str):
                return _safe_json(
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
            return _safe_json(
                {"ok": False, "error": "Please provide a valid email address."},
                status_code=400,
            )
        if not tos_accepted:
            return _safe_json(
                {"ok": False, "error": "Please accept the terms to join the waitlist."},
                status_code=400,
            )

        added = _save_waitlist_csv(runtime.data_dir, email, tos_accepted=True)
        # Aggregate-only telemetry — NEVER store email in telemetry.
        await runtime.acall(
            runtime.telemetry.record,
            "beta_signup",
            {"source": "landing", "tos_accepted": True, "duplicate": not added},
        )
        msg = "Added to the beta waitlist." if added else "You're already on the list."
        return _safe_json({"ok": True, "message": msg})

    # =========================================================================
    # Auth routes (invite-only signup, login, logout, me, profile)
    # =========================================================================

    @app.post("/api/auth/signup")
    async def api_signup(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return _safe_json({"ok": False, "error": "Invalid JSON body."}, status_code=400)
        if not isinstance(body, dict):
            return _safe_json({"ok": False, "error": "Invalid JSON body."}, status_code=400)

        def _do_signup():
            user, token, wave = runtime.auth.signup(
                email=body.get("email", ""),
                password=body.get("password", ""),
                invite_code=body.get("invite_code", ""),
                tos_accepted=body.get("tos_accepted") is True,
            )
            runtime.telemetry.record(
                "beta_accepted", {"wave": wave}, user_id=user.user_id
            )
            runtime.telemetry.record(
                "account_created",
                {"beta": True, "tos_version": user.tos_version, "wave": wave},
                user_id=user.user_id,
            )
            runtime.telemetry.record("session_login", {"via": "signup"}, user_id=user.user_id)
            return user, token

        try:
            user, token = await runtime.acall(_do_signup)
        except AuthError as exc:
            return _safe_json({"ok": False, "error": str(exc)}, status_code=exc.status_code)

        resp = _safe_json(
            {"ok": True, "user": _user_dict(user), "session_token": token},
            status_code=201,
        )
        resp.set_cookie(
            key=SESSION_COOKIE,
            value=token,
            httponly=True,
            samesite="lax",
            secure=(os.environ.get("SIFT_ENV", "dev").lower() == "production"),
            max_age=14 * 86400,
        )
        return resp

    @app.post("/api/auth/login")
    async def api_login(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return _safe_json({"ok": False, "error": "Invalid JSON body."}, status_code=400)
        if not isinstance(body, dict):
            return _safe_json({"ok": False, "error": "Invalid JSON body."}, status_code=400)

        def _do_login():
            user, token = runtime.auth.login(
                email=body.get("email", ""),
                password=body.get("password", ""),
            )
            runtime.telemetry.record("session_login", {"via": "password"}, user_id=user.user_id)
            return user, token

        try:
            user, token = await runtime.acall(_do_login)
        except AuthError as exc:
            return _safe_json({"ok": False, "error": str(exc)}, status_code=exc.status_code)

        resp = _safe_json({"ok": True, "user": _user_dict(user), "session_token": token})
        resp.set_cookie(
            key=SESSION_COOKIE,
            value=token,
            httponly=True,
            samesite="lax",
            secure=(os.environ.get("SIFT_ENV", "dev").lower() == "production"),
            max_age=14 * 86400,
        )
        return resp

    @app.post("/api/auth/logout")
    async def api_logout(request: Request) -> JSONResponse:
        token = _extract_token(request)
        await runtime.acall(runtime.auth.logout, token)
        resp = _safe_json({"ok": True})
        resp.delete_cookie(SESSION_COOKIE)
        return resp

    @app.get("/api/auth/me")
    async def api_me(request: Request) -> JSONResponse:
        user = await _require_user(request)
        return _safe_json({"ok": True, "user": _user_dict(user)})

    @app.patch("/api/auth/profile")
    async def api_profile(request: Request) -> JSONResponse:
        user = await _require_user(request)
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return _safe_json({"ok": False, "error": "Invalid JSON body."}, status_code=400)
        if not isinstance(body, dict) or not isinstance(body.get("kill_rate_opt_in"), bool):
            return _safe_json(
                {"ok": False, "error": "kill_rate_opt_in boolean is required."},
                status_code=400,
            )
        updated = await runtime.acall(
            runtime.auth.set_kill_rate_opt_in, user.user_id, body["kill_rate_opt_in"]
        )
        return _safe_json({"ok": True, "user": _user_dict(updated)})

    # =========================================================================
    # Mining endpoints (catalog, trigger run, polling, SSE stream)
    # =========================================================================

    @app.get("/api/mining/catalog")
    async def api_mining_catalog(request: Request) -> JSONResponse:
        await _require_user(request)
        items = await runtime.acall(runtime.catalog.list_public)
        return _safe_json({
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
        user = await _require_user(request)
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            body = {}
        if not isinstance(body, dict):
            return _safe_json({"ok": False, "error": "Invalid JSON body."}, status_code=400)

        # Fail closed on any attempt to disable S5 or inject raw symbols/prices.
        bad_keys = sorted(k for k in body if str(k).lower() in _FORBIDDEN_RUN_INPUT_KEYS)
        if bad_keys:
            return _safe_json(
                {
                    "ok": False,
                    "error": (
                        "Forbidden parameter(s): S5 registry-depth validation is "
                        "automatic (never opt-in) and raw symbols/prices are not accepted."
                    ),
                },
                status_code=400,
            )

        def _prepare():
            cid, scope_id, formation, open_z, close_z = runtime.mining.validate_config(
                candidate_id=body.get("candidate_id"),
                example=bool(body.get("example", False)),
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
            run_meta, pair_in = await runtime.acall(_prepare)
        except (MiningError, ValueError, TypeError) as exc:
            code = exc.status_code if isinstance(exc, MiningError) else 400
            return _safe_json({"ok": False, "error": str(exc)}, status_code=code)

        if pair_in is not None:
            runtime.submit(
                runtime.mining.execute_run,
                run_id=run_meta["run_id"],
                user_id=user.user_id,
                trial_id=run_meta["trial_id"],
                scope_id=run_meta["scope_id"],
                pair_in=pair_in,
            )

        return _safe_json({"ok": True, "run": run_meta}, status_code=202)

    @app.get("/api/mining/runs/{run_id}")
    async def api_get_mining_run(run_id: str, request: Request) -> JSONResponse:
        user = await _require_user(request)
        try:
            snap = await runtime.acall(
                runtime.mining.get_run_snapshot, run_id, user.user_id
            )
        except MiningError as exc:
            return _safe_json({"ok": False, "error": str(exc)}, status_code=exc.status_code)
        return _safe_json({"ok": True, "run": snap})

    @app.get("/api/mining/runs/{run_id}/stream")
    async def api_stream_mining_run(run_id: str, request: Request) -> Response:
        user = await _require_user(request)
        try:
            channel = await runtime.acall(
                runtime.mining.get_channel, run_id, user.user_id
            )
        except MiningError as exc:
            return _safe_json({"ok": False, "error": str(exc)}, status_code=exc.status_code)

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
                    guard_api_payload(ev_data)
                    yield f"event: {ev_name}\ndata: {json.dumps(ev_data, separators=(',', ':'))}\n\n"
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
                stages = [_stage_to_public(s) for s in runtime.ledger.stages(tid)]
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

        return _safe_json(await runtime.acall(_load))

    @app.get("/api/trials/{trial_id}")
    async def api_get_trial(trial_id: str, request: Request) -> JSONResponse:
        user = await _require_user(request)

        def _load():
            try:
                trial = runtime.ledger.get(trial_id)
                owner = runtime.ledger.trial_owner(trial_id)
            except LedgerError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            if owner != user.user_id and not runtime.runs.user_has_trial(
                user.user_id, trial_id
            ):
                raise HTTPException(status_code=404, detail="Trial not found.")
            stages = [_stage_to_public(s) for s in runtime.ledger.stages(trial_id)]
            killed_at = next(
                (s["stage"] for s in stages if s["status"] == "fail" and s["stage"] in ("s0", "s1", "s2")),
                None,
            )
            owned_card_id = runtime.mining._find_owned_card_for_trial(
                trial_id, user.user_id
            )
            return {
                "ok": True,
                "trial": {
                    "trial_id": trial.trial_id,
                    "family": trial.family,
                    "scope_id": trial.scope_id,
                    "provenance": trial.provenance,
                    "registered_at": trial.registered_at,
                    "verdict": trial.verdict,
                    "survivor_class": trial.survivor_class,
                    "killed_at": killed_at,
                    "card_id": owned_card_id,
                    "verdict_copy": _safe_verdict_copy(
                        trial.verdict,
                        killed_at=killed_at,
                        survivor_class=trial.survivor_class,
                        handle=user.user_id,
                    ),
                    "stages": stages,
                },
            }

        return _safe_json(await runtime.acall(_load))

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
                if flags.cards_v4_1_enabled(user.user_id):
                    try:
                        item["face"] = runtime.card_layer.face.render(
                            c.card_id,
                            viewer_handle=user.user_id,
                            current_abs_z=runtime.mining.current_abs_z(c.card_id),
                        )
                    except (FaceError, CardError):
                        item["face"] = None
                else:
                    item["face"] = None
                items.append(item)
            return {"ok": True, "cards": items}

        return _safe_json(await runtime.acall(_load))

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
                # Deterministic stylized aggregate z-score illustration seeded from
                # card_id only (zero raw market data / zero per-ticker series).
                import hashlib
                import math

                seed_val = int(hashlib.sha256(card.card_id.encode("utf-8")).hexdigest()[:8], 16)
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
            face = await runtime.acall(_render)
        except LayerError as exc:
            return _safe_json({"ok": False, "error": str(exc)}, status_code=503)
        except FaceError as exc:
            return _safe_json({"ok": False, "error": str(exc)}, status_code=503)
        except CardError as exc:
            msg = str(exc)
            if "belongs to" in msg or "non-transferable" in msg:
                return _safe_json(
                    {"ok": False, "error": "Card does not belong to this account (non-transferable)."},
                    status_code=403,
                )
            return _safe_json({"ok": False, "error": msg}, status_code=404)

        return _safe_json({"ok": True, "face": face})

    # =========================================================================
    # Vault & Saved Connections (`connection_saved`)
    # =========================================================================

    @app.get("/api/vault")
    async def api_vault(request: Request) -> JSONResponse:
        user = await _require_user(request)

        def _load():
            vault_cards = runtime.card_layer.vault.vault(user.user_id)
            archive_cards = runtime.card_layer.vault.archive(user.user_id)
            all_owned = runtime.card_layer.cards.cards_for_owner(user.user_id)
            known = {c.card_id for c in vault_cards} | {c.card_id for c in archive_cards}
            unassigned = [c for c in all_owned if c.card_id not in known]
            share_slots = runtime.card_layer.vault.share_slots(user.user_id)
            conns = runtime.connections.list_for_user(user.user_id)
            hint = _onboarding_payload(handle=user.user_id)["hit_loop_hint"]
            return {
                "ok": True,
                "vault": [_card_summary(c, "vault") for c in vault_cards],
                "archive": [_card_summary(c, "archive") for c in archive_cards],
                "unassigned": [_card_summary(c, "unassigned") for c in unassigned],
                "capacity": {
                    "used": len(vault_cards),
                    "share_slots": share_slots,
                    "by_tier": {
                        tier: base + share_slots
                        for tier, base in BASE_CAPACITY.items()
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
            }

        return _safe_json(await runtime.acall(_load))

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
            return _safe_json(await runtime.acall(_do))
        except (VaultError, CardError) as exc:
            msg = str(exc)
            if "flag_off" in msg:
                return _safe_json({"ok": False, "error": msg}, status_code=503)
            if "belongs to" in msg or "non-transferable" in msg:
                return _safe_json(
                    {"ok": False, "error": "Card does not belong to this account (non-transferable)."},
                    status_code=403,
                )
            return _safe_json({"ok": False, "error": msg}, status_code=400)

    @app.post("/api/vault/{card_id}/discard")
    async def api_vault_discard(card_id: str, request: Request) -> JSONResponse:
        user = await _require_user(request)

        def _do():
            res = runtime.card_layer.vault.discard(card_id, user.user_id)
            runtime.telemetry.record(
                "card_discarded",
                {
                    "card_id": card_id,
                    "pooled": bool(res["pooled"]),
                    "ghosts_shifted": int(res["ghosts_shifted"]),
                },
                user_id=user.user_id,
            )
            if int(res["ghosts_shifted"]) > 0:
                runtime.telemetry.record(
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
            return _safe_json(await runtime.acall(_do))
        except (VaultError, CardError) as exc:
            msg = str(exc)
            if "flag_off" in msg:
                return _safe_json({"ok": False, "error": msg}, status_code=503)
            if "belongs to" in msg or "non-transferable" in msg:
                return _safe_json(
                    {"ok": False, "error": "Card does not belong to this account (non-transferable)."},
                    status_code=403,
                )
            return _safe_json({"ok": False, "error": msg}, status_code=400)

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
            return _safe_json(await runtime.acall(_do))
        except (VaultError, CardError) as exc:
            msg = str(exc)
            if "flag_off" in msg:
                return _safe_json({"ok": False, "error": msg}, status_code=503)
            if "belongs to" in msg or "non-transferable" in msg:
                return _safe_json(
                    {"ok": False, "error": "Card does not belong to this account (non-transferable)."},
                    status_code=403,
                )
            return _safe_json({"ok": False, "error": msg}, status_code=400)

    @app.get("/api/connections")
    async def api_list_connections(request: Request) -> JSONResponse:
        user = await _require_user(request)

        def _load():
            conns = runtime.connections.list_for_user(user.user_id)
            hint = _onboarding_payload(handle=user.user_id)["hit_loop_hint"]
            out = []
            for cn in conns:
                face = None
                if flags.cards_v4_1_enabled(user.user_id):
                    try:
                        face = runtime.card_layer.face.render(
                            cn.card_id,
                            viewer_handle=user.user_id,
                            current_abs_z=runtime.mining.current_abs_z(cn.card_id),
                        )
                    except (FaceError, CardError):
                        face = None
                out.append({
                    "connection_id": cn.connection_id,
                    "card_id": cn.card_id,
                    "trial_id": cn.trial_id,
                    "label": cn.label,
                    "notify_enabled": cn.notify_enabled,
                    "created_at": cn.created_at,
                    "face": face,
                })
            return {"ok": True, "connections": out, "hit_loop_hint": hint}

        return _safe_json(await runtime.acall(_load))

    @app.post("/api/connections")
    async def api_save_connection(request: Request) -> JSONResponse:
        user = await _require_user(request)
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return _safe_json({"ok": False, "error": "Invalid JSON body."}, status_code=400)
        if not isinstance(body, dict) or not body.get("card_id"):
            return _safe_json({"ok": False, "error": "card_id is required."}, status_code=400)

        def _do():
            conn, newly_created = runtime.connections.save(
                card_layer=runtime.card_layer,
                user_id=user.user_id,
                card_id=str(body["card_id"]),
                label=body.get("label"),
                notify_enabled=bool(body.get("notify_enabled", False)),
            )
            if newly_created:
                runtime.telemetry.record(
                    "connection_saved",
                    {
                        "connection_id": conn.connection_id,
                        "card_id": conn.card_id,
                        "trial_id": conn.trial_id,
                    },
                    user_id=user.user_id,
                )
            return conn, newly_created

        try:
            conn, newly_created = await runtime.acall(_do)
        except ComplianceError:
            return _safe_json(
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
            return _safe_json({"ok": False, "error": str(exc)}, status_code=exc.status_code)
        except CardError as exc:
            msg = str(exc)
            if "belongs to" in msg or "non-transferable" in msg:
                return _safe_json(
                    {"ok": False, "error": "Card does not belong to this account (non-transferable)."},
                    status_code=403,
                )
            return _safe_json({"ok": False, "error": msg}, status_code=404)

        return _safe_json(
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
            await runtime.acall(
                runtime.connections.remove, connection_id, user.user_id
            )
        except ConnError as exc:
            return _safe_json({"ok": False, "error": str(exc)}, status_code=exc.status_code)
        return _safe_json({"ok": True, "removed": connection_id})

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
            return _safe_json({"ok": False, "error": "Not found"}, status_code=404)
        rel = Path(rel_path)
        if rel.is_absolute() or ".." in rel.parts:
            raise HTTPException(status_code=400, detail="Bad path")
        target = (WEB_DIR / rel).resolve()
        web_root = WEB_DIR.resolve()
        if web_root not in target.parents and target != web_root:
            raise HTTPException(status_code=403, detail="Forbidden")
        if target.suffix.lower() in _FORBIDDEN_STATIC_EXTS or target.name.startswith("."):
            raise HTTPException(status_code=404, detail="Not found")
        if not target.is_file():
            raise HTTPException(status_code=404, detail="Not found")
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
