"""End-to-end API, auth, mining (SSE + polling), vault, and connections tests."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sift import flags
from sift.cards import copy as card_copy
from sift.web.app import create_app

TEST_SECRET = b"beta-api-test-hmac-secret"


@pytest.fixture(autouse=True)
def _reset_flags():
    flags.reset()
    yield
    flags.reset()


@pytest.fixture()
def client(tmp_path: Path):
    app = create_app(
        data_dir=tmp_path / "data",
        card_secret=TEST_SECRET,
        allow_synthetic_examples=True,
        s2_B=200,
    )
    with TestClient(app) as tc:
        yield tc


def _mint_invite(client: TestClient, wave: str = "wave-1", max_uses: int = 1) -> str:
    rt = client.app.state.runtime
    inv = rt.call(rt.auth.create_invite, wave=wave, max_uses=max_uses)
    rt.call(
        rt.telemetry.record,
        "beta_invited",
        {"invite_id": inv["invite_id"], "wave": wave, "max_uses": max_uses},
    )
    return inv["code"]


def _signup_user(
    client: TestClient,
    email: str,
    password: str = "correct-horse-battery",
    invite_code: str | None = None,
) -> dict:
    code = invite_code or _mint_invite(client)
    res = client.post(
        "/api/auth/signup",
        json={
            "email": email,
            "password": password,
            "invite_code": code,
            "tos_accepted": True,
        },
    )
    assert res.status_code == 201, res.text
    return res.json()


def _wait_run(client: TestClient, run_id: str, headers: dict, timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/mining/runs/{run_id}", headers=headers)
        assert r.status_code == 200
        body = r.json()["run"]
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not complete within {timeout}s")


def _parse_sse(raw_text: str) -> list[tuple[str, dict]]:
    events = []
    cur_event = None
    for line in raw_text.splitlines():
        if line.startswith("event: "):
            cur_event = line[7:].strip()
        elif line.startswith("data: ") and cur_event is not None:
            events.append((cur_event, json.loads(line[6:])))
            cur_event = None
    return events


# ============================================================================
# 1. Existing landing page, /app, /api/onboarding, and /api/waitlist
# ============================================================================

def test_landing_onboarding_and_waitlist_preserved(client: TestClient, tmp_path: Path):
    r_home = client.get("/")
    assert r_home.status_code == 200
    assert "SIFT" in r_home.text

    r_app = client.get("/app")
    assert r_app.status_code == 200
    assert "Invited Beta" in r_app.text

    r_ob = client.get("/api/onboarding")
    assert r_ob.status_code == 200
    ob = r_ob.json()["onboarding"]
    assert ob["statement"] == card_copy.onboarding_statement()
    assert ob["tos_acceptance"] == card_copy.tos_acceptance_text()
    assert "hit_loop_hint" in ob
    assert "cards_are_candidates" in ob

    # Waitlist refuses without tos_accepted (fail closed)
    r_bad = client.post("/api/waitlist", json={"email": "quant@fund.io"})
    assert r_bad.status_code == 400

    # Waitlist accepts with tos_accepted=True and records aggregate beta_signup
    r_ok = client.post(
        "/api/waitlist",
        json={"email": "quant@fund.io", "tos_accepted": True},
    )
    assert r_ok.status_code == 200 and r_ok.json()["ok"] is True
    assert (tmp_path / "data" / "waitlist.csv").exists()

    # Static file safety: never serve .py or path traversal
    assert client.get("/api/waitlist.py").status_code == 404


# ============================================================================
# 2. Invite-only signup, ToS fail-closed, login, and cross-user isolation
# ============================================================================

def test_invite_only_auth_and_tos_fail_closed(client: TestClient):
    # Unauthenticated access refused
    for path in (
        "/api/auth/me",
        "/api/mining/catalog",
        "/api/trials",
        "/api/cards",
        "/api/vault",
        "/api/connections",
    ):
        assert client.get(path).status_code == 401

    invite = _mint_invite(client, wave="wave-1", max_uses=1)

    # Refuses missing / false tos_accepted
    r_no_tos = client.post(
        "/api/auth/signup",
        json={
            "email": "alice@quant.io",
            "password": "correct-horse-battery",
            "invite_code": invite,
            "tos_accepted": False,
        },
    )
    assert r_no_tos.status_code == 400

    # Refuses invalid invite code
    r_bad_inv = client.post(
        "/api/auth/signup",
        json={
            "email": "alice@quant.io",
            "password": "correct-horse-battery",
            "invite_code": "SIFT-INVALID-CODE",
            "tos_accepted": True,
        },
    )
    assert r_bad_inv.status_code == 403

    # Succeeds with valid invite + tos_accepted=True
    su = _signup_user(client, "alice@quant.io", invite_code=invite)
    uid = su["user"]["user_id"]
    token = su["session_token"]
    assert uid.startswith("u-")
    assert uid == uid.lower()  # lowercase opaque UUID never collides with _TICKER_RE
    assert su["user"]["beta"] is True

    # Invite is now exhausted -> cannot be reused
    r_reuse = client.post(
        "/api/auth/signup",
        json={
            "email": "bob@quant.io",
            "password": "correct-horse-battery",
            "invite_code": invite,
            "tos_accepted": True,
        },
    )
    assert r_reuse.status_code == 403

    # Login + /me + logout
    r_me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r_me.status_code == 200
    assert r_me.json()["user"]["user_id"] == uid
    assert "alice@quant.io" not in r_me.text

    r_out = client.post("/api/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert r_out.status_code == 200
    assert (
        client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code
        == 401
    )

    r_in = client.post(
        "/api/auth/login",
        json={"email": "alice@quant.io", "password": "correct-horse-battery"},
    )
    assert r_in.status_code == 200
    assert r_in.json()["user"]["user_id"] == uid


# ============================================================================
# 3. Mining flow: survivor (automatic S5 + SSE + mint + vault + connection)
#    AND honest kill + skip-list dedupe + S5 opt-out rejection
# ============================================================================

def test_mining_survivor_sse_vault_connections_and_isolation(client: TestClient):
    alice = _signup_user(client, "alice@quant.io")
    bob = _signup_user(client, "bob@quant.io")
    h_alice = {"Authorization": f"Bearer {alice['session_token']}"}
    h_bob = {"Authorization": f"Bearer {bob['session_token']}"}

    # Catalog lists opaque candidates + s5_automatic=True
    cat = client.get("/api/mining/catalog", headers=h_alice)
    assert cat.status_code == 200
    assert cat.json()["s5_automatic"] is True
    cids = [c["candidate_id"] for c in cat.json()["candidates"]]
    assert "example-pair-cointegrated" in cids
    assert "example-pair-noise" in cids

    # Attempt to skip S5 is refused (fail closed)
    r_skip = client.post(
        "/api/mining/runs",
        headers=h_alice,
        json={"candidate_id": "example-pair-cointegrated", "skip_s5": True},
    )
    assert r_skip.status_code == 400

    # Trigger guided example survivor run
    r_start = client.post(
        "/api/mining/runs",
        headers=h_alice,
        json={"example": True, "scope_id": "free_5y_top1k"},
    )
    assert r_start.status_code == 202
    run_meta = r_start.json()["run"]
    assert run_meta["cached"] is False

    # Stream SSE ceremony
    with client.stream("GET", run_meta["stream_url"], headers=h_alice) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        sse_raw = "".join(resp.iter_text())
    events = _parse_sse(sse_raw)
    ev_names = [e[0] for e in events]
    assert "registered" in ev_names
    assert "stage" in ev_names
    assert "verdict" in ev_names
    assert "done" in ev_names

    # Verify automatic S5 executed and minted a validated-green card
    stage_events = [e[1] for e in events if e[0] == "stage"]
    stages_seen = {s["stage"]: s["status"] for s in stage_events}
    assert stages_seen["s5"] == "pass"

    snap = _wait_run(client, run_meta["run_id"], h_alice)
    assert snap["verdict"] == "survivor"
    assert snap["survivor_class"] == "validated-green"
    card_id = snap["card_id"]
    assert card_id is not None and card_id.startswith("card-")
    assert snap["card_face"] is not None
    assert snap["card_face"]["explainer"] == card_copy.face_explainer(
        stages=len(snap["card_face"]["gauntlet_stats"]["stages_passed"]),
        handle=alice["user"]["user_id"],
    )

    # Cross-user isolation: Bob cannot view Alice's run or card face (non-transferable)
    assert client.get(f"/api/mining/runs/{run_meta['run_id']}", headers=h_bob).status_code == 404
    assert client.get(f"/api/cards/{card_id}/face", headers=h_bob).status_code == 403
    assert client.post(
        "/api/connections", headers=h_bob, json={"card_id": card_id}
    ).status_code == 403

    # Card face PNG rendering works server-side
    r_png = client.get(f"/api/cards/{card_id}/face?with_image=true", headers=h_alice)
    assert r_png.status_code == 200
    assert r_png.json()["face"]["image"]["png_1x_b64"]

    # Save connection (`connection_saved`)
    r_conn = client.post(
        "/api/connections",
        headers=h_alice,
        json={"card_id": card_id, "label": "my mean reversion monitor", "notify_enabled": True},
    )
    assert r_conn.status_code == 201
    conn_id = r_conn.json()["connection"]["connection_id"]
    assert r_conn.json()["hit_loop_hint"] == card_copy.hit_loop_hint(
        handle=alice["user"]["user_id"]
    )

    # Vault discard -> archive + pool, then promote back
    r_disc = client.post(f"/api/vault/{card_id}/discard", headers=h_alice)
    assert r_disc.status_code == 200
    assert r_disc.json()["entry"]["state"] == "archive"
    assert r_disc.json()["pooled"] is True

    r_prom = client.post(f"/api/vault/{card_id}/promote", headers=h_alice)
    assert r_prom.status_code == 200
    assert r_prom.json()["entry"]["state"] == "vault"

    # Skip-list idempotency: identical config returns cached=True and does NOT
    # emit a duplicate trial_registered event.
    rt = client.app.state.runtime
    before_reg = rt.call(rt.telemetry.count_events, "trial_registered")
    r_dup = client.post(
        "/api/mining/runs",
        headers=h_alice,
        json={"example": True, "scope_id": "free_5y_top1k"},
    )
    assert r_dup.status_code == 202
    assert r_dup.json()["run"]["cached"] is True
    assert r_dup.json()["run"]["trial_id"] == run_meta["trial_id"]
    after_reg = rt.call(rt.telemetry.count_events, "trial_registered")
    assert after_reg == before_reg

    # Delete connection
    assert client.delete(f"/api/connections/{conn_id}", headers=h_alice).status_code == 200


def test_honest_kill_framing_and_kill_rate_opt_in(client: TestClient):
    alice = _signup_user(client, "alice@quant.io")
    h = {"Authorization": f"Bearer {alice['session_token']}"}

    r_start = client.post(
        "/api/mining/runs",
        headers=h,
        json={"candidate_id": "example-pair-noise", "scope_id": "free_5y_top1k"},
    )
    assert r_start.status_code == 202
    snap = _wait_run(client, r_start.json()["run"]["run_id"], h)
    assert snap["verdict"] == "killed"
    assert snap["card_id"] is None
    assert "this is the product working" in snap["verdict_copy"].lower()

    # Toggle kill-rate badge opt-in
    r_prof = client.patch("/api/auth/profile", headers=h, json={"kill_rate_opt_in": True})
    assert r_prof.status_code == 200
    r_trials = client.get("/api/trials", headers=h)
    assert r_trials.status_code == 200
    ks = r_trials.json()["kill_rate_summary"]
    assert ks["opt_in"] is True
    assert ks["total_trials"] == 1
    assert ks["killed_count"] == 1


def test_cards_kill_switch_fails_closed(client: TestClient, monkeypatch):
    alice = _signup_user(client, "alice@quant.io")
    h = {"Authorization": f"Bearer {alice['session_token']}"}

    r_start = client.post("/api/mining/runs", headers=h, json={"example": True})
    snap = _wait_run(client, r_start.json()["run"]["run_id"], h)
    card_id = snap["card_id"]
    assert card_id is not None

    monkeypatch.setenv("SIFT_CARDS_V4_1", "0")
    flags.reset()
    # Card face & vault mutations fail closed (503) while /api/onboarding keeps serving fallback
    assert client.get(f"/api/cards/{card_id}/face", headers=h).status_code == 503
    assert client.post(f"/api/vault/{card_id}/discard", headers=h).status_code == 503
    r_ob = client.get("/api/onboarding")
    assert r_ob.status_code == 200
    assert "fallback" in r_ob.json()["onboarding"]["source"]
