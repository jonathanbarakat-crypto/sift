"""v4.1 mechanics 4 — server-rendered anonymized card face.

Fixtures: render a validated-green card -> face contains only aggregates +
per-family specifics + gauntlet survival stats + watermark(handle, ts);
no ticker-like tokens (guard raises on any); raw tickers never leave the
server (the renderer has no access to them by construction — it reads only
the ledger + card registry). Shareable object watermarked with the sharing
handle. Flag off -> FaceError. Foreign card -> CardError (non-transferable).
"""

import json

import pytest

from sift import flags
from sift.cards import CardError, CardRegistry
from sift.cards.face import CardFaceRenderer, FaceError
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


def test_face_has_only_aggregates_stats_and_watermark(ledger, cards, renderer):
    verdict, _ = _run_validated_green(ledger, cards, handle="alice")
    face = renderer.render(verdict["card_id"], viewer_handle="alice", now=1700000000.0)
    assert face["card_id"] == verdict["card_id"]
    assert face["family"] == "pairs"
    assert face["scope_id"] == "free_5y_top1k"
    assert face["survivor_class"] == "validated-green"
    # sector aggregate only — no tickers
    assert face["sector_aggregate"]["level"] == "aggregate"
    # gauntlet stats = survival facts
    assert "s5" in face["gauntlet_stats"]["stages_passed"]
    assert face["gauntlet_stats"]["s3_dsr"] is not None
    # watermark handle+timestamp
    assert face["watermark"] == {"handle": "alice", "ts": 1700000000.0,
                                 "note": face["watermark"]["note"]}
    # serialized face carries no ticker-like token (guard already ran; assert again
    # with the same 2-5 letter rule the renderer enforces)
    blob = json.dumps(face, default=str)
    import re
    assert not re.search(r"\b[A-Z]{2,5}(?:\.[A-Z]{1,2})?\b",
                         blob.replace("SIFT", "").replace("JSON", "").replace("P&L", ""))


def test_face_never_contains_raw_tickers(ledger, cards, renderer):
    """The card spec itself may name a pair — the FACE must not. Render and
    verify no ticker-shaped token appears even though the trial params may
    contain symbols server-side."""
    verdict, trial = _run_validated_green(ledger, cards, handle="alice")
    # plant a raw ticker in the trial params (server-side only) to prove the
    # renderer cannot leak it: params_json lives in the ledger, the face
    # reads only stage rows + card row.
    ledger.con.execute(
        "UPDATE trial SET params_json=? WHERE trial_id=?",
        (json.dumps({"pair": ["AAPL", "MSFT"], "formation": 60}), trial.trial_id),
    )
    ledger.con.commit()
    face = renderer.render(verdict["card_id"], viewer_handle="alice")
    blob = json.dumps(face, default=str)
    assert "AAPL" not in blob and "MSFT" not in blob


def test_share_object_watermarked_with_sharing_handle(ledger, cards, renderer):
    verdict, _ = _run_validated_green(ledger, cards, handle="alice")
    share = renderer.render_share(verdict["card_id"], owner_handle="alice", now=1700000001.0)
    assert share["watermark"]["handle"] == "alice"
    assert share["watermark"]["ts"] == 1700000001.0
    # a foreign handle cannot produce a share object for alice's card
    with pytest.raises(CardError):
        renderer.render_share(verdict["card_id"], owner_handle="eve")


def test_flag_off_refuses_render(ledger, cards, renderer):
    verdict, _ = _run_validated_green(ledger, cards, handle="alice")
    flags.disable_for_user("cards_v4_1", "alice")
    with pytest.raises(FaceError):
        renderer.render(verdict["card_id"], viewer_handle="alice")


def test_ghost_face_shows_fifo_slot(ledger, cards, renderer):
    verdict, trial = _run_validated_green(ledger, cards, handle="alice")
    t = ledger.get(trial.trial_id)
    s2, s3, s5 = (ledger.stage(trial.trial_id, s) for s in ("s2", "s3", "s5"))
    mk = dict(trial_id=t.trial_id, family=t.family, scope_id=t.scope_id,
              config_hash=t.config_hash, data_vintage_hash=t.data_vintage_hash,
              survivor_class="validated-green", owner_handle="alice",
              dsr=s3["dsr"], t_net_full=s5["statistic"], p_perm=s2["p_value"])
    ghost = cards.mint(**mk)
    face = renderer.render(ghost.card_id, viewer_handle="alice")
    assert face["kind"] == "ghost"
    assert face["ghost_no"] == 1
