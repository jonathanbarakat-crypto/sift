"""v4.1 mechanics item 7 — knowledge-reward badges, behind cards_v4_1.

Planted ledger fixtures (no gauntlet run needed — rows are planted directly
into the append-only ledger, which is exactly what the badges read):

- kill_rate: 12 decided trials for alice (11 killed) -> earned
  "12 trials, 11 killed"; below the 10-trial bar -> not earned.
- prereg_streak: 3 consecutive user_initiated survivors with S0-S4 pass ->
  streak 3 earned; an operator_sweep trial breaks the streak; pending
  trials are skipped.
- peer_retest: record_retest on a decided trial -> read-back artifact
  "re-tested by @bob: 47/50 robust"; the tester's badge aggregates it;
  retest_status lists it on the registry entry; invalid counts / pending /
  unknown trial -> BadgeError.
- scope_honesty: narrow-scope trial with S5 decided at full depth -> earned;
  registry_depth-scope trials and S5-pending trials do not count.
- fail-closed guard (like the card-face ticker guard): forbidden keys
  (pnl/win_rate/...), forbidden terms, ticker-like tokens and stray floats
  in any emitted artifact -> BadgeError; planted tickers in server-side
  params never leak into badge artifacts.
- opt-in: badges_for == [] until set_visibility(handle, True); opt-out
  respected; set_visibility is read back (GET after write).
- flag off (global set_flag, per-user disable_for_user, env
  SIFT_CARDS_V4_1=0) -> BadgeError(flag_off) on every path.
- leaderboard: validation survival rate ONLY (validated-green/decided),
  never P&L; deterministic order.
"""

import json

import pytest

from sift import flags
from sift.cards.badges import (
    BadgeEngine,
    BadgeError,
    KILL_RATE_MIN_TRIALS,
    PREREG_STREAK_MIN,
    badge_dict,
    Badge,
)
from sift.registry import Ledger


@pytest.fixture(autouse=True)
def _flags_reset():
    flags.reset()
    yield
    flags.reset()


@pytest.fixture()
def ledger(tmp_path):
    l = Ledger(tmp_path / "ledger.db")
    yield l
    l.close()


@pytest.fixture()
def engine(ledger, tmp_path):
    e = BadgeEngine(ledger, tmp_path / "badges.db")
    yield e
    e.close()


_PLANT_SEQ = [0]


def plant(ledger, handle, *, verdict, survivor_class="none", provenance="user_initiated",
          scope_id="free_5y_top1k", s0_s4_pass=None, s5=None, i=0):
    """Plant one trial row + stage rows directly in the ledger.
    Params are made unique per call so the skip-list never dedupes two
    planted trials into one row. By default a killed trial fails at S2
    (later stages stay pending, like the real runner's short-circuit);
    survivors pass S0-S4."""
    if s0_s4_pass is None:
        s0_s4_pass = verdict == "survivor"
    _PLANT_SEQ[0] += 1
    trial = ledger.register("pairs", scope_id,
                            {"planted": _PLANT_SEQ[0], "seed": i}, "v1",
                            provenance=provenance, discoverer_handle=handle)
    kill_at = None if s0_s4_pass else "s2"
    for s in ("s0", "s1", "s2", "s3", "s4"):
        if kill_at is not None and s == kill_at:
            ledger.record_stage(trial.trial_id, s, "fail")
        elif kill_at is not None and s > kill_at:
            break   # later stages stay pending (a stage not run cannot pass)
        else:
            ledger.record_stage(trial.trial_id, s, "pass", statistic=1.0)
    if s5 is not None:
        ledger.record_stage(trial.trial_id, "s5", s5, statistic=3.5)
    if verdict != "pending":
        ledger.decide(trial.trial_id, verdict, survivor_class)
    return trial


def _by_kind(badges, kind):
    return next(b for b in badges if b["kind"] == kind)


# -- kill-rate badge ------------------------------------------------------------

def test_kill_rate_badge_from_planted_ledger(ledger, engine):
    for i in range(11):
        plant(ledger, "alice", verdict="killed", i=i)
    plant(ledger, "alice", verdict="survivor", survivor_class="validated-green", i=11)
    badges = engine.compute_badges("alice", now=1700000000.0)
    kr = _by_kind(badges, "kill_rate")
    assert kr["earned"] is True
    assert kr["label"] == "12 trials, 11 killed"
    assert kr["detail"] == {"trials": 12, "killed": 11}
    assert kr["earned_at"] == 1700000000.0


def test_kill_rate_below_bar_not_earned(ledger, engine):
    for i in range(KILL_RATE_MIN_TRIALS - 1):
        plant(ledger, "alice", verdict="killed", i=i)
    kr = _by_kind(engine.compute_badges("alice"), "kill_rate")
    assert kr["earned"] is False
    assert kr["earned_at"] is None
    assert kr["detail"]["killed"] == KILL_RATE_MIN_TRIALS - 1


def test_kill_rate_counts_only_decided_trials(ledger, engine):
    for i in range(5):
        plant(ledger, "alice", verdict="killed", i=i)
    plant(ledger, "alice", verdict="pending", i=99)   # pending: not counted
    kr = _by_kind(engine.compute_badges("alice"), "kill_rate")
    assert kr["detail"] == {"trials": 5, "killed": 5}


def test_kill_rate_is_per_user(ledger, engine):
    for i in range(12):
        plant(ledger, "alice", verdict="killed", i=i)
    plant(ledger, "bob", verdict="survivor", survivor_class="validated-green", i=0)
    assert _by_kind(engine.compute_badges("alice"), "kill_rate")["earned"]
    assert _by_kind(engine.compute_badges("bob"), "kill_rate")["earned"] is False


# -- pre-registration streak ------------------------------------------------------

def test_prereg_streak_earned(ledger, engine):
    for i in range(PREREG_STREAK_MIN):
        plant(ledger, "alice", verdict="survivor", survivor_class="validated-green",
              provenance="user_initiated", s5="pass", i=i)
    pr = _by_kind(engine.compute_badges("alice"), "prereg_streak")
    assert pr["earned"] is True
    assert pr["detail"] == {"streak": PREREG_STREAK_MIN}
    assert pr["label"] == f"pre-registration streak: {PREREG_STREAK_MIN}"


def test_prereg_streak_breaks_on_non_prereg_provenance(ledger, engine):
    plant(ledger, "alice", verdict="survivor", survivor_class="validated-green",
          provenance="user_initiated", i=0)
    plant(ledger, "alice", verdict="survivor", survivor_class="validated-green",
          provenance="operator_sweep", i=1)          # not pre-registered -> break
    plant(ledger, "alice", verdict="survivor", survivor_class="validated-green",
          provenance="user_initiated", i=2)
    pr = _by_kind(engine.compute_badges("alice"), "prereg_streak")
    assert pr["detail"] == {"streak": 1}             # only the latest run counts
    assert pr["earned"] is False


def test_prereg_streak_breaks_on_s0_s4_fail(ledger, engine):
    for i in range(3):
        plant(ledger, "alice", verdict="survivor", survivor_class="validated-green", i=i)
    plant(ledger, "alice", verdict="killed", s0_s4_pass=False, i=10)  # killed at s0-s4
    pr = _by_kind(engine.compute_badges("alice"), "prereg_streak")
    assert pr["detail"] == {"streak": 0}


def test_prereg_streak_skips_pending(ledger, engine):
    for i in range(PREREG_STREAK_MIN):
        plant(ledger, "alice", verdict="survivor", survivor_class="validated-green", i=i)
    plant(ledger, "alice", verdict="pending", i=50)   # in flight: skipped, streak intact
    pr = _by_kind(engine.compute_badges("alice"), "prereg_streak")
    assert pr["detail"] == {"streak": PREREG_STREAK_MIN}


# -- peer re-test ladder ------------------------------------------------------------

def test_record_retest_read_back_and_artifact(ledger, engine):
    t = plant(ledger, "alice", verdict="survivor", survivor_class="validated-green",
              s5="pass", i=0)
    art = engine.record_retest(t.trial_id, "bob", n_retests=50, n_robust=47,
                               now=1700000000.0)
    # read-back: row must exist in the DB (GET after write)
    row = engine.con.execute(
        "SELECT * FROM peer_retest WHERE trial_id=? AND tester_handle=?",
        (t.trial_id, "bob"),
    ).fetchone()
    assert row is not None and row["n_robust"] == 47 and row["n_retests"] == 50
    assert art["label"] == "re-tested by @bob: 47/50 robust"
    assert art["trial_id"] == t.trial_id and art["recorded_at"] == 1700000000.0
    # the status artifact hangs on the registry entry (v3 §6.3)
    status = engine.retest_status(t.trial_id, requester_handle="alice")
    assert len(status) == 1 and status[0]["label"] == art["label"]


def test_peer_retest_badge_aggregates(ledger, engine):
    t1 = plant(ledger, "alice", verdict="survivor", survivor_class="validated-green", i=0)
    t2 = plant(ledger, "alice", verdict="killed", i=1)
    engine.record_retest(t1.trial_id, "bob", n_retests=50, n_robust=47)
    engine.record_retest(t2.trial_id, "bob", n_retests=10, n_robust=9)
    pr = _by_kind(engine.compute_badges("bob"), "peer_retest")
    assert pr["earned"] is True
    assert pr["detail"] == {"retests": 60, "robust": 56, "entries": 2}
    assert pr["label"] == "peer re-tests: 56/60 robust"


def test_record_retest_rejects_bad_counts(ledger, engine):
    t = plant(ledger, "alice", verdict="killed", i=0)
    with pytest.raises(BadgeError):
        engine.record_retest(t.trial_id, "bob", n_retests=0, n_robust=0)
    with pytest.raises(BadgeError):
        engine.record_retest(t.trial_id, "bob", n_retests=10, n_robust=11)
    with pytest.raises(BadgeError):
        engine.record_retest(t.trial_id, "bob", n_retests=10, n_robust=-1)
    with pytest.raises(BadgeError):
        engine.record_retest(t.trial_id, "bob", n_retests=10.5, n_robust=1)


def test_record_retest_rejects_pending_or_unknown_trial(ledger, engine):
    t = plant(ledger, "alice", verdict="pending", i=0)
    with pytest.raises(BadgeError):
        engine.record_retest(t.trial_id, "bob", n_retests=5, n_robust=5)
    with pytest.raises(BadgeError):
        engine.record_retest("trial-999999", "bob", n_retests=5, n_robust=5)
    with pytest.raises(BadgeError):
        engine.retest_status("trial-999999", requester_handle="bob")


# -- scope-honesty badge --------------------------------------------------------------

def test_scope_honesty_earned(ledger, engine):
    plant(ledger, "alice", verdict="survivor", survivor_class="validated-green",
          scope_id="free_5y_top1k", s5="pass", i=0)
    sh = _by_kind(engine.compute_badges("alice"), "scope_honesty")
    assert sh["earned"] is True
    assert sh["detail"] == {"full_depth_judged": 1}


def test_scope_honesty_requires_s5_decided_and_narrow_scope(ledger, engine):
    plant(ledger, "alice", verdict="survivor", survivor_class="validated-green",
          scope_id="free_5y_top1k", s5=None, i=0)          # s5 pending -> no
    plant(ledger, "alice", verdict="killed", scope_id="registry_depth", s5="fail",
          i=1)                                             # full-depth scope -> no
    plant(ledger, "alice", verdict="pending", scope_id="free_5y_top1k", s5=None,
          i=2)                                             # pending -> no
    sh = _by_kind(engine.compute_badges("alice"), "scope_honesty")
    assert sh["detail"] == {"full_depth_judged": 0}
    assert sh["earned"] is False
    # now one honest narrow trial judged at full depth
    plant(ledger, "alice", verdict="killed", scope_id="free_5y_top1k", s5="fail", i=3)
    sh = _by_kind(engine.compute_badges("alice"), "scope_honesty")
    assert sh["earned"] is True and sh["detail"]["full_depth_judged"] == 1


# -- fail-closed content guard -----------------------------------------------------

def test_guard_rejects_forbidden_fields_terms_tickers_floats():
    from sift.cards.badges import _guard_obj
    with pytest.raises(BadgeError):
        _guard_obj({"pnl": 12.3})
    with pytest.raises(BadgeError):
        _guard_obj({"win_rate": 0.9})
    with pytest.raises(BadgeError):
        _guard_obj({"label": "great win rate"})
    with pytest.raises(BadgeError):
        _guard_obj({"label": "buy the dip"})          # directional language
    with pytest.raises(BadgeError):
        _guard_obj({"label": "AAPL killed at s2"})    # ticker-like token
    with pytest.raises(BadgeError):
        _guard_obj({"killed": 41.5})                   # stray float (counts only)
    with pytest.raises(BadgeError):
        _guard_obj({"nested": [{"note": "net Sharpe 2.1"}]})
    # whitelisted float keys pass
    _guard_obj({"survival_rate": 0.5, "earned_at": 1700000000.0})


def test_badges_never_leak_planted_tickers_or_forbidden_keys(ledger, engine):
    t = plant(ledger, "alice", verdict="killed", i=0)
    # plant raw tickers server-side in the trial params — badges read counts only
    ledger.con.execute(
        "UPDATE trial SET params_json=? WHERE trial_id=?",
        (json.dumps({"pair": ["AAPL", "MSFT"], "side": "buy"}), t.trial_id),
    )
    ledger.con.commit()
    engine.record_retest(t.trial_id, "bob", n_retests=4, n_robust=4)
    arts = engine.compute_badges("alice")
    arts += engine.compute_badges("bob")
    arts.append(engine.badge_artifact("alice"))
    blob = json.dumps(arts, default=str)
    assert "AAPL" not in blob and "MSFT" not in blob and "buy" not in blob
    for forbidden in ("pnl", "win_rate", "sharpe", "ticker", "directional"):
        assert forbidden not in blob.lower().replace("p&l", "pnl")
    # no confetti / trading hooks anywhere
    assert engine.badge_artifact("alice")["hooks"] == []


def test_badge_dict_guard_runs_on_emit(engine):
    b = Badge("kill_rate", "alice", "3 trials, 3 killed",
              {"trials": 3, "killed": 3, "pnl": 1}, True, 1.0)
    with pytest.raises(BadgeError):
        badge_dict(b)


# -- opt-in visibility -------------------------------------------------------------

def test_opt_in_required_and_optout_respected(ledger, engine):
    for i in range(12):
        plant(ledger, "alice", verdict="killed", i=i)   # killed at s2: no streak
    # not opted in -> nothing displayed even though earned
    assert engine.badges_for("alice") == []
    assert engine.visibility("alice") is False
    # opt in -> earned badges displayed
    row = engine.set_visibility("alice", True, now=1700000000.0)
    assert row == {"handle": "alice", "opt_in": True, "updated_at": 1700000000.0}
    assert engine.visibility("alice") is True
    shown = engine.badges_for("alice")
    assert [b["kind"] for b in shown] == ["kill_rate"]
    assert engine.badge_artifact("alice")["display_opt_in"] is True
    # opt back out -> respected
    engine.set_visibility("alice", False)
    assert engine.badges_for("alice") == []
    assert engine.badge_artifact("alice")["badges"] == []


def test_visibility_readback_matches_db(engine):
    engine.set_visibility("carol", True, now=1.0)
    row = engine.con.execute(
        "SELECT opt_in, updated_at FROM badge_visibility WHERE handle='carol'"
    ).fetchone()
    assert row["opt_in"] == 1 and row["updated_at"] == 1.0
    assert engine.visibility_row("carol") == {"handle": "carol", "opt_in": True,
                                              "updated_at": 1.0}


# -- flag off -> BadgeError(flag_off) on every path ---------------------------------

def test_flag_off_global_raises_on_every_path(ledger, engine):
    t = plant(ledger, "alice", verdict="killed", i=0)
    flags.set_flag("cards_v4_1", False)
    with pytest.raises(BadgeError):
        engine.compute_badges("alice")
    with pytest.raises(BadgeError):
        engine.badges_for("alice")
    with pytest.raises(BadgeError):
        engine.badge_artifact("alice")
    with pytest.raises(BadgeError):
        engine.set_visibility("alice", True)
    with pytest.raises(BadgeError):
        engine.visibility("alice")
    with pytest.raises(BadgeError):
        engine.record_retest(t.trial_id, "bob", n_retests=2, n_robust=2)
    with pytest.raises(BadgeError):
        engine.retest_status(t.trial_id, requester_handle="alice")
    with pytest.raises(BadgeError):
        engine.leaderboard(requester_handle="alice")


def test_flag_off_per_user(ledger, engine):
    plant(ledger, "alice", verdict="killed", i=0)
    flags.disable_for_user("cards_v4_1", "mallory")
    with pytest.raises(BadgeError):
        engine.compute_badges("mallory")
    # other users unaffected
    assert engine.compute_badges("alice")


def test_flag_off_env_kill_switch(ledger, engine, monkeypatch):
    plant(ledger, "alice", verdict="killed", i=0)
    monkeypatch.setenv("SIFT_CARDS_V4_1", "0")
    flags.reset()
    assert flags.cards_v4_1_enabled() is False
    with pytest.raises(BadgeError):
        engine.compute_badges("alice")
    monkeypatch.delenv("SIFT_CARDS_V4_1")
    flags.reset()
    assert engine.compute_badges("alice")


# -- leaderboard: validation survival rate only ---------------------------------------

def test_leaderboard_survival_rate_only(ledger, engine):
    # alice: 2 decided, 1 validated-green -> 0.5
    plant(ledger, "alice", verdict="survivor", survivor_class="validated-green", s5="pass", i=0)
    plant(ledger, "alice", verdict="killed", i=1)
    # bob: 2 decided, 0 green -> 0.0
    plant(ledger, "bob", verdict="killed", i=0)
    plant(ledger, "bob", verdict="survivor", survivor_class="on-watch-amber", s5="fail", i=1)
    # pending trial must not enter the denominator
    plant(ledger, "bob", verdict="pending", i=2)
    lb = engine.leaderboard(requester_handle="alice")
    assert [e["handle"] for e in lb] == ["alice", "bob"]
    assert lb[0] == {"handle": "alice", "decided": 2, "validated_green": 1,
                     "survival_rate": 0.5}
    assert lb[1]["survival_rate"] == 0.0
    blob = json.dumps(lb, default=str)
    for forbidden in ("pnl", "win_rate", "sharpe", "ticker", "return"):
        assert forbidden not in blob.lower()


def test_leaderboard_deterministic_tiebreak(ledger, engine):
    for h in ("bob", "alice", "carol"):
        plant(ledger, h, verdict="killed", i=0)
    lb = engine.leaderboard()
    assert [e["handle"] for e in lb] == ["alice", "bob", "carol"]  # handle asc on ties
