"""Tests for Gate-1 append-only telemetry store, Wilson 95% CI, and beta_cli."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from sift.web import beta_cli
from sift.web.telemetry import (
    DAY_S,
    TelemetryError,
    TelemetryStore,
    gate1_report,
    qualitative_band,
    wilson_ci,
)


def test_wilson_ci_matches_beta_plan_benchmarks():
    # docs/beta-plan.md §0 benchmarks (z=1.96):
    # n=30, k=15 -> [33%, 67%], width ~34pp
    lo30, hi30 = wilson_ci(15, 30)
    assert round(lo30 * 100) == 33 and round(hi30 * 100) == 67
    # n=20, k=10 -> [30%, 70%], width ~40pp
    lo20, hi20 = wilson_ci(10, 20)
    assert round(lo20 * 100) == 30 and round(hi20 * 100) == 70
    # n=10, k=5 -> [24%, 76%], width ~53pp
    lo10, hi10 = wilson_ci(5, 10)
    assert round(lo10 * 100) == 24 and round(hi10 * 100) == 76
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_qualitative_band_thresholds():
    assert qualitative_band(0, 0) == "no_cohort"
    assert qualitative_band(10, 20) == "majority_retained"
    assert qualitative_band(6, 20) == "minority_retained"
    assert qualitative_band(8, 20) == "inconclusive_in_between"


def test_telemetry_append_only_triggers_and_pii_guard(tmp_path: Path):
    db = tmp_path / "beta.db"
    with TelemetryStore(db) as store:
        ev_id = store.record("account_created", {"beta": True}, user_id="u-abc123")
        assert ev_id == 1

        # Append-only SQLite triggers refuse UPDATE and DELETE
        with pytest.raises(sqlite3.IntegrityError):
            store.con.execute(
                "UPDATE telemetry_event SET event_type='tampered' WHERE event_id=?",
                (ev_id,),
            )
        with pytest.raises(sqlite3.IntegrityError):
            store.con.execute("DELETE FROM telemetry_event WHERE event_id=?", (ev_id,))

        # Fail-closed PII guards refuse email keys or email strings
        with pytest.raises(TelemetryError):
            store.record("beta_signup", {"email": "leak@quant.io"})
        with pytest.raises(TelemetryError):
            store.record("feedback_note", {"note": "reach me at leak@quant.io"})
        with pytest.raises(TelemetryError):
            store.record("session_login", {}, user_id="leak@quant.io")
        with pytest.raises(TelemetryError):
            store.record("unknown_event", {})


def test_d30_d60_d90_report_and_dedupe(tmp_path: Path):
    db = tmp_path / "beta.db"
    t0 = 1_800_000_000.0
    with TelemetryStore(db) as store:
        # 10 beta users created at t0
        users = [f"u-{i:08x}" for i in range(10)]
        for u in users:
            store.record("account_created", {"beta": True}, user_id=u, now=t0)

        # D30 (0 < t <= 30d): 7 active (users 0..6); user 0 also logs duplicate trial_id
        for u in users[:7]:
            store.record(
                "trial_registered",
                {"trial_id": f"trial-d30-{u}", "scope_id": "free_5y_top1k"},
                user_id=u,
                now=t0 + 10 * DAY_S,
            )
        store.record(
            "trial_registered",
            {"trial_id": f"trial-d30-{users[0]}", "scope_id": "free_5y_top1k"},
            user_id=users[0],
            now=t0 + 12 * DAY_S,
        )

        # D60 (30d < t <= 60d): 5 active (users 0..4)
        for u in users[:5]:
            store.record(
                "trial_registered",
                {"trial_id": f"trial-d60-{u}", "scope_id": "free_5y_top1k"},
                user_id=u,
                now=t0 + 45 * DAY_S,
            )

        # D90 (60d < t <= 90d): 3 active (users 0..2); users 0..3 hold cards
        for u in users[:4]:
            store.record("card_minted", {"card_id": f"card-{u}"}, user_id=u, now=t0 + 15 * DAY_S)
            store.record(
                "trial_verdict",
                {"trial_id": f"trial-d30-{u}", "verdict": "survivor", "survivor_class": "validated-green"},
                user_id=u,
                now=t0 + 15 * DAY_S,
            )
        for u in users[:3]:
            store.record(
                "trial_registered",
                {"trial_id": f"trial-d90-{u}", "scope_id": "free_5y_top1k"},
                user_id=u,
                now=t0 + 80 * DAY_S,
            )
        store.record("connection_saved", {"connection_id": "conn-1"}, user_id=users[0], now=t0 + 20 * DAY_S)

        rep = gate1_report(store, now=t0 + 91 * DAY_S)
        assert rep["cohort_n"] == 10
        assert rep["horizons"]["D30"]["k"] == 7
        assert rep["horizons"]["D30"]["n"] == 10
        assert rep["horizons"]["D30"]["qualitative_band"] == "majority_retained"
        assert rep["d30_early_stop_loss_triggered"] is False

        assert rep["horizons"]["D60"]["k"] == 5
        assert rep["horizons"]["D60"]["qualitative_band"] == "majority_retained"

        assert rep["horizons"]["D90"]["k"] == 3
        assert rep["horizons"]["D90"]["qualitative_band"] == "minority_retained"

        # Deduplication: 7 + 5 + 3 = 15 unique trials (duplicate trial-d30-u0 not double-counted)
        assert rep["secondary"]["unique_trials_registered"] == 15
        assert rep["secondary"]["connections_saved"] == 1
        assert rep["secondary"]["cards_minted"] == 4
        # G1-a counts-only split
        g1a = rep["secondary"]["g1_a_d90_split_counts_only"]
        assert g1a["card_holders"]["matured_n"] == 4
        assert g1a["card_holders"]["active_k"] == 3
        assert g1a["non_card_holders"]["matured_n"] == 6
        assert g1a["non_card_holders"]["active_k"] == 0


def test_beta_cli_invite_report_and_export(tmp_path: Path, capsys):
    data_dir = str(tmp_path / "data")
    assert beta_cli.main(["invite-create", "--data-dir", data_dir, "--wave", "wave-1", "--count", "2", "--json"]) == 0
    inv_out = json.loads(capsys.readouterr().out)
    assert len(inv_out["invites"]) == 2

    assert beta_cli.main(["report", "--data-dir", data_dir, "--json"]) == 0
    rep_out = json.loads(capsys.readouterr().out)
    assert rep_out["funnel_counts"]["beta_invited"] == 2
    assert "D30" in rep_out["horizons"]

    assert beta_cli.main(["report", "--data-dir", data_dir]) == 0
    text_out = capsys.readouterr().out
    assert "Wilson 95% CI" in text_out

    assert beta_cli.main(["export-events", "--data-dir", data_dir]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert len(lines) == 2
    assert all(ev["event_type"] == "beta_invited" for ev in lines)
