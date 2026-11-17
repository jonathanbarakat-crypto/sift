"""Anti-abuse attack-sim tests (SIMULATED — infrastructure validation, not
user-behavior evidence).

Runs the full §4.7 attack battery and asserts every control fires, the
append-only share_grant log is intact (UPDATE/DELETE blocked by triggers),
and the audit log is written. Also asserts the banner is on every report.
"""

import sqlite3

from sift.sim.antiabuse import run_antiabuse_sims
from sift.sim.header import SIM_HEADER
from sift.sim.report import render_text


def test_antiabuse_all_controls_fire(sim_env, tmp_path):
    res = run_antiabuse_sims(tmp_path / "ab", seed=20261117)
    assert res.report["sim"] is True
    assert res.report["header"] == SIM_HEADER
    assert render_text(res.report).splitlines()[0] == SIM_HEADER
    failed = [k for k, v in res.checks.items() if not v]
    assert not failed, f"controls that did not fire: {failed}"
    sections = res.report["sections"]
    # every attack section present
    for key in ("attack_1_device_ip_asn_cluster", "attack_2_day_zero_before_first_run",
                "attack_3_unverifiable_posts", "attack_4_duplicate_share_same_card",
                "attack_5_device_rate_limit", "attack_6_plus5_cap",
                "attack_7_abuse_revocation", "attack_8_fraud_queue_sla",
                "append_only_integrity"):
        assert key in sections, key
    assert sections["summary"]["all_checks_pass"] is True


def test_antiabuse_deterministic_same_seed(sim_env, tmp_path):
    """Same seed -> byte-identical report JSON (fixed clock + seeded RNG)."""
    r1 = run_antiabuse_sims(tmp_path / "a", seed=7)
    r2 = run_antiabuse_sims(tmp_path / "b", seed=7)
    assert r1.json_text == r2.json_text
    r3 = run_antiabuse_sims(tmp_path / "c", seed=8)
    assert r1.json_text != r3.json_text


def test_append_only_triggers_block_direct_sql(sim_env, tmp_path):
    """Direct SQL probe on the sim's shares.db: UPDATE and DELETE on
    share_grant must raise (triggers), and the row count is unchanged."""
    res = run_antiabuse_sims(tmp_path / "ab", seed=99)
    assert res.checks["share_grant_append_only_triggers_block"]
    db = tmp_path / "ab" / "cards" / "shares.db"
    con = sqlite3.connect(str(db))
    try:
        n_before = con.execute("SELECT COUNT(*) FROM share_grant").fetchone()[0]
        for stmt in ("UPDATE share_grant SET outcome='x' WHERE seq=1",
                     "DELETE FROM share_grant WHERE seq=1"):
            try:
                con.execute(stmt)
                raise AssertionError(f"{stmt} did NOT raise — log is mutable!")
            except sqlite3.IntegrityError:
                pass
        con.rollback()
        n_after = con.execute("SELECT COUNT(*) FROM share_grant").fetchone()[0]
        assert n_before == n_after and n_before > 0
    finally:
        con.close()
