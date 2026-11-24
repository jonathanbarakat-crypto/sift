"""TDD fixtures for soundness #6: S3 must ENFORCE the BH-FDR gate it advertises.

On main 669b6eb, run_s3 computes q_bh via stats.bh_fdr over the stratum's S2
p-values but `passed = dsr >= S3_DSR_MIN` only — bh_fdr['rejected'] is read
NOWHERE. Probe: p_value=0.9 -> q_bh=0.9 (BH never rejects) with DSR=0.975 ->
S3 PASSES. These tests pin path A (ENFORCE): S3 passes only when BOTH the DSR
threshold is met AND the candidate is in the BH rejection set at q=S3_BH_Q.

Written BEFORE the fix (RED), must pass after (GREEN).
"""

import numpy as np
import pytest

from sift.gauntlet import config as cfg
from sift.gauntlet import stats
from sift.gauntlet.runner import GauntletRunner
from sift.registry import Ledger


@pytest.fixture()
def ledger(tmp_path):
    led = Ledger(tmp_path / "ledger.db")
    yield led
    led.close()


# sr_hat=0.20 per bar clears DSR >= 0.95 for any stratum size used here
# (verified against stats.dsr: dsr=1.0 at n=11..1000, v floored at 1/T).
SR_PASS = 0.20


def _register_s2_rows(ledger, scope_id, p_values, family="pairs"):
    """Register one trial per p-value in the stratum and record its S2 row,
    so run_s3's stratum query sees them (the family = the stratum's trials)."""
    for i, p in enumerate(p_values):
        t = ledger.register(family, scope_id, {"probe": i}, "vintage-probe")
        ledger.record_stage(t.trial_id, "s2", "pass", p_value=float(p))


def _run_s3_with_p(ledger, scope_id, p_value):
    """Run run_s3 for a freshly registered candidate with the given S2 p."""
    t = ledger.register("pairs", scope_id, {"cand": p_value}, "vintage-probe")
    runner = GauntletRunner(ledger, seed=0)
    return runner.run_s3(t.trial_id, sr_hat_per_bar=SR_PASS, T=1260,
                         p_value=float(p_value), scope_id=scope_id)


def _stratum_pvals(ledger, scope_id, p_value):
    """The exact p-vector run_s3 feeds to bh_fdr: stratum S2 rows + candidate."""
    rows = ledger.con.execute(
        "SELECT sr.p_value FROM stage_result sr JOIN trial t"
        " ON sr.trial_id = t.trial_id WHERE t.scope_id=? AND sr.stage='s2'"
        " AND sr.p_value IS NOT NULL", (scope_id,)).fetchall()
    return [r[0] for r in rows] + [p_value]


def test_s3_fails_when_dsr_passes_but_bh_would_never_reject(ledger):
    """THE PROBE CASE: p_value=0.9 -> q_bh=0.9 > S3_BH_Q; BH rejects nothing.
    Even with DSR >= 0.95, S3 must FAIL (path A: BH gates)."""
    scope = "free_5y_top1k"
    _register_s2_rows(ledger, scope, [0.9] * 10)   # family of weak p-values
    res = _run_s3_with_p(ledger, scope, p_value=0.9)
    # sanity: DSR really does pass (the defect lets it through on DSR alone)
    assert res.dsr is not None and res.dsr >= cfg.S3_DSR_MIN, (
        f"fixture must have passing DSR to isolate the BH gate, got dsr={res.dsr}")
    # BH over 11 weak p-values rejects nothing -> candidate not in rejection set
    bh = stats.bh_fdr(np.array(_stratum_pvals(ledger, scope, 0.9)), q=cfg.S3_BH_Q)
    cand_idx = len(_stratum_pvals(ledger, scope, 0.9)) - 1
    assert cand_idx not in set(int(i) for i in bh["rejected"])
    assert res.q_bh > cfg.S3_BH_Q
    # THE FIX: S3 must fail — BH is not in the rejection set
    assert res.status == "fail", (
        "soundness #6: DSR>=0.95 with q_bh=0.9 > S3_BH_Q must FAIL S3 "
        "(BH-FDR is an advertised gate, not recorded-only evidence)")


def test_s3_passes_clean_family_with_rejected_candidate(ledger):
    """A clean family (many tiny stratum p-values + a strongly significant
    candidate) must PASS S3 when DSR also clears: BH rejects, DSR passes."""
    scope = "free_5y_top1k"
    _register_s2_rows(ledger, scope, [0.001] * 50)  # strong family signal
    res = _run_s3_with_p(ledger, scope, p_value=0.0001)
    assert res.dsr is not None and res.dsr >= cfg.S3_DSR_MIN
    pvals = _stratum_pvals(ledger, scope, 0.0001)
    bh = stats.bh_fdr(np.array(pvals), q=cfg.S3_BH_Q)
    cand_idx = len(pvals) - 1
    assert cand_idx in set(int(i) for i in bh["rejected"])   # BH rejects candidate
    assert res.q_bh <= cfg.S3_BH_Q
    assert res.status == "pass"


def test_s3_gate_reads_rejection_set_not_only_q(ledger):
    """Pin the exact gate semantics: candidate in bh['rejected'] <=> BH arm of
    the gate passes. Candidate p=0.001 at m=200 is NOT rejected (needs
    p <= q/m = 0.0005) even though it is the smallest p — S3 must fail."""
    scope = "free_5y_top1k"
    rng = np.random.Generator(np.random.PCG64(7))
    _register_s2_rows(ledger, scope, list(rng.uniform(0.05, 1.0, 199)))
    res = _run_s3_with_p(ledger, scope, p_value=0.001)
    assert res.dsr >= cfg.S3_DSR_MIN
    pvals = _stratum_pvals(ledger, scope, 0.001)
    bh = stats.bh_fdr(np.array(pvals), q=cfg.S3_BH_Q)
    cand_idx = len(pvals) - 1
    assert cand_idx not in set(int(i) for i in bh["rejected"])  # m=200: 0.001 > 0.0005
    assert res.q_bh > cfg.S3_BH_Q
    assert res.status == "fail"


def test_s3_q_bh_still_recorded_on_stage_row(ledger):
    """Whatever the gate, q_bh stays recorded on the stage row (evidence)."""
    scope = "free_5y_top1k"
    _register_s2_rows(ledger, scope, [0.9] * 5)
    res = _run_s3_with_p(ledger, scope, p_value=0.9)
    assert res.q_bh is not None and res.q_bh > cfg.S3_BH_Q
    # record() persists it on the s3 row
    t = ledger.con.execute(
        "SELECT trial_id FROM trial ORDER BY registered_at DESC LIMIT 1"
    ).fetchone()["trial_id"]
    GauntletRunner(ledger).record(t, res)
    row = ledger.stage(t, "s3")
    assert row["q_bh"] == pytest.approx(res.q_bh)
    assert row["status"] == "fail"
