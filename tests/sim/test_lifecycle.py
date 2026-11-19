"""Card-layer lifecycle sim tests (SIMULATED — infrastructure validation,
not user-behavior evidence).

mint -> ghost FIFO promotion on discard -> decay downgrade-only -> archive.
Asserts queue renumbering correctness and that tiers never upgrade.
"""

from sift.cards import CARD_TIERS
from sift.sim.header import SIM_HEADER
from sift.sim.lifecycle import run_lifecycle_sim
from sift.sim.report import render_text


def test_lifecycle_sim_all_checks(sim_env, tmp_path):
    report = run_lifecycle_sim(tmp_path / "life", seed=20261117)
    assert report["sim"] is True
    assert report["header"] == SIM_HEADER
    assert render_text(report).splitlines()[0] == SIM_HEADER
    sec = report["sections"]
    failed = [k for k, v in sec["checks"].items() if not v]
    assert not failed, f"lifecycle checks failed: {failed}"
    assert sec["all_checks_pass"] is True
    # ghost FIFO renumbering: [1,2] -> discard ghost 1 -> [1] (g2 promoted)
    assert sec["ghost_fifo"]["before_discard"] == [1, 2]
    assert sec["ghost_fifo"]["after_discard"] == [1]
    # decay: downgrade happened, never upgraded on the second pass
    assert sec["decay"]["downgraded"], "expected at least one downgrade"
    assert sec["decay"]["ledger_decay_events"] >= 1
    tiers = sec["decay"]["tiers_after_second_pass"]
    assert all(t in CARD_TIERS for t in tiers.values())


def test_lifecycle_deterministic_same_seed(sim_env, tmp_path):
    import json
    r1 = run_lifecycle_sim(tmp_path / "a", seed=5)
    r2 = run_lifecycle_sim(tmp_path / "b", seed=5)
    assert json.dumps(r1, sort_keys=True, default=str) == \
        json.dumps(r2, sort_keys=True, default=str)
