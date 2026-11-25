"""Unit tests for the append-only trial registry ledger (v3 §5).

Covers the four acceptance areas of gate-1 build 1/8:
register / dedupe (skip-list) / decide / stratum_stats, plus the
append-only invariants (stage rows and verdict transition exactly once).
"""

from __future__ import annotations

import math

import pytest

from sift.registry import (
    STAGES,
    Ledger,
    LedgerError,
    canonical_json,
    config_hash_of,
    sample_variance,
    skip_list_key,
)

VINTAGE_A = "vintage-aaaa"
VINTAGE_B = "vintage-bbbb"
SCOPE_FREE = "free-5yr-top1000"
SCOPE_PRO = "pro-full"

PARAMS = {"lookback": 20, "entry_z": 2.0, "exit_z": 0.5}
PARAMS_VARIANT = {"lookback": 20, "entry_z": 2.5, "exit_z": 0.5}


@pytest.fixture()
def ledger(tmp_path):
    with Ledger(tmp_path / "registry.db") as led:
        yield led


# ---------------------------------------------------------------- register
def test_register_mints_trial_id_and_stage_rows(ledger):
    t = ledger.register("pairs", SCOPE_FREE, PARAMS, VINTAGE_A)
    assert t.trial_id == "trial-000001"
    assert t.verdict == "pending"
    assert t.survivor_class == "none"
    assert t.family == "pairs"
    assert t.scope_id == SCOPE_FREE
    assert t.data_vintage_hash == VINTAGE_A
    stages = ledger.stages(t.trial_id)
    assert [s["stage"] for s in stages] == list(STAGES)  # s0..s5
    assert all(s["status"] == "pending" for s in stages)


def test_register_mints_sequential_ids(ledger):
    t1 = ledger.register("pairs", SCOPE_FREE, PARAMS, VINTAGE_A)
    t2 = ledger.register("pairs", SCOPE_FREE, PARAMS_VARIANT, VINTAGE_A)
    assert (t1.trial_id, t2.trial_id) == ("trial-000001", "trial-000002")
    assert ledger.trial_count() == 2


def test_register_rejects_invalid_provenance(ledger):
    with pytest.raises(LedgerError):
        ledger.register("pairs", SCOPE_FREE, PARAMS, VINTAGE_A, provenance="anon")


def test_get_unknown_trial_raises(ledger):
    with pytest.raises(LedgerError):
        ledger.get("trial-999999")


# ------------------------------------------------------------------- dedupe
def test_reregister_identical_config_returns_same_id(ledger):
    t1 = ledger.register("pairs", SCOPE_FREE, PARAMS, VINTAGE_A)
    t2 = ledger.register("pairs", SCOPE_FREE, dict(PARAMS), VINTAGE_A)
    assert t1.trial_id == t2.trial_id
    assert ledger.trial_count() == 1  # counted exactly once in N


def test_dedupe_is_on_canonical_json_not_key_order(ledger):
    p1 = {"a": 1, "b": {"x": 1, "y": 2}}
    p2 = {"b": {"y": 2, "x": 1}, "a": 1}
    assert canonical_json(p1) == canonical_json(p2)
    t1 = ledger.register("pairs", SCOPE_FREE, p1, VINTAGE_A)
    t2 = ledger.register("pairs", SCOPE_FREE, p2, VINTAGE_A)
    assert t1.trial_id == t2.trial_id


def test_new_params_mint_new_trial(ledger):
    t1 = ledger.register("pairs", SCOPE_FREE, PARAMS, VINTAGE_A)
    t2 = ledger.register("pairs", SCOPE_FREE, PARAMS_VARIANT, VINTAGE_A)
    assert t1.trial_id != t2.trial_id
    assert ledger.trial_count() == 2


def test_new_vintage_mints_new_trial(ledger):
    """A new data vintage is a new skip-list key (decay lineage)."""
    t1 = ledger.register("pairs", SCOPE_FREE, PARAMS, VINTAGE_A)
    t2 = ledger.register("pairs", SCOPE_FREE, PARAMS, VINTAGE_B)
    assert t1.trial_id != t2.trial_id


def test_new_scope_mints_new_trial(ledger):
    t1 = ledger.register("pairs", SCOPE_FREE, PARAMS, VINTAGE_A)
    t2 = ledger.register("pairs", SCOPE_PRO, PARAMS, VINTAGE_A)
    assert t1.trial_id != t2.trial_id


def test_skip_list_key_stable_and_vintage_sensitive():
    k1 = skip_list_key("pairs", SCOPE_FREE, PARAMS, VINTAGE_A)
    k2 = skip_list_key("pairs", SCOPE_FREE, dict(PARAMS), VINTAGE_A)
    k3 = skip_list_key("pairs", SCOPE_FREE, PARAMS, VINTAGE_B)
    assert k1 == k2
    assert k1 != k3
    assert len(k1) == 64  # sha256 hex


def test_config_hash_is_sha256_of_canonical_json():
    h = config_hash_of(PARAMS)
    assert len(h) == 64
    assert h == config_hash_of(dict(reversed(list(PARAMS.items()))))


# ------------------------------------------------------------- stage rows
def test_record_stage_pending_to_pass_once(ledger):
    t = ledger.register("pairs", SCOPE_FREE, PARAMS, VINTAGE_A)
    ledger.record_stage(t.trial_id, "s2", "pass", statistic=12.0, p_value=0.001,
                        seed=42)
    row = ledger.stage(t.trial_id, "s2")
    assert row["status"] == "pass"
    assert row["statistic"] == 12.0
    assert row["p_value"] == 0.001
    assert row["seed"] == 42


def test_record_stage_is_append_only(ledger):
    t = ledger.register("pairs", SCOPE_FREE, PARAMS, VINTAGE_A)
    ledger.record_stage(t.trial_id, "s1", "pass")
    with pytest.raises(LedgerError):
        ledger.record_stage(t.trial_id, "s1", "fail")  # already decided
    assert ledger.stage(t.trial_id, "s1")["status"] == "pass"


def test_record_stage_rejects_invalid_inputs(ledger):
    t = ledger.register("pairs", SCOPE_FREE, PARAMS, VINTAGE_A)
    with pytest.raises(LedgerError):
        ledger.record_stage(t.trial_id, "s9", "pass")       # bad stage
    with pytest.raises(LedgerError):
        ledger.record_stage(t.trial_id, "s0", "pending")    # not a decision
    with pytest.raises(LedgerError):
        ledger.record_stage("trial-999999", "s0", "pass")   # unknown trial


def test_unrun_stage_stays_pending_and_cannot_pass(ledger):
    t = ledger.register("pairs", SCOPE_FREE, PARAMS, VINTAGE_A)
    assert ledger.stage(t.trial_id, "s5")["status"] == "pending"


# ------------------------------------------------------------------- decide
def test_decide_killed(ledger):
    t = ledger.register("pairs", SCOPE_FREE, PARAMS, VINTAGE_A)
    ledger.decide(t.trial_id, "killed", "none")
    assert ledger.get(t.trial_id).verdict == "killed"
    assert ledger.get(t.trial_id).survivor_class == "none"


def test_decide_survivor_amber(ledger):
    t = ledger.register("pairs", SCOPE_FREE, PARAMS, VINTAGE_A)
    ledger.decide(t.trial_id, "survivor", "on-watch-amber")
    got = ledger.get(t.trial_id)
    assert got.verdict == "survivor"
    assert got.survivor_class == "on-watch-amber"


def test_decide_validated_green(ledger):
    t = ledger.register("pairs", SCOPE_FREE, PARAMS, VINTAGE_A,
                        parent_trial_id=None)
    ledger.decide(t.trial_id, "survivor", "validated-green")
    assert ledger.get(t.trial_id).survivor_class == "validated-green"


def test_decide_is_append_only(ledger):
    t = ledger.register("pairs", SCOPE_FREE, PARAMS, VINTAGE_A)
    ledger.decide(t.trial_id, "killed", "none")
    with pytest.raises(LedgerError):
        ledger.decide(t.trial_id, "survivor", "validated-green")


def test_decide_rejects_invalid_combinations(ledger):
    t = ledger.register("pairs", SCOPE_FREE, PARAMS, VINTAGE_A)
    with pytest.raises(LedgerError):
        ledger.decide(t.trial_id, "maybe", "none")                 # bad verdict
    with pytest.raises(LedgerError):
        ledger.decide(t.trial_id, "killed", "on-watch-amber")      # kill w/ class
    with pytest.raises(LedgerError):
        ledger.decide(t.trial_id, "survivor", "none")              # survivor w/o class
    with pytest.raises(LedgerError):
        ledger.decide("trial-999999", "killed", "none")            # unknown trial


# ----------------------------------------------------------- stratum_stats
def test_stratum_n_counts_kills_and_registrations(ledger):
    t1 = ledger.register("pairs", SCOPE_FREE, PARAMS, VINTAGE_A)
    ledger.register("pairs", SCOPE_FREE, PARAMS_VARIANT, VINTAGE_A)
    ledger.decide(t1.trial_id, "killed", "none")  # a kill still counts in N
    n, v = ledger.stratum_stats(SCOPE_FREE)
    assert n == 2          # includes the kill
    assert v == 0.0        # no S4 stats recorded yet


def test_strata_never_pooled_across_scopes(ledger):
    ledger.register("pairs", SCOPE_FREE, PARAMS, VINTAGE_A)
    ledger.register("pairs", SCOPE_FREE, PARAMS_VARIANT, VINTAGE_A)
    ledger.register("pairs", SCOPE_PRO, PARAMS, VINTAGE_A)
    n_free, _ = ledger.stratum_stats(SCOPE_FREE)
    n_pro, _ = ledger.stratum_stats(SCOPE_PRO)
    assert n_free == 2
    assert n_pro == 1
    assert ledger.stratum_stats("unknown-scope") == (0, 0.0)


def test_stratum_var_is_variance_of_s4_stats_within_scope(ledger):
    vals = [0.5, 1.0, 1.5, 2.0]
    for i, v in enumerate(vals):
        params = {"k": i}
        t = ledger.register("pairs", SCOPE_FREE, params, VINTAGE_A)
        ledger.record_stage(t.trial_id, "s4", "pass", statistic=v)
    ledger.update_var_stat(SCOPE_FREE)
    n, v = ledger.stratum_stats(SCOPE_FREE)
    assert n == 4
    # #1: s4 statistics are stored ANNUALIZED (sr_ann); update_var_stat
    # converts to per-bar (÷ sqrt(252)) BEFORE the variance, so V is per-bar
    # SR^2 — matching the per-bar sr_hat fed to stats.dsr (deck-path
    # convention). Variance scales by 1/BARS_PER_YEAR under the conversion.
    import math as _math

    from sift.registry import BARS_PER_YEAR as _BPY
    assert v == pytest.approx(sample_variance(vals) / _BPY)
    # sample variance of [0.5, 1.0, 1.5, 2.0] = 5/12 = 0.41666... annualized;
    # per-bar: (5/12)/252
    assert v == pytest.approx(5.0 / 12.0 / _BPY, abs=1e-12)
    assert v == pytest.approx(sample_variance([x / _math.sqrt(_BPY) for x in vals]))


def test_stratum_var_not_pooled_across_scopes(ledger):
    t_a = ledger.register("pairs", SCOPE_FREE, {"k": 0}, VINTAGE_A)
    t_b = ledger.register("pairs", SCOPE_PRO, {"k": 0}, VINTAGE_A)
    ledger.record_stage(t_a.trial_id, "s4", "pass", statistic=1.0)
    ledger.record_stage(t_b.trial_id, "s4", "pass", statistic=100.0)
    ledger.update_var_stat(SCOPE_FREE)
    ledger.update_var_stat(SCOPE_PRO)
    # single S4 stat per scope -> variance 0 in each, NOT the pooled 4950
    assert ledger.stratum_stats(SCOPE_FREE)[1] == pytest.approx(0.0)
    assert ledger.stratum_stats(SCOPE_PRO)[1] == pytest.approx(0.0)


def test_update_var_stat_ignores_null_and_other_stages(ledger):
    t = ledger.register("pairs", SCOPE_FREE, {"k": 0}, VINTAGE_A)
    ledger.record_stage(t.trial_id, "s2", "pass", statistic=999.0)  # not s4
    ledger.update_var_stat(SCOPE_FREE)
    assert ledger.stratum_stats(SCOPE_FREE)[1] == 0.0
    ledger.record_stage(t.trial_id, "s4", "fail", statistic=None)
    ledger.update_var_stat(SCOPE_FREE)
    assert ledger.stratum_stats(SCOPE_FREE)[1] == 0.0


def test_trials_in_scope_lists_all(ledger):
    ledger.register("pairs", SCOPE_FREE, PARAMS, VINTAGE_A)
    ledger.register("leadlag", SCOPE_FREE, PARAMS, VINTAGE_A)
    ledger.register("pairs", SCOPE_PRO, PARAMS, VINTAGE_A)
    ids = [t.trial_id for t in ledger.trials_in_scope(SCOPE_FREE)]
    assert ids == ["trial-000001", "trial-000002"]


# ------------------------------------------------------------- sample_var
def test_sample_variance():
    assert sample_variance([]) == 0.0
    assert sample_variance([3.0]) == 0.0
    assert sample_variance([1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert math.isclose(sample_variance([2.0, 4.0]), 2.0)
