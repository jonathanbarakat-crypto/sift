"""Tests for the gauntlet stage runner (S0-S4 wiring + ledger recording) and
the planted fixtures F-PAIR (must pass) and F-NULL (must fail S2/S3)."""

import pytest

from sift.gauntlet import stats
from sift.gauntlet.runner import (
    GauntletConfig,
    SQLiteLedger,
    run_gauntlet_s0_s4,
    stage_s2_permutation,
    stage_s3_multiplicity,
)
from tests.fixtures import F_NULL_N, make_f_null, make_f_pair


@pytest.fixture()
def ledger(tmp_path):
    return SQLiteLedger(tmp_path / "ledger.db")


@pytest.fixture(scope="module")
def f_pair():
    return make_f_pair()


class TestLedgerSchema:
    def test_register_and_dedupe(self, ledger):
        tid1 = ledger.register_trial("pairs", "free_5y_top1k", {"bars": 1260}, {"a": 1}, "v0")
        tid2 = ledger.register_trial("pairs", "free_5y_top1k", {"bars": 1260}, {"a": 1}, "v0")
        tid3 = ledger.register_trial("pairs", "free_5y_top1k", {"bars": 1260}, {"a": 2}, "v0")
        assert tid1 == tid2          # skip-list: identical config -> same id
        assert tid3 != tid1
        n, v = ledger.stratum_stats("free_5y_top1k")
        assert n == 2                # N counts unique trials incl. future kills

    def test_stage_rows_pending_then_recorded(self, ledger):
        tid = ledger.register_trial("pairs", "free_5y_top1k", {"bars": 1260}, {}, "v0")
        ledger.record_stage(tid, "s1", "pass", 2.5, 2.0, None, {"abs_t": 2.5})
        trial = ledger.load_trial(tid)
        assert trial.stages["s1"].status == "pass"
        assert trial.stages["s1"].statistic == 2.5
        assert trial.stages["s2"].status == "pending"
        ledger.set_verdict(tid, "survivor", "on-watch-amber")
        assert ledger.load_trial(tid).verdict == "survivor"

    def test_strata_never_pooled(self, ledger):
        ledger.register_trial("pairs", "free_5y_top1k", {"bars": 1260}, {"i": 1}, "v0")
        ledger.register_trial("pairs", "pro_full", {"bars": 4032}, {"i": 1}, "v0")
        n_free, _ = ledger.stratum_stats("free_5y_top1k")
        n_pro, _ = ledger.stratum_stats("pro_full")
        assert n_free == 1 and n_pro == 1  # separate denominators


class TestS0Calibration:
    def test_f_pair_reproduces(self, f_pair):
        from sift.gauntlet.runner import stage_s0_calibrate
        x, y, beta_star, phi_star = f_pair
        s0 = stage_s0_calibrate(x, y, beta_star, phi_star)
        assert s0.status == "pass"
        assert s0.detail["verdict"] == "reproduced"

    def test_wrong_fixture_fails(self):
        from sift.gauntlet.runner import stage_s0_calibrate
        import numpy as np
        rng = np.random.Generator(np.random.PCG64(0))
        x = (100 + np.cumsum(rng.standard_normal(300))).tolist()
        y = (100 + np.cumsum(rng.standard_normal(300))).tolist()
        s0 = stage_s0_calibrate(x, y, beta_star=2.0, phi_star=0.85)
        assert s0.status == "fail"


class TestS2Permutation:
    def test_f_pair_passes(self, f_pair):
        x, y, _, _ = f_pair
        s2 = stage_s2_permutation(x, y, B=200, seed=1)
        assert s2.p_value is not None and s2.p_value <= 0.05
        assert s2.detail["block_len"] == stats.block_length(len(x))

    def test_f_null_fails(self):
        pairs = make_f_null(n=30, T=250)
        passes = 0
        for x, y in pairs:
            s2 = stage_s2_permutation(x, y, B=200, seed=7)
            if s2.p_value is not None and s2.p_value <= 0.05:
                passes += 1
        # type-I control: at most ~5-10% of pure-noise pairs may pass S2
        assert passes / len(pairs) <= 0.10

    def test_seeded_deterministic(self, f_pair):
        x, y, _, _ = f_pair
        a = stage_s2_permutation(x, y, B=50, seed=42)
        b = stage_s2_permutation(x, y, B=50, seed=42)
        assert a.p_value == b.p_value


class TestS3Multiplicity:
    def test_f_pair_dsr_passes_at_small_N(self, ledger, f_pair):
        x, y, _, _ = f_pair
        tid = ledger.register_trial("pairs", "free_5y_top1k", {"bars": len(x)}, {}, "v0")
        obs = stats.pair_spread_strategy_returns(x, y)
        s3 = stage_s3_multiplicity(ledger, tid, "pairs", "free_5y_top1k",
                                   obs["sharpe_per_bar"], len(x), obs["returns"])
        assert s3.status == "pass"
        assert s3.detail["N_stratum"] == 1
        assert s3.detail["Vbar_stratum"] == pytest.approx(1 / len(x))

    def test_null_sr_fails_at_large_N(self, ledger):
        # a mediocre SR cannot clear DSR>=0.95 when the stratum is large
        s3 = stage_s3_multiplicity(ledger, "t-x", "pairs", "free_5y_top1k",
                                   sr_hat_per_bar=0.03, T=1260, returns=[0.001] * 1260)
        # N=0 -> treated as N=1; still compute honestly
        assert s3.detail["N_stratum"] >= 1
        # force a big stratum via ledger
        for i in range(999):
            ledger.register_trial("pairs", "free_5y_top1k", {"bars": 1260}, {"i": i}, "v0")
        s3b = stage_s3_multiplicity(ledger, "t-y", "pairs", "free_5y_top1k",
                                    sr_hat_per_bar=0.03, T=1260, returns=[0.001] * 1260)
        assert s3b.status == "fail"  # ann SR ~0.48 cannot beat E[max] at N=1000

    def test_denominators_from_stratum_only(self, ledger):
        # register trials in another scope; they must not inflate this stratum's N
        for i in range(500):
            ledger.register_trial("pairs", "pro_full", {"bars": 4032}, {"i": i}, "v0")
        ledger.register_trial("pairs", "free_5y_top1k", {"bars": 1260}, {"i": 0}, "v0")
        n, _ = ledger.stratum_stats("free_5y_top1k")
        assert n == 1


class TestS4WalkForward:
    def test_f_pair_passes(self, f_pair):
        from sift.gauntlet.runner import stage_s4_walk_forward, DEFAULT_COST_PARAMS
        x, y, _, _ = f_pair
        s4 = stage_s4_walk_forward(x, y, len(x), tau_cal=3.5, cost_params=DEFAULT_COST_PARAMS)
        assert s4.status == "pass", s4.detail
        assert s4.detail["t_net"] >= 3.5
        assert s4.detail["min_trl"] <= len(x)

    def test_null_fails_tau(self):
        from sift.gauntlet.runner import stage_s4_walk_forward, DEFAULT_COST_PARAMS
        pairs = make_f_null(n=20, T=250)
        n_pass = 0
        for x, y in pairs:
            s4 = stage_s4_walk_forward(x, y, len(x), tau_cal=3.5, cost_params=DEFAULT_COST_PARAMS)
            n_pass += s4.status == "pass"
        assert n_pass / len(pairs) <= 0.10


class TestGauntletEndToEnd:
    def test_f_pair_survives_and_is_recorded(self, ledger, f_pair):
        x, y, beta_star, phi_star = f_pair
        trial = run_gauntlet_s0_s4(
            ledger, x, y,
            scope_id="free_5y_top1k",
            window={"bars": len(x), "basis": "registered_scope"},
            params={"family": "pairs"},
            perm_seed=20261115,
            s0_fixture=(x, y, beta_star, phi_star),
            config=GauntletConfig.for_scope("free_5y_top1k"),
        )
        assert trial.verdict == "survivor"
        assert trial.survivor_class == "on-watch-amber"  # S5 automatic, task 3/8
        for s in ("s0", "s1", "s2", "s3", "s4"):
            assert trial.stages[s].status == "pass", (s, trial.stages[s])
        assert trial.stages["s2"].detail["seed"] == 20261115
        assert trial.stages["s3"].detail["N_stratum"] == 1
        # ledger rows persisted
        reloaded = ledger.load_trial(trial.trial_id)
        assert reloaded.verdict == "survivor"

    def test_f_null_killed_at_s2_or_s3(self, tmp_path):
        # F-NULL acceptance: 200 independent random walks in ONE shared
        # scope-stratified ledger (the real deployment shape — the ledger
        # counts every trial, so N=200 deflates DSR). Observed admission must
        # sit inside the FDR control band [0, q + 3*sqrt(q(1-q)/m)], q=0.10.
        pairs = make_f_null(n=F_NULL_N, T=250)
        led = SQLiteLedger(tmp_path / "fnull.db")
        x, y, beta_star, phi_star = make_f_pair()  # golden fixture for S0
        admitted = 0
        kill_stages = {}
        for i, (nx, ny) in enumerate(pairs):
            trial = run_gauntlet_s0_s4(
                led, nx, ny,
                scope_id="free_5y_top1k",
                window={"bars": len(nx), "basis": "registered_scope"},
                params={"i": i},
                perm_seed=1000 + i,
                s0_fixture=(x, y, beta_star, phi_star),
                config=GauntletConfig.for_scope("free_5y_top1k"),
            )
            if trial.verdict == "survivor":
                admitted += 1
            else:
                for s in ("s1", "s2", "s3", "s4"):
                    if trial.stages[s].status == "fail":
                        kill_stages[s] = kill_stages.get(s, 0) + 1
                        break
        assert admitted / F_NULL_N <= 0.10 + 3 * (0.10 * 0.90 / F_NULL_N) ** 0.5
        assert admitted <= 2  # 200 pure-noise walks: essentially all killed
        # kills happen at the gauntlet's decision stages: S1 screen + S2 null +
        # S3 DSR account for essentially all of them; S4 cost gate is rarely the
        # binding one on pure noise
        assert kill_stages.get("s1", 0) + kill_stages.get("s2", 0) + kill_stages.get("s3", 0) >= 0.99 * (F_NULL_N - admitted)
        assert kill_stages.get("s2", 0) > 0  # the permutation null does real work

    def test_s5_row_stays_pending(self, ledger, f_pair):
        # S5 is task 3/8's stage — the runner must never mark it pass itself
        x, y, beta_star, phi_star = f_pair
        trial = run_gauntlet_s0_s4(
            ledger, x, y, scope_id="free_5y_top1k", window={"bars": len(x)},
            params={}, perm_seed=5, s0_fixture=(x, y, beta_star, phi_star),
        )
        assert trial.stages["s5"].status == "pending"

    def test_s0_not_certified_without_fixture(self, ledger, f_pair):
        # A stage that is not run cannot pass: no golden fixture -> killed at S0.
        x, y, _, _ = f_pair
        trial = run_gauntlet_s0_s4(
            ledger, x, y, scope_id="free_5y_top1k", window={"bars": len(x)},
            params={"nofix": 1}, perm_seed=9,
        )
        assert trial.verdict == "killed"
        assert trial.stages["s0"].status == "fail"
        assert trial.stages["s0"].detail["verdict"] == "not_certified"
        assert trial.stages["s1"].status == "pending"  # never reached

    def test_ledger_is_append_only(self, ledger):
        tid = ledger.register_trial("pairs", "free_5y_top1k", {"bars": 1260}, {}, "v0")
        ledger.record_stage(tid, "s1", "pass", 2.5, 2.0, None, {})
        with pytest.raises(RuntimeError):
            ledger.record_stage(tid, "s1", "fail", 9.9, 2.0, None, {})  # rewrite blocked
        ledger.set_verdict(tid, "killed", None)
        with pytest.raises(RuntimeError):
            ledger.set_verdict(tid, "survivor", "on-watch-amber")  # verdict immutable
        assert ledger.load_trial(tid).stages["s1"].statistic == 2.5
        assert ledger.load_trial(tid).verdict == "killed"
