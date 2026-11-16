"""Tests for the pure gauntlet statistics (sift/gauntlet/stats.py)."""

import math

import pytest

from sift.gauntlet import stats


class TestBlockLength:
    def test_free_scope_T1260(self):
        assert stats.block_length(1260) == 11  # ceil(1260^(1/3))

    def test_registry_depth_T4032(self):
        assert stats.block_length(4032) == 16

    def test_min_floor(self):
        assert stats.block_length(8) == stats.MIN_BLOCK_LEN == 5

    def test_invalid(self):
        with pytest.raises(ValueError):
            stats.block_length(0)


class TestPermutationPValue:
    def test_formula(self):
        # p = (1 + #extreme) / (B + 1)
        assert stats.permutation_pvalue(3.0, [3.5, 1.0, -3.2, 0.5]) == (1 + 2) / 5

    def test_granularity(self):
        # B=1000 -> granularity 1/1001 ~= 0.001
        p = stats.permutation_pvalue(10.0, [0.1] * 1000)
        assert p == pytest.approx(1 / 1001)

    def test_empty(self):
        with pytest.raises(ValueError):
            stats.permutation_pvalue(1.0, [])


class TestExpectedMaxSharpe:
    """Worked numbers from the FL design doc SS2 S3 (recomputed)."""

    def test_free_scope_N1000(self):
        emax = stats.expected_max_sharpe(1000, 1 / 1260)
        assert emax * stats.ANNUALIZATION_DAILY == pytest.approx(1.46, abs=0.01)

    def test_registry_depth_N5000(self):
        emax = stats.expected_max_sharpe(5000, 1 / 4032)
        assert emax * stats.ANNUALIZATION_DAILY == pytest.approx(0.92, abs=0.01)

    def test_small_N(self):
        assert stats.expected_max_sharpe(12, 1 / 1260) * stats.ANNUALIZATION_DAILY == pytest.approx(0.74, abs=0.01)


class TestDSR:
    def test_candidate_bar_free_scope(self):
        # ann. SR ~= 2.19 needed for DSR >= 0.95 at T=1260, N=1000
        sr = 2.19 / stats.ANNUALIZATION_DAILY
        dsr = stats.deflated_sharpe_ratio(sr, 1000, 1 / 1260, 1260)
        assert dsr >= 0.95
        assert dsr == pytest.approx(0.9505, abs=0.01)

    def test_candidate_bar_registry_depth(self):
        # ann. SR ~= 1.33 needed at T=4032, N=5000 — full depth has a LOWER bar
        sr = 1.34 / stats.ANNUALIZATION_DAILY
        dsr = stats.deflated_sharpe_ratio(sr, 5000, 1 / 4032, 4032)
        assert dsr >= 0.95

    def test_psr_monotone_in_sr(self):
        sr_star = stats.expected_max_sharpe(10, 1 / 1260)
        assert stats.probabilistic_sharpe_ratio(0.2, sr_star, 1260) > \
               stats.probabilistic_sharpe_ratio(0.1, sr_star, 1260)

    def test_skew_kurt_deflate(self):
        # negative skew lowers PSR at same SR (g3 < 0 shrinks the denominator term)
        sr_star = stats.expected_max_sharpe(10, 1 / 1260)
        p_norm = stats.probabilistic_sharpe_ratio(0.15, sr_star, 1260, 0.0, 0.0)
        p_neg_skew = stats.probabilistic_sharpe_ratio(0.15, sr_star, 1260, -1.0, 3.0)
        assert p_neg_skew < p_norm


class TestMinTRL:
    def test_values(self):
        # per-bar SR 0.1 -> MinTRL ~= 271 bars; 0.05 -> ~= 1083 bars
        assert stats.min_trl(0.1) == pytest.approx(270.9, abs=0.5)
        assert stats.min_trl(0.05) == pytest.approx(1082.5, abs=0.5)

    def test_nonpositive_sr(self):
        assert stats.min_trl(0.0) == math.inf
        assert stats.min_trl(-0.1) == math.inf


class TestBH:
    def test_qvalues(self):
        p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]
        q = stats.benjamini_hochberg(p, q=0.10)
        # BH adjusted q-value for the smallest p is p_(1)*m/1 = 0.001*8
        assert q[0] == pytest.approx(0.008)
        # step-up rejection set: largest k with p_(k) <= k/m*q (m=8, q=0.10):
        # p_(5)=0.042 <= 5/8*0.10=0.0625, p_(6)=0.06 <= 0.075, p_(7)=0.074 <= 0.0875,
        # p_(8)=0.205 > 0.10  ->  k = 7
        order = sorted(range(len(p)), key=lambda i: p[i])
        k = 0
        for rank in range(1, len(p) + 1):
            if p[order[rank - 1]] <= rank / len(p) * 0.10:
                k = rank
        assert k == 7
        rejects = {order[r] for r in range(k)}
        assert rejects == {0, 1, 2, 3, 4, 5, 6}

    def test_empty(self):
        assert stats.benjamini_hochberg([], 0.1) == []

    def test_all_null(self):
        q = stats.benjamini_hochberg([0.5, 0.6, 0.7, 0.8], q=0.10)
        assert all(v > 0.10 for v in q)


class TestTAndSharpe:
    def test_t_zero_mean(self):
        assert abs(stats.t_statistic([1.0, -1.0, 1.0, -1.0])) < 1e-9

    def test_sharpe(self):
        r = [0.01] * 10
        assert stats.sharpe_ratio(r) == 0.0  # zero variance
        r2 = [0.01, 0.02, 0.005, 0.015]
        assert stats.sharpe_ratio(r2) > 0

    def test_skew_kurt_normal(self):
        # large normal sample -> g3, g4 near 0
        import numpy as np
        rng = np.random.Generator(np.random.PCG64(1))
        r = rng.standard_normal(20000).tolist()
        g3, g4 = stats.mean_skew_kurtosis(r)
        assert abs(g3) < 0.1 and abs(g4) < 0.2


class TestPairsMachinery:
    def test_ols_recovers_beta(self, f_pair):
        x, y, beta_star, _ = f_pair
        assert stats.ols_hedge_ratio(x, y) == pytest.approx(beta_star, abs=0.05)

    def test_half_life_recovers(self, f_pair):
        x, y, _, phi_star = f_pair
        beta = stats.ols_hedge_ratio(x, y)
        spread = [y[i] - beta * x[i] for i in range(len(x))]
        hl = stats.half_life(spread)
        hl_star = math.log(0.5) / math.log(phi_star)
        assert hl == pytest.approx(hl_star, rel=0.20)

    def test_strategy_t_positive_on_f_pair(self, f_pair):
        x, y, _, _ = f_pair
        res = stats.pair_spread_strategy_returns(x, y)
        assert res["t_stat"] > 3.5

    def test_walk_forward_positive_on_f_pair(self, f_pair):
        x, y, _, _ = f_pair
        wf = stats.walk_forward_net_returns(x, y, half_spread=0.01)
        assert wf["t_net"] > 3.5
        assert wf["sharpe_per_bar"] > 0
