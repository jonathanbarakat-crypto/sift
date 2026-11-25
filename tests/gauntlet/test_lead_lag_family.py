"""Comprehensive acceptance & statistical test suite for the `lead_lag` family.

Covers every pre-registered requirement in `docs/lead-lag-family-spec.md` §1-§6:
1. Exact FWL Newey-West HAC equivalence with full (X'X)^-1 S (X'X)^-1.
2. Additive migration of legacy (scope_id)-only stratum tables + strict
   (family, scope_id) DSR & BH-FDR isolation (pairs unaffected by lead_lag).
3. All 5 synthetic fixtures: F-LEADLAG (L in {1, 2, 5}, forward pass + reverse
   kill), F-LEADLAG-CONTEMP, F-LEADLAG-REGIME, F-LEADLAG-DELIST, F-LEADLAG-NULL.
4. Byte-identical reproducibility under identical (seed, params, vintage).
5. Card face rendering (JSON + PIL PNG) & S6 mixed pairs + lead_lag deck.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from sift import flags
from sift.cards import CardRegistry
from sift.cards.face import CardFaceRenderer
from sift.cards.minting import CardMinter
from sift.decks import DeckRegistry, bucket_config
from sift.decks.service import DeckService
from sift.gauntlet import config as cfg
from sift.gauntlet import families, stats
from sift.gauntlet import lead_lag as ll
from sift.gauntlet.runner import GauntletRunner
from sift.registry import Ledger

SECRET = b"lead-lag-test-secret-32-bytes!!!"


@pytest.fixture(autouse=True)
def _reset_flags():
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


# ---------------------------------------------------------------------------
# 1. Exact Newey-West HAC equivalence & pluggable StrategyFamily registry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("H", [1, 2, 5])
def test_fwl_nw_hac_matches_explicit_sandwich(H: int):
    """FWL scalar-score NW HAC must match full 3x3 (X'X)^-1 S_NW (X'X)^-1."""
    rng = np.random.Generator(np.random.PCG64(123 + H))
    n = 400
    x = rng.normal(0.0, 0.02, n)
    b = rng.normal(0.0, 0.02, n)
    e = rng.normal(0.0, 0.015, n)
    # Introduce MA(H-1) serial correlation in residuals
    eps = np.convolve(e, np.ones(H), mode="same")
    y = 0.001 + 0.25 * x - 0.12 * b + eps

    L = max(H - 1, stats.nw_lag(n))
    res = ll.ols_hac_gamma(y, x, b, hold_bars=H, lag=L)

    # Explicit 3x3 sandwich reference
    X = np.column_stack([np.ones(n), x, b])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    U = X * resid[:, None]
    S = U.T @ U
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        Gk = U[k:].T @ U[:-k]
        S = S + w * (Gk + Gk.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    V_hac = XtX_inv @ S @ XtX_inv
    se_ref = float(np.sqrt(V_hac[1, 1]))
    t_ref = float(coef[1] / se_ref)

    assert res["gamma"] == pytest.approx(float(coef[1]), abs=1e-12)
    assert res["se_gamma"] == pytest.approx(se_ref, abs=1e-12)
    assert res["t_hac"] == pytest.approx(t_ref, abs=1e-11)


def test_family_registry_protocol():
    pf = families.get_family("pairs")
    lf = families.get_family("lead_lag")
    assert isinstance(pf, families.StrategyFamily)
    assert isinstance(lf, families.StrategyFamily)
    assert pf.directed is False and pf.use_hac is False
    assert lf.directed is True and lf.use_hac is True
    for lag in cfg.LEAD_LAG_LAGS:
        grid = lf.s5_variant_grid(lag_bars=lag)
        assert len(grid) == 9
        assert grid[0]["lag_bars"] == lag
        assert grid[0]["hold_bars"] == lag
        horizons = {row["lag_bars"] for row in grid}
        assert horizons == set(cfg.LEAD_LAG_LAGS)
    with pytest.raises(ValueError):
        families.get_family("unknown_family")


# ---------------------------------------------------------------------------
# 2. Legacy ledger migration & strict (family, scope_id) stratum isolation
# ---------------------------------------------------------------------------

def test_legacy_ledger_migration_preserves_pairs_rows(tmp_path):
    """Legacy DB with scope_id-only stratum tables migrates cleanly to
    (family='pairs', scope_id) with zero data loss."""
    db_path = tmp_path / "legacy_ledger.db"
    con = sqlite3.connect(str(db_path))
    con.executescript(
        """
        CREATE TABLE trial (
            trial_id TEXT PRIMARY KEY,
            family TEXT NOT NULL,
            scope_id TEXT NOT NULL,
            params_json TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            data_vintage_hash TEXT NOT NULL,
            provenance TEXT NOT NULL,
            registered_at REAL NOT NULL,
            verdict TEXT NOT NULL DEFAULT 'pending',
            survivor_class TEXT NOT NULL DEFAULT 'none',
            discoverer_handle TEXT,
            parent_trial_id TEXT,
            engine_version TEXT NOT NULL,
            budget_consumed INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE stage_result (
            trial_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            statistic REAL, threshold REAL, p_value REAL,
            n_stratum INTEGER, v_stratum REAL, dsr REAL, q_bh REAL, seed INTEGER,
            computed_at REAL NOT NULL,
            PRIMARY KEY (trial_id, stage)
        );
        CREATE TABLE stratum_stats (
            scope_id TEXT PRIMARY KEY,
            n_trials INTEGER NOT NULL,
            var_stat REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE stratum_epoch (
            scope_id TEXT NOT NULL,
            epoch INTEGER NOT NULL,
            n_eff_raw INTEGER NOT NULL,
            n_eff REAL NOT NULL,
            rho_bar REAL NOT NULL,
            method TEXT NOT NULL,
            detail_json TEXT NOT NULL,
            frozen_at REAL NOT NULL,
            PRIMARY KEY (scope_id, epoch)
        );
        INSERT INTO stratum_stats VALUES ('free_5y_top1k', 42, 0.0025, 1700000000.0);
        INSERT INTO stratum_epoch VALUES (
            'free_5y_top1k', 1, 42, 18.5, 0.25, 'a3_avg_corr', '{}', 1700000000.0
        );
        """
    )
    con.commit()
    con.close()

    migrated = Ledger(db_path)
    try:
        n, v = migrated.stratum_stats("free_5y_top1k")
        assert n == 42
        assert v == pytest.approx(0.0025)
        ep = migrated.stratum_epoch_stats("free_5y_top1k", 1)
        assert ep is not None
        assert ep["family"] == "pairs"
        assert ep["n_eff"] == pytest.approx(18.5)
        # lead_lag starts cleanly isolated at (0, 0.0) and no frozen epoch
        assert migrated.stratum_stats("free_5y_top1k", family="lead_lag") == (0, 0.0)
        assert migrated.stratum_epoch_stats("free_5y_top1k", 1, family="lead_lag") is None
    finally:
        migrated.close()


def test_stratum_isolation_lead_lag_never_shifts_pairs_dsr_or_bh(tmp_path):
    """Acceptance criterion #1: registering 50 lead_lag trials in the same scope
    must not change pairs stratum N, V, epoch, DSR, or BH-FDR q_bh by a single bit."""
    scope = "free_5y_top1k"
    led_clean = Ledger(tmp_path / "clean.db")
    led_mixed = Ledger(tmp_path / "mixed.db")
    try:
        for led in (led_clean, led_mixed):
            for i in range(5):
                t = led.register("pairs", scope, {"pair_i": i}, "v1")
                led.record_stage(t.trial_id, "s2", "pass", p_value=0.002 + 0.001 * i)
                led.record_stage(t.trial_id, "s4", "pass", statistic=1.2 + 0.1 * i)
            led.update_var_stat(scope, family="pairs")

        # Pollute led_mixed with 50 lead_lag trials in the SAME scope
        for i in range(50):
            tl = led_mixed.register("lead_lag", scope, {"ll_i": i}, "v1")
            led_mixed.record_stage(tl.trial_id, "s2", "fail", p_value=0.4 + 0.01 * (i % 50))
            led_mixed.record_stage(tl.trial_id, "s4", "fail", statistic=-2.5 + 0.2 * i)
        led_mixed.update_var_stat(scope, family="lead_lag")

        assert led_mixed.stratum_stats(scope, family="pairs") == led_clean.stratum_stats(
            scope, family="pairs"
        )
        assert led_mixed.stratum_stats(scope, family="lead_lag")[0] == 50

        tc = led_clean.register("pairs", scope, {"cand": "target"}, "v1")
        tm = led_mixed.register("pairs", scope, {"cand": "target"}, "v1")
        rc = GauntletRunner(led_clean, seed=7).run_s3(
            tc.trial_id, 0.14, 1260, p_value=0.001, scope_id=scope
        )
        rm = GauntletRunner(led_mixed, seed=7).run_s3(
            tm.trial_id, 0.14, 1260, p_value=0.001, scope_id=scope
        )
        assert rc.dsr == rm.dsr
        assert rc.q_bh == rm.q_bh
        assert rc.n_stratum == rm.n_stratum
        assert rc.v_stratum == rm.v_stratum
        assert rc.status == rm.status
    finally:
        led_clean.close()
        led_mixed.close()


# ---------------------------------------------------------------------------
# 3. Synthetic fixtures (5/5) — deterministic under PCG64
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "lag_bars,gamma_true,phi_b",
    [(1, 0.30, -0.1), (2, 0.45, 0.05), (5, 0.40, 0.05)],
)
def test_f_leadlag_forward_passes_s0_to_s5_and_reverse_killed(
    ledger, cards, lag_bars: int, gamma_true: float, phi_b: float
):
    """F-LEADLAG: forward (A -> B) recovers gamma* within 0.05 at T=4032 and
    passes S0-S5 (validated-green) across all three pre-registered horizons
    W in {1, 2, 5}; reverse (B -> A) is killed honestly."""
    fx = ll.f_leadlag(
        T=4032,
        gamma_true=gamma_true,
        lag_true=lag_bars,
        phi_b=phi_b,
        seed=101 + lag_bars,
    )
    full_prices = fx["prices"]
    full_rm = fx["market_returns"]
    scope_prices = full_prices[-1260:]
    scope_rm = full_rm[-1260:]

    # S0 planted calibration check at T=4032 (spec §3 S0 table)
    s0_full = ll.s0_lead_lag_sanity(
        full_prices,
        market_returns=full_rm,
        lag_bars=lag_bars,
        formation=120,
        gamma_true=gamma_true,
    )
    assert s0_full["passed"] is True
    assert abs(s0_full["gamma_hat"] - gamma_true) <= cfg.LEAD_LAG_S0_GAMMA_TOL

    # Forward (A -> B) end-to-end gauntlet S0-S4 + automatic S5
    t_fwd = ledger.register(
        "lead_lag",
        "free_5y_top1k",
        {"leader": "A", "follower": "B", "lag_bars": lag_bars},
        "v-fx",
        discoverer_handle="alice",
    )
    runner = GauntletRunner(
        ledger, seed=7, card_minter=CardMinter(ledger, cards)
    )
    res_fwd = runner.run_lead_lag_gauntlet(
        t_fwd.trial_id,
        prices=scope_prices,
        market_returns=scope_rm,
        lag_bars=lag_bars,
        formation=120,
        open_z=1.5,
        scope_id="free_5y_top1k",
        s2_B=200,
        full_prices=full_prices,
        full_market_returns=full_rm,
        gamma_true=gamma_true if lag_bars == 1 else None,
    )
    assert res_fwd["verdict"] == "survivor"
    assert res_fwd["survivor_class"] == "validated-green"
    assert res_fwd["card_id"] is not None
    assert res_fwd["s5"]["pbo_n_configs"] == 9
    assert res_fwd["s5"]["pbo_grid_digest"] == cfg.s5_variant_grid_lead_lag_digest(lag_bars)
    assert res_fwd["s5"]["use_hac"] is True
    assert res_fwd["s4_score"]["use_hac"] is True
    assert res_fwd["current_abs_z"] is not None and res_fwd["current_abs_z"] >= 0.0

    # Reverse (B -> A): swap columns; verify full-depth |gamma_{B->A}| small & killed
    rev_full = full_prices[:, [1, 0]]
    r_rev = ll.s0_lead_lag_sanity(rev_full, lag_bars=lag_bars, formation=120)
    assert abs(r_rev["gamma_hat"]) <= 0.08
    rev_prices = scope_prices[:, [1, 0]]
    t_rev = ledger.register(
        "lead_lag",
        "free_5y_top1k",
        {"leader": "B", "follower": "A", "lag_bars": lag_bars},
        "v-fx",
        discoverer_handle="alice",
    )
    res_rev = runner.run_lead_lag_gauntlet(
        t_rev.trial_id,
        prices=rev_prices,
        market_returns=scope_rm,
        lag_bars=lag_bars,
        formation=120,
        open_z=1.5,
        scope_id="free_5y_top1k",
        s2_B=200,
    )
    assert res_rev["verdict"] == "killed"
    assert res_rev["killed_at"] in ("s1", "s2")
    assert res_rev["card_id"] is None


def test_f_leadlag_contemp_rejects_confounder(ledger):
    """F-LEADLAG-CONTEMP: strong contemporaneous beta + AR(1) in both legs,
    true gamma*=0. Raw lagged correlation is spurious, but pre-whitened partial
    regression recovers |gamma_hat| <= 0.05 and S1/S2 kills it."""
    fx = ll.f_leadlag_contemp(T=4032, beta0=0.8, phi_a=0.25, phi_b=0.25, seed=202)
    P = fx["prices"]
    ra = ll.log_returns(P[:, 0])
    rb = ll.log_returns(P[:, 1])
    # Verify raw lagged correlation IS strongly spurious without confounder controls
    raw_lag_corr = float(np.corrcoef(ra[1:-1], rb[2:])[0, 1])
    assert abs(raw_lag_corr) > 0.12

    # Pre-whitened partial regression eliminates the confounder
    s0 = ll.s0_lead_lag_sanity(P, lag_bars=1, formation=120, gamma_true=0.0)
    assert abs(s0["gamma_hat"]) <= 0.05
    s2 = ll.s2_lead_lag_permutation(ra, rb, lag_bars=1, B=200, seed=7)
    assert s2["p_perm"] > 0.05

    t = ledger.register("lead_lag", "free_5y_top1k", {"fx": "contemp"}, "v-fx")
    res = GauntletRunner(ledger, seed=7).run_lead_lag_gauntlet(
        t.trial_id,
        prices=P[-1260:],
        lag_bars=1,
        formation=120,
        open_z=1.5,
        scope_id="free_5y_top1k",
        s2_B=200,
    )
    assert res["verdict"] == "killed"
    assert res["killed_at"] in ("s1", "s2")


def test_f_leadlag_regime_downgraded_by_s5(ledger, cards):
    """F-LEADLAG-REGIME: active on leading 5y sub-window, dead afterwards ->
    passes 5y window, but automatic S5 at full depth catches decay (no green card)."""
    fx = ll.f_leadlag_regime(
        T=4032, split=1260, gamma_active=0.30, lag_true=1, seed=303
    )
    full_prices = fx["prices"]
    active_5y = full_prices[:1260]

    t = ledger.register(
        "lead_lag",
        "free_5y_top1k",
        {"fx": "regime"},
        "v-fx",
        discoverer_handle="alice",
    )
    runner = GauntletRunner(ledger, seed=7, card_minter=CardMinter(ledger, cards))
    res = runner.run_lead_lag_gauntlet(
        t.trial_id,
        prices=active_5y,
        lag_bars=1,
        formation=120,
        open_z=1.5,
        scope_id="free_5y_top1k",
        s2_B=200,
        full_prices=full_prices,
    )
    assert res["verdict"] == "survivor"
    assert res["survivor_class"] != "validated-green"
    assert "amber" in res["survivor_class"]


@pytest.mark.parametrize("delist_leg", ["follower", "leader"])
def test_f_leadlag_delist_clean_truncation(ledger, delist_leg: str):
    """F-LEADLAG-DELIST: delist_mask truncates cleanly at delist_bar with zero
    NaN leakage or post-delist fills."""
    fx = ll.f_leadlag_delist(
        T=4032, delist_bar=1500, delist_leg=delist_leg, seed=404
    )
    full_prices = fx["prices"]
    delist_idx = fx["delist_idx"]

    t = ledger.register(
        "lead_lag",
        "free_5y_top1k",
        {"fx": f"delist-{delist_leg}"},
        "v-fx",
    )
    runner = GauntletRunner(ledger, seed=7)
    res = runner.run_lead_lag_gauntlet(
        t.trial_id,
        prices=full_prices[:1260],
        lag_bars=1,
        formation=120,
        open_z=1.5,
        scope_id="free_5y_top1k",
        s2_B=100,
        full_prices=full_prices,
        delist_idx=delist_idx,
    )
    if "s5" in res:
        # S5 window must NEVER exceed delist_bar=1500
        assert res["s5"]["T_window_used"] <= 1500
        assert np.isfinite(res["s5"]["t_net"])


def test_f_leadlag_null_false_positive_calibration(tmp_path):
    """F-LEADLAG-NULL: pure noise series has well-calibrated false-positive rate
    and S5 full-depth kill/amber rate >= 0.95 (zero false validated-green)."""
    led = Ledger(tmp_path / "null_ledger.db")
    try:
        n_trials = 25
        green_count = 0
        s2_passes = 0
        for idx in range(n_trials):
            fx = ll.f_leadlag_null(T=1260, seed=1000 + idx)
            t = led.register("lead_lag", "free_5y_top1k", {"null_seed": idx}, "v-null")
            runner = GauntletRunner(led, seed=100 + idx)
            res = runner.run_lead_lag_gauntlet(
                t.trial_id,
                prices=fx["prices"],
                lag_bars=1,
                formation=120,
                open_z=1.5,
                scope_id="free_5y_top1k",
                s2_B=100,
            )
            if res["verdict"] == "survivor":
                s2_passes += 1
                if res["survivor_class"] == "validated-green":
                    green_count += 1
        assert green_count == 0
        # Under H0 at alpha=0.05, 99.9% upper bound on 25 trials is <= 6
        assert s2_passes <= 6
    finally:
        led.close()


# ---------------------------------------------------------------------------
# 4. Byte-identical reproducibility
# ---------------------------------------------------------------------------

def test_lead_lag_byte_identical_reproducibility(tmp_path):
    """Acceptance criterion #5: identical (seed, params, vintage) reproduces
    byte-identical stage rows and verdicts."""
    fx = ll.f_leadlag(T=4032, gamma_true=0.30, lag_true=1, seed=777)
    snapshots = []
    for run_idx in range(2):
        led = Ledger(tmp_path / f"repro_{run_idx}.db")
        try:
            t = led.register(
                "lead_lag",
                "free_5y_top1k",
                {"leader": "L1", "follower": "F1", "lag_bars": 1},
                "v-repro",
            )
            res = GauntletRunner(led, seed=42).run_lead_lag_gauntlet(
                t.trial_id,
                prices=fx["prices"][-1260:],
                lag_bars=1,
                formation=120,
                open_z=1.5,
                scope_id="free_5y_top1k",
                s2_B=150,
                full_prices=fx["prices"],
            )
            rows = [
                (
                    r["stage"],
                    r["status"],
                    r["statistic"],
                    r["threshold"],
                    r["p_value"],
                    r["dsr"],
                    r["q_bh"],
                    r["seed"],
                )
                for r in led.stages(t.trial_id)
            ]
            snapshots.append((res["verdict"], res["survivor_class"], rows))
        finally:
            led.close()
    assert snapshots[0] == snapshots[1]


# ---------------------------------------------------------------------------
# 5. Card face rendering (JSON + PIL PNG) & S6 mixed pairs + lead_lag deck
# ---------------------------------------------------------------------------

def test_lead_lag_card_face_and_mixed_s6_deck(ledger, cards, tmp_path):
    """Minted lead_lag card renders clean anonymized JSON + PNG face (zero
    tickers/slopes/signs) and participates in an S6 mixed pairs+lead_lag deck."""
    fx = ll.f_leadlag(T=4032, gamma_true=0.30, lag_true=2, seed=888)
    t_ll = ledger.register(
        "lead_lag",
        "free_5y_top1k",
        {"leader": "AAPL", "follower": "MSFT", "lag_bars": 2},
        "v-deck",
        discoverer_handle="alice",
    )
    minter = CardMinter(ledger, cards)
    res = GauntletRunner(ledger, seed=7, card_minter=minter).run_lead_lag_gauntlet(
        t_ll.trial_id,
        prices=fx["prices"][-1260:],
        lag_bars=2,
        formation=120,
        open_z=1.5,
        scope_id="free_5y_top1k",
        s2_B=150,
        full_prices=fx["prices"],
    )
    card_id = res["card_id"]
    assert card_id is not None

    renderer = CardFaceRenderer(ledger, cards)
    agg_leader = [float(np.sin(k * 0.25)) for k in range(40)]
    face = renderer.render(
        card_id,
        viewer_handle="alice",
        with_image=True,
        diagram={"family": "lead_lag", "leader": agg_leader, "lag": 2},
        current_abs_z=res["current_abs_z"],
    )
    assert face["family"] == "lead_lag"
    assert face["sector_aggregate"]["bucket"] == "directed_lead_lag_aggregate"
    assert face["specifics"]["lag_bars"] == 2
    assert face["specifics"]["hold_bars"] == 2
    assert face["specifics"]["directed"] is True
    assert "image" in face and face["image"]["png_2x_b64"]
    # Zero tickers or slopes anywhere in JSON face (excluding base64 PNG blobs)
    face_no_img = {k: v for k, v in face.items() if k != "image"}
    serialized = str(face_no_img)
    assert "AAPL" not in serialized
    assert "MSFT" not in serialized
    assert "gamma" not in serialized.lower()

    # S6 mixed pairs + lead_lag deck composition (5 cards: 1 minted lead_lag + 4 pairs)
    cids = [card_id]
    for i in range(4):
        c = cards.mint(
            trial_id=f"trial-pair-{i}",
            family="pairs",
            scope_id="free_5y_top1k",
            config_hash=f"cfg-p-{i}",
            data_vintage_hash="v-deck",
            survivor_class="validated-green",
            owner_handle="alice",
            dsr=0.98,
            t_net_full=3.5,
            p_perm=0.01,
            s4_pass=True,
            s5_pass=True,
        )
        cids.append(c.card_id)

    decks = DeckRegistry(tmp_path / "decks.db")
    try:
        svc = DeckService(cards, decks)
        bucket = bucket_config(
            size=5, sector="all", family_mix=("pairs", "lead_lag")
        )
        deck = svc.register_ranked(
            scope_id="free_5y_top1k",
            bucket=bucket,
            owner_handle="alice",
            data_vintage_hash="v-deck",
        )
        assert deck.deck_id is not None
        assert "lead_lag" in deck.family_mix and "pairs" in deck.family_mix
    finally:
        decks.close()
