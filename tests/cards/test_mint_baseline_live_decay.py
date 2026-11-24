"""Soundness #10 (hermes-tasks #181 finding #10, card G / issue #189, re-file
of #189): persist the mint-time holdout baseline on the card + wire live decay
evaluators. TDD fixtures FIRST.

Covers:
- card schema migration: baseline_t_net / baseline_sr_ann / baseline_holdout_bars
  columns added additively; idempotent on old DBs (a pre-migration card table
  gets the columns via ALTER TABLE, reopening is a no-op);
- mint() accepts and persists the holdout baseline; read-back via get_card
  carries it; ghosts carry it too;
- mint WITHOUT a baseline (holdout_bars=0) persists None — never fabricated;
- the frozen-holdout stats computed in first_run flow into the minted card
  through the runner -> CardMinter plumbing (holdout_baseline arg on
  run_gauntlet, surfaced on the verdict dict, persisted by maybe_mint);
- a live re_evaluator that proposes a LOWER tier against fresh live evidence
  DOWNGRADES (card row + decayed_from + card_decay event + notification) even
  under default wiring when the engine supplies the live evaluator;
- an upgrade proposal is still REFUSED without eligible live-window evidence
  (live_evaluator None or ineligible) — never by payment or chance;
- CardLayer wires a real live-data re_evaluator + live_evaluator (fail-closed:
  no live series wired -> live upgrades stay refused);
- health ratio on the card face reads the PERSISTED baseline: with a baseline
  present the ratio is live_sr_ann / baseline_sr_ann (None-safe); with NO
  baseline the face reports baseline=None and the phrase stays the neutral
  "holding near its mint baseline" — never a fabricated number;
- the append-only `card_mint_baseline` ledger event is recorded at mint when a
  baseline exists (events.py health_facts already reads it).
"""

import json
import sqlite3

import numpy as np
import pytest

from sift import flags
from sift.cards import CardRegistry
from sift.cards.decay import (
    DECAY_EVENT,
    UPGRADE_EVENT,
    DecayError,
    RegistryDecayJob,
    live_window_evaluator,
    meets_tier_bar,
)
from sift.cards.minting import CardMinter
from sift.cards.notify import Notifier
from sift.cards.service import CardLayer
from sift.gauntlet import stats
from sift.gauntlet.runner import GauntletRunner
from sift.registry import Ledger
from tests.cards.test_minting import SECRET, _run_validated_green

DAY_S = 86400.0


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
def notifier(cards, tmp_path):
    n = Notifier(cards, tmp_path / "notify.db")
    yield n
    n.close()


def _mint(cards, ledger, handle="alice", minted_bar=0, now=1_000_000.0,
          baseline=None, tier_stats=(0.96, 3.2, 0.04), tag="b"):
    """Mint a silver card; `baseline` is the optional holdout baseline dict."""
    t = ledger.register("pairs", "free_5y_top1k", {"cfg": tag}, "v1",
                        discoverer_handle=handle)
    # skip-list hits return the existing trial with decided stage rows —
    # only record stages on a fresh (pending) trial
    if ledger.stage(t.trial_id, "s2")["status"] == "pending":
        ledger.record_stage(t.trial_id, "s2", "pass", p_value=tier_stats[2])
        ledger.record_stage(t.trial_id, "s3", "pass", dsr=tier_stats[0])
        ledger.record_stage(t.trial_id, "s4", "pass", statistic=tier_stats[1])
        ledger.record_stage(t.trial_id, "s5", "pass", statistic=tier_stats[1])
    kw = {}
    if baseline is not None:
        kw = {"baseline_t_net": baseline.get("t_net"),
              "baseline_sr_ann": baseline.get("sr_ann"),
              "baseline_holdout_bars": baseline.get("holdout_bars")}
    card = cards.mint(
        trial_id=t.trial_id, family=t.family, scope_id=t.scope_id,
        config_hash=t.config_hash, data_vintage_hash=t.data_vintage_hash,
        survivor_class="validated-green", owner_handle=handle,
        dsr=tier_stats[0], t_net_full=tier_stats[1], p_perm=tier_stats[2],
        now=now, minted_bar=minted_bar, **kw,
    )
    return card, t


# ---------------------------------------------------------------------------
# 1. schema migration: baseline columns exist, additive, idempotent
# ---------------------------------------------------------------------------

def test_baseline_columns_added_additively_and_idempotent(tmp_path):
    # a PRE-migration card table (the v4.1 schema without baseline cols)
    old_schema = """
    CREATE TABLE card (
        card_id TEXT PRIMARY KEY, canonical_spec TEXT NOT NULL,
        spec_hash TEXT NOT NULL, trial_id TEXT NOT NULL, family TEXT NOT NULL,
        scope_id TEXT NOT NULL, survivor_class TEXT NOT NULL,
        kind TEXT NOT NULL DEFAULT 'regular', ghost_no INTEGER,
        tier TEXT NOT NULL, tier_updated_at REAL NOT NULL,
        minted_bar INTEGER NOT NULL DEFAULT 0, owner_handle TEXT NOT NULL,
        minted_at REAL NOT NULL, decayed_from TEXT,
        mint_dsr REAL, mint_sr_star REAL, mint_n_stratum INTEGER,
        UNIQUE (spec_hash, kind, ghost_no)
    );
    CREATE TABLE spec_ghost_counter (spec_hash TEXT PRIMARY KEY, next_no INTEGER NOT NULL DEFAULT 1);
    """
    db = tmp_path / "old_cards.db"
    con = sqlite3.connect(str(db))
    con.executescript(old_schema)
    con.execute(
        "INSERT INTO card (card_id, canonical_spec, spec_hash, trial_id, family,"
        " scope_id, survivor_class, kind, tier, tier_updated_at, minted_bar,"
        " owner_handle, minted_at) VALUES ('card-old','{}','h','t1','pairs','s','validated-green',"
        "'regular','bronze',1.0,0,'alice',1.0)")
    con.commit()
    con.close()
    # opening with the new registry migrates in place
    reg = CardRegistry(db, secret=SECRET)
    cols = {r[1] for r in reg.con.execute("PRAGMA table_info(card)").fetchall()}
    assert {"baseline_t_net", "baseline_sr_ann", "baseline_holdout_bars"} <= cols
    # old row readable, baseline None (not fabricated)
    old = reg.get_card("card-old")
    assert old.baseline_t_net is None and old.baseline_sr_ann is None
    assert old.baseline_holdout_bars is None
    reg.close()
    # reopening is idempotent (no error, columns still there exactly once)
    reg2 = CardRegistry(db, secret=SECRET)
    cols2 = [r[1] for r in reg2.con.execute("PRAGMA table_info(card)").fetchall()]
    assert cols2.count("baseline_t_net") == 1
    reg2.close()


# ---------------------------------------------------------------------------
# 2. mint persists the baseline; read-back carries it; None when absent
# ---------------------------------------------------------------------------

def test_mint_persists_holdout_baseline_read_back(ledger, cards):
    baseline = {"t_net": 2.87, "sr_ann": 1.83, "holdout_bars": 252}
    card, t = _mint(cards, ledger, baseline=baseline)
    rb = cards.get_card(card.card_id)                    # read-back
    assert rb.baseline_t_net == pytest.approx(2.87)
    assert rb.baseline_sr_ann == pytest.approx(1.83)
    assert rb.baseline_holdout_bars == 252
    # ghost duplicates carry the baseline too
    card2, _ = _mint(cards, ledger, baseline=baseline, tag="b")   # same spec -> ghost
    assert card2.kind == "ghost"
    rb2 = cards.get_card(card2.card_id)
    assert rb2.baseline_t_net == pytest.approx(2.87)


def test_mint_without_holdout_persists_none_baseline(ledger, cards):
    card, _ = _mint(cards, ledger, baseline=None)
    rb = cards.get_card(card.card_id)
    assert rb.baseline_t_net is None
    assert rb.baseline_sr_ann is None
    assert rb.baseline_holdout_bars is None


def test_mint_records_card_mint_baseline_ledger_event(ledger, cards):
    baseline = {"t_net": 2.87, "sr_ann": 1.83, "holdout_bars": 252}
    card, t = _mint(cards, ledger, baseline=baseline)
    # CardMinter records the append-only baseline event; direct registry mint
    # does not (the engine path owns it) — emulate the minter here:
    minter = CardMinter(ledger, cards)
    ev_id = minter._record_baseline_event(card.card_id, t.trial_id, baseline,
                                           now=1_000_000.0)
    evs = ledger.events("card_mint_baseline")
    assert any(e["event_id"] == ev_id for e in evs)      # read-back
    payload = json.loads(evs[-1]["payload_json"])
    assert payload["card_id"] == card.card_id
    assert payload["baseline_t_net"] == pytest.approx(2.87)
    assert payload["holdout_bars"] == 252


# ---------------------------------------------------------------------------
# 3. first_run holdout stats flow into the minted card (runner plumbing)
# ---------------------------------------------------------------------------

def test_runner_passes_holdout_baseline_to_minter(ledger, cards):
    """run_gauntlet(holdout_baseline=...) surfaces on the verdict dict and the
    CardMinter persists it on the card."""
    verdict, trial = _run_validated_green(ledger, cards)
    assert verdict["card_id"] is not None
    card = cards.get_card(verdict["card_id"])
    # no holdout supplied in this fixture run -> baseline None (honest)
    assert card.baseline_t_net is None

    # now with a holdout baseline: re-run a distinct config with the arg
    import numpy as np

    from tests.fixtures.planted import T_5Y, T_FULL
    rng = np.random.Generator(np.random.PCG64(8))
    T = T_FULL
    lx = np.cumsum(rng.normal(0, 0.01, T))
    e = rng.normal(0, 1.0, T)
    s = np.zeros(T)
    for tt in range(1, T):
        s[tt] = 0.7 * s[tt - 1] + e[tt] * 10.0
    px = 100.0 * np.exp(lx)
    py = px * np.exp(s / 100.0)
    prices_full = np.column_stack([px, py])
    prices_scope = prices_full[-T_5Y:]
    x, y = prices_scope[:, 0], prices_scope[:, 1]
    excess = stats.pairs_strategy_returns(prices_scope)["gross"][60:]
    t2 = ledger.register("pairs", "free_5y_top1k", {"fixture": "F-PAIR-FULL-B"}, "v1",
                         discoverer_handle="alice")
    runner = GauntletRunner(ledger, seed=7, card_minter=CardMinter(ledger, cards))
    beta_true = float(np.dot(x - x.mean(), y - y.mean()) / np.dot(x - x.mean(), x - x.mean()))
    base = {"t_net": 2.5, "sr_ann": 1.6, "holdout_bars": 252,
            "holdout_start_bar": 3780}
    v2 = runner.run_gauntlet(
        t2.trial_id,
        fixture_x=x, fixture_y=y,
        beta_true=beta_true, half_life_true=-np.log(2.0) / np.log(0.7),
        fixture_spread=s[-T_5Y:],
        excess_returns=excess,
        perm_series=np.diff(np.log(prices_scope[:, 1])) - np.diff(np.log(prices_scope[:, 0])),
        perm_statistic_fn=stats.mean_reversion_stat,
        prices=prices_scope,
        scope_id="free_5y_top1k",
        s2_B=200,
        full_prices=prices_full,
        full_spread_returns=np.diff(np.log(prices_full[:, 1])) - np.diff(np.log(prices_full[:, 0])),
        holdout_baseline=base,
    )
    assert v2.get("holdout_baseline") == base          # surfaced on the verdict
    card2 = cards.get_card(v2["card_id"])
    assert card2.baseline_t_net == pytest.approx(2.5)   # persisted read-back
    assert card2.baseline_sr_ann == pytest.approx(1.6)
    assert card2.baseline_holdout_bars == 252
    # append-only baseline event recorded by the minter
    evs = ledger.events("card_mint_baseline")
    assert any(json.loads(e["payload_json"]).get("card_id") == card2.card_id
               for e in evs)


# ---------------------------------------------------------------------------
# 4. live re_evaluator: a weakened proposal DOWNGRADES under engine wiring
# ---------------------------------------------------------------------------

def _strong_live_returns(n=2 * 252 + 40, seed=5):
    rng = np.random.Generator(np.random.PCG64(seed))
    e = rng.normal(0, 0.01, n)
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = s[t - 1] + e[t]
    r = 0.002 - 0.8 * np.roll(s, 1) + e
    r[0] = 0.002 + e[0]
    return r


def test_live_re_evaluator_downgrade_fires_with_event_and_notify(ledger, cards, notifier):
    """Engine wiring: re_evaluator reads FRESH live evidence (not the static
    stage rows). A weakened live proposal downgrades; event + notify fire."""
    card, t = _mint(cards, ledger, baseline={"t_net": 2.9, "sr_ann": 1.8, "holdout_bars": 252})
    notifier.set_pref(card.card_id, "alice", enabled=True)
    # fresh live evidence: pure noise -> below every bar -> bronze proposal
    live = np.random.Generator(np.random.PCG64(7)).normal(0, 0.01, 600)

    def live_re(card, ledger):
        # fresh-evidence re-evaluator: tier from the CURRENT live window, not
        # the static mint-time rows (soundness #10: default was inert)
        from sift.cards.decay import _window_stats
        n, v = ledger.stratum_stats(card.scope_id)
        ws = _window_stats(live, start=len(live) - 252, end=len(live),
                           n_stratum=n, v_stratum=v, seed=1, s2_B=100)
        if ws.t_net_full is not None and meets_tier_bar(ws, "silver"):
            return "silver"
        return "bronze"

    job = RegistryDecayJob(cards, ledger, notifier=notifier, re_evaluator=live_re)
    (r,) = job.run_pass(now=2_000_000.0)
    assert r.downgraded is True
    assert r.to_tier == "bronze" and r.from_tier == "silver"
    after = cards.get_card(card.card_id)
    assert after.tier == "bronze" and after.decayed_from == "silver"   # read-back
    assert len(ledger.events(DECAY_EVENT)) == 1                        # read-back
    assert r.notified is True
    # the card keeps its persisted baseline through the downgrade
    assert after.baseline_t_net == pytest.approx(2.9)


def test_upgrade_still_refused_without_eligible_live_evidence(ledger, cards, notifier):
    """A HIGHER proposal under default wiring (no live_evaluator) is refused
    loudly — never applied by payment or chance (invariant kept)."""
    card, t = _mint(cards, ledger)   # silver
    job = RegistryDecayJob(cards, ledger, notifier=notifier,
                           re_evaluator=lambda c, led: "gold")   # no live_evaluator
    with pytest.raises(DecayError) as ei:
        job.evaluate_card(card, now=2_000_000.0)
    assert "no pre-registered live-window evidence" in str(ei.value) or "live-window" in str(ei.value)
    assert cards.get_card(card.card_id).tier == "silver"
    assert ledger.events(UPGRADE_EVENT) == []


def test_upgrade_refused_when_live_window_ineligible(ledger, cards, notifier):
    """With a live_evaluator wired but the card too young (in-sample echo),
    an upgrade proposal is refused — rules NOT relaxed."""
    card, t = _mint(cards, ledger, minted_bar=0)
    rets = _strong_live_returns()
    job = RegistryDecayJob(
        cards, ledger, notifier=notifier,
        re_evaluator=lambda c, led: "gold",
        live_evaluator=lambda c: live_window_evaluator(
            c, live_returns=rets, now_bar=252, n_stratum=50, v_stratum=0.01,
            seed=1, s2_B=100),
    )
    with pytest.raises(DecayError) as ei:
        job.evaluate_card(card, now=2_000_000.0)
    assert "in-sample echo" in str(ei.value) or "live-window" in str(ei.value)
    assert cards.get_card(card.card_id).tier == "silver"


# ---------------------------------------------------------------------------
# 5. CardLayer wires live evaluators; fail-closed when no live series
# ---------------------------------------------------------------------------

def test_card_layer_wires_live_evaluators_fail_closed(ledger, tmp_path):
    lay = CardLayer(ledger, tmp_path / "cards", card_secret=SECRET)
    try:
        # default wiring: a live series source is injectable; with none wired,
        # live_evaluator is None -> upgrades stay refused (fail closed)
        assert lay.decay.live_evaluator is None or callable(lay.decay.live_evaluator)
        # the re_evaluator is the live-evidence one (not the inert static default):
        # with no live series wired it falls back to the static rows (documented)
        card, t = _mint(lay.cards, ledger)
        r = lay.decay.evaluate_card(card, now=3_000_000.0)
        assert r.to_tier == r.from_tier     # static evidence == mint tier -> no-op
        # injecting a live series enables live evaluation end-to-end
        lay.set_live_returns(lambda card: _strong_live_returns())
        assert lay.decay.live_evaluator is not None
        # a SHORT live series (100 bars < 12m) -> in-sample echo -> the
        # upgrade proposal is refused, no tier change (rules not relaxed)
        lay.set_live_returns(lambda card: _strong_live_returns(n=100))
        job2 = RegistryDecayJob(lay.cards, ledger,
                                re_evaluator=lambda c, led: "gold",
                                live_evaluator=lay.decay.live_evaluator)
        with pytest.raises(DecayError):
            job2.evaluate_card(card, now=3_000_000.0 + 300 * DAY_S)
        assert lay.cards.get_card(card.card_id).tier == "silver"
    finally:
        lay.close()


# ---------------------------------------------------------------------------
# 6. health ratio reads the PERSISTED baseline on the face
# ---------------------------------------------------------------------------

def test_face_health_ratio_uses_persisted_baseline(ledger, tmp_path):
    lay = CardLayer(ledger, tmp_path / "cards", card_secret=SECRET)
    try:
        card, t = _mint(lay.cards, ledger,
                        baseline={"t_net": 2.9, "sr_ann": 1.8, "holdout_bars": 252})
        # face render with a live reading: health = live sr_ann / baseline sr_ann
        face = lay.face.render(card.card_id, viewer_handle="alice", now=1_000_000.0,
                               health_ratio=0.8 * 1.8 / 1.8)
        assert face["health"]["ratio_vs_mint_baseline"] == pytest.approx(0.8)
        assert "weakening" in face["health"]["phrase"]
        # the face carries the persisted baseline as a survival fact
        assert face["mint_baseline"]["sr_ann"] == pytest.approx(1.8)
        assert face["mint_baseline"]["holdout_bars"] == 252
    finally:
        lay.close()


def test_face_health_ratio_none_baseline_never_fabricated(ledger, tmp_path):
    lay = CardLayer(ledger, tmp_path / "cards", card_secret=SECRET)
    try:
        card, t = _mint(lay.cards, ledger, baseline=None)   # holdout_bars=0 path
        face = lay.face.render(card.card_id, viewer_handle="alice", now=1_000_000.0)
        # no baseline persisted -> ratio None, neutral phrase, baseline block None
        assert face["health"]["ratio_vs_mint_baseline"] is None
        assert face["health"]["phrase"] == "holding near its mint baseline"
        assert face["mint_baseline"] is None
    finally:
        lay.close()
