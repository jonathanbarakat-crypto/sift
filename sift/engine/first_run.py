"""Server-side runner for the SIFT gate-1 first personal trial run.

Scope: `free_1y_top1k` — top-1,000 PIT universe by dollar volume, window =
the last ~1 year of daily EOD (the ONLY window the EODHD free tier serves:
20 calls/day, past-1-year EOD, live quotes; deep history 403s — see
data/eodhd_client.py DeepHistoryBlocked). This run does NOT claim 5y/16y.

Pairs family: for each candidate pair (i, j) from the universe, the gauntlet
S0-S4 (+ automatic S5 at registry_depth when full-window data exists) runs
server-side against the warehouse parquet bars. Every trial is REGISTERED in
the append-only ledger (sift/registry) BEFORE any computation; re-registering
the identical (family, scope, params, data-vintage) is a skip-list hit that
reuses the trial_id and costs ~0 compute (measured and reported).

Measurements written to JSON (default trials/first-run-measurements.json):
  * wall-clock per unique trial vs per cached (skip-list) trial
  * EODHD API cost per unique vs cached trial (calls spent; cached = 0)
  * peak RSS and the corr-matrix allocation (N x N float64 bytes)
  * ledger dump summary (verdicts, survivor classes, stage pass/fail counts)

Hard rules (v3 scope doc section 12): server-side only, no raw data leaves the
server, scope is labeled honestly, 'SIFT does not trade'.

Usage:
    python3 -m sift.engine.first_run --warehouse <path> --universe <topN parquet>
        --ledger trials/ledger.db --max-pairs 2000 --measurements trials/first-run.json
        [--holdout-bars 252]

`--holdout-bars N` (default 252) locks the most recent N trading days out of
S0-S4 discovery/walk-forward; survivors are evaluated once on that frozen
holdout (the mint-time live-baseline, persisted on the minted card via
`--cards` — soundness #10). The holdout is never used for tuning.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/..")
    import importlib
    wh = importlib.import_module("data.warehouse")
    from sift.cards import CardRegistry
    from sift.cards.minting import CardMinter
    from sift.gauntlet import config as cfg
    from sift.gauntlet import stats
    from sift.gauntlet.runner import GauntletRunner
    from sift.registry import Ledger
else:
    import importlib
    wh = importlib.import_module("data.warehouse")
    from ..cards import CardRegistry  # type: ignore
    from ..cards.minting import CardMinter  # type: ignore
    from ..gauntlet import config as cfg  # type: ignore
    from ..gauntlet import stats  # type: ignore
    from ..gauntlet.runner import GauntletRunner  # type: ignore
    from ..registry import Ledger  # type: ignore

SCOPE_ID = "free_1y_top1k"
FAMILY = "pairs"
ENGINE_VERSION = "gate1-build5/8-first-run"

# data-source -> (vendor label, research_grade). EODHD is the production
# vendor; yfinance/Yahoo is RESEARCH-ONLY (docs/vendor-licensing.md pull/26,
# docs/research-feeds.md): no ToS authorization for production redistribution,
# no delisted lists (survivorship-biased), no SLA.
SOURCES = {
    "eodhd": {"vendor": "eodhd", "research_grade": False, "default_scope": "free_1y_top1k"},
    "yf": {"vendor": "yfinance", "research_grade": True, "default_scope": None},
}
RESEARCH_SCOPE_PREFIX = "research_"


def _kfmt(n: int) -> str:
    return f"{n//1000}k" if n >= 1000 else str(n)


def scope_id_for_source(source: str, window: str, top_n: int | None = None) -> str:
    """Derive the honest scope id from source + window.

    Production EODHD keeps its registered production scope ids
    (``free_1y_top1k``). A research source (yf/yfinance) MUST get a
    ``research_``-prefixed id (e.g. ``research_5y_top1k``) so a trial can
    never be mislabeled as production scope (``free_1y_top1k``/``pro_full``).
    """
    spec = SOURCES.get(source)
    if spec is None:
        raise ValueError(f"unknown --source {source!r}; expected one of {sorted(SOURCES)}")
    if not spec["research_grade"]:
        return spec["default_scope"]
    tag = window
    if top_n:
        tag += f"_top{_kfmt(top_n)}"
    return f"{RESEARCH_SCOPE_PREFIX}{tag}"


def source_is_research(source: str) -> bool:
    return bool(SOURCES.get(source, {}).get("research_grade"))


def log(msg: str) -> None:
    print(f"[first_run {datetime.now(timezone.utc).strftime('%H:%M:%S')}Z] {msg}", flush=True)


def peak_rss_bytes() -> int:
    # Linux: ru_maxrss is in kilobytes
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def load_price_matrix(warehouse_root: str, symbols: list[str], exchange: str = "US") -> tuple[np.ndarray, list[str], dict]:
    """Load adjusted_close series for the given symbols from the warehouse,
    aligned on the union of trading dates (inner join on dates present for all
    symbols; missing cells are NaN and masked downstream). Returns
    (prices T x N, kept_symbols, info).

    Symbols are accepted both bare (``AAPL``) and exchange-suffixed
    (``AAPL.US``): the EODHD ingest path writes ``AAPL`` parquet files whose
    ``symbol`` column is ``AAPL.US``, and the yfinance path writes Yahoo-style
    bare tickers (``AAPL``, ``BRK-B``) whose ``symbol`` column is ``AAPL.US``.
    Both resolve to the same ``eod/<EX>/<base>.parquet`` file; the lookup
    strips the ``.EX`` suffix exactly like the EODHD path did, and falls back
    to the raw name for Yahoo-style class shares (``BRK-B``).
    """
    frames = {}
    ex_suffix = f".{exchange.upper()}"
    for sym in symbols:
        base = sym[:-len(ex_suffix)] if sym.upper().endswith(ex_suffix) else sym
        df = wh.read_parquet(wh.eod_parquet_path(warehouse_root, exchange, base))
        if (df is None or df.empty) and base != sym:
            df = wh.read_parquet(wh.eod_parquet_path(warehouse_root, exchange, sym))
        if df is None or df.empty:
            continue
        s = df.set_index("date")["adjusted_close"].astype(float)
        frames[sym] = s
    if not frames:
        return np.zeros((0, 0)), [], {"loaded": 0}
    panel = pd.DataFrame(frames).sort_index()
    # Align on the COMMON trading-date window of the kept symbols: a strict
    # full-panel dropna() is empty as soon as one of N names IPO'd later (or
    # has a single missing bar) inside the window, so instead drop the symbols
    # that do not span the window and inner-join on the dates all remaining
    # symbols share. Still no look-ahead: only dates every kept symbol has.
    min_bars = max(2, int(0.5 * panel.shape[0]))
    counts = panel.count()
    spanning = counts[counts >= min_bars].index.tolist()
    if len(spanning) < 2:
        # fall back to the top-2 by bar count so a degenerate universe still runs
        spanning = counts.sort_values(ascending=False).index.tolist()[:2]
    panel = panel[spanning]
    first_valid = panel.apply(lambda c: c.first_valid_index())
    last_valid = panel.apply(lambda c: c.last_valid_index())
    lo, hi = first_valid.max(), last_valid.min()
    if pd.isna(lo) or pd.isna(hi) or lo > hi:
        return np.zeros((0, 0)), [], {"loaded": len(frames), "dates": 0}
    panel = panel.loc[lo:hi].dropna()
    if panel.empty:
        return np.zeros((0, 0)), [], {"loaded": len(frames), "dates": 0}
    kept = list(panel.columns)
    return panel.to_numpy(dtype=float), kept, {"loaded": len(frames), "kept": len(kept), "dates": int(panel.shape[0])}


def corr_matrix(returns: np.ndarray) -> np.ndarray:
    """Full Pearson corr matrix of per-bar log returns (N x N float64).
    Memory measured explicitly: N*N*8 bytes."""
    return np.corrcoef(returns, rowvar=False)


def pair_series(prices: np.ndarray, i: int, j: int, formation: int) -> dict:
    """Build the per-bar series for pair (i, j) from a T x N price matrix:
    hedge via OLS on log prices over the formation window, spread = log(y) -
    beta*log(x), spread returns, and the walk-forward net returns used by S4.
    All no-look-ahead (estimates use only bars <= t)."""
    x = prices[:, i]
    y = prices[:, j]
    lx, ly = np.log(x), np.log(y)
    T = len(x)
    beta = stats.ols_hedge_ratio(lx[:formation], ly[:formation]) if formation <= T else float("nan")
    spread = ly - beta * lx
    rs = np.zeros(T)
    rs[1:] = np.diff(spread)
    # soundness #9: real warehouse prices -> per-name data-driven half-spreads
    # (Roll) instead of the flagged 100 bps placeholder; two-leg cost, fill-bar
    # timing and live slippage are applied inside walk_forward_returns.
    wf = stats.walk_forward_returns(prices[:, [i, j]], formation=formation,
                                    half_spreads="roll")
    return {"beta": beta, "beta_levels": stats.ols_hedge_ratio(x, y),
            "spread": spread, "spread_returns": rs, "net": wf["net"]}


def _holdout_baseline(holdout_prices: np.ndarray, i: int, j: int, formation: int,
                      *, holdout_bars: int, holdout_start_bar: int) -> dict | None:
    """Measure the frozen-holdout live-baseline for pair (i, j) ONCE
    (soundness #10): walk-forward net returns on the locked holdout bars
    -> t_net and annualized Sharpe. Returns None when the holdout has too
    few usable bars (never fabricates a baseline). Evaluation-only: the
    holdout never entered discovery/walk-forward."""
    if holdout_prices is None or holdout_prices.shape[0] < 4:
        return None
    # formation capped at a quarter of the holdout so at least one full
    # walk-forward block (formation + step) is traded inside the holdout —
    # a formation of half the holdout leaves only the formation window and
    # zero traded blocks (baseline would be a fabricated all-zero).
    form = min(formation, max(2, holdout_prices.shape[0] // 4))
    wf_h = stats.walk_forward_returns(holdout_prices[:, [i, j]], formation=form,
                                      half_spreads="roll")
    net_h = wf_h["net"]
    net_h = net_h[np.isfinite(net_h)]
    if net_h.size < 2:
        return None
    return {
        "t_net": float(stats.t_stat_sharpe(net_h)),
        "sr_ann": float(stats.sharpe_ratio(net_h)),
        "holdout_bars": int(holdout_bars),
        "holdout_start_bar": int(holdout_start_bar),
        "holdout_net_bars": int(net_h.size),
    }


def register_trial(ledger: Ledger, params: dict, vintage: str, provenance: str = "operator_sweep",
                   scope_id: str = SCOPE_ID) -> tuple:
    """Register BEFORE any computation (v3 section 5). Returns (trial, cached: bool)."""
    existing = ledger.con.execute(
        "SELECT trial_id FROM trial WHERE family=? AND scope_id=? AND config_hash=? AND data_vintage_hash=?",
        (FAMILY, scope_id, __import__("sift.registry", fromlist=["config_hash_of"]).config_hash_of(params), vintage),
    ).fetchone()
    t0 = time.perf_counter()
    trial = ledger.register(FAMILY, scope_id, params, vintage, provenance=provenance,
                            engine_version=ENGINE_VERSION, discoverer_handle="operator")
    dt = time.perf_counter() - t0
    return trial, (existing is not None), dt


def run_pairs(warehouse_root: str, universe_path: str, ledger_path: str,
              max_pairs: int, formation: int, s2_B: int, measurements_path: str,
              exchange: str = "US", max_universe: int = 1000,
              source: str = "eodhd", window: str = "1y",
              holdout_bars: int = 0,
              cards_path: str | None = None) -> dict:
    t_start = time.perf_counter()
    vintage = wh.read_vintage(warehouse_root) or "no-vintage"
    scope_id = scope_id_for_source(source, window, top_n=max_universe if source != "eodhd" else None)
    research = source_is_research(source)
    log(f"data vintage: {vintage[:16]}…  scope={scope_id} family={FAMILY} source={source}"
        + ("  [RESEARCH-GRADE: yfinance/Yahoo — not production data]" if research else ""))

    uni = pd.read_parquet(universe_path)
    symbols = [s for s in uni["symbol"].tolist()][:max_universe]
    log(f"universe: {len(symbols)} symbols from {os.path.basename(universe_path)}")

    prices, kept, info = load_price_matrix(warehouse_root, symbols, exchange)
    log(f"price matrix: {prices.shape[0]} bars x {prices.shape[1]} symbols ({info})")
    if prices.shape[1] < 2:
        raise SystemExit("need at least 2 symbols with overlapping bars")

    # -- 12m HOLDOUT (locked, never used for tuning) --------------------------
    # `--holdout-bars N` locks the most recent N trading days out of the
    # S0-S4 discovery/walk-forward window: discovery + walk-forward run on
    # prices[:T-N] only. Survivors are then evaluated ONCE on the frozen
    # holdout prices[T-N:] — that single frozen-holdout evaluation is the
    # mint-time live-baseline, persisted on the minted card (soundness #10;
    # needs `--cards` so a CardMinter is wired). The holdout is NEVER used
    # for tuning, screening, or threshold selection (owner decision
    # 2026-11-17); it only measures survivors after they are chosen.
    holdout_start = None
    holdout = None
    if holdout_bars and holdout_bars > 0:
        if holdout_bars >= prices.shape[0] - formation - 2:
            raise SystemExit(
                f"--holdout-bars {holdout_bars} leaves too little discovery window "
                f"({prices.shape[0]} bars, formation {formation})"
            )
        holdout_start = prices.shape[0] - holdout_bars
        holdout = prices[holdout_start:]
        discovery_prices = prices[:holdout_start]
        log(f"holdout locked: last {holdout_bars} bars frozen out of discovery/"
            f"walk-forward (discovery window {discovery_prices.shape[0]} bars, "
            f"holdout {holdout.shape[0]} bars) — holdout is evaluation-only, "
            f"never used for tuning")
    else:
        discovery_prices = prices

    rets = np.zeros_like(discovery_prices)
    rets[1:] = np.diff(np.log(discovery_prices), axis=0)
    corr = corr_matrix(rets[1:])
    n = corr.shape[0]
    corr_bytes = n * n * 8
    log(f"corr matrix: {n}x{n} float64 = {corr_bytes/1e6:.1f} MB (measured allocation); "
        f"peak RSS so far {peak_rss_bytes()/1e9:.2f} GB")

    # candidate pairs: rank by |corr| descending, take top max_pairs (screening
    # stage — the pairs family's discovery step), deterministic order.
    iu = np.triu_indices(n, k=1)
    cvals = corr[iu]
    order = np.argsort(-np.abs(np.nan_to_num(cvals)))
    cand = [(int(iu[0][k]), int(iu[1][k])) for k in order[:max_pairs]]
    log(f"candidate pairs: {len(cand)} (top by |corr|)")

    ledger = Ledger(ledger_path)
    # soundness #10: wire the card layer so survivors actually mint (the
    # engine previously wired no CardMinter at all) and the frozen-holdout
    # live-baseline computed below is persisted on the minted card.
    cards_reg = CardRegistry(cards_path) if cards_path else None
    card_minter = CardMinter(ledger, cards_reg) if cards_reg is not None else None
    runner = GauntletRunner(ledger, seed=20261115, card_minter=card_minter,
                            panel_corr=corr)
    # sift #59: pre-register (freeze) the evaluation epoch's admission
    # denominator BEFORE any trial runs. The pairs stratum's trials share
    # the panel's assets, so they are correlated; the DSR benchmark must use
    # the A.3 implied independent N, not the raw candidate count. M is the
    # pre-registered candidate-pair count (len(cand)), so every trial in the
    # epoch faces the SAME frozen bar regardless of run order (previously N
    # was read at evaluation time: 17k/118k/135k/166k for the 4 survivors).
    ep = runner.set_admission_epoch(scope_id, epoch=1, m_trials=len(cand),
                                    detail={"vintage": vintage[:16],
                                            "n_candidate_pairs": len(cand)})
    log(f"sift #59: admission epoch frozen: scope={scope_id} epoch=1 "
        f"M={ep['n_eff_raw']} rho_bar={ep['rho_bar']:.6f} "
        f"n_eff={ep['n_eff']:.1f} method={ep['method']}")
    calls_before = 0
    calls_state = os.path.join(warehouse_root, "state", "calls.json")
    if os.path.exists(calls_state):
        calls_before = json.load(open(calls_state)).get("calls_today", 0)

    unique_t, cached_t = [], []
    unique_cost_calls, cached_cost_calls = [], []
    rows = []
    holdout_rows = []
    for k, (i, j) in enumerate(cand):
        si, sj = kept[i], kept[j]
        params = {"pair": [si, sj], "formation": formation, "s2_B": s2_B,
                  "open_z": 2.0, "close_z": 0.5, "family": FAMILY, "scope": scope_id}
        if holdout_start is not None:
            # the skip-list key carries the holdout so a holdout run can never
            # collide with a full-window run of the same pair
            params["holdout_bars"] = int(holdout_bars)
        trial, cached, reg_dt = register_trial(ledger, params, vintage, scope_id=scope_id)
        t0 = time.perf_counter()
        if cached:
            # skip-list hit: identical trial already registered+run — no compute
            cached_t.append(time.perf_counter() - t0 + reg_dt)
            cached_cost_calls.append(0.0)
            rows.append({"pair": [si, sj], "trial_id": trial.trial_id, "cached": True,
                         "wall_s": cached_t[-1], "calls": 0.0,
                         "verdict": ledger.get(trial.trial_id).verdict})
            continue
        ps = pair_series(discovery_prices, i, j, formation)
        # soundness #10: compute the frozen-holdout live-baseline BEFORE the
        # gauntlet runs so it can be persisted on the minted card (the holdout
        # is evaluation-only: it is measured once here, never read by S0-S4).
        holdout_baseline = None
        if holdout is not None:
            hb = _holdout_baseline(holdout, i, j, formation,
                                   holdout_bars=int(holdout_bars),
                                   holdout_start_bar=int(holdout_start))
            if hb is not None:
                holdout_baseline = hb
        # S0 harness calibration on real data (finding #14, doc-only): the
        # harness must recover the pair's own OLS hedge computed on LEVELS
        # (the convention S0's ols_hedge_ratio asserts on), so beta_true =
        # beta_levels — on real data this beta assertion is therefore
        # TAUTOLOGICAL (beta_hat - beta_true = 0.0 exactly). The genuine
        # real-data gate is the finite-half-life (mean-reversion) check: an
        # explosive spread (phi_hat > 1 -> half-life = inf) FAILS S0. The
        # planted AR(1) fixture (F-PAIR) remains the real calibration of
        # both assertions. The half-life is measured and recorded, not a
        # planted constant on real data.
        _, hl_hat = stats.estimate_ar1(ps["spread"])
        res = runner.run_gauntlet(
            trial.trial_id,
            fixture_x=discovery_prices[:, i], fixture_y=discovery_prices[:, j],
            beta_true=ps["beta_levels"], half_life_true=hl_hat if math.isfinite(hl_hat) else 1e9,
            fixture_spread=ps["spread"],
            excess_returns=ps["spread_returns"][1:],
            perm_series=ps["spread_returns"][1:],
            perm_statistic_fn=stats.mean_reversion_stat,
            prices=discovery_prices[:, [i, j]],
            scope_id=scope_id,
            s2_B=s2_B,
            holdout_baseline=holdout_baseline,
        )
        dt = time.perf_counter() - t0 + reg_dt
        unique_t.append(dt)
        calls_state_now = json.load(open(calls_state)).get("calls_today", calls_before) if os.path.exists(calls_state) else calls_before
        unique_cost_calls.append(float(calls_state_now - calls_before))
        calls_before = calls_state_now
        v = res
        rows.append({"pair": [si, sj], "trial_id": trial.trial_id, "cached": False,
                     "wall_s": dt, "calls": 0.0,
                     "verdict": v["verdict"], "survivor_class": v["survivor_class"],
                     "killed_at": v.get("killed_at"),
                     "card_id": v.get("card_id"),
                     "s4_score": v.get("s4_score"),
                     "stages": {s.stage: s.status for s in v["stages"]}})
        # -- frozen-holdout evaluation of survivors (ONCE, evaluation only) --
        # The holdout bars never entered S0-S4 discovery/walk-forward above.
        # A survivor is measured exactly once on the frozen holdout; that
        # frozen-holdout result is the mint-time live-baseline, NOW persisted
        # on the minted card (soundness #10: previously it went to the
        # measurements JSON only). The holdout is never used for tuning.
        if holdout is not None and v["verdict"] == "survivor":
            hb = holdout_baseline or _holdout_baseline(
                holdout, i, j, formation,
                holdout_bars=int(holdout_bars), holdout_start_bar=int(holdout_start))
            if hb is not None:
                form_h = min(formation, max(2, holdout.shape[0] // 4))
                wf_h = stats.walk_forward_returns(holdout[:, [i, j]],
                                                  formation=form_h,
                                                  half_spreads="roll")
                net_h = wf_h["net"]
                net_h = net_h[np.isfinite(net_h)]
                t_net_h = stats.t_stat_sharpe(net_h) if net_h.size >= 2 else None
                sr_ann_h = stats.sharpe_ratio(net_h) if net_h.size >= 2 else None
                holdout_rows.append({
                    "pair": [si, sj], "trial_id": trial.trial_id,
                    "holdout_bars": int(holdout_bars),
                    "holdout_start_bar": int(holdout_start),
                    "holdout_net_bars": int(net_h.size),
                    "holdout_t_net": t_net_h,
                    "holdout_sr_ann": sr_ann_h,
                    "card_id": v.get("card_id"),
                    "baseline_persisted_on_card": bool(v.get("card_id")) and hb is not None,
                    "survivor_class": v["survivor_class"],
                    "note": "frozen-holdout evaluation of a survivor — mint-time "
                            "live-baseline, persisted on the card (soundness #10); "
                            "the holdout was never used for tuning",
                })
        if (k + 1) % 50 == 0:
            log(f"  {k+1}/{len(cand)} pairs done | unique mean {np.mean(unique_t):.3f}s | "
                f"cached mean {np.mean(cached_t) if cached_t else 0:.5f}s | peak RSS {peak_rss_bytes()/1e9:.2f} GB")

    t_total = time.perf_counter() - t_start
    # re-register every pair once more to measure the cached path honestly
    t0 = time.perf_counter()
    for r in rows:
        params = {"pair": r["pair"], "formation": formation, "s2_B": s2_B,
                  "open_z": 2.0, "close_z": 0.5, "family": FAMILY, "scope": scope_id}
        if holdout_start is not None:
            params["holdout_bars"] = int(holdout_bars)
        register_trial(ledger, params, vintage, scope_id=scope_id)
    cached_rereg_s = time.perf_counter() - t0

    calls_after = json.load(open(calls_state)).get("calls_today", calls_before) if os.path.exists(calls_state) else calls_before
    summary = ledger_summary(ledger)
    meas = {
        "scope": scope_id, "family": FAMILY, "data_vintage": vintage,
        "source": source, "research_grade": research,
        "window_bars": int(prices.shape[0]), "universe_size": int(prices.shape[1]),
        "discovery_bars": int(discovery_prices.shape[0]),
        "holdout_bars": int(holdout_bars) if holdout_start is not None else 0,
        "holdout_start_bar": int(holdout_start) if holdout_start is not None else None,
        "holdout_rows": holdout_rows,
        "n_pairs": len(cand),
        "wall_clock_total_s": t_total,
        "unique_trials": len(unique_t),
        "unique_wall_s_mean": float(np.mean(unique_t)) if unique_t else 0.0,
        "unique_wall_s_total": float(sum(unique_t)),
        "cached_trials": len(cached_t) + len(rows),  # all re-registered
        "cached_wall_s_mean_rereg": cached_rereg_s / max(1, len(rows)),
        "cached_wall_s_mean": float(np.mean(cached_t)) if cached_t else 0.0,
        "eodhd_calls_total": calls_after,
        "eodhd_calls_per_unique_trial_mean": float(np.mean(unique_cost_calls)) if unique_cost_calls else 0.0,
        "eodhd_calls_per_cached_trial": 0.0,
        "corr_matrix_bytes": int(corr_bytes), "corr_matrix_mb": corr_bytes / 1e6,
        "corr_matrix_n": int(n),
        "admission_epoch": ep,   # sift #59: frozen A.3 denominator (M, rho_bar, n_eff)
        "peak_rss_bytes": peak_rss_bytes(), "peak_rss_gb": peak_rss_bytes() / 1e9,
        "ledger": summary,
        "rows": rows,
        "ran_utc": datetime.now(timezone.utc).isoformat(),
        "note": ("S5 registry_depth needs the production full-window feed — s5 rows stay "
                 "pending/amber on any non-production vintage (free EODHD 1y OR yfinance "
                 "research 5y). "
                 + ("This run is RESEARCH-GRADE (yfinance/Yahoo): active-only universe, "
                    "survivorship-biased, delist_idx unavailable, scope "
                    f"{scope_id} — never labeled free_1y_top1k/pro_full."
                    if research else
                    "free tier covers ~last 1y EOD only; scope labeled free_1y_top1k, NOT 5y.")),
    }
    os.makedirs(os.path.dirname(measurements_path), exist_ok=True)
    with open(measurements_path, "w") as f:
        json.dump(meas, f, indent=2, default=str)
    if cards_reg is not None:
        cards_reg.close()
    ledger.close()
    log(f"done in {t_total:.1f}s | unique mean {meas['unique_wall_s_mean']:.3f}s | "
        f"cached re-register mean {meas['cached_wall_s_mean_rereg']*1e3:.2f}ms | "
        f"corr {meas['corr_matrix_mb']:.1f}MB | peak RSS {meas['peak_rss_gb']:.2f}GB | "
        f"calls {calls_after}")
    return meas


def ledger_summary(ledger: Ledger) -> dict:
    con = ledger.con
    trials = con.execute("SELECT verdict, survivor_class, COUNT(*) c FROM trial GROUP BY verdict, survivor_class").fetchall()
    stages = con.execute("SELECT stage, status, COUNT(*) c FROM stage_result GROUP BY stage, status ORDER BY stage, status").fetchall()
    strata = con.execute("SELECT scope_id, n_trials, var_stat FROM stratum_stats").fetchall()
    return {
        "by_verdict_class": [dict(r) for r in trials],
        "stages": [dict(r) for r in stages],
        "strata": [dict(r) for r in strata],
        "total_trials": con.execute("SELECT COUNT(*) FROM trial").fetchone()[0],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--warehouse", required=True)
    ap.add_argument("--universe", required=True, help="topN PIT parquet from data/universe.py")
    ap.add_argument("--ledger", default="trials/ledger.db")
    ap.add_argument("--measurements", default="trials/first-run-measurements.json")
    ap.add_argument("--max-pairs", type=int, default=2000)
    ap.add_argument("--max-universe", type=int, default=1000)
    ap.add_argument("--formation", type=int, default=60)
    ap.add_argument("--s2-B", type=int, default=cfg.S2_B)
    ap.add_argument("--source", default="eodhd", choices=sorted(SOURCES),
                    help="data source: 'eodhd' (production vendor, default) or 'yf' "
                         "(yfinance/Yahoo RESEARCH-ONLY — scope id gets a research_ prefix)")
    ap.add_argument("--window", default="1y",
                    help="window tag used to derive the scope id for research sources "
                         "(e.g. '5y' -> research_5y_top<N>; default '1y'; ignored for eodhd)")
    ap.add_argument("--holdout-bars", type=int, default=252,
                    help="lock the most recent N trading days out of S0-S4 "
                         "discovery/walk-forward; survivors are evaluated ONCE on "
                         "that frozen holdout (the mint-time live-baseline, "
                         "persisted on the card — soundness #10). The holdout is "
                         "NEVER used for tuning. "
                         "0 disables the holdout (full window, legacy behavior). "
                         "Default 252 (12 months of trading days).")
    ap.add_argument("--cards", default=None,
                    help="optional card-registry DB path (e.g. trials/cards.db): "
                         "when set, the engine wires a CardMinter so survivors "
                         "mint cards (cards_v4_1 flag) and the frozen-holdout "
                         "live-baseline is persisted on each minted card")
    args = ap.parse_args(argv)
    run_pairs(args.warehouse, args.universe, args.ledger, args.max_pairs,
              args.formation, args.s2_B, args.measurements, max_universe=args.max_universe,
              source=args.source, window=args.window,
              holdout_bars=args.holdout_bars, cards_path=args.cards)
    return 0


if __name__ == "__main__":
    sys.exit(main())
