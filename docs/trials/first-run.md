# SIFT gate-1 — first personal trial run (gate-0 proof) — `free_1y_top1k`

**Run date:** 2026-11-15 ~22:06Z · **engine:** `gate1-build5/8-first-run` · **family:** `pairs`
**Scope:** `free_1y_top1k` — point-in-time top-N (N capped by the daily free-tier call budget, see below) by average dollar volume, window = **the last ~1 year of daily EOD (250 trading bars, 2025-11-17 → 2026-11-13)**. This is the *only* window the EODHD free tier serves (20 API calls/day, past-1-year EOD, live quotes; pre-1-year requests 403 with `DeepHistoryBlocked`, never bypassed). **This run does NOT claim 5y or 16y.** S5 registry-depth validation needs full-window data (paid tier) — s5 rows stay `pending` here, which is the honest state ("a stage not run cannot pass").

**Built on unmerged upstream branches** (sift main is still empty; PRs open):
- ingestion: branch `build/gate1-eodhd-ingest` (PR pull/4) — `data/{eodhd_client,ingest_eodhd,universe,warehouse}.py`
- engine + gauntlet + ledger: branch `task/t_b888ba8c-s5-registry-depth` (PRs pull/8, pull/7, pull/6, pull/9, pull/10) — `sift/{registry,gauntlet,engine}`
- this run's branch `task/t_c3367cda-first-trial` adds `sift/engine/first_run.py` (the server-side pairs runner) on top of the S5 branch + the ingest branch's `data/`.

## Data vintage & API budget (measured, not claimed)

- `pass eodhd/api-token` present on the VM; token never printed/logged/committed.
- EODHD free-tier calls used **2026-11-15: 19/20** (persisted in `state/calls.json`; the `--max-calls 19` hard guard fired before the 20th request — see log `STOP: API call budget exhausted: 19/19`).
  - 1 call: `/exchanges-list` (70 exchanges) — the smoke test the task budgeted.
  - 1 call: `/eod/AAPL.US` — ONE ticker EOD fetch for the smoke test (250 bars, 2025-11-17..2026-11-13).
  - 17 calls: 16 further large-cap US tickers to give the run a real (if small) universe (AAPL MSFT GOOG AMZN NVDA META TSLA JPM V XOM JNJ PG UNH HD MA WFC GS).
  - symbol-list/delisted probes were covered by the cached exchanges list / skipped by budget; the free-tier delisted-list gap (403, expected) is recorded by the ingest layer, not re-probed.
- Data vintage (sha256 over all warehouse files): `66bc3b9a118228dab35a6fcf91468f33bb7c09cd74d9ebc6b048a676fd37fd72`. Every trial row in the ledger carries this hash — the skip-list key includes it, so a new ingest = new trial space.

## What ran

Server-side pairs-family sweep over the ingested universe (17 symbols with full 250-bar overlap → **136 candidate pairs**, top by |Pearson corr| of daily log returns, deterministic order). Every trial was **registered in the append-only ledger BEFORE any computation** (`register()` mints the trial_id first; re-registering identical (family, scope, config_hash, data_vintage) returns the same id — the skip-list). Each unique trial ran the full gauntlet S0→S4 via `GauntletRunner.run_gauntlet` (S5 pending — no full-window data on the free tier):

- S0 harness calibration: pass 136/136 (beta recovery on levels within 5% tol; spread half-life measured and recorded).
- S1 discovery screen |t|≥2.0 on spread returns: pass 2, fail 134.
- S2 block-permutation null (B=1000, block ⌈T^1/3⌉): fail 2, pending 134 (short-circuit).
- Verdicts: 136 killed (134 at s1, 2 at s2), 0 survivors — **expected and correct for gate-0 on real 1y data**: the purpose was to prove the pipeline runs end-to-end on real data, not to find edges.

## Measured cost (the numbers the task asked for)

| metric | unique trial | cached (skip-list) trial |
|---|---|---|
| wall-clock mean | **0.0448 s** (total 6.09 s / 136) | **78 µs** in-run skip-list hit; **192 µs** per re-register (26 ms / 136) |
| EODHD API calls | 0 per trial at run time; ingest spent 19 calls once, amortized = **19/136 = 0.14 calls per unique trial** | **0** |
| ratio | — | **~571× cheaper per in-run cached hit** (233× vs re-register), 0 API cost |

- Full run wall-clock: **6.50 s** for 136 unique trials + 136 cached re-registrations on 2 vCPU.
- Corr matrix: 17×17 float64 = **2,312 bytes (0.0023 MB)** at this universe size. Scaling: N=1,000 → 1000²×8 B = **8.0 MB**; the full top-1k×250-bar panel is 2.0 MB — memory is dominated by the panel, not the matrix. Peak RSS for the whole run: **0.158 GB** (includes interpreter + pandas).
- Ledger dump summary: 136 trials in stratum `free_1y_top1k` (N counts kills, never pooled); stage rows s0 pass 136, s1 pass 2 / fail 134, s2 fail 2, s3–s5 pending; verdicts all `killed`/`none`.

## How to reproduce

```bash
python3 -m data.ingest_eodhd --warehouse data/warehouse --tickers AAPL ... --max-calls 19   # 19/20 daily calls
python3 -m data.universe --warehouse data/warehouse --n 1000
python3 -m sift.engine.first_run --warehouse data/warehouse \
  --universe data/warehouse/universe/top1000_pit_<asof>.parquet \
  --ledger trials/ledger.db --measurements trials/first-run-measurements.json \
  --max-pairs 136 --s2-B 1000
python3 -m pytest tests/ -q    # 60 passed
```

Raw warehouse data (`data/warehouse/`), the ledger DB and tokens are gitignored — server-side only, no raw data to clients (v3 §12). **SIFT does not trade.**

## Caveats (honest scope labeling)

- Universe here is 17 large-caps, not the full top-1k: the free tier's 20 calls/day caps the *ingested* universe (1 call per ticker). The pipeline itself scales to top-1k (corr matrix 8 MB at N=1000); the gate-0 question — does the gauntlet run end-to-end on real data — is answered yes.
- Window is ~1y (250 bars), labeled `free_1y_top1k`; S4's `tau_cal` is configured for the 5y free scope (`free_5y_top1k`, T=1260) — with T=250 the MinTRL check is the binding constraint, one reason kills concentrate at S1/S2 here. S5 stays pending (needs paid-tier full history).
