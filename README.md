# SIFT

Server-side cross-stock pattern mining with a pre-registered statistical
gauntlet. **SIFT does not trade** — signals only, never execution. No raw
market data is ever served to clients — browsers receive derived outputs only
(v3 scope doc §3, §4.1, §12). Public content shows survival facts only — no
returns, no win rates, no directional calls. Every trial is recorded in an
append-only, scope-stratified trial registry (the honesty ledger), which is
the denominator for all multiplicity-corrected statistics.

Hard rules (v3 scope doc §12): server-side mining only; no raw data to any
client; registry-depth validation (S5) is automatic and never opt-in;
DSR denominators are scope-stratified; public cards show survival facts only.

## Layout

- `sift/registry/` — append-only trial ledger (SQLite now, Postgres-compatible
  schema)
- `sift/gauntlet/` — stages S0–S5: `stats.py` (pure stat functions),
  `s5.py` (registry-depth validation + delist-safe panel helpers),
  `config.py` (pre-registered thresholds), `runner.py` (stage wiring + ledger)
- `sift/cards/` — the whole v4.1 card layer behind `cards_v4_1` (registry,
  minting, vault, faces, anchored, share, notify, decay, metrics,
  `service.py` composition root) — see "Feature flags (v4.1)" below
- `sift/engine/` — server-side mining engine (first trial run)
- `sift/data/`, `sift/web/` — scaffolded packages (ingest/landing live at
  repo-root `data/` and `web/`)
- `data/` — server-side EODHD ingest layer + versioned warehouse (gate 1)
- `web/` — static landing page + email-only beta waitlist (gate 1)

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/
```

## Gauntlet stages

S0 calibration → S1 discovery screen (|t|≥2.0) → S2 block-permutation null
(PCG64, block ⌈T^1/3⌉, p=(1+#extreme)/(B+1)) → S3 scope-stratified DSR≥0.95 +
BH-FDR → S4 cost-aware walk-forward (|t_net| ≥ tau_cal: 3.5 at the 5-yr free
scope, 3.0 at full depth) → **S5 registry-depth validation**.

S5 (v3 §8.1-§8.4) runs **automatically on every S0-S4 survivor — never
opt-in**: the candidate is re-evaluated at `registry_depth` scope (full
window, full universe incl. later-delisted names) at full-strength thresholds
(t ≥ 3.0 full-depth vs 3.5 free-scope tau_cal, p_perm ≤ 0.05, DSR ≥ 0.95 with
the registry_depth stratum, PBO ≤ 0.5). Pass → `validated-green`; fail →
`on-watch-amber` (never validated-green). Post-delist prices are masked out
everywhere (F-DELIST). Acceptance fixtures: `F-REGIME` (edge only in years 0-7
of 16 — the 5y in-edge window shows t>2, the full window attenuates so S5
kills it) and `F-DELIST` (20% of a universe delists mid-window; the pipeline
must not crash and must not use post-delist prices). See
`tests/gauntlet/test_s5.py`.

### Statistical hygiene (soundness #12/#13/#14)

- **Serial correlation (#12).** `t_stat_sharpe`, the `sqrt(T-1)` in PSR and
  the `sqrt(252)` annualization assume iid returns. Under *positive*
  autocorrelation the iid t is anticonservative (std of t ≈
  √((1+φ)/(1−φ)) > 1); under mean reversion it is conservative. The pairs
  gauntlet path keeps the iid variants (the pairs family is
  mean-reversion — the conservative direction; switching it to HAC would
  *loosen* a currently conservative gate, so it is deliberately NOT
  switched). Positive-autocorrelation exposures (deck baskets, pull/37)
  must use `stats.t_stat_sharpe_hac` (Newey–West, Bartlett kernel, Andrews
  lag `floor(4·(T/100)^(2/9))`) or pass `stats.effective_sample_size` into
  `psr`/`dsr` via their `t_eff` argument — every switch is explicit and
  pre-registered, never silent.
- **Bias-corrected moments (#13).** `stats.skew_kurt` now returns the
  bias-corrected adjusted Fisher–Pearson G1 and the RAW (non-excess)
  kurtosis G2 (the old mean(z³)/mean(z⁴) was downward-biased: MC mean kurt
  2.80 at n=60 vs true 3.0). PSR/DSR/MinTRL consume the raw kurtosis
  unchanged (PSR expects RAW in its `(kurt−1)·sr²/4` term). n<4 / zero-
  variance guards unchanged.
- **S0 on real data (#14, doc-only).** On real data S0's beta assertion is
  tautological (beta_true = the pair's own levels OLS, so
  `beta_hat − beta_true = 0.0` exactly); the genuine real-data gate is the
  finite-half-life (mean-reversion) check — an explosive spread
  (φ̂ > 1 → half-life = inf) FAILS S0. The planted AR(1) fixture (F-PAIR)
  remains the real calibration of both assertions.

## Data pipeline (gate 1)

`data/` is the server-side ingest layer (v3 scope doc §3.2: versioned
warehouse, data vintage = content hash; §4.5/§12.8: survivorship-bias-free
data at every tier):

- `data/eodhd_client.py` — EODHD free-tier client. Hard limits encoded:
  20 API calls/day, past-1-year EOD only, live quotes included. Token comes
  from `--api-token` / `$EODHD_TOKEN` / `pass eodhd/api-token` and is never
  printed, logged, or committed. Deep-history (pre-1-year) probes raise a
  clear `DeepHistoryBlocked` error — expected on the free tier, never
  bypassed.
- `data/ingest_eodhd.py` — CLI: fetch exchanges-list once (cached), then
  per-ticker daily EOD for US stocks → one parquet per ticker under
  `data/warehouse/eod/US/`, plus the data-vintage content hash
  (`data/warehouse/VINTAGE`). Resumable: cached parquet covering the
  requested window is skipped without spending a call; one call per ticker
  per run max; `--max-calls` (default 19, under the hard 20/day) is enforced
  before every HTTP request and the daily count is persisted in
  `state/calls.json` so reruns don't re-spend calls.
- `data/universe.py` — CLI: point-in-time top-N (default 1,000 = free-tier
  universe) by average dollar volume over the available window as of
  `--asof`. **No survivorship bias**: names whose series ended before the
  as-of date stay in the ranking/listing with `ended_before_asof` /
  `delisted` flags and their `last_date`. On a **research-grade** vintage
  (yfinance/Yahoo) the universe is active-only: `delist_idx_available=False`
  and the manifest says `survivorship_bias: "BIASED …"` (v3 §4.5/§12.8).
- `data/warehouse.py` — parquet IO + data-vintage hash (sha256 over all
  warehouse data files).
- `data/yf_client.py` + `data/ingest_yf.py` — **RESEARCH-ONLY** yfinance
  (Yahoo) feed, ~5y daily for initial testing; writes `research_grade: true`
  + `vendor: yfinance` into the vintage. EODHD stays the production vendor —
  see `docs/research-feeds.md` and `docs/vendor-licensing.md` (pull/26).

```bash
python3 -m data.ingest_eodhd --warehouse data/warehouse --max-calls 19
python3 -m data.ingest_eodhd --tickers AAPL MSFT --refresh-live
python3 -m data.ingest_eodhd --exchanges-list-only
python3 -m data.ingest_eodhd --probe-deep-history AAPL   # expected to fail on free tier
python3 -m data.universe --warehouse data/warehouse --n 1000 --asof 2026-11-15
python3 -m pytest tests/ -q
# research feed (yfinance/Yahoo, RESEARCH-ONLY — never production):
python3 -m data.ingest_yf --warehouse data/warehouse --period 5y --seed-list sec --top 500 --sleep 1.5
python3 -m sift.engine.first_run --warehouse data/warehouse --universe data/warehouse/universe/top500_pit_<asof>.parquet --source yf --window 5y
```

`data/warehouse/` (raw vendor data + any tokens) is **gitignored** — this
repo is public; raw data lives only on the server under the internal
commercial license.

## Landing page + waitlist (gate 1)

```
web/                  static landing page (plain HTML/CSS/JS, no framework)
  index.html          landing page
  styles.css          styles
  app.js              waitlist form handling (progressive enhancement)
  api/waitlist.py     optional stdlib-only server: static host + POST /api/waitlist
  api/test_waitlist.py  stdlib unittest suite for the endpoint
```

The page deploys to **any static host** as-is (GitHub Pages, Netlify,
Cloudflare Pages, S3, nginx, ...). From this repo:

```bash
# Option A — any static server pointed at web/
python3 -m http.server 8080 --directory web

# Option B — the bundled stub, which also provides the waitlist endpoint
python3 web/api/waitlist.py            # http://127.0.0.1:8080/
PORT=9000 python3 web/api/waitlist.py  # custom port
```

The form POSTs to `/api/waitlist` with `{"email": "..."}` (JSON, or
form-encoded `email=...` as fallback); the bundled stdlib stub appends to
`data/waitlist.csv` (gitignored; duplicates ignored). Sign-ups are
email-only. The stub never serves `.py`/`.csv`/`.json`/`.md` under `web/`,
guards body size, blocks path traversal, neutralizes spreadsheet formula
prefixes, and serializes writes. Endpoint tests (stdlib only):

```bash
python3 -m unittest discover -s web/api -p "test_*.py" -v
```

## Beta

The first phase is a **limited alpha/beta**: an invited-only group of
**~10–30 users** (n small on purpose — the beta population is the
measurement instrument; results are reported as counts/ranges/qualitative
findings, never point estimates). No public signup — the landing page is an
email-only waitlist — and **no money changes hands** during the beta. ToS
"not an investment / no resale value" language is live from day one. Full
plan: `docs/beta-plan.md` (gate-1 measurement definition, small-n reporting
rules, §8 open-question statuses).

## Non-negotiables (scope doc v3 §12)

1. Signals, never execution.
2. No raw data reaches any user device — derived outputs only.
3. No tickers + directional calls in any SIFT-published content.
4. No win rates, no P&L leaderboards, no picks.
5. Registry-depth validation (S5) is automatic — never opt-in.
6. Scope is a first-class field on every trial; DSR denominators stratify by
   scope and are never pooled across scopes.
7. Public cards show survival facts only.

## Feature flags (v4.1)

`sift/flags.py` holds server-side flags. `cards_v4_1` wraps the whole v4.1
card layer and defaults **ON for the beta cohort**:

- `sift/cards/` — card registry + minting on validated-green survivors
  (pull/13), per-user vault + ghost FIFO (pull/14), server-rendered
  anonymized card faces with share watermark (pull/15), anchored run mode
  (pull/16)
- `sift/cards/share.py` — share-for-capacity: +1 permanent vault slot per
  verified share of a card's anonymized face, one per (account, card),
  hard cap +5, with the anti-abuse stack (device/IP clustering, per-device
  rate limits, live-post verification, append-only audit log, fraud queue)
  the trial registry (kill rate, pre-registration streak, peer re-test,
  scope honesty), opt-in display, fail-closed guard against any
  P&L/win-rate/directional payload
- `sift/cards/notify.py` — per-card opt-in hit notifications; free tier
  defaults to a rate-capped digest, paid may go live; signals-only. All
  user-facing copy is built only in `copy_for_event` and grepped fail-closed
  (banned phrases + whole-word directional bans + P&L/value language)
- `sift/cards/events.py` — rich v1 signal-event package for hit
  notifications: versioned payload (magnitude-only statistic + threshold,
  persistence/health facts from ledger stage rows only — never params_json,
  anonymized `cards_fired_today`), per-card channel prefs (email / RSS
  token feed / API webhook with HMAC signature), quiet hours; fail-closed
  on any sign/symbol/banned phrase
- `sift/cards/copy.py` — the newcomer-friendly copy + onboarding layer: the
  ONLY place the new UX/onboarding strings are built (face explainer,
  glossary tooltips, gauge labels, health phrases, onboarding statement,
  day-one ToS, hit-loop hint). Every string is scanned against
  `BANNED_PHRASES` (no buy/sell/long/short, no P&L/returns, no
  profit-expectation/"value" language, no urgency/countdown) and the builder
  raises `CopyError` on any hit — a banned phrase refuses the render. The
  card face carries the explainer + glossary + gauge (stretched/normal/
  compressed from |z| only, numeric |z| one click away) + health (plain
  ratio vs mint baseline); `render()`/`render_share()` take optional
  `current_abs_z` / `health_ratio` live readings (magnitude is always |z|,
  never a sign). Onboarding copy is also served by `GET /api/onboarding`
  (web/api/waitlist.py) and the waitlist signup requires ToS acceptance
  (`tos_accepted`, recorded in the CSV)
- `sift/cards/decay.py` — registry re-evaluation job: BIDIRECTIONAL tiers.
  Downgrades (amber) still happen any time from registry evidence; upgrades
  fire ONLY through pre-registered live-window rules (card age > 12m so the
  trailing window is fully out-of-sample, two consecutive annual windows at
  the mint-time bar in WINDOW-LENGTH-INVARIANT terms — annualized Sharpe,
  never raw t: |t_win| >= 4.0*sqrt(252/4032) = 1.0 AND p_perm <= 0.01 for
  silver->platinum, |t_win| >= 3.0*sqrt(252/4032) = 0.75 AND p_perm <= 0.05
  for bronze->silver, i.e. annSR >= 1.0 / 0.75 — the identical mint-time
  full-depth bar, not ~4x stricter (soundness #5); DSR stays out of the
  live-window bar per #170 — max one tier change per card per 90 days,
  seeded PCG64 re-evaluation, append-only `card_upgrade` ledger event with
  evidence), publicly explained, never by payment or chance; notification
  hook
- `sift/engine/first_run.py` — `--holdout-bars N` (default 252) locks the
  most recent N trading days out of S0–S4 discovery/walk-forward;
  survivors are evaluated once on that frozen holdout (the mint-time
  live-baseline stored on the card); the holdout is never used for tuning
- `sift/cards/metrics.py` — gate-1 metric hooks (instrumentation stubs over
  the server-side stores until the event-store measurement pipeline lands):
  D90 retention split card-holders vs non, ghost ratio, vault turnover,
  share-grant -> D30 retention, anchored-vs-unanchored hit rates
- `sift/cards/service.py` — `CardLayer`, the composition root wiring all of
  the above behind the flag

**Kill-switch:** set env `SIFT_CARDS_V4_1=0` (global) or call
`flags.set_flag("cards_v4_1", False)` / `flags.disable_for_user("cards_v4_1", handle)`
(per user). With the flag off every cards module fails closed with its own
error type (`VaultError` / `FaceError` / `ShareError` /
`NotifyError` / `DecayError` / `CopyError` / `LayerError`, all carrying
`flag_off`), the
runner mints no cards, and no new engagement is granted. Reads stay
available — already-minted cards are append-only registry records and are
never hidden, destroyed, or randomized by the kill-switch. Unknown flags
fail closed (`is_enabled` -> False, `set_flag` raises). The gauntlet itself
(S0-S5) is unchanged by the flag.

**Legal posture (v3 §12 + v4.1 mechanics, restated for the card layer):**
cards are **non-transferable** (ownership is asserted on every path);
share-for-capacity slots are non-transferable, expire with the account,
and are never sellable — no money changes hands; there is **no P&L, no win
rates, no directional calls, no market language** anywhere in card content
there is **no randomness for sale** (no lotteries, no paid random draws —
tier changes happen only through the pre-registered registry rules and never
re-randomize); and there is **no marketplace** (cards and slots cannot be
bought, sold, or traded).

## Deck layer (v4.2, spec §4 — hermes-tasks #166)

`sift/decks/` implements the v4.2 deck layer behind the `decks_v4_2` flag
(which additionally requires `cards_v4_1` — the deck layer sits on top of
the card layer):

- `sift/decks/__init__.py` — bucket registry + append-only deck ledger:
  the FIXED enumerable bucket grid (sizes {5,10,20,50,100} x optional GICS
  sector split ("all" + sectors) x family mix (pairs, lead-lag, mixing
  allowed) — all buckets free-tier), deterministic top-K membership by card
  tier (K = bucket size), dedupe on (scope, bucket config, vintage) like
  cards, F–S deck ranking (`deck_rank_from_stats`, distinct from card
  tiers), custom decks registered as `kind="custom"` (informational only:
  amber, unranked, unshareable), deck events append-only with read-back,
  decks non-transferable
- `sift/decks/service.py` — `DeckService`: ranked/custom registration and
  the S6 runner (`run_s6`): equal-weight basket, fixed rebalance cadence
  (21 bars), cost-aware walk-forward with realistic per-name half-spreads
  computed from the data (100 bps placeholder replaced, kept as the
  conservative bound), PBO at depth, pass bar net SR>0 / |t_net|>=
  tau_cal(scope) / MinTRL<=T / PBO<=0.5 multiplicity-corrected over the
  deck stratum (N = all registered deck trials in scope across all users,
  kills included) using EFFECTIVE N (independent bets — components sharing
  names correlate; 50 mixed cards != 50 independent bets)
- `sift/gauntlet/s6.py` — pure S6 statistics: data-driven half-spread
  estimator, equal-weight basket with fixed cadence + turnover costs,
  `effective_n` (name-overlap/Jaccard), `multiplicity_corrected_p`,
  `deck_gauntlet` verdict
- `sift/decks/copy.py` — the ONLY place deck strings are built; every deck
  string passes the SAME fail-closed banned-phrase guard as cards (v3 §12
  bans extended to ALL deck strings); beta posture: deck results are amber
  telemetry only, never performance claims

**Kill-switch:** `SIFT_DECKS_V4_2=0` disables only the deck layer;
`SIFT_CARDS_V4_1=0` disables both (decks require cards). Flag off -> every
deck path fails closed with `DeckError`/`DeckCopyError` (`flag_off`).

## Simulated-user harness (`sift/sim/`) — SIMULATED, infrastructure validation only

> **SIMULATED - infrastructure validation, not user-behavior evidence.**
> The sims in `sift/sim/` generate **synthetic users** (`sim-user-XXXX`) and
> **synthetic instruments** (`SIM####`) against an **ephemeral test DB**
> (`SIFT_ENV=test` is hard-required — the CLI refuses otherwise). They
> validate *infrastructure*: queueing, DB contention under concurrent lanes,
> the v4.1 §4.7 anti-abuse controls, append-only log integrity, and the
> card-layer lifecycle. Per spec §8, sim output says **nothing about real
> users** and must never be cited as user-behavior evidence. Every report
> (JSON and text) leads with the banner above, enforced in code
> (`sift/sim/header.py`).

- **Load soak** — N simulated users × runs-per-user mining runs through the
  real register → gauntlet S0–S5 → verdict → mint/admit path, on concurrent
  worker lanes with their own DB connections (real SQLite contention).
  Reports latency p50/p95/p99, queue depth, and asserts no
  deadlocks/timeouts/lane errors. Wall-clock timings are reported only
  under a `measured` block; `deterministic_view()` strips them so
  **same seed → same report** (seeded PCG64 RNG + fixed injected clock).
- **Anti-abuse attack sims** — device/IP/ASN clustering, day-zero sharing
  before first completed run, unverifiable share posts (grant must NOT
  fire), duplicate shares of the same card, per-device velocity limits
  (>3 grants/device/30d), the +5 slot cap, revocation on an abuse flag, and
  fraud-queue SLA routing. Asserts the append-only `share_grant` log
  rejects UPDATE/DELETE via its triggers and that the audit log is written.
- **Card-layer lifecycle sim** — mint → duplicate-spec ghost mint →
  FIFO promotion on discard → decay (downgrade-only, never upgrades) →
  archive. Behind the `cards_v4_1` flag, fail-closed when off; the
  `SIFT_CARDS_V4_1=0` kill-switch is respected (`tests/sim/test_killswitch.py`).

Run it (test env only):

```bash
SIFT_ENV=test python3 -m sift.sim.run --sim all --n-users 50 --runs-per-user 3 --workers 8 --out sim-report.json
```

CI runs the full suite plus a small-N (20 users) smoke soak
(`.github/workflows/ci.yml`, job `sim-smoke`); tests live in `tests/sim/`.

**Persona spec** — `sift/sim/personas.json` (72 simulated-user personas,
schema-validated by `sift/sim/personas.py`) + `sift/sim/bots.py` (seeded,
deterministic bot sessions) + rationale/distribution tables in
`docs/sim-user-personas.md`. Personas parameterize the bots ONLY — the mix
is a design choice for coverage, **not** a market estimate, and is never
user-behavior evidence (spec §8 is answered from the limited alpha/beta,
pull/25). Every bot run logs persona id + seed; every report carries the
SIMULATED banner.

## Non-negotiables (scope doc v3 §12)
