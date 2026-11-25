# SIFT v4.2 — Cards & Decks spec (supersedes v4.1 §2 card mechanics framing)

Owner decisions 2026-11-18. This doc reframes the card layer as PORTFOLIO CANDIDATES
and adds a separate DECK (portfolio) layer with its own gauntlet and ranking.
All v3 §12 hard rules and v4.1 §6 legal posture are UNCHANGED (see §5 below).

## 0. Canonical location & supersession map

This is the CURRENT cards & decks build-reference spec. Canonical home: `sift`
repo, `main` branch, `docs/cards-decks-spec-v4.2.md` — the default-branch copy is
the source of truth; feature branches are never "current".

Master index: `sift-research` repo, `main`, `docs/SPEC-INDEX.md`.

Predecessors (repo / branch / path):
- **v4.1** cards & vault layer — folded into v4; no standalone file remains.
  Exploratory draft quoted in hermes-tasks #121; fold decision hermes-tasks #124.
- **v4** consolidated product scope — `sift-research` / `main` /
  `docs/SIFT-Product-Scope-v4-Cards-Vault-Folded.md` (v3 + [v4.1] cards/vault folded).
- **v3** consolidated product scope — `sift-research` / `main` /
  `docs/SIFT-Product-Scope-v3-Research-Updated.md`.

Supersedes: the v4.1 **§2** card mechanics — gauntlet-survivor minting + evidence
tiers (folded into v4 as §6.4.1). The vault (v4.1 §3), anchored search (§3.1) and
badges (§4) are NOT superseded; they remain current via v4 §6.4.2–§6.4.4.
Note: the H1 title previously carried stale "§3–§4" numbering (v4.1 §3 = vault,
§4 = badges); corrected here to §2, the card mechanics actually superseded
(authoritative v4.1 draft: hermes-tasks #121; folded as v4 §6.4.1).
Inherits unchanged: v3 §12 hard rules; v4.1 §6 legal posture (see §5 below).
For product scope outside cards/decks mechanics, v4 (`sift-research`) remains current.

## 1. Two layers, two questions

- CARD = portfolio candidate. Question asked: "how strong is the evidence that this
  connection mean-reverts / leads-lags?" Answer: evidence tier + health. A card is
  NEVER "an instant win" — onboarding and all copy must say so explicitly.
- DECK = the tested unit. Question asked: "does this pre-registered set of cards,
  traded as a portfolio with realistic costs, clear the walk-forward bar?" Answer:
  deck rank (F–S) or amber/informational.

## 2. Card gauntlet (per-card) — scoring, not binary kill

Pipeline execution order (note: stage numbers are labels, not sequence):
S0 → S1 → S2 → S4 → S3 → S5.

- S0 fixture calibration, S1 discovery screen (|t|>=2.0), S2 block-permutation
  (p_perm <= 0.05, B=1000, PCG64) — unchanged.
- S4 walk-forward (formation 60 / step 120, cost-aware) becomes a SCORED stage:
  record net SR, t_net, MinTRL, T — do NOT binary-kill on S4 alone. A candidate that
  passes S0–S2 mints a card at the tier its evidence supports (bronze minimum),
  amber on research data. S4's thresholds stay exactly as-is (S4_NET_SR_MIN=0,
  |t_net|>=tau_cal, MinTRL<=T) — they define the GOLD/PLATINUM bar, not a kill line.
- S3 (DSR >= 0.95, BH-FDR q=0.10 over the scope stratum; N = all registered trials in
  scope across ALL users, kills included; dedupe on family+scope+config+vintage so
  repeated identical runs are one trial) — runs on S4 survivors as today. Owner
  decision 2026-11-18 (hermes-tasks #170): S3's DSR is the MINT-TIME ADMISSION GATE
  ONLY — clearing it admits the candidate above bronze at mint; it does NOT set the
  top tiers and is NEVER re-applied at current N. `dsr`, `sr_star`, `n_stratum`
  are recorded in the stage row as "admission evidence at mint-time N" context.
  Unchanged mechanism, clarified semantics: the luck ceiling scales with pool size
  at MINT time; the tier label itself is N-independent (see §3.1).
- S5 full-depth registry validation on production (EODHD) data only; pending/amber
  on yfinance research data (no delisted-name list → survivorship bias). Unchanged.

## 3. Card tiers & health

Owner decision 2026-11-18 (hermes-tasks #170): the TIER is N-INDEPENDENT and
HEALTH is deliberately N-SENSITIVE. The two properties answer different
questions, so they treat the pool size (stratum N) differently on purpose.

### 3.1 Tier — N-independent, raw stats only
- `tier_from_stats` takes ONLY raw, N-independent, ledger-recorded stats:
  S2 `p_perm`, S4 score bar (`s4_pass`), S5 full-depth t (`s5_pass`,
  `|t_net_full|`). DSR is NEVER an input to the tier.
- Bars (unchanged numbers where they map 1:1 to the pre-#170 pull/31 bars):
  - bronze = S0–S2 pass (minimum card tier; any passer mints);
  - silver = full-depth |t_net_full| >= 3.0 AND p_perm <= 0.05;
  - gold = S4 bar + p_perm <= 0.01 + full-depth |t_net_full| >= 3.0;
  - platinum = S4 bar + S5 pass + full-depth |t_net_full| >= 4.0 AND
    p_perm <= 0.01 (the pull/31 gold bar minus DSR, plus S5 pass).
  - Missing stats never inflate a tier (None components fail closed / drop
    to a lower tier).
- DSR/PSR is the MINT-TIME ADMISSION GATE ONLY: a candidate must clear
  DSR >= 0.95 at mint-time N to mint above bronze (fail => cap at bronze;
  None fails closed). `dsr`, `sr_star`, `n_stratum` are recorded in the
  ledger stage row and on the card row (`mint_dsr`, `mint_sr_star`,
  `mint_n_stratum`) as context, labeled "admission evidence at mint-time
  N". Rationale: DSR is deflated by stratum N (all trials in scope), so a
  card's label must not drift when OTHER users mine more.
- Decay / re-evaluation NEVER re-deflates by current N. Tier moves only via
  the #157 rules on RAW stats over fully out-of-sample trailing live windows
  (card age > 12m = 252 trading days, two consecutive 12m windows, max 1 tier
  change / 90 days, append-only ledger, rarity language only). Downgrades
  (amber) are allowed any time from registry evidence on raw stats.
  `decay.py::meets_tier_bar` uses the raw bars above only.

### 3.2 Health — N-sensitive on purpose (the asymmetry)
- Health is the time-varying property (markets change); tier is absolute
  evidence at mint. So health DOES respond to sample-size effects:
  on each trailing-12m live window re-evaluation, the card's live-window
  Sharpe is compared against the CURRENT stratum (current N, current V̄) —
  a current-window DSR-style deflation, i.e. the expected max Sharpe of
  N_current random strategies over the same window, as a ratio:
  health = SR_live / E[max SR of N_current random strategies].
- If many new candidates are minted in the same regime, the health bar rises
  with N and the card must beat the *current* luck ceiling to count as
  strengthening; if N is unchanged the ratio is stable.
- Health is display-only, never a tier, downgrade-only until fully
  out-of-sample (overlaps the mint window early).
- Deck membership may feed card HEALTH (component of a passing deck
  strengthens health) but NEVER card tier.

## 4. Deck (portfolio) layer — S6 gauntlet

- Deck buckets are a FIXED, enumerable grid (no free-form size): card-count
  {5, 10, 20, 50, 100} × optional sector split (GICS buckets + "all") × family mix
  (pairs, lead-lag, ... — mixing allowed). All buckets available on the free tier;
  this is NOT pay-gated.
- Ranked decks have DETERMINISTIC membership (e.g. "top-K by card tier within bucket",
  K fixed in advance = bucket size). Same deck + same vintage + same config →
  dedupe/cached like cards, so the cross-user pool strengthens the sample instead of
  inflating trial count.
- Custom hand-picked decks: allowed as personal sandbox; score computed and shown
  back to that user as INFORMATIONAL ONLY (amber, unranked, unshareable as a claim),
  because arbitrary C(N,k) selections enable try-until-it-passes.
- S6 portfolio gauntlet on ranked decks: pre-registered basket rules, equal-weight,
  cost-aware walk-forward with REALISTIC per-name half-spreads from the data.
  Two pre-registered price-only proxies run per name and the charged half-spread
  is max(liquidity_concentration, dispersion, 1 bps floor): the dispersion term
  (k·std(|Δlog p|)·1e4) is a VOLATILITY-scaled cost bound, not a liquidity
  measure; the liquidity-concentration term (participation ratio of |Δlog p|)
  is the liquidity-aware proxy, ~invariant to volatility at fixed quote
  continuity (no volume/ADV feed server-side, so a true Amihud is not
  computable from prices alone — soundness #8). The 100 bps placeholder is kept
  as the conservative fallback bound when a name has no usable evidence;
  costs are never understated, never assumed zero. The S6 stage detail records
  which proxy produced the cost (`half_spread_proxy`). Rebalance cadence fixed
  in advance. Pass bar: net SR > 0, |t_net| >= tau_cal(scope),
  MinTRL <= T, PBO <= 0.5 at depth — multiplicity-corrected over the deck stratum
  (N = all registered deck trials in scope, all users, kills included).
- Effective N = independent bets, NOT card count: components sharing names correlate;
  the S6 variance/MinTRL computation must use effective diversification, and the spec
  must state that 50 mixed cards ≠ 50 independent bets.
- Deck ranking: F D C B A S tiers (distinct from card bronze–platinum) = evidence
  strength of the PORTFOLIO claim. Beta: all deck results are amber telemetry,
  never displayed as performance.

## 5. Copy & legal (unchanged hard rules, extended to decks)

- v3 §12 + v4.1 §6: no sellable discovery credit; no tickers+directional calls in
  SIFT content; no P&L/returns/profit-expectation/"value" language anywhere (card
  faces, notifications, RSS/API payloads, deck rankings, onboarding, emails, UI) —
  rarity/evidence/survival language only; banned-phrase grep tests fail-closed on
  ALL copy including new deck strings; no direction/sign in any artifact (payloads
  carry magnitude only, from ledger stage rows, never params_json).
- Cards AND decks non-transferable, no marketplace/resale, no money in beta, ToS
  "not an investment / no resale value / no right to profit" day one, SIFT does not
  trade, signals-only. Paid creator spots showing "how I trade using SIFT" are
  PROHIBITED in beta (§9.8: no paid acquisition channels at all).
- Notifications (#158, merged pull/33): rich signal events — magnitude, threshold,
  trailing-12m hit count, median hold, half-life, health ratio, tier, aggregate
  cards-fired count; email/RSS/API per-card opt-in, quiet hours, free=digest /
  paid=live. Onboarding (#160, merged pull/32+): plain-language faces, gauge health,
  glossary, and the explicit line that no single card is an instant win — cards are
  evidence about connections; decks are the tested portfolios; the user decides.
- Raw market data server-side only; shared artifacts = anonymized aggregates +
  watermark.

### 5.1 Private owner spec & unified slot mining contract (owner decisions 2026-11-25)

Aligns v4 §12 row 3 ("owner-visible page naming tickers; the stricter anonymized card
face is what leaves the platform") with `sift-research/docs/mining-modes-policy.md`:

- **Public / general / shareable surfaces remain 100% zero-ticker fail-closed:**
  `guard_api_payload` stays unchanged on every catalog, run, SSE, trial-list, card-face,
  PNG, vault, connection, telemetry, email, RSS, and webhook payload. Off-platform hit
  notifications (`sift/cards/events.py`) remain strictly ticker-free (`card_id` +
  magnitude `|z|` + cue to check the authenticated Private Owner Spec in-app).
- **Dedicated Private Owner Spec (`/api/private/cards/{card_id}/spec` and
  `/api/private/trials/{trial_id}/spec`):** strictly authenticated, ownership-verified,
  rate-limited, non-cacheable (`Cache-Control: no-store, private`, `X-Robots-Tag:
  noindex, nofollow`), watermarked non-shareable, and guarded fail-closed by a separate
  `guard_owner_private_payload` enforcing:
  - **Allowed allowlist:** structured `instruments` (`[{"role": ..., "symbol": ...}]`,
    confined strictly to that single owned trial/card's exact symbols, never bulk dumps),
    user-declared statistical parameters (`family`, `mining_mode`, `subspace_filter`,
    `subspace_size`, `scope_id`, `formation`, `open_z`, `close_z`, `holdout_bars`),
    survival facts (`gauntlet_stats` with S0 `beta_hat` stripped), magnitude-only live
    signal (`current_abs_z`, `gauge.band`), durability/health facts, and `hit_loop_hint`.
  - **Hard exclusions (forbidden even in the private view):** raw market data/OHLCV/bars,
    raw `params_json` passthrough, vendor-fitted `slope` / `beta` / `beta_hat` /
    `beta_levels` / `hedge_ratio`, sign/direction (`+z`/`-z`, rich/cheap leg,
    buy/sell/long/short), P&L/returns/profit/"value" language, banned phrases, email PII.
  - **Eligibility:** owned cards are always eligible; trials are eligible iff
    `mining_mode == "direct"` OR `verdict == "survivor"`. Killed `anchored` and
    `uniform_random` trials never reveal server-drawn unknown legs.
- **Unified Slot A / Slot B mining modes (zero hidden pre-screening):** user-facing runs
  derive `mining_mode` deterministically from Slot A and Slot B with zero data-peeking
  server shortlists: `direct` (specific + specific), `anchored` (specific + `Random(All)`
  or `Random(<Sector>)`, seeded `PCG64` uniform draw, recording `anchor_id_hmac` +
  `subspace_filter` + `subspace_size`), and `uniform_random` (`Random` + `Random`,
  seeded `PCG64` uniform draw). S5 full-depth validation runs automatically (never opt-in,
  never auto-pass) across all three modes.

## 6. Data & caveats

- yfinance 5y = research feed (research_grade, active-only; S5 amber; never conflated
  with production scopes). EODHD = production vendor (docs/vendor-licensing.md,
  pull/26): server-side aggregate-only design is compliant with the personal plan.
- Full-run result 2026-11-18 (research_5y_top1k, 270,480 candidate pairs): 0 survivors
  at the old binary S4 bar; top S4-fail net SR 1.10 (APO/BJDX) failed on t_net/MinTRL;
  S2 survivors clustered on few common legs (BJDX, ALLR) — single-name artifacts S4
  correctly killed. This motivates the candidate-tier change (§2), NOT weaker
  thresholds.

## 7. Flag & kill-switch

All of the above behind `cards_v4_1` flag (deck layer adds `SIFT_DECKS_V4_2`; private
owner spec adds `SIFT_OWNER_PRIVATE_SPEC`), fail-closed when off, kill-switch envs
`SIFT_CARDS_V4_1=0`, `SIFT_DECKS_V4_2=0`, `SIFT_OWNER_PRIVATE_SPEC=0`.


