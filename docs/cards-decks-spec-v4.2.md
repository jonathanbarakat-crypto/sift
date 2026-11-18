# SIFT v4.2 — Cards & Decks spec (supersedes v4.1 §3–§4 mechanics framing)

Owner decisions 2026-11-18. This doc reframes the card layer as PORTFOLIO CANDIDATES
and adds a separate DECK (portfolio) layer with its own gauntlet and ranking.
All v3 §12 hard rules and v4.1 §6 legal posture are UNCHANGED (see §6 below).

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
  repeated identical runs are one trial) — runs on S4 survivors as today; sets the
  top card tiers. Unchanged mechanism, clarified semantics: the luck ceiling scales
  with pool size.
- S5 full-depth registry validation on production (EODHD) data only; pending/amber
  on yfinance research data (no delisted-name list → survivorship bias). Unchanged.

## 3. Card tiers & health

- Tiers bronze/silver/gold/platinum = evidence strength at mint (S2 p, S4 score,
  S3 DSR, S5 full-depth t) + durability. Bidirectional per hermes-tasks #157 (merged
  pull/31): upgrades only on pre-registered stats over fully out-of-sample trailing
  windows (card age > 12m, two consecutive 12m windows), max 1 tier change / 90 days,
  append-only ledger, rarity language only.
- Health = rolling trailing-12m sim vs mint baseline (overlaps mint window early —
  downgrade-only until fully live). Health is display-only, never a tier.
- Deck membership may feed card HEALTH (component of a passing deck strengthens
  health) but NEVER card tier.

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
  cost-aware walk-forward with REALISTIC per-name half-spreads from the data
  (replace the 100 bps placeholder; keep 100 bps as conservative bound), rebalance
  cadence fixed in advance. Pass bar: net SR > 0, |t_net| >= tau_cal(scope),
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

All of the above behind `cards_v4_1` flag (deck layer may add `SIFT_DECKS_V4_2`),
fail-closed when off, kill-switch env `SIFT_CARDS_V4_1=0`.

