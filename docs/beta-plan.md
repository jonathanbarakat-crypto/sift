# SIFT — Limited Alpha/Beta Telemetry Plan & Gate-1 Measurement Definition

**Status:** plan / spec (research + design; no code, no outreach executed)
**Date:** 2026-11-17 (rewrite; external sources originally accessed 2026-11-15 UTC)
**Task:** t_788cd399 (supersedes the 2026-11-15 cohort plan, pull/1–pull/3, task t_56db5ac0)
**Decision:** user decision 2026-11-17 (hermes-tasks #143): the 50–100-quant gate-1 cohort is **replaced by a limited alpha/beta** — the beta population (small invited group, **n ≈ 10–30**) **is the measurement instrument**.
**Spec source:** `SIFT-Product-Scope-v3-Research-Updated.md` on branch `docs/sift-scope-v3` of `jonathanbarakat-crypto/sift-research` (PR pull/3) — esp. §2.4 gate 1, §8.5 funnel, §10 metrics, §12 non-negotiables, §14 open decision #4; v4 scope fold on branch `docs/sift-scope-v4` (sift-research pull/7, §2.4/§2.4a — same rewrite mirrored there); v4.1 "middle ground" spec (hermes-tasks #121) §6 legal posture, §7 secondary metrics, §8 open questions. Supporting: `sift-research` `docs/market-sizing-verification.md` (42 cited sources), `docs/assessment.md`; FL `research/design/2026-11-15-sift-gauntlet-s5-design.md` and `research/notes/2026-11-15-006-sift-registry-discovery-credit-survivor-classes-design-spec.md` (trial-ledger schema: `trial_id`, `scope_record`, `provenance`, `config_hash`, `data_vintage_hash`, timestamps).
**Hard rules honored throughout (v3 §12 + v4.1 spec):** server-side mining only; no raw data to clients; S5 never opt-in; scope-stratified denominators; public cards show survival facts only; "SIFT does not trade." Cards non-transferable; no paid randomness; no profit-expectation or "value" language anywhere (UI copy, emails, notifications included). This plan adds: no PII beyond email in the beta measurement; no paid acquisition channels (§9.8); no money changes hands during the beta.

---

## 0. What changed on 2026-11-17 and why

The original plan (pull/1–pull/3, 2026-11-15) defined gate 1 as a **50–100-quant cohort with a count pass line (≥20 active at D90)**. That is replaced by a **limited alpha/beta telemetry plan**:

1. **Population:** an **invited-only beta of n ≈ 10–30 users**, invited in waves by the founder (§2). The landing page stays **waitlist-only (email-only) — no open signup during the beta**. **No money changes hands**: the beta is free; invited users may get Pro-tier trial *budget* for the beta window, disclosed in the invite, never a paid anything. The ToS "cards are not investments / no resale value / no right to profit" language is **live from day one** (v4.1 spec §6; §10 below).
2. **Measurement stance:** at n ≈ 10–30, point estimates are not decision-grade. Wilson 95% CI widths at these n are ~29–53 points (computed 2026-11-17, binomial Wilson score interval, z=1.96: n=30, k=15 → [33%, 67%], width 34pp; n=20, k=10 → [30%, 70%], width 40pp; n=10, k=5 → [24%, 76%], width 53pp). So all gate-1 and secondary metrics are reported as **counts, ranges, and qualitative findings — never as point estimates** — with the CI printed alongside every rate (§5, §7).
3. **Why smaller is more honest, not less:** the old n=50–100 count gate (≥20 of 100 = 20%, CI [13%, 29%]) could not actually resolve the ~50%-vs-market-baseline question it was meant to answer either; it just looked like a number. The limited beta commits to what small n *can* answer — gross failure vs. healthy engagement, qualitative splits, directional reads — and explicitly defers what it cannot (§6, §8). A solo operator can also genuinely onboard 10–30 people personally (screening call + first-run walkthrough each, v3 scope §11); 50–100 was already past the honest solo limit.
4. **Simulated users are infrastructure validation only** (load soak, anti-abuse attack sims — hermes-tasks #144). They are **never evidence for any user-behavior question** in §8; see §9. No document may claim "simulation proves behavior" — that is circular.

---

## 1. Goal

Answer §2.4 gate 1 honestly before scaling: **"Can a small invited group of serious quants complete mining runs and remain active at 90 days?"** This remains the direct test of the honesty-churn hypothesis (v3 §2.2, §16): the product's most common answer is "no," and the hypothesis is that users who accept that answer retain, while hype-driven users churn the day their first pattern dies. What changes is the *read*: a qualitative/range read over n ≈ 10–30, not a count pass/fail over 50–100.

---

## 2. The limited alpha/beta population

### 2.1 Size, invitation, money

- **n ≈ 10–30 invited users**, invited in waves by the founder. Invite-only: no public signup, no public offering; the waitlist collects emails only.
- **No money changes hands.** Free during the beta. Invited users may receive Pro-tier trial *budget* (scope/priority) for the beta window, disclosed in the invite — a product-exposure decision, not a purchase. This is what makes the beta a *limited* beta in the legal sense (§10).
- **ToS live from day one:** cards are not investments, have no resale value, confer no right to profit; no "value" language anywhere; cards non-transferable; no paid randomness (v4.1 spec §6; §10 below).

### 2.2 Who counts as an invitee (screening — answerable in a 15-min call or form)

1. **Seriousness:** ≥1 year running systematic/backtested strategies and writes code (Python/C#/Julia) — the "serious quant" filter from v3 §2.1.
2. **Motivation check:** asks about *validation/honesty* features, not "find me alpha." Anyone whose stated goal is passive income or signals-for-sale is a polite no — they are the negative-LTV segment v3 §2.1 explicitly excludes.
3. **Consent:** agrees to the measurement protocol in §5 (events listed, email as the only identifier, aggregate-only reporting, ranges/qualitative-only publication).

Explicitly **not** screened for: AUM, geography beyond English-language, platform preference. The old plan's **WTP screen is removed from the beta** — with no money changing hands, willingness-to-pay cannot be measured in beta; it is **deferred to post-launch paid-conversion analytics** (§8 status table).

### 2.3 Recruitment channels — founder-personal invites only (verified 2026-11-15)

All organic/community-led per v3 §9.3–9.4 and §9.8 (no paid social, no affiliates). During the beta these channels carry **founder-written, value-first posts that point to the email-only waitlist and describe the invited beta — never an open signup**. Expected yield into n ≈ 10–30 is an estimate, labeled as such.

| # | Channel | Concrete link / handle | Why it reaches the right people | Est. yield into 10–30 |
|---|---|---|---|---|
| C1 | QuantConnect community forum | https://www.quantconnect.com/forum/ (posting requires 30% Boot Camp — post as the founder account after completing it, or answer questions where posting is open) | 565K registered / ~50k MAU [S4][S5]; the densest concentration of exactly this user (LEAN users already backtesting) | 3–6 |
| C2 | QuantConnect Discord | https://www.quantconnect.com/discord (vanity `discord.gg/quantconnect`; ~5.8k members per disdex listing [S6]) | Real-time; ideal for the "first mining run" walkthroughs | 1–3 |
| C3 | r/algotrading | https://www.reddit.com/r/algotrading/ (~1.9M members [S7]); weekly discussion threads are the sanctioned self-promo venue; a dedicated post must add value (e.g. "we ran X through pre-registered gauntlet — here's the kill") | Densest serious-retail quant subreddit; skeptical audience that converts if answered honestly (§9.3) | 2–5 |
| C4 | r/quant + r/systematictrading | https://www.reddit.com/r/quant/ , https://www.reddit.com/r/systematictrading/ | Smaller, more academic/systematic; good for the screened invitees | 1–2 |
| C5 | X quant community ("stats corner of fintwit") | Handles: @lopezdeprado (24.9k, LdP), @echanQT (Ernie Chan, 3.7k), @aclenow, @alphaarchitect, @choffstein, @QuantEdges, @MebFaber, @GaryAntonacci, @investingidiocy (Rob Carver) [S8][S9]; search terms: #algotrading #quant #backtesting #stat arb | The stats-heavy corner where the honesty positioning lands; quote-post threads about overfitting/backtesting honesty | 1–3 |
| C6 | Discord servers (discoverable directory, searched 2026-11-15) | "Quant Talk" (~3.0k members, algo/quant/data-science discussions), "Quant Trading App" (~11k members, quant-as-edge), Numerai Discord `discord.gg/RD9uhjnJHP` (~9.3k members [S10]) + forum https://forum.numer.ai/ | Numerai community is the closest analog population: self-directed data-science market modelers, many already pay for tools/data | 1–3 |
| C7 | YouTube stat-arb / backtesting educators (partnership per §9.4) | Algovibes (~30k subs; "I backtested a pairs trading strategy… does it work" — documents simplifications incl. survivorship bias), QuantPy (Python quant implementation), Part Time Larry (IB API tutorials), Kevin Davey "Algo Trading With Kevin Davey" (~50k subs, backtesting methodology/overfitting), QuantInsti (~50k) [S11][S12] | "We tested this channel's pairs tutorial through pre-registered OOS checks" is the §9.4 asset; audience is already building systems | 1–2 |
| C8 | Hacker News Show HN + r/algotrading launch thread | https://news.ycombinator.com/show (precedent: QuanterLab Show HN 2026-11-15, quant backtesting tools are a proven HN category [S13]) | Technical credibility; the skeptics in the comments are the filter (§9.3) | 1–2 |
| C9 | Wilmott forum + Elite Trader | https://forum.wilmott.com/ (Trading Forum: 2,400 topics/17.7k posts, active Nov 2026 [S14]); https://www.elitetrader.com/ (120k members [S15]) | Professional/semi-pro quants; lower volume but highest seriousness; also the fund-pilot hunting ground (§2.4 gate 2) | 0–1 |
| C10 | Kaggle competition communities (organic mention) | Jane Street Real-Time Market Data Forecasting: 25.4k entrants / 3.6k participants [S16]; forum posts pointing to "how to validate without overfitting" content | Data-science traders; WTP-adjacent (they spend on compute/data) | 0–1 |

**Sequencing:** C1–C4 first (highest density, lowest cost), C5/C6 in parallel for the honesty-positioning content, C7–C10 at/after public beta open. Do **not** run C1–C4 as ads or affiliate posts — §9.8 prohibits; all posts are founder-written, value-first, answer skeptics in-thread, and end at the waitlist.

**Invite mechanics:** one landing page with (a) the §9.2 interactive teaser (synthetic data, no signup), (b) the free deflated-Sharpe/PBO calculator as lead magnet, (c) an **email-only waitlist form** (no account creation during the beta). The founder sends personal invites (email + screening call) in waves of ~5–10 until n ≈ 10–30 accepted.

---

## 3. Onboarding flow (day 0)

1. **Email + invite code** → account created with email as the only PII held (§5).
2. **Guided first run (the activation event, §8.5):** the builder opens pre-loaded with one example pattern (e.g. a same-sector pair) at the free scope (top-1,000, 5-yr, labeled). The user presses run; the server-side gauntlet executes (S0–S4 at registered scope, S5 queued automatically — never opt-in, §12.6) and streams the ceremony (flicker → amber control segment → verdict) over SSE. **Activation = first completed mining run** (§8.5).
3. **Honest-result framing is shown regardless of outcome:** if the example pattern is killed, the copy says so plainly ("killed at stage Sx — this is the product working"). The kill-rate badge (§6.1) is offered opt-in on the profile.
4. **Week-1 cadence:** 3 suggested runs (their own hypotheses encouraged — hypothesis space sampling, §5.3), one email digest (free tier) summarizing their trials and the registry's cleared space. No pushy lifecycle emails; max 1 email/week in beta.
5. **Support:** founder answers in the community channels (C1–C6); a shared "#beta" channel in our own Discord for the invitees.
6. **ToS acceptance is part of account creation** (day one, not later): "not an investment / no resale value / no right to profit", non-transferable cards, no paid randomness (§10). The signup form carries the mandatory checkbox (the endpoint refuses without `tos_accepted`) and the plain onboarding statement ("SIFT shows which relationships in your research held up under stress. It never tells you what to buy or sell; nothing here is advice. You make the calls — we make the evidence legible.") shown at signup and on the first card; the same copy is served by `GET /api/onboarding` and built only in `sift/cards/copy.py` (banned-phrase fail-closed).
7. **Newcomer-friendly card face (v4.1 UX layer, `sift/cards/copy.py` + `face.py` + `render.py`):** every card face carries a one-paragraph plain-language explainer ("this card certifies that a co-movement relationship survived the pre-registered stress test — nothing about what to do with it"), jargon (z-score, half-life, DSR, tier, health) behind glossary tooltips, a gauge-style magnitude reading (stretched / normal / compressed from |z| only, numeric |z| one click away), and health in plain ratio language vs the mint baseline ("weakening ~0.8x"). All strings are built only in the copy builders and grep-tested against banned phrases (no buy/sell/long/short, no P&L/returns, no value/profit language, no urgency) — any banned phrase refuses the render. On a hit, the in-app hint frames the loop as: check the pair in your OWN charting/broker tool; the notification means "your saved connection fired, here's how strong and how often it has fired" — never an instruction to trade. Artifacts stay watermarked + shareable because they carry no directional content (no tickers, no sign — the face still reads ledger stage rows only, never params_json).

---

## 4. Definition of "active" at D30/D60/D90 — mechanics unchanged, read changed

**Mechanics (unchanged):** ≥1 registered trial run in the trailing 30 days counts as active ("active at day D" = at least one trial registered in (D−30, D]). Count **new unique `trial_id`s** per user per 30 days — a cached identical config reuses the trial ID and does not count (skip-list), so "active" cannot be farmed by re-running the same pattern. Justification for the trial-registration definition (behavior event, matches the sourced baselines' event classes [S17][S4], robust to the honesty-churn mechanism) is unchanged from the 2026-11-15 plan.

**Read (changed for small n):** report active_D as a **count k of n** with its Wilson 95% CI printed alongside, plus a qualitative band — "majority retained" (k ≥ ⌈n/2⌉), "minority retained" (k < n/3). **No count pass line (the old "≥20" is retired).** At n ≈ 10–30 a 20-point retention difference is inside the CI noise floor (widths ~29–53pp, computed in §0); the honest read is directional, not a significance test.

**Secondary (reported, not gated) metrics:** logins/30d, saved connections count, discovery cards shared, % with ≥1 survivor (amber or green) — all from the ledger and account events, no PII.

---

## 5. Measurement instrumentation — computing the gate without PII beyond email

All events land in one append-only event store keyed by an opaque `user_id` (random UUID); the only join to identity is `user_id → email` in the auth DB, and the analytics export never includes email — it ships `user_id`, event, timestamp. Beta membership is a boolean column `beta=true` on the user row; the D30/60/90 report is computed by SQL over events, not by reading anyone's email.

**Events needed (minimal set):**

| Event | Source | Feeds |
|---|---|---|
| `beta_signup` (waitlist submit, consent=true) | landing page form | denominator; screen answers (stored hashed/aggregated, not per-user in reports) |
| `beta_invited` / `beta_accepted` (invite code) | founder invite flow | invite-only provenance; wave tracking |
| `account_created` | auth service | day-0 anchor per user |
| `trial_registered` {trial_id, user_id, scope_record, provenance=user_initiated, config_hash, registered_at} | registry (already exists server-side, FL pattern) | **the "active" metric**; §5.3 provenance split |
| `trial_verdict` {trial_id, verdict, survivor_class, stage} | registry | secondary metrics, kill-rate badge |
| `connection_saved` {connection_id} | app | aha metric (§8.5) |
| `card_minted` / `card_discarded` / `ghost_promoted` / `share_grant` (card-layer events, flag `cards_v4_1` ON) | card layer | §7 secondary metrics |
| `session_login` | auth | secondary login metric (not gated) |
| `email_delivered/opened/clicked` (aggregate) | email provider | onboarding health only |
| `feedback_note` (qualitative: exit interview / in-beta survey, free text) | founder interviews | the qualitative half of the beta read (§6, §8) |

**Landing page additions:** email-only waitlist form (§2.3) + consent checkbox + a privacy note stating exactly: "we store your email and an anonymous ID; product events are stored under the anonymous ID; reports are aggregate-only, ranges and qualitative findings at this small n; we never publish per-user data; you can request deletion." That is the whole PII surface: **email**.

**Registry additions:** none needed for the gate — `trial_registered` with `provenance` and `scope_record` already gives everything. One addition: `beta` flag copied onto the trial row at registration so the D30/60/90 query is a single-table scan and beta users can be compared to any later organic users.

**The gate query (pre-registered, run at beta day 30/60/90):**
```
active_D = count(distinct user_id in beta
             where exists trial_registered with registered_at in (day0+D-30, day0+D])
report active_30, active_60, active_90 as COUNT k of n, rate k/n WITH Wilson 95% CI,
       plus the qualitative band (majority / minority retained)
NO point-estimate pass line; the decision rule in §6 uses bands + qualitative criteria
```
Pre-registered analysis plan is frozen (hash) before day 30 — same discipline as the §5.3 provenance comparison in the FL registry spec §4.3.

**Small-n reporting rules (binding for every beta report, internal or public):**
1. Every rate ships with its Wilson 95% CI and the raw k/n. Never a bare percentage.
2. Subgroup splits (card-holders vs non-holders, sharers vs non, anchored vs unanchored) have tiny cells at n≈10–30 (often 3 vs 7). Report as **counts + qualitative description only** — no significance tests, no "X% vs Y%" headlines, no p-values on cells < 5.
3. Findings are stated as **ranges and qualitative patterns** ("a majority of card-holders were still active at D90; non-holders mostly churned after the first kill") — not point estimates.
4. **Anti-gaming / integrity notes:** (1) unique-trial counting defeats re-run farming via the skip-list; (2) the beta flag is set server-side at invite acceptance, not self-reported; (3) no peeking at interim results for marketing — the FL spec's no-peeking rule (§4.2) applies; (4) any non-invited organic user who appears later is tracked in the same schema for a beta-vs-organic qualitative comparison.

---

## 6. Gate-1 read and the honest kill criterion (rewritten for small n)

### 6.1 Read (qualitative bands, pre-registered)

- **PASS-in-substance:** a **majority of the beta (k ≥ ⌈n/2⌉) still active at D90**, *and* the empirical survival read sits **at or above the sourced market band** (D30 ≈ 87–91%, D60 ≈ 75–83%, D90 ≈ 65–76% — ASIC/NBER fit, from the 2026-11-15 plan §7) when read qualitatively. At n ≤ 30 we **cannot statistically resolve** a 10–20pp difference from that band (CI widths ~29–53pp); we can only detect gross divergence. That limit is stated, not hidden.
- **KILL-in-substance:** **fewer than a third of the beta (k < n/3) active at D90**, or a D30 leading-indicator failure (§6.2). This means even invited, screened, honesty-motivated quants churn at or above the market baseline — the honesty-churn hypothesis did not move behavior and the market is not there (v3 §2.4 "both no → the market is not there").
- **In-between (a third to a majority retained at D90):** inconclusive at this n — do not scale on it; extend the beta (another wave of invites, same protocol) or run post-launch analytics before reading it as a pass. **Never** read an in-between result as a pass to justify scaling.

### 6.2 Leading-indicator kill at D30 (early stop-loss)

If **fewer than half of the beta are active at D30** (k < ⌈n/2⌉ at D30), pause invites and diagnose (exit interviews with the churned — n is small but the interviews are qualitative gold): onboarding/positioning problem or product-market-fit problem; fix, then restart the clock with a fresh wave. Rationale: sourced baselines put D30 at ~87–93% for trading accounts [S17][S18]; a majority gone inside the first month is a gross-failure signal we *can* detect even at n ≈ 10–30.

**What a kill does NOT mean:** it does not mean the engine or registry is broken — the §10 S5/ρ metrics answer that separately. Gate 1 is purely about the market.

**What a pass does NOT mean:** pass ≠ venture-scale; v3 §16 caps the ambition at solo-business scope regardless. And a beta pass is a *qualitative* engagement read on our own product — it is **not** evidence about markets, regulators, or users at large (§10).

---

## 7. Secondary metrics — recomputed against the beta population (n ≈ 10–30)

Kept from the v4.1 spec §7 secondary-metrics list and the v4 scope §2.4a (G1-a…G1-e), recomputed against the beta population; the small-n rules in §5 apply to every row:

| # | Metric (beta population, flag `cards_v4_1` ON) | What it tests | Read at small n |
|---|---|---|---|
| G1-a | **D90 retention split: card-holders vs non-card-holders** (users who minted ≥1 card vs never) | Does the card/vault layer improve or hurt the core honesty-churn gate? (spec §8 Q1) | Counts + qualitative only; cells often 3 vs 7 — **no significance claims**. Direction: if card-holders clearly retain worse, the layer is predatory in effect — kill or redesign the flag (`SIFT_CARDS_V4_1=0`), not the gate. |
| G1-b | **Ghost ratio** (ghosts ÷ true cards, per canonical spec and cohort-wide) + FIFO promotion events | Does utility-without-ownership dilute the discovery signal; does queue promotion drive engagement? (spec §8 Q2) | Counts + range; watch the pattern, not a point estimate. |
| G1-c | **Anchored-vs-unanchored survivor rates**, scope-stratified by §12.11 scope records (anchor, subspace, N_eff) | Does anchored mode actually improve hit rates or just feel smarter? (spec §8 Q7) | Per-stratum counts; at this n report "anchored ≥ unanchored within stratum" as a directional read, never a p-value. If anchored ≤ unanchored across strata, the mode is placebo — re-scope or cut. |
| G1-d | **Share-grant → D30 retention** (verified-share recipients vs non-sharers, abuse-filtered grants only, anti-abuse stack) | Does share-for-capacity convert to real engagement, not farmed slots? | With n ≈ 10–30, likely 0–3 sharers — report qualitatively via interviews; abuse sims cover the anti-abuse side (§9). |
| G1-e | **Cards minted per user; vault turnover** (keep/discard split, archive vs aggregate pool); **re-test activity** (peer re-tests per card) | Is vault pressure engaging or annoying; is the re-test loop alive? (spec §7 secondary metrics) | Median cards/user as a range ("most invitees minted 0–2 cards"); discard-rate pattern; re-test counts. Kill-switch criterion: very high discard-to-archive + G1-a negative → vault pressure is noise, flag off by default. |

**Kill criterion for the card layer (not the product):** if G1-a reads negative (card-holders clearly retain worse) or G1-c reads placebo (anchored ≤ unanchored), the `cards_v4_1` flag defaults OFF in the next testing build (`SIFT_CARDS_V4_1=0`) while the v3 gates run unchanged — the card layer is measured by the beta, it does not gate the product.

---

## 8. Spec §8 open questions — beta-measured / deferred / dropped

Every v4.1 spec §8 open question (quoted in hermes-tasks #121) gets a status. **Beta-measured** = answered from beta telemetry + qualitative feedback (interviews/survey), reported per the small-n rules in §5. **Deferred** = post-launch analytics/ops. **Dropped** = not a beta question at all.

| Spec §8 question | Status | Rationale |
|---|---|---|
| Q1 — Does vault pressure improve or hurt D90 retention? | **beta-measured** (G1-a) | Card-holder vs non-holder D90 split, qualitative at small n. |
| Q2 — Ghost ratio: does utility-without-ownership dilute the discovery signal; does FIFO promotion drive engagement? | **beta-measured** (G1-b) | Ghost-ratio counts + promotion events + qualitative feedback. |
| Q3 — Card art pipeline cost/quality; who reviews? | **deferred** | Operational budget line, not a user-behavior question; decide with post-launch ops data. |
| Q4 — Should ghost cards show "discovered by someone else" (social proof) without naming? | **beta-measured** (qualitative) | Default yes; confirm via in-beta feedback interviews. |
| Q5 — Re-evaluation cadence for card-tier changes (monthly registry pass)? | **deferred** | Engine-side cadence decision; set from post-launch decay-report load, not beta telemetry. Tiers are now **bidirectional** (owner decision 2026-11-17): amber downgrades any time from registry evidence; green upgrades ONLY via pre-registered live-window rules — card age > 12m (fully out-of-sample trailing window), two consecutive annual windows at the mint-time bar (silver→gold: DSR >= 0.99 AND |t_net| >= 4.0 AND p_perm <= 0.01; bronze→silver: DSR >= 0.95 AND |t_net| >= 3.0), max one tier change per card per 90 days, seeded PCG64, append-only `card_upgrade` ledger event with evidence, publicly explained in rarity language only — never by payment or chance. Mint-time `--holdout-bars 252` on `sift.engine.first_run` locks the last 252 trading days out of S0–S4 discovery/walk-forward and evaluates survivors once on that frozen holdout (the mint-time live-baseline); the holdout is never used for tuning. |
| Q7 — Does anchored mode actually improve hit rates, or just feel smarter? | **beta-measured** (G1-c) | Scope-stratified anchored vs unanchored survivor rates; directional read only at this n. |
| Q8 — Anchor-leak check: card faces must never show the anchor ticker to non-owners | **dropped from beta telemetry** | A security/code-review gate before launch (audit of the anonymized-face render path), not a beta question; not answerable by telemetry. |
| WTP / paid conversion (the old 50–100 cohort's core screen) | **deferred** | No money changes hands in the beta, so WTP is unmeasurable here; measured post-launch in paid-conversion analytics. |
| "Do simulated users behave like real users?" | **dropped as a question** | Simulated users are infra validation only (§9); the question itself is circular and must not appear as a claim in any doc. |

---

## 9. Simulated users — infrastructure validation ONLY

The simulated-user harness (hermes-tasks #144) exists to soak **load** (concurrent mining runs, queue depth, vault pressure, DB contention; p50/p95 latency) and to attack the **anti-abuse controls** (device/IP/ASN clusters, day-zero share farms, unverifiable posts, duplicate shares, velocity limits, +5 cap, revocation, fraud-queue SLA, append-only `share_grant` log integrity). It runs against a test DB / ephemeral env only (`SIFT_ENV=test`), synthetic instruments only, behind `cards_v4_1` fail-closed with `SIFT_CARDS_V4_1=0` respected, and every report it emits carries the header **"SIMULATED — infrastructure validation, not user-behavior evidence."**

**Binding rule for all SIFT docs and reports:** simulated users are **never evidence** for any spec §8 user-behavior question and never evidence for gate-1 retention/engagement reads. Any doc that claims "the sim proves users will …" is circular (the sim was built to behave) and must be corrected. Beta telemetry from real invited users is the only behavioral evidence the beta produces.

---

## 10. Legal posture (unchanged) — and what beta telemetry does NOT do

Unchanged from v4.1 spec §6 and v3 §12: **invite-only beta, no public offering, no money changes hands**; cards **non-transferable**, no marketplace, no sale, no royalties; **no paid randomness**, no lotteries/loot boxes, no chance-based rewards; ToS "not an investment / no resale value / no right to profit" **live from day one**; no profit-expectation or "value" language anywhere (UI copy, emails, notifications included — rarity language only); notifications opt-in, rate-capped; signals-only, no execution.

**Explicit note (decision 2026-11-17, hermes-tasks #143):** beta telemetry **does not substitute for the FCA/IOSCO-evidence-based reasoning** in the v4.1 spec §6 legal posture. The regulatory design (non-transferable cards, no paid randomness, no profit-expectation marketing) rests on the cited regulatory evidence (SEC/CFTC 2026-03-17 interpretive release on digital collectibles; FCA/IOSCO findings that engagement hooks pushing *trading* harm users), **not** on beta numbers. Beta telemetry only measures **our own product's engagement** with a small invited group — it says nothing about how any other population would behave, and no doc may present it as such.

---

## 11. What this plan deliberately does not do

- No paid acquisition, no affiliates, no guru marketing (§9.8) — the channels in §2.3 are all founder-time organic, ending at an email-only waitlist.
- **No paid-creator spots in beta (v4.2 §5 / §9.8):** no paid acquisition channels at all, and specifically no "how I trade using SIFT" paid spots, sponsored trade-along posts, or creator partnerships that show SIFT output alongside a trading decision. SIFT content may describe evidence and survival facts only; any creator material must carry the same no-direction/no-P&L/no-value-language discipline as the product copy (banned-phrase fail-closed), and nothing in it may present a card or deck as a win, a pick, or a "top deck". This holds for the card layer (v4.1) and the deck layer (v4.2) alike.
- No public signup / open beta during this phase — invited-only.
- No money changes hands — no payments enabled for beta users at all.
- No PII beyond email; no per-user reporting outside the operator; aggregate-only, ranges-and-qualitative-only in any public mention of beta results (§5 rules; §12 and the privacy consent).
- No raw data to clients at any point (§12.2) — the beta runs on the same server-side architecture as launch.
- No S5 opt-in for anyone, beta included (§12.6); beta users get Pro-tier trial *budget* (scope/priority), never validation exemptions.
- No simulated-user results presented as user-behavior evidence (§9).
- No contact with funds/vendors from this plan — gate 2 (3 paying fund pilots) is a separate task with its own public-registry targeting criteria (sift-research#2 §3).
- No peeking at interim retention for marketing; pre-registered read at D30/60/90 only.

---

## 12. Sources

- [S1] QuantConnect pricing — Researcher $84/mo, Team $168/user (min 2), Trading Firm $480/user (min 2): https://www.quantconnect.com/pricing.md (accessed 2026-11-15)
- [S2] Trade Ideas pricing — TI Standard $127/mo, Premium $254/mo: https://www.trade-ideas.com/pricing/ (accessed 2026-11-15)
- [S3] Quantpedia pricing/FAQ — Prime $399/12mo … Pro $799/12mo (3-mo $449–$599): https://quantpedia.com/premium-pricing , https://quantpedia.com/how-it-works/faq (accessed 2026-11-15)
- [S4] QuantConnect homepage — 50k monthly users create 2,500 algos/month, 565K community, $100B/mo notional, 375K live strategies: https://www.quantconnect.com/?lang=en (accessed 2026-11-15)
- [S5] QuantConnect research page — "564,800 quants": https://www.quantconnect.com/research (accessed 2026-11-15)
- [S6] disdex listing of QuantConnect Discord — `discord.gg/quantconnect`, ~5,803 members: https://disdex.io/server/846109296973840424 (accessed 2026-11-15); official link https://www.quantconnect.com/discord
- [S7] r/algotrading ~1.9M members (aggregated third-party estimate, Senpi): https://senpi.ai/resources/blog/senpi-cursor-for-trading (accessed 2026-11-15); weekly discussion threads live: https://www.reddit.com/r/algotrading/comments/1xsk1p5/ (2026-11-10)
- [S8] X profiles: @lopezdeprado (24.9k followers), @echanQT (3.65k): https://x.com/lopezdeprado , https://x.com/echanqt (accessed 2026-11-15)
- [S9] Quant researcher handle list (Mar 2026 compile): @aclenow, @alphaarchitect, @choffstein, @QuantEdges, @MebFaber, @GaryAntonacci, @investingidiocy et al.: https://gist.github.com/videv93/71a3ef36205bc1a0102f77506599af34 (accessed 2026-11-15)
- [S10] Numerai Discord `discord.gg/RD9uhjnJHP`, ~9.3k members (discord listing + forum RSS): https://discord.com/servers/numerai-894652647515226152 , https://forum.numer.ai/t/the-tournament-roadmap/8377 (accessed 2026-11-15)
- [S11] Algovibes pairs-trading backtest video (survivorship-bias documentation): https://www.youtube.com/watch?v=qOsjfj5z_58 (accessed 2026-11-15); channel list incl. QuantPy, Part Time Larry, Kevin Davey, QuantInsti (~30–50k subs): https://algotradingmap.com/learning-center (accessed 2026-11-15)
- [S12] paperswithbacktest community-curated quant video list (412 videos, IB API tutorial by Part Time Larry, Quantopian pairs talk): https://paperswithbacktest.com/blog/quant-trading-videos-curated-by-the-community (accessed 2026-11-15)
- [S13] Hacker News — QuanterLab Show HN (quant backtesting launch, 2026-11-15): https://news.ycombinator.com/item?id=50380178 (accessed 2026-11-15)
- [S14] Wilmott forum — Trading Forum, 2,400 topics / 17,757 posts, last post Nov 7 2026: https://forum.wilmott.com/ (accessed 2026-11-15)
- [S15] Elite Trader — 120,000 members, 280k threads, 5M posts: https://www.elitetrader.com/et/ (accessed 2026-11-15)
- [S16] Kaggle Jane Street Real-Time Market Data Forecasting — 25,446 entrants / 3,643 participants / 3,757 teams: https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting (accessed 2026-11-15)
- [S17] ASIC REP 828 (20 Jan 2026) — quarterly retention of new CFD clients 100/65/45/33%; 67% stopped by year end: https://download.asic.gov.au/media/tq0he35c/rep828-published-20-january-2026.pdf (accessed 2026-11-15)
- [S18] Ben-David, Birru & Prokopenya, NBER w22146 — avg trader lifetime 181 days; 25% leave within 46 days; 50% exit within 155 days: https://www.nber.org/system/files/working_papers/w22146/w22146.pdf (accessed 2026-11-15)
- [S19] Recurly churn benchmarks July 2026 — median monthly churn 3.84% for $25–50/mo price tier (3.15% for $50–100): https://statisticsglobal.org/churn-rate-benchmarks.md (accessed 2026-11-15)
- [S20] ChartMogul 2025 — median NRR 82% B2B SaaS vs 49% B2C SaaS: https://statisticsglobal.org/churn-rate-benchmarks.md (accessed 2026-11-15)
- Internal: `sift-research` branch `docs/sift-scope-v3` (PR pull/3) — SIFT-Product-Scope-v3-Research-Updated.md §2.1–2.4, §5, §6, §8.5, §9.3–9.8, §10, §11, §12, §14#4, §16; `sift-research` branch `docs/sift-scope-v4` (PR pull/7) §2.4/§2.4a (same rewrite mirrored there); v4.1 "middle ground" spec quoted in hermes-tasks #121 (esp. §6 legal posture, §7 secondary metrics, §8 open questions); user decisions 2026-11-17 via hermes-tasks #143 (this rewrite) and #144 (sim harness = infra validation only); `docs/market-sizing-verification.md` (42-source verification ledger); FL `research/design/2026-11-15-sift-gauntlet-s5-design.md` and `research/notes/2026-11-15-006-...-design-spec.md` (trial ledger schema, skip-list semantics, pre-registration/no-peeking protocol). Wilson CI widths quoted in §0/§4 computed 2026-11-17 (binomial Wilson score interval, z=1.96). Original cohort plan: sift pull/1–pull/3 (task t_56db5ac0, 2026-11-15).
