# SIFT — Beta Cohort Plan & Gate-1 Measurement Definition

**Status:** plan / spec (research + design; no code, no outreach executed)
**Date:** 2026-11-15 (all external sources accessed 2026-11-15 UTC)
**Task:** t_56db5ac0 (SIFT gate-1 prep 8/8)
**Spec source:** `SIFT-Product-Scope-v3-Research-Updated.md` on branch `docs/sift-scope-v3` of `jonathanbarakat-crypto/sift-research` (PR pull/3) — esp. §2.4 gate 1, §8.5 funnel, §10 metrics, §12 non-negotiables, §14 open decision #4. Supporting: `sift-research` `docs/market-sizing-verification.md` (42 cited sources), `docs/assessment.md`; FL `research/design/2026-11-15-sift-gauntlet-s5-design.md` and `research/notes/2026-11-15-006-sift-registry-discovery-credit-survivor-classes-design-spec.md` (trial-ledger schema: `trial_id`, `scope_record`, `provenance`, `config_hash`, `data_vintage_hash`, timestamps).
**Hard rules honored throughout (v3 §12):** server-side mining only; no raw data to clients; S5 never opt-in; scope-stratified denominators; public cards show survival facts only; "SIFT does not trade." This plan adds: no PII beyond email in the cohort measurement; no paid acquisition channels (§9.8).

---

## 1. Goal

Answer §2.4 gate 1 honestly before scaling: **"Can 20 serious quants complete a mining run and remain active at 90 days?"** This is the direct test of the honesty-churn hypothesis (§2.2, §16): the product's most common answer is "no," and the hypothesis is that users who accept that answer retain, while hype-driven users churn the day their first pattern dies.

---

## 2. Cohort: size, target, screening

### 2.1 Size and why

**Target: 50–100 recruits** (v3 §14 open decision #4, source sift-research#2). Rationale, made quantitative:

- The gate is a count: **≥20 of the cohort still active at day 90**. In retention-rate terms that is 40% at n=50, 27% at n=75, 20% at n=100 — with Wilson 95% CIs of roughly [28%, 54%], [18%, 38%], [13%, 29%] respectively (computed from the binomial, script output quoted inline).
- The sourced attrition baseline (§4) says the *market* keeps ~65% of new trading-account clients at ~90 days (ASIC REP 828 quarterly curve 100→65→45→33%) and ~50% of forex traders at 155 days (NBER w22146). So **D90 retention ≥40% is already better-than-market**; the gate at n=50 (≥40%) is set at the better-than-market line, and at n=100 (≥20%) it is a permissive floor. Recommendation: **recruit toward the top of the range (n=75–100)** so the gate reads ≥20–27% D90 retention with a tighter CI, and pre-register the pass line as the raw count ≥20 (per §2.4 as written), reporting the rate with CI alongside.
- 50–100 is also the largest cohort one solo operator can onboard personally (§11) — each recruit needs a screening conversation and a first-run walkthrough; beyond ~100 the measurement itself becomes the bottleneck.

### 2.2 Who counts as a recruit (screening criteria — all answerable in a 15-min call or form)

1. **Seriousness:** ≥1 year running systematic/backtested strategies and writes code (Python/C#/Julia) — this is the "serious quant" filter from §2.1.
2. **WTP screen (the point of the cohort):** currently paying for ≥1 tool in the demonstrated bands — QuantConnect Researcher $84/mo [S1], Trade Ideas $127–254/mo [S2], Quantpedia $399–799/yr [S3], or equivalent (Backtrader/vectorbt Pro, TradingView paid, data subscriptions ≥$50/mo). Self-attested on the form; no receipts required (keeps PII minimal), but the question is asked verbatim.
3. **Motivation check:** asks about *validation/honesty* features, not "find me alpha." Anyone whose stated goal is passive income or signals-for-sale is a polite no — they are the negative-LTV segment §2.1 explicitly excludes.
4. **Consent:** agrees to the measurement protocol in §5 (events listed, email as the only identifier, aggregate-only reporting).

Explicitly **not** screened for: AUM, geography beyond English-language, platform preference. US daily equities focus means we do not exclude non-US residents, but onboarding copy is English-only in v1 (§2.1a SAM definition).

### 2.3 Recruitment channels — concrete, with links and handles (verified 2026-11-15)

All organic/community-led per §9.3–9.4 and §9.8 (no paid social, no affiliates). Expected yield per channel is an estimate, labeled as such.

| # | Channel | Concrete link / handle | Why it reaches the right people | Est. yield into 50–100 |
|---|---|---|---|---|
| C1 | QuantConnect community forum | https://www.quantconnect.com/forum/ (posting requires 30% Boot Camp — post as the founder account after completing it, or answer questions where posting is open) | 565K registered / ~50k MAU [S4][S5]; the densest concentration of exactly this user (LEAN users already backtesting) | 15–25 |
| C2 | QuantConnect Discord | https://www.quantconnect.com/discord (vanity `discord.gg/quantconnect`; ~5.8k members per disdex listing [S6]) | Real-time; ideal for the "first mining run" walkthroughs | 5–10 |
| C3 | r/algotrading | https://www.reddit.com/r/algotrading/ (~1.9M members [S7]); weekly discussion threads are the sanctioned self-promo venue; a dedicated post must add value (e.g. "we ran X through pre-registered gauntlet — here's the kill") | Densest serious-retail quant subreddit; skeptical audience that converts if answered honestly (§9.3) | 10–20 |
| C4 | r/quant + r/systematictrading | https://www.reddit.com/r/quant/ , https://www.reddit.com/r/systematictrading/ | Smaller, more academic/systematic; good for the WTP-screened recruits | 3–8 |
| C5 | X quant community ("stats corner of fintwit") | Handles: @lopezdeprado (24.9k, LdP), @echanQT (Ernie Chan, 3.7k), @aclenow, @alphaarchitect, @choffstein, @QuantEdges, @MebFaber, @GaryAntonacci, @investingidiocy (Rob Carver) [S8][S9]; search terms: #algotrading #quant #backtesting #stat arb | The stats-heavy corner where the honesty positioning lands; quote-post threads about overfitting/backtesting honesty | 5–10 |
| C6 | Discord servers (discoverable directory, searched 2026-11-15) | "Quant Talk" (~3.0k members, algo/quant/data-science discussions), "Quant Trading App" (~11k members, quant-as-edge), Numerai Discord `discord.gg/RD9uhjnJHP` (~9.3k members [S10]) + forum https://forum.numer.ai/ | Numerai community is the closest analog population: self-directed data-science market modelers, many already pay for tools/data | 5–10 |
| C7 | YouTube stat-arb / backtesting educators (partnership per §9.4) | Algovibes (~30k subs; "I backtested a pairs trading strategy… does it work" — documents simplifications incl. survivorship bias), QuantPy (Python quant implementation), Part Time Larry (IB API tutorials), Kevin Davey "Algo Trading With Kevin Davey" (~50k subs, backtesting methodology/overfitting), QuantInsti (~50k) [S11][S12] | "We tested this channel's pairs tutorial through pre-registered OOS checks" is the §9.4 asset; audience is already building systems | 5–10 |
| C8 | Hacker News Show HN + r/algotrading launch thread | https://news.ycombinator.com/show (precedent: QuanterLab Show HN 2026-11-15, quant backtesting tools are a proven HN category [S13]) | Technical credibility; the skeptics in the comments are the filter (§9.3) | 5–10 |
| C9 | Wilmott forum + Elite Trader | https://forum.wilmott.com/ (Trading Forum: 2,400 topics/17.7k posts, active Nov 2026 [S14]); https://www.elitetrader.com/ (120k members [S15]) | Professional/semi-pro quants; lower volume but highest seriousness; also the fund-pilot hunting ground (§2.4 gate 2) | 2–5 |
| C10 | Kaggle competition communities (organic mention) | Jane Street Real-Time Market Data Forecasting: 25.4k entrants / 3.6k participants [S16]; forum posts pointing to "how to validate without overfitting" content | Data-science traders; WTP-adjacent (they spend on compute/data) | 2–5 |

**Sequencing:** C1–C4 first (highest density, lowest cost), C5/C6 in parallel for the honesty-positioning content, C7–C10 at/after public beta open. Do **not** run C1–C4 as ads or affiliate posts — §9.8 prohibits; all posts are founder-written, value-first, and answer skeptics in-thread.

**Recruit mechanics:** one landing page with (a) the §9.2 interactive teaser (synthetic data, no signup), (b) the free deflated-Sharpe/PBO calculator as lead magnet, (c) a **beta application form** (email + the §2.2 screening questions + consent checkbox). Invite codes for the free tier per §14 #4 — the beta cohort gets Pro-tier trial budgets for the 90 days (so the gate tests product retention, not free-tier quota frustration), disclosed in the invite.

---

## 3. Onboarding flow (day 0)

1. **Email + invite code** → account created with email as the only PII held (§5).
2. **Guided first run (the activation event, §8.5):** the builder opens pre-loaded with one example pattern (e.g. a same-sector pair) at the free scope (top-1,000, 5-yr, labeled). The user presses run; the server-side gauntlet executes (S0–S4 at registered scope, S5 queued automatically — never opt-in, §12.6) and streams the ceremony (flicker → amber control segment → verdict) over SSE. **Activation = first completed mining run** (§8.5).
3. **Honest-result framing is shown regardless of outcome:** if the example pattern is killed, the copy says so plainly ("killed at stage Sx — this is the product working"). The kill-rate badge (§6.1) is offered opt-in on the profile.
4. **Week-1 cadence:** 3 suggested runs (their own hypotheses encouraged — hypothesis space sampling, §5.3), one email digest (free tier) summarizing their trials and the registry's cleared space. No pushy lifecycle emails; max 1 email/week in beta.
5. **Support:** founder answers in the community channels (C1–C6); a shared "#beta" channel in our own Discord for the cohort.

---

## 4. Definition of "active" at D30/D60/D90 — decision and justification

**Decision: ≥1 registered trial run in the trailing 30 days counts as active** (i.e. "active at day D" = at least one trial registered in (D−30, D]). The alternative (≥3 logins in 30 days) is rejected.

**Justification:**
- It is the metric the product actually cares about and can measure without PII: the server-side registry mints a `trial_id` per run (§3.1, §5.1 — registration is the only path to a run, so N is exact by construction). A login is measurable too, but it measures *visits*, not *value events*; a user who logs in three times to look at a dead card is not retained, and a user who runs one new trial a month is. §8.5 defines activation as first completed mining run and the funnel's retention row as "registry's cleared space + on-watch evolution" — trial registration is the same event class.
- It is the definition the sourced baseline uses: ASIC's attrition stat is "stopped *trading*" and QC's MAU is "created algorithms" — behavior events, not logins [S17][S4]. Comparing a login metric to those baselines would flatter us.
- It is robust to the honesty-churn mechanism: the hypothesis says hype users churn when their pattern dies; a user whose pattern was killed but who registers another trial is exhibiting exactly the behavior the hypothesis predicts retains. Login-count would miss that (they may never "log in" again after the kill — they just run the next one via API/SSE stream).
- Guard against gaming: trial budgets cap spam by construction (§6.2 rule 4) — a cached identical config reuses the trial ID and does not count as a new registration (§1.5 skip-list), so "active" cannot be farmed by re-running the same pattern. Count **new unique `trial_id`s** per user per 30 days, not run-starts.

**Secondary (reported, not gated) metrics:** logins/30d, saved connections count, discovery cards shared, % with ≥1 survivor (amber or green) — all from the ledger and account events, no PII.

---

## 5. Measurement instrumentation — computing the gate without PII beyond email

All events land in one append-only event store keyed by an opaque `user_id` (random UUID); the only join to identity is `user_id → email` in the auth DB, and the analytics export never includes email — it ships `user_id`, event, timestamp. Cohort membership is a boolean column `beta_cohort=true` on the user row; the D30/60/90 report is computed by SQL over events, not by reading anyone's email.

**Events needed (minimal set):**

| Event | Source | Feeds |
|---|---|---|
| `beta_signup` (form submit, consent=true) | landing page form | cohort denominator, screen answers (stored hashed/aggregated, not per-user in reports) |
| `account_created` | auth service | day-0 anchor per user |
| `trial_registered` {trial_id, user_id, scope_record, provenance=user_initiated, config_hash, registered_at} | registry (already exists server-side, FL pattern) | **the "active" metric**; also §5.3 provenance split |
| `trial_verdict` {trial_id, verdict, survivor_class, stage} | registry | secondary metrics, kill-rate badge |
| `connection_saved` {connection_id} | app | aha metric (§8.5) |
| `card_shared` {trial_id} | share button | viral metric (§8.5) |
| `session_login` | auth | secondary login metric (not gated) |
| `email_delivered/opened/clicked` (aggregate) | email provider | onboarding health only |

**Landing page additions:** the beta form (§2.3) + consent checkbox + a privacy note stating exactly: "we store your email and an anonymous ID; product events are stored under the anonymous ID; reports are aggregate-only; we never publish per-user data; you can request deletion." That is the whole PII surface: **email**.

**Registry additions:** none needed for the gate — `trial_registered` with `provenance` and `scope_record` already gives everything. One addition: `beta_cohort` flag copied onto the trial row at registration so the D30/60/90 query is a single-table scan and cohort users can be compared to non-cohort users (selection check: did the beta screen actually select for stickier users?).

**The gate query (pre-registered, run at cohort day 30/60/90):**
```
active_D = count(distinct user_id in beta_cohort
             where exists trial_registered with registered_at in (day0+D-30, day0+D])
pass = active_90 >= 20
report active_30, active_60, active_90 as counts AND rates with Wilson 95% CI
```
Pre-registered analysis plan is frozen (hash) before day 30 — same discipline as the §5.3 provenance comparison in the FL registry spec §4.3.

**Anti-gaming / integrity notes:** (1) unique-trial counting defeats re-run farming via the skip-list; (2) the cohort flag is set server-side at invite acceptance, not self-reported; (3) no peeking at interim results for marketing — the FL spec's no-peeking rule (§4.2) applies; (4) if a cohort user invites a non-cohort user, the non-cohort user is tracked in the same schema so we can report cohort-vs-organic retention split (expected: cohort ≥ organic, which itself tests whether screening selects sticky users).

---

## 6. Gate-1 pass criterion and the honest kill criterion

### 6.1 Pass (as written in v3 §2.4)
**≥20 of the 50–100 beta cohort users still active at day 90** (active per §4). Reported as count ≥20 **and** rate with Wilson CI. At n=100 that is ≥20% D90 retention (CI [13%, 29%]); at n=50, ≥40% ([28%, 54%]). Note for the reader: the *market* baseline is ~65% at ~90 days for new trading accounts [S17], so the gate as written is a **floor, not a beat-the-market line** — a pass at n=100 (20%) would be well *below* the sourced attrition baseline, which is itself a warning sign even though the raw count passes. We therefore pre-register a **two-part read**: (a) the raw §2.4 count (pass/fail as written), and (b) the comparison to baseline — if D90 retention < ~50% (i.e. worse than NBER's 50%-at-155-days when evaluated at 90 days, ~67%), gate 1 is a **fail in substance** regardless of the raw count. This is the honest reading of §16's warning: "if retention tracks the sourced attrition baseline, the honesty-churn hypothesis fails and the market is not there."

### 6.2 Kill criterion (what result means STOP)
Stop (do not scale; per §2.4 "both no → the market is not there; no GTM plan fixes that") if **either**:
1. **active_90 < 20** (raw gate fail), **or**
2. **D90 retention < 50%** with the cohort screened per §2.2 — i.e. even WTP-screened serious quants churn at or above the market baseline, meaning the honesty-churn hypothesis (the registry/discovery-credit mitigation, §2.2) did not move retention. In that case the honest read is: the product is not worse than the market, but it is not better, and the SAM attach (6.7–30%, §2.1a) is not achievable — pessimistic scenario only ($144k/yr, §2.3).
3. A **leading-indicator kill** at day 30 for early stop-loss (do not wait 90 days to learn the answer): if **active_30 < 60% of cohort** (i.e. more than 40% gone within the first month), the honesty-churn is happening *before* any gauntlet kill can explain it — onboarding/positioning problem or product-market-fit problem; pause recruitment, diagnose (exit interviews with the churned, n small but qualitative), fix, and only then restart the clock with a fresh cohort. Rationale: the sourced baselines put D30 at ~87–93% for trading accounts [S17][S18]; anything below ~70% at D30 in a WTP-screened cohort is a red flag worth acting on, and 60% is the pre-registered hard line.

**What a kill does NOT mean:** it does not mean the engine or registry is broken — the §10 S5/ρ metrics answer that separately. Gate 1 is purely about the market.

**What a pass does NOT mean:** pass ≠ venture-scale; §16 caps the ambition at solo-business scope regardless.

---

## 7. Expected D30/60/90 retention curves — honesty-churn hypothesis right vs wrong

Model: constant daily hazard h, retention(D) = e^(−h·D) — the standard exponential survival model; baselines fit to the cited sources (computed inline, script output quoted):

- **Sourced market baseline:** ASIC REP 828 quarterly curve (100→65→45→33% at ~Q1–Q4) [S17] and NBER w22146 (50% exit by day 155, avg lifetime 181 days) [S18] both imply h ≈ 0.0045–0.0048/day → **D30 ≈ 87%, D60 ≈ 75%, D90 ≈ 65%, D180 ≈ 43%**. The two-point ASIC fit (65% at d90, 33% at d365) gives h=0.0025 → D30 93%, D60 86%, D90 80%, D365 41%. We take **D30 ~87–91%, D60 ~75–83%, D90 ~65–76%** as the market band for new trading-account-like users.
- SaaS cross-check at our price point: Recurly July-2026 network data puts median monthly churn at 3.84% for the $25–50/mo tier [S19] → D90 = (1−0.0384)^3 ≈ 89%; B2C SaaS NRR 49%/yr [S20]. So subscription-SaaS tools retain *better* than trading accounts at 90 days — the trading-activity baseline is the more honest comparator for a tool whose value requires ongoing use, which is why we anchor on the ASIC/NBER band, not the SaaS band.

**Scenario table (constant-hazard curves; counts for n=50 and n=100):**

| Scenario | h (per day) | D30 | D60 | D90 | D180 | active@D90 (n=50 / n=100) | Gate read |
|---|---|---|---|---|---|---|---|
| **Market baseline** (ASIC/NBER band) | 0.0045–0.0048 | 87% | 75% | 65% | 43% | 33 / 65 | PASS raw; PASS substance (better-than-baseline line is ~50%) |
| **Honesty-churn hypothesis WRONG** (registry/discovery credit works; retention *beats* market) | 0.0032 | 91% | 83% | **75%** | 56% | 38 / 75 | PASS, strong |
| **Honesty-churn hypothesis RIGHT** (honesty does not save retention; hype segment churns fast, and even screened quants churn when patterns die — at or above market) | 0.0134 | 67% | 45% | **30%** | 9% | 15 / 30 | n=100: raw PASS (30≥20) but **substance FAIL** (30% < 50% baseline line) → kill per §6.2(2); n=50: raw FAIL (15<20) → kill per §6.2(1) |
| **Severe honesty-churn** (product actively repels: kills feel like failures, not culture) | ≥0.02 | ≤55% | ≤31% | **≤17%** | ≤4% | 8 / 17 | raw FAIL → kill |

**How to read the curves live:** plot the cohort's empirical survival (Kaplan–Meier on trial-registration events) against these three bands at D30/60/90. If the curve sits in the baseline band or above → hypothesis wrong (product retains like any good tool; the honesty framing is not a retention liability) → pass. If it sits at or below the honesty-churn-RIGHT band → the registry mitigation did not move behavior → kill per §6.2. The pre-registered decision rule uses the D90 point with Wilson CI, plus the D30 leading indicator (§6.2(3)).

**Leading indicators to watch between D0 and D30** (all ledger-derived): time-to-first-run < 1 day (target ≥80% of cohort — §10); % with a saved connection by D30 (target ≥50%); median trials/user in first 30 days (target ≥3); share of cohort whose first verdict was a kill and who still registered a second trial (the direct honesty-churn probe — target ≥60%; if <40%, the kill is driving churn → hypothesis right).

---

## 8. What this plan deliberately does not do

- No paid acquisition, no affiliates, no guru marketing (§9.8) — the channels in §2.3 are all founder-time organic.
- No PII beyond email; no per-user reporting outside the operator; aggregate-only in any public mention of beta results (consistent with §12 and the privacy consent).
- No raw data to clients at any point (§12.2) — the beta runs on the same server-side architecture as launch.
- No S5 opt-in for anyone, beta included (§12.6); beta users get Pro-tier trial *budget* (scope/priority), never validation exemptions.
- No contact with funds/vendors from this plan — gate 2 (3 paying fund pilots) is a separate task with its own public-registry targeting criteria (sift-research#2 §3).
- No peeking at interim retention for marketing; pre-registered read at D30/60/90 only.

---

## 9. Sources

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
- [S14] Wilmott forum — Trading Forum 2,400 topics / 17,757 posts, last post Nov 7 2026: https://forum.wilmott.com/ (accessed 2026-11-15)
- [S15] Elite Trader — 120,000 members, 280k threads, 5M posts: https://www.elitetrader.com/et/ (accessed 2026-11-15)
- [S16] Kaggle Jane Street Real-Time Market Data Forecasting — 25,446 entrants / 3,643 participants / 3,757 teams: https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting (accessed 2026-11-15)
- [S17] ASIC REP 828 (20 Jan 2026) — quarterly retention of new CFD clients 100/65/45/33%; 67% stopped by year end: https://download.asic.gov.au/media/tq0he35c/rep828-published-20-january-2026.pdf (accessed 2026-11-15)
- [S18] Ben-David, Birru & Prokopenya, NBER w22146 — avg trader lifetime 181 days; 25% leave within 46 days; 50% exit within 155 days: https://www.nber.org/system/files/working_papers/w22146/w22146.pdf (accessed 2026-11-15)
- [S19] Recurly churn benchmarks July 2026 — median monthly churn 3.84% for $25–50/mo price tier (3.15% for $50–100): https://statisticsglobal.org/churn-rate-benchmarks.md (accessed 2026-11-15)
- [S20] ChartMogul 2025 — median NRR 82% B2B SaaS vs 49% B2C SaaS: https://statisticsglobal.org/churn-rate-benchmarks.md (accessed 2026-11-15)
- Internal: `sift-research` branch `docs/sift-scope-v3` — SIFT-Product-Scope-v3-Research-Updated.md §2.1–2.4, §5, §6, §8.5, §9.3–9.8, §10, §11, §12, §14#4, §16; `docs/market-sizing-verification.md` (42-source verification ledger); FL `research/design/2026-11-15-sift-gauntlet-s5-design.md` and `research/notes/2026-11-15-006-...-design-spec.md` (trial ledger schema, skip-list semantics, pre-registration/no-peeking protocol).
