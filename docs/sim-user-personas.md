# Simulated-User Personas — SIFT Test-Bot Harness

> **SIMULATED — infrastructure validation, not user-behavior evidence.**
>
> Everything on this page describes **synthetic personas used to drive the
> bot harness in `sift/sim/`** (pull/27, follow-up issue #144). These personas
> exist ONLY to build deterministic user bots for infrastructure soak and
> v4.1 §4.7 anti-abuse testing. They are **NOT evidence of real user
> behavior**: the mix, shares, and every parameter below are a **design
> choice for behavior-space coverage, not a market estimate**, and must never
> be cited as user-behavior evidence in any doc, report, or metric. Spec §8
> human-response questions are answered from the **limited alpha/beta**
> telemetry plan (pull/25), never from sims.

**Machine-readable spec:** `sift/sim/personas.json` (72 personas, schema
validated by `sift/sim/personas.py::load_personas` and
`tests/sim/test_personas.py`).
**Bot driver:** `sift/sim/bots.py` (`run_persona_session` / `run_persona_sim`).

---

## 1. Why personas exist at all

The soak needs *varied* synthetic traffic to measure queueing, DB contention,
vault pressure, and to give the §4.7 controls realistic attack targets. A
single anonymous load shape cannot do that, so each bot is parameterized by a
persona: how it searches, how much it mints, how it shares, and whether it is
a deliberate abuse pattern. Every bot run is **deterministic**
(`world seed ⊕ persona.deterministic_seed`, PCG64 + fixed clock — same seed
→ same session log) and **CI-smoke sized** (a handful of runs per session).
Every session record and report logs `persona_id` + seed and carries the
SIMULATED banner (`sift/sim/header.py`).

## 2. Hard constraints baked into the spec

- **Synthetic instruments only.** Bots pick from the sim world's `SIM####`
  universe. Personas never name real tickers — bot inputs stay internal,
  never in product copy (v3 §12: no tickers + directional calls in anything
  SIFT ships).
- **No profit-expectation/"value" language anywhere**, including
  bot-generated share posts. Share posts are **non-directional test
  fixtures**: verifiable ones are just the anonymized card link +
  `#SIFT` attribution tag; unverifiable ones are garbage that must fail
  live-post verification.
- **Raw market data stays server-side.** Bots only ever read the same
  anonymized aggregates a real card-holder would (cards, vault, share state).
- **Test env only.** `SIFT_ENV=test` is hard-required (the harness refuses
  otherwise); ephemeral test DB only; card-layer paths behind `cards_v4_1`,
  fail-closed when off, kill-switch `SIFT_CARDS_V4_1=0` respected.
- Every harness output carries the header
  `SIMULATED - infrastructure validation, not user-behavior evidence`.

## 3. Schema (`personas.json`, one entry per persona)

| key | type / range | meaning (sim knob, not a claim about people) |
|---|---|---|
| `id` | lowercase slug, unique | stable persona id, logged on every bot event |
| `name` | string | human-readable label |
| `archetype` | one of 9 classes (§4) | coarse class |
| `sophistication` | int 1–5 | naive → sophisticated |
| `information_sources` | non-empty subset of `wsb, r/stocktips, substack_free, influencers, screeners, fundamentals, random, insider_ish` | where the bot "gets ideas" |
| `strategy` | `pairs / lead_lag / momentum / narrative / random / copycat` | connection family the bot hunts |
| `search_behavior` | `{query_patterns[], queries_per_session 1–200, session_minutes 1–600, ticks_per_session 1–50}` | session shape (ticks = synthetic instruments per session) |
| `minting_goal` | `{cards_per_week: 0–100}` | minting-intent load knob |
| `sharing_behavior` | `{posts_per_week: 0–200, verifiable: bool}` | share-post frequency + whether posts pass live verification |
| `abuse_flags` | subset of `day_zero, multi_device, ip_cluster, asn_cluster, velocity, unverifiable_shares, duplicate_shares, cap_buster, honeypot_probe, share_farm` | §4.7 attack-sim targeting knobs |
| `deterministic_seed` | int, unique per persona | bot stream = spawn(world seed, persona seed) → reproducible |
| `description` | 10–600 chars | short rationale |

## 4. Archetype families (72 personas)

The six owner seed archetypes (issue #144) are included verbatim as named
archetypes, then extended across the behavior space:

| family | count | contains |
|---|---|---|
| `anchor` | 8 | **Big-name anchor** + orbit/brand/headline/ETF/Dow/S&P variants |
| `trender` | 8 | **WSB trender** + DD/options/squeeze/comment/gainers/cross/reversal variants |
| `list_pairer` | 6 | **Top-of-list pairer** + screener/gainers/losers/active/upgrade variants |
| `verifier` | 8 | **Substack verifier** + weekly/portfolio/free-picks/Discord/Telegram/YouTube/pairs variants |
| `miner` | 6 | **Influencer miner** + X/FinTok/podcast/streamer/pair variants |
| `farmer` | 6 | **Card farmer** + easy-mint/ghost/share-slot/badge/copycat variants |
| `power_user` | 8 | systematic quant, lead-lag hunter, multifactor, PIT-aware, backtest-first, specialist, re-test, anchored-run |
| `casual` | 10 | weekend, lunch-break, occasional, news-reader, paper-trader, retirement, friend-referred, mobile-only, dormant, digest |
| `abusive` | 12 | day-zero sharer, burner, device farm, IP ring, VPN hopper, velocity hammer, cap buster, duplicate re-sharer, unverifiable poster, screenshot poster, honeypot prober, coordinated share farm |

### 4.1 Distributions (design choice — NOT a market estimate)

These tables describe the **spec file's own composition**, chosen for
coverage. They say nothing about the real user base.

| sophistication | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| personas | 13 | 35 | 15 | 6 | 3 |

| strategy | random | pairs | momentum | narrative | copycat | lead_lag |
|---|---|---|---|---|---|---|
| personas | 21 | 19 | 12 | 10 | 8 | 2 |

| information source | personas using it |
|---|---|
| random | 21 |
| screeners | 20 |
| wsb | 16 |
| influencers | 14 |
| fundamentals | 10 |
| substack_free | 8 |
| r/stocktips | 4 |

- 13 of 72 personas carry ≥1 `abuse_flags` (all in the `abusive` family) so
  every §4.7 control has a named attack target: day-zero, multi-device,
  IP/ASN cluster, velocity, unverifiable shares, duplicate shares, cap
  busting, honeypot probing, coordinated share farm.
- 9 personas post unverifiable share fixtures (grant must NOT fire).
- Aggregate load knobs across the spec: 220 cards/week total minting goal,
  737 posts/week total sharing intent — sized for CI-smoke soaks (each bot
  session runs at most a few runs; the soak's N scaling comes from
  `run_soak`, not from persona counts).

## 5. How bots consume personas (`sift/sim/bots.py`)

`run_persona_session(world, persona)` builds a deterministic synthetic
identity from the persona seed (handle `sim-bot-<persona>-NNN`; abuse flags
steer shared device/IP/ASN shapes), registers share-account prerequisites
(day-zero personas register age 0 / 0 runs), runs
`min(queries_per_session // 5, max_runs)` mining runs through the real
pull/27 path (`run_one_mining_run` — ledger register → gauntlet → verdict →
mint/admit), then attempts share grants sized by `sharing_behavior`:
verifiable personas post the anonymized card link + `#SIFT` tag; unverifiable
personas post garbage; `honeypot_probe` personas end with a canary probe.
Every event is logged with `persona_id` + `deterministic_seed`.

`run_persona_sim(db_dir, seed=…)` runs a CI-smoke selection (every 7th
honest persona + all abusive ones) and returns a bannered report.

## 6. What the harness asserts (tests/sim/test_personas.py)

1. `personas.json` validates: 50–100 entries, schema/ranges, unique ids and
   seeds, SIMULATED header on the file, the six seed archetypes present,
   abuse flags covering day-zero/multi-device/IP-cluster/velocity/
   unverifiable.
2. Negative validation cases raise `PersonaSpecError` (duplicate id, bad
   sophistication, unknown source, too few personas, missing header).
3. Three seeded bot sessions run end-to-end against the pull/27 harness; the
   persona-sim report (JSON + text) carries the SIMULATED header on all
   outputs and every session logs persona id + seed.
4. Determinism: same seed → identical session logs.
5. A day-zero + unverifiable persona driving the bot path gets **nothing
   granted** — the §4.7 controls fire through the persona-driven path too.

## 7. Anti-misuse rule

Any artifact derived from these personas (reports, dashboards, decks,
docs) must carry the SIMULATED header and must never be presented as
evidence about real users. §8 of the spec is answered from the limited
alpha/beta (pull/25). If you find a doc citing sims as user-behavior
evidence, that is a bug — fix the doc, not the sim.
