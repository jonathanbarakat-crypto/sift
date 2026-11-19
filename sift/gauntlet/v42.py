"""v4.2 per-card gauntlet: S4 becomes a SCORED stage (spec v4.2 §2-§3, issue #165).

Design (spec v4.2 §2): pipeline execution order S0 -> S1 -> S2 -> S4 -> S3 -> S5
(stage numbers are labels, not sequence). S4's walk-forward result is RECORDED
as a score (net SR, t_net, MinTRL, T in the stage detail) and is NO LONGER a
binary kill line: a candidate that passes S0-S2 mints a card at bronze+ (amber
on research data — research_ scopes stay on-watch-amber). S4's pre-registered
thresholds (S4_NET_SR_MIN, tau_cal(scope), MinTRL <= T) are UNCHANGED and now
define the GOLD/PLATINUM evidence bars, not a kill line.

v4.2 §3: tiers bronze/silver/gold/platinum come from the full evidence vector
(S2 p_perm, S4 score, S3 DSR, S5 full-depth t). bronze = S0-S2 pass. The
bidirectional decay rules of #157 (pull/31) are unchanged: upgrades only on
pre-registered stats over fully out-of-sample trailing 12m windows (age > 12m,
two consecutive windows), max 1 tier change / 90 days, append-only ledger,
rarity-language public explanation. Health = rolling trailing-12m sim vs mint
baseline, display-only, NEVER a tier; deck membership may feed health, never
tier (the deck layer is issue #166 and does not exist yet — nothing here reads
deck state).

Verdict language (v4.2 §3): `killed` is kept ONLY for S0-S2 failures. S0-S2
passers get verdict=survivor with a survivor_class that reflects the evidence:
"validated-green" (S5 pass at full depth), "on-watch-amber" (S5 fail or
pending after S4+S3 passed), "on-watch-amber-s4-evidence" / "on-watch-amber-
s3-evidence" (S0-S2 passed, the S4 or S3 bar not met — bronze-tier evidence).
"""

from __future__ import annotations

from sift.gauntlet import config as cfg

# -- v4.2 survivor classes -------------------------------------------------------
# "validated-green" / "on-watch-amber" keep their v3/v4.1 meaning (S5 pass /
# S5 fail-or-pending). The S0-S2 passers that fail the S4 or S3 bar are no
# longer "killed": they are survivors at the bronze evidence tier, with the
# class naming the stage whose bar they did not clear. Research-grade scopes
# (research_*) stay on-watch-amber — amber on research data.
SURVIVOR_CLASS_S4_EVIDENCE = "on-watch-amber-s4-evidence"   # passed S0-S2, S4 bar not met
SURVIVOR_CLASS_S3_EVIDENCE = "on-watch-amber-s3-evidence"   # passed S0-S2+S4, S3 DSR bar not met
SURVIVOR_CLASSES_V42 = (
    "none",
    "on-watch-amber",
    "validated-green",
    SURVIVOR_CLASS_S4_EVIDENCE,
    SURVIVOR_CLASS_S3_EVIDENCE,
)

# -- v4.2 pre-registered evidence bars (spec v4.2 §2-§3, owner decision
# 2026-11-18: TIER IS N-INDEPENDENT — DSR is the mint-time admission gate
# only and is NEVER an input to the tier) ---------------------------------
# S4's own thresholds are UNCHANGED and define the GOLD/PLATINUM evidence
# bars (net SR > S4_NET_SR_MIN, |t_net| >= tau_cal(scope), MinTRL <= T).
# The pull/31 (#157) mint-time gold bar (DSR >= 0.99 AND full-depth
# |t_net| >= 4.0 AND p_perm <= 0.01) mapped 1:1 to raw stats minus DSR, so
# it becomes the PLATINUM raw bar. Silver keeps the full-depth line.
# Bronze = S0-S2 pass (the minimum card tier).
# DSR/PSR stays as the MINT-TIME ADMISSION GATE (DSR >= S3_DSR_MIN = 0.95 at
# mint-time N to mint above bronze) — recorded as context on the ledger
# stage row ("admission evidence at mint-time N"), never re-deflated by a
# later N. See mint_admission_gate().
MINT_DSR_GATE = 0.95                       # DSR >= 0.95 at mint-time N to mint above bronze
SILVER_T_NET_FULL = 3.0                    # silver: full-depth |t_net| >= 3.0 + p_perm <= 0.05
SILVER_P_PERM = 0.05
GOLD_P_PERM = 0.01                         # gold: S4 bar + p_perm <= 0.01
PLATINUM_T_NET_FULL = 4.0                  # platinum: gold + S5 pass + full-depth |t_net| >= 4.0
# Legacy names kept for callers/tests that referenced the v4.2-draft bars;
# they now denote RAW bars (no DSR anywhere).
GOLD_DSR, GOLD_P_PERM = 0.99, 0.01         # deprecated alias: platinum's old name
SILVER_DSR, SILVER_T_NET = 0.95, 3.0       # deprecated alias: the admission gate / silver line


def s4_passed(detail: dict, scope_id: str) -> bool:
    """S4's pre-registered bar, evaluated on the recorded stage detail:
    net SR > S4_NET_SR_MIN AND |t_net| >= tau_cal(scope) AND MinTRL <= T.
    UNCHANGED thresholds — in v4.2 this is an evidence bar, not a kill."""
    sr_ann = detail.get("sr_ann")
    t_net = detail.get("t_net")
    min_trl = detail.get("min_trl")
    T = detail.get("T")
    if sr_ann is None or t_net is None or min_trl is None or T is None:
        return False
    tau = cfg.tau_cal(scope_id)
    return (sr_ann > cfg.S4_NET_SR_MIN) and (abs(t_net) >= tau) and (min_trl <= T)


def survivor_class_for(*, s4_pass: bool, s3_pass: bool, s5_pass: bool | None) -> str:
    """Map the recorded stage outcomes to the v4.2 survivor class.

    - S5 pass (full-depth validated)  -> validated-green
    - S4 met, S3 met, S5 fail/pending  -> on-watch-amber
    - S0-S2 passed but S4 bar not met  -> on-watch-amber-s4-evidence
    - S4 met but S3 DSR bar not met    -> on-watch-amber-s3-evidence
    """
    if s5_pass is True:
        return "validated-green"
    if not s4_pass:
        return SURVIVOR_CLASS_S4_EVIDENCE
    if not s3_pass:
        return SURVIVOR_CLASS_S3_EVIDENCE
    return "on-watch-amber"


def mint_admission_gate(dsr: float | None, *, dsr_min: float = MINT_DSR_GATE) -> bool:
    """v4.2 §2-§3: DSR/PSR is the MINT-TIME ADMISSION GATE only.

    A candidate must clear DSR >= `dsr_min` (0.95) AT MINT-TIME N to mint
    above bronze. This is the ONLY place the stratum-deflated DSR decides a
    card's standing; the tier itself is computed from raw, N-independent
    stats by `sift.cards.tier_from_stats` (see there). A None DSR fails
    closed (no admission)."""
    if dsr is None:
        return False
    return dsr >= dsr_min


__all__ = [
    "SURVIVOR_CLASS_S4_EVIDENCE",
    "SURVIVOR_CLASS_S3_EVIDENCE",
    "SURVIVOR_CLASSES_V42",
    "MINT_DSR_GATE",
    "SILVER_T_NET_FULL",
    "SILVER_P_PERM",
    "GOLD_DSR",
    "GOLD_P_PERM",
    "SILVER_DSR",
    "SILVER_T_NET",
    "PLATINUM_T_NET_FULL",
    "s4_passed",
    "survivor_class_for",
    "mint_admission_gate",
]
