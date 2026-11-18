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

# -- v4.2 pre-registered evidence bars (spec v4.2 §2-§3) -------------------------
# S4's own thresholds are UNCHANGED and now define the gold/platinum bars.
# The pull/31 (#157) mint-time gold bar was DSR >= 0.99 AND full-depth
# |t_net| >= 4.0 AND p_perm <= 0.01; v4.2 keeps that bar for PLATINUM (with the
# S5 full-depth pass) and makes GOLD = S4 bar (net SR > S4_NET_SR_MIN,
# |t_net| >= tau_cal(scope), MinTRL <= T) AND S3 DSR >= 0.99 AND p_perm <= 0.01.
# Silver keeps the S3/S5 admission line. Bronze = S0-S2 pass.
GOLD_DSR, GOLD_P_PERM = 0.99, 0.01          # unchanged from pull/31
SILVER_DSR, SILVER_T_NET = 0.95, 3.0         # unchanged from pull/31
PLATINUM_T_NET_FULL = 4.0                    # full-depth |t_net| >= 4.0 + S5 pass


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


__all__ = [
    "SURVIVOR_CLASS_S4_EVIDENCE",
    "SURVIVOR_CLASS_S3_EVIDENCE",
    "SURVIVOR_CLASSES_V42",
    "GOLD_DSR",
    "GOLD_P_PERM",
    "SILVER_DSR",
    "SILVER_T_NET",
    "PLATINUM_T_NET_FULL",
    "s4_passed",
    "survivor_class_for",
]
