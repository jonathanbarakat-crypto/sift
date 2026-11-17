"""v4.1 anchored run mode (mechanics 5, behind `cards_v4_1`).

An anchored run locks one or more ANCHOR slots (a pre-registered trial /
canonical spec that stays fixed in the run) while the remaining slots are
filled by RANDOM SAMPLING of partners / families / params. The run's scope
record (§12.11) carries: anchor slot(s), the sampled subspace, and N_eff
(the effective number of independent trials sampled — the multiplicity
denominator contribution), so the anchored design stays honest: the anchor
is pre-registered (not mined post-hoc), and the sampled part is counted in
N_eff.

The kill report shows per-stage kills (which stage each sampled candidate
died at), alongside the anchor's own stage rows.

Implementation: pure planning layer over the ledger. `plan_anchored_run`
builds the scope record deterministically from a seed (PCG64, recorded in
the scope record like every other seed in this repo); `run_anchored`
executes the sampled slots through the existing GauntletRunner and returns
a kill report with per-stage kill counts.

Flag off -> AnchoredRunError(flag_off).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

from sift import flags
from sift.gauntlet.runner import GauntletRunner
from sift.registry import Ledger, canonical_json


class AnchoredRunError(Exception):
    """Anchored-run misuse (flag off, bad anchor, bad subspace)."""


@dataclass
class AnchoredScopeRecord:
    """§12.11 scope record for an anchored run."""
    run_id: str
    anchors: list[str]                 # locked anchor slot(s): trial_ids
    subspace: dict                     # what the random slots sample
    n_eff: int                         # effective independent trials sampled
    seed: int
    created_at: float
    scope_id: str

    def to_json(self) -> str:
        return canonical_json(self.__dict__)


@dataclass
class KillReport:
    run_id: str
    per_stage_kills: dict[str, int]    # stage -> count of sampled candidates killed there
    survivors: list[str]               # sampled trial_ids that survived
    anchor_verdicts: dict[str, str]    # anchor trial_id -> verdict
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "per_stage_kills": self.per_stage_kills,
            "n_survivors": len(self.survivors),
            "survivors": self.survivors,
            "anchor_verdicts": self.anchor_verdicts,
            "detail": self.detail,
        }


def plan_anchored_run(
    ledger: Ledger,
    *,
    anchors: list[str],
    subspace: dict,
    n_eff: int,
    scope_id: str,
    seed: int,
    handle: str,
    now: float | None = None,
) -> AnchoredScopeRecord:
    """Build the §12.11 scope record: locked anchor slots + sampled subspace
    + N_eff, all recorded. The seed is recorded (PCG64) like every other
    seed. Anchors must be existing, decided trials (pre-registered)."""
    if not flags.cards_v4_1_enabled(handle):
        raise AnchoredRunError("flag_off: cards_v4_1 is disabled for this account")
    if not anchors:
        raise AnchoredRunError("anchored run needs at least one anchor slot")
    if n_eff < 1:
        raise AnchoredRunError("n_eff must be >= 1")
    for a in anchors:
        t = ledger.get(a)                       # raises if unknown
        if t.verdict == "pending":
            raise AnchoredRunError(f"anchor {a} is still pending — anchors must be pre-registered/decided")
    now = time.time() if now is None else now
    n = ledger.trial_count()
    run_id = f"anchored-{n + 1:06d}"
    return AnchoredScopeRecord(
        run_id=run_id, anchors=list(anchors), subspace=dict(subspace),
        n_eff=n_eff, seed=seed, created_at=now, scope_id=scope_id,
    )


def sample_partners(rng: np.random.Generator, universe: list, k: int) -> list:
    """Random sampling of partners from a universe without replacement
    (the anchored run's random partner slots)."""
    idx = rng.choice(len(universe), size=min(k, len(universe)), replace=False)
    return [universe[i] for i in sorted(int(i) for i in idx)]


def run_anchored(
    runner: GauntletRunner,
    ledger: Ledger,
    scope: AnchoredScopeRecord,
    sampler,
    handle: str,
) -> KillReport:
    """Execute an anchored run: the anchor slots are already decided (their
    verdicts are read back); the `sampler(i, rng)` callback produces the
    i-th sampled candidate's gauntlet inputs, which run through the normal
    GauntletRunner (registration -> S0-S5 -> verdict). The kill report
    aggregates per-stage kills across all sampled candidates.

    `sampler` must return a dict with the run_gauntlet kwargs + a `params`
    dict for registration. Deterministic given the recorded seed.
    """
    if not flags.cards_v4_1_enabled(handle):
        raise AnchoredRunError("flag_off: cards_v4_1 is disabled for this account")
    rng = np.random.Generator(np.random.PCG64(scope.seed))
    per_stage: dict[str, int] = {}
    survivors: list[str] = []
    for i in range(scope.n_eff):
        cand = sampler(i, rng)
        trial = ledger.register(
            cand.get("family", "pairs"), scope.scope_id, cand["params"],
            cand.get("data_vintage_hash", "v-anchored"),
            provenance=cand.get("provenance", "user_initiated"),
            discoverer_handle=handle,
            parent_trial_id=scope.anchors[0] if scope.anchors else None,
        )
        verdict = runner.run_gauntlet(trial.trial_id, **cand["gauntlet"])
        if verdict["verdict"] == "killed":
            stage = verdict.get("killed_at") or "unknown"
            per_stage[stage] = per_stage.get(stage, 0) + 1
        else:
            survivors.append(trial.trial_id)
    anchor_verdicts = {a: ledger.get(a).verdict for a in scope.anchors}
    return KillReport(
        run_id=scope.run_id,
        per_stage_kills=dict(sorted(per_stage.items())),
        survivors=survivors,
        anchor_verdicts=anchor_verdicts,
        detail={"n_eff": scope.n_eff, "seed": scope.seed,
                "subspace": scope.subspace, "anchors": scope.anchors},
    )


__all__ = [
    "AnchoredRunError",
    "AnchoredScopeRecord",
    "KillReport",
    "plan_anchored_run",
    "run_anchored",
    "sample_partners",
]
