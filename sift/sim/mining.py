"""Simulated mining-run execution against a SimWorld (synthetic instruments).

SIMULATED — infrastructure validation, not user-behavior evidence.

Each simulated "mining run" registers a trial in the ledger and executes the
gauntlet on SYNTHETIC data (a planted AR(1) spread over two synthetic
instruments from the sim world's panel — never real tickers, never warehouse
data). The run path exercises exactly what production runs exercise: ledger
registration, stage rows, verdict, and (behind cards_v4_1) card minting +
vault admission — so the soak measures real DB contention and queue
behavior of the actual code paths, under synthetic load.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from sift.cards.service import CardLayer
from sift.gauntlet import stats
from sift.gauntlet.runner import GauntletRunner
from sift.registry import Ledger

from .clock import SimClock, SimRng
from .world import SIM_SCOPE_ID, SIM_VINTAGE, SimWorld

# Registration is the transactional bottleneck (trial_id minted from the
# ledger count — the production design registers BEFORE compute, behind a
# single writer / sequence). The sim serializes registration through this
# lock and lets the expensive gauntlet compute run concurrently across
# lanes, which is where the soak's DB contention actually lives (stage-row
# writes, card mints, vault admits from separate connections).
_REGISTER_LOCK = threading.Lock()


class RunCtx(Protocol):
    """Anything run_one_mining_run needs: a ledger + card layer + rng/clock.
    SimWorld satisfies it; soak lanes build lightweight per-lane contexts
    with their OWN connections on the same DB files (real contention)."""

    ledger: Ledger
    layer: CardLayer
    rng: SimRng
    seed: int
    clock: SimClock
    universe: list[str]


@dataclass
class RunOutcome:
    """One simulated mining run's measured outcome."""
    handle: str
    trial_id: str
    verdict: str
    survivor_class: str
    card_id: str | None
    wall_s: float          # measured simulated wall time of the run
    queue_depth_at_start: int
    extra: dict = field(default_factory=dict)


def _planted_pair(ctx: RunCtx, salt: int, rng: SimRng,
                  T: int = 1260) -> dict:
    """A synthetic planted AR(1) pair over two synthetic instruments.

    Deterministic per (world seed, salt, rng stream). Uses synthetic symbols
    only (SIM####), generated in-memory — never real tickers, never warehouse
    data. Window = one synthetic 5y (1260 bars), same construction as the
    F-PAIR-FULL acceptance fixture (phi=0.7, sig_s=10, log-space), so S4's
    cost-aware walk-forward clears its hurdle and S5 validates at full depth."""
    a = ctx.universe[int(rng.integers(0, len(ctx.universe)))]
    b = a
    while b == a:
        b = ctx.universe[int(rng.integers(0, len(ctx.universe)))]
    # log-space construction (same shape as the F-PAIR-FULL acceptance
    # fixture): log px random walk, planted AR(1) spread s, log py =
    # log px + s/scale. S0's beta_true is the OLS level hedge over the
    # window (computed from the fixture itself), and the half-life assertion
    # runs on the planted log-space spread.
    phi, sig_s, sig_x, scale = 0.7, 10.0, 0.01, 100.0
    lx = np.cumsum(rng.normal(0, sig_x, T))
    e = np.asarray(rng.normal(0, 1.0, T))
    s = np.zeros(T)
    for t in range(1, T):
        s[t] = phi * s[t - 1] + e[t] * sig_s
    px = 100.0 * np.exp(lx)
    py = px * np.exp(s / scale)
    prices = np.column_stack([px, py])
    beta_true = float(np.dot(px - px.mean(), py - py.mean())
                      / np.dot(px - px.mean(), px - px.mean()))
    return {
        "symbols": (a, b), "prices": prices,
        "beta_true": beta_true,
        "half_life_true": -math.log(2.0) / math.log(phi),
        "spread": s,
        "spread_returns": np.diff(s),
    }


def run_one_mining_run(ctx: RunCtx, user, *, salt: int,
                       queue_depth: int = 0,
                       s2_B: int = 100,
                       rng: SimRng | None = None,
                       survivor_salt: int | None = None) -> RunOutcome:
    """Execute one simulated mining run end-to-end through the real code
    path (ledger register -> gauntlet S0-S5 -> verdict -> mint/admit).

    `ctx` is a SimWorld (single-threaded sims) or a per-lane context with
    its own DB connections on the same files (soak). Registration is
    serialized through _REGISTER_LOCK (production registers before compute
    behind a single writer); gauntlet compute + stage writes run on the
    lane's own connections — that concurrency is the DB-contention surface
    the soak measures.

    The planted synthetic pair draws from a stream derived from
    (ctx.seed, survivor_salt or salt), so every run is deterministic per
    seed regardless of thread scheduling. `survivor_salt` lets the soak
    pin a handful of runs to a known-validated planted pair (the soak's
    mint-path assertion) while the rest draw arbitrary synthetic pairs.
    """
    if rng is None:
        rng = SimRng((ctx.seed * 1_000_003 + int(survivor_salt if survivor_salt is not None else salt)) % (2**63))
    pair = _planted_pair(ctx, salt, rng)
    prices = pair["prices"]
    x, y = prices[:, 0], prices[:, 1]
    excess = stats.pairs_strategy_returns(prices)["gross"][60:]
    with _REGISTER_LOCK:
        trial = ctx.ledger.register(
            "pairs", SIM_SCOPE_ID,
            {"sim": True, "salt": salt, "symbols": list(pair["symbols"])},
            SIM_VINTAGE, provenance="user_initiated",
            discoverer_handle=user.handle,
        )
    from sift.cards.minting import CardMinter
    runner = GauntletRunner(ctx.ledger, seed=ctx.seed + salt,
                            card_minter=CardMinter(ctx.ledger, ctx.layer.cards))
    t0 = time.perf_counter()
    verdict = runner.run_gauntlet(
        trial.trial_id,
        fixture_x=x, fixture_y=y,
        beta_true=pair["beta_true"], half_life_true=pair["half_life_true"],
        fixture_spread=pair["spread"],
        excess_returns=excess,
        perm_series=pair["spread_returns"],
        perm_statistic_fn=stats.mean_reversion_stat,
        prices=prices,
        scope_id=SIM_SCOPE_ID,
        s2_B=s2_B,
        full_prices=prices,
        full_spread_returns=pair["spread_returns"],
    )
    wall_s = time.perf_counter() - t0
    card_id = verdict.get("card_id")
    if card_id is not None:
        ctx.layer.vault.admit(card_id, user.handle, now=ctx.clock.now())
    return RunOutcome(
        handle=user.handle, trial_id=trial.trial_id,
        verdict=verdict["verdict"], survivor_class=verdict["survivor_class"],
        card_id=card_id, wall_s=wall_s,
        queue_depth_at_start=queue_depth,
        extra={"killed_at": verdict.get("killed_at"),
               "symbols": list(pair["symbols"])},
    )


__all__ = ["RunOutcome", "run_one_mining_run"]
