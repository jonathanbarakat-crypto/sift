"""Pluggable StrategyFamily protocol and family registry (`pairs`, `lead_lag`).

Prerequisite #2 from `docs/lead-lag-family-spec.md` (§5.2): separates family-
specific S0/S1/S2/S4/S5 statistical adapters from the family-agnostic
`GauntletRunner` orchestration and `(family, scope_id)` multiplicity tracking.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from sift.gauntlet import config as cfg
from sift.gauntlet import lead_lag as ll
from sift.gauntlet import stats


@runtime_checkable
class StrategyFamily(Protocol):
    """Minimal pluggable statistical adapter interface for a SIFT strategy family."""

    family_id: str
    directed: bool
    use_hac: bool

    def s0_calibrate(self, **kwargs: Any) -> dict: ...

    def s1_screen(self, **kwargs: Any) -> dict: ...

    def s2_null(self, **kwargs: Any) -> dict: ...

    def s4_walk_forward(self, prices: np.ndarray, **kwargs: Any) -> dict: ...

    def s5_variant_grid(self, **kwargs: Any) -> tuple[dict, ...]: ...

    def s5_variant_grid_digest(self, **kwargs: Any) -> str: ...


class PairsFamily:
    """Mean-reversion pairs adapter (`pairs` — undirected, IID conservative t)."""

    family_id = "pairs"
    directed = False
    use_hac = False

    def s0_calibrate(
        self,
        *,
        x: np.ndarray,
        y: np.ndarray,
        beta_true: float,
        half_life_true: float,
        spread: np.ndarray | None = None,
    ) -> dict:
        r = stats.calibrate_ar1_fixture(
            x,
            y,
            beta_true,
            half_life_true,
            beta_tol=cfg.S0_BETA_TOL,
            hl_rel_tol=cfg.S0_HL_REL_TOL,
            spread=spread,
        )
        return {
            "passed": r.passed,
            "statistic": r.beta_hat,
            "threshold": beta_true,
            "detail": {
                "beta_hat": r.beta_hat,
                "beta_true": r.beta_true,
                "half_life_hat": r.half_life_hat,
                "half_life_true": r.half_life_true,
                "beta_ok": r.beta_ok,
                "half_life_ok": r.half_life_ok,
            },
        }

    def s1_screen(self, *, excess_returns: np.ndarray) -> dict:
        return stats.discovery_screen(excess_returns, threshold=cfg.S1_THRESHOLD)

    def s2_null(
        self,
        *,
        series: np.ndarray,
        statistic_fn,
        B: int = cfg.S2_B,
        seed: int = 0,
    ) -> dict:
        return stats.block_permutation_null(series, statistic_fn, B=B, seed=seed)

    def s4_walk_forward(
        self,
        prices: np.ndarray,
        *,
        formation: int = 60,
        step: int = 120,
        open_z: float = 2.0,
        close_z: float = 0.5,
        cost_model: stats.CostModel | None = None,
        half_spreads: str | tuple = "placeholder",
    ) -> dict:
        return stats.walk_forward_returns(
            prices,
            formation=formation,
            step=step,
            open_z=open_z,
            close_z=close_z,
            cost_model=cost_model,
            half_spreads=half_spreads,
        )

    def s5_variant_grid(self, **kwargs: Any) -> tuple[dict, ...]:
        return cfg.s5_variant_grid()

    def s5_variant_grid_digest(self, **kwargs: Any) -> str:
        return cfg.s5_variant_grid_digest()


class LeadLagFamily:
    """Directed cross-stock lead-lag adapter (`lead_lag` — directed, HAC mandatory)."""

    family_id = "lead_lag"
    directed = True
    use_hac = True

    def s0_calibrate(
        self,
        *,
        prices: np.ndarray,
        market_returns: np.ndarray | None = None,
        lag_bars: int = cfg.LEAD_LAG_DEFAULT_LAG,
        hold_bars: int | None = None,
        formation: int = cfg.LEAD_LAG_DEFAULT_FORMATION,
        step: int = cfg.LEAD_LAG_STEP,
        delist_idx: np.ndarray | None = None,
        gamma_true: float | None = None,
    ) -> dict:
        r = ll.s0_lead_lag_sanity(
            prices,
            market_returns=market_returns,
            lag_bars=lag_bars,
            hold_bars=hold_bars,
            formation=formation,
            step=step,
            delist_idx=delist_idx,
            gamma_true=gamma_true,
        )
        return {
            "passed": r["passed"],
            # Store non-directional magnitude |t_hac| or lag_bars? Keep
            # statistic as valid_bars or abs(t_hac) — NEVER expose raw signed
            # gamma_hat in stage_result.statistic where card face could echo it.
            "statistic": float(r["lag_bars"]),
            "threshold": float(r["min_bars"]),
            "detail": r,
        }

    def s1_screen(
        self,
        *,
        r_a_tilde: np.ndarray,
        r_b_tilde: np.ndarray,
        lag_bars: int = cfg.LEAD_LAG_DEFAULT_LAG,
        hold_bars: int | None = None,
    ) -> dict:
        return ll.s1_lead_lag_screen(
            r_a_tilde,
            r_b_tilde,
            lag_bars=lag_bars,
            hold_bars=hold_bars,
            threshold=cfg.S1_THRESHOLD,
        )

    def s2_null(
        self,
        *,
        r_a_tilde: np.ndarray,
        r_b_tilde: np.ndarray,
        lag_bars: int = cfg.LEAD_LAG_DEFAULT_LAG,
        hold_bars: int | None = None,
        B: int = cfg.S2_B,
        seed: int = 0,
    ) -> dict:
        return ll.s2_lead_lag_permutation(
            r_a_tilde,
            r_b_tilde,
            lag_bars=lag_bars,
            hold_bars=hold_bars,
            B=B,
            seed=seed,
        )

    def s4_walk_forward(
        self,
        prices: np.ndarray,
        *,
        market_returns: np.ndarray | None = None,
        lag_bars: int = cfg.LEAD_LAG_DEFAULT_LAG,
        hold_bars: int | None = None,
        formation: int = cfg.LEAD_LAG_DEFAULT_FORMATION,
        open_z: float = cfg.LEAD_LAG_DEFAULT_OPEN_Z,
        step: int = cfg.LEAD_LAG_STEP,
        cost_model: stats.CostModel | None = None,
        half_spreads: str | float | np.ndarray = "roll",
    ) -> dict:
        return ll.lead_lag_walk_forward(
            prices,
            market_returns=market_returns,
            lag_bars=lag_bars,
            hold_bars=hold_bars,
            formation=formation,
            open_z=open_z,
            step=step,
            cost_model=cost_model,
            half_spreads=half_spreads,
        )

    def s5_variant_grid(self, *, lag_bars: int = cfg.LEAD_LAG_DEFAULT_LAG) -> tuple[dict, ...]:
        return cfg.s5_variant_grid_lead_lag(lag_bars)

    def s5_variant_grid_digest(self, *, lag_bars: int = cfg.LEAD_LAG_DEFAULT_LAG) -> str:
        return cfg.s5_variant_grid_lead_lag_digest(lag_bars)


FAMILY_REGISTRY: dict[str, StrategyFamily] = {
    "pairs": PairsFamily(),
    "lead_lag": LeadLagFamily(),
}


def get_family(family_id: str) -> StrategyFamily:
    """Resolve a registered StrategyFamily adapter (fails closed on unknown ID)."""
    key = str(family_id or "").strip().lower()
    if key not in FAMILY_REGISTRY:
        raise ValueError(
            f"unknown strategy family {family_id!r}; registered: {sorted(FAMILY_REGISTRY)}"
        )
    return FAMILY_REGISTRY[key]


__all__ = [
    "StrategyFamily",
    "PairsFamily",
    "LeadLagFamily",
    "FAMILY_REGISTRY",
    "get_family",
]
