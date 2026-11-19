"""v4.2 deck layer — S6 runner: bucket membership + portfolio gauntlet (spec v4.2 §4, #166).

Composition (server-side only, behind `decks_v4_2` + `cards_v4_1`):

- `ranked_membership(cards, bucket)` — deterministic membership: top-K by
  card tier within the bucket (K = bucket size), ties broken by card_id
  (stable, deterministic). Family mix filters by card.family; the sector
  split filters by the card's sector bucket label (passed in per card —
  the registry stores labels, never tickers).
- `run_s6(deck, component_returns, ...)` — the S6 portfolio gauntlet:
  equal-weight basket, fixed rebalance cadence (S6_REBALANCE_CADENCE),
  cost-aware walk-forward with realistic per-name half-spreads computed
  from the data (sift/gauntlet/s6.py — replaces the 100 bps placeholder,
  keeps 100 bps as the conservative bound), PBO via CSCV at depth, and the
  pass bar net SR>0 / |t_net|>=tau_cal(scope) / MinTRL<=T / PBO<=0.5
  multiplicity-corrected over the deck stratum (N = all registered deck
  trials in scope across all users, kills included; EFFECTIVE N for the
  within-deck correlation — components sharing names correlate).
- Custom decks are scored identically but stay INFORMATIONAL ONLY (amber,
  unranked, unshareable) — `run_s6` never sets a rank for kind='custom'.

Read-back discipline (hard rule): every registration/trial returns the
deck re-read from the registry, and every event is read back via
events(). Flag off -> DeckError(flag_off), fail-closed.

Hard rules (v3 §12 + v4.2 §5): membership is card IDs only; component
returns are supplied server-side by the engine (raw data never leaves the
server); no tickers/direction/P&L in any artifact or log payload.
"""

from __future__ import annotations

import time

import numpy as np

from sift import flags
from sift.cards import Card, CardRegistry
from sift.decks import Deck, DeckError, DeckRegistry
from sift.gauntlet import config as gcfg
from sift.gauntlet import s6 as s6stats
from sift.gauntlet import stats

# Card-tier strength order for top-K membership (cards use bronze/silver/gold;
# higher index = stronger). Deck ranks F-S are separate (see decks/__init__).
_CARD_TIER_ORDER = {"bronze": 0, "silver": 1, "gold": 2}


def _require_flag(handle: str | None = None) -> None:
    if not flags.decks_v4_2_enabled(handle):
        raise DeckError(
            "flag_off: decks_v4_2 is disabled (kill-switch SIFT_DECKS_V4_2=0"
            " or SIFT_CARDS_V4_1=0) — the deck layer is inert"
        )


def ranked_membership(cards: list[Card], bucket: dict,
                      sector_of: dict[str, str] | None = None) -> list[str]:
    """Deterministic ranked-deck membership: top-K by card tier within the
    bucket, K = bucket size. Filters: family mix (card.family in the mix,
    matching pairs/lead_lag) and the optional sector split (sector_of maps
    card_id -> GICS bucket label; bucket sector 'all' keeps everything).
    Ties broken by card_id so membership is fully deterministic.

    Returns the sorted list of member card ids (length == K). Raises
    DeckError if the bucket has fewer than K eligible cards (a ranked deck
    is only registered when its membership is fully determined)."""
    K = int(bucket["size"])
    sector = bucket.get("sector", gcfg.DECK_SECTOR_ALL)
    fams = set(bucket.get("family_mix") or ["pairs"])
    fam_map = {"pairs": "pairs", "lead_lag": "lead_lag", "lead-lag": "lead_lag"}
    eligible = []
    for c in cards:
        if c.kind != "regular":
            continue
        if fam_map.get(c.family, c.family) not in fams and c.family not in fams:
            continue
        if sector != gcfg.DECK_SECTOR_ALL:
            lbl = (sector_of or {}).get(c.card_id)
            if lbl is not None and lbl != sector:
                continue
        eligible.append(c)
    eligible.sort(key=lambda c: (-_CARD_TIER_ORDER.get(c.tier, 0), c.card_id))
    if len(eligible) < K:
        raise DeckError(
            f"bucket size {K} not satisfiable: only {len(eligible)} eligible cards"
        )
    return [c.card_id for c in eligible[:K]]


class DeckService:
    """Server-side deck layer: bucket registry + S6 gauntlet runs."""

    def __init__(self, cards: CardRegistry, decks: DeckRegistry):
        self.cards = cards
        self.decks = decks

    def close(self) -> None:
        pass  # registries are closed by their owners

    # -- registration ---------------------------------------------------------
    def register_ranked(self, *, scope_id: str, bucket: dict, owner_handle: str,
                        data_vintage_hash: str,
                        sector_of: dict[str, str] | None = None,
                        pool: list[Card] | None = None,
                        now: float | None = None) -> Deck:
        """Register a ranked deck: deterministic top-K membership in the
        bucket, dedupe on (scope, bucket config, vintage). Read-back via the
        registry. Flag off -> DeckError(flag_off)."""
        _require_flag(owner_handle)
        if pool is None:
            pool = self.cards.all_cards()
        members = ranked_membership(pool, bucket, sector_of=sector_of)
        return self.decks.register_ranked(
            scope_id=scope_id, config=bucket, member_card_ids=members,
            owner_handle=owner_handle, data_vintage_hash=data_vintage_hash,
            now=now,
        )

    def register_custom(self, *, scope_id: str, member_card_ids: list[str],
                        owner_handle: str, data_vintage_hash: str,
                        now: float | None = None) -> Deck:
        """Custom hand-picked deck: INFORMATIONAL ONLY (amber, unranked,
        unshareable). Membership is arbitrary; S6 may score it for the
        owner's own view but no rank is ever assigned."""
        _require_flag(owner_handle)
        for cid in member_card_ids:
            self.cards.assert_owned(cid, owner_handle)  # non-transferable; owner's own cards only
        return self.decks.register_custom(
            scope_id=scope_id, member_card_ids=member_card_ids,
            owner_handle=owner_handle, data_vintage_hash=data_vintage_hash,
            now=now,
        )

    # -- S6 -------------------------------------------------------------------
    def run_s6(self, deck_id: str, *, component_returns: np.ndarray,
               component_names: list[set] | None = None,
               prices: np.ndarray | None = None,
               p_perm: float | None = None,
               owner_handle: str | None = None,
               seed: int = 0, now: float | None = None) -> dict:
        """Run the S6 portfolio gauntlet on one deck and record the trial.

        - equal-weight basket, fixed rebalance cadence, cost-aware
          walk-forward with realistic per-name half-spreads from the data
          (`prices`, when supplied — else the conservative 100 bps bound);
          `prices` may be a 1D single-name series or a 2D (T, K) per-name
          panel (bars x names, one column per name); 2D panels are
          estimated per name, never flattened across symbol boundaries;
        - PBO via CSCV at depth over the basket's rebalance blocks;
        - pass bar: net SR>0, |t_net|>=tau_cal(scope), MinTRL<=T_scope,
          PBO<=0.5 — multiplicity-corrected over the deck stratum
          (N = decks.stratum_n(scope) = all registered deck trials in scope
          across ALL users, kills included);
        - EFFECTIVE N from the components' name sets (components sharing
          names correlate; 50 mixed cards != 50 independent bets);
        - custom decks: scored identically but the result is recorded as
          informational_only and NO rank is set (unranked, unshareable).

        Returns the S6 evidence dict; the deck row + deck_trial event are
        read back before returning."""
        _require_flag(owner_handle)
        deck = self.decks.get_deck(deck_id)
        if owner_handle is not None:
            self.decks.assert_owned(deck_id, owner_handle)
        now = time.time() if now is None else now

        R = np.asarray(component_returns, float)
        if R.ndim != 2 or R.shape[1] != len(json_loads(deck.member_card_ids)):
            raise DeckError(
                f"component_returns must be (T, K={len(json_loads(deck.member_card_ids))})"
            )
        # realistic per-name half-spreads from the data (two pre-registered
        # proxies: max(liquidity_concentration, dispersion, floor); keep the
        # 100 bps conservative bound as fallback when no usable evidence)
        if prices is not None:
            est = s6stats.estimate_half_spread_bps(prices)
        else:
            est = {"median_bps": gcfg.S6_HALF_SPREAD_BPS_CONSERVATIVE,
                   "source": "conservative_bound",
                   "proxy": gcfg.S6_HALF_SPREAD_PROXY}
        cost = s6stats.DeckCostModel(
            half_spread_bps=s6stats.effective_half_spread_bps(est),
            conservative_bound_bps=gcfg.S6_HALF_SPREAD_BPS_CONSERVATIVE,
        )
        basket = s6stats.deck_basket_returns(R, cost_model=cost)
        net = basket["net"]

        # PBO at depth via CSCV (S=16) over the deck's OWN component
        # returns — a genuine multi-configuration matrix (K components).
        # Undefined design (K<2 or T<S) -> None -> the pbo check FAILS
        # CLOSED (soundness #2-deck); never a fabricated 0.0.
        pbo = _deck_pbo(R, seed=seed)

        # deck stratum N (all registered deck trials in scope, kills included)
        n_stratum = self.decks.stratum_n(deck.scope_id)
        # effective N from component name sets (independent bets)
        if component_names is not None and len(component_names) == R.shape[1]:
            neff = s6stats.effective_n(component_names)
            n_eff = neff["n_eff"]
            overlap = neff["overlap"]
        else:
            n_eff = float(R.shape[1])
            overlap = 0.0
        # stratum variance of S6 net-SR statistics across deck trials
        v_stratum = _deck_stratum_variance(self.decks, deck.scope_id)

        verdict = s6stats.deck_gauntlet(
            net, scope_id=deck.scope_id, p_perm=p_perm,
            n_stratum=n_stratum, v_stratum=v_stratum, n_eff=n_eff, pbo=pbo,
            n_cards=int(R.shape[1]),
        )
        detail = dict(verdict)
        detail.update({
            "cost_source": est.get("source", "conservative_bound"),
            "half_spread_proxy": est.get("proxy", gcfg.S6_HALF_SPREAD_PROXY),
            "half_spread_bps_used": cost.half_spread_bps,
            "conservative_bound_bps": cost.conservative_bound_bps,
            "total_cost_bps": basket["total_cost_bps"],
            "rebalance_cadence": basket["rebalance_cadence"],
            "K": int(R.shape[1]),
            "n_eff": n_eff, "name_overlap_mean": overlap,
            "pbo_computed": pbo is not None,
            "min_trl_ok": bool(verdict["min_trl"] <= verdict["T_scope"]),
            "informational_only": deck.kind == "custom",
        })
        rb = self.decks.record_s6(deck_id, passed=bool(verdict["pass"]),
                                  detail=detail, handle=owner_handle, now=now)
        # read-back of the event too (hard rule)
        evs = self.decks.events("deck_trial", deck_id=deck_id)
        return {"deck": rb, "s6": detail, "deck_trial_events": len(evs)}


def json_loads(s: str):
    import json
    return json.loads(s)


def _deck_pbo(component_returns: np.ndarray, S: int = 16, seed: int = 0) -> float | None:
    """PBO at depth via CSCV (S=16) over the DECK'S OWN component returns.

    The (T, K) component matrix IS a genuine multi-configuration returns
    matrix: each column is one component card's strategy return series, so
    the CSCV combinatorial splits run over K >= 2 real configurations —
    no synthetic perturbations, no hardcoded 0.0 (soundness #2-deck: the
    previous body always returned 0.0 while S6_PBO_MAX=0.5 was advertised,
    making the gate vacuous).

    Returns None when the design is undefined (K < 2 components or T < S
    blocks) — callers must fail the pbo check closed on None, never pass
    a fabricated 0.0. Uses stats.cscv_pbo with the corrected sign
    (mean(lambda >= 0), rank 0 = best; the sign fix is shared with pull/52
    — if that PR has not merged, this local copy of the convention is
    identical and will consolidate when it lands).
    """
    R = np.asarray(component_returns, float)
    if R.ndim != 2:
        # a single net-basket series carries no configuration dimension:
        # CSCV is undefined -> None (fail closed)
        return None
    R = R[np.all(np.isfinite(R), axis=1)] if R.size else R
    return stats.cscv_pbo(R, S=S, seed=seed)


def _deck_stratum_variance(decks: DeckRegistry, scope_id: str) -> float:
    """Sample variance of the scope's recorded S6 net-SR statistics
    (floored at 1/T by dsr()); computed strictly within the scope."""
    import json as _json
    vals = []
    for d in decks.decks_in_scope(scope_id):
        row = decks.con.execute(
            "SELECT s6_detail_json FROM deck WHERE deck_id=?", (d.deck_id,)
        ).fetchone()
        if row and row["s6_detail_json"]:
            try:
                det = _json.loads(row["s6_detail_json"])
                if det.get("sr_ann") is not None:
                    vals.append(float(det["sr_ann"]) / 252.0 ** 0.5)
            except Exception:
                continue
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return sum((v - m) ** 2 for v in vals) / (len(vals) - 1)


__all__ = ["DeckService", "ranked_membership"]
