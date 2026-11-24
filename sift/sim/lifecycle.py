"""Card-layer lifecycle sim: mint -> ghost FIFO promotion on discard ->
decay downgrade-only -> archive.

SIMULATED — infrastructure validation, not user-behavior evidence.

All card-layer sims run behind `cards_v4_1` and fail closed when the flag is
off (kill-switch `SIFT_CARDS_V4_1=0` respected — tested in
tests/sim/test_killswitch.py).

Asserts:
- mint of a validated-green synthetic survivor mints exactly one regular
  card; a duplicate spec mints a numbered ghost (FIFO);
- discarding ghost k promotes ghosts k+1.. down one slot (queue
  renumbering: active ghosts stay numbered 1..m with no gaps);
- decay pass may only DOWNGRADE tiers (gold->silver->bronze), never
  upgrade, and every downgrade appends a `card_decay` ledger event;
- archived cards stay in the personal archive + registry pool (never
  destroyed).
"""

from __future__ import annotations

import json
from pathlib import Path

from sift.cards import CARD_TIERS
from sift.cards.decay import DECAY_EVENT

from .clock import DEFAULT_SEED
from .report import build_report
from .world import SIM_SCOPE_ID, SIM_VINTAGE, SimWorld


def _mint(world: SimWorld, handle: str, tag: str,
          stats=(0.995, 4.5, 0.005, True, False)) -> str:
    trial = world.ledger.register(
        "pairs", SIM_SCOPE_ID, {"sim": True, "tag": tag},
        SIM_VINTAGE, provenance="user_initiated", discoverer_handle=handle,
    )
    world.ledger.decide(trial.trial_id, "survivor", "validated-green")
    card = world.layer.cards.mint(
        trial_id=trial.trial_id, family=trial.family, scope_id=trial.scope_id,
        config_hash=f"sim-h-{tag}", data_vintage_hash=trial.data_vintage_hash,
        survivor_class="validated-green", owner_handle=handle,
        dsr=stats[0], t_net_full=stats[1], p_perm=stats[2],
        s4_pass=stats[3], s5_pass=stats[4],
        now=world.clock.now(),
    )
    return card.card_id


def _same_spec_twice(world: SimWorld, handle: str, tag: str) -> tuple[str, str]:
    """Mint the SAME canonical spec twice: first -> regular, second -> ghost.

    One registered trial carries the verdict; both mints use its canonical
    spec (same config_hash), so the second mint is a duplicate -> ghost."""
    trial = world.ledger.register(
        "pairs", SIM_SCOPE_ID, {"sim": True, "tag": tag},
        SIM_VINTAGE, provenance="user_initiated", discoverer_handle=handle,
    )
    world.ledger.decide(trial.trial_id, "survivor", "validated-green")
    kw: dict = dict(
        trial_id=trial.trial_id, family=trial.family, scope_id=trial.scope_id,
        config_hash=f"sim-h-{tag}", data_vintage_hash=trial.data_vintage_hash,
        survivor_class="validated-green", owner_handle=handle,
        dsr=0.995, t_net_full=4.5, p_perm=0.005,
    )
    c1 = world.layer.cards.mint(now=world.clock.now(), **kw)
    c2 = world.layer.cards.mint(now=world.clock.now(), **kw)  # duplicate spec
    return c1.card_id, c2.card_id


def run_lifecycle_sim(db_dir: str | Path, *, seed: int = DEFAULT_SEED) -> dict:
    """Run the lifecycle sim; returns the bannered report."""
    checks: dict[str, bool] = {}
    with SimWorld(db_dir, seed=seed, n_users=1) as world:
        world.layer.require_layer(world.users[0].handle)  # fail-closed (flag on)
        handle = world.users[0].handle
        now = world.clock.now()

        # -- 1. mint: one regular card on a validated-green synthetic run ----
        regular = _mint(world, handle, "life-regular")
        card = world.layer.cards.get_card(regular)
        checks["mint_one_regular_card"] = card.kind == "regular" and card.tier == "gold"
        world.layer.vault.admit(regular, handle, now=now)

        # -- 2. duplicate spec -> numbered ghosts, FIFO ------------------------
        base_id, g1 = _same_spec_twice(world, handle, "life-ghost")
        # mint the same canonical spec once more -> ghost no 2
        spec_kwargs: dict = dict(
            trial_id=world.ledger.get(
                world.layer.cards.get_card(g1).trial_id).trial_id,
            family="pairs", scope_id=SIM_SCOPE_ID,
            config_hash="sim-h-life-ghost", data_vintage_hash=SIM_VINTAGE,
            survivor_class="validated-green", owner_handle=handle,
            dsr=0.995, t_net_full=4.5, p_perm=0.005,
        )
        g2_card = world.layer.cards.mint(now=world.clock.now(), **spec_kwargs)
        g2 = g2_card.card_id
        ghosts = world.layer.cards.ghosts_for_spec(
            world.layer.cards.get_card(g1).spec_hash)
        checks["ghost_numbered_fifo"] = (
            [g.ghost_no for g in ghosts] == [1, 2]
            and world.layer.cards.get_card(g1).kind == "ghost"
            and world.layer.cards.get_card(g2).ghost_no == 2
        )
        for gid in (g1, g2):
            world.layer.vault.admit(gid, handle, now=now)

        # -- 3. discard ghost 1 -> ghost 2 promotes to slot 1 (FIFO shift) ------
        spec_hash = world.layer.cards.get_card(g1).spec_hash
        before_slots = [g.ghost_no for g in world.layer.cards.ghosts_for_spec(spec_hash)]
        disc = world.layer.vault.discard(g1, handle, now=now)
        after = world.layer.cards.ghosts_for_spec(spec_hash)
        after_slots = [g.ghost_no for g in after]
        checks["ghost_fifo_promotion_on_discard"] = (
            before_slots == [1, 2] and disc["ghosts_shifted"] == 1
            and after_slots == [1]
            and after[0].card_id == g2
            and disc["pooled"] is True
        )
        # discarded ghost left the active queue (negative slot) but stays in registry
        discarded_card = world.layer.cards.get_card(g1)
        checks["discarded_ghost_never_destroyed"] = (
            discarded_card.ghost_no is not None and discarded_card.ghost_no <= 0
        )

        # -- 4. archive: discard the regular card -> archive + pool ------------
        arc = world.layer.vault.discard(regular, handle, now=now)
        checks["discard_moves_to_archive_and_pool"] = (
            arc["entry"].state == "archive" and arc["pooled"] is True
            and world.layer.vault.pool_size() >= 2
            and any(c.card_id == regular for c in world.layer.vault.archive(handle))
        )

        # -- 5. decay: downgrade-only, ledger event, never upgrades -------------
        # weaken the ledger stats so re-derivation downgrades gold -> bronze
        trial_id = world.layer.cards.get_card(regular).trial_id
        world.ledger.con.execute(
            "UPDATE stage_result SET dsr=0.80 WHERE trial_id=? AND stage='s3'",
            (trial_id,))
        world.ledger.con.execute(
            "UPDATE stage_result SET statistic=1.0 WHERE trial_id=? AND stage='s5'",
            (trial_id,))
        world.ledger.con.commit()
        results = world.layer.decay.run_pass(now=now + 3600)
        by_id = {r.card_id: r for r in results}
        r_reg = by_id[regular]
        checks["decay_downgrades_gold_to_lower"] = (
            r_reg.downgraded is True
            and CARD_TIERS.index(r_reg.to_tier) < CARD_TIERS.index("gold")
        )
        # strengthen stats -> a second pass must NOT upgrade (downgrade-only)
        world.ledger.con.execute(
            "UPDATE stage_result SET dsr=0.999 WHERE trial_id=? AND stage='s3'",
            (trial_id,))
        world.ledger.con.execute(
            "UPDATE stage_result SET statistic=9.0 WHERE trial_id=? AND stage='s5'",
            (trial_id,))
        world.ledger.con.commit()
        results2 = world.layer.decay.run_pass(now=now + 7200)
        tiers_after = {c.card_id: c.tier for c in world.layer.cards.all_cards()}
        checks["decay_never_upgrades"] = all(
            CARD_TIERS.index(tiers_after[c.card_id])
            <= CARD_TIERS.index(by_id[c.card_id].to_tier)
            for c in world.layer.cards.all_cards()
        ) and all(not r.downgraded or r.to_tier == tiers_after[r.card_id]
                  for r in results2)
        # every downgrade appended a card_decay event to the ledger
        decay_events = world.ledger.events(DECAY_EVENT)
        checks["decay_appends_ledger_event"] = (
            len(decay_events) >= 1
            and all(json.loads(e["payload_json"])["to_tier"] for e in decay_events)
        )
        sections = {
            "mint": {"regular_card": regular, "tier": card.tier,
                     "assertion": "validated-green synthetic survivor mints exactly one regular card"},
            "ghost_fifo": {"before_discard": before_slots, "after_discard": after_slots,
                           "shifted": disc["ghosts_shifted"],
                           "assertion": "discard of ghost k shifts ghosts below up one slot; active queue stays 1..m no gaps"},
            "archive": {"state": arc["entry"].state, "pool_size": world.layer.vault.pool_size(),
                        "assertion": "discard -> personal archive + registry pool; never destroyed"},
            "decay": {"downgraded": {r.card_id: [r.from_tier, r.to_tier] for r in results if r.downgraded},
                      "tiers_after_second_pass": tiers_after,
                      "ledger_decay_events": len(decay_events),
                      "assertion": "decay is downgrade-only (never upgrades); every downgrade appends a card_decay ledger event"},
            "checks": {k: bool(v) for k, v in sorted(checks.items())},
            "all_checks_pass": all(checks.values()),
        }
        return build_report("card_lifecycle_sim", seed, sections)


__all__ = ["run_lifecycle_sim"]
