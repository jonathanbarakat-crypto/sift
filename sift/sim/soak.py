"""Load soak sim: N simulated users running mining runs concurrently.

SIMULATED — infrastructure validation, not user-behavior evidence.

Soaks the real server-side code paths (ledger registration, gauntlet stage
rows, verdicts, card mint + vault admission behind cards_v4_1) with N
simulated users submitting mining runs through a bounded worker pool
(a queue). Measures run latency, queue depth, vault capacity pressure and
DB contention; asserts no deadlocks/timeouts/errors; reports p50/p95.

N is configurable (default 50; CI smoke uses 20). Runs are dispatched to a
ThreadPoolExecutor with `workers` lanes — the "queue" — and each lane's
in-flight count is tracked as queue depth. All randomness (user order,
per-run salt) comes from the seeded RNG; the clock is fixed. Same seed ->
same report.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from sift.cards.service import CardLayer
from sift.registry import Ledger

from .clock import DEFAULT_SEED, SimClock, SimRng
from .mining import RunOutcome, run_one_mining_run
from .report import build_report, dump_json, percentiles
from .world import SimWorld

DEFAULT_N_USERS = 50
DEFAULT_RUNS_PER_USER = 3
DEFAULT_WORKERS = 8
SOAK_TIMEOUT_S = 120.0  # hard per-run timeout: no deadlocks/timeouts allowed
BUSY_TIMEOUT_MS = 30_000  # per-connection busy timeout (client-side contention bound)


@dataclass
class LaneCtx:
    """A soak lane's private context: its OWN connections to the same test
    DB files as the world (real cross-connection contention on the SQLite
    files), plus the shared seeded rng/clock/universe."""

    ledger: Ledger
    layer: CardLayer
    rng: SimRng
    seed: int
    clock: SimClock
    universe: list[str]

    def close(self) -> None:
        try:
            self.layer.close()
        finally:
            self.ledger.close()


def _open_lane(db_dir: Path, seed: int, rng: SimRng, clock: SimClock, universe) -> LaneCtx:
    ledger = Ledger(db_dir / "ledger.db")
    layer = CardLayer(ledger, db_dir / "cards", card_secret=b"sim-hmac-secret-test-only")
    # client-side busy timeout so contending writers wait instead of failing
    for con in (ledger.con, layer.cards.con, layer.vault.con, layer.shares.con,
                layer.notify.con):
        con.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    return LaneCtx(ledger=ledger, layer=layer, rng=rng, seed=seed,
                   clock=clock, universe=universe)


@dataclass
class SoakResult:
    report: dict
    outcomes: list[RunOutcome]
    json_text: str


def run_soak(db_dir: str | Path, *, n_users: int = DEFAULT_N_USERS,
             runs_per_user: int = DEFAULT_RUNS_PER_USER,
             workers: int = DEFAULT_WORKERS,
             seed: int = DEFAULT_SEED,
             timeout_s: float = SOAK_TIMEOUT_S) -> SoakResult:
    """Run the load soak. Returns the bannered report + raw outcomes.

    Assertions (all must hold, else RuntimeError):
    - every submitted run completed within `timeout_s` (no timeouts);
    - no exceptions/deadlocks escaped a lane;
    - no sqlite 'database is locked' errors (DB contention bounded by the
      short transactions in the real code paths);
    - p50/p95 latencies computed and reported.
    """
    with SimWorld(db_dir, seed=seed, n_users=n_users) as world:
        world.layer.require_layer(world.users[0].handle)  # fail-closed check (flag on in tests)
        jobs: list[tuple[object, int]] = []
        for u_i, user in enumerate(world.users):
            for r in range(runs_per_user):
                jobs.append((user, u_i * 1000 + r))
        # seeded submission order — but drawn from a DERIVED stream so the
        # shuffle does not consume the world stream that also builds the
        # users/panel (keeping every later draw deterministic per seed).
        order_rng = world.rng.spawn(777_777)
        order = list(range(len(jobs)))
        order_rng.shuffle(order)
        jobs = [jobs[i] for i in order]

        outcomes: list[RunOutcome] = []
        errors: list[str] = []
        in_flight = 0
        max_queue_depth = 0
        lock = threading.Lock()
        done_count = 0
        lanes: list[LaneCtx] = []

        # stratum effect: N (and V̄) grow as trials register in the shared
        # ledger, deflating DSR for later runs. The soak pins every job to a
        # known-validated planted pair (survivor_salt=0 for this seed) so the
        # full 40-run load exercises the mint/admit path end-to-end under
        # concurrency; the gauntlet's kill path is covered by tests/sim's
        # single-threaded verdict probes and by the existing gauntlet suite.
        pin_salt = 0

        def lane(job):
            nonlocal in_flight, max_queue_depth, done_count
            user, salt = job
            lane_ctx = _open_lane(Path(db_dir), seed, world.rng, world.clock,
                                  world.universe)
            with lock:
                lanes.append(lane_ctx)
            with lock:
                in_flight += 1
                max_queue_depth = max(max_queue_depth, in_flight)
            try:
                out = run_one_mining_run(lane_ctx, user, salt=salt,
                                         queue_depth=in_flight,
                                         survivor_salt=pin_salt)
                with lock:
                    outcomes.append(out)
            except Exception as exc:  # noqa: BLE001 — collect, assert after
                with lock:
                    errors.append(f"{type(exc).__name__}: {exc}")
            finally:
                with lock:
                    in_flight -= 1
                    done_count += 1
                lane_ctx.close()

        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(lane, job) for job in jobs]
            for f in futures:
                f.result(timeout=timeout_s)   # raises TimeoutError -> no timeouts allowed
        wall_s = time.perf_counter() - t_start

        if errors:
            raise RuntimeError(
                f"soak: {len(errors)} lane errors (deadlock/timeout/contention): "
                f"{errors[:5]}"
            )
        if done_count != len(jobs):
            raise RuntimeError(f"soak: only {done_count}/{len(jobs)} runs completed")

        latencies = [o.wall_s for o in outcomes]
        pct = percentiles(latencies, ps=(50, 95, 99))
        queue_depths = [o.queue_depth_at_start for o in outcomes]

        # vault capacity pressure: how many vault entries / how full
        con = sqlite3.connect(str(world.db_dir / "cards" / "vault.db"))
        try:
            vault_rows = con.execute(
                "SELECT COUNT(*) FROM vault_entry WHERE state='vault'").fetchone()[0]
            owners = con.execute(
                "SELECT COUNT(DISTINCT owner) FROM vault_entry").fetchone()[0]
            max_slots = con.execute(
                "SELECT COALESCE(MAX(extra_slots),0) FROM vault_owner_slots").fetchone()[0]
        finally:
            con.close()
        cards_minted = len(world.layer.cards.all_cards())
        trials = world.ledger.trial_count()

        # DB contention probe: count of trials actually persisted = all jobs
        # (each job registered exactly one trial; ledger is the contention
        # surface under concurrent writes)
        assert trials == len(jobs), f"ledger contention: {trials} != {len(jobs)} trials"

        sections = {
            "config": {
                "n_users": n_users, "runs_per_user": runs_per_user,
                "total_runs": len(jobs), "worker_lanes": workers,
                "seed": seed, "timeout_s": timeout_s,
                "env": "test (SIFT_ENV=test, ephemeral DB)",
            },
            # everything below "measured" is seed-deterministic (sim decisions
            # + DB-derived counts); wall-clock timings live under "measured"
            # and are real measurements (vary run to run even for one seed).
            "measured": {
                "latency_s": {
                    "p50": round(pct["p50"], 6), "p95": round(pct["p95"], 6),
                    "p99": round(pct["p99"], 6),
                    "min": round(min(latencies), 6), "max": round(max(latencies), 6),
                    "mean": round(sum(latencies) / len(latencies), 6),
                },
                "queue": {
                    "max_queue_depth": max_queue_depth,
                    "max_in_flight_lanes": workers,
                    "mean_queue_depth_at_start": round(
                        sum(queue_depths) / len(queue_depths), 3),
                },
                "wall_clock_s": round(wall_s, 3),
            },
            "vault_capacity_pressure": {
                "vault_entries": vault_rows, "owners_with_entries": owners,
                "cards_minted": cards_minted,
                "max_permanent_share_slots": max_slots,
            },
            "run_verdicts": {
                "survivor_validated_green": sum(
                    1 for o in outcomes if o.survivor_class == "validated-green"),
                "killed": sum(1 for o in outcomes if o.verdict == "killed"),
            },
            "db_contention": {
                "ledger_trials_persisted": trials,
                "expected_trials": len(jobs),
                "locked_errors": len([e for e in errors if "locked" in e]),
                "lane_errors": len(errors),
            },
            "assertions": {
                "no_timeouts": True, "no_deadlocks": True,
                "no_lane_errors": len(errors) == 0,
                "all_runs_completed": done_count == len(jobs),
            },
        }
        report = build_report("load_soak", seed, sections)
        return SoakResult(report=report, outcomes=outcomes,
                          json_text=dump_json(report))


__all__ = ["DEFAULT_N_USERS", "SoakResult", "run_soak"]
