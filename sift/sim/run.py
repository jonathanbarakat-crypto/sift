"""Sim harness CLI entrypoint.

SIMULATED — infrastructure validation, not user-behavior evidence.

Usage:
    SIFT_ENV=test python3 -m sift.sim.run --sim soak --n-users 20 --out report.json
    SIFT_ENV=test python3 -m sift.sim.run --sim antiabuse --out report.json
    SIFT_ENV=test python3 -m sift.sim.run --sim lifecycle --out report.json
    SIFT_ENV=test python3 -m sift.sim.run --sim all --seed 20261117

Hard constraint enforced in code (sift/sim/world.assert_test_env): the CLI
refuses to run unless SIFT_ENV=test. Every emitted report (JSON or text)
starts with the SIM banner.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

from . import SIMULATED  # noqa: F401 — re-exported marker
from .antiabuse import run_antiabuse_sims
from .clock import DEFAULT_SEED
from .header import SIM_HEADER, header_lines
from .lifecycle import run_lifecycle_sim
from .report import dump_json, render_text
from .soak import DEFAULT_N_USERS, run_soak


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--sim", choices=("soak", "antiabuse", "lifecycle", "all"),
                    default="all")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--n-users", type=int, default=DEFAULT_N_USERS,
                    help="soak: number of simulated users (CI smoke: 20)")
    ap.add_argument("--runs-per-user", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None,
                    help="write the JSON report here (bannered); text to stdout always")
    args = ap.parse_args(argv)

    # HARD CONSTRAINT: test env only — never production data.
    if os.environ.get("SIFT_ENV") != "test":
        print(
            "REFUSED: sift/sim runs only with SIFT_ENV=test "
            "(ephemeral test DB; never production data)",
            file=sys.stderr,
        )
        return 2

    print(SIM_HEADER, flush=True)
    print("\n".join(header_lines(seed=args.seed)), flush=True)

    reports = {}
    with tempfile.TemporaryDirectory(prefix="sift-sim-") as tmp:
        if args.sim in ("soak", "all"):
            res = run_soak(os.path.join(tmp, "soak"),
                           n_users=args.n_users, runs_per_user=args.runs_per_user,
                           workers=args.workers, seed=args.seed)
            reports["load_soak"] = res.report
            print(f"[soak] n_users={args.n_users} runs={args.n_users * args.runs_per_user} "
                  f"p50={res.report['sections']['measured']['latency_s']['p50']:.4f}s "
                  f"p95={res.report['sections']['measured']['latency_s']['p95']:.4f}s "
                  f"max_queue_depth={res.report['sections']['measured']['queue']['max_queue_depth']} "
                  f"OK", flush=True)
        if args.sim in ("antiabuse", "all"):
            res = run_antiabuse_sims(os.path.join(tmp, "antiabuse"), seed=args.seed)
            reports["antiabuse_attack_sims"] = res.report
            failed = [k for k, v in res.checks.items() if not v]
            if failed:
                raise RuntimeError(f"anti-abuse controls did NOT fire as expected: {failed}")
            print(f"[antiabuse] {len(res.checks)} checks passed "
                  f"(controls fired, append-only log intact) OK", flush=True)
        if args.sim in ("lifecycle", "all"):
            rep = run_lifecycle_sim(os.path.join(tmp, "lifecycle"), seed=args.seed)
            reports["card_lifecycle_sim"] = rep
            if not rep["sections"]["all_checks_pass"]:
                raise RuntimeError(
                    f"lifecycle checks failed: "
                    f"{[k for k, v in rep['sections']['checks'].items() if not v]}"
                )
            print("[lifecycle] mint->ghost FIFO->decay downgrade-only->archive OK", flush=True)

    if args.out:
        payload = {
            "sim": True,
            "header": SIM_HEADER,
            "seed": args.seed,
            "reports": reports,
        }
        with open(args.out, "w") as f:
            f.write(json.dumps(payload, indent=2, sort_keys=True, default=str))
        print(f"[report] written to {args.out} (bannered, SIMULATED)", flush=True)
    # text render of the first report to stdout
    if reports:
        print(render_text(next(iter(reports.values()))), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
