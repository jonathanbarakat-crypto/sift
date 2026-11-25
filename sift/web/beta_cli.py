"""Operator CLI for the SIFT Gate-1 invited beta.

Database isolation:
- `auth.db`      — invites, sessions, and `beta_user` (the ONLY file with email)
- `telemetry.db` — append-only `telemetry_event` keyed by opaque `user_id`
                   (zero email column; safe to copy/export for analytics)
- `app.db`       — mining run records and saved connections (zero email)

Subcommands:
    invite-create   Mint one or more invite codes in auth.db and log
                    `beta_invited` to telemetry.db.
    report          Read-only Gate-1 D30/D60/D90 Wilson 95% CI report from
                    telemetry.db ONLY (never opens auth.db).
    export-events   Read-only JSONL export from telemetry.db ONLY.
    delete-user     Permanently erase a user's email/sessions from auth.db
                    and deactivate their saved connections in app.db.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from sift.web.auth import AuthStore
from sift.web.connections import ConnectionStore
from sift.web.telemetry import TelemetryStore, gate1_report


def _parse_as_of(raw: str | None) -> float | None:
    if not raw:
        return None
    s = raw.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def _format_report_text(rep: dict) -> str:
    lines = [
        "=== SIFT Gate-1 Limited Beta Report (docs/beta-plan.md §4-§7) ===",
        f"As of (UTC): {datetime.fromtimestamp(rep['as_of_ts'], tz=UTC).isoformat()}",
        f"Beta cohort size (n): {rep['cohort_n']}",
        f"Note: {rep['reporting_rules']}",
        "",
        "Active-user retention (>= 1 unique trial_registered in trailing 30d):",
    ]
    for label in ("D30", "D60", "D90"):
        h = rep["horizons"][label]
        lo_pct = h["wilson_95_ci"][0] * 100.0
        hi_pct = h["wilson_95_ci"][1] * 100.0
        rate_pct = h["rate"] * 100.0
        lines.append(
            f"  {label}: {h['k']}/{h['n']} matured ({rate_pct:.1f}%, "
            f"Wilson 95% CI [{lo_pct:.1f}%, {hi_pct:.1f}%], width {h['ci_width_pp']:.1f}pp) "
            f"— band={h['qualitative_band']}"
        )
    lines.append(
        f"D30 early stop-loss triggered (k < ceil(n/2)): "
        f"{'YES — pause invites & diagnose' if rep['d30_early_stop_loss_triggered'] else 'no'}"
    )
    sec = rep["secondary"]
    surv = sec["users_with_survivor"]
    lines.extend([
        "",
        "Secondary metrics (counts + Wilson 95% CI / small-n rules):",
        f"  Unique trials registered: {sec['unique_trials_registered']}",
        f"  Logins (trailing 30d):    {sec['logins_trailing_30d']}",
        f"  Connections saved:        {sec['connections_saved']}",
        (
            f"  Users with >=1 survivor:  {surv['k']}/{surv['n']} "
            f"(Wilson 95% CI [{surv['wilson_95_ci'][0]*100:.1f}%, "
            f"{surv['wilson_95_ci'][1]*100:.1f}%])"
        ),
        (
            f"  Card-layer events:        minted={sec['cards_minted']} "
            f"discarded={sec['cards_discarded']} "
            f"ghosts_promoted={sec['ghosts_promoted']} "
            f"share_grants={sec['share_grants']}"
        ),
    ])
    g1a = sec["g1_a_d90_split_counts_only"]
    lines.append(
        "  G1-a D90 split (counts only, no p-values): "
        f"card_holders={g1a['card_holders']['active_k']}/{g1a['card_holders']['matured_n']} "
        f"({g1a['card_holders']['band']}), "
        f"non_card_holders={g1a['non_card_holders']['active_k']}/{g1a['non_card_holders']['matured_n']} "
        f"({g1a['non_card_holders']['band']})"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SIFT Gate-1 Beta operator CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    default_dir = os.environ.get("SIFT_DATA_DIR", "data")

    p_inv = sub.add_parser("invite-create", help="Mint invite code(s) for beta wave")
    p_inv.add_argument("--data-dir", default=default_dir)
    p_inv.add_argument("--wave", default="wave-1")
    p_inv.add_argument("--count", type=int, default=1)
    p_inv.add_argument("--max-uses", type=int, default=1)
    p_inv.add_argument("--json", action="store_true", dest="as_json")

    p_rep = sub.add_parser("report", help="Read-only Gate-1 D30/D60/D90 Wilson CI report")
    p_rep.add_argument("--data-dir", default=default_dir)
    p_rep.add_argument("--as-of", default=None, help="ISO-8601 UTC timestamp")
    p_rep.add_argument("--json", action="store_true", dest="as_json")

    p_exp = sub.add_parser("export-events", help="Read-only PII-free JSONL event export")
    p_exp.add_argument("--data-dir", default=default_dir)
    p_exp.add_argument("--event-type", default=None)

    p_del = sub.add_parser("delete-user", help="Permanently erase a user's PII from auth.db")
    p_del.add_argument("--data-dir", default=default_dir)
    p_del.add_argument("--user-id", required=True)

    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir)
    auth_db = data_dir / "auth.db"
    tel_db = data_dir / "telemetry.db"
    app_db = data_dir / "app.db"

    if args.cmd == "invite-create":
        created = []
        with AuthStore(auth_db) as auth, TelemetryStore(tel_db) as tel:
            for _ in range(max(1, int(args.count))):
                inv = auth.create_invite(wave=args.wave, max_uses=args.max_uses)
                tel.record(
                    "beta_invited",
                    {
                        "invite_id": inv["invite_id"],
                        "wave": inv["wave"],
                        "max_uses": inv["max_uses"],
                    },
                )
                created.append(inv)
        if args.as_json:
            print(json.dumps({"invites": created}, indent=2))
        else:
            for inv in created:
                print(
                    f"{inv['code']}  (invite_id={inv['invite_id']}, "
                    f"wave={inv['wave']}, max_uses={inv['max_uses']})"
                )
        return 0

    if args.cmd == "report":
        as_of = _parse_as_of(args.as_of)
        # Reads ONLY telemetry.db (never opens auth.db).
        with TelemetryStore(tel_db) as tel:
            rep = gate1_report(tel, now=as_of)
        if args.as_json:
            print(json.dumps(rep, indent=2))
        else:
            print(_format_report_text(rep))
        return 0

    if args.cmd == "export-events":
        # Reads ONLY telemetry.db (never opens auth.db).
        with TelemetryStore(tel_db) as tel:
            events = tel.export_events(event_type=args.event_type)
        for ev in events:
            print(json.dumps(ev, sort_keys=True))
        return 0

    if args.cmd == "delete-user":
        with AuthStore(auth_db) as auth:
            deleted = auth.delete_account(args.user_id)
        with ConnectionStore(app_db) as conns:
            conns.delete_for_user(args.user_id)
        print(json.dumps({"ok": True, "deleted": deleted, "user_id": args.user_id}))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
