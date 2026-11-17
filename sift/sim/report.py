"""Report builder: deterministic, bannered, machine-readable sim reports.

SIMULATED — infrastructure validation, not user-behavior evidence.

Every report dict/text produced here carries the SIM_HEADER banner and a
`sim: true` machine field. `same seed -> same report`: the report contains
no wall-clock timestamps (only the fixed sim clock), no random ids beyond
the seeded ones, and sorted keys.
"""

from __future__ import annotations

import json

from .header import SIM_HEADER, SIM_HEADER_MACHINE, header_lines


def percentiles(values: list[float], ps=(50, 95)) -> dict[str, float]:
    """Deterministic percentile computation (linear interpolation, sorted
    input — no RNG involved)."""
    vals = sorted(float(v) for v in values)
    if not vals:
        return {f"p{p}": 0.0 for p in ps}
    out = {}
    for p in ps:
        if len(vals) == 1:
            out[f"p{p}"] = vals[0]
            continue
        k = (len(vals) - 1) * (p / 100.0)
        lo = int(k)
        hi = min(lo + 1, len(vals) - 1)
        frac = k - lo
        out[f"p{p}"] = vals[lo] * (1 - frac) + vals[hi] * frac
    return out


def build_report(sim_name: str, seed: int, sections: dict) -> dict:
    """Assemble the report dict. Always bannered: `sim: true`,
    `header: SIM_HEADER`. Deterministic: json.dumps with sort_keys."""
    report = {
        "sim": True,
        "header": SIM_HEADER,
        "header_machine": SIM_HEADER_MACHINE,
        "sim_name": sim_name,
        "seed": int(seed),
        "determinism": "same seed -> same report (seeded RNG + fixed clock)",
        "sections": sections,
    }
    return report


def render_text(report: dict) -> str:
    """Human-readable report text. The FIRST line is always the banner."""
    lines = header_lines(seed=report.get("seed"), sim_name=report.get("sim_name"))
    lines.append("")
    lines.append(_render_sections(report.get("sections", {}), indent=0))
    return "\n".join(lines) + "\n"


def _render_sections(obj, indent: int) -> str:
    pad = "  " * indent
    out = []
    if isinstance(obj, dict):
        for k in sorted(obj):
            v = obj[k]
            if isinstance(v, (dict, list)):
                out.append(f"{pad}{k}:")
                out.append(_render_sections(v, indent + 1))
            else:
                out.append(f"{pad}{k}: {v}")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                out.append(f"{pad}-")
                out.append(_render_sections(item, indent + 1))
            else:
                out.append(f"{pad}- {item}")
    else:
        out.append(f"{pad}{obj}")
    return "\n".join(out)


def deterministic_view(report: dict) -> dict:
    """The seed-deterministic projection of a report: strips the wall-clock
    `measured` block (latencies/queue depth are real measurements and vary
    run to run even with the same seed). Everything else — sim decisions,
    DB-derived counts, verdicts, grant outcomes — is produced by the seeded
    RNG + fixed clock and must be byte-identical for the same seed."""
    import copy

    _STRIPPED_KEYS = ("measured", "latency_s", "queue", "wall_clock_s")

    def strip(obj):
        if isinstance(obj, dict):
            return {k: strip(v) for k, v in obj.items() if k not in _STRIPPED_KEYS}
        if isinstance(obj, list):
            return [strip(v) for v in obj]
        return obj

    out = strip(copy.deepcopy(report))
    return out if isinstance(out, dict) else {}


def dump_json(report: dict) -> str:
    """Canonical JSON (sorted keys, no wall-clock) — byte-identical for the
    same seed, which is what the determinism assertion compares."""
    return json.dumps(report, indent=2, sort_keys=True, default=str)


__all__ = ["build_report", "deterministic_view", "dump_json", "percentiles", "render_text"]
