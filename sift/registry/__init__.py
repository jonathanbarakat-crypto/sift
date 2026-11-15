"""Append-only trial registry ledger (scope doc v3 §5).

See `sift.registry.ledger` for the schema and semantics: register() mints a
trial ID before any data access, the canonical-JSON params + config hash +
data-vintage hash form the skip-list key, stage rows s0..s5 are append-only,
and stratum stats are scope-stratified and never pooled.
"""

from sift.registry.ledger import (
    PROVENANCES,
    STAGES,
    SURVIVOR_CLASSES,
    VERDICTS,
    Ledger,
    LedgerError,
    Trial,
    canonical_json,
    config_hash_of,
    sample_variance,
    skip_list_key,
)

__all__ = [
    "PROVENANCES",
    "STAGES",
    "SURVIVOR_CLASSES",
    "VERDICTS",
    "Ledger",
    "LedgerError",
    "Trial",
    "canonical_json",
    "config_hash_of",
    "sample_variance",
    "skip_list_key",
]
