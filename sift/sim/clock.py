"""Deterministic sim clock + seeded RNG.

The harness never reads the wall clock for sim decisions: every sim gets a
`SimClock` with a fixed start time, and every random draw comes from a
PCG64 generator seeded from the sim seed. Same seed -> same report.
"""

from __future__ import annotations

import numpy as np

DEFAULT_SEED = 20261117
DEFAULT_T0 = 1_800_000_000.0  # fixed epoch for all sims (deterministic)


class SimClock:
    """A fixed, manually advanced clock. `now` never moves unless `tick` is
    called, so a sim run is reproducible."""

    def __init__(self, t0: float = DEFAULT_T0):
        self.t0 = float(t0)
        self._now = float(t0)

    def now(self) -> float:
        return self._now

    def tick(self, dt: float) -> float:
        self._now += float(dt)
        return self._now

    def advance(self, days: float) -> float:
        return self.tick(days * 86400.0)


class SimRng:
    """Seeded RNG wrapper (PCG64). All sim randomness flows through here."""

    def __init__(self, seed: int = DEFAULT_SEED):
        self.seed = int(seed)
        self.gen = np.random.Generator(np.random.PCG64(self.seed))

    def integers(self, low: int, high: int | None = None, size: int | None = None):
        return self.gen.integers(low, high, size=size)

    def choice(self, seq, size: int | None = None, replace: bool = True):
        return self.gen.choice(np.asarray(seq), size=size, replace=replace)

    def normal(self, loc: float = 0.0, scale: float = 1.0, size: int | None = None):
        return self.gen.normal(loc, scale, size=size)

    def random(self, size: int | None = None):
        return self.gen.random(size=size)

    def shuffle(self, seq) -> None:
        self.gen.shuffle(seq)

    def spawn(self, salt: int) -> "SimRng":
        """A derived, independent stream for one sim user/attack (stable
        per salt, so per-user behavior is reproducible)."""
        return SimRng((self.seed * 1_000_003 + int(salt)) % (2**63))


__all__ = ["DEFAULT_SEED", "DEFAULT_T0", "SimClock", "SimRng"]
