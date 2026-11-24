#!/usr/bin/env python3
"""Probe for sift issue #59: raw vs A.3 correlation-adjusted independent N
in the S3 DSR admission bar, using the 2026-11-20 EODHD eval numbers.

Run: python3 scripts/probe_s3_n_eff.py  (from the repo root)
"""
import math
import sys
from pathlib import Path
from statistics import NormalDist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sift.gauntlet import stats  # noqa: E402

nd = NormalDist()

N_RAW = 208_981     # raw stratum registration count (correlated pairs, 647-name panel)
K = 104             # panel return-correlation eigenvalues > 1
N_EFF = K * (K - 1) // 2          # C(104,2) = 5,356 (asset-cluster count)
V = 0.00136         # measured per-bar stratum variance
T = 1260
BARS = 252

rho_bar = (N_EFF - N_RAW) / (1.0 - N_RAW)   # A.3-implied average inter-trial correlation
print(f"panel: 647 names, {K}/647 eigenvalues > 1 -> n_eff_pairs = C({K},2) = {N_EFF}")
print(f"raw stratum N (pairs registered)            = {N_RAW}")
print(f"A.3 implied rho_bar at M={N_RAW} -> N_eff={N_EFF}: rho_bar = {rho_bar:.6f}")
print(f"A.3 check: implied_independent_n({N_RAW}, {rho_bar:.6f}) = "
      f"{stats.implied_independent_n(N_RAW, rho_bar):.1f}")
print()

for label, n_eff in (("RAW N (pre-fix)", None), ("A.3 n_eff (post-fix)", N_EFF)):
    for sr_ann in (2.90, 2.0, 1.5):
        sr_bar = sr_ann / math.sqrt(BARS)
        d = stats.dsr(sr_bar, T, N_RAW, V, n_eff=n_eff)
        print(f"{label:22s} sr_ann={sr_ann:4.2f}  sr_star_ann={d['sr_star']*math.sqrt(BARS):.4f}"
              f"  DSR={d['dsr']:.3e}  (n_used={d['n_eff']:.0f})")
    print()

# paper Exhibit 4 sanity: M=100, rho=0.5 -> N_hat = 50.5
print(f"paper Exhibit 4 check: implied_independent_n(100, 0.5) = "
      f"{stats.implied_independent_n(100, 0.5)} (paper: 50.5)")
# issue #59 headline: sr_star_ann 2.90 at raw N vs 2.43 at n_eff, V=0.00136
d_raw = stats.dsr(0.0, T, N_RAW, V, n_eff=None)
d_eff = stats.dsr(0.0, T, N_RAW, V, n_eff=N_EFF)
print(f"issue #59 headline: sr_star_ann raw={d_raw['sr_star']*math.sqrt(BARS):.2f} "
      f"vs n_eff={d_eff['sr_star']*math.sqrt(BARS):.2f} at V={V}")
