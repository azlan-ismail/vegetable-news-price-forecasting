"""Compare a fresh diagnostic re-analysis run with the committed reference outputs.

Usage: python src/check_diagnostics.py [--rtol 1e-6]
Reads build/diagnostics/ and results/diagnostic_reanalysis/. Exits non-zero on mismatch.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "build" / "diagnostics"
REF = ROOT / "results" / "diagnostic_reanalysis"

ap = argparse.ArgumentParser()
ap.add_argument("--rtol", type=float, default=1e-6)
ap.add_argument("--atol", type=float, default=1e-9)
args = ap.parse_args()

failures = []
for ref_path in sorted(REF.glob("*")):
    new_path = NEW / ref_path.name
    if not new_path.exists():
        failures.append(f"{ref_path.name}: missing from fresh run")
        continue
    if ref_path.suffix == ".npy":
        a, b = np.load(new_path), np.load(ref_path)
        ok = a.shape == b.shape and np.allclose(a, b, rtol=args.rtol, atol=args.atol)
        diff = float(np.max(np.abs(a - b))) if a.shape == b.shape else float("nan")
    elif ref_path.suffix == ".csv":
        a, b = pd.read_csv(new_path), pd.read_csv(ref_path)
        ok = list(a.columns) == list(b.columns) and len(a) == len(b)
        diff = 0.0
        if ok:
            for col in a.columns:
                if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
                    x, y = a[col].to_numpy(float), b[col].to_numpy(float)
                    if not np.allclose(x, y, rtol=args.rtol, atol=args.atol, equal_nan=True):
                        ok = False
                    diff = max(diff, float(np.nanmax(np.abs(x - y))) if len(x) else 0.0)
                elif not a[col].astype(str).equals(b[col].astype(str)):
                    ok = False
    else:
        continue
    print(f"{'OK  ' if ok else 'FAIL'} {ref_path.name}  max abs diff = {diff:.3g}")
    if not ok:
        failures.append(ref_path.name)

if failures:
    print("Mismatched outputs:", ", ".join(failures))
    sys.exit(1)
print("All diagnostic outputs match the reference within tolerance.")
