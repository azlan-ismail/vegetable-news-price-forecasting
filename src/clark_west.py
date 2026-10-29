"""Clark-West (2007) adjusted-MSPE tests for nested forecast comparisons.

Null: the text block adds no predictive content (price-only model is the true
model). Alternative (one-sided): the text-augmented model has lower population
MSPE. Reads build/diagnostics/diagnostic_predictions.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

root = Path(__file__).resolve().parents[1]
src = root / "build" / "diagnostics"
pred = pd.read_csv(src / "diagnostic_predictions.csv", parse_dates=["target_date"])


def cw(f: np.ndarray) -> tuple[float, float]:
    n = len(f)
    t = f.mean() / (f.std(ddof=1) / np.sqrt(n))
    return float(t), float(stats.norm.sf(t))


rows = []
for target in ("level", "return"):
    for fit in ("published", "expanding"):
        small = pred[f"{target}|{fit}|none (price-only)"].to_numpy()
        for block in ("all 42 text features", "corpus-wide 4"):
            col = f"{target}|{fit}|{block}"
            if col not in pred:
                continue
            large = pred[col].to_numpy()
            y = pred["y_true"].to_numpy()
            f = (y - small) ** 2 - ((y - large) ** 2 - (small - large) ** 2)
            frame = pred[["target_date", "item"]].assign(f=f)
            # Pooled: average across commodities within a date (45 date-level terms).
            t, p = cw(frame.groupby("target_date")["f"].mean().to_numpy())
            rows.append({"target": target, "fit": fit, "text_block": block, "scope": "pooled (date means)",
                         "n": frame.target_date.nunique(), "cw_stat": t, "p_one_sided": p})
            for item, g in frame.groupby("item"):
                t, p = cw(g["f"].to_numpy())
                rows.append({"target": target, "fit": fit, "text_block": block, "scope": item,
                             "n": len(g), "cw_stat": t, "p_one_sided": p})
out = pd.DataFrame(rows)
out.to_csv(src / "clark_west.csv", index=False)
print(out[out.scope.str.startswith("pooled")].round(4).to_string(index=False))
print(out.groupby(["target", "fit", "text_block"]).apply(
    lambda g: int((g.p_one_sided < 0.05).sum()), include_groups=False).rename("n_p_lt_0.05"))
