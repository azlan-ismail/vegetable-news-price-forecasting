"""Figures for the diagnostic re-analysis (reads build/diagnostics/)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
src = root / "build" / "diagnostics"
out = root / "results" / "figures"
out.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a9a8a1"
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "savefig.dpi": 300, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})

ablation = pd.read_csv(src / "ablation_by_block.csv")
placebo = pd.read_csv(src / "placebo.csv").set_index("target")
naive = float((ablation.test_rmse / ablation.ratio_vs_naive).iloc[0])

# Figure: placebo distributions ------------------------------------------------
titles = {"level": "(a) Price-level target", "return": "(b) Log-return target"}
fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.9))
for ax, target in zip(axes, ("level", "return")):
    perm = np.load(src / f"placebo_perm_{target}.npy")
    noise = np.load(src / f"placebo_noise_{target}.npy")
    price_only = float(ablation.query("target==@target and grid=='published' and n_text==0").test_rmse.iloc[0])
    lo = min(perm.min(), noise.min(), naive, price_only) * 0.97
    hi = max(np.percentile(np.r_[perm, noise], 99), placebo.loc[target, "real_text_rmse"]) * 1.02
    n_out = int((perm > hi).sum() + (noise > hi).sum())
    bins = np.linspace(lo, hi, 40)
    ax.set_xlim(lo, hi)
    ax.hist(noise, bins=bins, color=GRAY, alpha=0.45, label="Gaussian-noise block", edgecolor="white", linewidth=0.4)
    ax.hist(perm, bins=bins, histtype="step", color=INK, linewidth=1.2, label="Date-permuted news")
    ax.axvline(placebo.loc[target, "real_text_rmse"], color=BLUE, linewidth=2, label="Real news (42 features)")
    ax.axvline(price_only, color=ORANGE, linewidth=2, linestyle="--", label="Price-only Ridge")
    ax.axvline(naive, color=MUTED, linewidth=1.5, linestyle=":", label="Naive lag-1")
    ax.set_title(titles[target], fontsize=9, loc="left", color=INK)
    ax.text(0.98, 0.95, f"{n_out} of 1,000 draws beyond axis", transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color=MUTED)
    ax.set_xlabel("Test RMSE (RM/kg)")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("Placebo draws")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(out / "placebo_distributions.png", bbox_inches="tight")
plt.close(fig)

# Figure: RMSE ratio by text block ---------------------------------------------
order = ["all 42 text features", "sentiment only (22)", "topic counts only (20)",
         "training-fitted PCA, 5 PC", "training-fitted PCA, 3 PC", "corpus-wide 4",
         "corpus-wide sentiment only (3)", "training-fitted PCA, 1 PC",
         "article count only (1)", "mean sentiment only (1)"]
labels = {"all 42 text features": "All 42 text features", "sentiment only (22)": "Sentiment block (22)",
          "topic counts only (20)": "Count block (20)", "training-fitted PCA, 5 PC": "Text PCA, 5 components",
          "training-fitted PCA, 3 PC": "Text PCA, 3 components", "corpus-wide 4": "Corpus-wide (4)",
          "corpus-wide sentiment only (3)": "Corpus-wide sentiment (3)",
          "training-fitted PCA, 1 PC": "Text PCA, 1 component", "article count only (1)": "Article count (1)",
          "mean sentiment only (1)": "Mean sentiment (1)"}
pub = ablation[ablation.grid == "published"]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
y = np.arange(len(order))[::-1]
for target, color, marker, dy in (("level", BLUE, "o", 0.13), ("return", ORANGE, "s", -0.13)):
    vals = pub[pub.target == target].set_index("text_block").loc[order, "ratio_vs_price_only"].to_numpy()
    ax.scatter(vals, y + dy, color=color, marker=marker, s=34, zorder=3,
               edgecolor="white", linewidth=0.8,
               label="Price-level target" if target == "level" else "Log-return target")
ax.axvline(1.0, color=MUTED, linewidth=1, zorder=1)
ax.set_yticks(y, [labels[o] for o in order])
ax.set_xscale("log")
ticks = [0.95, 1.0, 1.1, 1.25, 1.5, 1.8]
ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
ax.minorticks_off()
ax.set_xlabel("Test RMSE relative to matched price-only Ridge (log scale)")
ax.grid(axis="x", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(out / "rmse_ratio_by_text_block.png", bbox_inches="tight")
plt.close(fig)
print("figures written")
