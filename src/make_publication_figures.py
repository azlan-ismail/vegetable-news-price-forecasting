from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


root = Path(__file__).resolve().parents[1]
source = root / "results" / "metrics"
calendar_source = root / "data" / "derived" / "modelling"
output = root / "results" / "figures"
output.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)

# Conditional evaluation calendar
calendar = pd.read_csv(calendar_source / "conditional_evaluation_calendar.csv", parse_dates=["target_date"])
split_order = ["train", "validation", "test"]
split_labels = {"train": "Training (n=30)", "validation": "Validation (n=15)", "test": "Test (n=45)"}
split_colors = {"train": "#4C78A8", "validation": "#F58518", "test": "#E45756"}

fig, ax = plt.subplots(figsize=(8.0, 2.35))
for position, split in enumerate(split_order):
    subset = calendar.loc[calendar["split"].eq(split), "target_date"]
    ax.scatter(
        subset,
        [position] * len(subset),
        s=27,
        color=split_colors[split],
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )

ax.set_yticks(range(len(split_order)), [split_labels[split] for split in split_order])
ax.set_xlabel("Target date")
ax.set_ylim(-0.45, 2.45)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n2025"))
ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.8)
ax.set_axisbelow(True)
fig.tight_layout(pad=0.6)
fig.savefig(output / "conditional_evaluation_dates.png", bbox_inches="tight")
plt.close(fig)

# Pooled conditional-test RMSE
metrics = pd.read_csv(source / "conditional_test_metrics.csv")
pooled = metrics.loc[metrics["scope"].eq("pooled")].set_index("model")
model_order = ["naive_lag_1", "price_only_ridge", "text_enhanced_ridge"]
model_labels = ["Naive lag-1", "Price-only Ridge", "Text-enhanced Ridge"]
values = pooled.loc[model_order, "rmse"].to_numpy()
colors = ["#59A14F", "#4E79A7", "#E15759"]

fig, ax = plt.subplots(figsize=(6.2, 3.25))
bars = ax.bar(model_labels, values, color=colors, width=0.62)
ax.set_ylabel("RMSE (RM/kg)")
ax.set_ylim(0, max(values) * 1.18)
ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + max(values) * 0.025,
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
fig.tight_layout(pad=0.7)
fig.savefig(output / "conditional_test_rmse.png", bbox_inches="tight")
plt.close(fig)
