"""Diagnostic re-analysis of the matched price-only versus text-enhanced comparison.

Runs entirely from the released public-safe aggregate data. It asks *why* the
text-enhanced pooled Ridge model under-performs, using:

  A. Specification: level target (as published) versus a scale-free log-return
     target in which the naive forecast is the zero-coefficient special case.
  B. Placebo: the real date-level text vectors versus text vectors permuted across
     dates within each split (same marginal distribution, timing destroyed) and
     versus Gaussian noise blocks of the same width.
  C. Dimensionality / feature-block ablation, including training-fitted PCA.
  D. Covariate shift of the text block between development and test dates.
  E. Expanding-window refits through the test period.
  F. Date-level bootstrap confidence intervals for RMSE ratios.

Outputs are written to build/diagnostics/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

PRICE_COLUMNS = [
    "price_lag_1", "price_lag_2", "price_lag_3", "price_lag_7", "price_lag_14", "price_lag_30",
    "price_roll_mean_7", "price_roll_std_7", "price_roll_min_7", "price_roll_max_7",
    "price_roll_mean_14", "price_roll_std_14", "price_roll_min_14", "price_roll_max_14",
    "price_roll_mean_30", "price_roll_std_30", "price_roll_min_30", "price_roll_max_30",
    "prior_return_1", "target_dow_sin", "target_dow_cos", "target_month_sin", "target_month_cos",
]
CALENDAR = ["target_dow_sin", "target_dow_cos", "target_month_sin", "target_month_cos"]
PUBLISHED_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]
WIDE_GRID = list(np.logspace(-3, 5, 17))


# --------------------------------------------------------------------------- data

def load(model_path: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    d = pd.read_csv(model_path, parse_dates=["target_date"])
    text = [c for c in d.columns
            if c in {"all_article_count", "all_sentiment_mean", "positive_share", "negative_share"}
            or (c.startswith("topic_") and c != "topic_minus_one_count")]
    counts = [c for c in text if "count" in c]
    for c in counts:
        d[f"log1p_{c}"] = np.log1p(d[c].astype(float))
    text_cols = [f"log1p_{c}" if c in counts else c for c in text]

    # Scale-free (return) representation of the price block.
    d = d.copy()
    lag1 = d["price_lag_1"]
    new = {}
    ret_cols = []
    for c in PRICE_COLUMNS:
        if c in CALENDAR or c == "prior_return_1":
            ret_cols.append(c)
            continue
        if c == "price_lag_1":
            continue
        name = f"r_{c}"
        if "roll_std" in c:
            new[name] = d[c] / lag1
        else:
            new[name] = np.log(d[c] / lag1)
        ret_cols.append(name)
    new["log_return"] = np.log(d["y_true"] / lag1)
    d = pd.concat([d, pd.DataFrame(new, index=d.index)], axis=1)
    d = d.sort_values(["target_date", "item"]).reset_index(drop=True)
    return d, ret_cols, text_cols


def pipe(cols: list[str], alpha: float) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), cols),
        ("item", OneHotEncoder(handle_unknown="ignore"), ["item"]),
    ])
    return Pipeline([("pre", pre), ("reg", Ridge(alpha=alpha))])


def to_price(frame: pd.DataFrame, pred: np.ndarray, target: str) -> np.ndarray:
    if target == "level":
        return pred
    return frame["price_lag_1"].to_numpy() * np.exp(pred)


def target_values(frame: pd.DataFrame, target: str) -> pd.Series:
    return frame["y_true"] if target == "level" else frame["log_return"]


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def select_and_test(d: pd.DataFrame, cols: list[str], target: str, grid: list[float]) -> dict:
    """Validation-selected alpha, refit on train+validation, predict test (price scale)."""
    tr, va, te = (d[d.split == s] for s in ("train", "validation", "test"))
    scores = []
    for a in grid:
        m = pipe(cols, a).fit(tr[cols + ["item"]], target_values(tr, target))
        scores.append((rmse(va["y_true"], to_price(va, m.predict(va[cols + ["item"]]), target)), a))
    val_rmse, alpha = sorted(scores)[0]
    dev = pd.concat([tr, va])
    m = pipe(cols, alpha).fit(dev[cols + ["item"]], target_values(dev, target))
    pred = to_price(te, m.predict(te[cols + ["item"]]), target)
    return {"alpha": float(alpha), "val_rmse": float(val_rmse),
            "test_rmse": rmse(te["y_true"], pred), "pred": pred,
            "alpha_at_grid_edge": alpha in (grid[0], grid[-1])}


# ------------------------------------------------------------------ text variants

def permute_text(d: pd.DataFrame, text_cols: list[str], rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle whole date-level text vectors across dates within each split."""
    out = d.copy()
    by_date = d.groupby("target_date")[text_cols].first()
    split_of = d.groupby("target_date")["split"].first()
    mapping = {}
    for s in ("train", "validation", "test"):
        dates = split_of.index[split_of == s].to_numpy()
        mapping.update(dict(zip(dates, rng.permutation(dates))))
    src = out["target_date"].map(mapping)
    out[text_cols] = by_date.loc[src].to_numpy()
    return out


def noise_text(d: pd.DataFrame, k: int, rng: np.random.Generator) -> tuple[pd.DataFrame, list[str]]:
    out = d.copy()
    dates = out["target_date"].unique()
    z = pd.DataFrame(rng.standard_normal((len(dates), k)), index=dates,
                     columns=[f"noise_{i}" for i in range(k)])
    out[z.columns] = z.loc[out["target_date"]].to_numpy()
    return out, list(z.columns)


def add_pca(d: pd.DataFrame, text_cols: list[str], k: int) -> tuple[pd.DataFrame, list[str]]:
    """PCA of the date-level text block fitted on training dates only."""
    out = d.copy()
    by_date = d.groupby("target_date")[text_cols].first()
    split_of = d.groupby("target_date")["split"].first()
    sc = StandardScaler().fit(by_date[split_of == "train"])
    pca = PCA(n_components=k, random_state=SEED).fit(sc.transform(by_date[split_of == "train"]))
    pcs = pd.DataFrame(pca.transform(sc.transform(by_date)), index=by_date.index,
                       columns=[f"text_pc{i+1}" for i in range(k)])
    out[pcs.columns] = pcs.loc[out["target_date"]].to_numpy()
    return out, list(pcs.columns)


# ----------------------------------------------------------------- expanding refit

def expanding(d: pd.DataFrame, cols: list[str], target: str, alpha: float) -> np.ndarray:
    """Refit on all earlier eligible dates before each test date; alpha held fixed."""
    te_dates = sorted(d.loc[d.split == "test", "target_date"].unique())
    preds = pd.Series(index=d.index[d.split == "test"], dtype=float)
    for t in te_dates:
        hist = d[d.target_date < t]
        cur = d[d.target_date == t]
        m = pipe(cols, alpha).fit(hist[cols + ["item"]], target_values(hist, target))
        preds.loc[cur.index] = to_price(cur, m.predict(cur[cols + ["item"]]), target)
    return preds.to_numpy()


# ---------------------------------------------------------------------- bootstrap

def bootstrap_ratio(te: pd.DataFrame, pa: np.ndarray, pb: np.ndarray, n_boot: int,
                    rng: np.random.Generator, block: int = 1) -> tuple[float, float, float]:
    """RMSE(a)/RMSE(b) with a (moving-block) bootstrap over test dates."""
    frame = te[["target_date", "y_true"]].copy()
    frame["ea"] = (pa - frame["y_true"]) ** 2
    frame["eb"] = (pb - frame["y_true"]) ** 2
    per_date = frame.groupby("target_date")[["ea", "eb"]].sum().sort_index()
    ea, eb = per_date["ea"].to_numpy(), per_date["eb"].to_numpy()
    n = len(ea)
    point = float(np.sqrt(ea.sum() / eb.sum()))
    stats_ = []
    starts_max = n - block + 1
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block)
                              for s in rng.integers(0, starts_max, int(np.ceil(n / block)))])[:n]
        stats_.append(np.sqrt(ea[idx].sum() / eb[idx].sum()))
    lo, hi = np.percentile(stats_, [2.5, 97.5])
    return point, float(lo), float(hi)


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-data", type=Path,
                    default=ROOT / "data/derived/modelling/conditional_model_ready_dataset.csv")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "build/diagnostics")
    ap.add_argument("--n-placebo", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    d, ret_cols, text_cols = load(args.model_data)
    te = d[d.split == "test"]
    naive = te["price_lag_1"].to_numpy()
    naive_rmse = rmse(te["y_true"], naive)
    price_cols = {"level": PRICE_COLUMNS, "return": ret_cols}
    results: dict = {"naive_test_rmse": naive_rmse, "n_text_features": len(text_cols)}

    # A + C. Specifications and feature blocks ---------------------------------
    corpus4 = ["log1p_all_article_count", "all_sentiment_mean", "positive_share", "negative_share"]
    counts = [c for c in text_cols if c.startswith("log1p_")]
    sents = [c for c in text_cols if not c.startswith("log1p_")]
    blocks = {
        "none (price-only)": [],
        "all 42 text features": text_cols,
        "topic counts only (20)": counts,
        "sentiment only (22)": sents,
        "corpus-wide 4": corpus4,
        "corpus-wide sentiment only (3)": corpus4[1:],
        "article count only (1)": corpus4[:1],
        "mean sentiment only (1)": ["all_sentiment_mean"],
    }
    pca_frames = {}
    for k in (1, 3, 5):
        pca_frames[k] = add_pca(d, text_cols, k)

    rows, preds = [], {}
    for target in ("level", "return"):
        for grid_name, grid in (("published", PUBLISHED_GRID), ("wide", WIDE_GRID)):
            for name, blk in blocks.items():
                r = select_and_test(d, price_cols[target] + blk, target, grid)
                rows.append({"target": target, "grid": grid_name, "text_block": name,
                             "n_text": len(blk), **{k: v for k, v in r.items() if k != "pred"}})
                preds[(target, grid_name, name)] = r["pred"]
            for k, (dk, pcs) in pca_frames.items():
                r = select_and_test(dk, price_cols[target] + pcs, target, grid)
                name = f"training-fitted PCA, {k} PC"
                rows.append({"target": target, "grid": grid_name, "text_block": name,
                             "n_text": k, **{kk: v for kk, v in r.items() if kk != "pred"}})
                preds[(target, grid_name, name)] = r["pred"]
    ablation = pd.DataFrame(rows)
    base = ablation[ablation.text_block == "none (price-only)"].set_index(["target", "grid"])["test_rmse"]
    ablation["ratio_vs_price_only"] = ablation.apply(
        lambda r: r.test_rmse / base[(r.target, r.grid)], axis=1)
    ablation["ratio_vs_naive"] = ablation["test_rmse"] / naive_rmse
    ablation.to_csv(out / "ablation_by_block.csv", index=False)

    # B. Placebo ---------------------------------------------------------------
    placebo_rows = []
    for target in ("level", "return"):
        real = ablation.query("target==@target and grid=='published' and text_block=='all 42 text features'")
        real_rmse = float(real.test_rmse.iloc[0])
        perm, noise = [], []
        for _ in range(args.n_placebo):
            dp = permute_text(d, text_cols, rng)
            perm.append(select_and_test(dp, price_cols[target] + text_cols, target, PUBLISHED_GRID)["test_rmse"])
            dn, ncols = noise_text(d, len(text_cols), rng)
            noise.append(select_and_test(dn, price_cols[target] + ncols, target, PUBLISHED_GRID)["test_rmse"])
        perm, noise = np.array(perm), np.array(noise)
        placebo_rows.append({
            "target": target, "real_text_rmse": real_rmse,
            "permuted_median": float(np.median(perm)),
            "permuted_p2_5": float(np.percentile(perm, 2.5)),
            "permuted_p97_5": float(np.percentile(perm, 97.5)),
            "share_permuted_le_real": float(np.mean(perm <= real_rmse)),
            "noise_median": float(np.median(noise)),
            "noise_p2_5": float(np.percentile(noise, 2.5)),
            "noise_p97_5": float(np.percentile(noise, 97.5)),
            "share_noise_le_real": float(np.mean(noise <= real_rmse)),
            "n_draws": args.n_placebo,
        })
        np.save(out / f"placebo_perm_{target}.npy", perm)
        np.save(out / f"placebo_noise_{target}.npy", noise)
    pd.DataFrame(placebo_rows).to_csv(out / "placebo.csv", index=False)

    # D. Covariate shift -------------------------------------------------------
    by_date = d.groupby("target_date")[text_cols + ["split"]].first()
    dev = by_date[by_date.split != "test"][text_cols]
    tst = by_date[by_date.split == "test"][text_cols]
    sd = dev.std(ddof=1).replace(0, np.nan)
    smd = ((tst.mean() - dev.mean()) / sd).rename("std_mean_diff_test_vs_dev")
    shift = smd.to_frame()
    shift["dev_mean"] = dev.mean()
    shift["test_mean"] = tst.mean()
    shift.to_csv(out / "text_covariate_shift.csv")
    raw_counts = d.groupby("target_date")[["all_article_count", "split"]].first()
    results["mean_articles_per_date"] = raw_counts.groupby("split")["all_article_count"].mean().to_dict()
    results["share_text_features_abs_smd_gt_0_5"] = float((smd.abs() > 0.5).mean())
    results["median_abs_smd"] = float(smd.abs().median())

    # E. Expanding refits --------------------------------------------------------
    exp_rows = []
    for target in ("level", "return"):
        for name in ("none (price-only)", "all 42 text features", "corpus-wide 4"):
            a = float(ablation.query("target==@target and grid=='published' and text_block==@name").alpha.iloc[0])
            p = expanding(d, price_cols[target] + blocks[name], target, a)
            preds[(target, "expanding", name)] = p
            exp_rows.append({"target": target, "text_block": name, "alpha": a,
                             "test_rmse": rmse(te["y_true"], p)})
    exp = pd.DataFrame(exp_rows)
    exp.to_csv(out / "expanding_refit.csv", index=False)

    # F. Bootstrap CIs -----------------------------------------------------------
    boot_rows = []
    comparisons = [
        ("level", "published", "all 42 text features", "none (price-only)"),
        ("level", "published", "corpus-wide 4", "none (price-only)"),
        ("level", "published", "training-fitted PCA, 1 PC", "none (price-only)"),
        ("level", "expanding", "all 42 text features", "none (price-only)"),
        ("level", "expanding", "corpus-wide 4", "none (price-only)"),
        ("return", "published", "all 42 text features", "none (price-only)"),
        ("return", "wide", "all 42 text features", "none (price-only)"),
        ("return", "published", "corpus-wide 4", "none (price-only)"),
        ("return", "published", "training-fitted PCA, 1 PC", "none (price-only)"),
        ("return", "expanding", "all 42 text features", "none (price-only)"),
        ("return", "expanding", "corpus-wide 4", "none (price-only)"),
    ]
    for target, grid, a_name, b_name in comparisons:
        for block in (1, 5):
            pt, lo, hi = bootstrap_ratio(te, preds[(target, grid, a_name)],
                                         preds[(target, grid, b_name)], args.n_boot, rng, block)
            boot_rows.append({"target": target, "fit": grid, "model": a_name, "reference": b_name,
                              "block_len_dates": block, "rmse_ratio": pt, "ci_low": lo, "ci_high": hi})
    for target, fit in (("level", "published"), ("return", "published"),
                        ("level", "expanding"), ("return", "expanding")):
        for block in (1, 5):
            pt, lo, hi = bootstrap_ratio(te, preds[(target, fit, "none (price-only)")],
                                         naive, args.n_boot, rng, block)
            boot_rows.append({"target": target, "fit": fit, "model": "none (price-only)",
                              "reference": "naive lag-1", "block_len_dates": block,
                              "rmse_ratio": pt, "ci_low": lo, "ci_high": hi})
    pd.DataFrame(boot_rows).to_csv(out / "bootstrap_rmse_ratios.csv", index=False)

    # Per-commodity table for the return specification -------------------------
    item_rows = []
    for name in ("none (price-only)", "all 42 text features", "corpus-wide 4"):
        for target in ("level", "return"):
            p = preds[(target, "published", name)]
            tmp = te[["item", "y_true"]].assign(p=p, n=naive)
            for item, g in tmp.groupby("item"):
                item_rows.append({"target": target, "text_block": name, "item": item,
                                  "rmse": rmse(g.y_true, g.p), "naive_rmse": rmse(g.y_true, g.n)})
    pd.DataFrame(item_rows).to_csv(out / "commodity_rmse.csv", index=False)

    # Save predictions -----------------------------------------------------------
    pred_frame = te[["target_date", "item", "y_true", "price_lag_1"]].copy()
    for (target, grid, name), p in preds.items():
        pred_frame[f"{target}|{grid}|{name}"] = p
    pred_frame.to_csv(out / "diagnostic_predictions.csv", index=False)
    (out / "summary.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
