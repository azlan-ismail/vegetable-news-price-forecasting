"""Reproduce the matched Ridge forecasting comparison from public-safe data.

This script starts from the released model-ready aggregate dataset. It does not
require or redistribute publisher-owned news article text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PRICE_COLUMNS = [
    "price_lag_1",
    "price_lag_2",
    "price_lag_3",
    "price_lag_7",
    "price_lag_14",
    "price_lag_30",
    "price_roll_mean_7",
    "price_roll_std_7",
    "price_roll_min_7",
    "price_roll_max_7",
    "price_roll_mean_14",
    "price_roll_std_14",
    "price_roll_min_14",
    "price_roll_max_14",
    "price_roll_mean_30",
    "price_roll_std_30",
    "price_roll_min_30",
    "price_roll_max_30",
    "prior_return_1",
    "target_dow_sin",
    "target_dow_cos",
    "target_month_sin",
    "target_month_cos",
]
ALPHA_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]


def make_pipeline(numeric_columns: list[str], alpha: float) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            ("item", OneHotEncoder(handle_unknown="ignore", drop=None), ["item"]),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=alpha)),
        ]
    )


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred)
    values = np.divide(
        2 * np.abs(y_pred - y_true),
        denominator,
        out=np.zeros_like(np.asarray(y_true, dtype=float)),
        where=denominator != 0,
    )
    return float(np.mean(values) * 100)


def dm_hln(loss_baseline: pd.Series, loss_hybrid: pd.Series, h: int = 1) -> tuple[float, float]:
    differential = np.asarray(loss_baseline) - np.asarray(loss_hybrid)
    n = len(differential)
    if n < 3 or np.isclose(differential.std(ddof=1), 0):
        return np.nan, np.nan
    statistic = differential.mean() / (differential.std(ddof=1) / np.sqrt(n))
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    statistic *= correction
    p_value = 2 * stats.t.sf(abs(statistic), df=n - 1)
    return float(statistic), float(p_value)


def holm_adjust(p_values: pd.Series) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def compare_csv(actual_path: Path, reference_path: Path, sort_columns: list[str]) -> None:
    actual = pd.read_csv(actual_path).sort_values(sort_columns).reset_index(drop=True)
    reference = pd.read_csv(reference_path).sort_values(sort_columns).reset_index(drop=True)
    if list(actual.columns) != list(reference.columns):
        raise AssertionError(f"Column mismatch for {actual_path.name}")
    if len(actual) != len(reference):
        raise AssertionError(f"Row-count mismatch for {actual_path.name}")
    for column in actual.columns:
        if pd.api.types.is_numeric_dtype(actual[column]) and pd.api.types.is_numeric_dtype(reference[column]):
            if not np.allclose(
                actual[column].to_numpy(dtype=float),
                reference[column].to_numpy(dtype=float),
                rtol=1e-8,
                atol=1e-10,
                equal_nan=True,
            ):
                delta = np.nanmax(
                    np.abs(
                        actual[column].to_numpy(dtype=float)
                        - reference[column].to_numpy(dtype=float)
                    )
                )
                raise AssertionError(
                    f"Numeric mismatch for {actual_path.name}:{column}; max absolute difference={delta}"
                )
        else:
            left = actual[column].fillna("<NA>").astype(str)
            right = reference[column].fillna("<NA>").astype(str)
            if not left.equals(right):
                raise AssertionError(f"Value mismatch for {actual_path.name}:{column}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the public-safe forecasting and evaluation stage."
    )
    parser.add_argument(
        "--model-data",
        type=Path,
        default=ROOT / "data/derived/modelling/conditional_model_ready_dataset.csv",
    )
    parser.add_argument(
        "--price-features",
        type=Path,
        default=ROOT / "data/derived/preprocessing/price_features_causal.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "build/reproduced",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=ROOT / "results",
    )
    parser.add_argument(
        "--check-reference",
        action="store_true",
        help="Fail if reproduced tables differ from the committed reference results.",
    )
    args = parser.parse_args()

    np.random.seed(42)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_data = pd.read_csv(
        args.model_data,
        parse_dates=["target_date", "forecast_origin", "text_cutoff_day"],
    )
    missing_price_columns = sorted(set(PRICE_COLUMNS).difference(model_data.columns))
    if missing_price_columns:
        raise ValueError(f"Model-ready data are missing price columns: {missing_price_columns}")

    text_columns = [
        column
        for column in model_data.columns
        if column in {"all_article_count", "all_sentiment_mean", "positive_share", "negative_share"}
        or (column.startswith("topic_") and column != "topic_minus_one_count")
    ]
    count_columns = [column for column in text_columns if "count" in column]
    for column in count_columns:
        model_data[f"log1p_{column}"] = np.log1p(model_data[column].astype(float))
    text_model_columns = [
        f"log1p_{column}" if column in count_columns else column for column in text_columns
    ]

    train = model_data.loc[model_data["split"].eq("train")].copy()
    validation = model_data.loc[model_data["split"].eq("validation")].copy()
    test = model_data.loc[model_data["split"].eq("test")].copy()
    expected_sizes = {"train": 240, "validation": 120, "test": 360}
    observed_sizes = {"train": len(train), "validation": len(validation), "test": len(test)}
    if observed_sizes != expected_sizes:
        raise ValueError(f"Unexpected split sizes: {observed_sizes}")

    specifications = {
        "price_only_ridge": PRICE_COLUMNS,
        "text_enhanced_ridge": PRICE_COLUMNS + text_model_columns,
    }
    selection_rows: list[dict[str, object]] = []
    selected_alpha: dict[str, float] = {}
    for model_name, numeric_columns in specifications.items():
        model_rows: list[dict[str, object]] = []
        for alpha in ALPHA_GRID:
            candidate = make_pipeline(numeric_columns, alpha)
            candidate.fit(train[numeric_columns + ["item"]], train["y_true"])
            prediction = candidate.predict(validation[numeric_columns + ["item"]])
            row = {
                "model": model_name,
                "alpha": alpha,
                "validation_rmse": mean_squared_error(validation["y_true"], prediction) ** 0.5,
            }
            selection_rows.append(row)
            model_rows.append(row)
        candidates = pd.DataFrame(model_rows).sort_values(["validation_rmse", "alpha"])
        selected_alpha[model_name] = float(candidates.iloc[0]["alpha"])

    model_selection = pd.DataFrame(selection_rows)
    model_selection["selected"] = model_selection.apply(
        lambda row: row["alpha"] == selected_alpha[row["model"]], axis=1
    )
    model_selection.to_csv(output_dir / "model_selection_validation.csv", index=False)

    development = pd.concat([train, validation], ignore_index=True)
    prediction_frames: list[pd.DataFrame] = []
    for model_name, numeric_columns in specifications.items():
        fitted = make_pipeline(numeric_columns, selected_alpha[model_name])
        fitted.fit(development[numeric_columns + ["item"]], development["y_true"])
        prediction = test[
            ["target_date", "forecast_origin", "text_cutoff_day", "item", "y_true"]
        ].copy()
        prediction["model"] = model_name
        prediction["y_pred"] = fitted.predict(test[numeric_columns + ["item"]])
        prediction_frames.append(prediction)

    naive = test[
        ["target_date", "forecast_origin", "text_cutoff_day", "item", "y_true"]
    ].copy()
    naive["model"] = "naive_lag_1"
    naive["y_pred"] = test["price_lag_1"].to_numpy()
    prediction_frames.append(naive)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output_dir / "conditional_test_predictions.csv", index=False)

    pair_check = (
        predictions.loc[
            predictions["model"].isin(["price_only_ridge", "text_enhanced_ridge"])
        ]
        .pivot(
            index=["target_date", "forecast_origin", "item", "y_true"],
            columns="model",
            values="y_pred",
        )
        .reset_index()
    )
    if len(pair_check) != 45 * 8:
        raise ValueError("The paired Ridge comparison is not aligned on 45 dates x 8 commodities.")

    metric_rows: list[dict[str, object]] = []
    for (model_name, item), group in predictions.groupby(["model", "item"]):
        metric_rows.append(
            {
                "scope": "commodity",
                "item": item,
                "model": model_name,
                "n": len(group),
                "mae": mean_absolute_error(group["y_true"], group["y_pred"]),
                "rmse": mean_squared_error(group["y_true"], group["y_pred"]) ** 0.5,
                "smape_pct": smape(group["y_true"].to_numpy(), group["y_pred"].to_numpy()),
            }
        )
    for model_name, group in predictions.groupby("model"):
        metric_rows.append(
            {
                "scope": "pooled",
                "item": "ALL",
                "model": model_name,
                "n": len(group),
                "mae": mean_absolute_error(group["y_true"], group["y_pred"]),
                "rmse": mean_squared_error(group["y_true"], group["y_pred"]) ** 0.5,
                "smape_pct": smape(group["y_true"].to_numpy(), group["y_pred"].to_numpy()),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "conditional_test_metrics.csv", index=False)

    dm_rows: list[dict[str, object]] = []
    for item, group in pair_check.groupby("item"):
        baseline_loss = (group["y_true"] - group["price_only_ridge"]) ** 2
        hybrid_loss = (group["y_true"] - group["text_enhanced_ridge"]) ** 2
        statistic, p_value = dm_hln(baseline_loss, hybrid_loss, h=1)
        dm_rows.append(
            {
                "item": item,
                "n_irregular_test_dates": len(group),
                "dm_hln_statistic": statistic,
                "p_value_unadjusted": p_value,
                "mean_squared_loss_difference_baseline_minus_hybrid": float(
                    (baseline_loss - hybrid_loss).mean()
                ),
                "hybrid_win_rate": float((hybrid_loss < baseline_loss).mean()),
                "interpretation": "Sensitivity only: irregular conditional dates",
            }
        )
    dm_results = pd.DataFrame(dm_rows)
    valid = dm_results["p_value_unadjusted"].notna()
    dm_results["p_value_holm"] = np.nan
    dm_results.loc[valid, "p_value_holm"] = holm_adjust(
        dm_results.loc[valid, "p_value_unadjusted"]
    )
    dm_results.to_csv(output_dir / "dm_hln_irregular_date_sensitivity.csv", index=False)

    price_features = pd.read_csv(
        args.price_features, parse_dates=["target_date", "forecast_origin"]
    )
    validation_start = validation["target_date"].min()
    history = price_features.loc[
        price_features["target_date"].lt(validation_start)
        & price_features["target_observed"].astype(bool)
    ].copy()
    history["abs_return"] = np.abs(history["y_true"] / history["price_lag_1"] - 1)
    thresholds = (
        history.groupby("item")["abs_return"].quantile(0.75).rename("high_volatility_threshold")
    )

    test_with_volatility = test[["target_date", "item", "y_true", "price_lag_1"]].copy()
    test_with_volatility["abs_return"] = np.abs(
        test_with_volatility["y_true"] / test_with_volatility["price_lag_1"] - 1
    )
    test_with_volatility = test_with_volatility.join(thresholds, on="item")
    test_with_volatility["volatility_regime"] = np.where(
        test_with_volatility["abs_return"].ge(
            test_with_volatility["high_volatility_threshold"]
        ),
        "high",
        "lower",
    )

    volatility_rows: list[dict[str, object]] = []
    for (item, regime), dates in test_with_volatility.groupby(
        ["item", "volatility_regime"]
    ):
        keys = dates[["target_date", "item"]]
        for model_name in ["price_only_ridge", "text_enhanced_ridge"]:
            model_predictions = predictions.loc[predictions["model"].eq(model_name)].merge(
                keys, on=["target_date", "item"], how="inner", validate="one_to_one"
            )
            volatility_rows.append(
                {
                    "item": item,
                    "volatility_regime": regime,
                    "model": model_name,
                    "n": len(model_predictions),
                    "mae": mean_absolute_error(
                        model_predictions["y_true"], model_predictions["y_pred"]
                    ),
                    "rmse": mean_squared_error(
                        model_predictions["y_true"], model_predictions["y_pred"]
                    )
                    ** 0.5,
                    "inferential_status": (
                        "descriptive_only"
                        if len(model_predictions) >= 10
                        else "descriptive_too_small_n_lt_10"
                    ),
                }
            )
    volatility_results = pd.DataFrame(volatility_rows)
    volatility_results.to_csv(
        output_dir / "volatility_conditioned_descriptive_metrics.csv", index=False
    )
    thresholds.reset_index().to_csv(
        output_dir / "training_volatility_thresholds.csv", index=False
    )

    checks = {
        "model_selection_validation.csv": ["model", "alpha"],
        "conditional_test_predictions.csv": ["target_date", "item", "model"],
        "conditional_test_metrics.csv": ["scope", "item", "model"],
        "dm_hln_irregular_date_sensitivity.csv": ["item"],
        "training_volatility_thresholds.csv": ["item"],
        "volatility_conditioned_descriptive_metrics.csv": [
            "item",
            "volatility_regime",
            "model",
        ],
    }
    verification = "not_requested"
    if args.check_reference:
        for filename, sort_columns in checks.items():
            reference_subdir = "predictions" if "predictions" in filename else "metrics"
            compare_csv(
                output_dir / filename,
                args.reference_root / reference_subdir / filename,
                sort_columns,
            )
        verification = "passed"

    pooled = metrics.loc[metrics["scope"].eq("pooled"), ["model", "rmse"]]
    summary = {
        "status": "reproduced",
        "seed": 42,
        "selected_alpha": selected_alpha,
        "pooled_test_rmse": dict(zip(pooled["model"], pooled["rmse"])),
        "reference_verification": verification,
    }
    (output_dir / "reproduction_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
