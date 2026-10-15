from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(
    description=(
        "Create leakage-safe price features, evaluation calendars, and audited "
        "news corpora from a joined PriceCatcher extract and a Google News export."
    )
)
parser.add_argument(
    "--price-file",
    type=Path,
    required=True,
    help="CSV with date, item, and price columns. Item names must already be joined from the PriceCatcher lookup table.",
)
parser.add_argument(
    "--news-file",
    type=Path,
    required=True,
    help="CSV with the source columns documented in data/README.md.",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("build/preprocessed"),
    help="Output directory (default: build/preprocessed).",
)
args = parser.parse_args()

PRICE_FILE = args.price_file.expanduser().resolve()
NEWS_FILE = args.news_file.expanduser().resolve()
OUT = args.output_dir.expanduser().resolve()
OUT.mkdir(parents=True, exist_ok=True)

TARGET_ITEMS = {
    "CILI AKAR HIJAU",
    "CILI AKAR MERAH",
    "CILI HIJAU",
    "CILI MERAH - KULAI",
    "CILI MERAH - MINYAK",
    "SAWI HIJAU",
    "TIMUN",
    "TOMATO",
}
PLACEHOLDERS = {"failed to extract text", "no text found"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text_key(value: str) -> str:
    return re.sub(r"[^\w]+", " ", str(value).casefold(), flags=re.UNICODE).strip()


def canonical_url(value: str) -> str:
    value = str(value).strip()
    if not value:
        return ""
    try:
        parsed = urlsplit(value)
        path = re.sub(r"/+$", "", parsed.path)
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, "", ""))
    except Exception:
        return value


def robust_mad(values: np.ndarray) -> float:
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def days_since_last_observation(observed: pd.Series) -> pd.Series:
    result = []
    distance = 0
    for value in observed.astype(bool):
        if value:
            distance = 0
        else:
            distance += 1
        result.append(distance)
    return pd.Series(result, index=observed.index, dtype="int64")


# ---------------------------------------------------------------------------
# Price preprocessing: national daily medians, causal fill, no overwritten raw
# ---------------------------------------------------------------------------
price = pd.read_csv(PRICE_FILE, low_memory=False)
price["date"] = pd.to_datetime(price["date"], errors="coerce").dt.normalize()
price["price"] = pd.to_numeric(price["price"], errors="coerce")
price["item"] = price["item"].astype(str).str.strip()
price = price.dropna(subset=["date", "price", "item"])
price = price[price["item"].isin(TARGET_ITEMS)].copy()
missing_items = sorted(TARGET_ITEMS.difference(price["item"].unique()))
if missing_items:
    raise ValueError(f"Price input is missing required manuscript commodities: {missing_items}")

daily_observed = (
    price.groupby(["date", "item"], as_index=False)
    .agg(price_observed=("price", "median"), n_obs=("price", "size"))
    .sort_values(["item", "date"])
)

calendar = pd.date_range(daily_observed["date"].min(), daily_observed["date"].max(), freq="D")
price_daily_parts = []
feature_parts = []

for item, group in daily_observed.groupby("item"):
    item_frame = group.set_index("date").reindex(calendar)
    item_frame.index.name = "date"
    item_frame["item"] = item
    item_frame["target_observed"] = item_frame["price_observed"].notna()
    item_frame["was_missing"] = ~item_frame["target_observed"]
    item_frame["n_obs"] = item_frame["n_obs"].fillna(0).astype(int)
    item_frame["price_causal"] = item_frame["price_observed"].ffill()
    item_frame["days_since_observed"] = days_since_last_observation(item_frame["target_observed"])

    prior = item_frame["price_causal"].shift(1)
    trailing_median = prior.rolling(14, min_periods=7).median()
    trailing_mad = prior.rolling(14, min_periods=7).apply(robust_mad, raw=True).replace(0, np.nan)
    robust_z = (item_frame["price_causal"] - trailing_median) / (1.4826 * trailing_mad)
    item_frame["causal_robust_z"] = robust_z
    item_frame["causal_outlier_flag"] = robust_z.abs().gt(4).fillna(False)
    item_frame["price_outlier_sensitivity"] = item_frame["price_causal"].where(
        ~item_frame["causal_outlier_flag"], trailing_median
    )

    item_frame = item_frame.reset_index()
    price_daily_parts.append(item_frame)

    series = item_frame.set_index("date")["price_causal"]
    observed_target = item_frame.set_index("date")["price_observed"]
    features = pd.DataFrame(index=calendar)
    features.index.name = "target_date"
    features["item"] = item
    features["forecast_origin"] = features.index - pd.Timedelta(days=1)
    features["y_true"] = observed_target.reindex(calendar)
    features["target_observed"] = features["y_true"].notna()
    for lag in [1, 2, 3, 7, 14, 30]:
        features[f"price_lag_{lag}"] = series.shift(lag)
    for window in [7, 14, 30]:
        history = series.shift(1).rolling(window, min_periods=window)
        features[f"price_roll_mean_{window}"] = history.mean()
        features[f"price_roll_std_{window}"] = history.std(ddof=0)
        features[f"price_roll_min_{window}"] = history.min()
        features[f"price_roll_max_{window}"] = history.max()
    prior_price = series.shift(1)
    features["prior_return_1"] = prior_price.pct_change(fill_method=None)
    features["target_dow_sin"] = np.sin(2 * np.pi * features.index.dayofweek / 7)
    features["target_dow_cos"] = np.cos(2 * np.pi * features.index.dayofweek / 7)
    features["target_month_sin"] = np.sin(2 * np.pi * features.index.month / 12)
    features["target_month_cos"] = np.cos(2 * np.pi * features.index.month / 12)
    feature_parts.append(features.reset_index())

price_daily = pd.concat(price_daily_parts, ignore_index=True)
price_features = pd.concat(feature_parts, ignore_index=True)

# The evaluation target dates must have observed outcomes for every commodity.
observed_pivot = daily_observed.pivot(index="date", columns="item", values="price_observed").sort_index()
common_observed_dates = observed_pivot.dropna().index
test_dates = common_observed_dates[-60:]
validation_dates = common_observed_dates[common_observed_dates < test_dates.min()][-30:]

date_calendar = pd.concat([
    pd.DataFrame({"split": "validation", "target_date": validation_dates}),
    pd.DataFrame({"split": "test", "target_date": test_dates}),
], ignore_index=True)
date_calendar["forecast_origin"] = date_calendar["target_date"] - pd.Timedelta(days=1)
# Conservative convention because the raw publication timestamps have no zone.
date_calendar["text_cutoff_day"] = date_calendar["forecast_origin"] - pd.Timedelta(days=1)
date_calendar["target_observed_all_items"] = True

item_calendar = date_calendar.merge(pd.DataFrame({"item": sorted(price["item"].unique())}), how="cross")
item_calendar = item_calendar.merge(
    daily_observed.rename(columns={"date": "target_date"})[["target_date", "item", "price_observed"]],
    on=["target_date", "item"], how="left", validate="many_to_one"
)
item_calendar["target_observed"] = item_calendar["price_observed"].notna()


# ---------------------------------------------------------------------------
# News preprocessing: preserve provenance, deduplicate, separate body quality
# ---------------------------------------------------------------------------
news_raw = pd.read_csv(NEWS_FILE, low_memory=False)
news_raw["published_ts_naive"] = pd.to_datetime(
    news_raw["published_date"], errors="coerce", format="mixed"
)
news_raw["published_day"] = news_raw["published_ts_naive"].dt.normalize()
news_raw["title_clean"] = news_raw["title"].fillna("").astype(str).str.strip()
news_raw["article_text_clean"] = news_raw["article_text"].fillna("").astype(str).str.strip()
news_raw["normalized_title"] = news_raw["title_clean"].map(normalize_text_key)
news_raw["canonical_publisher_url"] = news_raw["publisher_url"].fillna("").map(canonical_url)
news_raw["title_words"] = news_raw["title_clean"].str.split().str.len()
news_raw["body_words"] = news_raw["article_text_clean"].str.split().str.len()
body_lower = news_raw["article_text_clean"].str.casefold()
news_raw["placeholder_body"] = body_lower.isin(PLACEHOLDERS) | news_raw["article_text_clean"].eq("")

news_raw["document_type"] = np.select(
    [
        (~news_raw["placeholder_body"]) & (news_raw["body_words"] >= 5),
        news_raw["placeholder_body"] & (news_raw["title_words"] >= 5),
        (~news_raw["placeholder_body"]) & (news_raw["body_words"] < 5) & (news_raw["title_words"] >= 5),
    ],
    ["full_text", "title_only", "title_only"],
    default="unusable",
)
news_raw["quality_rank"] = news_raw["document_type"].map({"full_text": 2, "title_only": 1, "unusable": 0})
news_raw["model_text"] = np.where(
    news_raw["document_type"].eq("full_text"),
    news_raw["title_clean"] + ". " + news_raw["article_text_clean"],
    news_raw["title_clean"],
)

raw_columns = [
    "topic", "title", "published_date", "publisher", "publisher_site",
    "google_news_url", "publisher_url", "article_text", "extract_datetime",
]
news = news_raw.drop_duplicates(subset=raw_columns).copy()
news = news[news["published_day"].notna() & news["normalized_title"].ne("")].copy()
news["article_identity"] = (
    news["normalized_title"] + "|" + news["published_day"].dt.strftime("%Y-%m-%d")
)

topic_map = news.groupby("article_identity")["topic"].agg(
    lambda s: "|".join(sorted({str(v).strip() for v in s.dropna() if str(v).strip()}))
)
publisher_count = news.groupby("article_identity")["publisher"].nunique(dropna=True)
duplicate_group_size = news.groupby("article_identity").size()

news = news.sort_values(
    ["article_identity", "quality_rank", "body_words", "canonical_publisher_url"],
    ascending=[True, False, False, True],
)
deduplicated = news.drop_duplicates("article_identity", keep="first").copy()
deduplicated["source_query_topics"] = deduplicated["article_identity"].map(topic_map)
deduplicated["duplicate_group_size"] = deduplicated["article_identity"].map(duplicate_group_size).astype(int)
deduplicated["publishers_in_group"] = deduplicated["article_identity"].map(publisher_count).astype(int)
deduplicated["timestamp_timezone_status"] = "timezone_not_recorded"
deduplicated["language_processing_status"] = "not_recomputed_translation_artifact_unavailable"

news_columns = [
    "article_identity", "published_ts_naive", "published_day", "publisher",
    "canonical_publisher_url", "title_clean", "article_text_clean", "model_text",
    "document_type", "title_words", "body_words", "source_query_topics",
    "duplicate_group_size", "publishers_in_group", "timestamp_timezone_status",
    "language_processing_status",
]
news_primary = deduplicated[deduplicated["document_type"].eq("full_text")][news_columns].copy()
news_sensitivity = deduplicated[
    deduplicated["document_type"].isin(["full_text", "title_only"])
][news_columns].copy()

news_calendar = pd.DataFrame({
    "date": pd.date_range(news_raw["published_day"].min(), news_raw["published_day"].max(), freq="D")
})
raw_daily = news_raw.groupby("published_day").size().rename("raw_rows")
primary_daily = news_primary.groupby("published_day").size().rename("primary_full_text_articles")
sensitivity_daily = news_sensitivity.groupby("published_day").size().rename("sensitivity_articles")
title_only_daily = (
    news_sensitivity[news_sensitivity["document_type"].eq("title_only")]
    .groupby("published_day").size().rename("title_only_articles")
)
news_daily = news_calendar.set_index("date").join(
    [raw_daily, primary_daily, sensitivity_daily, title_only_daily]
).fillna(0).astype(int).reset_index()

date_calendar = date_calendar.merge(
    news_daily.rename(columns={"date": "text_cutoff_day"})[
        ["text_cutoff_day", "primary_full_text_articles", "sensitivity_articles", "title_only_articles"]
    ], on="text_cutoff_day", how="left", validate="many_to_one"
)
for column in ["primary_full_text_articles", "sensitivity_articles", "title_only_articles"]:
    date_calendar[column] = date_calendar[column].fillna(0).astype(int)


# ---------------------------------------------------------------------------
# Outputs and reproducibility checks
# ---------------------------------------------------------------------------
price_daily.to_csv(OUT / "price_daily_causal.csv", index=False)
price_features.to_csv(OUT / "price_features_causal.csv", index=False)
date_calendar.to_csv(OUT / "forecast_calendar_common_observed.csv", index=False)
item_calendar.to_csv(OUT / "forecast_item_calendar.csv", index=False)
news_primary.to_csv(OUT / "news_primary_full_text_pretranslation.csv", index=False)
news_sensitivity.to_csv(OUT / "news_sensitivity_with_title_only_pretranslation.csv", index=False)
news_daily.to_csv(OUT / "news_daily_availability.csv", index=False)

validation_exclusions = pd.date_range(validation_dates.min(), validation_dates.max()).difference(validation_dates)
test_exclusions = pd.date_range(test_dates.min(), test_dates.max()).difference(test_dates)

checks = {
    "eight_price_items": set(price_daily["item"].unique()) == TARGET_ITEMS,
    "one_row_per_item_date": not price_daily.duplicated(["item", "date"]).any(),
    "no_backward_fill_first_day": bool(price_daily.groupby("item").first()["price_causal"].notna().all()),
    "validation_dates": len(validation_dates) == 30,
    "test_dates": len(test_dates) == 60,
    "all_calendar_targets_observed": bool(item_calendar["target_observed"].all()),
    "primary_news_unique_identity": not news_primary["article_identity"].duplicated().any(),
    "sensitivity_news_unique_identity": not news_sensitivity["article_identity"].duplicated().any(),
}

manifest = {
    "status": "corrected_preprocessing_complete_pretranslation",
    "raw_files": {
        "price": {"name": PRICE_FILE.name, "sha256": file_sha256(PRICE_FILE), "rows": int(len(pd.read_csv(PRICE_FILE, usecols=["date"])))},
        "news": {"name": NEWS_FILE.name, "sha256": file_sha256(NEWS_FILE), "rows": int(len(news_raw))},
    },
    "price": {
        "items": sorted(price_daily["item"].unique().tolist()),
        "daily_rows": int(len(price_daily)),
        "missing_item_days": int(price_daily["was_missing"].sum()),
        "causal_outlier_flags": int(price_daily["causal_outlier_flag"].sum()),
        "primary_series": "national daily median with forward fill only; no outlier replacement",
        "sensitivity_series": "causal trailing-median replacement where |robust z| > 4",
    },
    "news": {
        "raw_rows": int(len(news_raw)),
        "exact_rows_after_deduplication": int(len(news)),
        "unique_title_day_articles": int(len(deduplicated)),
        "primary_full_text_articles": int(len(news_primary)),
        "sensitivity_articles": int(len(news_sensitivity)),
        "sensitivity_title_only_articles": int(news_sensitivity["document_type"].eq("title_only").sum()),
        "language_stage": "not executed because the translated intermediate file and translation dependencies were not provided",
        "timestamp_rule": "raw naive timestamp retained; conservative text_cutoff_day is one day before forecast_origin",
    },
    "evaluation": {
        "validation_observations": int(len(validation_dates)),
        "validation_span": [str(validation_dates.min().date()), str(validation_dates.max().date())],
        "validation_excluded_calendar_dates": [str(d.date()) for d in validation_exclusions],
        "test_observations": int(len(test_dates)),
        "test_span": [str(test_dates.min().date()), str(test_dates.max().date())],
        "test_excluded_calendar_dates": [str(d.date()) for d in test_exclusions],
    },
    "checks": checks,
    "remaining_before_modeling": [
        "Run deterministic language identification and translation using a frozen, recorded tool/version.",
        "Fit BERTopic only on news available before the validation period, then freeze it.",
        "Run ProsusAI/finbert on text that has not duplicated the title for title-only records.",
        "Construct lagged topic/sentiment features using text_cutoff_day.",
    ],
}

if not all(checks.values()):
    raise RuntimeError(f"Preprocessing validation failed: {checks}")

(OUT / "preprocessing_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
)

readme = f"""# Corrected preprocessing outputs

This directory was generated from the supplied price and Google News files without modifying them.

## Completed

- The eight manuscript commodities were selected using an explicit allow-list.
- National daily prices aggregated by median.
- Missing prices handled causally with forward fill only; no backward fill.
- Primary price series does not replace outliers. A trailing-only replacement is retained as a sensitivity series.
- Forecast calendars contain explicit `forecast_origin`, `target_date`, and conservative `text_cutoff_day` fields.
- Validation and test targets contain observed prices for all eight commodities.
- News records are separated into full-text and title-only corpora and deduplicated by normalized title and publication day.

## Evaluation windows

- Validation: 30 common observed dates, {validation_dates.min().date()} to {validation_dates.max().date()}; excluded calendar dates: {', '.join(str(d.date()) for d in validation_exclusions)}.
- Test: 60 common observed dates, {test_dates.min().date()} to {test_dates.max().date()}; excluded calendar dates: {', '.join(str(d.date()) for d in test_exclusions)}.

## Not yet executed

Language identification/translation, BERTopic, FinBERT, and forecasting are intentionally not claimed as complete. The current environment does not contain the translated intermediate file or the model dependencies/checkpoints. The CSV corpora are the verified inputs for those stages.

## Redistribution warning

The generated news-corpus CSV files contain publisher text. Do not place them in a public repository unless the relevant redistribution rights have been confirmed.
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")

print(json.dumps(manifest, indent=2, ensure_ascii=False))
