# Evaluating News-Derived Signals for Vegetable Price Forecasting

- **Repository:** <https://github.com/azlan-ismail/vegetable-news-price-forecasting>
- **Status:** Private working repository pending scholarly release.

Reproducibility materials for the manuscript:

> **Incremental Predictive Utility of News-Derived Signals for Short-Horizon
> Vegetable Price Forecasting: A Coverage-Aware Conditional Evaluation**

The study tests whether BERTopic--FinBERT news features improve one-day-ahead
price forecasts for eight Malaysian vegetables. Its central comparison uses
matched pooled Ridge models, chronological model selection, training-only topic
fitting, and only target dates with retained full text dated before the forecast
origin.

## Main result

| Model | Pooled test RMSE |
|---|---:|
| Naive lag-1 | 0.4351 |
| Price-only Ridge | 0.5071 |
| Text-enhanced Ridge | 0.9055 |

The text-enhanced model's RMSE was 78.6% higher than that of the matched
price-only model. This is a conditional negative result for 45 irregular test
dates. It is not evidence about arbitrary daily forecasting or the causal effect
of news.

## Repository contents

| Path | Contents |
|---|---|
| `notebooks/conditional_text_rerun.ipynb` | Authoritative end-to-end Google Colab workflow |
| `src/preprocess.py` | Causal price and news preprocessing with explicit commodity selection |
| `src/reproduce_forecasting.py` | Lightweight reproduction of the matched Ridge comparison from released aggregate data |
| `src/make_publication_figures.py` | Regenerates the two result figures |
| `data/derived/` | Public-safe preprocessing outputs and aggregate/model-ready data |
| `results/` | Validation results, predictions, metrics, diagnostics, and figures |
| `environment/` | Minimal pinned dependencies for lightweight and full-pipeline runs |

## Quick reproduction from public-safe data

The following commands reproduce the forecasting comparison without requiring
publisher-owned news article text:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r environment/requirements-forecast.txt
python src/reproduce_forecasting.py --check-reference
python src/make_publication_figures.py
```

Successful verification writes outputs to `build/reproduced/` and reports
`"reference_verification": "passed"`.

## Full text-processing workflow

The full BERTopic--FinBERT workflow is in
`notebooks/conditional_text_rerun.ipynb`. It requires a GPU runtime and the
audited preprocessing package described in `data/README.md`.

1. Open the notebook in Google Colab and select a GPU runtime.
2. Run its setup cells and supply the authorized preprocessing package.
3. Inspect the training-period topic summary and representative samples.
4. Set `TOPIC_REVIEW_APPROVED = True` only after completing that review.
5. Run the remaining cells and compare the generated tables with `results/`.

The expected split is 30 training, 15 validation, and 45 test dates across eight
commodities, giving 360 held-out commodity--date observations.

## Reproducibility scope

The workflow fixes seed 42 and records the principal package versions. BERTopic
is fitted only through the final training cutoff and then frozen. Validation
selects the Ridge penalties; the test set is evaluated once. Bit-for-bit GPU
reproduction is not claimed because deterministic PyTorch algorithms were not
enabled and pretrained-model commit revisions were not pinned.

This repository supports exact reproduction of the forecasting and evaluation
stage from the included aggregate data. It does not provide a fully
self-contained rerun from raw news text because publisher article bodies and
translated text are not redistributed.

## Data availability

Vegetable prices originate from the Department of Statistics Malaysia (DOSM)
[PriceCatcher dataset](https://open.dosm.gov.my/data-catalogue/pricecatcher),
released under CC BY 4.0. The repository includes derived price series and cites
the source in `data/README.md`.

The news corpus was assembled from publicly accessible publisher pages through
Google News. Public accessibility does not establish redistribution rights.
Consequently, the repository excludes article bodies, translations,
representative text excerpts, and headline-derived identifiers. It provides
the preprocessing code, input hashes, conditional calendars, aggregate
features, predictions, metrics, and diagnostics.

## Citation

Until final publication metadata and a repository DOI are available, use the
metadata in `CITATION.cff`. After creating a GitHub release, archiving it with
Zenodo is recommended so the exact research compendium can receive a DOI.

## Release status

This private working repository is technically validated. Public release still
requires approval of the licences for the authors' materials, confirmation of
the target journal's anonymity requirements, creation of a versioned release,
and archival of that release with a persistent DOI. See
`RELEASE_CHECKLIST.md`.
