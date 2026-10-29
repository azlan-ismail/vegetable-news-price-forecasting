# Evaluating News-Derived Signals for Vegetable Price Forecasting

- **Repository:** <https://github.com/azlan-ismail/vegetable-news-price-forecasting>
- **Status:** Public research compendium accompanying a manuscript under review.
- **Licences:** MIT for code; CC BY 4.0 for derived data, figures, and documentation (see `LICENSE.md`).

Reproducibility materials for the manuscript:

> **Separating Information from Overfitting: A Placebo-Benchmarked Evaluation
> of News-Derived Signals for Short-Horizon Vegetable Price Forecasting**

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
price-only model. The diagnostic re-analysis shows that this loss is a cost of
the text block's dimensionality: news vectors permuted across dates raise RMSE
by a similar amount (median 73.6%), compact text representations change RMSE by
-1.2% to +7.3%, and bootstrap intervals bound any RMSE reduction from news at a
few per cent. Results concern 45 irregular, coverage-defined test dates.

## Repository contents

| Path | Contents |
|---|---|
| `notebooks/conditional_text_rerun.ipynb` | Authoritative end-to-end Google Colab workflow |
| `src/preprocess.py` | Causal price and news preprocessing with explicit commodity selection |
| `src/reproduce_forecasting.py` | Lightweight reproduction of the matched Ridge comparison from released aggregate data |
| `src/make_publication_figures.py` | Regenerates the two baseline result figures |
| `src/diagnostic_reanalysis.py` | Placebo text blocks, block/PCA ablation, return specification, covariate shift, expanding refits, bootstrap intervals |
| `src/clark_west.py` | Clark--West nested-model tests on the diagnostic predictions |
| `src/make_diagnostic_figures.py` | Regenerates the placebo and ablation figures |
| `data/derived/` | Public-safe preprocessing outputs and aggregate/model-ready data |
| `results/` | Validation results, predictions, metrics, diagnostics, and figures |
| `results/diagnostic_reanalysis/` | Reference outputs of the diagnostic re-analysis |
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

To run the diagnostic re-analysis (about 10 minutes on a laptop CPU):

```bash
python src/diagnostic_reanalysis.py      # writes build/diagnostics/
python src/clark_west.py
python src/make_diagnostic_figures.py
python src/check_diagnostics.py       # compares with results/diagnostic_reanalysis/
```

Reference outputs are in `results/diagnostic_reanalysis/`. The placebo draws
use seed 42; Ridge fits are deterministic.

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

The repository is public and technically validated. A versioned release
archived with Zenodo will add a persistent DOI. See `RELEASE_CHECKLIST.md`.
