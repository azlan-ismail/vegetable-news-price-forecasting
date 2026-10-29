# Public-release checklist

- [x] Retain only the final conditional notebook.
- [x] Exclude article bodies, translations, representative passages, and
      headline-derived identifiers.
- [x] Include public-safe aggregate/model-ready data and exact reference
      results.
- [x] Parameterize local paths in the preprocessing script.
- [x] Validate lightweight result reproduction.
- [x] Remove manuscript source, build products, bibliography, template files,
      and manuscript-only figures from the repository and its reachable history.
- [x] Obtain agreement from all rights holders on code/document/data licences;
      replace `LICENSE.md` accordingly (MIT for code; CC BY 4.0 otherwise).
- [x] Decide whether the target journal permits repository access during
      review (manuscript is submitted with author names; no anonymized release).
- [x] Create the private GitHub repository:
      <https://github.com/azlan-ismail/vegetable-news-price-forecasting>.
- [x] Not applicable: review is not double-anonymous.
- [x] After resolving the licence and review-anonymity items above, change the
      repository visibility to public when scholarly release is appropriate.
- [ ] Create a tagged release (suggested: `v0.1.0`).
- [ ] Archive the release with Zenodo, then add the DOI to `CITATION.cff` and
      the README.
