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
- [ ] Obtain agreement from all rights holders on code/document/data licences;
      replace `LICENSE.md` accordingly.
- [ ] Decide whether the target journal permits repository access during
      double-anonymous review; if needed, prepare an anonymized release.
- [x] Create the private GitHub repository:
      <https://github.com/azlan-ismail/vegetable-news-price-forecasting>.
- [ ] If repository materials are supplied during double-anonymous review, use
      the journal's approved anonymized reviewer-access route.
- [ ] After resolving the licence and review-anonymity items above, change the
      repository visibility to public when scholarly release is appropriate.
- [ ] Create a tagged release (suggested: `v0.1.0`).
- [ ] Archive the release with Zenodo, then add the DOI to `CITATION.cff` and
      the README.
