# Changelog

## 0.2.0 — 2026-08-16

- Added text and embedding-native reranking through `IMRNNAdapter`.
- Added Moore–Penrose vocabulary-concept explanations.
- Standardized one query/document modulation architecture with a square
  identity-initialized projector, 128 hidden units, and zero dropout.
- Standardized improvement-margin training with dense top-100 hard negatives,
  Adam optimization, deterministic splits, and validation-only early stopping.
- Added strict base-versus-adapted evaluation and complete checkpoint metadata.
- Added the validated MiniLM–SciFact checkpoint and three-dataset validation
  report.
- Added local BEIR dataset support, tests, CI, packaging, and release tooling.
- Licensed the distribution under CC BY 4.0 with an attribution notice for the
  authors.
