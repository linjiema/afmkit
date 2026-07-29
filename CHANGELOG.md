# Changelog

All notable changes to **afmkit** are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
and the format is inspired by [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

Nothing yet. The next batch will land with v0.2 (GUI, sawtooth peak
detection, .ibw round-trip, eWLC / FJC plugin packages).

## [0.1.0] — 2026-07-30

First public release. Core library + CLI; no GUI yet.

**Install**

```bash
pip install "afmkit @ git+https://github.com/linjiema/afmkit.git@v0.1.0"
```

### Added

- **Core data model** — `ForceCurve` and `CurveBatch` (xarray-backed,
  immutable, full unit tests).
- **WLC model** — Marko-Siggia worm-like chain, 1:1 compatible with the
  original Igor `LVFitWLC` formula. Registered as the default
  `"wlc"` model in `MODEL_REGISTRY`.
- **JPK 4-column `.txt` loader** — `JPKTxtLoader` and `load_jpk_txt()`,
  bit-for-bit identical to the original Igor `FXImport()`. Handles
  optional text headers, k_cantilever in pN/nm, and returns a
  `CurveBatch` of two `ForceCurve`s (approach / retract).
- **HDF5 native store** — `HDF5Store`, `load_hdf5()`, `save_hdf5()`.
  gzip-compressed, JSON-encoded metadata, ragged batches supported.
- **Exporters** — `to_csv` (wide column-block), `to_csv_fits`
  (one-row-per-fit), `to_mat` (v5), `to_parquet` (pyarrow / fastparquet),
  `to_markdown` (human-readable report), and an `export()` dispatch
  helper that picks the format from `--format` or the file suffix.
- **Fitting engine** — `LmfitEngine` wraps `lmfit` and exposes a uniform
  `fit(curve, model=..., x_range=..., p0=...)` helper. Returns a
  `FitResult` with best-fit params, standard errors, χ², reduced χ²,
  covariance, and the data the fit was performed on.
- **CLI** — `afmkit version / info / import / fit / export` subcommands
  (typer + rich), end-to-end wired.
- **Plugin system** — pluggy hookspecs (`afmkit.plugins`) for loaders,
  models, baselines, and fitters. Model registry is the v0.1
  short-form; entry-point discovery lands in v0.2.
- **Dev toolchain** — `ruff`, `mypy` (non-strict in v0.1, strict by
  v0.2), `pytest`, `hypothesis`, `pre-commit`, GitHub Actions CI on
  macOS / Linux × Python 3.11 / 3.12 / 3.13, mkdocs-material docs.
- **Documentation** — quickstart, API reference, migration guide from
  Igor Pro, plugin authoring guide, team / agent workflow doc,
  end-to-end runnable example notebook (`examples/01_quickstart.ipynb`).

### Known limitations (v0.2 roadmap)

- Igor `.ibw` round-trip is **not** in v0.1 — read / write support is
  planned for v0.2 once the `igor` optional extra is unblocked on
  Python 3.13.
- eWLC and FJC models are plugin-only in v0.1; the registry is wired
  but no first-party implementations are shipped.
- Automated sawtooth peak detection is v0.2.
- GUI is v0.2.

[Unreleased]: https://github.com/linjiema/afmkit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/linjiema/afmkit/releases/tag/v0.1.0
