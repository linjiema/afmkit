# Changelog

All notable changes to **afmkit** are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
and the format is inspired by [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

Nothing yet. The next batch will land with v0.3 (peak-pick interactive
review, FJC plugin, GUI plot overlay).

## [0.2.0] — 2026-07-30

Four big pieces, each on its own feature branch and shipped together
once the full test matrix was green.

**Install**

```bash
pip install "afmkit @ git+https://github.com/linjiema/afmkit.git@v0.2.0"
```

To pull in the new optional extras:

```bash
pip install "afmkit[igor] @ ..."     # Igor .ibw round-trip
pip install "afmkit[gui]  @ ..."     # Textual TUI
pip install "afmkit[parquet] @ ..."  # pyarrow backend for to_parquet
```

### Added

- **Sawtooth peak detection** — `afmkit.analysis.find_sawtooth_peaks(curve, …)` returns a list of `Peak` dataclasses (extension, force, prominence, width, height_drop, confidence) for the retract sweep. Algorithm: centered moving-average smoothing → `scipy.signal.find_peaks` with prominence / width thresholds → per-peak confidence from the median prominence. Use this as the upstream of "how many unfolding steps did this molecule show, and at what forces".
- **Igor .ibw round-trip** — `afmkit.io.load_ibw`, `load_ibw_batch`, `save_ibw`, and the `IgorIBWLoader` class. The reader delegates to the optional `igor` PyPI package; the writer is a stdlib-only v2 binary emitter we wrote in-tree (the released `igor==0.3` ships a `save()` that raises `NotImplementedError`). The wave note carries an `afmkit=2col` marker so the reader can disambiguate. F/B files in the same folder are paired as approach/retract by basename.
- **Extensible WLC model** — `afmkit.models.EWLCModel` and the `"ewlc"` registry entry. The Wang 1997 / Odijk 1995 interpolation with the new `K0` (stretch modulus, pN) parameter; reduces to Marko-Siggia WLC as K0 → ∞ and is finite on the physical range [0, L] (the singularity lives at `x = L·(1 + 1/K0) > L` — strictly past the contour length, which is the practical advantage over WLC).
- **Textual TUI** — `afmkit gui` (or `python -m afmkit.presentation.gui.app`). Three-panel layout: directory input → data table → status line. Keybindings: `o` open directory, `f` fit selected curve with WLC, `e` export batch to CSV, `q` quit. No X server, no PySide6 — runs in any terminal over SSH.
- **Plugin system hardening** — `afmkit.models.register_model` and `MODEL_REGISTRY` are now the v0.2 first-class extension point. eWLC is registered as `"ewlc"` alongside the existing `"wlc"`. A pluggy entry-point (`afmkit-fjc`, etc.) for separately-installable plugin packages is documented but not yet exercised.

### Infrastructure

- Optional dependencies: `[igor]`, `[gui]`, `[parquet]`, `[plot]`, `[all]`.
- `afmkit.io.__init__` uses PEP 562 `__getattr__` to lazy-import the optional `.ibw` surface — `from afmkit.io import load_ibw` works but a minimal install (no `igor`) doesn't fail at package import.
- `pyproject.toml` pins `textual.*` and `afmkit.presentation.gui.*` with `disable_error_code = ["misc"]` to silence the mypy-version-dependent "Class cannot subclass value of type Any" false positive on the TUI base classes.
- 298 tests pass on every cell of the CI matrix (3 OS × 3 Python), up from 222 in v0.1.0. ~85 % line coverage on the core.

### Known limitations (v0.3 roadmap)

- Peak detection does not yet support manual review / re-fit of a single peak — that's a v0.3 interactive piece in the TUI.
- FJC model and `afmkit-fjc` standalone plugin package are v0.3.
- `.ibw` writer is v2 only (v5 in Igor supports 32-bit ints, dependency tracking, etc.). v2 covers everything afmkit produces; v5 support can come later if someone needs it.
- TUI is terminal-only — a real GUI (Qt / web) is still a future piece.

## [0.1.0] — 2026-07-30

First public release. Core library + CLI; no GUI yet.

**Install**

```bash
pip install "afmkit @ git+https://github.com/linjiema/afmkit.git@v0.1.0"
```

### Added

- **Core data model** — `ForceCurve` and `CurveBatch` (xarray-backed, immutable, full unit tests).
- **WLC model** — Marko-Siggia worm-like chain, 1:1 compatible with the original Igor `LVFitWLC` formula. Registered as the default `"wlc"` model in `MODEL_REGISTRY`.
- **JPK 4-column `.txt` loader** — `JPKTxtLoader` and `load_jpk_txt()`, bit-for-bit identical to the original Igor `FXImport()`. Handles optional text headers, k_cantilever in pN/nm, and returns a `CurveBatch` of two `ForceCurve`s (approach / retract).
- **HDF5 native store** — `HDF5Store`, `load_hdf5()`, `save_hdf5()`. gzip-compressed, JSON-encoded metadata, ragged batches supported.
- **Exporters** — `to_csv` (wide column-block), `to_csv_fits` (one-row-per-fit), `to_mat` (v5), `to_parquet` (pyarrow / fastparquet), `to_markdown` (human-readable report), and an `export()` dispatch helper that picks the format from `--format` or the file suffix.
- **Fitting engine** — `LmfitEngine` wraps `lmfit` and exposes a uniform `fit(curve, model=..., x_range=..., p0=...)` helper. Returns a `FitResult` with best-fit params, standard errors, χ², reduced χ², covariance, and the data the fit was performed on.
- **CLI** — `afmkit version / info / import / fit / export` subcommands (typer + rich), end-to-end wired.
- **Plugin system** — pluggy hookspecs (`afmkit.plugins`) for loaders, models, baselines, and fitters. Model registry is the v0.1 short-form; entry-point discovery lands in v0.2.
- **Dev toolchain** — `ruff`, `mypy` (non-strict in v0.1, strict by v0.2), `pytest`, `hypothesis`, `pre-commit`, GitHub Actions CI on macOS / Linux × Python 3.11 / 3.12 / 3.13, mkdocs-material docs.
- **Documentation** — quickstart, API reference, migration guide from Igor Pro, plugin authoring guide, team / agent workflow doc, end-to-end runnable example notebook (`examples/01_quickstart.ipynb`).

### Known limitations (v0.2 roadmap)

- Igor `.ibw` round-trip — read / write support was planned for v0.2 once the `igor` optional extra is unblocked on Python 3.13.
- eWLC and FJC models are plugin-only in v0.1; the registry is wired but no first-party implementations are shipped.
- Automated sawtooth peak detection is v0.2.
- GUI is v0.2.

[Unreleased]: https://github.com/linjiema/afmkit/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/linjiema/afmkit/releases/tag/v0.2.0
[0.1.0]: https://github.com/linjiema/afmkit/releases/tag/v0.1.0
