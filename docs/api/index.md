# API reference

The full API is auto-generated from docstrings by mkdocstrings. Browse
by module below.

## `afmkit.core`

- [`afmkit.core.curve`](curve.md) — `ForceCurve`, `CurveBatch`
  (xarray-backed, immutable).

## `afmkit.io`

- `afmkit.io.jpk_txt` — JPK 4-column `.txt` loader
  (`load_jpk_txt`, `JPKTxtLoader`); 1:1 with the original Igor
  `FXImport()`.
- `afmkit.io.igor_ibw` — Igor Binary Wave round-trip. Reader
  (`load_ibw`, `load_ibw_batch`, `IgorIBWLoader`) and writer
  (`save_ibw`) cover both v2 (default) and v5 (Igor Pro 6.00+)
  layouts. The v0.5+ reader re-hydrates every scalar `key=value`
  from the wave `note`; the `roundtrip_ibw(curve, path, *, version)`
  convenience function writes + reads + asserts the round-trip.
- `afmkit.io.hdf5_store` — native HDF5 storage (`save_hdf5`,
  `load_hdf5`); gzip-compressed, JSON-encoded metadata, ragged
  batches supported.
- `afmkit.io.exporters` — wide-column CSV, one-row-per-fit
  `to_csv_fits`, `to_mat` (v5), `to_parquet` (pyarrow /
  fastparquet), `to_markdown`, and `to_ibw` (via `save_ibw`).
  The `to_csv_fits` / `to_markdown` / `to_mat` / `to_parquet`
  functions accept `reviewers={curve_index: PeakReviewer}` to
  plumb the per-peak review state into the output.

## `afmkit.models`

- `afmkit.models.base` — `PolymerModel` protocol + `MODEL_REGISTRY`.
- `afmkit.models.wlc` — Marko-Siggia WLC (`WLCModel`); the default
  `"wlc"` registry entry; 1:1 with the Igor `LVFitWLC` formula.
- `afmkit.models.ewlc` — Wang 1997 / Odijk 1995 eWLC
  (`EWLCModel`); the `"ewlc"` registry entry; reduces to
  Marko-Siggia as K0 → ∞.
- `afmkit.models.fjc` — Freely Jointed Chain (`FJCModel`); the
  `"fjc"` registry entry; classical Padé [2,2] inverse Langevin
  approximation (b Kuhn length, Lc contour length). Registered
  both in `MODEL_REGISTRY` and as a pluggy entry point so it can
  be re-implemented out of tree.
- `afmkit.plugins` — pluggy hookspecs (`register_loader`,
  `register_model`, `register_baseline`, `register_fitter`).

## `afmkit.fitting`

- `afmkit.fitting.engine` — `Fitter` protocol, `LmfitEngine`
  implementation.
- `afmkit.fitting.report` — `FitResult` dataclass (best-fit params,
  standard errors, χ², reduced χ², covariance, residual array).

## `afmkit.analysis`

- `afmkit.analysis.peak_detection` — `find_sawtooth_peaks(curve,
  …)`; centered moving-average smoothing + `scipy.signal.find_peaks`
  with prominence / width thresholds; returns a list of `Peak`
  dataclasses.
- `afmkit.analysis.peak_review` — `PeakReviewer`,
  `ReviewedPeak`; accept / reject / override force / re-fit a
  single peak in a local window / attach a note. Plumbed through
  every exporter as of v0.4 / v0.5.

## `afmkit.presentation`

- `afmkit.presentation.cli` — the `afmkit` shell command (typer):
  `import` / `info` / `fit` / `export` / `gui`.
- `afmkit.presentation.gui` — the Textual TUI: `AFMkitApp`,
  `ForceExtensionPlot` (matplotlib-in-Textual widget), the
  peak-review modal.
