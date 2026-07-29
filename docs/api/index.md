# API reference

The full API is auto-generated from docstrings by mkdocstrings. Browse by
module below.

## `afmkit.core`

- [`afmkit.core.curve`](curve.md) — `ForceCurve`, `CurveBatch`
- `afmkit.core.session` — `Session` (analysis context)
- `afmkit.core.types` — shared Protocols and TypedDicts
- `afmkit.core.units` — optional pint-based unit handling

## `afmkit.io`

- `afmkit.io.base` — `Loader` protocol
- `afmkit.io.jpk_txt` — JPK 4-column `.txt` loader
- `afmkit.io.igor_ibw` — legacy Igor Binary Wave loader
- `afmkit.io.hdf5_store` — native HDF5 storage
- `afmkit.io.exporters` — CSV, Matlab, Parquet, Markdown, IBW

## `afmkit.models`

- `afmkit.models.base` — `PolymerModel` protocol
- `afmkit.models.wlc` — Marko-Siggia WLC

## `afmkit.fitting`

- `afmkit.fitting.engine` — `Fitter`, `LmfitEngine`
- `afmkit.fitting.report` — `FitResult` dataclass

## `afmkit.processing`

- `afmkit.processing.smooth`
- `afmkit.processing.baseline`
- `afmkit.processing.peaks`
- `afmkit.processing.calibration`

## `afmkit.analysis`

- `afmkit.analysis.single`
- `afmkit.analysis.batch`
- `afmkit.analysis.auto_peaks`
- `afmkit.analysis.statistics`

## `afmkit.presentation`

- `afmkit.presentation.cli` — `afmkit` command (typer)
