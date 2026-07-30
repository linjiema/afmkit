# Migration from Igor Pro

This page maps the original `FX_Analysis_NJU_20110330.ipf` and
`Load_JPK_FX_Data_20110514.ipf` workflow to afmkit. If you have been
running the Igor scripts for years, this is your starting point —
conventions, units, the WLC formula, and the directory layout are all
preserved 1:1.

!!! tip "v0.1 scope"
    v0.1 ships the data model, the JPK loader, the WLC model, the WLC
    fitting engine, and the exporters. Everything that touches the GUI
    (peak picking panel, cursor handling) and the `.ibw` round-trip
    lives in v0.2. See the [status table](#status) at the bottom for
    the per-feature state.

## Loading data

| Igor (Load_JPK_FX_Data) | afmkit | Typical use case |
|---|---|---|
| `FXImport()` — pop a folder picker, read a 4-column `.txt` | `afmkit.io.jpk_txt.load_jpk_txt(path, k_cantilever=0.1)` | Loading a single approach/retract pair. |
| `RefoldingImport()` — multi-cycle refolding data, slice 4000–7999 | not in v0.1 — use `load_jpk_txt` per file and slice with `curve.select_range(50, 200)` | Folded / refolded constructs. The Igor code hard-codes the row range `[4000, 7999]`; the afmkit equivalent is an extension-based window. |
| Batch directory ingest (e.g. `filelist = IndexedFile(root, ...)` loop in `FXImport`) | glob the directory, call `load_jpk_txt` per file, concatenate into a `CurveBatch`:<br>`CurveBatch(chain.from_iterable(load_jpk_txt(p, k_cantilever=k) for p in Path("data").glob("*.txt")))` | One folder = one batch. |
| `IgorImport()` — re-use already-loaded Igor waves | **not in v0.1.** Re-export the waves to CSV/Tab-Separated from Igor, then wrap each in `ForceCurve`. The native `.ibw` reader is v0.2. | Translating existing in-memory Igor data. |
| Stored as `Force_F{n}`, `Extension_F{n}`, `Force_B{n}`, `Extension_B{n}` | `CurveBatch` of `ForceCurve` objects with `metadata` dict | In-memory representation. |

!!! note "Python import paths"
    `load_jpk_txt` and `JPKTxtLoader` are **not** re-exported from
    `afmkit.io.__init__` in v0.1 — import them from
    `afmkit.io.jpk_txt` directly. The top-level `import afmkit` only
    exposes the version string.

The unit conversions (`1e12` N→pN, `1e9` m→nm, `-F/k` cantilever
correction) and the per-direction baseline subtraction (mean of the
first 200 force points, last-point extension subtraction) are
preserved 1:1 — see [`afmkit/io/jpk_txt.py`][jpk-py] for the
reference implementation.

[jpk-py]: https://github.com/linjiema/afmkit/blob/main/src/afmkit/io/jpk_txt.py

```python
from pathlib import Path
import itertools as it
from afmkit.io.jpk_txt import load_jpk_txt
from afmkit.core.curve import CurveBatch

batches = [
    load_jpk_txt(p, k_cantilever=0.06) for p in sorted(Path("data").glob("*.txt"))
]
batch = CurveBatch(
    it.chain.from_iterable(batches),
    name="data",
    metadata={"k_cantilever": 0.06, "source": "data/"},
)
print(batch)  # CurveBatch('data' n_curves=2 * len(files))
```

## Before / after: one full workflow

Below is a side-by-side of the same lab session in the old Igor macros
and in afmkit. The order is the same; only the surface syntax changes.

### Igor (FX_Analysis_NJU + Load_JPK_FX_Data)

```igorpro
// 1. Load a folder of JPK .txt files (loads Force_F{n}, Extension_F{n}, ...).
NVAR spring_constant = root:Data:spring_constant          // prompted by FXImport
FXImport()                                                // folder picker + ingest
// → waves root:Data:Force_F{0..N-1}, Extension_F{0..N-1}, Force_B{n}, Extension_B{n}

// 2. Open the analysis panel and set the L range from cursor A.
Force_Extension_Analysis()
NVAR f = root:Fits:G_SetLtoCursor                          // 0 = free, 2 = cursor A..B
Cursor /A /W=Force_Extension $("Extension_F0") 50          // cursor A at 50 nm
Cursor /B /W=Force_Extension $("Extension_F0") 180         // cursor B at 180 nm

// 3. Fit one curve, then run the auto-pick across the batch.
FitToCursor()                                              // fit curve 0 in [50, 180]
Make/O/N=(N) WLC_p_fit=WLC_p, WLC_L_fit=WLC_L              // allocate result waves
for(i=0; i<N; i+=1)
    WLC_p_fit[i] = WLCurves_P(i)                           // fit each curve
endfor

// 4. Export selected waves to .txt for plotting in Origin.
FX_Export()                                                // dump current panel waves
```

### afmkit (Python)

```python
import numpy as np
from afmkit.io.jpk_txt import load_jpk_txt
from afmkit.fitting import fit
from afmkit.io.exporters import to_csv

# 1. Load a folder of JPK .txt files.
# → CurveBatch of ForceCurve, each with metadata={"source_file", "k_cantilever", "direction"}.
batch = load_jpk_txt("data/curve_001.txt", k_cantilever=0.06)
approach = batch[0]  # first direction="approach" curve
print(
    approach
)  # ForceCurve(n_points=2000, ext=[0.0, 199.5] nm, F=[0.1, 65.4] pN, k=0.06)

# 2. The "cursor A to B as fit range" is now an extension window.
#    Pick a 130 nm window centred on the elastic regime, away from the
#    WLC divergence near x = L and the contact region near x = 0.
x_range = (20.0, 180.0)

# 3. Fit one curve, then run the same fit across the batch.
result = fit(approach, model="wlc", x_range=x_range)
print(result.summary())
# FitResult[WLCModel]
# ------------------
#   status            : ok
#   parameters:
#     p = 0.42  +/- 0.003
#     L = 198.7  +/- 0.5

fits = [
    fit(curve, model="wlc", x_range=x_range)
    for curve in batch
    if curve.metadata["direction"] == "approach"
]

# 4. Export the curves. CSV wide-format drops straight into Origin /
#    Matlab / Excel. (to_csv_fits / to_markdown need an extra bridge
#    step today — see the gotcha at the bottom of this page.)
to_csv(batch, "curves.csv")
```

What changed, line by line:

1. **Folder pick → function call.** `FXImport()` pops a dialog;
   `load_jpk_txt(path, k_cantilever=0.06)` reads a single file.
   Loop over a glob to replicate the batch behaviour.
2. **Waves → objects.** Igor uses 4 separate global waves
   (`Force_F0`, `Extension_F0`, `Force_B0`, `Extension_B0`); afmkit
   packages each direction into a single `ForceCurve` and the whole
   session into a `CurveBatch`. Metadata travels with the curve.
3. **Cursor A/B → `x_range=(min, max)`.** No more `G_SetLtoCursor`:
   pass the extension window directly to `fit()`. The same heuristic
   (skip the near-`L` divergence and the contact region) applies;
   see the [WLC fit range](#tuning-the-wlc-fit) section.
4. **`FitToCursor` → `afmkit.fit`.** Same Marko-Siggia formula,
   same `Levenberg-Marquardt` solver, same default bounds. The result
   is a structured `FitResult` dataclass — `result.params["p"]`,
   `result.redchi`, `result.r_squared`, etc. — instead of waves
   stuffed into a data folder.
5. **`FX_Export` → `to_csv` / `to_csv_fits` / `to_mat` / `to_parquet` /
   `to_markdown`.** Each curve is a column pair
   (`ext_000`, `force_000`, `ext_001`, `force_001`, …), so the
   leading block of the CSV is directly plottable in Origin, Excel,
   and Matlab without any post-processing. Use `to_mat` for the
   round-trip into a Matlab script that already has the rest of your
   data.

## From waves to `ForceCurve`

The old Igor code stored each curve as a separate wave:

```
root:Data:Extension_F0     (wave of 5000 doubles, units = "nm")
root:Data:Force_F0         (wave of 5000 doubles, units = "pN")
```

afmkit packages these together with metadata in a single object:

```python
from afmkit.core.curve import ForceCurve

curve = ForceCurve(
    extension=extension_array,  # nm
    force=force_array,  # pN
    metadata={
        "k_cantilever": 0.1,  # pN/nm
        "temperature": 298.0,  # K
        "source_file": "trace_001.txt",
        "direction": "approach",  # or "retract"
        "sampling_rate_hz": 5000.0,
    },
)
```

A [`CurveBatch`][CurveBatch] is then an ordered collection of
`ForceCurve` objects, backed internally by `xarray.Dataset` for fast
slicing, masking, and HDF5 round-trip.

[ForceCurve]: api/curve.md
[CurveBatch]: api/curve.md

## Units and sign conventions

The loaders in afmkit are calibrated to be bit-for-bit compatible with
the original Igor macros, but the unit conversions are now done
**once, in one place** — the loader. The rest of the library stays in
pN / nm.

| Quantity | Igor waves | afmkit | Conversion |
|---|---|---|---|
| Deflection (input column) | newtons (N) | piconewtons (pN) | `× 1e12` |
| Piezo position (input column) | metres (m) | nanometres (nm) | `× 1e9` |
| Cantilever-corrected extension | nm | nm | `ext_nm = z*1e9 - F_pN / k` |
| Force baseline | mean of first 200 points | mean of first 200 points (same) | identical, see below |
| Extension baseline | last point of the trace | last point of the trace (same) | identical, see below |
| Sign of force | positive = repulsive | positive = repulsive | identical (Igor stores `-F` to get this; afmkit applies the same sign flip in `_convert_one`) |
| Sign of extension | increasing during approach | increasing during approach | identical |

**The two baseline corrections are the most common source of double-
or zero-correction bugs.** Both are applied *automatically* by
`afmkit.io.jpk_txt.load_jpk_txt`, so the resulting `ForceCurve` is
already baseline-subtracted:

- The force mean is computed from the **first 200 points** of the
  trace (`force_pn -= force_pn[:200].mean()`). This mirrors
  `wavestats/Q/R=[0,199] Force_for_old` in the original `FXImport()`.
- The extension is shifted so that the **last point of the trace
  sits at zero** (`ext_nm -= ext_nm[-1]`). This is the
  `Ext_for_old=Ext_for_old-Ef_ini` line.

If you have been manually subtracting a baseline in your Igor script
on top of `FXImport`, you must drop that step when you move to afmkit
— otherwise you will be subtracting the baseline twice and your
curves will be shifted.

```python
# DO NOT do this — the loader has already baseline-subtracted.
mean = curve.force[:200].mean()
corrected = curve.with_force(curve.force - mean)
```

For refolding / multi-cycle data the original code re-uses the same
heuristic on a different point range (`V_npnts-201` to
`V_npnts-1`). v0.1 does not have a dedicated refolding loader; load
each cycle with `load_jpk_txt` and either pre-trim the input or call
`curve.select_range(50, 200)` to crop the cycle you want.

## Tuning the WLC fit

The Marko-Siggia WLC implemented in afmkit is bit-for-bit identical to
the Igor `LVFitWLC`:

```python
def wlc(x, p, L):
    return (4.1 / p) * (0.25 * (1 - x / L) ** -2 - 0.25 + x / L)
```

The hard-coded `kB*T = 4.1 pN·nm` matches the lab's existing Igor
fits; do not "correct" it to 4.047 or 4.11 without re-running the
full legacy-vs-new equivalence test suite.

If your existing Igor fits used `G_SetLtoCursor = 2` (cursor A to B
as fit range), the afmkit equivalent is:

```python
from afmkit.fitting import fit

result = fit(
    curve,
    model="wlc",
    x_range=(curve.extension[idx_a], curve.extension[idx_b]),
    p0={"p": 0.4, "L": curve.extension[-1]},
)
```

!!! warning "Fit range must be interior to the WLC regime"
    The Marko-Siggia formula has a `(1 - x/L)^-2` divergence at
    `x = L` and is dominated by the contact region at `x ≈ 0`. A fit
    that includes either end will either NaN out or be pulled off by
    data the model was never meant to describe.

    For a typical 200 nm Lc curve, a safe interior range is
    **`x_range=(20, 180)`** — i.e. trim ~10 % off each end. For a
    100 nm Lc curve use `(10, 90)`. The fitter will refuse a range
    that selects no data (`ValueError("x_range ... selects no data
    points")`).

If `p0` is omitted, afmkit uses `WLCModel.guess_params(x, y)`, which
sets `L = max(x) * 1.1` and `p = 0.4` — a robust default for
unfolded protein data, but freely movable by the fitter.

The result is a [`FitResult`][FitResult] dataclass with best-fit
parameters, standard errors, the full covariance matrix, chi-square,
reduced chi-square, AIC, BIC, R², the residual array, and the data
the fit was performed on. `result.summary()` prints a multi-line
human-readable dump; `result.to_dict()` / `to_json()` serialise it
for HDF5 storage.

[FitResult]: api/curve.md

## Reading the new code

The rest of this guide assumes you are comfortable with the data
model and the fit API. If you want a single end-to-end reference,
`examples/01_quickstart.ipynb` is the canonical notebook — the
authors of afmkit run it as the smoke test on every release.

!!! note "Notebook status"
    The quickstart notebook is checked in alongside the v0.1.0
    release. If you are running off a development checkout and the
    file is missing locally, the `01_quickstart.ipynb` source of
    truth is in the GitHub repo; pull it before following along.

The notebook walks through the same five steps as the [before/after
example](#before-after-one-full-workflow) above, one cell at a time:

1. **Load** — `load_jpk_txt` on a single file, then a glob over a
   folder.
2. **Slice** — `curve.select_range(50, 200)` to crop the elastic
   regime, with the same `[50, 200]` heuristic the lab has been
   using in Igor for years.
3. **Fit** — `fit(curve, model="wlc", x_range=(20, 180))` with the
   interior window.
4. **Batch** — list comprehension over the batch, dropping failed
   fits (`result.metadata["success"] is False`).
5. **Export** — `to_csv` for the data, `to_csv_fits` for the per-fit
   results, `to_markdown` for the human-readable summary.

For deeper dives:

- [`docs/quickstart.md`](quickstart.md) — install, the
  `[all]` extra, and the `afmkit` shell command.
- [`docs/api/curve.md`](api/curve.md) — full `ForceCurve` /
  `CurveBatch` API reference, including `to_xarray` and
  `from_xarray` for the power-user xarray interop.
- [`docs/api/index.md`](api/index.md) — auto-generated reference for
  every public module.
- [`CONTRIBUTING.md`](https://github.com/linjiema/afmkit/blob/main/CONTRIBUTING.md)
  — for writing your own polymer model or loader as a plugin.

## Common gotchas

A short list of "this would have eaten an afternoon of debugging"
moments, with the fix for each.

### 1. The `.ibw` round-trip is **not** in v0.1

`afmkit.io.read_ibw` does not exist in v0.1. The native Igor Binary
Wave reader is v0.2, pending the `igor` optional extra being
unblocked on Python 3.13.

If you have a folder of `.ibw` files you want to load today, the
fastest workaround is to export them from Igor as plain text and
wrap each pair of columns in a `ForceCurve`:

```python
import numpy as np
from afmkit.core.curve import CurveBatch, ForceCurve

ext = np.loadtxt("trace_001.dat", usecols=0)  # adjust to your export
frc = np.loadtxt("trace_001.dat", usecols=1)
curve = ForceCurve(ext, frc, metadata={"source_file": "trace_001.dat"})
batch = CurveBatch([curve], name="trace_001")
```

The unit conversion that the JPK loader normally does for you
(`1e12`, `1e9`, `-F/k`) is **not** applied here — you have to do it
yourself, since the `.ibw` files store whatever units your Igor
script chose.

### 2. eWLC and FJC are plugin-only in v0.1

The `MODEL_REGISTRY` ships with exactly one entry, `"wlc"`. eWLC
(Wang 1997), FJC, and twist-WLC are first-class extension points
— see [`src/afmkit/plugins.py`][plugins-py] — but no first-party
implementations are shipped. To use eWLC, write a class that
satisfies the [`PolymerModel`][PolymerModel] protocol and call
`afmkit.models.register_model("ewlc", EwlcModel)` once at import
time. The fitter will pick it up via `model="ewlc"`.

[plugins-py]: https://github.com/linjiema/afmkit/blob/main/src/afmkit/plugins.py
[PolymerModel]: api/curve.md

### 3. The CLI `import` requires `--k`

JPK `.txt` exports do not store the cantilever spring constant —
the file format predates the metadata convention. `load_jpk_txt`
therefore requires `k_cantilever` as a keyword argument, and the
`afmkit import` CLI subcommand requires `--k`:

```bash
afmkit import data/*.txt -o curves.h5 --k 0.06
```

`afmkit import` and the other CLI subcommands (`info`, `fit`,
`export`) are **stubs in v0.1** — they print a friendly "coming
with v0.1" message and exit. Use the Python API in the meantime.

### 4. The fit `x_range` should be **interior** of the curve

Already covered in the [WLC tuning](#tuning-the-wlc-fit) section
above; restating because it is the most common first-attempt
failure mode. The Marko-Siggia formula diverges at `x = L` and is
dominated by the contact region at `x ≈ 0`. For a 200 nm Lc curve,
use `x_range=(20, 180)`. For other Lc, scale proportionally: 10 %
off each end is a reasonable starting point.

### 5. Don't double-correct the baseline

Also covered above; restating because it is the second most common
bug. The loader applies the `[0:200]` mean force correction and the
`ext[-1]` extension correction for you. Do not re-apply them in
user code; do not pass already-baselined waves to the loader.

### 6. `afmkit.fit` swallows solver errors

The fitter never raises on a non-converging fit — a NaN in the
data, a wildly wrong starting point, anything. Instead it returns
a `FitResult` with `metadata["success"] = False` and
`metadata["message"]` set to the lmfit diagnostic. In a 100-curve
batch fit, raising would lose the (good) fits that came before;
swallowing is deliberate. Check `result.metadata["success"]` (or
the `model` / `nfev` fields) in your batch loop:

```python
fits = [fit(c, model="wlc", x_range=(20, 180)) for c in batch]
ok = [r for r in fits if r.metadata["success"]]
print(f"{len(ok)}/{len(fits)} fits converged")
```

### 7. There are two `FitResult` classes in v0.1 (bridge required)

`afmkit.fitting.fit()` returns `afmkit.fitting.report.FitResult`
with fields named `chi_square`, `reduced_chi_square`, `x_fit`,
`y_fit`. The exporters `to_csv_fits` and `to_markdown` accept the
**separate** `afmkit.io.exporters.FitResult` (fields named `chi2`,
`redchi`, `x`, `y`). They are not the same class and not
subclasses of each other; passing the fitting result directly
raises `AttributeError: 'FitResult' object has no attribute 'chi2'`.

Until the two are unified, bridge across with one explicit copy:

```python
from afmkit.fitting import fit
from afmkit.io.exporters import FitResult, to_csv_fits, to_markdown


# `fit_result` is afmkit.fitting.FitResult; `exported` is the
# afmkit.io.exporters.FitResult that the exporters want.
def to_exporters(fit_result):
    return FitResult(
        model_name=fit_result.model_name,
        params=fit_result.params,
        param_stderr=fit_result.stderr,
        chi2=fit_result.chi_square,
        redchi=fit_result.reduced_chi_square,
        n_data=fit_result.n_data,
        covariance=fit_result.covariance,
        x=fit_result.x_fit,
        y=fit_result.y_fit,
    )


fits = [fit(c, model="wlc", x_range=(20, 180)) for c in batch]
to_csv_fits([to_exporters(r) for r in fits], "fits.csv")
to_markdown(batch, [to_exporters(r) for r in fits], "report.md")
```

This is a known v0.1 wart, tracked for a v0.2 cleanup.

## Status

Per-feature status of the v0.1.0 release, sourced from the code
(grep for `TODO`, `v0.2`, `deferred` in `src/afmkit/`) and the
`CHANGELOG.md`. Features not listed here are either already
documented above or are part of the core data model (which is fully
shipped and tested).

| Feature | Status | Where |
|---|---|---|
| `ForceCurve` / `CurveBatch` data model | **v0.1 (shipped)** | `src/afmkit/core/curve.py` |
| JPK 4-column `.txt` loader (`load_jpk_txt`) | **v0.1 (shipped)** | `src/afmkit/io/jpk_txt.py` |
| HDF5 store (`HDF5Store`, `save_hdf5`, `load_hdf5`) | **v0.1 (shipped)** | `src/afmkit/io/hdf5_store.py` |
| WLC model (`WLCModel`) + `LmfitEngine` | **v0.1 (shipped)** | `src/afmkit/models/wlc.py`, `src/afmkit/fitting/engine.py` |
| Exporters: CSV (wide), `csv_fits`, Matlab v5, Parquet, Markdown | **v0.1 (shipped)** | `src/afmkit/io/exporters.py` |
| Matlab v7.3 (HDF5 backend) | not planned (v5 is portable enough; document if requested) | docstring TODO in `exporters.py:33` |
| Igor `.ibw` reader & writer | v0.2 (blocked on the `igor` extra on Python 3.13) | `CHANGELOG.md:57` |
| eWLC, FJC, twist-WLC models | v0.2 (plugin only — registry is wired in v0.1) | `src/afmkit/models/__init__.py`, `CHANGELOG.md:60` |
| Automated sawtooth peak detection | v0.2 | `CHANGELOG.md:62` |
| Cursor / baseline / Savitzky-Golay smoothing (analysis panel) | v0.2 (the `afmkit.processing` package is a docstring placeholder in v0.1) | `src/afmkit/processing/__init__.py` |
| PyQt6 GUI (parity with `FX_Analysis` panel) | v0.2 | `CHANGELOG.md:63` |
| CLI subcommands (`afmkit import`, `fit`, `export`, `info`) | v0.1 stubs (print a friendly message) — wired end-to-end in v0.2 | `src/afmkit/presentation/cli.py` |
| Refolding multi-cycle loader | not planned (load each cycle with `load_jpk_txt` and slice) | — |

If there is a specific Igor macro you cannot live without, please
[open an issue](https://github.com/linjiema/afmkit/issues).
