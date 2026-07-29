# Migration from Igor Pro

> **Status: draft.** This page is being expanded as the v0.1 API stabilizes.

This page maps the original `FX_Analysis_NJU_20110330.ipf` and
`Load_JPK_FX_Data_20110514.ipf` workflow to afmkit equivalents. If you
have been running the Igor scripts for years, this is your starting point.

## Loading data

| Igor (Load_JPK_FX_Data) | afmkit |
|---|---|
| `FXImport()` — pop a folder picker, read 4-column `.txt` | `afmkit.load_jpk_txt(glob, k_cantilever=0.1)` |
| `RefoldingImport()` — multi-cycle refolding data | `afmkit.load_jpk_refolding(glob, k_cantilever=0.1)` |
| `IgorImport()` — re-use already-loaded Igor waves | `afmkit.io.read_ibw(path)` |
| Stored as `Force_F{n}`, `Extension_F{n}`, `Force_B{n}`, `Extension_B{n}` | `CurveBatch` of `ForceCurve` objects with metadata |

The unit conversions (`1e12` N→pN, `1e9` m→nm, `-F/k` cantilever
correction) and the end-of-trace baseline subtraction are preserved
1:1 — see `afmkit.io.jpk_txt` for the reference implementation.

## Reading legacy `.ibw` data

```python
import afmkit

batch = afmkit.io.read_ibw("legacy_experiment.ibw")
print(batch)
```

## The analysis panel

| Igor `FX_Analysis` macro | afmkit equivalent |
|---|---|
| `Force_Extension_Analysis()` — the main panel | GUI in v0.2; until then, the `analysis` module |
| `WLCurves(p, L, ΔL, n)` — draw n WLC curves | `afmkit.models.WLCModel().plot(curve, ...)` |
| `FitToCursor()` — read L from cursor A | `afmkit.fit(curve, model="wlc", x_range=...)` |
| `AutoFindForcePeaks` | `afmkit.analysis.auto_peaks.find_sawtooth_peaks(curve)` (v0.2) |
| `Enter` / `EnterAllDataPoints` | `afmkit.exporters.to_csv(results)` |
| `FX_Export` | `afmkit.exporters.to_hdf5(session, path)` or `to_csv` / `to_mat` / `to_ibw` |
| `Addpeak` / `Deletepeak` | Manual review in GUI (v0.2) |

## From waves to `ForceCurve`

The old Igor code stored each curve as a separate wave:

```
root:Data:Extension_F0     (wave of 5000 doubles, units = "nm")
root:Data:Force_F0         (wave of 5000 doubles, units = "pN")
```

afmkit packages these together with metadata in a single object:

```python
curve = ForceCurve(
    extension=extension_array,    # nm
    force=force_array,            # pN
    metadata={
        "k_cantilever": 0.1,      # pN/nm
        "temperature": 298.0,     # K
        "source_file": "trace_001.txt",
        "direction": "approach",  # or "retract"
        "sampling_rate_hz": 5000.0,
    },
)
```

A `CurveBatch` is then an ordered collection of `ForceCurve` objects,
backed internally by `xarray.Dataset` for fast slicing, masking, and
HDF5 round-trip.

## Tuning the WLC fit

The Marko-Siggia WLC implemented in afmkit is bit-for-bit identical to
the Igor `LVFitWLC`:

```python
def wlc(x, p, L):
    return (4.1 / p) * (0.25 * (1 - x / L) ** -2 - 0.25 + x / L)
```

If your existing Igor fits used `G_SetLtoCursor = 2` (cursor A to B as
fit range), the afmkit equivalent is:

```python
result = afmkit.fit(
    curve,
    model="wlc",
    x_range=(curve.extension[idx_a], curve.extension[idx_b]),
    p0={"p": 0.4, "L": curve.extension[-1]},
)
```

## Coming next

The migration table above will grow as we add features. If there is a
specific Igor macro you cannot live without, please open an issue.
