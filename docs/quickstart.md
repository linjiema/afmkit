# Quickstart

> **Status: shipped in v0.5.0.** Every example on this page runs
> against the current release. If you find a discrepancy, please
> [open an issue](https://github.com/linjiema/afmkit/issues).

## Install

```bash
pip install "afmkit @ git+https://github.com/linjiema/afmkit.git@v0.5.0"
```

Or pull in everything an end-user might want (JPK loader, HDF5
storage, Igor `.ibw` round-trip, matplotlib plot panel, Textual
TUI, pyarrow Parquet):

```bash
pip install "afmkit[all] @ git+https://github.com/linjiema/afmkit.git@v0.5.0"
```

### Optional extras

| Extra | What it adds | Install example |
|---|---|---|
| `io` | HDF5 storage backend | `pip install "afmkit[io] @ ..."` |
| `igor` | Read / write legacy `.ibw` files (v2 + v5) | `pip install "afmkit[igor] @ ..."` |
| `matlab` | Matlab `.mat` round-trip (v5) | `pip install "afmkit[matlab] @ ..."` |
| `parquet` | pyarrow-backed Parquet export | `pip install "afmkit[parquet] @ ..."` |
| `plot` | Static matplotlib paper figures | `pip install "afmkit[plot] @ ..."` |
| `gui` | Textual TUI (works over SSH, no X server) | `pip install "afmkit[gui] @ ..."` |
| `all` | Everything above | `pip install "afmkit[all] @ ..."` |
| `dev` | pytest, ruff, mypy, pre-commit | (developer setup, see [Contributing](contributing.md)) |

## Your first fit

```python
from pathlib import Path
from afmkit.io.jpk_txt import load_jpk_txt
from afmkit.fitting import fit
from afmkit.analysis import find_sawtooth_peaks, PeakReviewer
from afmkit.io.exporters import to_csv_fits

# 1. Load a folder of JPK 4-column .txt files (k in pN/nm).
batch = load_jpk_txt("./data/*.txt", k_cantilever=0.1)

# 2. Auto-detect sawtooth peaks on the first retract curve.
peaks = find_sawtooth_peaks(batch.retract_curves()[0], min_prominence_pN=15.0)
print(f"Auto-detected {len(peaks)} unfolding peaks")

# 3. Review them — accept, reject, override force, re-fit a single peak.
reviewer = PeakReviewer(peaks, batch.retract_curves()[0])
reviewer.accept(0)
reviewer.reject(1)
reviewer.override(2, 42.5)  # user-override force in pN

# 4. Fit WLC on the curve, export per-fit + per-peak review state.
result = fit(batch[0], model="wlc", x_range=(20.0, 180.0))
print(f"p  = {result.params['p']:.3f} nm,  Lc = {result.params['L']:.1f} nm")

to_csv_fits([result], "fits.csv", reviewers={0: reviewer})
```

## From the shell

```bash
afmkit import ./data/*.txt -o curves.h5 --k 0.1
afmkit fit curves.h5 --model wlc --output fits.csv
afmkit info curves.h5
afmkit export curves.h5 --format md -o report.md
afmkit gui                                       # launch the Textual TUI
```

## Try the TUI in 30 seconds

```bash
pip install "afmkit[gui,plot] @ git+https://github.com/linjiema/afmkit.git@v0.5.0"
afmkit gui
# → press `o`, type a directory of JPK .txt files, hit Enter
# → arrow-keys to a curve, press `f` to fit, `P` to toggle the plot panel
# → press `p` to enter peak review, `j`/`k` to navigate,
#   `a` accept, `r` reject, `o` override force, `R` re-fit a single peak
```

## Round-trip an Igor `.ibw` file (v0.5+)

`save_ibw` embeds every scalar metadata key in the wave `note`,
and `load_ibw` re-hydrates them all — so the round-trip is
loss-less without the caller passing `k_cantilever` (or anything
else) explicitly.

```python
from afmkit.core.curve import ForceCurve
from afmkit.io.igor_ibw import save_ibw, load_ibw, roundtrip_ibw
import numpy as np

x = np.linspace(0, 200, 500)
f = np.sin(x) * 30
curve = ForceCurve(
    x, f,
    metadata={
        "k_cantilever": 0.085,
        "temperature": 297.5,
        "experiment_id": "exp-2026-07-31",
        "in_liquid": True,
    },
)

# One-call round-trip with verification (asserts the data and
# metadata round-trip cleanly):
loaded = roundtrip_ibw(curve, "out.ibw", version=5)
assert loaded.metadata["k_cantilever"] == 0.085
assert loaded.metadata["temperature"] == 297.5
assert loaded.metadata["in_liquid"] is True
```

## Where to go next

- [Migration from Igor Pro](migration.md) — every original Igor
  macro mapped to its afmkit equivalent.
- [API reference](api/index.md) — auto-generated from docstrings.
- [Contributing](contributing.md) — plugin authoring and dev
  setup.
- [Git workflow](git-workflow.md) — branch model, PR / commit
  rules, release flow.
