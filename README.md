# afmkit

> Modern Python toolkit for single-molecule force spectroscopy data analysis.

[![CI](https://github.com/linjiema/afmkit/actions/workflows/ci.yml/badge.svg)](https://github.com/linjiema/afmkit/actions/workflows/ci.yml)
[![Docs](https://github.com/linjiema/afmkit/actions/workflows/docs.yml/badge.svg)](https://linjiema.github.io/afmkit/)
[![GitHub release](https://img.shields.io/github/v/release/linjiema/afmkit)](https://github.com/linjiema/afmkit/releases)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Typing: mypy](https://img.shields.io/badge/typing-mypy--strict-blue)](https://mypy.readthedocs.io)

**afmkit** is a clean, extensible Python reimplementation of the workflow
that single-molecule biophysics labs have been running in Igor Pro for two
decades. It reads AFM force-extension curves (JPK Nanowizzard / ForceRobot,
plus legacy Igor Binary Wave), fits the standard polymer models
(WLC / eWLC / FJC), picks and interactively reviews sawtooth unfolding
peaks, and exports the results in formats that drop straight into
**Origin**, **Matlab**, and the **Python** data stack.

The goal is a tool a grad student can use from a notebook **and** a lab can
build a reproducible pipeline on top of.

> **Install from GitHub** — afmkit is not on PyPI. Pin a tag (`@v0.3.0`)
> for reproducibility, or `@main` for the bleeding edge. See
> [What's new in v0.3.0](https://github.com/linjiema/afmkit/releases/tag/v0.3.0).

---

## ✨ Features (v0.3.0)

### Reading and writing

- 📂 **JPK 4-column `.txt` loader** — `load_jpk_txt()`, bit-for-bit identical
  to the original Igor `FXImport()`. Returns a `CurveBatch` of approach /
  retract `ForceCurve`s with the same k_cantilever convention (pN/nm).
- 🗃️ **HDF5 native store** — gzip-compressed, JSON-encoded metadata, ragged
  batches supported. `save_hdf5` / `load_hdf5` round-trip.
- 🔁 **Igor `.ibw` round-trip** (optional `[igor]` extra) — `load_ibw`,
  `load_ibw_batch`, `save_ibw`. F/B files in the same folder are paired
  as approach/retract by basename. The writer is a stdlib-only v2 binary
  emitter (the released `igor==0.3` PyPI package's `save()` is still
  `NotImplementedError` upstream).
- 📊 **Exporters** — wide-column CSV (Origin / Matlab friendly), `.mat`
  v5, Parquet (pyarrow / fastparquet), Markdown, and a native HDF5
  round-trip.

### Models

- 🧬 **Marko-Siggia WLC** — 1:1 with the original `LVFitWLC` formula,
  `F = (4.1 / p) · [0.25·(1 − x/L)⁻² − 0.25 + x/L]`. The default
  `"wlc"` registry entry.
- 📐 **eWLC** — Wang 1997 / Odijk 1995 interpolation with the
  `K0` stretch modulus (pN). Reduces to Marko-Siggia as K0 → ∞.
  Finite on `[0, L]`; the singularity lives strictly past the contour
  length (`x = L·(1 + 1/K0) > L`) — the practical advantage over WLC.
- 🪢 **FJC** — Classical Freely Jointed Chain with the Padé [2,2]
  inverse Langevin approximation (b Kuhn length, Lc contour length).
  Registered both in `MODEL_REGISTRY["fjc"]` and via pluggy
  entry-point — the first model exposed as a separable plugin.
- 🧩 **Plugin architecture** (pluggy) — add a new file format or a new
  model with a single `pip install`. The `afmkit.models` entry-point
  group is the v0.3 first-class extension point.

### Fitting

- 🔬 **lmfit-backed fitting** with a thin `LmfitEngine` and a high-level
  `fit(curve, model="wlc", x_range=...)` helper. Returns `FitResult`
  with best-fit params, standard errors, χ², reduced χ², covariance.

### Analysis

- 📈 **Sawtooth peak detection** — `find_sawtooth_peaks(curve, …)` returns
  `Peak` dataclasses (extension, force, prominence, width, height_drop,
  confidence) for the retract sweep. Centered moving-average smoothing
  → `scipy.signal.find_peaks` with prominence / width thresholds.
- 🖊️ **Peak review** — `PeakReviewer` + `ReviewedPeak` (frozen).
  Accept / reject / override force / re-fit a single peak in a local
  window / attach a free-form note. `to_dict()` round-trips into the
  per-fit CSV / Markdown export.

### Presentation

- 🖥️ **Textual TUI** — `afmkit gui`. Three-panel layout: directory
  input → data table → status line. Keybindings: `o` open directory,
  `f` fit, `e` export, `p` peak review, `P` toggle plot panel, `q` quit.
  No X server, no PySide6 — runs in any terminal over SSH.
- 🖼️ **Matplotlib plot widget** — `ForceExtensionPlot` renders a
  force-extension curve with optional peak markers and WLC fit overlay
  via the matplotlib Agg backend, mounted inside the TUI panel.
- 💻 **CLI** — `afmkit version / info / import / fit / export / gui`
  subcommands (typer + rich), end-to-end wired.

---

## 🚀 Quick start

```bash
pip install "afmkit[all] @ git+https://github.com/linjiema/afmkit.git@v0.3.0"
```

The `[all]` extra pulls in `[io,parquet,igor,matlab,plot,gui]` — everything
the headline workflows need. Drop it for a minimal install and add extras
on demand:

```bash
pip install "afmkit @ git+https://github.com/linjiema/afmkit.git@v0.3.0"           # core only
pip install "afmkit[igor] @ git+https://github.com/linjiema/afmkit.git@v0.3.0"    # .ibw round-trip
pip install "afmkit[gui]  @ git+https://github.com/linjiema/afmkit.git@v0.3.0"    # Textual TUI
pip install "afmkit[plot] @ git+https://github.com/linjiema/afmkit.git@v0.3.0"    # matplotlib panel
```

### From Python — load, fit, review

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
reviewer = PeakReviewer(peaks)
reviewer.accept(peaks[0])
reviewer.reject(peaks[1])
reviewer.override_force(peaks[2], manual_force_pN=42.5, note="looks like a doublet")

# 4. Fit WLC on the curve, export per-fit + per-peak review state.
result = fit(batch[0], model="wlc", x_range=(20.0, 180.0))
print(f"p  = {result.params['p']:.3f} nm,  Lc = {result.params['L']:.1f} nm")

to_csv_fits([result], "fits.csv", reviewers={0: reviewer})
```

### From the shell

```bash
afmkit import ./data/*.txt -o curves.h5 --k 0.1
afmkit fit curves.h5 --model wlc --output fits.csv
afmkit info curves.h5
afmkit export curves.h5 --format md -o report.md
afmkit gui                                       # launch the Textual TUI
```

### Try the TUI in 30 seconds

```bash
pip install "afmkit[gui,plot] @ git+https://github.com/linjiema/afmkit.git@v0.3.0"
afmkit gui
# → press `o`, type a directory of JPK .txt files, hit Enter
# → arrow-keys to a curve, press `f` to fit, `P` to toggle the plot panel
# → press `p` to enter peak review, `j`/`k` to navigate,
#   `a` accept, `r` reject, `o` override force, `R` re-fit a single peak
```

---

## 🧱 Architecture

```
afmkit/
├── core/          ← ForceCurve, CurveBatch (xarray-backed)
├── io/            ← Loaders (JPK .txt, Igor .ibw, HDF5) + exporters (CSV, MAT, Parquet, MD)
├── models/        ← WLCModel, EWLCModel, FJCModel + MODEL_REGISTRY (plugin-extensible)
├── fitting/       ← LmfitEngine, fit() helper, FitResult
├── processing/    ← smoothing, baseline
├── analysis/      ← find_sawtooth_peaks, PeakReviewer
└── presentation/  ← typer-based CLI; Textual TUI; matplotlib plot widget
```

Every layer only depends on the layers below it. Compute never imports a
GUI library. Models never import IO code. You can swap any layer
(graphical or not) without rewriting the rest.

---

## 🔌 Plugins

Add a new file format or polymer model without forking afmkit. Two
extension points:

```python
# 1. In-tree: register a class on the public MODEL_REGISTRY.
from afmkit.models import register_model
from afmkit.models.base import PolymerModel

@register_model("my_custom_model")
class MyCustomModel:
    # implements the PolymerModel protocol
    ...
```

```toml
# 2. Out-of-tree: a separately-installable package that registers
# itself via a pyproject.toml entry-point. pluggy discovers it
# at afmkit.plugins.PM time. The FJC model is the in-tree demo.
# pyproject.toml of a third-party plugin:
[project.entry-points."afmkit.models"]
my_model = "my_pkg.models:MyCustomModel"
```

See [docs/contributing.md](docs/contributing.md) for the full plugin
author guide.

---

## 🔁 Migrating from Igor Pro

If you have an existing `FX_Analysis` workflow, see
[docs/migration.md](docs/migration.md) for a one-to-one mapping of the
original functions (`FXImport`, `WLCurves`, `FitToCursor`, …) to their
afmkit equivalents, plus the units-and-sign-conventions cheat sheet
and the gotchas list.

The headline v0.3 changes vs. the original Igor workflow:

| Igor | afmkit v0.3 |
|---|---|
| `Load_JPK_FX_Data` / `FXImport` | `afmkit.io.jpk_txt.load_jpk_txt` |
| `WLCurves` / manual peak picking | `afmkit.analysis.find_sawtooth_peaks` |
| `FitToCursor` per peak (manual) | `PeakReviewer.re_fit(peak, …)` in the TUI |
| `LVFitWLC` (formula verbatim) | `afmkit.models.WLCModel` |
| `.ibw` save | `afmkit.io.save_ibw` (v0.3 writer is v2; v5 is v0.4) |
| Save curves + notes | `afmkit.io.hdf5_store.save_hdf5` |

---

## 📦 Output formats

| Format | Use case | Reader |
|---|---|---|
| **HDF5** (default) | Single-file archive: raw + metadata, JSON-encoded | h5py, Matlab (HDF5), Origin 2024+ |
| **CSV** (wide) | Drop into Origin / Excel | any spreadsheet |
| **Matlab `.mat`** (v5) | Hand off to Matlab collaborators | Matlab R2007b+ |
| **Parquet** | Big batches, columnar, future ML | pandas, polars, Arrow |
| **Markdown report** | One-page summary per curve for the PI | any text editor |
| **Igor `.ibw`** | Round-trip with existing Igor pipelines | Igor Pro 6+ |

---

## ✅ Verified on

CI matrix is green on every cell. Local gates are `ruff check` + `ruff format --check` + `mypy --strict` + `pytest`.

| OS | Python 3.11 | Python 3.12 | Python 3.13 |
|---|---|---|---|
| ubuntu-latest | ✅ | ✅ | ✅ |
| macos-latest | ✅ | ✅ | ✅ |
| windows-latest | ✅ | ✅ | ✅ |

Test count: 380 unit + 12 doctest (6 doctest marked `+SKIP` for examples
that depend on a runtime data folder).

---

## 🧪 Development

```bash
git clone https://github.com/linjiema/afmkit
cd afmkit
git checkout develop            # daily integration branch (main is release-only)
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all,dev]"
pytest tests/ -q --doctest-modules src/afmkit
```

The dev extras pull in `pytest`, `ruff`, `mypy`, `h5py`, `hypothesis`,
`pre-commit`, the optional `[igor]`, `[gui]`, `[plot]` extras for the
full test matrix, and the in-tree `pre-commit` hooks (not yet enforced
in CI; v0.4 plan).

### Working on a change

```bash
git checkout -b feature/my-change          # off develop
# ... edit, commit (Conventional Commits: feat/fix/chore/docs/test/...)
git push -u origin feature/my-change
gh pr create --base develop                # PR against develop, not main
```

See [docs/git-workflow.md](docs/git-workflow.md) for the full branch
model, commit rules, PR template, and hotfix procedure. Releases follow
[docs/release-checklist.md](docs/release-checklist.md).

---

## 📚 Documentation

- [Quick start](docs/quickstart.md)
- [API reference](docs/api/index.md)
- [Migration from Igor Pro](docs/migration.md)
- [Plugin authoring](docs/contributing.md)
- [Git workflow](docs/git-workflow.md)
- [Release checklist](docs/release-checklist.md)
- [Tutorials](docs/tutorials/)
- [Roadmap](docs/roadmap.md)

Full site (built with mkdocs + Material) is published at
<https://linjiema.github.io/afmkit/>.

---

## 🤝 Contributing

Issues and pull requests are welcome. For significant changes, please
open an issue first to discuss what you'd like to change. See
[CONTRIBUTING.md](CONTRIBUTING.md) and [docs/git-workflow.md](docs/git-workflow.md)
for the dev workflow.

## 📄 Citation

If afmkit helped your research, please cite it:

```bibtex
@software{afmkit,
  title = {afmkit: a Python toolkit for single-molecule force spectroscopy},
  url = {https://github.com/linjiema/afmkit},
  version = {0.3.0},
  year = {2026},
}
```

A CITATION.cff is also provided for GitHub's built-in citation parser.

## 📝 License

[MIT](LICENSE) — do what you want, just keep the copyright notice.
