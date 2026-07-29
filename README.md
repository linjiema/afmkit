# afmkit

> Modern Python toolkit for single-molecule force spectroscopy data analysis.

[![CI](https://github.com/linjiema/afmkit/actions/workflows/ci.yml/badge.svg)](https://github.com/linjiema/afmkit/actions/workflows/ci.yml)
[![GitHub release](https://img.shields.io/github/v/release/linjiema/afmkit)](https://github.com/linjiema/afmkit/releases)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Typing: mypy](https://img.shields.io/badge/typing-mypy-blue)](https://mypy.readthedocs.io)

**afmkit** is a clean, extensible Python reimplementation of the workflow that
single-molecule biophysics labs have been running in Igor Pro for two decades.
It reads AFM force-extension curves (JPK Nanowizzard, ForceRobot, plus legacy
Igor Binary Wave), fits the standard polymer models (WLC / eWLC / FJC),
picks sawtooth unfolding peaks, and exports the results in formats that
drop straight into **Origin**, **Matlab**, and the **Python** data stack.

The goal is a tool that a grad student can use from a notebook **and** a
lab can build a reproducible pipeline on top of.

> **Install from GitHub** — afmkit is not on PyPI. Pin a tag (`@v0.1.0`)
> for reproducibility, or `@main` for the bleeding edge.

---

## ✨ Features (v0.1)

- 📂 **Loaders** for JPK 4-column `.txt` and a native HDF5 format — plugin
  hooks for new formats.
- 🧬 **Marko-Siggia WLC** model, 1:1 compatible with the original Igor
  implementation. eWLC and FJC land as plugin packages in v0.2.
- 🔬 **lmfit-backed fitting** with a thin `LmfitEngine` and a high-level
  `fit(curve, model="wlc", x_range=...)` helper.
- 📊 **Exports** to wide-column CSV (Origin / Matlab friendly), `.mat`,
  Parquet, Markdown, and a native HDF5 round-trip.
- 🧩 **Plugin architecture** (pluggy): add a new file format or a new model
  with a single `pip install`.
- 🖥️ **CLI** (`afmkit import / fit / export / info / version`) — GUI in v0.2.

## 🚀 Quick start

```bash
pip install "afmkit @ git+https://github.com/linjiema/afmkit.git@v0.1.0"
```

```python
from pathlib import Path
from afmkit.io.jpk_txt import load_jpk_txt
from afmkit.fitting import fit
from afmkit.io.exporters import to_csv, to_csv_fits
from afmkit.io.hdf5_store import save_hdf5

# Load a folder of JPK 4-column .txt files (k in pN/nm)
batch = load_jpk_txt("./data/*.txt", k_cantilever=0.1)
print(f"Loaded {batch.n_curves} curves from {len(set(c.metadata.get('source_file', '') for c in batch))} files")

# Fit the first curve with the standard WLC model
result = fit(batch[0], model="wlc", x_range=(20.0, 180.0))
print(f"p  = {result.params['p']:.3f} nm")
print(f"Lc = {result.params['L']:.1f} nm")
print(f"Reduced χ² = {result.redchi:.3f}")

# Wide-column CSV that opens cleanly in Origin / Excel / Matlab
to_csv(batch, "results.csv")

# Per-fit summary table
fits = [fit(curve, model="wlc", x_range=(20.0, 180.0)) for curve in batch]
to_csv_fits(fits, "fits.csv")

# Self-contained archive (curves + metadata, JSON-encoded)
save_hdf5(batch, "session.h5")
```

Or from the shell:

```bash
afmkit import ./data/*.txt -o curves.h5 --k 0.1
afmkit fit curves.h5 --model wlc --output fits.csv
afmkit info curves.h5
afmkit export curves.h5 --format md -o report.md
```

## 🧱 Architecture

```
afmkit/
├── core/      ← ForceCurve, CurveBatch (xarray-backed)
├── io/        ← Loaders (JPK, HDF5) and exporters (CSV, MAT, Parquet, MD)
├── models/    ← WLCModel + MODEL_REGISTRY (plugin-extensible)
├── fitting/   ← LmfitEngine, fit() helper, FitResult
├── processing/← smoothing, baseline (v0.2)
├── analysis/  ← end-to-end workflows (v0.2)
└── presentation/ ← typer-based CLI; GUI in v0.2
```

Every layer only depends on the layers below it. Compute never imports a
GUI library. Models never import IO code. You can swap any layer
(graphical or not) without rewriting the rest.

## 🔌 Plugins

Add a new file format or polymer model without forking afmkit — register
a class via the public `MODEL_REGISTRY`:

```python
from afmkit.models import register_model
from afmkit.models.base import PolymerModel

@register_model("fjc")
class FJCModel:
    # implements the PolymerModel protocol
    ...
```

A pluggy-based entry-point system for separately-installable plugins
(an `afmkit-fjc` package) is wired in v0.2. See
[docs/contributing.md](docs/contributing.md) for the full plugin author guide.

## 🔁 Migrating from Igor Pro

If you have an existing `FX_Analysis` workflow, see
[docs/migration.md](docs/migration.md) for a one-to-one mapping of the
original functions (`FXImport`, `WLCurves`, `FitToCursor`, …) to their
afmkit equivalents, plus the units-and-sign-conventions cheat sheet and
the gotchas list (`.ibw` is v0.2; the 200-point baseline is applied for
you by `load_jpk_txt`, so don't double-correct).

## 📦 Output formats

| Format | Use case | Reader |
|---|---|---|
| **HDF5** (default) | Single-file archive: raw + metadata, JSON-encoded | h5py, Matlab (HDF5), Origin 2024+ |
| **CSV** (wide) | Drop into Origin / Excel | any spreadsheet |
| **Matlab `.mat`** (v5) | Hand off to Matlab collaborators | Matlab R2007b+ |
| **Parquet** | Big batches, columnar, future ML | pandas, polars, Arrow |
| **Markdown report** | One-page summary per curve for the PI | any text editor |
| **Igor `.ibw`** | Round-trip with existing Igor pipelines (v0.2) | Igor Pro 6+ |

## 🧪 Development

```bash
git clone https://github.com/linjiema/afmkit
cd afmkit
python -m venv .venv
source .venv/bin/activate
pip install -e ".[io,dev]"
pytest
```

The dev extras pull in `pytest`, `ruff`, `mypy`, `h5py`, `hypothesis`,
and the pre-commit hooks. No PyPI publish — we install from this repo.

## 📚 Documentation

- [Quick start](docs/quickstart.md)
- [API reference](docs/api/index.md)
- [Migration from Igor Pro](docs/migration.md)
- [Plugin authoring](docs/contributing.md)
- [Tutorials](docs/tutorials/)

## 🤝 Contributing

Issues and pull requests are welcome. For significant changes, please open
an issue first to discuss what you'd like to change. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the dev workflow.

## 📄 Citation

If afmkit helped your research, please cite it:

```bibtex
@software{afmkit,
  title = {afmkit: a Python toolkit for single-molecule force spectroscopy},
  url = {https://github.com/linjiema/afmkit},
  version = {<version>},
  year = {<year>},
}
```

A CITATION.cff is also provided for GitHub's built-in citation parser.

## 📝 License

[MIT](LICENSE) — do what you want, just keep the copyright notice.
