# afmkit

> Modern Python toolkit for single-molecule force spectroscopy data analysis.

[![CI](https://github.com/linjiema/afmkit/actions/workflows/ci.yml/badge.svg)](https://github.com/linjiema/afmkit/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/afmkit.svg)](https://pypi.org/project/afmkit/)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Typing: mypy --strict](https://img.shields.io/badge/typing-mypy--strict-blue)](https://mypy.readthedocs.io)

**afmkit** is a clean, extensible Python reimplementation of the workflow that
single-molecule biophysics labs have been running in Igor Pro for two decades.
It reads AFM force-extension curves (JPK Nanowizzard, ForceRobot, plus legacy
Igor Binary Wave), fits the standard polymer models (WLC / eWLC / FJC),
picks sawtooth unfolding peaks, and exports the results in formats that
drop straight into **Origin**, **Matlab**, and the **Python** data stack.

The goal is a tool that a grad student can use from a notebook **and** a
lab can build a reproducible pipeline on top of.

---

## ✨ Features

- 📂 **Loaders** for JPK 4-column `.txt`, legacy Igor `.ibw`, and a native
  HDF5 format — plus a plugin system for new formats.
- 🧬 **Polymer models**: Marko-Siggia WLC (1:1 compatible with the original
  Igor implementation), extensible WLC (eWLC, Wang 1997 interpolation),
  Freely Jointed Chain.
- 🔬 **Automated sawtooth peak detection** with optional manual review.
- 📊 **Exports** to CSV, Matlab `.mat`, Parquet, Markdown report, and HDF5
  — drop into Origin / Matlab / pandas / R without reformatting.
- 🧩 **Plugin architecture** (pluggy-based): add a new file format or a new
  model with a single `pip install`.
- 🐍 **Type-hinted, strictly typed, fully tested** core (mypy --strict,
  pytest, hypothesis).
- 🖥️ **CLI** today, GUI coming in v0.2.

## 🚀 Quick start

```bash
pip install afmkit[all]
```

```python
import afmkit

# Load a folder of JPK .txt files
batch = afmkit.load_jpk_txt("./data/*.txt", k_cantilever=0.1)  # pN/nm

# Fit the first curve with the standard WLC model
result = afmkit.fit(batch[0], model="wlc")
print(f"p = {result.params['p']:.2f} nm")
print(f"Lc = {result.params['L']:.1f} nm")
print(f"Reduced χ² = {result.redchi:.2f}")

# Save a CSV that opens cleanly in Origin
afmkit.exporters.to_csv(batch.fit_all(model="wlc"), "results.csv")

# And a self-contained HDF5 with everything (curves + fits + metadata)
afmkit.exporters.to_hdf5(session, "session.h5")
```

Or from the shell:

```bash
afmkit import ./data/*.txt -o curves.h5 --k 0.1
afmkit fit curves.h5 --model wlc --output results.csv
afmkit info curves.h5
```

## 🧱 Architecture

```
afmkit/
├── core/      ← ForceCurve, CurveBatch, Session (xarray + pydantic)
├── io/        ← Loaders and exporters, pluggy-registered
├── models/    ← WLC, eWLC, FJC, … (PolymerModel protocol)
├── fitting/   ← lmfit wrapper, robust / MCMC strategies
├── processing/← smoothing, baseline, peak detection
├── analysis/  ← end-to-end workflows
└── presentation/ ← CLI (typer); GUI in v0.2
```

Every layer only depends on the layers below it. Compute never imports a
GUI library. Models never import IO code. You can swap any layer
(graphical or not) without rewriting the rest.

## 🔌 Plugins

Add a new file format or polymer model without forking afmkit — implement
the relevant protocol and register via entry points:

```toml
# your-plugin/pyproject.toml
[project.entry-points."afmkit.models"]
fjc = "afmkit_fjc:FJCModel"
```

```bash
pip install afmkit-fjc
```

`afmkit.fit(..., model="fjc")` now works. See
[docs/contributing.md](docs/contributing.md) for the full plugin author guide.

## 🔁 Migrating from Igor Pro

If you have an existing `FX_Analysis` workflow, see
[docs/migration.md](docs/migration.md) for a one-to-one mapping of the
original functions (`FXImport`, `WLCurves`, `AutoFindForcePeaks`, …) to
their afmkit equivalents, plus a recipe for reading your old `.ibw` data
into afmkit and writing it back as HDF5.

## 📦 Output formats

| Format | Use case | Reader |
|---|---|---|
| **HDF5** (default) | Single-file archive: raw + processed + fits + metadata | h5py, Matlab HDF5, Origin 2024+ |
| **CSV** (wide) | Drop into Origin / Excel | any spreadsheet |
| **Matlab `.mat`** (v7.3 / HDF5 backend) | Hand off to Matlab collaborators | Matlab R2007b+ |
| **Parquet** | Big batches, columnar, future ML | pandas, polars, Arrow |
| **Markdown report** | One-page summary per curve for the PI | any text editor |
| **Igor `.ibw`** | Round-trip with existing Igor pipelines | Igor Pro 6+ |

## 🧪 Development

```bash
git clone https://github.com/linjiema/afmkit
cd afmkit
pip install -e ".[all,dev]"
pre-commit install
pytest
```

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
