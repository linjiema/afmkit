# Quickstart

> **Status: in progress.** This page will populate as v0.1 features land.

## Install

```bash
pip install afmkit[all]
```

Extras:

| Extra | What it adds |
|---|---|
| `io` | HDF5 storage backend |
| `igor` | Read legacy `.ibw` files |
| `matlab` | Matlab `.mat` round-trip |
| `plot` | Static matplotlib paper figures |
| `all` | Everything above |
| `dev` | pytest, ruff, mypy, pre-commit |

## Your first fit

```python
import afmkit

# 1. Load a folder of JPK .txt files
batch = afmkit.load_jpk_txt("./data/*.txt", k_cantilever=0.1)  # pN/nm

# 2. Pick one curve and fit it
curve = batch[0]
result = afmkit.fit(curve, model="wlc")

# 3. Inspect the result
print(result)
print(f"p = {result.params['p']:.3f} nm")
print(f"L = {result.params['L']:.1f} nm")
print(f"Reduced χ² = {result.redchi:.2f}")

# 4. Save it
afmkit.exporters.to_csv(result, "curve_0.csv")
```

## From the shell

```bash
afmkit import ./data/*.txt -o curves.h5 --k 0.1
afmkit fit curves.h5 --model wlc --output results.csv
afmkit info curves.h5
```

## Where to go next

- [Migration from Igor Pro](migration.md)
- [API reference](api/index.md)
- [Contributing](contributing.md)
