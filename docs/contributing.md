# Contributing and plugin authoring

For the full contribution guide (dev setup, code style, PR workflow,
release process), see
[CONTRIBUTING.md](https://github.com/linjiema/afmkit/blob/main/CONTRIBUTING.md)
at the repository root. This page focuses on **writing plugins** for
afmkit.

## Why plugins?

afmkit is meant to be a small, well-tested core. Everything that is
specific to a particular instrument, polymer variant, or analysis
strategy lives in a separate package. This keeps the core stable while
letting the ecosystem grow.

## Hooks available

The plugin system is built on [pluggy](https://pluggy.readthedocs.io),
the same hook mechanism pytest uses. The four hookspecs are declared in
:mod:`afmkit.plugins`:

| Hookspec | Purpose | Returns |
|---|---|---|
| `register_loader` | New file format | `type[Loader]` |
| `register_model`  | New polymer model | `type[PolymerModel]` |
| `register_baseline` | New baseline correction | `type[BaselineCorrector]` |
| `register_fitter` | New fitting engine | `type[Fitter]` |

## Skeleton: a new model

Suppose you want to add a Freely Jointed Chain (FJC) model as a plugin.

### 1. Project layout

```
afmkit-fjc/
├── pyproject.toml
├── src/
│   └── afmkit_fjc/
│       ├── __init__.py
│       └── fjc.py
└── tests/
    └── test_fjc.py
```

### 2. The model

```python
# src/afmkit_fjc/fjc.py
from __future__ import annotations

import numpy as np
from afmkit.models.base import PolymerModel


class FJCModel(PolymerModel):
    """Freely Jointed Chain (FJC) model.

    F(x) = (kB T / b) * [L/x - 1/4 + x/L]  (with Kuhn length b)
    """

    param_names: tuple[str, ...] = ("b", "L")
    param_bounds: tuple[tuple[float, float], ...] = ((0.1, 10.0), (10.0, 1000.0))
    param_hints: dict[str, str] = {"b": "Kuhn length (nm)", "L": "Contour length (nm)"}

    def __call__(self, x: np.ndarray, b: float, L: float) -> np.ndarray:
        # ... evaluate the FJC formula
        ...

    def guess_params(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        return {"b": 1.0, "L": float(x.max())}
```

### 3. Register via entry points

```toml
# pyproject.toml
[project]
name = "afmkit-fjc"
dependencies = ["afmkit>=0.1"]

[project.entry-points."afmkit.models"]
fjc = "afmkit_fjc.fjc:FJCModel"
```

### 4. Publish to PyPI

```bash
pip install afmkit-fjc
```

afmkit auto-discovers the plugin on next import. Users can now do:

```python
result = afmkit.fit(curve, model="fjc")
```

## What you should NOT do

- Don't import afmkit's GUI modules from a plugin.
- Don't mutate afmkit's module-level state.
- Don't ship a plugin with a hard dependency on a non-trivial GUI
  library (matplotlib is fine, PyQt6 is not — keep the core importable
  in headless environments).
