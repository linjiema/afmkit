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

Suppose you want to add a twist-WLC model as a plugin. The
first-party FJC model is shipped in-tree as `afmkit.models.fjc`
(v0.3+, registered both in `MODEL_REGISTRY["fjc"]` and as a
pluggy entry point so it can be re-implemented out of tree);
WLC (`"wlc"`) and eWLC (`"ewlc"`) are also first-party. Use
this skeleton for a *new* polymer model — e.g. twist-WLC,
Marko-Siggia with a stretch modulus, or a polymer variant
specific to your lab.

### 1. Project layout

```
afmkit-twlc/
├── pyproject.toml
├── src/
│   └── afmkit_twlc/
│       ├── __init__.py
│       └── twlc.py
└── tests/
    └── test_twlc.py
```

### 2. The model

```python
# src/afmkit_twlc/twlc.py
from __future__ import annotations

import numpy as np
from afmkit.models.base import PolymerModel


class TwistWLCModel(PolymerModel):
    """Twist worm-like chain (placeholder for the lab's specific
    polymer variant — replace with the actual formula).
    """

    param_names: tuple[str, ...] = ("p", "L", "C")
    param_bounds: tuple[tuple[float, float], ...] = (
        (0.1, 5.0),    # p (persistence length, nm)
        (10.0, 1000.0),  # L (contour length, nm)
        (0.0, 1000.0),   # C (twist rigidity, kB T·nm)
    )
    param_hints: dict[str, str] = {
        "p": "Persistence length (nm)",
        "L": "Contour length (nm)",
        "C": "Twist rigidity (kB T·nm)",
    }

    def __call__(self, x: np.ndarray, p: float, L: float, C: float) -> np.ndarray:
        # ... evaluate the twist-WLC formula
        ...

    def guess_params(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        return {"p": 0.4, "L": float(x.max()), "C": 100.0}
```

### 3. Register via entry points

```toml
# pyproject.toml
[project]
name = "afmkit-twlc"
dependencies = ["afmkit>=0.5"]

[project.entry-points."afmkit.models"]
twlc = "afmkit_twlc.twlc:TwistWLCModel"
```

### 4. Publish to PyPI

```bash
pip install afmkit-twlc
```

afmkit auto-discovers the plugin on next import. Users can now do:

```python
result = afmkit.fit(curve, model="twlc")
```

## What you should NOT do

- Don't import afmkit's GUI modules from a plugin.
- Don't mutate afmkit's module-level state.
- Don't ship a plugin with a hard dependency on a non-trivial GUI
  library (matplotlib is fine, PyQt6 is not — keep the core importable
  in headless environments).
