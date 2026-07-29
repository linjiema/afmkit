"""afmkit — single-molecule force spectroscopy analysis in Python.

A modern, extensible toolkit for analyzing atomic force microscopy (AFM)
force-extension curves. Designed to replace Igor Pro workflows with a clean
Python API, pluggable backends, and first-class interoperability with
Origin / Matlab / Python ecosystems.

Architecture
------------
- :mod:`afmkit.core`     — Data model (ForceCurve, CurveBatch, Session)
- :mod:`afmkit.io`       — File I/O loaders and exporters (HDF5, JPK, Igor IBW, CSV, …)
- :mod:`afmkit.models`   — Polymer physics models (WLC, eWLC, FJC, …)
- :mod:`afmkit.fitting`  — Curve fitting engines (LM, robust, MCMC)
- :mod:`afmkit.processing` — Signal processing (smoothing, baseline, peak finding)
- :mod:`afmkit.analysis` — High-level analysis workflows
- :mod:`afmkit.presentation` — CLI (and GUI in a future release)
- :mod:`afmkit.plugins`  — pluggy-based extension points

Quick start (v0.1, core only)
-----------------------------
The data model is the foundation. Higher-level IO and fitting APIs land
in upcoming releases — see README.md for the roadmap.

>>> import numpy as np
>>> from afmkit.core.curve import ForceCurve
>>> ext = np.linspace(0.5, 199.5, 1000)
>>> p, lc = 0.4, 200.0
>>> force = (4.1 / p) * (0.25 * (1 - ext / lc) ** -2 - 0.25 + ext / lc)
>>> curve = ForceCurve(ext, force, metadata={"k_cantilever": 0.1})
>>> curve.n_points
1000
>>> sub = curve.select_range(50, 100)
>>> sub.n_points
251
"""

from __future__ import annotations

from afmkit._version import __version__

__all__ = ["__version__"]
