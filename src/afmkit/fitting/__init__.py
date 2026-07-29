"""Curve fitting engines.

This module wraps ``lmfit`` (and optionally other backends) behind a uniform
:class:`~afmkit.fitting.engine.LmfitEngine` interface, so that swapping fit
strategies (Levenberg-Marquardt, robust, Bayesian MCMC, global) is a
one-line change at the call site.

The result of every fit is captured as a :class:`~afmkit.fitting.report.FitResult`
dataclass, which holds the parameter estimates, uncertainties, covariance
matrix, residual statistics, and the original data — a single object you
can save, plot, or hand to the GUI.

Typical use
-----------
>>> import numpy as np
>>> from afmkit.fitting import LmfitEngine, fit
>>> from afmkit.models import WLCModel
>>>
>>> # Synthesise a noise-free WLC curve (p=0.4, L=200).
>>> x = np.linspace(1.0, 199.0, 200)
>>> model = WLCModel()
>>> y = model(x, p=0.4, L=200.0)
>>>
>>> # Lower-level: instantiate an engine, call .fit() many times.
>>> engine = LmfitEngine(WLCModel())
>>> result = engine.fit(x, y, x_range=(20.0, 180.0))
>>> result.metadata["success"]
True
>>> abs(result.params["p"] - 0.4) < 0.01
True
>>> abs(result.params["L"] - 200.0) < 1.0
True
>>>
>>> # Higher-level: pick a model by name, get a FitResult directly.
>>> from afmkit.core.curve import ForceCurve
>>> curve = ForceCurve(x, y, metadata={"k_cantilever": 0.1})
>>> result = fit(curve, model="wlc")
>>> result.metadata["success"]
True
"""

from __future__ import annotations

from afmkit.fitting.engine import LmfitEngine, fit
from afmkit.fitting.report import FitResult

__all__ = ["FitResult", "LmfitEngine", "fit"]
