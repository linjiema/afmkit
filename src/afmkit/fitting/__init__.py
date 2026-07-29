"""Curve fitting engines.

This module wraps ``lmfit`` (and optionally other backends) behind a uniform
:class:`~afmkit.fitting.engine.Fitter` interface, so that swapping fit
strategies (Levenberg-Marquardt, robust, Bayesian MCMC, global) is a
one-line change at the call site.

The result of every fit is captured as a :class:`~afmkit.fitting.report.FitResult`
dataclass, which holds the parameter estimates, uncertainties, covariance
matrix, residual statistics, and the original data — a single object you
can save, plot, or hand to the GUI.
"""

from __future__ import annotations
