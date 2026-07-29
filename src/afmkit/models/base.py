"""Polymer model protocol.

This module defines :class:`PolymerModel`, the structural type every
polymer physics model in afmkit must satisfy. Concrete models (WLC, eWLC,
FJC, …) live in sibling modules and are registered with the model
registry; they do not need to inherit from anything — Python's structural
duck typing does the rest.

Why a Protocol, not an ABC?
----------------------------
A :class:`~typing.Protocol` keeps the model contract declarative and
side-effect-free:

- No method-resolution surprises when users mix in their own base
  classes.
- Static type checkers (mypy) verify conformance from the call site
  without requiring an explicit ``isinstance`` check.
- A model packaged as a third-party plugin can live in its own
  distribution and still satisfy the contract.

The contract
------------
Every model exposes four pieces of metadata and two callables:

==========================  ===================================================
Attribute / method          Meaning
==========================  ===================================================
``param_names``             Names of the model parameters, in canonical order.
``param_bounds``            Closed-open ``(lo, hi)`` tuples for the fitter, in
                            the same order as ``param_names``.
``param_hints``             Human-readable label per parameter, keyed by
                            ``param_name``. Used by the GUI and the CLI help.
``__call__(x, **params)``   Evaluate the model. ``x`` is an array of
                            extensions in **nm**; keyword arguments match
                            ``param_names``. Returns forces in **pN**.
``guess_params(x, y)``      Produce a starting point for non-linear fitting
                            from a sample of the data. ``x`` and ``y`` are
                            the extension and force arrays; the returned
                            ``dict`` must have exactly the keys in
                            ``param_names``.
==========================  ===================================================

Units
-----
- Extension ``x``: **nm**.
- Force ``F``: **pN**.
- All model-specific parameters (persistence length, contour length,
  stretch modulus, …) are in **nm** and **pN** to match the rest of
  afmkit's data model.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["PolymerModel"]


@runtime_checkable
class PolymerModel(Protocol):
    """Structural type for polymer force-extension models.

    Any class that exposes the attributes and methods below is a valid
    afmkit polymer model — no inheritance required.

    Examples
    --------
    A minimal concrete implementation:

    >>> import numpy as np
    >>> from afmkit.models.base import PolymerModel
    >>> class LinearModel:
    ...     param_names = ("k",)
    ...     param_bounds = ((0.0, 10.0),)
    ...     param_hints = {"k": "Stiffness (pN/nm)"}
    ...     def __call__(self, x, *, k):
    ...         return np.asarray(x, dtype=float) * k
    ...     def guess_params(self, x, y):
    ...         return {"k": float(np.asarray(y).mean() / max(np.asarray(x).mean(), 1e-12))}
    >>> m = LinearModel()
    >>> isinstance(m, PolymerModel)
    True
    >>> m(np.array([1.0, 2.0, 3.0]), k=2.0)
    array([2., 4., 6.])
    """

    #: Canonical parameter names, in the order the fitting engine
    #: passes them to :meth:`__call__`.
    param_names: tuple[str, ...]

    #: Closed search bounds for the fitter. Each entry is a ``(lo, hi)``
    #: pair; the lower bound is inclusive, the upper bound is inclusive
    #: too (afmkit uses bounded least-squares, not strict inequalities).
    param_bounds: tuple[tuple[float, float], ...]

    #: Human-readable description per parameter, keyed by name. Surfaced
    #: in the GUI and in CLI ``--help`` output.
    param_hints: dict[str, str]

    def __call__(self, x: np.ndarray, **params: float) -> np.ndarray:
        """Evaluate the model at ``x`` with the given parameters.

        Parameters
        ----------
        x
            1-D array of extensions in **nm**.
        **params
            Parameter values, keyed by :attr:`param_names`.

        Returns
        -------
        np.ndarray
            1-D array of forces in **pN**, same shape as ``x``.
        """
        ...

    def guess_params(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Produce a starting point for non-linear fitting.

        Parameters
        ----------
        x, y
            The observed data (extensions in nm, forces in pN). Both are
            1-D arrays of the same length.

        Returns
        -------
        dict
            A dictionary with exactly the keys in :attr:`param_names`.
            Values must lie inside :attr:`param_bounds` whenever
            possible; the fitter will clamp otherwise.
        """
        ...
