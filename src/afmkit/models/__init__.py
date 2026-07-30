"""Polymer physics models for fitting force-extension curves.

Each model is a small class implementing the
:class:`~afmkit.models.base.PolymerModel` protocol:

- ``__call__(x, **params) -> y`` — evaluate the model.
- ``guess_params(x, y) -> dict`` — produce a starting point for fitting.
- ``param_names`` / ``param_bounds`` / ``param_hints`` — metadata for the
  fitting engine and the GUI.

Built-in models
~~~~~~~~~~~~~~~
- :class:`~afmkit.models.wlc.WLCModel` — Marko-Siggia worm-like chain
  (the default; 1:1 compatible with the original Igor implementation).
- :class:`~afmkit.models.ewlc.EWLCModel` — extensible WLC (Wang 1997
  interpolation) with finite stretch modulus ``K0``. Reduces to the
  WLC as ``K0 → ∞``.
- More models (FJC, twist-WLC) are planned as optional plugin packages.

Custom models
~~~~~~~~~~~~~
A user-defined model is just a class that satisfies the protocol and is
registered with the model registry — no monkey-patching, no global state.
Third-party packages can also register a model through the
``[project.entry-points."afmkit.models"]`` pluggy group — see
:mod:`afmkit.plugins` and the example below.

Plugin-registration example
~~~~~~~~~~~~~~~~~~~~~~~~~~~
A third-party package (say ``afmkit-fjc``) can ship an FJC model
without modifying afmkit. The model's class is declared in the
plugin package and then advertised as an entry point in the
plugin's ``pyproject.toml``::

    # src/afmkit_fjc/__init__.py
    from afmkit.models.base import PolymerModel
    import numpy as np

    class FJCModel:
        param_names = ("a", "b", "kT")  # Kuhn length, contour, ...
        param_bounds = ((0.1, 10.0), (10.0, 1000.0), (0.1, 10.0))
        param_hints = {"a": "Kuhn length (nm)", ...}
        def __call__(self, x, *, a, b, kT):
            ...
        def guess_params(self, x, y):
            return {"a": 1.0, "b": 100.0, "kT": 4.1}

    # pyproject.toml of the afmkit-fjc distribution
    [project.entry-points."afmkit.models"]
    fjc = "afmkit_fjc:FJCModel"

After ``pip install afmkit-fjc``, the model is discoverable through
:func:`afmkit.models.get_model` and through the pluggy plugin
manager (:func:`afmkit.plugins.get_plugin_manager`). The afmkit
core stays untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from afmkit.models.base import PolymerModel
from afmkit.models.ewlc import EWLCModel
from afmkit.models.fjc import FJCModel
from afmkit.models.wlc import WLCModel

if TYPE_CHECKING:
    # Imported only for type checkers; the dict literal below is the
    # runtime source of truth so a circular import with the registry
    # helpers is impossible.
    pass

__all__ = [
    "EWLCModel",
    "FJCModel",
    "MODEL_REGISTRY",
    "PolymerModel",
    "WLCModel",
    "get_model",
    "register_model",
]


# -- Model registry ------------------------------------------------------
#
# A plain string → class dict. We don't use pluggy here yet because
# polymer models are pure data — no setup, no teardown, no need for
# hookspec signatures. Third-party packages can call
# :func:`register_model` from their ``__init__.py`` to add themselves,
# or advertise themselves through the
# ``[project.entry-points."afmkit.models"]`` group — see
# :mod:`afmkit.plugins` and the example in this module's docstring.
# The long-term plan is to fold this dict into the pluggy-based plugin
# system, at which point the public surface stays the same and only
# the discovery mechanism changes.

MODEL_REGISTRY: dict[str, type[PolymerModel]] = {
    # WLCModel and EWLCModel satisfy the PolymerModel Protocol
    # structurally (the dataclasses expose param_names/param_bounds/
    # param_hints/__call__/guess_params with matching signatures), but
    # mypy does not infer that relationship for Protocol-typed dict
    # values.
    "wlc": WLCModel,  # type: ignore[dict-item]
    "ewlc": EWLCModel,  # type: ignore[dict-item]
    "fjc": FJCModel,  # type: ignore[dict-item]
}


def register_model(name: str, cls: type[PolymerModel]) -> None:
    """Register a model class under a short string name.

    Parameters
    ----------
    name
        Lookup key (lowercase, e.g. ``"wlc"``, ``"ewlc"``). Used by
        :func:`get_model` and by the ``model=`` argument of
        :func:`afmkit.fitting.engine.fit`.
    cls
        A class satisfying the :class:`PolymerModel` protocol. **Must
        be a class, not an instance** — the engine instantiates per-fit
        so the same class can be shared across curves without
        surprising cross-talk.

    Raises
    ------
    TypeError
        If ``cls`` is not a class.
    ValueError
        If ``name`` is already registered. Use the ``override=True``
        path of a future ``replace_model`` API if you really mean to
        clobber a built-in; the engine deliberately makes accidental
        overwrites hard.
    """
    if not isinstance(cls, type):
        raise TypeError(f"register_model expects a class, got an instance of {type(cls).__name__}")
    if name in MODEL_REGISTRY:
        raise ValueError(
            f"model {name!r} is already registered; choose a different name "
            f"or call afmkit.models.unregister_model({name!r}) first"
        )
    MODEL_REGISTRY[name] = cls


def get_model(name: str) -> PolymerModel:
    """Look up a model by name and return a fresh instance.

    Parameters
    ----------
    name
        The registry key (e.g. ``"wlc"``).

    Returns
    -------
    PolymerModel
        A **new instance** of the registered class. Instances are
        cheap (most models are frozen dataclasses with no state), so
        per-call construction is fine and avoids subtle aliasing
        bugs in long-running batch fits.

    Raises
    ------
    KeyError
        If no model is registered under ``name``. The message lists
        the currently-known names to make typos easy to spot.
    """
    try:
        cls = MODEL_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_REGISTRY)) or "(none)"
        raise KeyError(f"unknown model {name!r}; known models: {known}") from exc
    return cls()
