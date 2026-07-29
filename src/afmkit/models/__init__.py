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
- More models (eWLC, FJC, twist-WLC) are planned as optional plugin packages.

Custom models
~~~~~~~~~~~~~
A user-defined model is just a class that satisfies the protocol and is
registered with the model registry — no monkey-patching, no global state.
"""

from __future__ import annotations
