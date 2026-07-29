"""pluggy-based plugin extension points.

Third-party packages can extend afmkit by implementing one of the hookspec
functions below and registering their implementation through the standard
Python entry-points mechanism in their own ``pyproject.toml``:

.. code-block:: toml

    [project.entry-points."afmkit.loaders"]
    bruker = "afmkit_bruker:BrukerLoader"

    [project.entry-points."afmkit.models"]
    fjc = "afmkit_fjc:FJCModel"

Once installed (``pip install afmkit-bruker``), afmkit will auto-discover
the loader and make it available through :func:`afmkit.io.registry.get_loader`.

Hookspecs
---------
- :func:`register_loader`  — register a new file format loader.
- :func:`register_model`   — register a new polymer physics model.
- :func:`register_baseline`— register a new baseline correction strategy.
- :func:`register_fitter`  — register a new fitting engine backend.
"""

from __future__ import annotations

import pluggy

# Single global hookspec namespace. Using pluggy keeps the registration
# semantics identical to pytest, so plugin authors have one mental model.
hookspec = pluggy.HookspecMarker("afmkit")
hookimpl = pluggy.HookimplMarker("afmkit")

# Project name used by pluggy to discover third-party entry points.
PROJECT_NAME = "afmkit"


# -- Hookspecs ------------------------------------------------------------


@hookspec
def register_loader() -> type:
    """Return a ``Loader`` subclass to register as a new file format.

    The class must implement the protocol declared in
    :mod:`afmkit.io.base`. Returning a class (not an instance) lets afmkit
    defer construction until it actually needs to read a file.
    """


@hookspec
def register_model() -> type:
    """Return a ``PolymerModel`` subclass to register."""


@hookspec
def register_baseline() -> type:
    """Return a ``BaselineCorrector`` subclass to register."""


@hookspec
def register_fitter() -> type:
    """Return a ``Fitter`` subclass to register."""


# -- Plugin manager singleton --------------------------------------------


def get_plugin_manager() -> pluggy.PluginManager:
    """Return the singleton plugin manager, with all built-ins loaded.

    Built-in entry points (those shipped in ``afmkit`` itself) are
    registered directly. Third-party entry points are discovered from
    installed distributions.
    """
    pm = pluggy.PluginManager(PROJECT_NAME)
    pm.add_hookspecs(__name__)
    # Built-in registrations are added by the IO/models submodules at
    # import time via register().
    pm.load_setuptools_entrypoints(PROJECT_NAME)
    return pm
