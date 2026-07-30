"""File I/O: loaders and exporters.

This module is the boundary between afmkit's internal data model and the
outside world.

Loaders (read)
~~~~~~~~~~~~~~
- :class:`~afmkit.io.jpk_txt.JPKTxtLoader` — legacy JPK 4-column ``.txt`` (the
  format produced by JPK Nanowizzard / ForceRobot exports).
- :class:`~afmkit.io.igor_ibw.IgorIBWLoader` — read legacy Igor Binary Wave
  (``.ibw``) data for backward compatibility. Optional — requires
  ``pip install 'afmkit[igor]'``.
- :class:`~afmkit.io.hdf5_store.HDF5Store` — read/write afmkit's native
  HDF5 format (bidirectional; not a :class:`~afmkit.io.base.Loader`).

Exporters (write)
~~~~~~~~~~~~~~~~~
- :mod:`afmkit.io.exporters` — CSV, Matlab ``.mat``, Parquet, Markdown report,
  and (optionally) Igor ``.ibw`` round-trip.

Plugin extension
~~~~~~~~~~~~~~~~
All loaders and exporters register through the central
:class:`~afmkit.io.registry.LoaderRegistry`. Third-party packages can
register new formats via the ``afmkit.loaders`` entry point — see
:mod:`afmkit.plugins`.

Lazy igor_ibw import
~~~~~~~~~~~~~~~~~~~~
The .ibw loader depends on the optional ``igor`` package. Importing
it eagerly at package-load time would make every test that touches
``afmkit.io`` (even indirectly) fail with ImportError on a minimal
install. The public names from :mod:`afmkit.io.igor_ibw` are
therefore exposed through PEP 562 ``__getattr__`` — they resolve on
first access and raise a clear ``ImportError`` pointing to the
``afmkit[igor]`` extra if the package is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from afmkit.io.hdf5_store import HDF5Store, load_hdf5, save_hdf5

# Names exported from afmkit.io.igor_ibw that should be re-exported
# here lazily (so the test suite and other consumers can import
# afmkit.io without having the optional 'igor' package installed).
_IGOR_IBW_NAMES = ("IgorIBWLoader", "load_ibw", "load_ibw_batch", "save_ibw")

__all__ = [
    "HDF5Store",
    "IgorIBWLoader",
    "load_hdf5",
    "load_ibw",
    "load_ibw_batch",
    "save_hdf5",
    "save_ibw",
]

if TYPE_CHECKING:  # pragma: no cover - typing-only re-exports
    from afmkit.io.igor_ibw import (
        IgorIBWLoader,
        load_ibw,
        load_ibw_batch,
        save_ibw,
    )


def __getattr__(name: str) -> Any:
    """PEP 562 lazy import for the optional igor_ibw surface."""
    if name in _IGOR_IBW_NAMES:
        from afmkit.io import igor_ibw

        value = getattr(igor_ibw, name)
        # Cache on the module so subsequent attribute lookups skip the
        # import dance.
        globals()[name] = value
        return value
    raise AttributeError(f"module 'afmkit.io' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
