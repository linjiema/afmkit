"""File I/O: loaders and exporters.

This module is the boundary between afmkit's internal data model and the
outside world.

Loaders (read)
~~~~~~~~~~~~~~
- :class:`~afmkit.io.jpk_txt.JPKTxtLoader` — legacy JPK 4-column ``.txt`` (the
  format produced by JPK Nanowizzard / ForceRobot exports).
- :class:`~afmkit.io.igor_ibw.IgorIBWLoader` — read legacy Igor Binary Wave
  (``.ibw``) data for backward compatibility.
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
"""

from __future__ import annotations

from afmkit.io.hdf5_store import HDF5Store, load_hdf5, save_hdf5
from afmkit.io.igor_ibw import (
    IgorIBWLoader,
    load_ibw,
    load_ibw_batch,
    save_ibw,
)

__all__ = [
    "HDF5Store",
    "IgorIBWLoader",
    "load_hdf5",
    "load_ibw",
    "load_ibw_batch",
    "save_hdf5",
    "save_ibw",
]
