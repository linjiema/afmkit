"""Loader protocol — the common interface for all file-format loaders.

Every concrete loader (JPK 4-column ``.txt``, Igor ``.ibw``, HDF5, …) is
expected to expose a ``name`` attribute and two methods:
:meth:`Loader.can_load` and :meth:`Loader.load`. Loaders are *stateless*
in spirit — they transform an on-disk file into a
:class:`~afmkit.core.curve.CurveBatch`, which is the canonical in-memory
representation used by the rest of afmkit.

A class satisfying this protocol is the unit of registration for both
the built-in registry (:mod:`afmkit.io.registry`) and the pluggy entry
point ``afmkit.loaders`` — see :mod:`afmkit.plugins`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from afmkit.core.curve import CurveBatch

__all__ = ["Loader"]


class Loader(Protocol):
    """The interface every file-format loader must satisfy.

    Loaders should be cheap to construct and stateless across calls —
    all per-file options are passed through ``**kwargs`` of
    :meth:`load`, not constructor arguments. This keeps the registry /
    plugin system simple: a single shared instance is reused for every
    file.

    Attributes
    ----------
    name
        Short, lowercase identifier for the format (e.g. ``"jpk_txt"``,
        ``"igor_ibw"``). Used as the key in the
        :class:`~afmkit.io.registry.LoaderRegistry`.
    """

    name: str

    def can_load(self, path: Path) -> bool:
        """Return ``True`` if this loader can read the file at ``path``.

        Implementations should perform only cheap checks (file existence,
        extension, magic bytes, first-line sniff). Avoid opening or
        parsing the full file — :meth:`can_load` is called on every
        candidate file during auto-discovery.
        """
        ...

    def load(self, path: Path, **kwargs: Any) -> CurveBatch:
        """Load ``path`` and return a :class:`CurveBatch`.

        Format-specific options (e.g. cantilever spring constant for JPK)
        are passed as keyword arguments; each implementation documents
        the supported keys in its own docstring.
        """
        ...
