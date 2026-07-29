"""Native HDF5 store for afmkit.

This module implements :class:`HDF5Store`, the canonical round-trip
serialiser for :class:`~afmkit.core.curve.CurveBatch`. The on-disk format
is a single ``.h5`` file with one group per curve; all per-curve and
batch-level metadata is JSON-encoded into HDF5 attributes so the file is
fully self-describing and forward-compatible.

File layout
-----------
::

    <file>/
      afmkit_version: attr  (str, e.g. "0.1.0")            ─┐ root
      batch_name:      attr  (str, may be empty)            │ attrs
      batch_metadata:  attr  (str, JSON-serialised dict)   ─┘
      curves/
        curve_000/
          extension:    dataset  (1-D float64, nm)
          force:        dataset  (1-D float64, pN)
          n_points:     attr     (int)
          metadata:     attr     (str, JSON-serialised dict)
        curve_001/...
        ...

Compression defaults to ``gzip`` level 4 — a good size / CPU tradeoff
for typical force-extension arrays. Datasets with different point
counts (ragged batches) are stored in separate groups, so the ragged
shape is preserved exactly.

The store is *intentionally not* a :class:`~afmkit.io.base.Loader`
implementation: it is bidirectional (read **and** write), whereas the
``Loader`` protocol is read-only. Loader-style auto-discovery is not
needed either — HDF5 files are typically opened explicitly by the user.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from afmkit import __version__
from afmkit.core.curve import CurveBatch, ForceCurve

__all__ = ["HDF5Store", "load_hdf5", "save_hdf5"]


# -- Constants ------------------------------------------------------------

#: Group / dataset / attribute names used in the on-disk format. Centralised
#: so that the writer, reader, and any future migration code agree.
_CURVES_GROUP = "curves"
_VERSION_ATTR = "afmkit_version"
_BATCH_NAME_ATTR = "batch_name"
_BATCH_METADATA_ATTR = "batch_metadata"
_CURVE_METADATA_ATTR = "metadata"
_CURVE_NPOINTS_ATTR = "n_points"
_CURVE_EXT_DSET = "extension"
_CURVE_FORCE_DSET = "force"

#: Number of digits used to name curve groups (``curve_000``, ``curve_001``, …).
#: Allows up to 10**4 - 1 = 9999 curves per file before naming collides —
#: comfortably above the ~100 curves/folder lab workflow.
_CURVE_NAME_DIGITS = 4

#: Modes that :meth:`HDF5Store.save` accepts. ``h5py`` itself accepts more
#: (``r+``, ``x``, …) but the documented afmkit contract is just these two.
_SUPPORTED_MODES = frozenset({"w", "a"})


# -- JSON helpers ---------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """JSON encoder fallback for non-native types in metadata dicts.

    Handles the common cases that appear in afmkit metadata:

    - ``np.ndarray`` → list (via :meth:`ndarray.tolist`).
    - ``np.generic`` scalars (``np.float64``, ``np.int32``, …) → Python
      native via :meth:`ndarray.item`.
    - :class:`pathlib.Path` → string.
    - Anything else → string via :func:`str`. This is a best-effort
      fallback; intentionally permissive so user-defined metadata
      survives a round-trip without raising.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _encode_json(obj: Any) -> str:
    """Serialise a metadata dict to a JSON string.

    The result is always valid UTF-8 text and never contains NaN/Inf
    markers (``allow_nan=False``) — afmkit metadata is curated user data
    and a stray NaN there indicates a bug, not a legitimate value.
    """
    return json.dumps(obj, default=_json_default, allow_nan=False, sort_keys=True)


def _decode_json(text: str) -> Any:
    """Inverse of :func:`_encode_json`. Always returns a Python object."""
    return json.loads(text)


# -- Internal save / load helpers ----------------------------------------


def _curve_group_name(index: int) -> str:
    """Format a zero-padded curve group name (``curve_0001``)."""
    return f"curve_{index:0{_CURVE_NAME_DIGITS}d}"


def _write_curve(
    grp: h5py.Group,
    curve: ForceCurve,
    *,
    compression: str,
    compression_opts: int | None,
) -> None:
    """Write a single :class:`ForceCurve` into the supplied group."""
    grp.attrs[_CURVE_NPOINTS_ATTR] = int(curve.n_points)
    grp.attrs[_CURVE_METADATA_ATTR] = _encode_json(curve.metadata)
    grp.create_dataset(
        _CURVE_EXT_DSET,
        data=curve.extension,
        compression=compression,
        compression_opts=compression_opts,
    )
    grp.create_dataset(
        _CURVE_FORCE_DSET,
        data=curve.force,
        compression=compression,
        compression_opts=compression_opts,
    )


def _next_curve_index(curves_group: h5py.Group) -> int:
    """Return the smallest curve index that does not yet exist in ``curves_group``.

    Used by :meth:`HDF5Store.save` in ``mode="a"`` to avoid clobbering
    existing curve groups.
    """
    used: set[int] = set()
    for name in curves_group:
        if not isinstance(name, str):
            continue
        prefix = "curve_"
        if not name.startswith(prefix):
            continue
        try:
            used.add(int(name[len(prefix) :]))
        except ValueError:
            continue
    if not used:
        return 0
    return max(used) + 1


def _sorted_curve_names(curves_group: h5py.Group) -> list[str]:
    """Return curve group names sorted by their trailing integer."""
    items: list[tuple[int, str]] = []
    for name in curves_group:
        if not isinstance(name, str) or not name.startswith("curve_"):
            continue
        try:
            idx = int(name[len("curve_") :])
        except ValueError:
            continue
        items.append((idx, name))
    items.sort()
    return [name for _, name in items]


def _read_curve(grp: h5py.Group) -> ForceCurve:
    """Reconstruct a :class:`ForceCurve` from a per-curve group.

    Validates the required datasets / attributes and raises
    :class:`ValueError` with a file-specific message otherwise.
    """
    for required in (_CURVE_EXT_DSET, _CURVE_FORCE_DSET):
        if required not in grp:
            raise ValueError(f"curve group {grp.name!r} is missing required dataset {required!r}")

    ext = np.asarray(grp[_CURVE_EXT_DSET][...], dtype=np.float64)
    force = np.asarray(grp[_CURVE_FORCE_DSET][...], dtype=np.float64)

    # Per-curve metadata is optional — missing or empty means {}.
    if _CURVE_METADATA_ATTR in grp.attrs:
        meta_text = grp.attrs[_CURVE_METADATA_ATTR]
        metadata = _decode_json(meta_text) if meta_text else {}
        if not isinstance(metadata, dict):
            raise ValueError(
                f"curve group {grp.name!r} has a non-dict metadata value: {type(metadata).__name__}"
            )
    else:
        metadata = {}

    return ForceCurve(ext, force, metadata=metadata)


# -- HDF5Store ------------------------------------------------------------


class HDF5Store:
    """Read / write afmkit data to HDF5 (built on :mod:`h5py`).

    The class is stateless across calls — all per-file options are
    passed as method arguments. This keeps the public surface area
    small and makes it easy to share a single instance across an
    analysis pipeline.

    Examples
    --------
    >>> import numpy as np
    >>> from afmkit.core.curve import CurveBatch, ForceCurve
    >>> from afmkit.io.hdf5_store import HDF5Store
    >>> store = HDF5Store()
    >>> c = ForceCurve(np.linspace(0.0, 100.0, 50), np.zeros(50),
    ...                metadata={"k_cantilever": 0.1})
    >>> batch = CurveBatch([c], name="demo")
    """

    #: Short identifier. The store is *not* a :class:`~afmkit.io.base.Loader`
    #: (it is bidirectional), but the name is still useful for logging.
    name: str = "hdf5"

    # -- Public API -------------------------------------------------------

    def save(
        self,
        batch: CurveBatch,
        path: Path | str,
        *,
        mode: str = "w",
        compression: str = "gzip",
        compression_opts: int | None = 4,
    ) -> None:
        """Save a :class:`CurveBatch` to ``path``.

        Parameters
        ----------
        batch
            The batch to serialise. Per-curve ragged shapes are preserved
            — each curve lives in its own subgroup.
        path
            Destination ``.h5`` file. Parent directories are created
            if missing. May be a :class:`str` or :class:`pathlib.Path`.
        mode
            ``"w"`` truncates any existing file before writing (the
            default). ``"a"`` opens an existing afmkit file and appends
            the new curves; the file's top-level attributes (``batch_name``,
            ``batch_metadata``, ``afmkit_version``) are left untouched.
            Any other mode string is rejected with :class:`ValueError`.
        compression
            HDF5 compression filter passed to
            :meth:`h5py.Group.create_dataset` (e.g. ``"gzip"``, ``"lzf"``,
            or ``None`` for no compression).
        compression_opts
            Filter-specific options (e.g. gzip level 0-9). ``None`` means
            "use the library default" for the chosen filter.
        """
        if mode not in _SUPPORTED_MODES:
            raise ValueError(f"mode must be one of {sorted(_SUPPORTED_MODES)}; got {mode!r}")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(path, mode) as fh:
            if mode == "w":
                fh.attrs[_VERSION_ATTR] = __version__
                fh.attrs[_BATCH_NAME_ATTR] = batch.name or ""
                fh.attrs[_BATCH_METADATA_ATTR] = _encode_json(batch.metadata)
                curves_group = fh.require_group(_CURVES_GROUP)
                # Wipe any prior curves so a re-save starts from zero.
                for name in list(curves_group):
                    del curves_group[name]
                start_index = 0
            else:  # mode == "a"
                curves_group = fh.require_group(_CURVES_GROUP)
                start_index = _next_curve_index(curves_group)

            for offset, curve in enumerate(batch):
                grp = curves_group.create_group(_curve_group_name(start_index + offset))
                _write_curve(
                    grp,
                    curve,
                    compression=compression,
                    compression_opts=compression_opts,
                )

    def load(self, path: Path | str) -> CurveBatch:
        """Load a :class:`CurveBatch` from ``path``.

        Parameters
        ----------
        path
            An HDF5 file previously written by :meth:`save`.

        Returns
        -------
        CurveBatch
            The reconstructed batch, with all per-curve and batch-level
            metadata restored.

        Raises
        ------
        ValueError
            If ``path`` is not a valid afmkit HDF5 store — either
            because the file is not HDF5 at all, or because the
            required ``afmkit_version`` root attribute is missing.
        """
        path = Path(path)
        try:
            fh = h5py.File(path, "r")
        except OSError as exc:
            # h5py raises OSError for two distinct failure modes we
            # need to disambiguate: (a) the file is not HDF5, (b) the
            # file is missing on disk. Both should surface as
            # ValueError per the API contract, with messages that
            # clearly point at the underlying cause.
            if path.exists():
                raise ValueError(f"file {path} is not a valid HDF5 file: {exc}") from exc
            raise ValueError(f"HDF5 file not found: {path}") from exc

        try:
            if _VERSION_ATTR not in fh.attrs:
                raise ValueError(
                    f"{path} is not an afmkit HDF5 store: "
                    f"missing required root attribute {_VERSION_ATTR!r}"
                )

            batch_name_raw = fh.attrs.get(_BATCH_NAME_ATTR, "")
            batch_name = str(batch_name_raw) if batch_name_raw else None

            if _BATCH_METADATA_ATTR in fh.attrs:
                batch_meta_text = fh.attrs[_BATCH_METADATA_ATTR]
                batch_metadata = _decode_json(batch_meta_text) if batch_meta_text else {}
                if not isinstance(batch_metadata, dict):
                    raise ValueError(
                        f"{path} has a non-dict batch_metadata value: "
                        f"{type(batch_metadata).__name__}"
                    )
            else:
                batch_metadata = {}

            curves_group = fh.get(_CURVES_GROUP)
            if curves_group is None:
                # A valid afmkit store always has a `curves` group,
                # even if empty. Tolerate a missing group rather than
                # failing the load — the file is still afmkit-shaped.
                curves: list[ForceCurve] = []
            else:
                curves = [
                    _read_curve(curves_group[name]) for name in _sorted_curve_names(curves_group)
                ]

            return CurveBatch(curves, name=batch_name, metadata=batch_metadata)
        finally:
            fh.close()


# -- Module-level convenience --------------------------------------------


def save_hdf5(batch: CurveBatch, path: Path | str, **kwargs: Any) -> None:
    """Convenience wrapper for :meth:`HDF5Store.save`.

    Any additional keyword arguments are forwarded to
    :meth:`HDF5Store.save` (``mode``, ``compression``, …).
    """
    HDF5Store().save(batch, path, **kwargs)


def load_hdf5(path: Path | str) -> CurveBatch:
    """Convenience wrapper for :meth:`HDF5Store.load`."""
    return HDF5Store().load(path)
