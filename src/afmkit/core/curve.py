"""Core data model: :class:`ForceCurve` and :class:`CurveBatch`.

The two classes defined here are the lingua franca of afmkit. Every other
layer (IO, processing, fitting, analysis, presentation) consumes or
produces them, so the contracts documented below are the public API of
the library.

Conventions
-----------
- All extension values are stored in **nanometres [nm]**.
- All force values are stored in **piconewtons [pN]**.
- The ``extension`` axis is named ``"point"`` in the underlying xarray
  Dataset; downstream code should not assume specific numeric indices.
- Metadata is a free-form ``dict`` (validated in later versions).
- A :class:`ForceCurve` is treated as **immutable** in spirit: every
  transformation returns a new instance.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import xarray as xr

__all__ = ["CurveBatch", "ForceCurve"]


# -- Helpers --------------------------------------------------------------


def _ensure_1d_finite(name: str, arr: Any) -> np.ndarray:
    """Coerce to a 1D float64 array and reject NaN/Inf.

    Parameters
    ----------
    name
        Used in error messages — should describe what ``arr`` represents.
    arr
        Anything convertible to a numpy array.

    Raises
    ------
    TypeError
        If the input cannot be coerced to a 1D numeric array.
    ValueError
        If the input contains NaN or Inf, or has zero length.
    """
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {out.shape}")
    if out.size == 0:
        raise ValueError(f"{name} must contain at least one point")
    if not np.all(np.isfinite(out)):
        n_bad = int(np.sum(~np.isfinite(out)))
        raise ValueError(f"{name} contains {n_bad} non-finite value(s)")
    return out


# -- ForceCurve -----------------------------------------------------------


class ForceCurve:
    """A single force-extension measurement.

    Wraps an :class:`xarray.Dataset` with two data variables:

    - ``"extension"`` — extension in **nm**.
    - ``"force"``     — force in **pN**.

    All acquisition-time metadata (cantilever ``k``, temperature, source
    file, direction, …) lives in the underlying Dataset's ``attrs``
    dictionary and is exposed as :attr:`metadata`.

    Parameters
    ----------
    extension : array-like
        Extension values, in nm. Must be 1-D, finite, and the same length
        as ``force``.
    force : array-like
        Force values, in pN. Must be 1-D and finite.
    metadata : dict, optional
        Free-form metadata. Common keys:

        - ``"k_cantilever"`` — spring constant, pN/nm.
        - ``"temperature"``  — sample temperature, K.
        - ``"source_file"``  — origin file path or name.
        - ``"direction"``    — ``"approach"`` or ``"retract"``.

    Examples
    --------
    >>> import numpy as np
    >>> from afmkit.core.curve import ForceCurve
    >>> ext = np.linspace(0.5, 199.5, 1000)  # avoid WLC singularity at x=L
    >>> p, lc = 0.4, 200.0
    >>> force = (4.1 / p) * (0.25 * (1 - ext / lc) ** -2 - 0.25 + ext / lc)
    >>> curve = ForceCurve(ext, force, metadata={"k_cantilever": 0.1})
    >>> curve.n_points
    1000
    >>> curve.extension.shape
    (1000,)
    """

    __slots__ = ("_ds",)

    # Names of the two data variables — kept as class constants so that
    # downstream code (IO, exporters) can refer to them without
    # hard-coding strings.
    EXTENSION: str = "extension"
    FORCE: str = "force"
    POINT: str = "point"

    def __init__(
        self,
        extension: Any,
        force: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        ext = _ensure_1d_finite("extension", extension)
        frc = _ensure_1d_finite("force", force)
        if ext.shape != frc.shape:
            raise ValueError(
                f"extension and force must have the same length; got {ext.shape} vs {frc.shape}"
            )
        ds = xr.Dataset(
            {
                self.EXTENSION: (self.POINT, ext),
                self.FORCE: (self.POINT, frc),
            },
            attrs=dict(metadata) if metadata else {},
        )
        self._ds = ds

    # -- Properties -------------------------------------------------------

    @property
    def extension(self) -> np.ndarray:
        """Extension values in nm (read-only view)."""
        return self._ds[self.EXTENSION].values

    @property
    def force(self) -> np.ndarray:
        """Force values in pN (read-only view)."""
        return self._ds[self.FORCE].values

    @property
    def metadata(self) -> dict[str, Any]:
        """Metadata dict. The returned dict is a shallow copy — mutating
        it does **not** affect the underlying curve. Use
        :meth:`with_metadata` to produce a new curve with updates."""
        return dict(self._ds.attrs)

    @property
    def n_points(self) -> int:
        """Number of data points in the curve."""
        return int(self._ds.sizes[self.POINT])

    # -- xarray interop ---------------------------------------------------

    def to_xarray(self) -> xr.Dataset:
        """Return a deep copy of the underlying xarray Dataset.

        The returned object is independent of this curve; modifying it
        will not affect the curve.
        """
        return self._ds.copy(deep=True)

    @classmethod
    def from_xarray(cls, ds: xr.Dataset) -> ForceCurve:
        """Build a :class:`ForceCurve` from an existing xarray Dataset.

        The dataset must contain both the ``"extension"`` and ``"force"``
        data variables on the ``"point"`` dimension; everything else
        (``attrs``, coordinate variables) is preserved.
        """
        missing = [v for v in (cls.EXTENSION, cls.FORCE) if v not in ds.data_vars]
        if missing:
            raise ValueError(f"xarray Dataset is missing required variables: {missing}")
        # Re-run the construction path so the same validation rules apply.
        return cls(
            ds[cls.EXTENSION].values,
            ds[cls.FORCE].values,
            metadata=dict(ds.attrs),
        )

    # -- Transformations (return new ForceCurves) -------------------------

    def select_range(self, x_min: float, x_max: float, *, inclusive: bool = True) -> ForceCurve:
        """Return a new curve restricted to ``x_min <= extension <= x_max``.

        Parameters
        ----------
        x_min, x_max
            Extension bounds in nm. Order is enforced; ``x_max < x_min``
            raises :class:`ValueError`.
        inclusive
            If True (default), both endpoints are included.

        Notes
        -----
        The implementation handles non-monotonic extension axes (e.g.
        approach+retract within a single curve) by indexing where
        ``x_min <= ext <= x_max`` rather than slicing by index — this is
        a robustness improvement over the original Igor implementation,
        which assumed a monotonically increasing axis.
        """
        if x_min > x_max:
            raise ValueError(f"x_min ({x_min}) must be <= x_max ({x_max})")
        ext = self.extension
        mask = (ext >= x_min) & (ext <= x_max) if inclusive else (ext > x_min) & (ext < x_max)
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            raise ValueError(f"select_range({x_min}, {x_max}) produced an empty curve")
        sub = self._ds.isel({self.POINT: indices})
        return ForceCurve.from_xarray(sub)

    def with_metadata(self, **updates: Any) -> ForceCurve:
        """Return a new curve with the given metadata keys replaced/added.

        Existing keys not mentioned in ``updates`` are preserved.

        >>> c = ForceCurve([0, 1], [0, 1], metadata={"k_cantilever": 0.1})
        >>> c2 = c.with_metadata(k_cantilever=0.12, operator="alice")
        >>> c2.metadata["k_cantilever"]
        0.12
        >>> c2.metadata["operator"]
        'alice'
        """
        new_meta = {**self.metadata, **updates}
        ds = self._ds.copy()
        ds.attrs = new_meta
        return ForceCurve.from_xarray(ds)

    def with_force(self, force: Any) -> ForceCurve:
        """Return a new curve with the force axis replaced."""
        frc = _ensure_1d_finite("force", force)
        if frc.shape != self.extension.shape:
            raise ValueError(
                f"new force array must match extension length "
                f"({self.extension.shape}); got {frc.shape}"
            )
        ds = self._ds.copy()
        ds[self.FORCE] = (self.POINT, frc)
        return ForceCurve.from_xarray(ds)

    # -- Misc -------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_points

    def __repr__(self) -> str:
        meta_summary = (
            f" k={self.metadata['k_cantilever']}" if "k_cantilever" in self.metadata else ""
        )
        return (
            f"ForceCurve(n_points={self.n_points}, "
            f"ext=[{self.extension.min():.1f}, {self.extension.max():.1f}] nm, "
            f"F=[{self.force.min():.1f}, {self.force.max():.1f}] pN,"
            f"{meta_summary})"
        )


# -- CurveBatch -----------------------------------------------------------


class CurveBatch:
    """An ordered collection of :class:`ForceCurve` objects.

    A :class:`CurveBatch` is the unit returned by every loader and the
    unit consumed by every batch operation in :mod:`afmkit.analysis`.
    It is intentionally a thin wrapper over a Python list — for
    100-curve lab folders this is faster and simpler than a fancier
    2-D structure, and avoids alignment issues when curves have
    different point counts.

    Parameters
    ----------
    curves
        Iterable of :class:`ForceCurve` instances.
    name : str, optional
        Human-readable label, e.g. the source folder name.
    metadata : dict, optional
        Shared batch-level metadata (e.g. a single cantilever
        calibration applying to all curves).
    """

    __slots__ = ("_curves", "metadata", "name")

    def __init__(
        self,
        curves: Iterable[ForceCurve],
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        items = list(curves)
        for c in items:
            if not isinstance(c, ForceCurve):
                raise TypeError(
                    f"CurveBatch only accepts ForceCurve instances; got {type(c).__name__}"
                )
        self._curves: list[ForceCurve] = items
        self.name = name
        self.metadata: dict[str, Any] = dict(metadata) if metadata else {}

    # -- Container protocol ----------------------------------------------

    def __len__(self) -> int:
        return len(self._curves)

    def __iter__(self) -> Iterator[ForceCurve]:
        return iter(self._curves)

    def __getitem__(self, idx: int | slice) -> ForceCurve | CurveBatch:
        if isinstance(idx, slice):
            return CurveBatch(self._curves[idx], name=self.name, metadata=self.metadata)
        return self._curves[idx]

    def __repr__(self) -> str:
        label = f"'{self.name}' " if self.name else ""
        return f"CurveBatch({label}n_curves={len(self._curves)})"

    # -- Convenience -----------------------------------------------------

    @property
    def n_curves(self) -> int:
        """Number of curves in the batch."""
        return len(self._curves)

    def extensions(self) -> Iterator[np.ndarray]:
        """Iterate over extension arrays of all curves."""
        return (c.extension for c in self._curves)

    def forces(self) -> Iterator[np.ndarray]:
        """Iterate over force arrays of all curves."""
        return (c.force for c in self._curves)

    def select(self, indices: Iterable[int]) -> CurveBatch:
        """Return a new :class:`CurveBatch` containing only the selected indices."""
        idx_list = list(indices)
        chosen = [self._curves[i] for i in idx_list]
        return CurveBatch(chosen, name=self.name, metadata=self.metadata)

    def to_xarray(self) -> xr.Dataset:
        """Combine all curves into a single :class:`xarray.Dataset`.

        The resulting dataset has one dimension per curve (``curve``) and
        a second ragged dimension (``point``) — meaning the dataset is
        effectively a list, not a matrix. This is useful for bulk
        exporters (CSV, parquet) but is *not* the canonical in-memory
        representation: downstream code should keep iterating over the
        :class:`ForceCurve` objects.
        """
        if not self._curves:
            return xr.Dataset(attrs=dict(self.metadata))

        ext_list = [c.extension for c in self._curves]
        force_list = [c.force for c in self._curves]
        meta_list = [c.metadata for c in self._curves]

        # Pad to a common length with NaN to make the Dataset rectangular.
        # (We don't lose data — the original ForceCurves are still here.)
        n_max = max(e.size for e in ext_list)
        ext_pad = np.full((len(self._curves), n_max), np.nan)
        force_pad = np.full((len(self._curves), n_max), np.nan)
        for i, (e, f) in enumerate(zip(ext_list, force_list, strict=True)):
            ext_pad[i, : e.size] = e
            force_pad[i, : f.size] = f

        return xr.Dataset(
            {
                ForceCurve.EXTENSION: (("curve", "point"), ext_pad),
                ForceCurve.FORCE: (("curve", "point"), force_pad),
            },
            attrs={
                "batch_name": self.name or "",
                "batch_metadata": self.metadata,
                "curve_metadata": meta_list,
            },
        )
