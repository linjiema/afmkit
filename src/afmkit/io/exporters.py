"""Multi-format exporters for afmkit data.

This module is the **write** counterpart to the loaders in
:mod:`afmkit.io`. It turns a :class:`~afmkit.core.curve.CurveBatch`
(and optionally a list of :class:`FitResult`) into the file formats
analysts actually use day to day:

==================  =============================================================
Function            Output
==================  =============================================================
:func:`to_csv`      Wide column-block CSV (Origin / Excel / Matlab friendly).
:func:`to_csv_fits` One-row-per-fit CSV for fit-result tables.
:func:`to_mat`      Matlab v5 ``.mat`` file. (v7.3 hdf5 backend is a TODO.)
:func:`to_parquet`  Parquet via :mod:`pyarrow` (fastparquet fallback).
:func:`to_markdown` Human-readable Markdown report.
:func:`export`      Dispatch helper — picks the format from ``fmt`` or
                    from the file suffix.
==================  =============================================================

Design notes
------------
- The **wide column-block CSV** is the workhorse format: each curve
  contributes a pair of columns ``ext_NNN`` / ``force_NNN`` (with
  ``NNN`` the zero-padded curve index), so the leading columns of the
  file are directly plottable in Origin, Excel, and Matlab without any
  post-processing.
- The **Markdown** report is a side-channel for sharing — it embeds the
  batch metadata, per-curve statistics, and (optionally) a fit-results
  table in a single self-contained ``.md`` file.
- The **Matlab** writer uses the scipy v5 format (the default of
  :func:`scipy.io.savemat`), which is portable and round-trips through
  loadmat. The v7.3 hdf5 backend — needed for arrays > 2 GB and for
  richer struct support — is a documented TODO.
- The **Parquet** writer prefers :mod:`pyarrow`; if pyarrow is missing
  it falls back to :mod:`fastparquet`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.io

from afmkit.core.curve import CurveBatch

__all__ = [
    "FitResult",
    "to_csv",
    "to_csv_fits",
    "to_mat",
    "to_parquet",
    "to_markdown",
    "export",
]


# -- FitResult -----------------------------------------------------------


@dataclass
class FitResult:
    """Container for the outcome of a single curve fit.

    A minimal, plain-Python dataclass that captures everything an
    exporter (or a downstream visualisation) needs:

    - the **model name** (e.g. ``"WLC"``)
    - the **best-fit parameters** and their **standard errors**
    - the **chi-squared** and **reduced chi-squared** statistics
    - the **number of data points** the fit used
    - the optional **covariance matrix** and the original data

    The fitting engine (see :mod:`afmkit.fitting`) will return objects
    that are structurally compatible with this dataclass — they can be
    passed directly to :func:`to_csv_fits` and :func:`to_markdown`.

    Attributes
    ----------
    model_name
        Human-readable model identifier (e.g. ``"WLC"``).
    params
        Best-fit parameter values, keyed by parameter name. The keys
        match the model's ``param_names``. Example:
        ``{"p": 0.42, "L": 198.0}``.
    param_stderr
        Standard errors for the parameters, keyed by the same names as
        :attr:`params`.
    chi2
        Sum of squared residuals (weighted by measurement
        uncertainties, if known to the fitter).
    redchi
        ``chi2 / (n_data - n_params)`` — the reduced chi-squared.
    n_data
        Number of data points the fit was performed on.
    covariance
        Optional covariance matrix of the parameter estimates.
    x, y
        Optional copies of the data the fit was performed on.
    """

    model_name: str
    params: dict[str, float]
    param_stderr: dict[str, float] = field(default_factory=dict)
    chi2: float = 0.0
    redchi: float = 0.0
    n_data: int = 0
    covariance: np.ndarray | None = None
    x: np.ndarray | None = None
    y: np.ndarray | None = None


# -- Internal helpers ----------------------------------------------------


#: Standard per-curve metadata keys promoted to dedicated CSV columns.
#: The order here is the order the columns appear in the CSV.
_METADATA_KEYS: tuple[str, ...] = ("source_file", "direction", "k_cantilever")


def _padded(arr: np.ndarray, n_max: int) -> np.ndarray:
    """Right-pad ``arr`` with NaN to length ``n_max``."""
    out = np.full(n_max, np.nan)
    out[: arr.size] = arr
    return out


def _metadata_value(meta: dict[str, Any], key: str) -> Any:
    """Return a coerced metadata value, or ``None`` if not usable."""
    if key not in meta:
        return None
    value = meta[key]
    if value is None:
        return None
    return value


def _build_wide_dataframe(batch: CurveBatch, *, include_metadata: bool) -> pd.DataFrame:
    """Build the wide column-block DataFrame used by CSV and Parquet.

    The DataFrame has one row per point index (padded with NaN to the
    longest curve in the batch) and two data columns per curve,
    suffixed with the zero-padded curve index (``ext_000``,
    ``force_000``, ``ext_001``, ``force_001``, ...).

    If ``include_metadata`` is True, three additional per-curve columns
    are appended after the data block: ``source_file_NNN``,
    ``direction_NNN``, and ``k_cantilever_NNN``.
    """
    if batch.n_curves == 0:
        raise ValueError("Cannot export an empty batch")

    n_max = max(c.n_points for c in batch)
    data: dict[str, Any] = {}

    # First pass: data columns (ext, force) for every curve, in
    # interleaved order so the leading columns are directly plottable
    # X/Y pairs.
    for i, curve in enumerate(batch):
        data[f"ext_{i:03d}"] = _padded(curve.extension, n_max)
        data[f"force_{i:03d}"] = _padded(curve.force, n_max)

    # Second pass: per-curve metadata columns, appended after the data
    # block. Strings are stored as object dtype (lists of repeated
    # values); floats are stored as float64 arrays (NaN when missing).
    if include_metadata:
        for i, curve in enumerate(batch):
            meta = curve.metadata
            sf = _metadata_value(meta, "source_file")
            dr = _metadata_value(meta, "direction")
            data[f"source_file_{i:03d}"] = [str(sf)] * n_max if sf is not None else [""] * n_max
            data[f"direction_{i:03d}"] = [str(dr)] * n_max if dr is not None else [""] * n_max
            k = _metadata_value(meta, "k_cantilever")
            try:
                k_float = float(k) if k is not None else float("nan")
            except (TypeError, ValueError):
                k_float = float("nan")
            if np.isfinite(k_float):
                data[f"k_cantilever_{i:03d}"] = np.full(n_max, k_float)
            else:
                data[f"k_cantilever_{i:03d}"] = np.full(n_max, np.nan)

    return pd.DataFrame(data)


# -- to_csv --------------------------------------------------------------


def to_csv(
    batch: CurveBatch,
    path: Path | str,
    *,
    include_metadata: bool = True,
) -> None:
    """Write the batch as a **wide column-block** CSV.

    The output has one row per point index, padded with NaN to the
    length of the longest curve in the batch. Each curve contributes a
    pair of columns — ``ext_NNN`` and ``force_NNN`` — so the leading
    block of the file is directly plottable in Origin, Excel, or
    Matlab::

        ext_000,force_000,ext_001,force_001,...
        0.0,    0.5,      1.0,    0.6,...
        ...

    Parameters
    ----------
    batch
        The :class:`~afmkit.core.curve.CurveBatch` to export.
    path
        Destination file path.
    include_metadata
        When True (default), append three per-curve metadata columns
        — ``source_file_NNN``, ``direction_NNN``, ``k_cantilever_NNN`` —
        after the data columns. These make the file fully
        self-describing when passed to collaborators.
    """
    df = _build_wide_dataframe(batch, include_metadata=include_metadata)
    df.to_csv(Path(path), index=False)


# -- to_csv_fits ---------------------------------------------------------


def to_csv_fits(
    fits: list[FitResult],
    path: Path | str,
) -> None:
    """Write a one-row-per-fit CSV.

    Each row contains the model name, the best-fit parameter values
    (one column per parameter, in the order the parameters first
    appear across the input list), the parameter standard errors
    (columns named ``<name>_stderr``), and the goodness-of-fit
    statistics (``chi2``, ``redchi``, ``n_data``).

    For a batch of WLC fits this yields the column order ``model``,
    ``p``, ``p_stderr``, ``L``, ``L_stderr``, ``chi2``, ``redchi``,
    ``n_data`` — matching the WLC model's
    :attr:`~afmkit.models.wlc.WLCModel.param_names`.

    Parameters
    ----------
    fits
        List of :class:`FitResult` instances.
    path
        Destination file path.
    """
    if not fits:
        raise ValueError("Cannot export an empty list of fits")

    # Collect the union of parameter names, in the order they first
    # appear. For a homogeneous batch of WLC fits this is
    # ``["p", "L"]``.
    all_params: list[str] = []
    for fit in fits:
        for name in fit.params:
            if name not in all_params:
                all_params.append(name)

    rows: list[dict[str, Any]] = []
    for fit in fits:
        row: dict[str, Any] = {
            "model": fit.model_name,
            "chi2": fit.chi2,
            "redchi": fit.redchi,
            "n_data": fit.n_data,
        }
        for name in all_params:
            row[name] = fit.params.get(name, np.nan)
            row[f"{name}_stderr"] = fit.param_stderr.get(name, np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(Path(path), index=False)


# -- to_mat --------------------------------------------------------------


def to_mat(batch: CurveBatch, path: Path | str) -> None:
    """Write the batch as a Matlab v5 ``.mat`` file.

    The file contains a top-level struct with the following fields:

    ``name``
        Batch name (empty string if unset).
    ``n_curves``
        Number of curves in the batch.
    ``n_points``
        Per-curve point counts, shape ``(n_curves,)`` (saved as a
        1-by-N row vector by scipy; recover the 1-D array with
        ``.squeeze()``).
    ``n_points_max``
        Length of the padded data arrays.
    ``extension``
        NaN-padded extensions, shape ``(n_curves, n_points_max)``.
    ``force``
        NaN-padded forces, same shape as ``extension``.
    ``meta_<key>``
        One field per batch-level metadata entry.
    ``curve_metadata_json``
        JSON-encoded per-curve metadata (one string per curve).

    Round-trip
    ----------
    ::

        scipy.io.savemat(path, scipy.io.loadmat(path))   # round-trip
        data = scipy.io.loadmat(path)
        n_curves = int(data["n_curves"].item())
        ext = data["extension"]                          # (n_curves, n_max)
        meta = json.loads(data["curve_metadata_json"][0])

    .. note::
       This implementation uses the scipy v5 format (the default of
       :func:`scipy.io.savemat`). The **v7.3 hdf5 backend** — required
       for arrays > 2 GB and for richer nested struct support — is a
       documented TODO. The migration path is to depend on
       ``hdf5storage`` and pass ``format="7.3"`` to ``savemat``; the
       public API of this function does not need to change.
    """
    if batch.n_curves == 0:
        raise ValueError("Cannot export an empty batch")

    n_max = max(c.n_points for c in batch)
    ext_2d = np.full((batch.n_curves, n_max), np.nan)
    force_2d = np.full((batch.n_curves, n_max), np.nan)
    n_points = np.zeros(batch.n_curves, dtype=np.int64)

    for i, curve in enumerate(batch):
        ext_2d[i, : curve.n_points] = curve.extension
        force_2d[i, : curve.n_points] = curve.force
        n_points[i] = curve.n_points

    mdict: dict[str, Any] = {
        "name": str(batch.name) if batch.name else "",
        "n_curves": np.int64(batch.n_curves),
        "n_points": n_points,
        "n_points_max": np.int64(n_max),
        "extension": ext_2d,
        "force": force_2d,
    }

    # Batch-level metadata as flat ``meta_<key>`` fields. We coerce
    # values to numpy types so scipy.io.savemat doesn't reject them.
    for key, value in batch.metadata.items():
        mdict[f"meta_{key}"] = _coerce_for_matlab(value)

    # Per-curve metadata as a 1-D string array of JSON blobs. Using
    # the inferred string dtype (not ``object``) keeps the layout
    # simple and round-trips through ``loadmat`` without surprises.
    mdict["curve_metadata_json"] = np.array([json.dumps(c.metadata) for c in batch])

    scipy.io.savemat(str(Path(path)), mdict)


def _coerce_for_matlab(value: Any) -> Any:
    """Coerce a Python value into something :func:`scipy.io.savemat` accepts."""
    if isinstance(value, int | float | str | bool):
        return value
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(value)
    except (TypeError, ValueError):
        return str(value)


# -- to_parquet ----------------------------------------------------------


def to_parquet(batch: CurveBatch, path: Path | str) -> None:
    """Write the batch as a wide Parquet file.

    The on-disk layout matches :func:`to_csv` (one row per point index,
    two data columns per curve). Parquet's binary encoding and rich
    schema metadata make this the preferred format for downstream
    pandas / polars analysis.

    Prefers :mod:`pyarrow` as the backend. Falls back to
    :mod:`fastparquet` if pyarrow is not installed. Raises
    :class:`ImportError` if neither backend is available.
    """
    df = _build_wide_dataframe(batch, include_metadata=False)
    path = Path(path)

    try:
        import pyarrow as pa  # type: ignore[import-untyped]
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError as exc:
        try:
            df.to_parquet(str(path), engine="fastparquet")
        except (ImportError, ValueError) as exc2:
            # ``ValueError`` is raised by pandas when the requested
            # engine isn't installed (e.g. older pandas versions).
            raise ImportError(
                "to_parquet requires either pyarrow or fastparquet; "
                f"neither is available (pyarrow: {exc}; fastparquet: {exc2})"
            ) from exc2
    else:
        table = pa.Table.from_pandas(df)
        pq.write_table(table, str(path))


# -- to_markdown ---------------------------------------------------------


def to_markdown(
    batch: CurveBatch,
    fits: list[FitResult] | None,
    path: Path | str,
) -> None:
    """Write a human-readable Markdown report of the batch.

    The report contains three sections:

    1. **Batch summary** — name, number of curves, cantilever spring
       constant, and source folder.
    2. **Per-curve summary** — point count, extension range, and force
       range for each curve.
    3. **Fit results** (if ``fits`` is provided) — one row per fit,
       with the parameter estimates and the goodness-of-fit
       statistics.

    Parameters
    ----------
    batch
        The :class:`~afmkit.core.curve.CurveBatch` to report on.
    fits
        Optional list of :class:`FitResult` instances. If ``None`` (or
        empty), the fit-results section is omitted.
    path
        Destination file path.
    """
    path = Path(path)
    batch_name = batch.name or "Unnamed"

    lines: list[str] = []
    lines.append(f"# {batch_name} — afmkit export")
    lines.append("")

    # -- Batch summary -----------------------------------------------------
    lines.append("## Batch summary")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Name | {batch_name} |")
    lines.append(f"| n_curves | {batch.n_curves} |")
    k_val = batch.metadata.get("k_cantilever", None)
    if isinstance(k_val, int | float) and np.isfinite(float(k_val)):
        lines.append(f"| k_cantilever | {float(k_val):.4g} |")
    else:
        lines.append("| k_cantilever | N/A |")
    src = ""
    if batch.n_curves > 0:
        src = str(batch[0].metadata.get("source_file", "")) or ""
    lines.append(f"| Source | {src} |")
    lines.append("")

    # -- Per-curve summary -------------------------------------------------
    lines.append("## Per-curve summary")
    lines.append("")
    lines.append("| Index | n_points | ext_min | ext_max | force_min | force_max |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for i, curve in enumerate(batch):
        ext = curve.extension
        force = curve.force
        lines.append(
            f"| {i} | {curve.n_points} | {ext.min():.4g} | {ext.max():.4g} | "
            f"{force.min():.4g} | {force.max():.4g} |"
        )
    lines.append("")

    # -- Fit results -------------------------------------------------------
    if fits:
        all_params: list[str] = []
        for fit in fits:
            for name in fit.params:
                if name not in all_params:
                    all_params.append(name)
        lines.append("## Fit results")
        lines.append("")
        header = ["#", "model", *all_params, "chi2", "redchi", "n_data"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for idx, fit in enumerate(fits):
            row: list[str] = [str(idx), fit.model_name]
            for name in all_params:
                val = fit.params.get(name, float("nan"))
                row.append(f"{val:.4g}")
            row.append(f"{fit.chi2:.4g}")
            row.append(f"{fit.redchi:.4g}")
            row.append(str(fit.n_data))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# -- export --------------------------------------------------------------


_FORMAT_FROM_SUFFIX: dict[str, str] = {
    ".csv": "csv",
    ".mat": "mat",
    ".parquet": "parquet",
    ".md": "markdown",
}


def export(
    batch: CurveBatch,
    path: Path | str,
    *,
    fmt: str | None = None,
    fits: list[FitResult] | None = None,
    **kwargs: Any,
) -> None:
    """Dispatch to the right exporter based on ``fmt`` or the file suffix.

    Parameters
    ----------
    batch
        The :class:`~afmkit.core.curve.CurveBatch` to export.
    path
        Destination file path. If ``fmt`` is not given, the suffix is
        used to pick the format (``.csv``, ``.mat``, ``.parquet``,
        ``.md``).
    fmt
        Explicit format identifier. One of ``"csv"``, ``"mat"``,
        ``"parquet"``, ``"markdown"``, ``"csv_fits"``. Case-insensitive.
    fits
        Optional list of :class:`FitResult` instances, used by
        ``fmt="markdown"`` and ``fmt="csv_fits"``.
    **kwargs
        Forwarded to the underlying exporter (e.g. ``include_metadata``
        for :func:`to_csv`).

    Raises
    ------
    ValueError
        If the format is unknown or cannot be inferred from the suffix.
    """
    path = Path(path)

    if fmt is None:
        suffix = path.suffix.lower()
        if suffix not in _FORMAT_FROM_SUFFIX:
            supported = sorted(set(_FORMAT_FROM_SUFFIX.values()) | {"csv_fits"})
            raise ValueError(
                f"Cannot infer format from suffix {suffix!r}. "
                f"Pass fmt= explicitly. Supported: {supported}"
            )
        fmt = _FORMAT_FROM_SUFFIX[suffix]

    fmt_lower = fmt.lower()
    if fmt_lower == "csv":
        to_csv(batch, path, **kwargs)
    elif fmt_lower == "mat":
        to_mat(batch, path, **kwargs)
    elif fmt_lower == "parquet":
        to_parquet(batch, path, **kwargs)
    elif fmt_lower == "markdown":
        to_markdown(batch, fits, path)
    elif fmt_lower == "csv_fits":
        if fits is None:
            raise ValueError("fmt='csv_fits' requires the `fits` argument")
        to_csv_fits(fits, path)
    else:
        supported = sorted(set(_FORMAT_FROM_SUFFIX.values()) | {"csv_fits"})
        raise ValueError(f"Unknown format: {fmt!r}. Supported: {supported}")
