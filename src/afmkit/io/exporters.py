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
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.io

from afmkit.core.curve import CurveBatch

if TYPE_CHECKING:
    from afmkit.analysis.peak_review import PeakReviewer

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


#: Columns emitted by :func:`to_csv_fits` for every peak row when
#: ``reviewers`` is provided. Mirrors the keys of
#: :meth:`PeakReviewer.to_dict` with the addition of ``curve_index`` so
#: the output is joinable back to the per-fit table on
#: ``(curve_index, peak_index)``.
_PEAK_REVIEW_COLUMNS: tuple[str, ...] = (
    "curve_index",
    "peak_index",
    "extension_nm",
    "force_pN",
    "manual_force_pN",
    "accepted",
    "confidence",
    "prominence_pN",
    "width_points",
    "height_drop_pN",
    "note",
)


def to_csv_fits(
    fits: list[FitResult],
    path: Path | str,
    *,
    reviewers: dict[int, PeakReviewer] | None = None,
) -> None:
    """Write a one-row-per-fit CSV, optionally extended with per-peak review rows.

    When ``reviewers`` is ``None`` (the default), the output is one row
    per fit: the model name, the best-fit parameter values (one
    column per parameter, in the order the parameters first appear
    across the input list), the parameter standard errors (columns
    named ``<name>_stderr``), and the goodness-of-fit statistics
    (``chi_square``, ``reduced_chi_square``, ``n_data``).

    For a batch of WLC fits this yields the column order ``model``,
    ``p``, ``p_stderr``, ``L``, ``L_stderr``, ``chi_square``,
    ``reduced_chi_square``, ``n_data`` — matching the WLC model's
    :attr:`~afmkit.models.wlc.WLCModel.param_names`.

    When ``reviewers`` is provided, the output shape changes:
    **one row per (curve, peak)** instead of one row per fit. Each
    row carries the fit's columns (model, params, std-errors,
    goodness-of-fit) plus the peak columns emitted by
    :meth:`PeakReviewer.to_dict`. Curves with a reviewer but zero
    peaks get a single "no peaks" row with all peak columns as
    empty / NaN. Curves without a reviewer (or with a reviewer
    missing from the dict) get a single row with the same empty
    peak columns, so every fit in the input list is still
    represented.

    Parameters
    ----------
    fits
        List of :class:`~afmkit.fitting.report.FitResult` instances,
        in curve-index order.
    path
        Destination file path.
    reviewers
        Optional ``{curve_index: PeakReviewer}`` mapping. When
        provided, the per-peak accept / reject / manual_force / note
        state flows into the output. The mapping is keyed by the
        0-based curve index (``fits[i]`` corresponds to
        ``reviewers[i]``). Curve indices that have a fit but no
        reviewer entry get a row with empty peak columns. Curve
        indices in the mapping without a corresponding fit raise
        :class:`ValueError`.
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

    if reviewers is None:
        _write_per_fit_csv(fits, all_params, Path(path))
    else:
        _write_per_peak_csv(fits, all_params, reviewers, Path(path))


def _write_per_fit_csv(
    fits: list[FitResult],
    all_params: list[str],
    path: Path,
) -> None:
    """Write the legacy one-row-per-fit CSV (the v0.1 → v0.3 shape)."""
    rows: list[dict[str, Any]] = []
    for fit in fits:
        # The exporters accept both the new (afmkit.fitting.report)
        # FitResult and the legacy io-side dataclass — fall back to the
        # legacy attribute names so older code paths keep working.
        stderr = getattr(fit, "stderr", None) or getattr(fit, "param_stderr", {}) or {}
        chi2 = getattr(fit, "chi_square", None)
        if chi2 is None:
            chi2 = getattr(fit, "chi2", float("nan"))
        redchi = getattr(fit, "reduced_chi_square", None)
        if redchi is None:
            redchi = getattr(fit, "redchi", float("nan"))

        row: dict[str, Any] = {
            "model": fit.model_name,
            "chi_square": chi2,
            "reduced_chi_square": redchi,
            "n_data": fit.n_data,
        }
        for name in all_params:
            row[name] = fit.params.get(name, np.nan)
            row[f"{name}_stderr"] = stderr.get(name, np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _write_per_peak_csv(
    fits: list[FitResult],
    all_params: list[str],
    reviewers: dict[int, PeakReviewer],
    path: Path,
) -> None:
    """Write the per-peak CSV (the v0.4+ shape when ``reviewers`` is set).

    Every input fit contributes at least one row. A fit whose
    curve index has a reviewer with at least one peak contributes
    one row per peak. A fit whose curve index has a reviewer with
    zero peaks, or no reviewer at all, contributes a single row
    with the peak columns empty.
    """
    _validate_reviewers_mapping(reviewers, len(fits))
    rows = _build_peak_review_rows(fits, all_params, reviewers)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _build_peak_review_rows(
    fits: list[FitResult],
    all_params: list[str],
    reviewers: dict[int, PeakReviewer],
) -> list[dict[str, Any]]:
    """Build the per-(curve, peak) review rows shared by all v0.4+ exporters.

    Each row is a flat dict with the per-fit columns (model, params,
    std-errors, goodness-of-fit) plus the per-peak review columns
    (``curve_index``, ``peak_index``, ``extension_nm``, ``force_pN``,
    ``manual_force_pN``, ``accepted``, ``confidence``, ``prominence_pN``,
    ``width_points``, ``height_drop_pN``, ``note``). The CSV writer
    drops these straight into a :class:`pandas.DataFrame`; the
    ``.mat`` writer coerces them into a struct array; the parquet
    writer passes them to ``pyarrow.Table.from_pandas``.

    A fit whose curve index has a reviewer with at least one peak
    contributes one row per peak. A fit whose curve index has a
    reviewer with zero peaks, or no reviewer at all, contributes a
    single row with the peak columns set to their empty values
    (NaN / ``""`` / ``False`` / ``-1`` for ints) so every fit is
    still represented.
    """
    rows: list[dict[str, Any]] = []
    for curve_idx, fit in enumerate(fits):
        fit_cols = _fit_columns(fit, all_params)
        reviewer = reviewers.get(curve_idx)
        if reviewer is None or len(reviewer) == 0:
            # No peaks to emit — but the fit still gets a row, with
            # the peak columns empty so the per-fit shape is
            # preserved.
            row = dict(fit_cols)
            row["curve_index"] = curve_idx
            for col in _PEAK_REVIEW_COLUMNS[1:]:
                row[col] = _empty_peak_value(col)
            rows.append(row)
            continue
        for peak_dict in reviewer.to_dict():
            row = dict(fit_cols)
            row["curve_index"] = curve_idx
            # Map the to_dict() keys onto the v0.4 CSV column names
            # (units-suffixed for clarity when the file is opened in
            # Excel / Origin).
            row["peak_index"] = int(peak_dict["index"])
            row["extension_nm"] = float(peak_dict["extension"])
            row["force_pN"] = float(peak_dict["force"])
            row["manual_force_pN"] = (
                float(peak_dict["manual_force"])
                if peak_dict["manual_force"] is not None
                else float("nan")
            )
            row["accepted"] = bool(peak_dict["accepted"])
            row["confidence"] = float(peak_dict["confidence"])
            row["prominence_pN"] = float(peak_dict["prominence"])
            row["width_points"] = int(peak_dict["width"])
            row["height_drop_pN"] = float(peak_dict["height_drop"])
            row["note"] = str(peak_dict["note"])
            rows.append(row)
    return rows


def _build_fit_rows(
    fits: list[FitResult],
    all_params: list[str],
) -> list[dict[str, Any]]:
    """Build the one-row-per-fit table shared by ``to_mat`` and ``to_parquet``.

    The schema is the v0.3 CSV one-row-per-fit shape (``model``,
    ``p``, ``p_stderr``, ``L``, ``L_stderr``, ``chi_square``,
    ``reduced_chi_square``, ``n_data`` for a batch of WLC fits) with
    the ``curve_index`` column appended so the per-fit table joins
    back to the per-peak / per-curve tables on the curve index.
    """
    rows: list[dict[str, Any]] = []
    for curve_idx, fit in enumerate(fits):
        row = _fit_columns(fit, all_params)
        row["curve_index"] = curve_idx
        rows.append(row)
    return rows


def _validate_reviewers_mapping(
    reviewers: dict[int, PeakReviewer],
    n_fits: int,
) -> None:
    """Raise ``ValueError`` if any reviewer key is out of range.

    Catches the off-by-one mistake of indexing by ``curve.metadata``
    position instead of batch position, or a typo'd curve index.
    """
    stray = [i for i in reviewers if i >= n_fits or i < -n_fits]
    if stray:
        raise ValueError(
            f"reviewers mapping has entries for curve indices not in the "
            f"fit list: {sorted(stray)} (fit list has {n_fits} entries)"
        )


def _fit_columns(fit: FitResult, all_params: list[str]) -> dict[str, Any]:
    """Build the per-fit column block shared by the per-fit and per-peak writers.

    Pulls the ``stderr`` / ``chi_square`` / ``reduced_chi_square``
    fields with the same back-compat fallback used in the legacy
    :func:`to_csv_fits` path, so both writers handle the new
    :class:`afmkit.fitting.report.FitResult` and the legacy
    io-side dataclass identically.
    """
    stderr = getattr(fit, "stderr", None) or getattr(fit, "param_stderr", {}) or {}
    chi2 = getattr(fit, "chi_square", None)
    if chi2 is None:
        chi2 = getattr(fit, "chi2", float("nan"))
    redchi = getattr(fit, "reduced_chi_square", None)
    if redchi is None:
        redchi = getattr(fit, "redchi", float("nan"))

    out: dict[str, Any] = {
        "model": fit.model_name,
        "chi_square": chi2,
        "reduced_chi_square": redchi,
        "n_data": fit.n_data,
    }
    for name in all_params:
        out[name] = fit.params.get(name, np.nan)
        out[f"{name}_stderr"] = stderr.get(name, np.nan)
    return out


def _empty_peak_value(col: str) -> Any:
    """Return the empty value used for a peak column on a no-peak row.

    All columns are returned as a single ``NaN`` (pandas writes this
    as an empty cell and reads it back as ``NaN``; the ``.mat``
    writer converts NaN to the empty string for the struct fields
    that have to be string-typed). The bool column defaults to
    ``False`` — the convention is "no review yet, so the peak is
    un-accepted" — but downstream readers should treat the
    ``curve_index`` row as a fit-only row by checking for NaN in
    the peak columns.
    """
    if col in (
        "extension_nm",
        "force_pN",
        "manual_force_pN",
        "prominence_pN",
        "height_drop_pN",
        "confidence",
        "note",
    ):
        return float("nan")
    if col in ("width_points", "peak_index"):
        return -1
    if col == "accepted":
        return False
    return float("nan")


# -- to_mat --------------------------------------------------------------


def to_mat(
    batch: CurveBatch,
    path: Path | str,
    *,
    fits: list[FitResult] | None = None,
    reviewers: dict[int, PeakReviewer] | None = None,
) -> None:
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
    ``fits`` (v0.5+)
        One-row-per-fit table (model, params, std-errors,
        goodness-of-fit, ``curve_index``). Only present when
        ``fits`` is provided. Each parameter column is named
        ``<name>`` and the standard error is ``<name>_stderr``;
        the column order matches the v0.3 ``to_csv_fits`` shape.
    ``peak_review`` (v0.5+)
        One-row-per-(curve, peak) table carrying the per-peak
        review state (``peak_index``, ``extension_nm``,
        ``force_pN``, ``manual_force_pN``, ``accepted``,
        ``confidence``, ``prominence_pN``, ``width_points``,
        ``height_drop_pN``, ``note``) joined with the fit
        columns and the ``curve_index`` column. Only present
        when ``reviewers`` is provided. Curves with a reviewer
        but zero peaks (or no reviewer at all) get a single row
        with the peak columns empty so every fit is
        represented.

    Round-trip
    ----------
    ::

        scipy.io.savemat(path, scipy.io.loadmat(path))   # round-trip
        data = scipy.io.loadmat(path)
        n_curves = int(data["n_curves"].item())
        ext = data["extension"]                          # (n_curves, n_max)
        meta = json.loads(data["curve_metadata_json"][0])
        if "peak_review" in data:
            peak_df = pd.DataFrame(data["peak_review"])

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

    # v0.5+ fit / peak-review tables. Both are emitted as Matlab
    # struct arrays (1-D structured numpy ndarrays) so they pass
    # through ``savemat`` cleanly and round-trip back to a
    # ``dtype.names``-bearing struct on the load side (squeeze the
    # loaded (1, N) array to recover the canonical 1-D layout).
    # The legacy v0.4 shape (no fits, no reviewers) is preserved
    # because the kwargs default to None.
    if fits is not None:
        if not fits:
            raise ValueError("Cannot export an empty list of fits")
        all_params: list[str] = []
        for fit in fits:
            for name in fit.params:
                if name not in all_params:
                    all_params.append(name)
        mdict["fits"] = _build_matlab_table(
            _build_fit_rows(fits, all_params), fit_param_names=all_params
        )

    if reviewers is not None:
        if fits is None:
            raise ValueError(
                "to_mat: `reviewers` requires `fits` — the per-peak "
                "table joins the fit columns, so a fit list is needed"
            )
        _validate_reviewers_mapping(reviewers, len(fits))
        all_params = []
        for fit in fits:
            for name in fit.params:
                if name not in all_params:
                    all_params.append(name)
        mdict["peak_review"] = _build_matlab_table(
            _build_peak_review_rows(fits, all_params, reviewers),
            fit_param_names=all_params,
        )

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


# Per-field dtypes used by :func:`_build_matlab_table` to assemble a
# structured numpy array for the per-fit / per-peak tables. The
# ``note`` and ``manual_force_pN`` columns are variable-width
# unicode (U32 is enough for a one-line note; the round-trip
# preserves the content as a ``numpy.str_`` scalar). Numeric
# columns are float64 so NaN round-trips cleanly; integer
# columns are int32 to match scipy's default. The ``accepted``
# bool is the only non-numeric / non-string column.
_MATLAB_FIELD_DTYPES: dict[str, str] = {
    "curve_index": "i4",
    "peak_index": "i4",
    "extension_nm": "f8",
    "force_pN": "f8",
    "manual_force_pN": "f8",
    "accepted": "?",
    "confidence": "f8",
    "prominence_pN": "f8",
    "width_points": "i4",
    "height_drop_pN": "f8",
    "note": "U32",
    "model": "U32",
    "chi_square": "f8",
    "reduced_chi_square": "f8",
    "n_data": "i4",
}


def _build_matlab_table(
    rows: list[dict[str, Any]],
    fit_param_names: list[str] | None = None,
) -> np.ndarray:
    """Assemble a list of per-fit / per-peak row dicts into a structured ndarray.

    The output is a 1-D structured ``numpy.ndarray`` with one
    entry per row and one named field per column. The struct
    passes through :func:`scipy.io.savemat` as a Matlab struct
    array and round-trips back through :func:`scipy.io.loadmat`
    (the loaded shape is ``(1, N)`` because Matlab stores struct
    arrays as 2-D; ``.squeeze()`` recovers the 1-D layout).

    Column types are looked up from :data:`_MATLAB_FIELD_DTYPES`;
    the fit parameter columns (``p``, ``L``, ``K0``, ``b``, ``Lc``,
    ``<name>_stderr``) are added with the float64 dtype. Missing
    fields in a row are filled with NaN / ``""`` / ``False`` / ``-1``
    so the struct dtype stays fixed.
    """
    dtype_list: list[tuple[str, str]] = []
    seen: set[str] = set()

    # Stable column order: the peak / fit schema columns in their
    # declared order, then any fit parameters the caller passed in.
    for col in _PEAK_REVIEW_COLUMNS:
        if col in _MATLAB_FIELD_DTYPES and col not in seen:
            dtype_list.append((col, _MATLAB_FIELD_DTYPES[col]))
            seen.add(col)
    # Fit-block columns (model, chi_square, reduced_chi_square,
    # n_data) come from the per-fit table and join the peak table
    # via ``_build_peak_review_rows``.
    for col in ("model", "chi_square", "reduced_chi_square", "n_data"):
        if col in _MATLAB_FIELD_DTYPES and col not in seen:
            dtype_list.append((col, _MATLAB_FIELD_DTYPES[col]))
            seen.add(col)
    if fit_param_names is not None:
        for name in fit_param_names:
            for col in (name, f"{name}_stderr"):
                if col not in seen:
                    dtype_list.append((col, "f8"))
                    seen.add(col)

    if not rows:
        # Empty table — return a 1-D structured array of length 0
        # with the right dtype. ``savemat`` writes this as a 0x0
        # Matlab struct, which round-trips cleanly.
        return np.zeros(0, dtype=dtype_list)

    # Coerce each row's values into the dtype-friendly form.
    records: list[tuple[Any, ...]] = []
    for row in rows:
        record: list[Any] = []
        for col, dtype in dtype_list:
            value = row.get(col, _empty_matlab_value(dtype))
            record.append(_coerce_for_matlab_field(value, dtype))
        records.append(tuple(record))
    return np.array(records, dtype=dtype_list)


def _empty_matlab_value(dtype: str) -> Any:
    """The "no data" marker for a given numpy dtype string."""
    if dtype == "?":
        return False
    if dtype.startswith("i") or dtype.startswith("u"):
        return -1
    if dtype.startswith("f"):
        return float("nan")
    if dtype.startswith("U") or dtype.startswith("S"):
        return ""
    return float("nan")


def _coerce_for_matlab_field(value: Any, dtype: str) -> Any:
    """Coerce a single row value into a value compatible with ``dtype``.

    Handles the four cases the v0.5 schema actually needs:
    numeric (``int``/``float``/``NaN``/``np.nan``), bool, string,
    and the "no data" markers produced by :func:`_empty_matlab_value`.
    """
    if dtype == "?":
        # ``bool`` field. ``NaN`` is treated as "un-reviewed" →
        # ``False`` (matches the CSV behaviour: placeholder rows
        # default ``accepted=False``).
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return False
        return bool(value)
    if dtype.startswith("i") or dtype.startswith("u"):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return -1
        return int(value)
    if dtype.startswith("f"):
        if value is None or (isinstance(value, float) and np.isnan(value)) or value == "":
            return float("nan")
        return float(value)
    if dtype.startswith("U") or dtype.startswith("S"):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        return str(value)
    return value


# -- to_parquet ----------------------------------------------------------


def to_parquet(
    batch: CurveBatch,
    path: Path | str,
    *,
    fits: list[FitResult] | None = None,
    reviewers: dict[int, PeakReviewer] | None = None,
) -> None:
    """Write the batch as a Parquet file (or set of files).

    The on-disk layout matches :func:`to_csv` (one row per point
    index, two data columns per curve). Parquet's binary encoding
    and rich schema metadata make this the preferred format for
    downstream pandas / polars analysis.

    When ``fits`` or ``reviewers`` are provided, the additional
    tables are written as **sibling files** next to the main
    curves file, not as additional row groups inside the same
    file. The naming convention is:

    ============================  ==================================
    Suffix                         Contents
    ============================  ==================================
    ``<path>.parquet``             The curves (the v0.4 layout).
    ``<path>.fits.parquet``        One row per fit (model, params,
                                   std-errors, goodness-of-fit,
                                   ``curve_index``). Only written
                                   when ``fits`` is provided.
    ``<path>.peaks.parquet``       One row per (curve, peak) with
                                   the per-peak review state joined
                                   to the fit columns. Only
                                   written when ``reviewers`` is
                                   provided. The curves in the
                                   batch without a reviewer entry
                                   (or with a reviewer but zero
                                   peaks) get a single row with
                                   the peak columns empty so every
                                   fit is represented.

    The sibling-file layout is chosen over a single Parquet
    Dataset because the three tables have fundamentally different
    shapes (one row per point vs one row per fit vs one row per
    (curve, peak)) and pandas / polars users typically want to
    load them independently. To load all three:

    .. code-block:: python

        curves = pd.read_parquet("data.parquet")
        fits = pd.read_parquet("data.fits.parquet")
        peaks = pd.read_parquet("data.peaks.parquet")

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
        _arrow_available = False
        _pyarrow_exc = exc
    else:
        _arrow_available = True

    if not _arrow_available:
        # fastparquet fallback only supports the main curves file.
        # ``fits`` / ``reviewers`` paths would need to be skipped
        # or written through a different backend; we surface a
        # clear ImportError rather than silently dropping the
        # extra tables.
        if fits is not None or reviewers is not None:
            raise ImportError(
                "to_parquet(fits=..., reviewers=...) requires pyarrow; "
                f"pyarrow is not available ({_pyarrow_exc}). Install "
                "pyarrow to write the fit / peak-review sibling files."
            ) from _pyarrow_exc
        try:
            df.to_parquet(str(path), engine="fastparquet")
        except (ImportError, ValueError) as exc2:
            raise ImportError(
                "to_parquet requires either pyarrow or fastparquet; "
                f"neither is available (pyarrow: {_pyarrow_exc}; "
                f"fastparquet: {exc2})"
            ) from exc2
        return

    # Main curves file (the v0.4 layout).
    table = pa.Table.from_pandas(df)
    pq.write_table(table, str(path))

    # Sibling files for the per-fit and per-peak tables. The
    # ``curve_index`` column is the join key across all three
    # files.
    if fits is not None:
        if not fits:
            raise ValueError("Cannot export an empty list of fits")
        all_params: list[str] = []
        for fit in fits:
            for name in fit.params:
                if name not in all_params:
                    all_params.append(name)
        fit_df = pd.DataFrame(_build_fit_rows(fits, all_params))
        fit_path = path.with_suffix(".fits.parquet")
        pq.write_table(pa.Table.from_pandas(fit_df), str(fit_path))

    if reviewers is not None:
        if fits is None:
            raise ValueError(
                "to_parquet: `reviewers` requires `fits` — the per-peak "
                "table joins the fit columns, so a fit list is needed"
            )
        _validate_reviewers_mapping(reviewers, len(fits))
        all_params = []
        for fit in fits:
            for name in fit.params:
                if name not in all_params:
                    all_params.append(name)
        peak_df = pd.DataFrame(_build_peak_review_rows(fits, all_params, reviewers))
        peak_path = path.with_suffix(".peaks.parquet")
        pq.write_table(pa.Table.from_pandas(peak_df), str(peak_path))


# -- to_markdown ---------------------------------------------------------


def to_markdown(
    batch: CurveBatch,
    fits: list[FitResult] | None,
    path: Path | str,
    *,
    reviewers: dict[int, PeakReviewer] | None = None,
) -> None:
    """Write a human-readable Markdown report of the batch.

    The report contains three sections by default:

    1. **Batch summary** — name, number of curves, cantilever spring
       constant, and source folder.
    2. **Per-curve summary** — point count, extension range, and force
       range for each curve.
    3. **Fit results** (if ``fits`` is provided) — one row per fit,
       with the parameter estimates and the goodness-of-fit
       statistics.

    When ``reviewers`` is provided, a fourth section is appended:

    4. **Peak review** — one row per (curve, peak) with the
       auto-detected extension / force, the user-override force (if
       any), the accept / reject flag, the auto-detected
       confidence, and any free-form note. The table is grouped
       by curve index so a reader can scan it in the same order
       the curves appear in the per-curve summary.

    Parameters
    ----------
    batch
        The :class:`~afmkit.core.curve.CurveBatch` to report on.
    fits
        Optional list of :class:`FitResult` instances. If ``None`` (or
        empty), the fit-results section is omitted.
    path
        Destination file path.
    reviewers
        Optional ``{curve_index: PeakReviewer}`` mapping. When
        provided, a per-peak review table is appended to the
        report. Same indexing rules as
        :func:`to_csv_fits` (``reviewers[i]`` corresponds to
        ``fits[i]`` and ``batch[i]``).
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
        header = ["#", "model", *all_params, "chi_square", "reduced_chi_square", "n_data"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for idx, fit in enumerate(fits):
            # Accept both the new (afmkit.fitting.report) FitResult
            # and the legacy io-side dataclass.
            chi2 = getattr(fit, "chi_square", None)
            if chi2 is None:
                chi2 = getattr(fit, "chi2", float("nan"))
            redchi = getattr(fit, "reduced_chi_square", None)
            if redchi is None:
                redchi = getattr(fit, "redchi", float("nan"))
            row: list[str] = [str(idx), fit.model_name]
            for name in all_params:
                val = fit.params.get(name, float("nan"))
                row.append(f"{val:.4g}")
            row.append(f"{chi2:.4g}")
            row.append(f"{redchi:.4g}")
            row.append(str(fit.n_data))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # -- Peak review -------------------------------------------------------
    if reviewers:
        lines.append("## Peak review")
        lines.append("")
        lines.append(
            "One row per auto-detected peak. `force` is the post-review "
            "value (user override if set, else auto-detected). `manual_force` "
            "is shown only when the user has set an override."
        )
        lines.append("")
        header = [
            "curve",
            "peak",
            "ext (nm)",
            "force (pN)",
            "manual_force (pN)",
            "accepted",
            "confidence",
            "prominence (pN)",
            "width (pts)",
            "height_drop (pN)",
            "note",
        ]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for curve_idx in sorted(reviewers):
            reviewer = reviewers[curve_idx]
            if len(reviewer) == 0:
                # Show the curve index with a single placeholder row
                # so the report reflects the empty-reviewer case.
                lines.append(f"| {curve_idx} | — | — | — | — | — | — | — | — | — | (no peaks) |")
                continue
            for peak_dict in reviewer.to_dict():
                manual = peak_dict["manual_force"]
                note = str(peak_dict["note"]) or "—"
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(curve_idx),
                            str(int(peak_dict["index"])),
                            f"{float(peak_dict['extension']):.4g}",
                            f"{float(peak_dict['force']):.4g}",
                            (f"{float(manual):.4g}" if manual is not None else "—"),
                            "yes" if bool(peak_dict["accepted"]) else "no",
                            f"{float(peak_dict['confidence']):.4g}",
                            f"{float(peak_dict['prominence']):.4g}",
                            str(int(peak_dict["width"])),
                            f"{float(peak_dict['height_drop']):.4g}",
                            note,
                        ]
                    )
                    + " |"
                )
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
