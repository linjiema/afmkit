"""Unit tests for :mod:`afmkit.io.exporters`.

These tests focus on the round-trip contract of every supported output
format — CSV, CSV-fits, Matlab ``.mat``, Parquet, and Markdown — plus the
:func:`export` dispatch helper that picks the right writer from ``fmt``
or from the file suffix.

Conventions
-----------
- All file operations use the ``tmp_path`` pytest fixture so the working
  tree is never touched.
- Synthetic :class:`ForceCurve` objects are small (≤ 200 points) so the
  tests stay fast — exporters don't depend on curve size.
- :class:`FitResult` instances are built directly with positional
  arguments; we don't exercise the lmfit/fitting engine here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.io

from afmkit.core.curve import CurveBatch, ForceCurve
from afmkit.io.exporters import (
    FitResult,
    export,
    to_csv,
    to_csv_fits,
    to_markdown,
    to_mat,
    to_parquet,
)

# -- Helpers --------------------------------------------------------------


def _make_curve(
    n: int,
    *,
    k: float = 0.1,
    source_file: str = "synthetic.jpk",
    direction: str = "approach",
    **extra: Any,
) -> ForceCurve:
    """Build a synthetic :class:`ForceCurve` of ``n`` points.

    The extension axis is a linspace so the round-trip is unambiguous;
    the force axis is a non-trivial ramp so accidental zero-fills would
    show up immediately in the assertions.
    """
    ext = np.linspace(0.5, 100.0, n)
    force = np.linspace(-1.0, 5.0, n) + 0.01 * np.arange(n)
    meta: dict[str, Any] = {
        "k_cantilever": k,
        "source_file": source_file,
        "direction": direction,
    }
    meta.update(extra)
    return ForceCurve(ext, force, metadata=meta)


def _make_batch(n_curves: int = 3, name: str = "test_batch") -> CurveBatch:
    """Build a small CurveBatch with deterministic but ragged curve lengths."""
    lengths = [50, 30, 20][:n_curves]
    curves = [
        _make_curve(
            n,
            k=0.05 + 0.01 * i,
            source_file=f"curve_{i:02d}.jpk",
            direction="approach" if i % 2 == 0 else "retract",
        )
        for i, n in enumerate(lengths)
    ]
    return CurveBatch(
        curves,
        name=name,
        metadata={"k_cantilever": 0.08, "operator": "tester"},
    )


def _make_fit(
    *,
    model: str = "WLC",
    p: float = 0.4,
    lc: float = 200.0,
    p_err: float = 0.01,
    lc_err: float = 1.0,
    chi2: float = 1.2,
    redchi: float = 0.4,
    n_data: int = 500,
) -> FitResult:
    """Build a synthetic :class:`FitResult` with WLC-shaped parameters."""
    return FitResult(
        model_name=model,
        params={"p": p, "L": lc},
        param_stderr={"p": p_err, "L": lc_err},
        chi2=chi2,
        redchi=redchi,
        n_data=n_data,
    )


# -- to_csv --------------------------------------------------------------


class TestToCsv:
    """The wide column-block CSV must round-trip through pandas."""

    def test_produces_file_with_expected_header_and_row_count(self, tmp_path: Path) -> None:
        batch = _make_batch(n_curves=3)
        n_max = max(c.n_points for c in batch)
        p = tmp_path / "out.csv"

        to_csv(batch, p)

        assert p.exists()
        text = p.read_text(encoding="utf-8")
        header = text.splitlines()[0].split(",")
        # 2 data columns per curve (ext_NNN, force_NNN) + 3 metadata
        # columns per curve (source_file, direction, k_cantilever).
        assert len(text.splitlines()) == n_max + 1
        # Data columns come first, in interleaved order.
        for i in range(3):
            assert f"ext_{i:03d}" in header
            assert f"force_{i:03d}" in header

    def test_opens_with_pandas_read_csv(self, tmp_path: Path) -> None:
        batch = _make_batch(n_curves=2)
        p = tmp_path / "out.csv"

        to_csv(batch, p)
        df = pd.read_csv(p)

        assert len(df) == max(c.n_points for c in batch)
        assert "ext_000" in df.columns
        assert "force_000" in df.columns
        assert "ext_001" in df.columns
        assert "force_001" in df.columns

    def test_round_trips_extension_and_force_per_curve(self, tmp_path: Path) -> None:
        batch = _make_batch(n_curves=3)
        p = tmp_path / "round.csv"

        to_csv(batch, p)
        df = pd.read_csv(p)

        for i, curve in enumerate(batch):
            ext_col = df[f"ext_{i:03d}"]
            force_col = df[f"force_{i:03d}"]
            # The leading n_points rows must match the input; the
            # remainder are NaN (padded). Use almost-equal because the
            # CSV round-trip introduces last-bit float jitter.
            np.testing.assert_array_almost_equal(
                ext_col.iloc[: curve.n_points].to_numpy(),
                curve.extension,
            )
            np.testing.assert_array_almost_equal(
                force_col.iloc[: curve.n_points].to_numpy(),
                curve.force,
            )
            assert ext_col.iloc[curve.n_points :].isna().all()
            assert force_col.iloc[curve.n_points :].isna().all()

    def test_include_metadata_false_omits_meta_columns(self, tmp_path: Path) -> None:
        batch = _make_batch(n_curves=2)
        p = tmp_path / "nomet.csv"

        to_csv(batch, p, include_metadata=False)
        df = pd.read_csv(p)

        assert "source_file_000" not in df.columns
        assert "direction_000" not in df.columns
        assert "k_cantilever_000" not in df.columns
        # Data columns remain.
        assert "ext_000" in df.columns and "force_000" in df.columns

    def test_k_cantilever_metadata_is_numeric(self, tmp_path: Path) -> None:
        batch = _make_batch(n_curves=2)
        p = tmp_path / "kcol.csv"

        to_csv(batch, p)
        df = pd.read_csv(p)

        # The k_cantilever column should be a constant float (the
        # per-curve spring constant) repeated along the column. Use
        # almost-equal because the CSV round-trip introduces last-bit
        # float jitter.
        for i, curve in enumerate(batch):
            col = df[f"k_cantilever_{i:03d}"]
            assert col.dtype.kind == "f"
            np.testing.assert_array_almost_equal(
                col.to_numpy(),
                np.full(len(col), curve.metadata["k_cantilever"]),
            )


# -- to_csv_fits ---------------------------------------------------------


class TestToCsvFits:
    """One row per fit; column order follows first-seen parameter order."""

    def test_three_fits_round_trip_through_pandas(self, tmp_path: Path) -> None:
        fits = [
            _make_fit(p=0.40, lc=200.0, p_err=0.01, lc_err=1.0, n_data=500),
            _make_fit(p=0.42, lc=198.0, p_err=0.02, lc_err=2.0, n_data=480),
            _make_fit(p=0.38, lc=205.0, p_err=0.015, lc_err=1.5, n_data=520),
        ]
        p = tmp_path / "fits.csv"

        to_csv_fits(fits, p)
        df = pd.read_csv(p)

        assert len(df) == 3
        # The implementation uses dict insertion order: model is
        # first, then chi2/redchi/n_data, then the parameter columns
        # in the order they first appear across the input list.
        assert list(df.columns) == [
            "model",
            "chi2",
            "redchi",
            "n_data",
            "p",
            "p_stderr",
            "L",
            "L_stderr",
        ]
        # All rows are WLC fits.
        assert (df["model"] == "WLC").all()
        # Spot-check round-trip values.
        np.testing.assert_array_almost_equal(df["p"].to_numpy(), [0.40, 0.42, 0.38])
        np.testing.assert_array_almost_equal(df["L"].to_numpy(), [200.0, 198.0, 205.0])
        np.testing.assert_array_equal(df["n_data"].to_numpy(), [500, 480, 520])

    def test_empty_fits_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="empty"):
            to_csv_fits([], tmp_path / "nope.csv")


# -- to_mat --------------------------------------------------------------


class TestToMat:
    """The .mat file must re-open through scipy.io.loadmat with matching shapes."""

    def test_round_trip_shapes(self, tmp_path: Path) -> None:
        batch = _make_batch(n_curves=3, name="matlab_test")
        p = tmp_path / "round.mat"

        to_mat(batch, p)
        data = scipy.io.loadmat(str(p))

        # Top-level scalar fields. ``name`` round-trips as a 1-D string
        # array of length 1; ``n_curves`` and ``n_points_max`` as
        # 0-D numpy arrays.
        name_arr = data["name"].squeeze()
        assert isinstance(name_arr, np.ndarray)
        assert str(name_arr).strip() == "matlab_test"
        assert int(data["n_curves"].item()) == 3
        n_max = max(c.n_points for c in batch)
        assert int(data["n_points_max"].item()) == n_max

        # 2-D padded arrays.
        ext = data["extension"]
        force = data["force"]
        assert ext.shape == (3, n_max)
        assert force.shape == (3, n_max)

        # Per-curve point counts.
        n_points = data["n_points"].squeeze()
        np.testing.assert_array_equal(n_points.astype(int), np.array([c.n_points for c in batch]))

        # Spot-check a few cell values.
        np.testing.assert_array_almost_equal(ext[0, : batch[0].n_points], batch[0].extension)

    def test_per_curve_metadata_round_trips_as_json(self, tmp_path: Path) -> None:
        batch = _make_batch(n_curves=2)
        p = tmp_path / "meta.mat"

        to_mat(batch, p)
        data = scipy.io.loadmat(str(p))

        meta_blobs = data["curve_metadata_json"]
        # scipy stores string arrays as 1-D when there's no row/col
        # orientation; the 2-curve batch has shape (2,).
        assert meta_blobs.shape == (2,)
        for i, curve in enumerate(batch):
            decoded = json.loads(str(meta_blobs[i]))
            assert decoded["k_cantilever"] == curve.metadata["k_cantilever"]
            assert decoded["direction"] == curve.metadata["direction"]

    def test_empty_batch_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="empty"):
            to_mat(CurveBatch([]), tmp_path / "void.mat")


# -- to_parquet ----------------------------------------------------------


class TestToParquet:
    """Parquet round-trips through pandas; layout matches CSV."""

    def test_round_trip_through_pandas(self, tmp_path: Path) -> None:
        batch = _make_batch(n_curves=2)
        p = tmp_path / "out.parquet"

        to_parquet(batch, p)
        assert p.exists()

        df = pd.read_parquet(p)
        n_max = max(c.n_points for c in batch)
        assert len(df) == n_max
        # Parquet path does not include metadata columns (matches the
        # implementation contract — see to_parquet docstring).
        assert "ext_000" in df.columns and "force_000" in df.columns
        assert "source_file_000" not in df.columns

        # Data round-trip.
        np.testing.assert_array_equal(
            df["ext_000"].to_numpy()[: batch[0].n_points],
            batch[0].extension,
        )


# -- to_markdown ---------------------------------------------------------


class TestToMarkdown:
    """Markdown must be a non-empty, batch-aware report."""

    def test_non_empty_contains_batch_name_and_curve_count(self, tmp_path: Path) -> None:
        batch = _make_batch(n_curves=3, name="my_batch")
        p = tmp_path / "report.md"

        to_markdown(batch, fits=None, path=p)

        assert p.exists()
        text = p.read_text(encoding="utf-8")
        assert text  # non-empty
        # Batch name appears in the heading and the summary table.
        assert "my_batch" in text
        # Curve count appears at least once in the summary section.
        assert "3" in text

    def test_fit_results_section_appears_when_fits_provided(self, tmp_path: Path) -> None:
        batch = _make_batch(n_curves=2)
        fits = [_make_fit(p=0.41, lc=201.0)]
        p = tmp_path / "with_fits.md"

        to_markdown(batch, fits=fits, path=p)
        text = p.read_text(encoding="utf-8")

        assert "Fit results" in text
        assert "WLC" in text
        # Per-curve summary should still be present.
        assert "Per-curve summary" in text


# -- export --------------------------------------------------------------


class TestExportDispatch:
    """:func:`export` picks the right writer from ``fmt`` or the suffix."""

    def test_fmt_csv_dispatches_to_to_csv(self, tmp_path: Path) -> None:
        batch = _make_batch(n_curves=2)
        p = tmp_path / "out.csv"

        export(batch, p, fmt="csv")

        # The CSV file exists and has the wide-column header.
        assert p.exists()
        header = p.read_text(encoding="utf-8").splitlines()[0]
        assert "ext_000" in header and "force_000" in header

    def test_fmt_unknown_raises_value_error(self, tmp_path: Path) -> None:
        batch = _make_batch(n_curves=2)
        with pytest.raises(ValueError, match="Unknown format"):
            export(batch, tmp_path / "x.xlsx", fmt="xlsx")

    def test_fmt_markdown_dispatches_to_markdown(self, tmp_path: Path) -> None:
        batch = _make_batch(n_curves=2, name="dispatch_test")
        p = tmp_path / "report.md"

        export(batch, p, fmt="markdown")

        text = p.read_text(encoding="utf-8")
        assert "dispatch_test" in text

    def test_no_fmt_infers_from_suffix(self, tmp_path: Path) -> None:
        batch = _make_batch(n_curves=2)
        p = tmp_path / "auto.csv"

        export(batch, p)

        # Should have produced a valid CSV (same header shape as to_csv).
        df = pd.read_csv(p)
        assert "ext_000" in df.columns and "force_001" in df.columns

    def test_no_fmt_unknown_suffix_raises(self, tmp_path: Path) -> None:
        batch = _make_batch(n_curves=2)
        with pytest.raises(ValueError, match="Cannot infer format"):
            export(batch, tmp_path / "mystery.weird")

    def test_fmt_mat_dispatches_to_to_mat(self, tmp_path: Path) -> None:
        batch = _make_batch(n_curves=2)
        p = tmp_path / "auto.mat"

        export(batch, p, fmt="mat")

        data = scipy.io.loadmat(str(p))
        assert int(data["n_curves"].item()) == 2

    def test_fmt_parquet_dispatches_to_to_parquet(self, tmp_path: Path) -> None:
        batch = _make_batch(n_curves=2)
        p = tmp_path / "auto.parquet"

        export(batch, p, fmt="parquet")

        df = pd.read_parquet(p)
        assert "ext_000" in df.columns

    def test_fmt_csv_fits_requires_fits_argument(self, tmp_path: Path) -> None:
        batch = _make_batch(n_curves=2)
        with pytest.raises(ValueError, match="csv_fits"):
            export(batch, tmp_path / "fits.csv", fmt="csv_fits")

    def test_fmt_csv_fits_with_fits_dispatches(self, tmp_path: Path) -> None:
        batch = _make_batch(n_curves=2)
        fits = [_make_fit()]
        p = tmp_path / "fits.csv"

        export(batch, p, fmt="csv_fits", fits=fits)

        df = pd.read_csv(p)
        assert len(df) == 1
        assert "p" in df.columns and "L" in df.columns
