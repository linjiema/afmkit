"""Unit tests for the :mod:`afmkit.presentation.cli` command-line interface.

These tests drive the typer app through ``typer.testing.CliRunner`` —
the same harness a real shell would see, minus the process boundary.
The fixtures used here come from ``tests/conftest.py`` (a synthetic
WLC :class:`ForceCurve`) and from small helpers in this file that
materialise the curves onto disk as JPK ``.txt`` and HDF5 files.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from typer.testing import CliRunner

from afmkit import __version__
from afmkit.core.curve import CurveBatch, ForceCurve
from afmkit.io.hdf5_store import save_hdf5
from afmkit.presentation.cli import app

if TYPE_CHECKING:
    pass

# A fresh runner per test keeps the captured stdout / stderr from
# leaking between cases. The default ``mix_stderr=False`` is fine —
# the CLI explicitly writes errors to stderr via the rich console.
runner = CliRunner()


# -- Local fixtures -------------------------------------------------------


def _save_synthetic_hdf5(
    tmp_path: Path, *, n_curves: int = 2, n_points: int = 300
) -> Path:
    """Write a small afmkit HDF5 session with ``n_curves`` WLC curves.

    The extension / force data is generated with the same WLC formula
    used in ``tests/conftest.py`` so the fitter has realistic numbers
    to work with — ``load_jpk_txt`` is exercised elsewhere, this is
    the "the fitter gets sensible data" baseline.
    """
    ext = np.linspace(0.5, 199.5, n_points)
    p, lc = 0.4, 200.0
    force = (4.1 / p) * (0.25 * (1.0 - ext / lc) ** -2 - 0.25 + ext / lc)
    curves = [
        ForceCurve(
            ext,
            force,
            metadata={
                "k_cantilever": 0.1,
                "source_file": f"curve_{i:03d}.txt",
                "direction": "retract" if i % 2 else "approach",
            },
        )
        for i in range(n_curves)
    ]
    batch = CurveBatch(curves, name="synthetic", metadata={"k_cantilever": 0.1})
    out = tmp_path / "session.h5"
    save_hdf5(batch, out)
    return out


def _save_synthetic_jpk_txt(path: Path, *, n_points: int = 300, k: float = 0.1) -> Path:
    """Write a JPK 4-column ``.txt`` file with the loader's own contract.

    We reuse :class:`JPKTxtLoader` to produce real, loadable data via
    the existing test machinery (round-trips through itself), then
    save the same ``.txt`` content the loader would write if it were
    an exporter — i.e. an on-disk file matching the JPK 4-column shape.
    """
    ext = np.linspace(0.5, 199.5, n_points)
    p, lc = 0.4, 200.0
    force = (4.1 / p) * (0.25 * (1.0 - ext / lc) ** -2 - 0.25 + ext / lc)
    # Reconstruct the raw JPK columns from the converted units.
    piezo = (ext + force / k) * 1e-9  # m
    defl = -force * 1e-12  # N (signed; JPK uses positive-down)
    # The .txt file has 4 columns: forward_piezo, forward_defl,
    # backward_piezo, backward_defl. Use the same data for both
    # directions — the loader will produce 2 curves per file.
    data = np.column_stack([piezo, defl, piezo, defl])
    # No header — the loader accepts a headerless 4-column file too.
    np.savetxt(path, data, fmt="%.6e", delimiter="\t")
    return path


# -- version --------------------------------------------------------------


class TestVersion:
    def test_subcommand_prints_version(self) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0, result.output
        assert __version__ in result.output

    def test_global_flag_prints_version_and_exits(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output


# -- info -----------------------------------------------------------------


class TestInfo:
    def test_missing_file_exits_1(self, tmp_path: Path) -> None:
        ghost = tmp_path / "does_not_exist.h5"
        result = runner.invoke(app, ["info", str(ghost)])
        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "not found" in (result.stderr or "").lower()

    def test_hdf5_reports_n_curves(self, tmp_path: Path) -> None:
        h5 = _save_synthetic_hdf5(tmp_path, n_curves=3)
        result = runner.invoke(app, ["info", str(h5)])
        assert result.exit_code == 0, result.output
        # The "3" must surface somewhere — both the table cell and any
        # status line. The substring "3" alone is too generic; anchor
        # on a label we know the table uses.
        assert "n_curves" in result.output
        assert "3" in result.output

    def test_jpk_txt_reports_row_count(self, tmp_path: Path) -> None:
        jpk = _save_synthetic_jpk_txt(tmp_path / "curve.txt", n_points=300)
        result = runner.invoke(app, ["info", str(jpk)])
        assert result.exit_code == 0, result.output
        assert "JPK" in result.output
        assert "k required" in result.output
        # 300 numeric rows + no header → 300.
        assert "300" in result.output

    def test_ibw_is_handled_gracefully(self, tmp_path: Path) -> None:
        ibw = tmp_path / "legacy.ibw"
        ibw.write_bytes(b"\x00\x00\x00\x00")  # contents don't matter
        result = runner.invoke(app, ["info", str(ibw)])
        # The spec says "exit 0" for the .ibw "not yet wired" case.
        assert result.exit_code == 0
        assert "not yet wired" in result.output

    def test_unknown_extension_exits_1(self, tmp_path: Path) -> None:
        weird = tmp_path / "data.xyz"
        weird.write_text("whatever\n")
        result = runner.invoke(app, ["info", str(weird)])
        assert result.exit_code == 1
        assert "Unknown" in (result.output + (result.stderr or ""))


# -- import ---------------------------------------------------------------


class TestImport:
    def test_directory_of_two_txt_files_produces_hdf5(self, tmp_path: Path) -> None:
        # Build a folder with 2 JPK .txt files.
        d = tmp_path / "raw"
        d.mkdir()
        _save_synthetic_jpk_txt(d / "a.txt", k=0.1)
        _save_synthetic_jpk_txt(d / "b.txt", k=0.1)
        out = tmp_path / "session.h5"

        result = runner.invoke(
            app, ["import", str(d), "--output", str(out), "--k", "0.1"]
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
        # 2 files * 2 directions = 4 curves in the merged batch.
        from afmkit.io.hdf5_store import load_hdf5

        loaded = load_hdf5(out)
        assert loaded.n_curves == 4

    def test_missing_k_for_directory_exits_1(self, tmp_path: Path) -> None:
        d = tmp_path / "raw"
        d.mkdir()
        _save_synthetic_jpk_txt(d / "a.txt")
        result = runner.invoke(
            app, ["import", str(d), "--output", str(tmp_path / "session.h5")]
        )
        assert result.exit_code == 1
        combined = result.output + (result.stderr or "")
        assert "--k" in combined or "k" in combined.lower()

    def test_recursive_flag_finds_nested_files(self, tmp_path: Path) -> None:
        # Place one .txt at the top level and one in a sub-folder.
        d = tmp_path / "raw"
        d.mkdir()
        nested = d / "sub"
        nested.mkdir()
        _save_synthetic_jpk_txt(d / "top.txt", k=0.1)
        _save_synthetic_jpk_txt(nested / "deep.txt", k=0.1)

        # Non-recursive walk should find only "top.txt" → 2 curves.
        out_a = tmp_path / "nonrec.h5"
        r_a = runner.invoke(
            app, ["import", str(d), "--output", str(out_a), "--k", "0.1"]
        )
        assert r_a.exit_code == 0, r_a.output

        # Recursive walk should find both → 4 curves.
        out_b = tmp_path / "rec.h5"
        r_b = runner.invoke(
            app,
            [
                "import",
                str(d),
                "--output",
                str(out_b),
                "--k",
                "0.1",
                "--recursive",
            ],
        )
        assert r_b.exit_code == 0, r_b.output

        from afmkit.io.hdf5_store import load_hdf5

        assert load_hdf5(out_a).n_curves == 2
        assert load_hdf5(out_b).n_curves == 4


# -- fit ------------------------------------------------------------------


class TestFit:
    def test_fit_writes_csv_with_expected_columns(self, tmp_path: Path) -> None:
        h5 = _save_synthetic_hdf5(tmp_path, n_curves=2)
        out_csv = tmp_path / "fits.csv"
        result = runner.invoke(
            app,
            [
                "fit",
                str(h5),
                "--model",
                "wlc",
                "--output",
                str(out_csv),
                "--x-range",
                "20",
                "180",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_csv.exists()
        # The Markdown sibling is written next to the CSV.
        assert out_csv.with_suffix(".md").exists()

        # Parse the CSV and check the column shape that ``to_csv_fits``
        # is contracted to produce: ``model``, ``p``, ``p_stderr``,
        # ``L``, ``L_stderr``, ``chi_square``, ``reduced_chi_square``,
        # ``n_data``.
        import pandas as pd

        df = pd.read_csv(out_csv)
        assert "model" in df.columns
        assert "p" in df.columns
        assert "p_stderr" in df.columns
        assert "L" in df.columns
        assert "L_stderr" in df.columns
        assert "reduced_chi_square" in df.columns
        assert len(df) == 2  # one row per curve

    def test_fit_missing_session_exits_1(self, tmp_path: Path) -> None:
        ghost = tmp_path / "nope.h5"
        result = runner.invoke(app, ["fit", str(ghost), "--output", str(tmp_path / "f.csv")])
        assert result.exit_code == 1
        combined = result.output + (result.stderr or "")
        assert "not found" in combined.lower()

    def test_fit_unknown_model_exits_1(self, tmp_path: Path) -> None:
        h5 = _save_synthetic_hdf5(tmp_path, n_curves=1)
        result = runner.invoke(
            app,
            [
                "fit",
                str(h5),
                "--model",
                "definitely_not_a_model",
                "--output",
                str(tmp_path / "f.csv"),
            ],
        )
        assert result.exit_code == 1
        combined = result.output + (result.stderr or "")
        assert "model" in combined.lower()


# -- export ---------------------------------------------------------------


class TestExport:
    @pytest.mark.parametrize("ext,fmt", [("csv", "csv"), ("mat", "mat"), ("parquet", "parquet")])
    def test_export_round_trips_each_format(
        self, tmp_path: Path, ext: str, fmt: str
    ) -> None:
        h5 = _save_synthetic_hdf5(tmp_path, n_curves=2)
        out = tmp_path / f"results.{ext}"
        result = runner.invoke(
            app,
            ["export", str(h5), "--format", fmt, "--output", str(out)],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
        assert out.stat().st_size > 0

        if fmt == "csv":
            import pandas as pd

            df = pd.read_csv(out)
            # 2 curves → 4 leading columns: ext_000, force_000, ext_001, force_001.
            assert "ext_000" in df.columns
            assert "force_000" in df.columns
            assert "ext_001" in df.columns
            assert "force_001" in df.columns
        elif fmt == "mat":
            import scipy.io

            data = scipy.io.loadmat(out)
            assert int(data["n_curves"].item()) == 2
            assert "extension" in data
            assert "force" in data
        elif fmt == "parquet":
            import pandas as pd

            df = pd.read_parquet(out)
            assert "ext_000" in df.columns
            assert "force_000" in df.columns

    def test_export_format_inferred_from_suffix(self, tmp_path: Path) -> None:
        h5 = _save_synthetic_hdf5(tmp_path, n_curves=1)
        out = tmp_path / "results.md"
        result = runner.invoke(app, ["export", str(h5), "--output", str(out)])
        assert result.exit_code == 0, result.output
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "afmkit" in text
        # n_curves = 1 should appear in the per-curve summary.
        assert "n_curves" in text

    def test_export_missing_session_exits_1(self, tmp_path: Path) -> None:
        ghost = tmp_path / "nope.h5"
        result = runner.invoke(
            app, ["export", str(ghost), "--output", str(tmp_path / "x.csv")]
        )
        assert result.exit_code == 1
        combined = result.output + (result.stderr or "")
        assert "not found" in combined.lower()
