"""Unit tests for :mod:`afmkit.io.jpk_txt`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from afmkit.core.curve import CurveBatch, ForceCurve
from afmkit.io.jpk_txt import JPKTxtLoader, load_jpk_txt

# -- Helpers --------------------------------------------------------------


def _write_jpk(path: Path, data: np.ndarray, header: str | None = None) -> None:
    """Write a 4-column whitespace-separated JPK ``.txt`` file.

    Parameters
    ----------
    path
        Destination file.
    data
        2-D array of shape (N, 4): forward ext, forward defl, backward
        ext, backward defl.
    header
        Optional text line written before the data.
    """
    lines: list[str] = []
    if header is not None:
        lines.append(header)
    for row in data:
        lines.append(" ".join(f"{v:.10g}" for v in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _forward_zeros(N: int) -> np.ndarray:
    """Build a 4-column array with a clean forward sweep and a mirrored
    backward sweep, both with zero deflection. Useful for structure
    tests that don't care about physical values."""
    ext_m = np.linspace(0.0, 300e-9, N)
    defl_N = np.zeros(N)
    return np.column_stack([ext_m, defl_N, ext_m[::-1], defl_N[::-1]])


# -- can_load -------------------------------------------------------------


class TestCanLoad:
    """Sniffing logic must be cheap and unambiguous."""

    def test_valid_no_header_returns_true(self, tmp_path: Path) -> None:
        p = tmp_path / "ok.txt"
        _write_jpk(p, _forward_zeros(100))
        assert JPKTxtLoader().can_load(p) is True

    def test_valid_with_header_returns_true(self, tmp_path: Path) -> None:
        p = tmp_path / "hdr.txt"
        _write_jpk(p, _forward_zeros(100), header="ext_f defl_f ext_b defl_b")
        assert JPKTxtLoader().can_load(p) is True

    def test_header_with_units_returns_true(self, tmp_path: Path) -> None:
        # JPK exports sometimes use tab-separated header fields with units.
        p = tmp_path / "hdr_units.txt"
        _write_jpk(
            p,
            _forward_zeros(50),
            header="piezo_f[m]\tdefl_f[N]\tpiezo_b[m]\tdefl_b[N]",
        )
        assert JPKTxtLoader().can_load(p) is True

    def test_nonexistent_returns_false(self, tmp_path: Path) -> None:
        assert JPKTxtLoader().can_load(tmp_path / "missing.txt") is False

    def test_non_txt_extension_returns_false(self, tmp_path: Path) -> None:
        p = tmp_path / "ok.csv"
        p.write_text("1 2 3 4\n", encoding="utf-8")
        assert JPKTxtLoader().can_load(p) is False

    def test_three_column_file_returns_false(self, tmp_path: Path) -> None:
        p = tmp_path / "3col.txt"
        p.write_text("1.0 2.0 3.0\n4.0 5.0 6.0\n", encoding="utf-8")
        assert JPKTxtLoader().can_load(p) is False

    def test_five_column_file_returns_false(self, tmp_path: Path) -> None:
        p = tmp_path / "5col.txt"
        p.write_text("1 2 3 4 5\n", encoding="utf-8")
        assert JPKTxtLoader().can_load(p) is False

    def test_empty_file_returns_false(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.txt"
        p.write_text("", encoding="utf-8")
        assert JPKTxtLoader().can_load(p) is False

    def test_blank_only_file_returns_false(self, tmp_path: Path) -> None:
        p = tmp_path / "blanks.txt"
        p.write_text("\n\n   \n", encoding="utf-8")
        assert JPKTxtLoader().can_load(p) is False

    def test_directory_returns_false(self, tmp_path: Path) -> None:
        assert JPKTxtLoader().can_load(tmp_path) is False


# -- load: shape and structure -------------------------------------------


class TestLoadStructure:
    """The loader always returns 2 curves per file (approach + retract)."""

    def test_no_header_returns_two_curves(self, tmp_path: Path) -> None:
        p = tmp_path / "ok.txt"
        N = 300
        _write_jpk(p, _forward_zeros(N))
        batch = JPKTxtLoader().load(p, k_cantilever=0.1)
        assert isinstance(batch, CurveBatch)
        assert len(batch) == 2
        for c in batch:
            assert isinstance(c, ForceCurve)
            assert c.n_points == N

    def test_with_header_returns_two_curves(self, tmp_path: Path) -> None:
        p = tmp_path / "hdr.txt"
        N = 500
        _write_jpk(p, _forward_zeros(N), header="ext_f defl_f ext_b defl_b")
        batch = JPKTxtLoader().load(p, k_cantilever=0.05)
        assert len(batch) == 2
        assert batch[0].n_points == N
        assert batch[1].n_points == N

    def test_trailing_blank_lines_are_tolerated(self, tmp_path: Path) -> None:
        p = tmp_path / "trail.txt"
        N = 200
        _write_jpk(p, _forward_zeros(N))
        with p.open("a", encoding="utf-8") as fh:
            fh.write("\n\n\n")
        batch = JPKTxtLoader().load(p, k_cantilever=0.1)
        assert batch[0].n_points == N

    def test_curves_are_force_curves_with_finite_values(self, tmp_path: Path) -> None:
        p = tmp_path / "ok.txt"
        _write_jpk(p, _forward_zeros(200))
        batch = JPKTxtLoader().load(p, k_cantilever=0.1)
        for c in batch:
            assert np.all(np.isfinite(c.extension))
            assert np.all(np.isfinite(c.force))


# -- load: unit conversion -----------------------------------------------


class TestUnitConversion:
    """The conversion math must match the original Igor block exactly."""

    def test_force_sign_flip_and_scale_to_pN(self, tmp_path: Path) -> None:
        # Deflection in N is negative on the sample side. After the
        # -F*1e12 transform it becomes positive (1 pN per 1e-12 N).
        N = 300
        piezo_m = np.linspace(0.0, 300e-9, N)
        defl_f_N = -1e-12 * np.ones(N)
        defl_b_N = -1e-12 * np.ones(N)
        data = np.column_stack([piezo_m, defl_f_N, piezo_m[::-1], defl_b_N[::-1]])
        p = tmp_path / "force.txt"
        _write_jpk(p, data)
        batch = JPKTxtLoader().load(p, k_cantilever=0.1)
        # Pre-baseline force is +1 pN everywhere; baseline subtracts 1.
        np.testing.assert_allclose(batch[0].force, np.zeros(N), atol=1e-12)
        np.testing.assert_allclose(batch[1].force, np.zeros(N), atol=1e-12)

    def test_extension_subtracts_cantilever_deflection(self, tmp_path: Path) -> None:
        # A constant -2e-12 N deflection produces a +2 pN force, which
        # at k=0.1 pN/nm corresponds to a 20 nm cantilever correction.
        N = 300
        piezo_m = np.linspace(0.0, 300e-9, N)
        defl_N = -2e-12 * np.ones(N)
        data = np.column_stack([piezo_m, defl_N, piezo_m[::-1], defl_N[::-1]])
        p = tmp_path / "ext.txt"
        _write_jpk(p, data)
        k = 0.1  # pN/nm
        batch = JPKTxtLoader().load(p, k_cantilever=k)
        piezo_nm = piezo_m * 1e9
        cantilever_nm = 2.0 / k  # = 20 nm

        # Forward: ext[i] = piezo_nm[i] - 20, then shift so last = 0.
        pre_fwd = piezo_nm - cantilever_nm
        post_fwd = pre_fwd - pre_fwd[-1]
        np.testing.assert_allclose(batch[0].extension, post_fwd, atol=1e-9)

        # Backward uses a reversed piezo axis, so its baseline shift
        # is the negative end of the sweep. ext[i] = piezo_nm[N-1-i] - 20.
        piezo_nm_b = piezo_m[::-1] * 1e9
        pre_bwd = piezo_nm_b - cantilever_nm
        post_bwd = pre_bwd - pre_bwd[-1]
        np.testing.assert_allclose(batch[1].extension, post_bwd, atol=1e-9)

    def test_zero_deflection_gives_unchanged_piezo_in_nm(self, tmp_path: Path) -> None:
        # With zero deflection, the only post-processing is shifting
        # the extension axis so the last point lands on 0 nm.
        N = 250
        piezo_m = np.linspace(10e-9, 200e-9, N)
        data = _forward_zeros(0)  # placeholder
        data = np.column_stack([piezo_m, np.zeros(N), piezo_m[::-1], np.zeros(N)])
        p = tmp_path / "zero_defl.txt"
        _write_jpk(p, data)
        batch = JPKTxtLoader().load(p, k_cantilever=0.1)
        np.testing.assert_allclose(batch[0].force, np.zeros(N), atol=1e-15)
        piezo_nm = piezo_m * 1e9
        np.testing.assert_allclose(batch[0].extension, piezo_nm - piezo_nm[-1], atol=1e-9)


# -- load: baseline correction -------------------------------------------


class TestBaseline:
    """The baseline is the mean of the first 200 force points and the
    last extension point — computed per direction."""

    def test_force_baseline_is_mean_of_first_200(self, tmp_path: Path) -> None:
        N = 500
        piezo_m = np.linspace(0.0, 300e-9, N)
        idx = np.arange(N)
        # Construct a force that ramps linearly so the first-200 mean
        # is distinguishable from the global mean.
        # Pre-baseline force: 0.5 - 0.01*i  (in pN)
        defl_f_N = -(0.5e-12 - 0.01e-12 * idx)
        # Backward: same shape, reversed index.
        defl_b_N = -(0.5e-12 - 0.01e-12 * idx[::-1])
        data = np.column_stack([piezo_m, defl_f_N, piezo_m[::-1], defl_b_N])
        p = tmp_path / "baseline.txt"
        _write_jpk(p, data)
        batch = JPKTxtLoader().load(p, k_cantilever=0.1)

        # Forward expectations
        pre_fwd = 0.5 - 0.01 * np.arange(N)
        baseline_fwd = pre_fwd[:200].mean()
        np.testing.assert_allclose(batch[0].force, pre_fwd - baseline_fwd, atol=1e-9)

        # Backward expectations
        pre_bwd = 0.5 - 0.01 * (N - 1 - np.arange(N))
        baseline_bwd = pre_bwd[:200].mean()
        np.testing.assert_allclose(batch[1].force, pre_bwd - baseline_bwd, atol=1e-9)

    def test_ext_baseline_is_last_point(self, tmp_path: Path) -> None:
        N = 400
        piezo_m = np.linspace(0.0, 300e-9, N)
        data = np.column_stack([piezo_m, np.zeros(N), piezo_m[::-1], np.zeros(N)])
        p = tmp_path / "ext_baseline.txt"
        _write_jpk(p, data)
        batch = JPKTxtLoader().load(p, k_cantilever=0.1)
        # The last point of the corrected extension is always 0.
        assert batch[0].extension[-1] == pytest.approx(0.0)
        assert batch[1].extension[-1] == pytest.approx(0.0)
        # The first point of forward ext becomes -300 nm (full piezo
        # sweep minus the last-point shift).
        assert batch[0].extension[0] == pytest.approx(-300.0, abs=1e-9)

    def test_baseline_handles_short_curves(self, tmp_path: Path) -> None:
        # Files with fewer than 200 points should still load — the
        # baseline just uses all available points. (``[:200]`` on a
        # shorter array is a no-op truncation, not an error.)
        N = 50
        piezo_m = np.linspace(0.0, 50e-9, N)
        defl_N = -1e-12 * np.ones(N)
        data = np.column_stack([piezo_m, defl_N, piezo_m[::-1], defl_N[::-1]])
        p = tmp_path / "short.txt"
        _write_jpk(p, data)
        batch = JPKTxtLoader().load(p, k_cantilever=0.1)
        # Mean of all 50 points is 1 pN — every force point should be 0.
        np.testing.assert_allclose(batch[0].force, np.zeros(N), atol=1e-12)


# -- load: error handling -------------------------------------------------


class TestLoadErrors:
    def test_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises((FileNotFoundError, OSError)):
            JPKTxtLoader().load(tmp_path / "missing.txt", k_cantilever=0.1)

    def test_k_cantilever_zero_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "ok.txt"
        _write_jpk(p, _forward_zeros(100))
        with pytest.raises(ValueError, match="k_cantilever"):
            JPKTxtLoader().load(p, k_cantilever=0.0)

    def test_k_cantilever_negative_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "ok.txt"
        _write_jpk(p, _forward_zeros(100))
        with pytest.raises(ValueError, match="k_cantilever"):
            JPKTxtLoader().load(p, k_cantilever=-0.1)

    def test_k_cantilever_nan_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "ok.txt"
        _write_jpk(p, _forward_zeros(100))
        with pytest.raises(ValueError, match="k_cantilever"):
            JPKTxtLoader().load(p, k_cantilever=float("nan"))

    def test_k_cantilever_inf_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "ok.txt"
        _write_jpk(p, _forward_zeros(100))
        with pytest.raises(ValueError, match="k_cantilever"):
            JPKTxtLoader().load(p, k_cantilever=float("inf"))

    def test_wrong_column_count_raises_informative_error(self, tmp_path: Path) -> None:
        p = tmp_path / "3col.txt"
        p.write_text("1.0 2.0 3.0\n4.0 5.0 6.0\n", encoding="utf-8")
        with pytest.raises(ValueError, match=r"4"):
            JPKTxtLoader().load(p, k_cantilever=0.1)

    def test_non_numeric_content_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "junk.txt"
        p.write_text("a b c d\ne f g h\n", encoding="utf-8")
        with pytest.raises(ValueError):
            JPKTxtLoader().load(p, k_cantilever=0.1)

    def test_header_only_file_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "hdr_only.txt"
        p.write_text("ext_f defl_f ext_b defl_b\n", encoding="utf-8")
        with pytest.raises(ValueError):
            JPKTxtLoader().load(p, k_cantilever=0.1)


# -- metadata -------------------------------------------------------------


class TestMetadata:
    """The loader threads source / direction / k_cantilever through both
    the per-curve and the batch-level metadata dicts."""

    def test_per_curve_metadata(self, tmp_path: Path) -> None:
        p = tmp_path / "ok.txt"
        _write_jpk(p, _forward_zeros(200))
        k = 0.07
        batch = JPKTxtLoader().load(p, k_cantilever=k)
        assert batch[0].metadata["direction"] == "approach"
        assert batch[1].metadata["direction"] == "retract"
        assert batch[0].metadata["k_cantilever"] == k
        assert batch[1].metadata["k_cantilever"] == k
        assert batch[0].metadata["source_file"] == str(p)
        assert batch[1].metadata["source_file"] == str(p)

    def test_batch_level_metadata(self, tmp_path: Path) -> None:
        p = tmp_path / "ok.txt"
        _write_jpk(p, _forward_zeros(200))
        k = 0.07
        batch = JPKTxtLoader().load(p, k_cantilever=k)
        assert batch.metadata["k_cantilever"] == k
        assert batch.metadata["source"] == str(p)

    def test_metadata_does_not_mutate_between_curves(self, tmp_path: Path) -> None:
        # Editing one curve's metadata must not leak into the other.
        p = tmp_path / "ok.txt"
        _write_jpk(p, _forward_zeros(200))
        batch = JPKTxtLoader().load(p, k_cantilever=0.1)
        batch[0].with_metadata(operator="alice")
        assert "operator" not in batch[1].metadata


# -- module-level helper --------------------------------------------------


class TestLoadHelper:
    def test_helper_matches_loader(self, tmp_path: Path) -> None:
        p = tmp_path / "ok.txt"
        _write_jpk(p, _forward_zeros(150))
        a = load_jpk_txt(p, k_cantilever=0.05)
        b = JPKTxtLoader().load(p, k_cantilever=0.05)
        assert len(a) == len(b) == 2
        np.testing.assert_array_equal(a[0].force, b[0].force)
        np.testing.assert_array_equal(a[0].extension, b[0].extension)
        np.testing.assert_array_equal(a[1].force, b[1].force)
        np.testing.assert_array_equal(a[1].extension, b[1].extension)

    def test_helper_accepts_string_path(self, tmp_path: Path) -> None:
        p = tmp_path / "ok.txt"
        _write_jpk(p, _forward_zeros(120))
        batch = load_jpk_txt(str(p), k_cantilever=0.05)
        assert isinstance(batch, CurveBatch)
        assert len(batch) == 2
        assert batch[0].metadata["source_file"] == str(p)

    def test_helper_propagates_k_validation(self, tmp_path: Path) -> None:
        p = tmp_path / "ok.txt"
        _write_jpk(p, _forward_zeros(100))
        with pytest.raises(ValueError, match="k_cantilever"):
            load_jpk_txt(p, k_cantilever=0.0)


# -- Protocol compliance -------------------------------------------------


class TestProtocolCompliance:
    """The class should be usable as a :class:`Loader`."""

    def test_has_name_attribute(self) -> None:
        assert JPKTxtLoader.name == "jpk_txt"

    def test_load_signature_accepts_kwargs(self, tmp_path: Path) -> None:
        # Extra kwargs must be silently absorbed (LSP compliance with
        # the Loader Protocol's ``**kwargs``).
        p = tmp_path / "ok.txt"
        _write_jpk(p, _forward_zeros(100))
        loader = JPKTxtLoader()
        batch = loader.load(p, k_cantilever=0.1, some_future_option=True)
        assert len(batch) == 2
