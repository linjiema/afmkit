"""Unit tests for :mod:`afmkit.io.hdf5_store`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from afmkit import __version__
from afmkit.core.curve import CurveBatch, ForceCurve
from afmkit.io.hdf5_store import HDF5Store, load_hdf5, save_hdf5

# -- Helpers --------------------------------------------------------------


def _make_curve(n: int, *, k: float = 0.1, **extra: Any) -> ForceCurve:
    """Build a synthetic :class:`ForceCurve` of ``n`` points.

    The values are deliberately non-trivial (linspace + a ramp) so that
    an accidental zero-fill of either axis shows up in the tests.
    """
    ext = np.linspace(0.0, 100.0, n)
    force = np.linspace(-5.0, 25.0, n)
    meta: dict[str, Any] = {"k_cantilever": k, "n": n}
    meta.update(extra)
    return ForceCurve(ext, force, metadata=meta)


def _assert_curves_equal(a: ForceCurve, b: ForceCurve) -> None:
    """Strict per-curve equality: arrays exact, metadata dict-equal."""
    np.testing.assert_array_equal(a.extension, b.extension)
    np.testing.assert_array_equal(a.force, b.force)
    assert a.metadata == b.metadata
    assert a.n_points == b.n_points


# -- Round-trip: shape & data --------------------------------------------


class TestRoundTrip:
    """The store must round-trip every field losslessly."""

    def test_three_curves_ragged_lengths(self, tmp_path: Path) -> None:
        p = tmp_path / "ragged.h5"
        curves = [
            _make_curve(100, k=0.1),
            _make_curve(50, k=0.2, direction="retract"),
            _make_curve(7, k=0.05, direction="approach", tag="tiny"),
        ]
        batch = CurveBatch(curves, name="ragged", metadata={"source": "synthetic"})

        save_hdf5(batch, p)
        loaded = load_hdf5(p)

        assert loaded.name == "ragged"
        assert loaded.metadata == {"source": "synthetic"}
        assert len(loaded) == 3
        for orig, back in zip(curves, loaded, strict=True):
            _assert_curves_equal(orig, back)

    def test_round_trip_preserves_dtype_and_shape(self, tmp_path: Path) -> None:
        p = tmp_path / "dtype.h5"
        c = _make_curve(123, k=0.07)
        save_hdf5(CurveBatch([c]), p)
        loaded = load_hdf5(p)
        assert loaded[0].extension.dtype == np.float64
        assert loaded[0].force.dtype == np.float64
        assert loaded[0].extension.shape == (123,)
        assert loaded[0].force.shape == (123,)

    def test_round_trip_via_class_matches_helper(self, tmp_path: Path) -> None:
        p = tmp_path / "api.h5"
        batch = CurveBatch([_make_curve(10)], name="api")
        HDF5Store().save(batch, p)
        loaded_a = load_hdf5(p)
        loaded_b = HDF5Store().load(p)
        assert loaded_a.name == loaded_b.name == "api"
        _assert_curves_equal(loaded_a[0], loaded_b[0])

    def test_load_accepts_string_path(self, tmp_path: Path) -> None:
        p = tmp_path / "str.h5"
        save_hdf5(CurveBatch([_make_curve(5)]), str(p))
        loaded = load_hdf5(str(p))
        assert len(loaded) == 1


# -- Metadata: variety of types ----------------------------------------


class TestMetadata:
    """Arbitrary metadata dicts must round-trip, including numpy types."""

    def test_batch_metadata_round_trips(self, tmp_path: Path) -> None:
        p = tmp_path / "bmd.h5"
        meta: dict[str, Any] = {
            "k_cantilever": 0.123,
            "operator": "alice",
            "tags": ["WT", "pH7.4", "2024-08"],
            "nested": {"a": 1, "b": [2, 3, 4]},
        }
        batch = CurveBatch([_make_curve(20)], name="x", metadata=meta)
        save_hdf5(batch, p)
        loaded = load_hdf5(p)
        assert loaded.metadata == meta

    def test_numpy_arrays_in_metadata_become_lists(self, tmp_path: Path) -> None:
        p = tmp_path / "nparr.h5"
        meta = {
            "waveform": np.array([1.0, 2.0, 3.0]),
            "scalar_int": np.int64(42),
            "scalar_float": np.float64(3.14),
            "scalar_bool": np.bool_(True),
        }
        save_hdf5(CurveBatch([_make_curve(10)], metadata=meta), p)
        loaded = load_hdf5(p)
        # Arrays become lists on the way through JSON; scalars become
        # native Python types. Compare by value, not by np type.
        assert loaded.metadata["waveform"] == [1.0, 2.0, 3.0]
        assert loaded.metadata["scalar_int"] == 42
        assert loaded.metadata["scalar_float"] == 3.14
        assert loaded.metadata["scalar_bool"] is True

    def test_per_curve_metadata_round_trips(self, tmp_path: Path) -> None:
        p = tmp_path / "cmd.h5"
        c1 = _make_curve(20, k=0.1, direction="approach")
        c2 = _make_curve(30, k=0.2, direction="retract", operator="bob")
        save_hdf5(CurveBatch([c1, c2]), p)
        loaded = load_hdf5(p)
        assert loaded[0].metadata == c1.metadata
        assert loaded[1].metadata == c2.metadata

    def test_unknown_metadata_keys_preserved(self, tmp_path: Path) -> None:
        # Future afmkit versions may add metadata keys; load must
        # tolerate them silently rather than stripping.
        p = tmp_path / "future.h5"
        c = _make_curve(5, experimental_flag=True, future_key=42)
        save_hdf5(CurveBatch([c]), p)
        loaded = load_hdf5(p)
        assert loaded[0].metadata["experimental_flag"] is True
        assert loaded[0].metadata["future_key"] == 42


# -- File format: attrs & layout ---------------------------------------


class TestFileFormat:
    """The on-disk shape must match the spec exactly."""

    def test_root_attrs_present(self, tmp_path: Path) -> None:
        p = tmp_path / "fmt.h5"
        save_hdf5(CurveBatch([_make_curve(5)], name="named", metadata={"x": 1}), p)
        with h5py.File(p, "r") as fh:
            assert fh.attrs["afmkit_version"] == __version__
            assert fh.attrs["batch_name"] == "named"
            assert "batch_metadata" in fh.attrs
            # The batch_metadata attr must be a JSON string we can parse.
            import json

            assert json.loads(fh.attrs["batch_metadata"]) == {"x": 1}

    def test_curve_groups_have_datasets_and_attrs(self, tmp_path: Path) -> None:
        p = tmp_path / "layout.h5"
        c = _make_curve(8, k=0.1, direction="approach")
        save_hdf5(CurveBatch([c]), p)
        with h5py.File(p, "r") as fh:
            grp = fh["curves/curve_0000"]
            assert "extension" in grp
            assert "force" in grp
            assert grp.attrs["n_points"] == 8
            assert "metadata" in grp.attrs
            # Datasets should be 1-D float64.
            assert grp["extension"].shape == (8,)
            assert grp["extension"].dtype == np.float64
            assert grp["force"].shape == (8,)
            assert grp["force"].dtype == np.float64

    def test_curves_ordered_by_name_on_load(self, tmp_path: Path) -> None:
        # Even if the on-disk groups are written out of order, load
        # must return them in index order so the user can rely on
        # position.
        p = tmp_path / "order.h5"
        with h5py.File(p, "w") as fh:
            fh.attrs["afmkit_version"] = __version__
            fh.attrs["batch_name"] = ""
            fh.attrs["batch_metadata"] = "{}"
            curves_group = fh.create_group("curves")
            # Write curve_0001 first, then curve_0000, then curve_0002.
            for idx, n in [(1, 30), (0, 10), (2, 20)]:
                grp = curves_group.create_group(f"curve_{idx:04d}")
                grp.attrs["n_points"] = n
                grp.attrs["metadata"] = "{}"
                grp.create_dataset("extension", data=np.arange(n, dtype=np.float64))
                grp.create_dataset("force", data=np.arange(n, dtype=np.float64))
        loaded = load_hdf5(p)
        # The stored `n` values uniquely identify each curve.
        assert [c.n_points for c in loaded] == [10, 30, 20]

    def test_curve_groups_use_zero_padded_names(self, tmp_path: Path) -> None:
        # 5 curves -> curve_0000 through curve_0004, all 4 digits.
        p = tmp_path / "pad.h5"
        save_hdf5(CurveBatch([_make_curve(2) for _ in range(5)]), p)
        with h5py.File(p, "r") as fh:
            names = sorted(fh["curves"].keys())
        assert names == [f"curve_{i:04d}" for i in range(5)]


# -- Compression ---------------------------------------------------------


class TestCompression:
    """Default gzip compression must actually shrink large datasets."""

    def test_gzip_shrinks_repetitive_data(self, tmp_path: Path) -> None:
        # Build a long curve with many runs of identical float values
        # on *both* axes — the kind of structure gzip actually
        # exploits. (Smooth numeric signals at full float64 precision
        # are *not* very compressible because the low bits keep
        # changing, so a pure sin/linspace baseline would inflate the
        # uncompressed side without giving gzip much to work with.)
        # Both axes are staircases: 50 plateaus of 1000 points each,
        # giving 50_000 points total.
        ext = np.repeat(np.linspace(0.0, 1000.0, 50), 1000)
        force = np.repeat(np.linspace(-5.0, 25.0, 50), 1000)
        c = ForceCurve(ext, force)
        batch = CurveBatch([c])

        p_compressed = tmp_path / "cmp.h5"
        p_uncompressed = tmp_path / "uncmp.h5"
        save_hdf5(batch, p_compressed, compression="gzip", compression_opts=4)
        # When compression is None, h5py rejects a non-None
        # compression_opts — so pass None explicitly here.
        save_hdf5(batch, p_uncompressed, compression=None, compression_opts=None)

        size_cmp = p_compressed.stat().st_size
        size_uncmp = p_uncompressed.stat().st_size
        assert (
            size_cmp < size_uncmp
        ), f"gzip ({size_cmp} B) should be smaller than uncompressed ({size_uncmp} B)"
        # A double-staircase signal at 8-byte floats should compress
        # by a comfortable margin with gzip; we assert a conservative
        # 3x to avoid flakiness across zlib versions.
        assert (
            size_uncmp / size_cmp > 3.0
        ), f"compression ratio {size_uncmp / size_cmp:.2f}x is below the 3x floor"


# -- Validation / error handling ----------------------------------------


class TestValidation:
    """Loading bad files must raise informative ``ValueError``s."""

    def test_non_hdf5_file_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "junk.h5"
        p.write_text("this is not an HDF5 file\n", encoding="utf-8")
        with pytest.raises(ValueError, match="not a valid HDF5 file"):
            load_hdf5(p)

    def test_hdf5_without_afmkit_version_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "nover.h5"
        with h5py.File(p, "w") as fh:
            fh.create_dataset("foo", data=np.arange(5))
        with pytest.raises(ValueError, match="afmkit_version"):
            load_hdf5(p)

    def test_hdf5_with_wrong_attr_raises(self, tmp_path: Path) -> None:
        # An HDF5 file with a different marker attr is still not an
        # afmkit store.
        p = tmp_path / "other.h5"
        with h5py.File(p, "w") as fh:
            fh.attrs["some_other_marker"] = "v1"
            fh.create_dataset("data", data=np.arange(3))
        with pytest.raises(ValueError, match="afmkit_version"):
            load_hdf5(p)

    def test_unsupported_mode_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad_mode.h5"
        with pytest.raises(ValueError, match="mode"):
            save_hdf5(CurveBatch([_make_curve(3)]), p, mode="r")

    def test_curve_missing_dataset_raises(self, tmp_path: Path) -> None:
        # Hand-craft a malformed afmkit file: curve group lacks
        # `extension`. Load must refuse rather than silently dropping
        # the curve.
        p = tmp_path / "malformed.h5"
        with h5py.File(p, "w") as fh:
            fh.attrs["afmkit_version"] = __version__
            fh.attrs["batch_name"] = ""
            fh.attrs["batch_metadata"] = "{}"
            grp = fh.create_group("curves/curve_0000")
            grp.create_dataset("force", data=np.arange(5.0))
        with pytest.raises(ValueError, match="extension"):
            load_hdf5(p)


# -- Empty batch ---------------------------------------------------------


class TestEmptyBatch:
    """An empty CurveBatch is a valid batch — must round-trip trivially."""

    def test_empty_batch_round_trip(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.h5"
        batch = CurveBatch([], name="void", metadata={"k_cantilever": 0.0})
        save_hdf5(batch, p)
        loaded = load_hdf5(p)
        assert len(loaded) == 0
        assert loaded.name == "void"
        assert loaded.metadata == {"k_cantilever": 0.0}

    def test_empty_batch_still_marks_file_as_afmkit(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.h5"
        save_hdf5(CurveBatch([]), p)
        with h5py.File(p, "r") as fh:
            assert fh.attrs["afmkit_version"] == __version__


# -- Mode handling -------------------------------------------------------


class TestMode:
    """``mode='w'`` overwrites; ``mode='a'`` appends without replacing."""

    def test_mode_w_overwrites(self, tmp_path: Path) -> None:
        p = tmp_path / "w.h5"
        # First save: 2 curves, name "first".
        save_hdf5(CurveBatch([_make_curve(10), _make_curve(20)], name="first"), p)
        # Second save: 1 curve, different name -> whole file is replaced.
        save_hdf5(CurveBatch([_make_curve(5)], name="second"), p, mode="w")
        loaded = load_hdf5(p)
        assert loaded.name == "second"
        assert len(loaded) == 1
        assert loaded[0].n_points == 5

    def test_mode_a_appends_curves(self, tmp_path: Path) -> None:
        p = tmp_path / "a.h5"
        # Seed: 2 curves, name "shared".
        save_hdf5(
            CurveBatch(
                [_make_curve(10), _make_curve(20)],
                name="shared",
                metadata={"tag": "original"},
            ),
            p,
        )
        # Append: 1 more curve. Top-level attrs are preserved.
        save_hdf5(
            CurveBatch(
                [_make_curve(30, k=0.9)],
                name="shared",
                metadata={"tag": "original"},
            ),
            p,
            mode="a",
        )
        loaded = load_hdf5(p)
        # Top-level identity preserved.
        assert loaded.name == "shared"
        assert loaded.metadata == {"tag": "original"}
        # Curves from both writes are present, in original-then-new order.
        assert [c.n_points for c in loaded] == [10, 20, 30]
        assert loaded[2].metadata["k_cantilever"] == 0.9

    def test_mode_a_continues_numbering(self, tmp_path: Path) -> None:
        # Make sure the appended curves get *new* group names, not
        # clobbering curve_0000 / curve_0001.
        p = tmp_path / "num.h5"
        save_hdf5(CurveBatch([_make_curve(5), _make_curve(7)]), p)
        save_hdf5(CurveBatch([_make_curve(9)]), p, mode="a")
        with h5py.File(p, "r") as fh:
            names = sorted(fh["curves"].keys())
        assert names == ["curve_0000", "curve_0001", "curve_0002"]


# -- Convenience helpers -------------------------------------------------


class TestModuleLevelHelpers:
    def test_save_load_match(self, tmp_path: Path) -> None:
        p = tmp_path / "helpers.h5"
        b = CurveBatch([_make_curve(10), _make_curve(20)], name="helpers")
        save_hdf5(b, p)
        loaded = load_hdf5(p)
        assert loaded.name == "helpers"
        assert [c.n_points for c in loaded] == [10, 20]

    def test_save_kwargs_forwarded(self, tmp_path: Path) -> None:
        # `mode="a"` keyword must reach the underlying store.
        p = tmp_path / "kw.h5"
        save_hdf5(CurveBatch([_make_curve(3)]), p)
        save_hdf5(CurveBatch([_make_curve(4)]), p, mode="a")
        assert len(load_hdf5(p)) == 2
