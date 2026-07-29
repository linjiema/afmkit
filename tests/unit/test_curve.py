"""Unit tests for :mod:`afmkit.core.curve`."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from afmkit.core.curve import CurveBatch, ForceCurve

# -- Construction ---------------------------------------------------------


class TestForceCurveConstruction:
    def test_basic_construction(self) -> None:
        ext = np.linspace(0, 100, 100)
        force = np.zeros_like(ext)
        curve = ForceCurve(ext, force)
        assert curve.n_points == 100
        np.testing.assert_array_equal(curve.extension, ext)
        np.testing.assert_array_equal(curve.force, force)

    def test_metadata_is_stored(self) -> None:
        curve = ForceCurve([0, 1, 2], [0, 1, 4], metadata={"k_cantilever": 0.1})
        assert curve.metadata == {"k_cantilever": 0.1}

    def test_default_metadata_is_empty_dict(self) -> None:
        curve = ForceCurve([0, 1], [0, 1])
        assert curve.metadata == {}

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            ForceCurve([0, 1, 2], [0, 1])

    def test_nan_in_force_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            ForceCurve([0, 1, 2], [0, 1, np.nan])

    def test_inf_in_extension_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            ForceCurve([0, 1, np.inf], [0, 1, 2])

    def test_empty_array_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one point"):
            ForceCurve([], [])

    def test_multidimensional_raises(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            ForceCurve(np.zeros((3, 3)), np.zeros((3, 3)))

    def test_input_lists_are_coerced_to_float64(self) -> None:
        curve = ForceCurve([0, 1, 2], [0, 1, 4])
        assert curve.extension.dtype == np.float64
        assert curve.force.dtype == np.float64

    def test_repr_contains_k_if_present(self) -> None:
        curve = ForceCurve([0, 1], [0, 1], metadata={"k_cantilever": 0.1})
        assert "k=0.1" in repr(curve)

    def test_len(self) -> None:
        curve = ForceCurve([0, 1, 2, 3], [0, 1, 2, 3])
        assert len(curve) == 4


# -- xarray interop -------------------------------------------------------


class TestForceCurveXarrayInterop:
    def test_to_xarray_roundtrip(self) -> None:
        original = ForceCurve([0, 1, 2], [0, 1, 4], metadata={"k_cantilever": 0.2})
        ds = original.to_xarray()
        assert "extension" in ds.data_vars
        assert "force" in ds.data_vars
        assert ds.attrs["k_cantilever"] == 0.2

        restored = ForceCurve.from_xarray(ds)
        np.testing.assert_array_equal(restored.extension, original.extension)
        np.testing.assert_array_equal(restored.force, original.force)
        assert restored.metadata == original.metadata

    def test_to_xarray_returns_independent_copy(self) -> None:
        curve = ForceCurve([0, 1, 2], [0, 1, 4])
        ds = curve.to_xarray()
        ds["force"].values[0] = 999.0
        assert curve.force[0] == 0.0  # original curve unchanged

    def test_from_xarray_missing_variable_raises(self) -> None:
        ds = xr.Dataset({"extension": ("point", [0, 1, 2])})
        with pytest.raises(ValueError, match="missing required variables"):
            ForceCurve.from_xarray(ds)


# -- Transformations -----------------------------------------------------


class TestForceCurveTransformations:
    def test_select_range_basic(self) -> None:
        ext = np.linspace(0, 100, 101)
        force = ext.copy()
        curve = ForceCurve(ext, force)
        sub = curve.select_range(20, 40)
        assert sub.n_points == 21
        assert sub.extension.min() == pytest.approx(20.0)
        assert sub.extension.max() == pytest.approx(40.0)

    def test_select_range_exclusive(self) -> None:
        ext = np.array([0, 10, 20, 30, 40], dtype=float)
        force = ext.copy()
        curve = ForceCurve(ext, force)
        sub = curve.select_range(10, 30, inclusive=False)
        np.testing.assert_array_equal(sub.extension, [20.0])

    def test_select_range_inverted_raises(self) -> None:
        curve = ForceCurve([0, 1, 2], [0, 1, 2])
        with pytest.raises(ValueError, match="must be <="):
            curve.select_range(50, 10)

    def test_select_range_empty_raises(self) -> None:
        curve = ForceCurve([0, 1, 2], [0, 1, 2])
        with pytest.raises(ValueError, match="empty curve"):
            curve.select_range(100, 200)

    def test_select_range_handles_non_monotonic(self) -> None:
        # Approach + retract: extension goes 0 -> 100 -> 0.
        ext = np.array([0, 25, 50, 75, 100, 75, 50, 25, 0], dtype=float)
        force = np.arange(ext.size, dtype=float)
        curve = ForceCurve(ext, force)
        # [40, 80] picks up 50 and 75 from both halves of the cycle.
        sub = curve.select_range(40, 80)
        # Should include all points where ext is in [40, 80], regardless of order.
        np.testing.assert_array_equal(sub.extension, [50, 75, 75, 50])

    def test_with_metadata_adds_keys(self) -> None:
        curve = ForceCurve([0, 1], [0, 1])
        updated = curve.with_metadata(k_cantilever=0.1, operator="alice")
        assert updated.metadata == {"k_cantilever": 0.1, "operator": "alice"}

    def test_with_metadata_preserves_existing(self) -> None:
        curve = ForceCurve([0, 1], [0, 1], metadata={"k_cantilever": 0.1})
        updated = curve.with_metadata(operator="alice")
        assert updated.metadata == {"k_cantilever": 0.1, "operator": "alice"}

    def test_with_metadata_does_not_mutate_original(self) -> None:
        curve = ForceCurve([0, 1], [0, 1], metadata={"k_cantilever": 0.1})
        curve.with_metadata(operator="alice")
        assert "operator" not in curve.metadata

    def test_with_force_replaces_axis(self) -> None:
        curve = ForceCurve([0, 1, 2], [0, 1, 4])
        updated = curve.with_force([10, 20, 30])
        np.testing.assert_array_equal(updated.force, [10, 20, 30])
        np.testing.assert_array_equal(curve.force, [0, 1, 4])  # original intact

    def test_with_force_length_mismatch_raises(self) -> None:
        curve = ForceCurve([0, 1, 2], [0, 1, 4])
        with pytest.raises(ValueError, match="match extension length"):
            curve.with_force([0, 1])

    def test_metadata_property_returns_copy(self) -> None:
        curve = ForceCurve([0, 1], [0, 1], metadata={"k_cantilever": 0.1})
        meta = curve.metadata
        meta["operator"] = "alice"
        assert "operator" not in curve.metadata  # original unchanged


# -- CurveBatch ----------------------------------------------------------


class TestCurveBatch:
    def test_construction_from_iterable(self) -> None:
        curves = [ForceCurve([0, 1], [0, 1]) for _ in range(3)]
        batch = CurveBatch(curves)
        assert batch.n_curves == 3

    def test_construction_rejects_non_ForceCurve(self) -> None:
        with pytest.raises(TypeError, match="ForceCurve"):
            CurveBatch([1, 2, 3])  # type: ignore[list-item]

    def test_construction_from_generator(self) -> None:
        gen = (ForceCurve([0, 1], [0, 1]) for _ in range(5))
        batch = CurveBatch(gen)
        assert batch.n_curves == 5

    def test_indexing_returns_curve(self) -> None:
        c0 = ForceCurve([0, 1], [0, 1])
        c1 = ForceCurve([0, 1], [2, 3])
        batch = CurveBatch([c0, c1])
        assert batch[0] is c0
        assert batch[1] is c1

    def test_slicing_returns_batch(self) -> None:
        curves = [ForceCurve([i], [i * 2.0]) for i in range(5)]
        batch = CurveBatch(curves)
        sub = batch[1:4]
        assert isinstance(sub, CurveBatch)
        assert sub.n_curves == 3
        np.testing.assert_array_equal(sub[0].extension, [1.0])

    def test_iteration(self) -> None:
        curves = [ForceCurve([i], [i * 2.0]) for i in range(3)]
        batch = CurveBatch(curves)
        seen = [c.extension[0] for c in batch]
        np.testing.assert_array_equal(seen, [0, 1, 2])

    def test_len(self) -> None:
        batch = CurveBatch([ForceCurve([0, 1], [0, 1]) for _ in range(7)])
        assert len(batch) == 7

    def test_select(self) -> None:
        curves = [ForceCurve([i], [i * 2.0]) for i in range(5)]
        batch = CurveBatch(curves)
        sub = batch.select([0, 2, 4])
        assert sub.n_curves == 3
        np.testing.assert_array_equal([c.extension[0] for c in sub], [0, 2, 4])

    def test_extensions_and_forces_iterators(self) -> None:
        curves = [ForceCurve([i], [i * 2.0]) for i in range(3)]
        batch = CurveBatch(curves)
        assert list(batch.extensions()) == [[0.0], [1.0], [2.0]]
        assert list(batch.forces()) == [[0.0], [2.0], [4.0]]

    def test_to_xarray_empty_batch(self) -> None:
        ds = CurveBatch([]).to_xarray()
        assert isinstance(ds, xr.Dataset)
        assert len(ds.data_vars) == 0

    def test_to_xarray_combines_curves(self) -> None:
        curves = [
            ForceCurve([0, 1, 2], [0, 1, 4]),
            ForceCurve([0, 1, 2, 3], [0, 1, 4, 9]),
        ]
        batch = CurveBatch(curves, name="exp1", metadata={"k_cantilever": 0.1})
        ds = batch.to_xarray()
        assert ds.sizes["curve"] == 2
        assert ds.sizes["point"] == 4
        assert "extension" in ds.data_vars
        assert "force" in ds.data_vars
        assert ds.attrs["batch_name"] == "exp1"
        # Shorter curve should be NaN-padded.
        assert np.isnan(ds["force"].values[0, 3])
        assert ds["force"].values[1, 3] == 9.0

    def test_repr(self) -> None:
        batch = CurveBatch([ForceCurve([0, 1], [0, 1])], name="exp1")
        assert "CurveBatch" in repr(batch)
        assert "exp1" in repr(batch)
        assert "n_curves=1" in repr(batch)


# -- Physics sanity ------------------------------------------------------


class TestForceCurvePhysics:
    """Sanity checks that the data model preserves numerical relationships."""

    def test_wlc_force_is_monotonic_in_extension(self) -> None:
        # The Marko-Siggia WLC is monotonically increasing for x in (0, L).
        ext = np.linspace(0.5, 199, 2000)  # avoid x=0 singularity
        p, lc = 0.4, 200.0
        force = (4.1 / p) * (0.25 * (1.0 - ext / lc) ** -2 - 0.25 + ext / lc)
        curve = ForceCurve(ext, force)
        assert np.all(np.diff(curve.force) > 0)

    def test_known_p_L_can_be_recovered(self) -> None:
        # We use the fact that the force at x = L/2 is a known function of p.
        p, lc = 0.4, 200.0
        x = lc / 2
        f_expected = (4.1 / p) * (0.25 * (1.0 - x / lc) ** -2 - 0.25 + x / lc)
        ext = np.array([x])
        force = np.array([f_expected])
        curve = ForceCurve(ext, force)
        assert curve.n_points == 1
        assert curve.force[0] == pytest.approx(f_expected, rel=1e-12)
