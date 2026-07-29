"""Unit tests for :mod:`afmkit.fitting.engine`.

The fitting engine is the bridge between :mod:`afmkit.models` and the
underlying :mod:`lmfit` solver. These tests pin down the contract that
the lab's batch pipelines depend on:

- A noise-free WLC fit must recover the ground-truth parameters
  to within 1 %.
- The ``x_range`` argument must actually shrink the data handed to
  the solver (and surface on :class:`FitResult` as ``fit_range``).
- ``p0`` overrides the model's :meth:`guess_params` and is **clamped**
  to the model's bounds before being handed to the solver.
- A fit that cannot converge must return a :class:`FitResult` with
  ``metadata["success"] = False`` — not raise.

A custom "broken" :class:`PolymerModel` is used in the failed-fit test
to deterministically trigger a model-output NaN, which is the most
reliable way to drive lmfit's least-squares solver into a
``ValueError``.
"""

from __future__ import annotations

import numpy as np
import pytest

from afmkit.core.curve import ForceCurve
from afmkit.fitting import LmfitEngine, fit
from afmkit.fitting.report import FitResult
from afmkit.models import WLCModel

P_TRUTH: float = 0.4
L_TRUTH: float = 200.0
# Restrict the fit to a range well clear of both the singularity at x=L
# and the low-force region near x=0. The conftest fixtures use 0–300 nm
# and (L=200) nm is right in the middle; without `x_range` the few
# points near the singularity dominate the chi-square.
X_RANGE: tuple[float, float] = (20.0, 180.0)


# -- Recovery on a noise-free WLC ----------------------------------------


class TestWLCFitRecovery:
    """A WLC fit on noise-free data must recover the truth to high precision."""

    def test_recover_p_and_L_within_one_percent(
        self, synthetic_extension: np.ndarray, synthetic_wlc_force: np.ndarray
    ) -> None:
        engine = LmfitEngine(WLCModel())
        result = engine.fit(synthetic_extension, synthetic_wlc_force, x_range=X_RANGE)
        assert result.metadata["success"]
        # 1 % tolerance is generous for a noise-free, well-conditioned
        # nonlinear least-squares fit — the Levenberg-Marquardt solver
        # typically converges to ~1e-6 on this data.
        assert abs(result.params["p"] - P_TRUTH) / P_TRUTH < 0.01
        assert abs(result.params["L"] - L_TRUTH) / L_TRUTH < 0.01

    def test_recovered_params_have_finite_stderrs(
        self, synthetic_extension: np.ndarray, synthetic_wlc_force: np.ndarray
    ) -> None:
        engine = LmfitEngine(WLCModel())
        result = engine.fit(synthetic_extension, synthetic_wlc_force, x_range=X_RANGE)
        # Both parameters should have well-defined 1-sigma uncertainties
        # — a non-finite stderr here would signal a covariance failure.
        for name in ("p", "L"):
            assert np.isfinite(result.stderr[name]), f"stderr for {name} is not finite"

    def test_n_data_matches_filtered_grid(
        self, synthetic_extension: np.ndarray, synthetic_wlc_force: np.ndarray
    ) -> None:
        engine = LmfitEngine(WLCModel())
        result = engine.fit(synthetic_extension, synthetic_wlc_force, x_range=X_RANGE)
        # Every x_fit must lie within the requested range.
        assert np.all(result.x_fit >= X_RANGE[0])
        assert np.all(result.x_fit <= X_RANGE[1])
        # And the count must match the filtered size.
        expected = int(
            np.sum((synthetic_extension >= X_RANGE[0]) & (synthetic_extension <= X_RANGE[1]))
        )
        assert result.n_data == expected


# -- x_range filter ------------------------------------------------------


class TestXRangeFilter:
    """The ``x_range`` argument must restrict both the data and the reported stats."""

    def test_x_range_shrinks_n_data(
        self, synthetic_extension: np.ndarray, synthetic_wlc_force: np.ndarray
    ) -> None:
        engine = LmfitEngine(WLCModel())
        result_full = engine.fit(synthetic_extension, synthetic_wlc_force)
        result_sub = engine.fit(synthetic_extension, synthetic_wlc_force, x_range=X_RANGE)
        assert result_sub.n_data < result_full.n_data

    def test_x_range_is_recorded_on_metadata(
        self, synthetic_extension: np.ndarray, synthetic_wlc_force: np.ndarray
    ) -> None:
        engine = LmfitEngine(WLCModel())
        result = engine.fit(synthetic_extension, synthetic_wlc_force, x_range=X_RANGE)
        # The engine stores the range as a list for JSON-serialisability.
        assert result.metadata.get("fit_range") == [X_RANGE[0], X_RANGE[1]]

    def test_inverted_range_raises(
        self, synthetic_extension: np.ndarray, synthetic_wlc_force: np.ndarray
    ) -> None:
        engine = LmfitEngine(WLCModel())
        # x_min > x_max is a programming error, not a failed fit.
        with pytest.raises(ValueError, match="x_min"):
            engine.fit(synthetic_extension, synthetic_wlc_force, x_range=(180.0, 20.0))

    def test_empty_range_raises(
        self, synthetic_extension: np.ndarray, synthetic_wlc_force: np.ndarray
    ) -> None:
        engine = LmfitEngine(WLCModel())
        # The conftest data lives in [0, 300]; [500, 600] selects nothing.
        with pytest.raises(ValueError, match="no data points"):
            engine.fit(synthetic_extension, synthetic_wlc_force, x_range=(500.0, 600.0))

    def test_x_range_none_omits_fit_range_key(
        self, synthetic_extension: np.ndarray, synthetic_wlc_force: np.ndarray
    ) -> None:
        engine = LmfitEngine(WLCModel())
        result = engine.fit(synthetic_extension, synthetic_wlc_force)
        assert "fit_range" not in result.metadata


# -- p0 override ---------------------------------------------------------


class TestP0Override:
    """``p0`` overrides the model's :meth:`guess_params` and is clamped to bounds."""

    def test_p0_at_truth_yields_successful_fit(
        self, synthetic_extension: np.ndarray, synthetic_wlc_force: np.ndarray
    ) -> None:
        # Starting exactly at the truth: the fitter converges in zero
        # iterations and the result is still correct.
        engine = LmfitEngine(WLCModel())
        result = engine.fit(
            synthetic_extension,
            synthetic_wlc_force,
            x_range=X_RANGE,
            p0={"p": P_TRUTH, "L": L_TRUTH},
        )
        assert result.metadata["success"]
        assert abs(result.params["p"] - P_TRUTH) / P_TRUTH < 0.01
        assert abs(result.params["L"] - L_TRUTH) / L_TRUTH < 0.01

    def test_p0_overrides_guess_params(
        self, synthetic_extension: np.ndarray, synthetic_wlc_force: np.ndarray
    ) -> None:
        # The whole point of p0 is to override the model's
        # :meth:`guess_params` heuristic. With x_range=(20, 180),
        # guess_params would have returned ``L ≈ 180 * 1.1 = 198``;
        # passing p0={"L": 150.0} must change the actual starting
        # point handed to the solver.
        engine = LmfitEngine(WLCModel())
        result = engine.fit(
            synthetic_extension,
            synthetic_wlc_force,
            x_range=X_RANGE,
            p0={"p": 0.5, "L": 150.0},
        )
        init = result.metadata["initial_params"]
        assert init["p"] == 0.5
        assert init["L"] == 150.0

    def test_p0_in_same_basin_converges_to_truth(
        self, synthetic_extension: np.ndarray, synthetic_wlc_force: np.ndarray
    ) -> None:
        # A starting point that is wrong but inside the **same basin
        # of attraction** as the truth must still converge. We pick
        # p0 close enough to (0.4, 200) that the WLC chi-square
        # landscape slopes back to the truth from there.
        engine = LmfitEngine(WLCModel())
        result = engine.fit(
            synthetic_extension,
            synthetic_wlc_force,
            x_range=X_RANGE,
            p0={"p": 0.5, "L": 180.0},
        )
        assert result.metadata["success"]
        assert abs(result.params["p"] - P_TRUTH) / P_TRUTH < 0.01
        assert abs(result.params["L"] - L_TRUTH) / L_TRUTH < 0.01

    def test_p0_extra_keys_are_ignored(
        self, synthetic_extension: np.ndarray, synthetic_wlc_force: np.ndarray
    ) -> None:
        engine = LmfitEngine(WLCModel())
        # `foo` is not a WLC parameter — must be silently dropped.
        result = engine.fit(
            synthetic_extension,
            synthetic_wlc_force,
            x_range=X_RANGE,
            p0={"p": P_TRUTH, "L": L_TRUTH, "foo": 99.0},
        )
        assert result.metadata["success"]
        # The bogus key must not have leaked into the result's params.
        assert "foo" not in result.params


# -- Bounds clamping -----------------------------------------------------


class TestBoundsClamping:
    """Out-of-bounds p0 values are clamped before being passed to the solver."""

    def test_p0_above_upper_bound_is_clamped(
        self, synthetic_extension: np.ndarray, synthetic_wlc_force: np.ndarray
    ) -> None:
        engine = LmfitEngine(WLCModel())
        # WLC bounds: p ∈ (0.05, 5.0), L ∈ (10.0, 1000.0).
        result = engine.fit(
            synthetic_extension,
            synthetic_wlc_force,
            x_range=X_RANGE,
            p0={"p": 100.0, "L": 5000.0},
        )
        # The clamped starting point is what the solver actually saw.
        init = result.metadata["initial_params"]
        assert init["p"] == 5.0
        assert init["L"] == 1000.0

    def test_p0_below_lower_bound_is_clamped(
        self, synthetic_extension: np.ndarray, synthetic_wlc_force: np.ndarray
    ) -> None:
        engine = LmfitEngine(WLCModel())
        result = engine.fit(
            synthetic_extension,
            synthetic_wlc_force,
            x_range=X_RANGE,
            p0={"p": 0.001, "L": 0.1},
        )
        init = result.metadata["initial_params"]
        assert init["p"] == 0.05
        assert init["L"] == 10.0

    def test_p0_inside_bounds_is_unchanged(
        self, synthetic_extension: np.ndarray, synthetic_wlc_force: np.ndarray
    ) -> None:
        engine = LmfitEngine(WLCModel())
        result = engine.fit(
            synthetic_extension,
            synthetic_wlc_force,
            x_range=X_RANGE,
            p0={"p": 0.4, "L": 200.0},
        )
        init = result.metadata["initial_params"]
        assert init["p"] == 0.4
        assert init["L"] == 200.0


# -- Failed fit ----------------------------------------------------------


class _AlwaysNaNModel:
    """A :class:`PolymerModel` that always returns NaN.

    Used to deterministically drive the solver into a failed fit
    (lmfit's least-squares with ``nan_policy='raise'`` aborts as soon
    as the model output is NaN). Satisfies the structural
    :class:`~afmkit.models.base.PolymerModel` protocol.
    """

    param_names = ("p",)
    param_bounds = ((0.1, 10.0),)
    param_hints = {"p": "fake"}

    def __call__(self, x: np.ndarray, *, p: float) -> np.ndarray:
        return np.full(np.asarray(x).shape, np.nan)

    def guess_params(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        return {"p": 1.0}


class TestFailedFit:
    """A non-converging fit must yield a :class:`FitResult` with ``success=False``."""

    def test_broken_model_returns_failed_result(self) -> None:
        engine = LmfitEngine(_AlwaysNaNModel())  # type: ignore[arg-type]
        ext = np.linspace(1.0, 10.0, 50)
        y = np.zeros_like(ext)
        result = engine.fit(ext, y)
        assert isinstance(result, FitResult)
        assert result.metadata["success"] is False
        # The diagnostic is on `message` — caller can log it.
        assert "message" in result.metadata
        # The failed result still has the right shape — caller can
        # decide whether to drop the curve or retry without crashing.
        assert result.n_data == ext.size
        assert result.n_params == 1
        # And the model-eval arrays are NaN (the model never produced
        # a finite output for this fit attempt).
        assert np.all(np.isnan(result.y_fit))
        assert np.all(np.isnan(result.residual))


# -- Top-level fit() end-to-end ------------------------------------------


class TestFitEndToEnd:
    """The :func:`fit` helper takes a :class:`ForceCurve` and returns a FitResult."""

    def test_fit_via_force_curve(self, synthetic_force_curve: ForceCurve) -> None:
        result = fit(synthetic_force_curve, model="wlc", x_range=X_RANGE)
        assert isinstance(result, FitResult)
        assert result.metadata["success"]
        assert abs(result.params["p"] - P_TRUTH) / P_TRUTH < 0.01
        assert abs(result.params["L"] - L_TRUTH) / L_TRUTH < 0.01
        # The x_fit / y_fit / residual arrays should all be 1-D and
        # match the requested range.
        assert result.x_fit.ndim == 1
        assert result.y_fit.ndim == 1
        assert result.residual.ndim == 1
        assert result.x_fit.size == result.n_data
        assert result.y_fit.size == result.n_data
        assert result.residual.size == result.n_data

    def test_fit_unknown_model_raises(self, synthetic_force_curve: ForceCurve) -> None:
        # A typo in the model name is a programming error — must raise
        # so the caller catches it, not silently fit the wrong model.
        with pytest.raises(KeyError, match="unknown model"):
            fit(synthetic_force_curve, model="not_a_real_model")

    def test_fit_default_model_is_wlc(self, synthetic_force_curve: ForceCurve) -> None:
        # The convenience helper defaults to model="wlc"; the two
        # calls must produce equivalent (not necessarily identical —
        # lmfit may take different paths) recovered parameters.
        r_default = fit(synthetic_force_curve, x_range=X_RANGE)
        r_explicit = fit(synthetic_force_curve, model="wlc", x_range=X_RANGE)
        assert r_default.model_name == r_explicit.model_name
        assert abs(r_default.params["p"] - r_explicit.params["p"]) < 1e-6
        assert abs(r_default.params["L"] - r_explicit.params["L"]) < 1e-6


# -- residual() helper ---------------------------------------------------


class TestResidualHelper:
    """:meth:`LmfitEngine.residual` exposes ``y - model(x, **params)`` for power users."""

    def test_residual_is_zero_at_truth(
        self, synthetic_extension: np.ndarray, synthetic_wlc_force: np.ndarray
    ) -> None:
        engine = LmfitEngine(WLCModel())
        # Evaluating the residual at the truth parameters: every
        # point of `synthetic_wlc_force` was generated by exactly that
        # model, so the residual is numerically zero.
        residual = engine.residual(
            {"p": P_TRUTH, "L": L_TRUTH}, synthetic_extension, synthetic_wlc_force
        )
        np.testing.assert_allclose(residual, np.zeros_like(synthetic_extension), atol=1e-9)

    def test_residual_shape_matches_input(
        self, synthetic_extension: np.ndarray, synthetic_wlc_force: np.ndarray
    ) -> None:
        engine = LmfitEngine(WLCModel())
        residual = engine.residual({"p": 0.5, "L": 180.0}, synthetic_extension, synthetic_wlc_force)
        assert residual.shape == synthetic_extension.shape


# -- Input validation ----------------------------------------------------


class TestInputValidation:
    """The engine must reject malformed inputs at the boundary."""

    def test_non_finite_x_raises(
        self, synthetic_extension: np.ndarray, synthetic_wlc_force: np.ndarray
    ) -> None:
        engine = LmfitEngine(WLCModel())
        bad_x = synthetic_extension.copy()
        bad_x[10] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            engine.fit(bad_x, synthetic_wlc_force, x_range=X_RANGE)

    def test_mismatched_x_y_length_raises(self) -> None:
        engine = LmfitEngine(WLCModel())
        x = np.linspace(1.0, 100.0, 50)
        y = np.zeros(49)  # one short
        with pytest.raises(ValueError, match="same length"):
            engine.fit(x, y)

    def test_mismatched_weights_length_raises(
        self, synthetic_extension: np.ndarray, synthetic_wlc_force: np.ndarray
    ) -> None:
        engine = LmfitEngine(WLCModel())
        bad_weights = np.ones(synthetic_extension.size - 1)
        with pytest.raises(ValueError, match="weights must have the same length"):
            engine.fit(
                synthetic_extension,
                synthetic_wlc_force,
                x_range=X_RANGE,
                weights=bad_weights,
            )

    def test_multidimensional_x_raises(self) -> None:
        engine = LmfitEngine(WLCModel())
        x = np.zeros((10, 2))
        y = np.zeros((10, 2))
        with pytest.raises(ValueError, match="1-D"):
            engine.fit(x, y)
