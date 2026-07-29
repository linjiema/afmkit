"""Unit tests for :mod:`afmkit.fitting.report`.

The :class:`FitResult` is the single object every fit engine returns;
all downstream consumers (GUI, plot, exporter, batch) read it through
``summary()``, ``to_dict()`` and the ``r_squared`` property. These
tests pin down the contract for those three access paths plus the
``from_lmfit`` factory.

Test data is built by a real :class:`LmfitEngine` run on the conftest
noise-free WLC fixtures, so the tests verify the full
:mod:`afmkit.fitting` stack as a black box.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import lmfit
import numpy as np
import pytest

from afmkit.fitting import LmfitEngine
from afmkit.fitting.report import FitResult
from afmkit.models import WLCModel

P_TRUTH: float = 0.4
L_TRUTH: float = 200.0
X_RANGE: tuple[float, float] = (20.0, 180.0)


# -- Fixtures ------------------------------------------------------------


@pytest.fixture
def good_fit(synthetic_extension: np.ndarray, synthetic_wlc_force: np.ndarray) -> FitResult:
    """A FitResult from a successful, noise-free WLC fit.

    Used as the basis for every test in this module. The fit recovers
    the truth to ~1e-6, so the test assertions can rely on
    ``r_squared == 1.0`` and finite stderrs.
    """
    engine = LmfitEngine(WLCModel())
    return engine.fit(synthetic_extension, synthetic_wlc_force, x_range=X_RANGE)


# -- summary() -----------------------------------------------------------


class TestSummary:
    """``summary()`` is the human-readable rendering of a FitResult."""

    def test_summary_is_non_empty(self, good_fit: FitResult) -> None:
        s = good_fit.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_summary_contains_model_name(self, good_fit: FitResult) -> None:
        assert "WLCModel" in good_fit.summary()

    def test_summary_lists_every_parameter_with_stderr(self, good_fit: FitResult) -> None:
        s = good_fit.summary()
        for name in good_fit.params:
            # Parameter name must appear, and its line must include a
            # value plus a stderr (``+/-`` or ``n/a`` for fixed / n/a).
            assert name in s, f"parameter {name!r} missing from summary"
        for line in s.splitlines():
            if "=" not in line:
                continue
            assert "+/-" in line or "n/a" in line, f"parameter line has no stderr marker: {line!r}"

    def test_summary_status_is_ok_for_successful_fit(self, good_fit: FitResult) -> None:
        assert "ok" in good_fit.summary()
        assert "FAILED" not in good_fit.summary()

    def test_summary_for_failed_fit_includes_message(self) -> None:
        # A hand-constructed failed FitResult: status block should
        # surface both "FAILED" and the diagnostic message.
        nan = float("nan")
        r = FitResult(
            model_name="wlc",
            params={"p": 1.0, "L": 100.0},
            stderr={"p": nan, "L": nan},
            covariance=None,
            chi_square=nan,
            reduced_chi_square=nan,
            n_data=10,
            n_params=2,
            aic=nan,
            bic=nan,
            residual=np.zeros(10),
            x_fit=np.arange(10, dtype=np.float64),
            y_fit=np.zeros(10),
            metadata={"success": False, "message": "solver exploded"},
        )
        s = r.summary()
        assert "FAILED" in s
        assert "solver exploded" in s


# -- to_dict / from_dict round-trip --------------------------------------


class TestSerializationRoundTrip:
    """``to_dict`` → ``from_dict`` must preserve every public field."""

    def test_to_dict_has_all_keys(self, good_fit: FitResult) -> None:
        d = good_fit.to_dict()
        for key in (
            "schema_version",
            "model_name",
            "params",
            "stderr",
            "covariance",
            "chi_square",
            "reduced_chi_square",
            "n_data",
            "n_params",
            "aic",
            "bic",
            "residual",
            "x_fit",
            "y_fit",
            "metadata",
        ):
            assert key in d, f"to_dict() missing key: {key!r}"

    def test_round_trip_preserves_scalar_fields(self, good_fit: FitResult) -> None:
        d = good_fit.to_dict()
        restored = FitResult.from_dict(d)
        assert restored.model_name == good_fit.model_name
        assert restored.params == good_fit.params
        assert restored.stderr == good_fit.stderr
        assert restored.n_data == good_fit.n_data
        assert restored.n_params == good_fit.n_params
        assert restored.chi_square == good_fit.chi_square
        assert restored.reduced_chi_square == good_fit.reduced_chi_square
        assert restored.aic == good_fit.aic
        assert restored.bic == good_fit.bic
        assert restored.metadata == good_fit.metadata

    def test_round_trip_preserves_arrays(self, good_fit: FitResult) -> None:
        d = good_fit.to_dict()
        restored = FitResult.from_dict(d)
        np.testing.assert_array_equal(restored.residual, good_fit.residual)
        np.testing.assert_array_equal(restored.x_fit, good_fit.x_fit)
        np.testing.assert_array_equal(restored.y_fit, good_fit.y_fit)

    def test_round_trip_preserves_covariance(self, good_fit: FitResult) -> None:
        d = good_fit.to_dict()
        restored = FitResult.from_dict(d)
        if good_fit.covariance is None:
            assert restored.covariance is None
        else:
            assert restored.covariance is not None
            np.testing.assert_allclose(restored.covariance, good_fit.covariance)

    def test_to_json_is_valid_json(self, good_fit: FitResult) -> None:
        s = good_fit.to_json()
        # Round-trip through json to make sure the on-the-wire form
        # is parseable — ``to_json`` is what callers hit when persisting
        # results into a notebook or HTTP body.
        d = json.loads(s)
        assert d["model_name"] == good_fit.model_name
        assert d["n_data"] == good_fit.n_data

    def test_from_dict_rejects_newer_schema(self) -> None:
        # A schema_version above what this build knows must be
        # rejected explicitly so callers can be told to upgrade.
        with pytest.raises(ValueError, match="schema version"):
            FitResult.from_dict(
                {
                    "schema_version": FitResult.SCHEMA_VERSION + 1000,
                    "model_name": "wlc",
                    "params": {},
                    "stderr": {},
                    "covariance": None,
                    "chi_square": 0.0,
                    "reduced_chi_square": 0.0,
                    "n_data": 0,
                    "n_params": 0,
                    "aic": 0.0,
                    "bic": 0.0,
                    "residual": [],
                    "x_fit": [],
                    "y_fit": [],
                    "metadata": {},
                }
            )


# -- r_squared -----------------------------------------------------------


class TestRSquared:
    """``r_squared`` is the primary goodness-of-fit number surfaced to users."""

    def test_r_squared_is_one_for_perfect_fit(self, good_fit: FitResult) -> None:
        # The fixture is a noise-free WLC fit, so the recovered model
        # matches the data to numerical precision → r_squared = 1.0.
        assert good_fit.r_squared == pytest.approx(1.0, abs=1e-6)

    def test_r_squared_in_unit_interval_for_perfect_fit(self, good_fit: FitResult) -> None:
        # For a non-pathological fit, R² must lie in [0, 1].
        assert 0.0 <= good_fit.r_squared <= 1.0

    def test_r_squared_nan_when_data_is_constant(self) -> None:
        # SS_tot = 0 when every y_data is identical — R² is undefined
        # in that degenerate case, and we return NaN rather than 0 or
        # a div-by-zero.
        nan = float("nan")
        r = FitResult(
            model_name="const",
            params={"a": 5.0},
            stderr={"a": nan},
            covariance=None,
            chi_square=0.0,
            reduced_chi_square=0.0,
            n_data=5,
            n_params=1,
            aic=0.0,
            bic=0.0,
            residual=np.zeros(5),
            x_fit=np.arange(5, dtype=np.float64),
            y_fit=np.full(5, 5.0),  # y_data = y_fit + residual = all 5.0
            metadata={},
        )
        assert math.isnan(r.r_squared)

    def test_r_squared_is_finite_for_real_fit(self, good_fit: FitResult) -> None:
        # Sanity: the helper must never return inf on a normal fit.
        assert math.isfinite(good_fit.r_squared)


# -- from_lmfit factory --------------------------------------------------


class TestFromLmfit:
    """``from_lmfit`` is the boundary between lmfit and the FitResult dataclass."""

    def test_from_lmfit_produces_valid_fit_result(
        self, synthetic_extension: np.ndarray, synthetic_wlc_force: np.ndarray
    ) -> None:
        # Build a real ModelResult by running the engine's underlying
        # lmfit Model directly — this exercises the from_lmfit path
        # with non-trivial parameter recovery.
        engine = LmfitEngine(WLCModel())
        params = engine.lm_model.make_params()
        params["p"].set(value=0.4, min=0.05, max=5.0)
        params["L"].set(value=200.0, min=10.0, max=1000.0)
        lm_result = engine.lm_model.fit(
            synthetic_wlc_force, params, x=synthetic_extension, method="leastsq"
        )

        fr = FitResult.from_lmfit(
            model_name="WLCModel",
            result=lm_result,
            x=synthetic_extension,
            y=synthetic_wlc_force,
        )

        assert isinstance(fr, FitResult)
        # Param dict has the model's parameters in canonical order.
        assert set(fr.params.keys()) == set(WLCModel.param_names)
        # stderr dict shape matches params dict shape — fixed / no-cov
        # entries are NaN rather than missing.
        assert set(fr.stderr.keys()) == set(fr.params.keys())
        # The standard statistics come through with finite values.
        assert fr.n_data == synthetic_extension.size
        assert fr.n_params == len(WLCModel.param_names)
        assert math.isfinite(fr.chi_square)
        # Metadata is populated with the lmfit diagnostic bag.
        assert "success" in fr.metadata
        assert "message" in fr.metadata

    def test_from_lmfit_with_none_covariance(self) -> None:
        # lmfit sets `result.covar` to None when the covariance could
        # not be estimated (e.g. a non-converging fit). from_lmfit
        # must propagate None rather than synthesise a zero matrix.
        #
        # We construct a synthetic ModelResult with covar=None
        # because scipy.optimize.leastsq rejects under-determined
        # problems at the input-validation stage, so a real fit
        # can't actually reach the "no covariance" state.
        params = lmfit.Parameters()
        params.add("a", value=1.0)
        params.add("b", value=2.0)
        # Make every parameter fixed so result.covar is None.
        for p in params.values():
            p.vary = False
        # Best_values for all-fixed params is just the initial values.
        best = {"a": 1.0, "b": 2.0}
        x = np.arange(10, dtype=np.float64)
        y = x.astype(np.float64) * 1.0 + 2.0
        result = SimpleNamespace(
            best_values=best,
            best_fit=y,
            residual=np.zeros_like(y),
            params=params,
            covar=None,
            success=True,
            message="all parameters fixed",
            method="leastsq",
            chisqr=0.0,
            redchi=0.0,
            ndata=10,
            nvarys=0,
            aic=float("nan"),
            bic=float("nan"),
            rsquared=1.0,
        )
        fr = FitResult.from_lmfit(model_name="dummy", result=result, x=x, y=y)
        # Under-determined system → covariance is None.
        assert fr.covariance is None
        # Stderr entries are NaN when the covariance is unavailable
        # (or for fixed parameters — both are reported as NaN).
        assert set(fr.stderr.keys()) == {"a", "b"}
        for v in fr.stderr.values():
            assert math.isnan(v)

    def test_from_lmfit_includes_fixed_parameters(self) -> None:
        # Fixed parameters (vary=False) don't appear in `best_values`
        # but must still be reported in the result's `params` dict
        # with NaN stderr — otherwise downstream code can't tell what
        # the locked value was.
        params = lmfit.Parameters()
        params.add("a", value=1.0)
        params.add("b", value=2.0, vary=False)  # fixed
        model = lmfit.Model(
            lambda x, a, b: np.asarray(x, dtype=float) * a + b,
            independent_vars=["x"],
        )
        x = np.arange(10, dtype=np.float64)
        y = x.astype(np.float64) * 1.0 + 2.0  # matches a=1, b=2 exactly
        lm_result = model.fit(y, params, x=x, method="leastsq")
        fr = FitResult.from_lmfit(model_name="dummy", result=lm_result, x=x, y=y)
        # Both parameters surface in `params`, with `b` carrying the
        # fixed value.
        assert fr.params["a"] == pytest.approx(1.0, abs=1e-6)
        assert fr.params["b"] == 2.0
        # Fixed params get NaN stderr — they're not part of the fit.
        assert math.isnan(fr.stderr["b"])
        # The varying param `a` does have a stderr.
        assert math.isfinite(fr.stderr["a"])


# -- __repr__ smoke test -------------------------------------------------


class TestRepr:
    def test_repr_mentions_model_and_stats(self, good_fit: FitResult) -> None:
        s = repr(good_fit)
        assert "WLCModel" in s
        assert "n_data" in s
        assert "R^2" in s
