"""Unit tests for :mod:`afmkit.models.wlc` and :mod:`afmkit.models.base`.

The WLC tests in particular form a **numerical-correctness contract**:
the new Python implementation must match the original Igor
``LVFitWLC`` routine to floating-point precision, because the lab's
historical analyses were processed through that Igor code. Any
discrepancy (even a few ulp) would invalidate longitudinal
comparisons with prior datasets.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from afmkit.models.base import PolymerModel
from afmkit.models.wlc import WLCModel

# -- Reference parameters -------------------------------------------------
# A representative (p, L) pair from a typical protein-unfolding curve in
# the lab. Pulled from the same values used in the synthetic test
# fixtures (see conftest.py), so the test set is internally consistent.

P_DEFAULT: float = 0.4
L_DEFAULT: float = 200.0
KBT_PNNM: float = 4.1  # the hardcoded thermal energy, in pN·nm


def _wlc_formula_igor(x: np.ndarray, p: float, L: float) -> np.ndarray:
    """The Igor `LVFitWLC` formula, written out longhand.

    Kept as a **separate** function so that the test for the
    :class:`WLCModel` class is genuinely testing the class — any
    transcription error in either place would be caught by a mismatch.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        reduced = x_arr / L
        return (KBT_PNNM / p) * (0.25 * (1.0 - reduced) ** -2 - 0.25 + reduced)


# -- Model construction & metadata ---------------------------------------


class TestWLCModelMetadata:
    """The class-level metadata that the fitter and GUI rely on."""

    def test_param_names(self) -> None:
        assert WLCModel.param_names == ("p", "L")

    def test_param_bounds_physical(self) -> None:
        # Bounds must be a 2-tuple of (lo, hi) pairs, one per param.
        assert len(WLCModel.param_bounds) == 2
        for lo, hi in WLCModel.param_bounds:
            assert lo > 0.0  # all physical parameters are strictly positive
            assert hi > lo  # well-formed range
        p_bounds, L_bounds = WLCModel.param_bounds
        # Persistence length: 0.05–5 nm covers the biopolymer range.
        assert p_bounds == (0.05, 5.0)
        # Contour length: 10–1000 nm covers the lab's typical constructs.
        assert L_bounds == (10.0, 1000.0)

    def test_param_hints_complete(self) -> None:
        # Every parameter must have a human-readable hint, otherwise the
        # GUI will render "(no hint)" placeholders.
        assert set(WLCModel.param_hints.keys()) == set(WLCModel.param_names)
        for name, hint in WLCModel.param_hints.items():
            assert isinstance(name, str)
            assert isinstance(hint, str)
            assert len(hint) > 0

    def test_param_names_and_bounds_aligned(self) -> None:
        # Catching a misaligned (param_names, param_bounds) pair early
        # is critical — the fitter zips them by index, so a mismatch
        # would silently mislabel the bounds.
        assert len(WLCModel.param_names) == len(WLCModel.param_bounds)

    def test_is_frozen_dataclass(self) -> None:
        # The model is mathematical — no per-instance state. A frozen
        # dataclass enforces that at runtime (setattr raises FrozenInstanceError).
        assert dataclasses.is_dataclass(WLCModel)
        assert dataclasses.fields(WLCModel) == ()  # no instance fields
        model = WLCModel()
        with pytest.raises(dataclasses.FrozenInstanceError):
            model.param_names = ("x",)  # type: ignore[misc]

    def test_construction_takes_no_args(self) -> None:
        # Pure dataclass with no instance fields — WLCModel() must work.
        model = WLCModel()
        assert isinstance(model, WLCModel)

    def test_repr_does_not_crash(self) -> None:
        # The default dataclass repr must be at least stringifiable.
        # (We don't pin the exact format — dataclass may evolve it.)
        assert "WLCModel" in repr(WLCModel())


# -- Numerical correctness (the bit-exact contract) ----------------------


class TestWLCModelNumericalCorrectness:
    """Tests that the Python implementation matches the Igor formula."""

    def test_at_x_zero_force_is_zero(self) -> None:
        # F(0) = (4.1/p) * [0.25 * 1 - 0.25 + 0] = 0
        model = WLCModel()
        F = model(np.array([0.0]), p=P_DEFAULT, L=L_DEFAULT)
        assert F.shape == (1,)
        assert F[0] == 0.0  # bit-exact, not just "close"

    def test_at_x_zero_via_call_returns_finite(self) -> None:
        # The test above checks the formula; this one checks the limit
        # value is exposed correctly through `np.asarray` coercion.
        model = WLCModel()
        F = model([0.0], p=P_DEFAULT, L=L_DEFAULT)  # list input
        assert F[0] == 0.0

    def test_at_half_contour_force_is_analytic(self) -> None:
        # F(L/2) = (4.1/p) * [0.25 * 4 - 0.25 + 0.5] = (4.1/p) * 1.25
        model = WLCModel()
        x = L_DEFAULT / 2.0
        expected = (KBT_PNNM / P_DEFAULT) * 1.25
        F = model(np.array([x]), p=P_DEFAULT, L=L_DEFAULT)
        assert F[0] == pytest.approx(expected, rel=1e-12)

    def test_at_quarter_contour_matches_direct_formula(self) -> None:
        # F(L/4) = (4.1/p) * [0.25 * (3/4)^(-2) - 0.25 + 0.25]
        #        = (4.1/p) * (0.25 * 16/9)
        #        = (4.1/p) * (4/9)
        model = WLCModel()
        x = L_DEFAULT / 4.0
        expected = (KBT_PNNM / P_DEFAULT) * (4.0 / 9.0)
        F = model(np.array([x]), p=P_DEFAULT, L=L_DEFAULT)
        assert F[0] == pytest.approx(expected, rel=1e-12)

    def test_bit_exact_match_with_igor_formula_random_grid(self) -> None:
        # The headline test: across a random grid of (p, L, x), the
        # model output and the hand-written formula must agree to
        # floating-point precision. We use `assert_array_equal`
        # (bit-exact), not `assert_allclose`, because that is the
        # contract with the Igor code.
        rng = np.random.default_rng(seed=20240729)
        p_vals = rng.uniform(0.1, 2.0, size=20)
        L_vals = rng.uniform(50.0, 500.0, size=20)
        x_grids = [rng.uniform(0.0, L - 1.0, size=200) for L in L_vals]

        model = WLCModel()
        for p, L, x in zip(p_vals, L_vals, x_grids, strict=True):
            F_model = model(x, p=p, L=L)
            F_ref = _wlc_formula_igor(x, p=p, L=L)
            np.testing.assert_array_equal(F_model, F_ref)

    def test_matches_inline_formula_dense_grid(self) -> None:
        # Same check on a dense, evenly-spaced grid — the typical
        # fitter input. Catches any subtle numpy broadcasting bug.
        model = WLCModel()
        x = np.linspace(0.5, L_DEFAULT - 0.5, 1000)
        F_model = model(x, p=P_DEFAULT, L=L_DEFAULT)
        F_ref = _wlc_formula_igor(x, p=P_DEFAULT, L=L_DEFAULT)
        np.testing.assert_array_equal(F_model, F_ref)

    def test_output_shape_matches_input(self) -> None:
        model = WLCModel()
        for n in (1, 5, 100, 1000):
            x = np.linspace(0.1, L_DEFAULT - 0.1, n)
            F = model(x, p=P_DEFAULT, L=L_DEFAULT)
            assert F.shape == (n,)

    def test_output_dtype_is_float(self) -> None:
        model = WLCModel()
        F = model(np.array([1.0, 2.0]), p=P_DEFAULT, L=L_DEFAULT)
        assert np.issubdtype(F.dtype, np.floating)

    def test_list_input_is_accepted(self) -> None:
        # Real callers often pass Python lists; the model must accept them
        # and return a numpy array.
        model = WLCModel()
        F = model([1.0, 2.0, 3.0], p=P_DEFAULT, L=L_DEFAULT)
        assert isinstance(F, np.ndarray)
        assert F.shape == (3,)


# -- Physics invariants --------------------------------------------------


class TestWLCModelPhysics:
    """The WLC is a monotonically increasing function on (0, L)."""

    def test_strictly_increasing_on_open_interval(self) -> None:
        model = WLCModel()
        # Stay strictly inside (0, L) — the endpoints have trivial
        # behaviour that we test elsewhere.
        x = np.linspace(0.1, L_DEFAULT - 0.1, 5000)
        F = model(x, p=P_DEFAULT, L=L_DEFAULT)
        diffs = np.diff(F)
        # Strictly positive on a dense grid (no ties within float
        # round-off at these spacings).
        assert np.all(diffs > 0)

    def test_increasing_across_realistic_force_range(self) -> None:
        # Same invariant for a few representative (p, L) pairs.
        model = WLCModel()
        for p, L in ((0.4, 200.0), (0.7, 150.0), (1.5, 50.0), (2.0, 100.0)):
            x = np.linspace(0.5, L - 0.5, 500)
            F = model(x, p=p, L=L)
            assert np.all(np.diff(F) > 0), f"WLC not monotone for p={p}, L={L}"

    def test_zero_force_only_at_zero_extension(self) -> None:
        # F(x) = 0 iff x = 0 (within float precision, in the physical
        # domain). The check at x = 0 is already in
        # `test_at_x_zero_force_is_zero`; here we check strictly
        # positive forces everywhere else.
        model = WLCModel()
        x = np.linspace(0.01, L_DEFAULT - 0.01, 1000)
        F = model(x, p=P_DEFAULT, L=L_DEFAULT)
        assert np.all(F > 0.0)


# -- Singularity handling -----------------------------------------------


class TestWLCModelSingularity:
    """At x = L the WLC formula diverges. We return +inf, not an error."""

    def test_at_x_equals_L_returns_inf(self) -> None:
        model = WLCModel()
        F = model(np.array([L_DEFAULT]), p=P_DEFAULT, L=L_DEFAULT)
        assert F.shape == (1,)
        assert np.isinf(F[0])
        assert F[0] > 0  # +inf, not -inf

    def test_singularity_does_not_raise(self) -> None:
        # The whole point: the fitter will hand us x = L on a few
        # iterations. Raising would kill the fit.
        model = WLCModel()
        # No `pytest.raises` — the call must complete normally.
        result = model(np.array([L_DEFAULT]), p=P_DEFAULT, L=L_DEFAULT)
        assert np.isposinf(result[0])

    def test_near_singularity_grows_without_bound(self) -> None:
        # Approaching x = L from below, F must blow up. We check the
        # sequence F(x_n) is increasing as x_n → L.
        model = WLCModel()
        # Geometric approach to L: 0.999 L, 0.9999 L, 0.99999 L.
        x = np.array([0.999, 0.9999, 0.99999]) * L_DEFAULT
        F = model(x, p=P_DEFAULT, L=L_DEFAULT)
        assert F[0] < F[1] < F[2]
        # And the last value is already large enough to be
        # "approaching infinity" in any practical sense.
        assert F[-1] > 1e5

    def test_array_containing_singularity_returns_inf_at_singularity(self) -> None:
        # When the input array spans x = L, only the entry at x = L
        # is inf; the rest are finite.
        model = WLCModel()
        x = np.array([10.0, 100.0, L_DEFAULT, 10.0, 100.0])
        F = model(x, p=P_DEFAULT, L=L_DEFAULT)
        assert np.isfinite(F[0])
        assert np.isfinite(F[1])
        assert np.isposinf(F[2])
        assert np.isfinite(F[3])
        assert np.isfinite(F[4])

    def test_no_runtime_warning_at_singularity(self) -> None:
        # The model is supposed to suppress the divide-by-zero warning
        # at x = L. If numpy starts leaking warnings, downstream
        # loguru capture will start spamming — catch it here.
        model = WLCModel()
        with np.errstate(divide="raise", invalid="raise"):
            # The internal `errstate` should mask the singularity.
            F = model(np.array([L_DEFAULT]), p=P_DEFAULT, L=L_DEFAULT)
        assert np.isposinf(F[0])


# -- Parametrisation & fitting support ----------------------------------


class TestWLCModelGuessParams:
    """The starting-point heuristic for the fitter."""

    def test_returns_dict_with_required_keys(self) -> None:
        model = WLCModel()
        x = np.linspace(1.0, L_DEFAULT, 100)
        y = np.linspace(0.0, 50.0, 100)
        guess = model.guess_params(x, y)
        assert isinstance(guess, dict)
        assert set(guess.keys()) == set(model.param_names)

    def test_returns_finite_values(self) -> None:
        model = WLCModel()
        for x_max in (10.0, 50.0, 200.0, 800.0):
            x = np.linspace(1.0, x_max, 100)
            y = model(x, p=P_DEFAULT, L=x_max * 1.1)  # any y is fine
            guess = model.guess_params(x, y)
            for k, v in guess.items():
                assert math.isfinite(v), f"guess_params returned non-finite {k}={v}"

    def test_l_guess_exceeds_x_max(self) -> None:
        # The WLC diverges at x = L, so L must be > every observed x.
        model = WLCModel()
        x = np.linspace(0.5, 150.0, 200)
        y = np.zeros_like(x)
        guess = model.guess_params(x, y)
        assert guess["L"] > x.max()

    def test_guess_respects_param_bounds(self) -> None:
        # The starting point should lie inside the search bounds —
        # otherwise the fitter has to immediately snap back to the
        # boundary, defeating the purpose of a "guess".
        model = WLCModel()
        for x_max in (1.0, 5.0, 15.0, 200.0, 999.0):
            x = np.linspace(0.1, x_max, 50)
            y = np.zeros_like(x)
            guess = model.guess_params(x, y)
            for name, (lo, hi), value in zip(
                model.param_names, model.param_bounds, guess.values(), strict=True
            ):
                assert (
                    lo <= value <= hi
                ), f"guess_params({name}={value}) out of bounds [{lo}, {hi}] for x_max={x_max}"

    def test_p_guess_is_protein_default(self) -> None:
        # The spec fixes the p default at 0.4 nm — a common value for
        # unfolded proteins. Changing this default would invalidate
        # every "auto-fit" comparison against prior batches.
        model = WLCModel()
        x = np.linspace(1.0, L_DEFAULT, 100)
        y = np.zeros_like(x)
        guess = model.guess_params(x, y)
        assert guess["p"] == 0.4

    def test_guess_ignores_y_argument(self) -> None:
        # The current heuristic doesn't use y — the spec accepts that
        # (the Protocol requires the signature for API symmetry with
        # other models that may need it). Verify the function is at
        # least y-tolerant and doesn't crash on a different y.
        model = WLCModel()
        x = np.linspace(1.0, 100.0, 50)
        guess1 = model.guess_params(x, np.zeros_like(x))
        guess2 = model.guess_params(x, np.ones_like(x) * 1e6)
        assert guess1 == guess2

    def test_guess_on_short_curve_clamps_l(self) -> None:
        # When x.max() * 1.1 falls below the L lower bound (10 nm), the
        # guess must clamp to the lower bound — never return a value
        # outside `param_bounds`.
        model = WLCModel()
        x = np.linspace(0.1, 5.0, 50)  # x.max() * 1.1 = 5.5 < 10
        y = np.zeros_like(x)
        guess = model.guess_params(x, y)
        lo, _ = model.param_bounds[1]
        assert guess["L"] >= lo


# -- Protocol conformance -----------------------------------------------


class TestPolymerModelProtocol:
    """The model must satisfy the :class:`PolymerModel` protocol."""

    def test_satisfies_protocol(self) -> None:
        # With `@runtime_checkable`, isinstance() works as a structural
        # check on the required attributes.
        model = WLCModel()
        assert isinstance(model, PolymerModel)

    def test_exposes_required_attributes(self) -> None:
        # Belt-and-suspenders: check each Protocol attribute by name
        # so a future rename in either place fails loudly.
        model = WLCModel()
        assert hasattr(model, "param_names")
        assert hasattr(model, "param_bounds")
        assert hasattr(model, "param_hints")
        assert callable(model)
        assert callable(model.guess_params)

    def test_protocol_attribute_types(self) -> None:
        # The metadata attributes have specific shapes — catch drift
        # in the next refactor.
        model = WLCModel()
        assert isinstance(model.param_names, tuple)
        assert all(isinstance(n, str) for n in model.param_names)
        assert isinstance(model.param_bounds, tuple)
        assert all(isinstance(b, tuple) and len(b) == 2 for b in model.param_bounds)
        assert isinstance(model.param_hints, dict)
        assert all(isinstance(v, str) for v in model.param_hints.values())


# -- Behaviour under parameter variation --------------------------------


class TestWLCModelParametrisation:
    """Sanity checks on how the model responds to p and L."""

    def test_smaller_p_gives_larger_force(self) -> None:
        # At fixed x and L, F ∝ 1/p — a stiffer chain (larger p)
        # produces a smaller force at the same extension.
        model = WLCModel()
        x = np.array([L_DEFAULT / 2.0])
        F_stiff = model(x, p=1.0, L=L_DEFAULT)
        F_soft = model(x, p=0.2, L=L_DEFAULT)
        assert F_soft[0] > F_stiff[0]

    def test_force_at_half_contour_scales_inversely_with_p(self) -> None:
        # F(L/2) = (4.1/p) * 1.25 — the 1.25 factor is constant, only
        # the prefactor changes. So F(L/2; p1) / F(L/2; p2) = p2 / p1.
        model = WLCModel()
        x = L_DEFAULT / 2.0
        F1 = model(np.array([x]), p=0.4, L=L_DEFAULT)[0]
        F2 = model(np.array([x]), p=0.8, L=L_DEFAULT)[0]
        assert F1 / F2 == pytest.approx(0.8 / 0.4)

    def test_scaling_in_p_is_inverse_proportional(self) -> None:
        # More general: F(x; alpha*p) = F(x; p) / alpha for any x < L.
        # Check across the full curve (excluding the singularity).
        model = WLCModel()
        x = np.linspace(1.0, L_DEFAULT - 1.0, 50)
        alpha = 2.5
        F_p = model(x, p=P_DEFAULT, L=L_DEFAULT)
        F_alpha_p = model(x, p=P_DEFAULT * alpha, L=L_DEFAULT)
        np.testing.assert_allclose(F_p, alpha * F_alpha_p, rtol=1e-12)

    def test_scaling_in_L_homogeneous(self) -> None:
        # The WLC is scale-invariant: rescaling both x and L by the
        # same factor leaves F unchanged. (Mathematical property of
        # the formula, sanity check that we haven't lost a factor.)
        model = WLCModel()
        x = np.linspace(0.1, 0.9, 50)
        scale = 3.7
        F1 = model(x * L_DEFAULT, p=P_DEFAULT, L=L_DEFAULT)
        F2 = model(x * L_DEFAULT * scale, p=P_DEFAULT, L=L_DEFAULT * scale)
        np.testing.assert_allclose(F1, F2, rtol=1e-12)
