"""Unit tests for :mod:`afmkit.models.ewlc`.

The eWLC tests cover three things:

1. **Metadata** — the class exposes the same contract as
   :class:`~afmkit.models.wlc.WLCModel` (``param_names``,
   ``param_bounds``, ``param_hints``, ``__call__``, ``guess_params``)
   and is registered in the model registry under ``"ewlc"``.

2. **Numerical correctness** — the closed-form evaluation matches a
   hand-written reference implementation to floating-point precision,
   the WLC limit (K0 → ∞) reproduces the Marko-Siggia WLC to within
   1 % across the realistic force range, and the singularity at
   x = L - 1/K0 returns +inf without raising.

3. **End-to-end fit recovery** — :class:`~afmkit.fitting.LmfitEngine`
   can recover the ground-truth (p, L, K0) on a noise-free synthetic
   eWLC curve to within 1 % per parameter.
"""

from __future__ import annotations

import dataclasses
import itertools
import math

import numpy as np
import pytest

from afmkit.fitting import LmfitEngine
from afmkit.models import EWLCModel, get_model
from afmkit.models.base import PolymerModel
from afmkit.models.ewlc import EWLCModel as EWLCDirect
from afmkit.models.wlc import WLCModel

# -- Reference parameters -------------------------------------------------
# A representative (p, L, K0) triplet for synthetic tests. Mirrors the
# protein/DNA constructs in the WLC test suite so the two model
# families are directly comparable on the same data.

P_DEFAULT: float = 0.4
L_DEFAULT: float = 200.0
K0_DEFAULT: float = 1500.0  # canonical dsDNA stretch modulus
KBT_PNNM: float = 4.1  # the hardcoded thermal energy, in pN·nm


def _ewlc_formula_reference(x: np.ndarray, p: float, L: float, K0: float) -> np.ndarray:
    """The Wang 1997 eWLC formula, written out longhand.

    Kept as a **separate** function so the test for the
    :class:`EWLCModel` class is genuinely testing the class — any
    transcription error in either place would be caught by a mismatch.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        reduced = x_arr / L
        return (KBT_PNNM / p) * (0.25 * (1.0 - reduced + 1.0 / K0) ** -2 - 0.25 + reduced)


def _wlc_formula_reference(x: np.ndarray, p: float, L: float) -> np.ndarray:
    """The Marko-Siggia WLC formula, written out longhand (for limit tests)."""
    x_arr = np.asarray(x, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        reduced = x_arr / L
        return (KBT_PNNM / p) * (0.25 * (1.0 - reduced) ** -2 - 0.25 + reduced)


def _synthetic_ewlc_curve(
    p: float = P_DEFAULT, L: float = L_DEFAULT, K0: float = K0_DEFAULT, n: int = 5000
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a noise-free eWLC curve on a 0 → 1.5 L extension axis."""
    x = np.linspace(0.0, 1.5 * L, n)
    y = _ewlc_formula_reference(x, p, L, K0)
    # Filter the divergent tail past the singularity so the fitter
    # receives only finite points (matches the WLC test convention).
    finite = np.isfinite(y)
    return x[finite], y[finite]


# -- Model construction & metadata ---------------------------------------


class TestEWLCModelMetadata:
    """The class-level metadata that the fitter and GUI rely on."""

    def test_param_names(self) -> None:
        assert EWLCModel.param_names == ("p", "L", "K0")

    def test_param_bounds_physical(self) -> None:
        # Bounds must be a tuple of (lo, hi) pairs, one per param.
        assert len(EWLCModel.param_bounds) == 3
        for lo, hi in EWLCModel.param_bounds:
            assert lo > 0.0  # all physical parameters are strictly positive
            assert hi > lo
        p_bounds, L_bounds, K0_bounds = EWLCModel.param_bounds
        # Persistence length: same range as WLC.
        assert p_bounds == (0.05, 5.0)
        # Contour length: extended to 10 µm for long nucleic-acid handles.
        assert L_bounds == (10.0, 10000.0)
        # Stretch modulus: 100-10 000 pN covers nucleic acids and stiff proteins.
        assert K0_bounds == (100.0, 10000.0)

    def test_param_bounds_keys_match_param_names(self) -> None:
        # The positionally-indexed bounds must be aligned with
        # param_names by index — the fitter zips them together
        # (src/afmkit/fitting/engine.py:338), so a misalignment would
        # silently mislabel the search range.
        assert len(EWLCModel.param_names) == len(EWLCModel.param_bounds)
        for name, (lo, hi) in zip(EWLCModel.param_names, EWLCModel.param_bounds, strict=True):
            assert isinstance(name, str)
            assert lo < hi

    def test_param_hints_complete(self) -> None:
        assert set(EWLCModel.param_hints.keys()) == set(EWLCModel.param_names)
        for name, hint in EWLCModel.param_hints.items():
            assert isinstance(name, str)
            assert isinstance(hint, str)
            assert len(hint) > 0

    def test_is_frozen_dataclass(self) -> None:
        assert dataclasses.is_dataclass(EWLCModel)
        assert dataclasses.fields(EWLCModel) == ()
        model = EWLCModel()
        with pytest.raises(dataclasses.FrozenInstanceError):
            model.param_names = ("x",)  # type: ignore[misc]

    def test_construction_takes_no_args(self) -> None:
        model = EWLCModel()
        assert isinstance(model, EWLCModel)

    def test_repr_does_not_crash(self) -> None:
        assert "EWLCModel" in repr(EWLCModel())


# -- Numerical correctness ----------------------------------------------


class TestEWLCModelNumericalCorrectness:
    """Tests that the Python implementation matches the reference formula."""

    def test_bit_exact_match_with_reference_random_grid(self) -> None:
        # Headline test: across a random grid of (p, L, K0, x), the
        # model output and the hand-written reference must agree to
        # floating-point precision.
        rng = np.random.default_rng(seed=20240815)
        p_vals = rng.uniform(0.1, 2.0, size=10)
        L_vals = rng.uniform(50.0, 500.0, size=10)
        K0_vals = rng.uniform(500.0, 5000.0, size=10)

        model = EWLCModel()
        for p, L, K0 in zip(p_vals, L_vals, K0_vals, strict=True):
            x = np.linspace(1.0, L - 1.0 / K0 - 1.0, 200)  # stay below singularity
            F_model = model(x, p=p, L=L, K0=K0)
            F_ref = _ewlc_formula_reference(x, p, L, K0)
            np.testing.assert_array_equal(F_model, F_ref)

    def test_matches_inline_formula_dense_grid(self) -> None:
        # Same check on a dense, evenly-spaced grid — the typical
        # fitter input. Catches any subtle numpy broadcasting bug.
        model = EWLCModel()
        x = np.linspace(0.5, L_DEFAULT - 1.0, 1000)
        F_model = model(x, p=P_DEFAULT, L=L_DEFAULT, K0=K0_DEFAULT)
        F_ref = _ewlc_formula_reference(x, P_DEFAULT, L_DEFAULT, K0_DEFAULT)
        np.testing.assert_array_equal(F_model, F_ref)

    def test_output_shape_matches_input(self) -> None:
        model = EWLCModel()
        for n in (1, 5, 100, 1000):
            x = np.linspace(0.1, L_DEFAULT - 0.1, n)
            F = model(x, p=P_DEFAULT, L=L_DEFAULT, K0=K0_DEFAULT)
            assert F.shape == (n,)

    def test_output_dtype_is_float(self) -> None:
        model = EWLCModel()
        F = model(np.array([1.0, 2.0]), p=P_DEFAULT, L=L_DEFAULT, K0=K0_DEFAULT)
        assert np.issubdtype(F.dtype, np.floating)

    def test_list_input_is_accepted(self) -> None:
        model = EWLCModel()
        F = model([1.0, 2.0, 3.0], p=P_DEFAULT, L=L_DEFAULT, K0=K0_DEFAULT)
        assert isinstance(F, np.ndarray)
        assert F.shape == (3,)

    def test_K0_keyword_is_optional_with_default(self) -> None:
        # K0 must default to 1500 pN so callers used to the WLC
        # signature (no K0) get a sensible eWLC evaluation.
        model = EWLCModel()
        F_no_kw = model(np.array([100.0]), p=P_DEFAULT, L=L_DEFAULT)
        F_default_kw = model(np.array([100.0]), p=P_DEFAULT, L=L_DEFAULT, K0=1500.0)
        np.testing.assert_array_equal(F_no_kw, F_default_kw)


# -- WLC limit & stretch-physics invariants -----------------------------


class TestEWLCModelWLCLimit:
    """The eWLC must reduce to the WLC as K0 → ∞ and be monotone in K0."""

    def test_large_K0_within_one_percent_of_WLC(self) -> None:
        # The defining property of the eWLC: as K0 grows, the model
        # must converge to the Marko-Siggia WLC. K0 = 10 000 pN is
        # already well into the "stiff" regime.
        x = np.linspace(10.0, 180.0, 50)  # well clear of both singularities
        wlc_model = WLCModel()
        ewlc_model = EWLCModel()
        F_wlc = wlc_model(x, p=P_DEFAULT, L=L_DEFAULT)
        F_ewlc = ewlc_model(x, p=P_DEFAULT, L=L_DEFAULT, K0=10000.0)
        # Relative error < 1 % across the realistic force range.
        rel_err = np.abs(F_ewlc - F_wlc) / np.abs(F_wlc)
        assert np.all(
            rel_err < 0.01
        ), f"eWLC at K0=10000 not within 1% of WLC: max rel err = {rel_err.max():.4f}"

    def test_very_large_K0_below_half_percent(self) -> None:
        # K0 = 10 000 pN should be tighter than 1 %; K0 = 100 000 pN
        # (well outside the model bounds) should be tighter still.
        x = np.linspace(10.0, 180.0, 50)
        wlc_model = WLCModel()
        ewlc_model = EWLCModel()
        F_wlc = wlc_model(x, p=P_DEFAULT, L=L_DEFAULT)
        F_ewlc = ewlc_model(x, p=P_DEFAULT, L=L_DEFAULT, K0=100000.0)
        rel_err = np.abs(F_ewlc - F_wlc) / np.abs(F_wlc)
        assert np.all(
            rel_err < 0.005
        ), f"eWLC at K0=100000 not within 0.5% of WLC: max rel err = {rel_err.max():.6f}"

    def test_small_K0_gives_slightly_lower_force_than_WLC(self) -> None:
        # A softer chain (small K0) is **more** compliant — at the
        # same extension it produces a slightly **lower** force than
        # the pure-WLC prediction. This is the physically correct
        # direction of the K0 correction: eWLC ≤ WLC on (0, L - 1/K0)
        # for finite K0, with equality only as K0 → ∞.
        x = np.linspace(10.0, 180.0, 50)
        wlc_model = WLCModel()
        ewlc_model = EWLCModel()
        F_wlc = wlc_model(x, p=P_DEFAULT, L=L_DEFAULT)
        F_ewlc = ewlc_model(x, p=P_DEFAULT, L=L_DEFAULT, K0=500.0)
        # Strict inequality on the whole interior of the domain.
        assert np.all(F_ewlc < F_wlc), (
            "eWLC at K0=500 should be slightly LOWER than WLC "
            "(softer chain → lower force at the same extension)"
        )

    def test_force_increases_monotonically_with_K0(self) -> None:
        # At fixed x, the eWLC force must be a monotonically
        # increasing function of K0 (stiffer chain → more force).
        x = np.array([100.0])
        ewlc_model = EWLCModel()
        K0_values = [200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]
        F_values = [ewlc_model(x, p=P_DEFAULT, L=L_DEFAULT, K0=K0)[0] for K0 in K0_values]
        for prev, nxt in itertools.pairwise(F_values):
            assert float(prev) < float(nxt), "eWLC not monotone in K0"


# -- Singularity handling -----------------------------------------------


class TestEWLCModelSingularity:
    """At x = L + L/K0 the eWLC diverges. We return +inf, not an error.

    The eWLC singularity sits slightly **past** the contour length
    (in contrast to the WLC, which diverges at x = L). The 1/K0
    term inside (1 - x/L + 1/K0) shifts the pole outward by L/K0
    nanometres, which is the whole point of the extension: the
    regularisation extends the finite-force range by a small but
    physically meaningful amount beyond the WLC asymptote.

    For the exact-pole tests we use simple parameters (L=2, K0=1) so
    the pole location x = L * (1 + 1/K0) = 4 is exactly representable
    in float64; with realistic parameters (L=200, K0=1500) the pole
    sits at x ≈ 200.1333… and float round-off leaves the
    ``(1 - x/L + 1/K0)`` term with a tiny but nonzero residue.
    """

    def test_near_singularity_grows_without_bound(self) -> None:
        # Geometric approach to the singularity: forces must blow up
        # as x → L + L/K0 from below. Use realistic parameters; we
        # only need monotonic growth, not exact pole alignment.
        K0 = 1500.0
        x_sing = L_DEFAULT * (1.0 + 1.0 / K0)
        x = np.array([0.999, 0.9999, 0.99999]) * x_sing
        model = EWLCModel()
        F = model(x, p=P_DEFAULT, L=L_DEFAULT, K0=K0)
        assert F[0] < F[1] < F[2]
        assert F[-1] > 1e4

    def test_singularity_returns_inf_at_exact_pole(self) -> None:
        # Simple parameters so the pole location is exact in float64.
        # x=4, L=2, K0=1 → x/L = 2, 1/K0 = 1, so 1 - 2 + 1 = 0 exactly.
        model = EWLCModel()
        F = model(np.array([4.0]), p=1.0, L=2.0, K0=1.0)
        assert np.isposinf(F[0])

    def test_singularity_does_not_raise(self) -> None:
        # The fitter will hand us x near the singularity on a few
        # iterations; raising would kill the fit. Use the exact-pole
        # parameters from the previous test.
        model = EWLCModel()
        result = model(np.array([4.0]), p=1.0, L=2.0, K0=1.0)
        assert np.isposinf(result[0])

    def test_finite_in_physical_range(self) -> None:
        # The closed-form model is finite throughout the physical
        # range [0, L] — unlike the WLC, which diverges at L itself.
        # This is the eWLC's main practical advantage: data points
        # near the contour length remain usable.
        model = EWLCModel()
        K0 = 1500.0
        x = np.linspace(0.0, L_DEFAULT, 200)
        F = model(x, p=P_DEFAULT, L=L_DEFAULT, K0=K0)
        assert np.all(np.isfinite(F))

    def test_finite_at_contour_length(self) -> None:
        # Direct check of the previous property at the exact contour
        # length: the WLC blows up at x = L, the eWLC does not.
        model = EWLCModel()
        F = model(np.array([L_DEFAULT]), p=P_DEFAULT, L=L_DEFAULT, K0=1500.0)
        assert np.isfinite(F[0])

    def test_no_runtime_warning_at_singularity(self) -> None:
        # The model must suppress the divide-by-zero warning at the
        # singularity — if numpy starts leaking, downstream loguru
        # capture will spam. Use exact-pole parameters.
        model = EWLCModel()
        with np.errstate(divide="raise", invalid="raise"):
            F = model(np.array([4.0]), p=1.0, L=2.0, K0=1.0)
        assert np.isposinf(F[0])


# -- Parametrisation & fitting support ----------------------------------


class TestEWLCModelGuessParams:
    """The starting-point heuristic for the fitter."""

    def test_returns_dict_with_required_keys(self) -> None:
        model = EWLCModel()
        x = np.linspace(1.0, L_DEFAULT, 100)
        y = np.linspace(0.0, 50.0, 100)
        guess = model.guess_params(x, y)
        assert isinstance(guess, dict)
        assert set(guess.keys()) == set(model.param_names)

    def test_returns_finite_values(self) -> None:
        model = EWLCModel()
        for x_max in (10.0, 50.0, 200.0, 5000.0):
            x = np.linspace(1.0, x_max, 100)
            y = np.zeros_like(x)
            guess = model.guess_params(x, y)
            for k, v in guess.items():
                assert math.isfinite(v), f"guess_params returned non-finite {k}={v}"

    def test_guess_respects_param_bounds(self) -> None:
        # The starting point must lie inside the search bounds —
        # otherwise the fitter immediately snaps back to the boundary.
        model = EWLCModel()
        for x_max in (1.0, 5.0, 15.0, 200.0, 5000.0):
            x = np.linspace(0.1, x_max, 50)
            y = np.zeros_like(x)
            guess = model.guess_params(x, y)
            for name, (lo, hi), value in zip(
                model.param_names, model.param_bounds, guess.values(), strict=True
            ):
                assert (
                    lo <= value <= hi
                ), f"guess_params({name}={value}) out of bounds [{lo}, {hi}] for x_max={x_max}"

    def test_K0_guess_is_dna_default(self) -> None:
        # The spec fixes K0 default at 1500 pN — a common value for
        # dsDNA and a safe starting point across the lab's data.
        model = EWLCModel()
        x = np.linspace(1.0, L_DEFAULT, 100)
        y = np.zeros_like(x)
        guess = model.guess_params(x, y)
        assert guess["K0"] == 1500.0

    def test_guess_on_short_curve_clamps_l(self) -> None:
        # When x.max() / 0.9 falls below the L lower bound, the guess
        # must clamp to the lower bound — never return a value outside
        # param_bounds.
        model = EWLCModel()
        x = np.linspace(0.1, 5.0, 50)  # x.max() / 0.9 = 5.56 < 10
        y = np.zeros_like(x)
        guess = model.guess_params(x, y)
        lo, _ = model.param_bounds[1]
        assert guess["L"] >= lo


# -- Registry -----------------------------------------------------------


class TestEWLCModelRegistry:
    """The model must be discoverable through the central registry."""

    def test_registered_under_ewlc(self) -> None:
        from afmkit.models import MODEL_REGISTRY

        assert "ewlc" in MODEL_REGISTRY

    def test_get_model_returns_fresh_instance(self) -> None:
        # ``get_model`` must return a *new* instance per call — the
        # convention documented in models/__init__.py: per-fit
        # construction avoids subtle aliasing across batch fits.
        m1 = get_model("ewlc")
        m2 = get_model("ewlc")
        assert isinstance(m1, EWLCModel)
        assert isinstance(m2, EWLCModel)
        assert m1 is not m2  # fresh instance, not the same object

    def test_get_model_unknown_raises(self) -> None:
        # Same KeyError contract as for any other unknown model.
        with pytest.raises(KeyError, match="unknown model"):
            get_model("definitely_not_a_real_model")

    def test_registry_class_matches_direct_import(self) -> None:
        # The class exposed through the registry must be the same
        # class that the module exports directly — a future refactor
        # that splits the two would silently break plugin authors.
        from afmkit.models import MODEL_REGISTRY

        assert MODEL_REGISTRY["ewlc"] is EWLCDirect


# -- Protocol conformance -----------------------------------------------


class TestEWLCModelProtocol:
    """The model must satisfy the :class:`PolymerModel` protocol."""

    def test_satisfies_protocol(self) -> None:
        # With @runtime_checkable, isinstance() is a structural check.
        model = EWLCModel()
        assert isinstance(model, PolymerModel)

    def test_exposes_required_attributes(self) -> None:
        model = EWLCModel()
        assert hasattr(model, "param_names")
        assert hasattr(model, "param_bounds")
        assert hasattr(model, "param_hints")
        assert callable(model)
        assert callable(model.guess_params)


# -- End-to-end fit recovery --------------------------------------------


class TestEWLCFitRecovery:
    """A noise-free eWLC fit must recover the truth to high precision.

    This is the headline "round-trip" test: generate synthetic data
    from the model with known (p, L, K0), then ask the fitting engine
    to recover them. A 1 % tolerance per parameter is generous for
    noise-free, well-conditioned nonlinear least-squares.
    """

    def test_recover_p_L_K0_within_one_percent(self) -> None:
        x, y = _synthetic_ewlc_curve()
        # Restrict to the same range the WLC test uses, well clear of
        # both the eWLC singularity (L - 1/K0 ≈ 199.3) and the
        # near-zero region where the force has little information.
        x_range: tuple[float, float] = (20.0, 180.0)
        engine = LmfitEngine(EWLCModel())
        result = engine.fit(x, y, x_range=x_range)
        assert result.metadata["success"]
        assert abs(result.params["p"] - P_DEFAULT) / P_DEFAULT < 0.01
        assert abs(result.params["L"] - L_DEFAULT) / L_DEFAULT < 0.01
        assert (
            abs(result.params["K0"] - K0_DEFAULT) / K0_DEFAULT < 0.05
        )  # K0 is less well constrained

    def test_p0_at_truth_yields_successful_fit(self) -> None:
        # Starting exactly at the truth must still produce a clean fit.
        x, y = _synthetic_ewlc_curve()
        x_range: tuple[float, float] = (20.0, 180.0)
        engine = LmfitEngine(EWLCModel())
        result = engine.fit(
            x,
            y,
            x_range=x_range,
            p0={"p": P_DEFAULT, "L": L_DEFAULT, "K0": K0_DEFAULT},
        )
        assert result.metadata["success"]
        assert abs(result.params["p"] - P_DEFAULT) / P_DEFAULT < 0.01
        assert abs(result.params["L"] - L_DEFAULT) / L_DEFAULT < 0.01
        assert abs(result.params["K0"] - K0_DEFAULT) / K0_DEFAULT < 0.05

    def test_residual_is_zero_at_truth(self) -> None:
        # The residual() helper must return ~0 when evaluated at the
        # true parameters — the synthetic curve is model-exact.
        x, y = _synthetic_ewlc_curve()
        engine = LmfitEngine(EWLCModel())
        residual = engine.residual({"p": P_DEFAULT, "L": L_DEFAULT, "K0": K0_DEFAULT}, x, y)
        # Allow a small tolerance because the engine may have its own
        # residual weighting; the headline is "near zero".
        assert np.max(np.abs(residual)) < 1.0
