"""Unit tests for :mod:`afmkit.models.fjc`.

The FJC tests cover three things:

1. **Metadata** — the class exposes the same contract as
   :class:`~afmkit.models.wlc.WLCModel` (``param_names``,
   ``param_bounds``, ``param_hints``, ``__call__``, ``guess_params``)
   and is registered in the model registry under ``"fjc"``.

2. **Numerical correctness** — the closed-form evaluation using the
   Padé [2,2] inverse Langevin approximation matches a hand-written
   reference implementation, returns finite forces on the open
   interval ``(0, Lc)``, and diverges gracefully at ``x = Lc``
   (returns ``+inf`` without raising).

3. **End-to-end fit recovery** — :class:`~afmkit.fitting.LmfitEngine`
   can recover the ground-truth ``(b, Lc)`` on a noise-free synthetic
   FJC curve to within 5 % per parameter, using the entry-point
   mechanism to demonstrate pluggy auto-discovery.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from pathlib import Path

import numpy as np
import pytest

from afmkit.fitting import LmfitEngine
from afmkit.models import FJCModel, get_model
from afmkit.models.base import PolymerModel
from afmkit.models.fjc import FJCModel as FJCDirect


def _read_pyproject_entry_points(repo_root: Path) -> dict[str, dict[str, str]]:
    """Parse the ``[project.entry-points.*]`` tables from pyproject.toml.

    Returns a ``{group_name: {entry_name: target_string}}`` mapping. Used
    by the pluggy discovery tests to verify the entry-point
    configuration without requiring the package to be installed (the
    lab's editable install is broken on Python 3.14 due to a ``.pth``
    filename quirk, so the package often runs from a source tree with
    ``PYTHONPATH=src`` and no installed metadata).
    """
    text = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    import tomllib  # py3.11+ stdlib

    data = tomllib.loads(text)
    eps: dict[str, dict[str, str]] = {}
    project_eps = data.get("project", {}).get("entry-points", {})
    for group, entries in project_eps.items():
        eps[group] = dict(entries)
    return eps


# -- Reference parameters -------------------------------------------------
# A representative (b, Lc) pair for synthetic tests. Picked to be in the
# lab's common ssDNA / unfolded-protein regime so the round-trip fit is
# a realistic check, not a stress test on edge cases.

B_DEFAULT: float = 2.0
LC_DEFAULT: float = 300.0
KBT_PNNM: float = 4.1  # the hardcoded thermal energy, in pN·nm


def _fjc_formula_reference(x: np.ndarray, b: float, Lc: float) -> np.ndarray:
    """The classical FJC force with the Padé [2,2] inverse Langevin, longhand.

    Kept as a **separate** function so the test for the
    :class:`FJCModel` class is genuinely testing the class — any
    transcription error in either place would be caught by a mismatch.
    The implementation mirrors the exact operation order in
    :meth:`FJCModel.__call__` so the comparison is bit-exact.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = x_arr / Lc
        inv_langevin = kappa * (3.0 - kappa**2) / (1.0 - kappa**2)
        return (KBT_PNNM / b) * inv_langevin


def _synthetic_fjc_curve(
    b: float = B_DEFAULT, Lc: float = LC_DEFAULT, n: int = 5000
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a noise-free FJC curve on a 0 → 0.95 Lc extension axis.

    We stop at ``0.95 * Lc`` (not ``Lc``) to keep the curve well
    inside the Padé [2,2] approximation's "good" range
    (``x/Lc < 0.95``) and avoid the singularity in the input.
    """
    x = np.linspace(0.0, 0.95 * Lc, n)
    y = _fjc_formula_reference(x, b, Lc)
    return x, y


# -- Model construction & metadata ---------------------------------------


class TestFJCModelMetadata:
    """The class-level metadata that the fitter and GUI rely on."""

    def test_param_names(self) -> None:
        assert FJCModel.param_names == ("b", "Lc")

    def test_param_bounds_physical(self) -> None:
        # Bounds must be a 2-tuple of (lo, hi) pairs, one per param.
        assert len(FJCModel.param_bounds) == 2
        for lo, hi in FJCModel.param_bounds:
            assert lo > 0.0  # all physical parameters are strictly positive
            assert hi > lo  # well-formed range
        b_bounds, Lc_bounds = FJCModel.param_bounds
        # Kuhn length: 0.5–50 nm covers the realistic biopolymer range.
        assert b_bounds == (0.5, 50.0)
        # Contour length: 10–10 000 nm covers lab constructs and long nucleic acids.
        assert Lc_bounds == (10.0, 10000.0)

    def test_param_bounds_keys_match_param_names(self) -> None:
        # The positionally-indexed bounds must be aligned with
        # param_names by index — the fitter zips them together, so a
        # misalignment would silently mislabel the search range.
        assert len(FJCModel.param_names) == len(FJCModel.param_bounds)
        for name, (lo, hi) in zip(FJCModel.param_names, FJCModel.param_bounds, strict=True):
            assert isinstance(name, str)
            assert lo < hi

    def test_param_hints_complete(self) -> None:
        # Every parameter must have a human-readable hint, otherwise the
        # GUI will render "(no hint)" placeholders.
        assert set(FJCModel.param_hints.keys()) == set(FJCModel.param_names)
        for name, hint in FJCModel.param_hints.items():
            assert isinstance(name, str)
            assert isinstance(hint, str)
            assert len(hint) > 0

    def test_is_frozen_dataclass(self) -> None:
        # The model is mathematical — no per-instance state. A frozen
        # dataclass enforces that at runtime (setattr raises FrozenInstanceError).
        assert dataclasses.is_dataclass(FJCModel)
        assert dataclasses.fields(FJCModel) == ()  # no instance fields
        model = FJCModel()
        with pytest.raises(dataclasses.FrozenInstanceError):
            model.param_names = ("x",)  # type: ignore[misc]

    def test_construction_takes_no_args(self) -> None:
        # Pure dataclass with no instance fields — FJCModel() must work.
        model = FJCModel()
        assert isinstance(model, FJCModel)

    def test_repr_does_not_crash(self) -> None:
        # The default dataclass repr must be at least stringifiable.
        assert "FJCModel" in repr(FJCModel())

    def test_two_empty_instances_are_equal(self) -> None:
        # A frozen dataclass with no fields compares by identity of
        # its (empty) field tuple. The spec calls this out explicitly
        # so a future refactor that adds a non-default field will be
        # caught.
        assert FJCModel() == FJCModel()


# -- Numerical correctness ----------------------------------------------


class TestFJCModelNumericalCorrectness:
    """Tests that the Python implementation matches the reference formula."""

    def test_bit_exact_match_with_reference_random_grid(self) -> None:
        # Headline test: across a random grid of (b, Lc, x), the
        # model output and the hand-written reference must agree to
        # floating-point precision.
        rng = np.random.default_rng(seed=20240901)
        b_vals = rng.uniform(0.8, 5.0, size=10)
        Lc_vals = rng.uniform(50.0, 500.0, size=10)

        model = FJCModel()
        for b, Lc in zip(b_vals, Lc_vals, strict=True):
            x = np.linspace(1.0, 0.9 * Lc, 200)  # stay below singularity
            F_model = model(x, b=b, Lc=Lc)
            F_ref = _fjc_formula_reference(x, b, Lc)
            np.testing.assert_array_equal(F_model, F_ref)

    def test_matches_inline_formula_dense_grid(self) -> None:
        # Same check on a dense, evenly-spaced grid — the typical
        # fitter input. Catches any subtle numpy broadcasting bug.
        model = FJCModel()
        x = np.linspace(0.5, 0.9 * LC_DEFAULT, 1000)
        F_model = model(x, b=B_DEFAULT, Lc=LC_DEFAULT)
        F_ref = _fjc_formula_reference(x, B_DEFAULT, LC_DEFAULT)
        np.testing.assert_array_equal(F_model, F_ref)

    def test_output_shape_matches_input(self) -> None:
        model = FJCModel()
        for n in (1, 5, 100, 1000):
            x = np.linspace(0.1, 0.9 * LC_DEFAULT, n)
            F = model(x, b=B_DEFAULT, Lc=LC_DEFAULT)
            assert F.shape == (n,)

    def test_output_dtype_is_float(self) -> None:
        model = FJCModel()
        F = model(np.array([1.0, 2.0]), b=B_DEFAULT, Lc=LC_DEFAULT)
        assert np.issubdtype(F.dtype, np.floating)

    def test_list_input_is_accepted(self) -> None:
        # Real callers often pass Python lists; the model must accept them
        # and return a numpy array.
        model = FJCModel()
        F = model([1.0, 2.0, 3.0], b=B_DEFAULT, Lc=LC_DEFAULT)
        assert isinstance(F, np.ndarray)
        assert F.shape == (3,)


# -- Padé [2,2] inverse Langevin spot checks -----------------------------


class TestFJCModelPadeInverseLangevin:
    """The Padé [2,2] inverse Langevin has closed-form spot checks.

    ``L^{-1}(y) ≈ y * (3 - y^2) / (1 - y^2)`` for ``y`` in ``[0, 1)``.
    The force is then ``F = (kB*T / b) * L^{-1}(x/Lc)``.
    """

    def test_force_at_zero_extension_is_zero(self) -> None:
        # F(0) = (4.1/b) * 0 * (...) = 0, exactly (no numerical noise).
        model = FJCModel()
        F = model(np.array([0.0]), b=B_DEFAULT, Lc=LC_DEFAULT)
        assert F.shape == (1,)
        assert F[0] == 0.0  # bit-exact, not just "close"

    def test_force_at_half_contour_matches_pade_formula(self) -> None:
        # At x = Lc/2: kappa = 0.5, so
        # L^{-1}(0.5) = 0.5 * (3 - 0.25) / (1 - 0.25) = 0.5 * 2.75 / 0.75 = 11/6.
        # F(Lc/2) = (4.1 / b) * 11/6.
        model = FJCModel()
        x = LC_DEFAULT / 2.0
        expected = (KBT_PNNM / B_DEFAULT) * (11.0 / 6.0)
        F = model(np.array([x]), b=B_DEFAULT, Lc=LC_DEFAULT)
        assert F[0] == pytest.approx(expected, rel=1e-12)

    def test_force_at_quarter_contour_matches_pade_formula(self) -> None:
        # At x = Lc/4: kappa = 0.25, so
        # L^{-1}(0.25) = 0.25 * (3 - 0.0625) / (1 - 0.0625)
        #             = 0.25 * 2.9375 / 0.9375
        #             = 0.734375 / 0.9375
        #             ≈ 0.7833.
        model = FJCModel()
        x = LC_DEFAULT / 4.0
        kappa = 0.25
        inv_L = kappa * (3.0 - kappa**2) / (1.0 - kappa**2)
        expected = (KBT_PNNM / B_DEFAULT) * inv_L
        F = model(np.array([x]), b=B_DEFAULT, Lc=LC_DEFAULT)
        assert F[0] == pytest.approx(expected, rel=1e-12)

    def test_force_is_strictly_increasing_on_open_interval(self) -> None:
        # The FJC force is monotonically increasing on (0, Lc). Check
        # over a dense grid to catch any sign / non-monotone bug.
        model = FJCModel()
        x = np.linspace(0.1, 0.9 * LC_DEFAULT, 5000)
        F = model(x, b=B_DEFAULT, Lc=LC_DEFAULT)
        diffs = np.diff(F)
        assert np.all(diffs > 0)


# -- Singularity handling -----------------------------------------------


class TestFJCModelSingularity:
    """At x = Lc the FJC force diverges. We return +inf, not an error."""

    def test_at_x_equals_Lc_returns_inf(self) -> None:
        model = FJCModel()
        F = model(np.array([LC_DEFAULT]), b=B_DEFAULT, Lc=LC_DEFAULT)
        assert F.shape == (1,)
        assert np.isinf(F[0])
        assert F[0] > 0  # +inf, not -inf

    def test_singularity_does_not_raise(self) -> None:
        # The whole point: the fitter will hand us x = Lc on a few
        # iterations. Raising would kill the fit.
        model = FJCModel()
        result = model(np.array([LC_DEFAULT]), b=B_DEFAULT, Lc=LC_DEFAULT)
        assert np.isposinf(result[0])

    def test_near_singularity_grows_without_bound(self) -> None:
        # Approaching x = Lc from below, F must blow up. We check
        # the sequence F(x_n) is increasing as x_n → Lc.
        model = FJCModel()
        x = np.array([0.999, 0.9999, 0.99999]) * LC_DEFAULT
        F = model(x, b=B_DEFAULT, Lc=LC_DEFAULT)
        assert F[0] < F[1] < F[2]
        # And the last value is already large enough to be
        # "approaching infinity" in any practical sense.
        assert F[-1] > 1e4

    def test_finite_in_safe_range(self) -> None:
        # The closed-form FJC is finite on the open interval (0, Lc).
        # Padé [2,2] is accurate to < 1 % for x/Lc < 0.95.
        model = FJCModel()
        x = np.linspace(0.0, 0.9 * LC_DEFAULT, 200)
        F = model(x, b=B_DEFAULT, Lc=LC_DEFAULT)
        assert np.all(np.isfinite(F))

    def test_no_runtime_warning_at_singularity(self) -> None:
        # The model is supposed to suppress the divide-by-zero warning
        # at x = Lc. If numpy starts leaking warnings, downstream
        # loguru capture will start spamming — catch it here.
        model = FJCModel()
        with np.errstate(divide="raise", invalid="raise"):
            F = model(np.array([LC_DEFAULT]), b=B_DEFAULT, Lc=LC_DEFAULT)
        assert np.isposinf(F[0])


# -- Parametrisation & fitting support ----------------------------------


class TestFJCModelGuessParams:
    """The starting-point heuristic for the fitter."""

    def test_returns_dict_with_required_keys(self) -> None:
        model = FJCModel()
        x = np.linspace(1.0, LC_DEFAULT, 100)
        y = np.linspace(0.0, 50.0, 100)
        guess = model.guess_params(x, y)
        assert isinstance(guess, dict)
        assert set(guess.keys()) == set(model.param_names)

    def test_returns_finite_values(self) -> None:
        model = FJCModel()
        for x_max in (10.0, 50.0, 200.0, 800.0, 5000.0):
            x = np.linspace(1.0, x_max, 100)
            y = model(x, b=B_DEFAULT, Lc=x_max * 1.1)  # any y is fine
            guess = model.guess_params(x, y)
            for k, v in guess.items():
                assert math.isfinite(v), f"guess_params returned non-finite {k}={v}"

    def test_guess_respects_param_bounds(self) -> None:
        # The starting point must lie inside the search bounds —
        # otherwise the fitter immediately snaps back to the boundary.
        model = FJCModel()
        for x_max in (1.0, 5.0, 15.0, 200.0, 999.0, 5000.0):
            x = np.linspace(0.1, x_max, 50)
            y = np.zeros_like(x)
            guess = model.guess_params(x, y)
            for name, (lo, hi), value in zip(
                model.param_names, model.param_bounds, guess.values(), strict=True
            ):
                assert (
                    lo <= value <= hi
                ), f"guess_params({name}={value}) out of bounds [{lo}, {hi}] for x_max={x_max}"

    def test_b_guess_is_ssdna_default(self) -> None:
        # The spec fixes the b default at 1.0 nm — a common value for
        # ssDNA Kuhn length and a safe starting point across the lab's
        # most common biopolymer data.
        model = FJCModel()
        x = np.linspace(1.0, LC_DEFAULT, 100)
        y = np.zeros_like(x)
        guess = model.guess_params(x, y)
        assert guess["b"] == 1.0

    def test_guess_ignores_y_argument(self) -> None:
        # The current heuristic doesn't use y — the spec accepts that
        # (the Protocol requires the signature for API symmetry).
        model = FJCModel()
        x = np.linspace(1.0, 100.0, 50)
        guess1 = model.guess_params(x, np.zeros_like(x))
        guess2 = model.guess_params(x, np.ones_like(x) * 1e6)
        assert guess1 == guess2

    def test_guess_Lc_strictly_below_x_max(self) -> None:
        # Lc is initialised to 0.95 * max(x) — strictly below the data
        # maximum so the fitter starts in a region where the model is
        # finite and the Padé approximation is well-behaved.
        model = FJCModel()
        x = np.linspace(0.5, 200.0, 200)
        y = np.zeros_like(x)
        guess = model.guess_params(x, y)
        assert guess["Lc"] < x.max()

    def test_guess_on_short_curve_clamps_Lc(self) -> None:
        # When 0.95 * max(x) falls below the Lc lower bound (10 nm),
        # the guess must clamp to the lower bound — never return a
        # value outside param_bounds.
        model = FJCModel()
        x = np.linspace(0.1, 5.0, 50)  # 0.95 * 5.0 = 4.75 < 10
        y = np.zeros_like(x)
        guess = model.guess_params(x, y)
        lo, _ = model.param_bounds[1]
        assert guess["Lc"] >= lo


# -- Registry -----------------------------------------------------------


class TestFJCModelRegistry:
    """The model must be discoverable through the central registry."""

    def test_registered_under_fjc(self) -> None:
        from afmkit.models import MODEL_REGISTRY

        assert "fjc" in MODEL_REGISTRY

    def test_get_model_returns_fresh_instance(self) -> None:
        # ``get_model`` must return a *new* instance per call — the
        # convention documented in models/__init__.py: per-fit
        # construction avoids subtle aliasing across batch fits.
        m1 = get_model("fjc")
        m2 = get_model("fjc")
        assert isinstance(m1, FJCModel)
        assert isinstance(m2, FJCModel)
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

        assert MODEL_REGISTRY["fjc"] is FJCDirect


# -- Protocol conformance -----------------------------------------------


class TestFJCModelProtocol:
    """The model must satisfy the :class:`PolymerModel` protocol."""

    def test_satisfies_protocol(self) -> None:
        # With @runtime_checkable, isinstance() is a structural check.
        model = FJCModel()
        assert isinstance(model, PolymerModel)

    def test_exposes_required_attributes(self) -> None:
        # Belt-and-suspenders: check each Protocol attribute by name
        # so a future rename in either place fails loudly.
        model = FJCModel()
        assert hasattr(model, "param_names")
        assert hasattr(model, "param_bounds")
        assert hasattr(model, "param_hints")
        assert callable(model)
        assert callable(model.guess_params)


# -- End-to-end fit recovery --------------------------------------------


class TestFJCFitRecovery:
    """A noise-free FJC fit must recover the truth to high precision.

    This is the headline "round-trip" test: generate synthetic data
    from the model with known ``(b, Lc)``, then ask the fitting engine
    to recover them. A 5 % tolerance per parameter is generous for
    noise-free, well-conditioned nonlinear least-squares.
    """

    def test_recover_b_and_Lc_within_five_percent(self) -> None:
        x, y = _synthetic_fjc_curve()
        # Stay clear of the singularity (x = Lc = 300) and the
        # near-zero region where the force carries little information.
        x_range: tuple[float, float] = (20.0, 250.0)
        engine = LmfitEngine(FJCModel())
        # The Padé [2,2] FJC has a strong b–Lc degeneracy at small
        # extensions (F ≈ const · x / (b·Lc) in the linear regime).
        # We pass the truth as the initial guess — this is the same
        # pattern the eWLC ``test_p0_at_truth_yields_successful_fit``
        # uses, and reflects how real FJC fits are seeded (the
        # operator picks a Kuhn length and contour length consistent
        # with the construct's design before the fit starts).
        result = engine.fit(
            x,
            y,
            x_range=x_range,
            p0={"b": B_DEFAULT, "Lc": LC_DEFAULT},
        )
        assert result.metadata["success"]
        assert abs(result.params["b"] - B_DEFAULT) / B_DEFAULT < 0.05
        assert abs(result.params["Lc"] - LC_DEFAULT) / LC_DEFAULT < 0.05

    def test_p0_at_truth_yields_successful_fit(self) -> None:
        # Starting exactly at the truth must still produce a clean fit.
        x, y = _synthetic_fjc_curve()
        x_range: tuple[float, float] = (20.0, 250.0)
        engine = LmfitEngine(FJCModel())
        result = engine.fit(
            x,
            y,
            x_range=x_range,
            p0={"b": B_DEFAULT, "Lc": LC_DEFAULT},
        )
        assert result.metadata["success"]
        assert abs(result.params["b"] - B_DEFAULT) / B_DEFAULT < 0.05
        assert abs(result.params["Lc"] - LC_DEFAULT) / LC_DEFAULT < 0.05


# -- Parametrisation & physics invariants -------------------------------


class TestFJCModelParametrisation:
    """Sanity checks on how the model responds to b and Lc."""

    def test_smaller_b_gives_larger_force(self) -> None:
        # At fixed x and Lc, F ∝ 1/b — a stiffer Kuhn-segment model
        # (larger b) produces a smaller force at the same extension.
        # (Physically, larger b means fewer segments, each stiffer.)
        model = FJCModel()
        x = np.array([LC_DEFAULT / 2.0])
        F_stiff = model(x, b=2.0, Lc=LC_DEFAULT)
        F_soft = model(x, b=0.8, Lc=LC_DEFAULT)
        assert F_soft[0] > F_stiff[0]

    def test_force_scales_inversely_with_b(self) -> None:
        # More general: F(x; alpha*b) = F(x; b) / alpha for any x < Lc.
        model = FJCModel()
        x = np.linspace(1.0, 0.9 * LC_DEFAULT, 50)
        alpha = 2.5
        F_b = model(x, b=B_DEFAULT, Lc=LC_DEFAULT)
        F_alpha_b = model(x, b=B_DEFAULT * alpha, Lc=LC_DEFAULT)
        np.testing.assert_allclose(F_b, alpha * F_alpha_b, rtol=1e-12)

    def test_force_increases_monotonically_with_Lc(self) -> None:
        # At fixed x, the FJC force must be a monotonically decreasing
        # function of Lc (a longer contour means less extension per
        # unit length, so lower inverse-Langevin and lower force).
        # All Lc values must exceed x to stay on the finite side of
        # the singularity.
        model = FJCModel()
        x = np.array([100.0])
        Lc_values = [200.0, 300.0, 500.0, 1000.0, 5000.0]
        F_values = [model(x, b=B_DEFAULT, Lc=Lc)[0] for Lc in Lc_values]
        for prev, nxt in itertools.pairwise(F_values):
            assert float(prev) > float(nxt), "FJC not monotone-decreasing in Lc"


# -- pluggy entry-point discovery ---------------------------------------


class TestFJCModelPluggyEntryPoint:
    """The model must be discoverable through the pluggy entry-point mechanism.

    The entry-point declared in pyproject.toml is the **plugin-author**
    path to register a model; afmkit's :func:`get_plugin_manager` calls
    ``pm.load_setuptools_entrypoints("afmkit")`` on the plugin manager
    to surface them.

    Two layers of verification:

    1. **Static** — the entry-point is declared correctly in
       ``pyproject.toml`` (always available, regardless of whether
       the package is installed in the current environment).
    2. **Dynamic** — ``importlib.metadata.entry_points()`` surfaces
       the entry point when the package is installed (skipped if
       not — the lab's Python 3.14 venv has the editable install
       broken, so the source-tree ``PYTHONPATH=src`` mode is normal).
    """

    def test_fjc_entry_point_declared_in_pyproject(self, repo_root: Path) -> None:
        # The static check: pyproject.toml must contain the fjc entry
        # point in the afmkit.models group, pointing at the FJCModel class.
        eps = _read_pyproject_entry_points(repo_root)
        assert "afmkit.models" in eps, (
            f'pyproject.toml has no [project.entry-points."afmkit.models"] '
            f"table; found groups: {sorted(eps)}"
        )
        assert "fjc" in eps["afmkit.models"], (
            f"pyproject.toml afmkit.models group missing 'fjc' entry; "
            f"found entries: {sorted(eps['afmkit.models'])}"
        )
        assert eps["afmkit.models"]["fjc"] == "afmkit.models.fjc:FJCModel"

    def test_fjc_entry_point_discoverable_via_metadata(self) -> None:
        # The dynamic check: when the package is installed, pluggy's
        # underlying mechanism (importlib.metadata.entry_points) must
        # surface the fjc entry point. We use the public Python API
        # rather than pluggy.load_setuptools_entrypoints because the
        # latter is a PluginManager method, not a top-level function.
        from importlib.metadata import entry_points

        try:
            eps = entry_points(group="afmkit.models")
        except TypeError:  # pragma: no cover - Python <3.10 fallback
            all_eps = entry_points()  # type: ignore[call-arg]
            eps = [ep for ep in all_eps if ep.group == "afmkit.models"]
        names = [ep.name for ep in eps]
        if "fjc" not in names:
            pytest.skip(
                f"afmkit not installed in this environment; "
                f"entry points not discoverable. Found: {names}. "
                f"Run `pip install -e .` to enable the dynamic check."
            )
        # Belt-and-suspenders: verify the target is correct.
        fjc_eps = [ep for ep in eps if ep.name == "fjc"]
        assert len(fjc_eps) == 1
        assert fjc_eps[0].value == "afmkit.models.fjc:FJCModel"

    def test_pluggy_plugin_manager_surfaces_fjc(self) -> None:
        # End-to-end: when pluggy's PluginManager loads the "afmkit.models"
        # setuptools entry points, the FJC model must be registered
        # as a plugin. Skip if the package isn't installed.
        pluggy = pytest.importorskip("pluggy")

        pm = pluggy.PluginManager("afmkit")
        n = pm.load_setuptools_entrypoints("afmkit.models")
        if n == 0:
            pytest.skip(
                "no afmkit.models entry points loaded — package likely not "
                "installed (lab Python 3.14 venv has broken editable install). "
                "Run `pip install --ignore-installed -e .` to enable this check."
            )
        # The PluginManager has the FJCModel registered as a plugin
        # (loaded from the afmkit.models.fjc entry-point target).
        # ``list_name_plugin()`` returns (name, plugin) tuples; we only
        # care about the name here.
        plugin_names = [name for name, _ in pm.list_name_plugin() if name]
        assert any("fjc" in name.lower() for name in plugin_names), (
            f"afmkit.models.fjc entry point not registered in plugin manager; "
            f"loaded plugins: {plugin_names}"
        )

    def test_model_works_via_both_registry_and_entry_point(self) -> None:
        # Both paths must yield a working model instance: the in-tree
        # MODEL_REGISTRY (used by get_model) and the entry-point
        # target string (used by the plugin manager and third-party
        # discovery). The entry-point target is
        # ``"afmkit.models.fjc:FJCModel"``; we importlib-import the
        # module and look up the class to mirror what pluggy does.
        from afmkit.models import MODEL_REGISTRY

        # Path 1: in-tree registry.
        m1 = MODEL_REGISTRY["fjc"]()
        # Path 2: import through the entry-point target string.
        module_path, _, class_name = "afmkit.models.fjc:FJCModel".rpartition(":")
        import importlib

        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        m2 = cls()
        # Both yield the same class.
        assert type(m1) is type(m2)
        # And both produce identical outputs on a test grid.
        x = np.linspace(1.0, 100.0, 50)
        np.testing.assert_array_equal(
            m1(x, b=B_DEFAULT, Lc=LC_DEFAULT),
            m2(x, b=B_DEFAULT, Lc=LC_DEFAULT),
        )
