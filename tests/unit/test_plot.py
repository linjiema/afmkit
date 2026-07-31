"""Unit tests for :mod:`afmkit.presentation.gui.plot`.

These tests cover the matplotlib-backed :class:`ForceExtensionPlot`
widget. The widget is deliberately headless — its main work is the
:func:`_render_to_pillow` helper, which builds a Pillow image from a
curve + (peaks) + (fit overlay) using matplotlib's Agg backend. The
tests therefore call :meth:`render_curve` and then introspect
:attr:`ForceExtensionPlot._image` rather than spinning up a Textual
``App``.

Optional deps
-------------
``matplotlib`` and ``Pillow`` are ``[plot]`` extras — both are
imported at the top of the test module via :func:`pytest.importorskip`
so a minimal install without the ``[plot]`` extra skips this whole
module cleanly. ``textual`` is similarly optional but required for
the widget's class hierarchy, so we ``importorskip`` it too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

# Optional-deps gates — must come before the afmkit imports so the
# skip is honoured when the package is missing.
matplotlib = pytest.importorskip("matplotlib")
pytest.importorskip("PIL")
textual = pytest.importorskip("textual")

import matplotlib.pyplot as _plt  # noqa: E402  (after importorskip)
import numpy as np  # noqa: E402
from rich.console import Group as _RichGroup  # noqa: E402

from afmkit.analysis.peak_detection import Peak  # noqa: E402
from afmkit.core.curve import ForceCurve  # noqa: E402
from afmkit.fitting.report import FitResult  # noqa: E402
from afmkit.presentation.gui import plot as plot_mod  # noqa: E402
from afmkit.presentation.gui.plot import ForceExtensionPlot  # noqa: E402

if TYPE_CHECKING:
    pass


# -- Helpers --------------------------------------------------------------


def _make_peaks(curve: ForceCurve) -> list[Peak]:
    """Return 3 synthetic peaks at evenly-spaced points on the curve.

    The exact positions are not important — the test only verifies
    that *some* peak overlay is drawn (which is the part the brief
    asks us to exercise). The peaks are placed inside the curve's
    range and use the curve's own force at the candidate index,
    so the WLC fit overlay and the data trace can co-exist.
    """
    ext = curve.extension
    force = curve.force
    n = len(ext)
    picks = [n // 4, n // 2, 3 * n // 4]
    peaks: list[Peak] = []
    for i, idx in enumerate(picks):
        peaks.append(
            Peak(
                index=idx,
                extension=float(ext[idx]),
                force=float(force[idx]),
                prominence=10.0 + i,
                width=5,
                height_drop=5.0,
                confidence=0.8,
            )
        )
    return peaks


def _make_fake_wlc_fit(curve: ForceCurve) -> FitResult:
    """Build a hand-rolled :class:`FitResult` for a WLC fit.

    The numeric values are plausible (p=0.4 nm, L≈max(curve.x)*1.1)
    but not actually fitted — the test only needs an object that
    carries ``model_name="wlc"`` and the right ``params`` keys so
    the overlay path runs end-to-end.
    """
    ext = curve.extension
    n = 100
    x_lo, x_hi = float(ext.min()), float(ext.max())
    x_fit = np.linspace(x_lo + 0.1 * (x_hi - x_lo), x_hi - 0.05 * (x_hi - x_lo), n)
    return FitResult(
        model_name="wlc",
        params={"p": 0.4, "L": float(ext.max()) * 1.1},
        stderr={"p": 0.01, "L": 1.0},
        covariance=None,
        chi_square=0.0,
        reduced_chi_square=0.0,
        n_data=n,
        n_params=2,
        aic=0.0,
        bic=0.0,
        residual=np.zeros(n, dtype=np.float64),
        x_fit=x_fit,
        y_fit=np.zeros(n, dtype=np.float64),
        metadata={"success": True},
    )


# -- Construction ---------------------------------------------------------


def test_construction(synthetic_force_curve: ForceCurve) -> None:
    """``ForceExtensionPlot(40, 10)`` constructs without error.

    The constructor does **not** touch matplotlib or Pillow — those
    imports are deferred to :meth:`render_curve`. This is the
    cheapest possible smoke test that the textual inheritance chain
    is wired up correctly.
    """
    plot = ForceExtensionPlot(width=40, height=10)
    assert plot is not None
    # The cached bitmap starts as ``None`` — before the first
    # render_curve call, the widget is in the empty-placeholder state.
    assert plot._image is None


# -- render_curve: basic (no peaks, no fit) ------------------------------


def test_render_curve_produces_pillow_image(
    synthetic_force_curve: ForceCurve,
) -> None:
    """A bare ``render_curve(curve)`` produces a non-empty Pillow image.

    The test does not pixel-test the matplotlib output — that would
    be brittle across matplotlib versions. Instead we verify the
    two invariants that matter for the test matrix:

    1. :attr:`ForceExtensionPlot._image` is a real ``PIL.Image.Image``
       after the call (not ``None``).
    2. The image's dimensions are ``width_chars * 10`` ×
       ``height_chars * 10`` pixels (i.e. the widget's character
       dimensions, translated to pixels at :data:`_PX_PER_CHAR`).
    """
    from PIL import Image as _PILImage  # local — keeps the import optional

    plot = ForceExtensionPlot(width=40, height=10)
    plot.render_curve(synthetic_force_curve)

    img = plot._image
    assert img is not None
    assert isinstance(img, _PILImage.Image)
    # The bitmap is non-empty (size > 0 in both axes) and matches
    # the requested widget dimensions translated to pixels.
    w, h = img.size
    assert w > 0
    assert h > 0
    # Matplotlib's tight-bbox-free figure gives us exactly width*PX ×
    # height*PX. We use >= rather than == to stay robust against
    # any future pixel-rounding tweak.
    assert w == 40 * plot_mod._PX_PER_CHAR
    assert h == 10 * plot_mod._PX_PER_CHAR


# -- render_curve: peak markers ------------------------------------------


def test_render_curve_with_peaks(synthetic_force_curve: ForceCurve) -> None:
    """``render_curve(curve, peaks=[3 peaks])`` regenerates the image.

    The brief explicitly says we should *not* try to OCR peaks from
    the bitmap — that would couple the test to the matplotlib
    version and the renderer's anti-aliasing choices. The test
    instead verifies two cheaper invariants:

    1. The widget's cached image is regenerated (i.e. the call
       runs through to a successful Pillow ``Image.open``).
    2. The image is non-empty after the call, which it would not
       be if the peak overlay had raised mid-draw.
    """
    plot = ForceExtensionPlot(width=40, height=10)
    peaks = _make_peaks(synthetic_force_curve)
    # Sanity: the helper actually returns the requested 3 peaks.
    assert len(peaks) == 3
    plot.render_curve(synthetic_force_curve, peaks=peaks)
    assert plot._image is not None


# -- render_curve: fit overlay -------------------------------------------


def test_render_curve_with_wlc_fit(synthetic_force_curve: ForceCurve) -> None:
    """A WLC ``FitResult`` exercises the fit overlay path.

    The test does not pixel-test the dashed WLC line — it only
    verifies that :meth:`render_curve` completes without error
    when given a non-trivial ``fit`` argument. The overlay path
    is the only place that imports :class:`WLCModel` and calls it
    on a freshly-generated ``x_range`` array, so reaching the
    end of the function is the signal that the integration is
    wired up.
    """
    plot = ForceExtensionPlot(width=40, height=10)
    fit = _make_fake_wlc_fit(synthetic_force_curve)
    plot.render_curve(synthetic_force_curve, fit=fit)
    assert plot._image is not None


# -- clear() -------------------------------------------------------------


def test_clear_resets_image(synthetic_force_curve: ForceCurve) -> None:
    """``clear()`` after ``render_curve()`` resets ``_image`` to ``None``.

    The placeholder is what the :meth:`render` method checks to
    decide whether to draw the half-block renderable or the
    ``"no data"`` text — the test guards that branch by
    verifying the underlying state.
    """
    plot = ForceExtensionPlot(width=40, height=10)
    plot.render_curve(synthetic_force_curve)
    assert plot._image is not None
    plot.clear()
    assert plot._image is None
    # ``render()`` after ``clear()`` should now be a 2-element
    # Group: title (empty here) + "no data" placeholder. The
    # simplest invariant is "no exception raised and the output
    # is a rich Group".
    out = plot.render()
    assert isinstance(out, _RichGroup)


# -- x_range argument -----------------------------------------------------


def test_x_range_restricts_xlim(
    monkeypatch: pytest.MonkeyPatch, synthetic_force_curve: ForceCurve
) -> None:
    """``x_range=(lo, hi)`` sets the matplotlib axes' xlim to ``(lo, hi)``.

    The test captures the :class:`matplotlib.axes.Axes` produced
    by ``plt.subplots`` via a monkeypatch, then asserts
    ``ax.get_xlim() == (lo, hi)`` after :meth:`render_curve`
    returns. This is a tighter test than checking ``_image`` —
    it exercises the actual user-visible axis range, which is
    what the v0.3 roadmap promises.
    """
    # Capture the real subplots *before* the monkeypatch so the
    # wrapper below can call it without recursing.
    real_subplots = _plt.subplots
    captured: dict[str, object] = {}

    def fake_subplots(*args: object, **kwargs: object) -> tuple[object, object]:
        """Wrap ``plt.subplots`` to grab the returned Axes for assertion."""
        fig, ax = real_subplots(*args, **kwargs)
        captured["ax"] = ax
        return fig, ax

    monkeypatch.setattr(_plt, "subplots", fake_subplots)

    lo, hi = 20.0, 180.0
    plot = ForceExtensionPlot(width=40, height=10)
    plot.render_curve(synthetic_force_curve, x_range=(lo, hi))

    ax = captured["ax"]
    assert ax is not None
    # ``get_xlim`` returns the autoscaled limits — for a manually
    # set range with no out-of-range data being drawn, it should
    # be exactly the requested (lo, hi) up to matplotlib's
    # internal margin handling. Allow a small tolerance for the
    # latter.
    x_lo, x_hi = ax.get_xlim()  # type: ignore[attr-defined]
    assert abs(x_lo - lo) < 1e-6
    assert abs(x_hi - hi) < 1e-6


# -- render() placeholder behaviour --------------------------------------


def test_render_placeholder_when_no_data() -> None:
    """Before any ``render_curve`` call, ``render()`` returns a Group with the placeholder.

    The placeholder is the literal string ``"no data"`` — we check
    the renderable is a rich :class:`rich.console.Group` so the
    :meth:`render` contract (which Textual calls) is honoured.
    """
    plot = ForceExtensionPlot(width=40, height=10, title="hello")
    out = plot.render()
    assert isinstance(out, _RichGroup)
    # The Group has at least the title (since we passed one) plus
    # the "no data" placeholder.
    assert len(out.renderables) >= 2  # type: ignore[attr-defined]


# -- ImportError when matplotlib is missing ------------------------------


def test_render_curve_raises_import_error_when_matplotlib_missing(
    monkeypatch: pytest.MonkeyPatch, synthetic_force_curve: ForceCurve
) -> None:
    """When matplotlib is unavailable, ``render_curve`` raises ``ImportError``.

    The brief allows either of two patterns: skip the test when
    matplotlib is missing (covered by the module-level
    ``importorskip``), or simulate the missing-import scenario via
    ``monkeypatch``. We do the latter so the test exercises the
    *real* user-facing error path — the install hint and the
    re-raise chain — rather than a no-op skip.
    """
    # Patch the module's import-error sentinel to simulate the
    # "matplotlib not installed" case without actually uninstalling
    # the package (which would break every other test in this file).
    fake_err = ImportError("No module named 'matplotlib' (simulated)")
    monkeypatch.setattr(plot_mod, "_MPL_IMPORT_ERROR", fake_err)

    plot = ForceExtensionPlot(width=40, height=10)
    with pytest.raises(ImportError) as excinfo:
        plot.render_curve(synthetic_force_curve)
    # The error message must point the user at the [plot] extra —
    # a bare ImportError would leave them wondering which package
    # to install.
    msg = str(excinfo.value)
    assert "matplotlib" in msg
    assert "[plot]" in msg or "afmkit[plot]" in msg
    # And the chained cause is the simulated ImportError, so the
    # original error is preserved in the traceback for diagnostics.
    assert excinfo.value.__cause__ is fake_err
    # The widget must NOT have cached a half-built image on failure.
    assert plot._image is None


# -- Native terminal image renderable (v0.6+ / [plot-native] extra) ----
#
# The :class:`ForceExtensionPlot` widget picks between the v0.4
# half-block :class:`_PillowImageRenderable` and the v0.6+ native
# terminal-image renderable (Sixel / Kitty / iTerm graphics via the
# ``textual-image`` package, the new ``[plot-native]`` extra).  The
# choice is controlled by the ``prefer_native`` constructor
# argument; these tests exercise the dispatch logic.


class TestNativeImageRenderable:
    """Tests for the ``[plot-native]`` extra integration."""

    def test_widget_prefer_native_type_guard(self) -> None:
        """``prefer_native`` only accepts ``None``, ``True``, ``False``.

        ``int`` values are tolerated (``1 == True`` and ``0 == False``
        in Python) so the guard accepts them as the obvious bool
        equivalent; only true non-bool/non-None inputs (strings,
        lists, custom objects) are rejected.
        """
        with pytest.raises(TypeError, match="prefer_native"):
            ForceExtensionPlot(width=40, height=10, prefer_native="yes")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="prefer_native"):
            ForceExtensionPlot(width=40, height=10, prefer_native=[])  # type: ignore[arg-type]

    def _make_pillow_image(self, synthetic_force_curve: ForceCurve) -> Any:
        """Render a small Pillow image from a synthetic curve for dispatch tests."""
        plot = ForceExtensionPlot(width=8, height=4)
        return plot._render_to_pillow(
            curve=synthetic_force_curve,
            peaks=None,
            fit=None,
            x_range=None,
            width_px=80,
            height_px=40,
        )

    def test_choose_renderable_force_half_block(self, synthetic_force_curve: ForceCurve) -> None:
        """``prefer_native=False`` always returns the half-block renderable,
        even when ``textual-image`` is installed.
        """
        image = self._make_pillow_image(synthetic_force_curve)
        # ``prefer_native=False`` ignores the import flag and returns
        # the v0.4 half-block path regardless.
        renderable = plot_mod._choose_renderable(image, prefer_native=False)
        assert isinstance(renderable, plot_mod._PillowImageRenderable)

    def test_choose_renderable_force_native_missing_package(
        self, monkeypatch: pytest.MonkeyPatch, synthetic_force_curve: ForceCurve
    ) -> None:
        """``prefer_native=True`` raises :class:`ImportError` with a
        clear install hint when ``textual-image`` is missing.
        """
        # Simulate the "textual-image not installed" case without
        # actually uninstalling the package (which would break
        # other tests in this module that use ``importorskip``).
        fake_err = ImportError("No module named 'textual_image' (simulated)")
        monkeypatch.setattr(plot_mod, "_native_renderable_module", None)
        monkeypatch.setattr(plot_mod, "_NATIVE_IMPORT_ERROR", fake_err)

        image = self._make_pillow_image(synthetic_force_curve)
        with pytest.raises(ImportError) as excinfo:
            plot_mod._choose_renderable(image, prefer_native=True)
        # The error message must point the user at the
        # ``[plot-native]`` extra — a bare ImportError would
        # leave them wondering which package to install.
        msg = str(excinfo.value)
        assert "textual-image" in msg or "[plot-native]" in msg
        # And the chained cause is the simulated ImportError so
        # the original error is preserved for diagnostics.
        assert excinfo.value.__cause__ is fake_err

    def test_choose_renderable_auto_with_native_installed(
        self, monkeypatch: pytest.MonkeyPatch, synthetic_force_curve: ForceCurve
    ) -> None:
        """``prefer_native=None`` with ``textual-image`` installed
        returns a native-image renderable (not the half-block fallback).
        """

        # Simulate the "textual-image is installed" case with a
        # stub module that exposes an ``Image`` class.  The dispatch
        # logic only checks that ``_native_renderable_module`` is
        # not None and calls ``_native_renderable_module.Image(pil_image)``,
        # so a trivial stub is enough to exercise the auto-detect
        # path.
        class _StubImage:
            def __init__(self, pil_image: object) -> None:
                self.pil_image = pil_image

        class _StubModule:
            Image = _StubImage

        monkeypatch.setattr(plot_mod, "_native_renderable_module", _StubModule())
        monkeypatch.setattr(plot_mod, "_NATIVE_IMPORT_ERROR", None)

        image = self._make_pillow_image(synthetic_force_curve)
        renderable = plot_mod._choose_renderable(image, prefer_native=None)
        # Not the half-block fallback.
        assert not isinstance(renderable, plot_mod._PillowImageRenderable)
        assert isinstance(renderable, _StubImage)
        # And the underlying PIL image is the one we passed in.
        assert renderable.pil_image is image  # type: ignore[attr-defined]

    def test_choose_renderable_auto_without_native(
        self, monkeypatch: pytest.MonkeyPatch, synthetic_force_curve: ForceCurve
    ) -> None:
        """``prefer_native=None`` without ``textual-image`` falls back
        to the half-block renderable (the v0.4 default contract).
        """
        monkeypatch.setattr(plot_mod, "_native_renderable_module", None)
        monkeypatch.setattr(
            plot_mod,
            "_NATIVE_IMPORT_ERROR",
            ImportError("simulated missing package"),
        )

        image = self._make_pillow_image(synthetic_force_curve)
        renderable = plot_mod._choose_renderable(image, prefer_native=None)
        assert isinstance(renderable, plot_mod._PillowImageRenderable)

    def test_widget_prefer_native_true_raises_when_missing(
        self, monkeypatch: pytest.MonkeyPatch, synthetic_force_curve: ForceCurve
    ) -> None:
        """``ForceExtensionPlot(prefer_native=True).render()`` raises
        a clear :class:`ImportError` when ``[plot-native]`` is missing
        *and* there is data to render.  Constructing the widget does
        not raise — only the render call does, so the App can mount
        the widget and fail at the first render tick instead of
        crashing at import time.
        """
        monkeypatch.setattr(plot_mod, "_native_renderable_module", None)
        monkeypatch.setattr(
            plot_mod,
            "_NATIVE_IMPORT_ERROR",
            ImportError("simulated missing package"),
        )

        plot = ForceExtensionPlot(width=40, height=10, prefer_native=True)
        # Construct succeeds.
        plot.render_curve(synthetic_force_curve)  # populates _image
        assert plot._image is not None
        # But render() raises with a clear install hint.
        with pytest.raises(ImportError) as excinfo:
            plot.render()
        assert "textual-image" in str(excinfo.value) or "[plot-native]" in str(excinfo.value)

    def test_widget_prefer_native_false_always_half_block(
        self, monkeypatch: pytest.MonkeyPatch, synthetic_force_curve: ForceCurve
    ) -> None:
        """``ForceExtensionPlot(prefer_native=False).render()`` returns
        a half-block renderable, even when ``[plot-native]`` is
        installed.  This is the explicit "force the v0.4 path" opt-out.
        """
        # Even with native available, the user said "no native" — we
        # must respect that and fall back to half-block.  Track
        # whether the stub ``Image`` was ever called.
        stub_calls: list[object] = []

        class _StubImage:
            def __init__(self, pil_image: object) -> None:
                stub_calls.append(pil_image)

        class _StubModule:
            Image = _StubImage

        monkeypatch.setattr(plot_mod, "_native_renderable_module", _StubModule())
        monkeypatch.setattr(plot_mod, "_NATIVE_IMPORT_ERROR", None)

        plot = ForceExtensionPlot(width=40, height=10, prefer_native=False)
        plot.render_curve(synthetic_force_curve)
        out = plot.render()
        assert isinstance(out, _RichGroup)
        # Walk the Group to find the image renderable; it must be
        # the half-block one, not the stub.
        renderables = out.renderables  # type: ignore[attr-defined]
        image_renderables = [
            r for r in renderables if isinstance(r, plot_mod._PillowImageRenderable)
        ]
        assert len(image_renderables) == 1
        # And the stub's __init__ must not have been called.
        assert stub_calls == []
