"""Unit tests for the :mod:`afmkit.presentation.gui` Textual TUI.

Textual apps are hard to unit-test in isolation — they need a
running asyncio event loop and a terminal. We deliberately keep
this module tiny and focus on the things we *can* test cheaply:

- The module imports cleanly (the import chain is valid).
- :class:`AFMkitApp` is a real :class:`textual.app.App` subclass.
- The keybindings expose the 4 documented actions (``o``, ``f``,
  ``e``, ``q``).
- The ``gui`` CLI subcommand is wired up and reaches the
  ``AFMkitApp.run()`` call without raising (we monkeypatch
  ``run`` so the test doesn't actually start the event loop).

End-to-end coverage of the TUI (modal screen, data table, fit
action) lives in the manual smoke-test recipe in
``docs/gui-usage.md`` rather than here — a full TUI test would
need a headless pilot harness, which is overkill for v0.2.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

# Textual is an optional [gui] extra; skip the whole module cleanly
# when it isn't installed so the test matrix doesn't break. The
# import must come before the afmkit imports so the skip is honoured
# when textual is missing.
textual = pytest.importorskip("textual")

from afmkit.presentation.cli import app as cli_app  # noqa: E402
from afmkit.presentation.gui.app import AFMkitApp  # noqa: E402  (after importorskip)

# CliRunner is shared across cases so we don't rebuild the capture
# buffers in every test.
runner = CliRunner()


# -- Module & class shape -------------------------------------------------


def test_gui_module_imports() -> None:
    """The ``afmkit.presentation.gui.app`` module imports without errors.

    This is the cheapest possible "is the code at least syntactically
    valid and free of circular imports" check; the heavyweight
    attribute tests below build on it.
    """
    import afmkit.presentation.gui.app as gui_app

    assert gui_app is not None


def test_afmkit_app_is_textual_app_subclass() -> None:
    """``AFMkitApp`` is a real :class:`textual.app.App` subclass.

    Confirms the optional-dep shim resolved to the live Textual
    implementation rather than the ``ImportError``-raising
    placeholder defined when textual is missing.
    """
    from textual.app import App

    assert issubclass(AFMkitApp, App)


def test_bindings_cover_v0_3_surface() -> None:
    """The documented v0.3 keybindings are registered.

    v0.2 had four (``o``, ``f``, ``e``, ``q``); v0.3 adds two more
    for the peak-review and plot-panel features (``P`` shift-p for
    the plot, ``p`` for the peak-review toggle). Missing any of
    these is a regression that would block the documented workflow.
    """
    binding_keys = {b.key for b in AFMkitApp.BINDINGS}
    assert binding_keys == {"o", "f", "e", "q", "P", "p"}


def test_bindings_actions_resolve_to_methods() -> None:
    """Each binding's action string maps to a real ``action_*`` method.

    Textual resolves ``Binding("o", "open_dir", ...)`` to
    ``self.action_open_dir()`` at dispatch time. The v0.3 action
    names here must each be present, and most must have matching
    methods on :class:`AFMkitApp`. (``quit`` is a built-in Textual
    action, so no matching method is required.)
    """
    expected_actions = {
        "open_dir",
        "fit_selected",
        "export_csv",
        "toggle_plot",
        "toggle_review",
        "quit",
    }
    found_actions = {b.action for b in AFMkitApp.BINDINGS}
    assert expected_actions <= found_actions

    # The three custom actions must have matching methods.
    for action in ("open_dir", "fit_selected", "export_csv"):
        assert hasattr(AFMkitApp, f"action_{action}"), f"AFMkitApp is missing action_{action}()"


def test_gui_subcommand_does_not_crash_on_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invoking ``afmkit gui`` resolves and reaches ``AFMkitApp.run()``.

    The Textual event loop is real — calling ``App.run()`` from
    CliRunner would block on terminal I/O for the lifetime of the
    TUI, which is not what unit tests are for. We monkeypatch
    :meth:`AFMkitApp.run` to a no-op and assert that the CLI
    subcommand resolves, the import path inside the body succeeds,
    and ``run()`` is actually called. That covers everything the
    CLI-level test can cover without driving a real TUI session.
    """
    calls: list[None] = []

    def fake_run(self: AFMkitApp) -> None:
        """Record that the CLI body reached the ``run()`` call."""
        calls.append(None)

    monkeypatch.setattr(AFMkitApp, "run", fake_run)

    result = runner.invoke(cli_app, ["gui"], catch_exceptions=False)

    # The CLI body must have called the (monkeypatched) run() —
    # i.e. the import path resolved and execution reached the
    # ``AFMkitApp().run()`` line. If any step before that had
    # raised, ``calls`` would be empty.
    assert calls, (
        f"afmkit gui subcommand did not reach AFMkitApp.run(); " f"output={result.output!r}"
    )
    # And nothing in the CLI / import chain raised an unhandled
    # exception that bubbled out of the runner.
    assert result.exception is None or isinstance(result.exception, SystemExit)


# -- v0.4 plot-panel integration -----------------------------------------


#: pytest.importorskip gate at module top already pulled in textual.
#: The v0.4 test surface for the plot-panel integration depends on
#: the matplotlib widget, so we import-skip it as a second gate.
plot_mod = pytest.importorskip("afmkit.presentation.gui.plot")
pytest.importorskip("matplotlib")
pytest.importorskip("PIL")

from afmkit.presentation.gui.plot import ForceExtensionPlot  # noqa: E402


def test_plot_widget_class_is_cached_on_init() -> None:
    """``__init__`` pre-imports the matplotlib widget class.

    The class is cached so ``compose()`` can ``yield`` an actual
    :class:`ForceExtensionPlot` instance without a per-render
    import dance. When the ``[plot]`` extra is missing, the
    cache holds ``None`` and the import error string is set.
    """
    app = AFMkitApp()
    # The widget class is importable in this test env, so the cache
    # must point at the real class — not the textual Widget base,
    # not a placeholder.
    assert app._plot_widget_cls is ForceExtensionPlot
    assert app._plot_widget_import_error == ""


def test_render_plot_calls_widget_render_curve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_render_plot`` renders onto the mounted widget, not a throwaway.

    v0.3 created a fresh :class:`ForceExtensionPlot` on every
    call and discarded the rendered image, only surfacing a
    textual summary. v0.4 must call :meth:`render_curve` on the
    widget that :meth:`compose` already mounted, so the half-block
    image is what actually shows up in the TUI.
    """
    # Build the app and stub out the Textual surface it would
    # otherwise need to be mounted. ``_render_plot`` only calls
    # ``self.query_one`` and ``self._set_status`` plus the widget's
    # ``render_curve`` — we replace the first two with mocks.
    app = AFMkitApp()
    app._show_plot = True
    # One synthetic curve so the render path has data to plot.
    from afmkit.core.curve import ForceCurve

    ext = np.linspace(0.0, 300.0, 101)
    force = np.linspace(0.0, 5.0, 101) + 0.01 * np.arange(101)
    curve = ForceCurve(ext, force, metadata={"k_cantilever": 0.1})
    app._curves = [curve]
    app._current_curve_idx = 0
    app._last_fit = None
    app._reviewer = None

    # Mock the plot widget that compose() would have mounted.
    render_calls: list[dict[str, Any]] = []

    class _StubWidget:
        def render_curve(
            self,
            curve: Any,
            *,
            peaks: Any = None,
            fit: Any = None,
            x_range: Any = None,
        ) -> None:
            render_calls.append(
                {
                    "curve": curve,
                    "peaks": peaks,
                    "fit": fit,
                    "x_range": x_range,
                }
            )

    def fake_query_one(selector: str, *_args: Any, **_kwargs: Any) -> Any:
        assert selector == "#plot-widget", f"unexpected query_one selector: {selector}"
        return _StubWidget()

    monkeypatch.setattr(app, "query_one", fake_query_one)
    status_calls: list[str] = []

    def fake_set_status(self: AFMkitApp, msg: str) -> None:
        status_calls.append(msg)

    monkeypatch.setattr(AFMkitApp, "_set_status", fake_set_status)

    app._render_plot()

    # The widget's render_curve must have been called once, with
    # the curve, no peaks, and no fit (we did not set _last_fit).
    assert len(render_calls) == 1
    call = render_calls[0]
    assert call["curve"] is curve
    assert call["peaks"] is None
    assert call["fit"] is None
    # The status line should mention the rendered curve.
    assert any("plot rendered" in m for m in status_calls), status_calls


def test_render_plot_passes_reviewed_peaks_and_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_render_plot`` plumbs the reviewer's peaks and the last fit through.

    Both pieces of state are v0.3 additions and the TUI must
    surface them on the plot when present. This test confirms
    the per-peak list (from the reviewer) and the FitResult
    (from the most recent fit action) make it to the widget.
    """
    from afmkit.analysis.peak_detection import Peak
    from afmkit.analysis.peak_review import PeakReviewer
    from afmkit.core.curve import ForceCurve
    from afmkit.io.exporters import FitResult

    app = AFMkitApp()
    app._show_plot = True

    ext = np.linspace(0.0, 300.0, 101)
    force = np.linspace(0.0, 5.0, 101)
    curve = ForceCurve(ext, force, metadata={"k_cantilever": 0.1})
    app._curves = [curve]
    app._current_curve_idx = 0
    fit = FitResult(
        model_name="WLC",
        params={"p": 0.42, "L": 200.0},
        param_stderr={"p": 0.01, "L": 1.0},
        chi2=1.1,
        redchi=0.4,
        n_data=500,
    )
    app._last_fit = fit
    peaks = [
        Peak(
            index=0,
            extension=50.0,
            force=20.0,
            prominence=10.0,
            width=5,
            height_drop=5.0,
            confidence=0.6,
        ),
        Peak(
            index=1,
            extension=120.0,
            force=35.0,
            prominence=12.0,
            width=4,
            height_drop=6.0,
            confidence=0.7,
        ),
    ]
    app._reviewer = PeakReviewer(peaks, curve)

    render_calls: list[dict[str, Any]] = []

    class _StubWidget:
        def render_curve(
            self,
            curve: Any,
            *,
            peaks: Any = None,
            fit: Any = None,
            x_range: Any = None,
        ) -> None:
            render_calls.append({"curve": curve, "peaks": peaks, "fit": fit, "x_range": x_range})

    monkeypatch.setattr(
        app,
        "query_one",
        lambda *_a, **_kw: _StubWidget(),
    )
    monkeypatch.setattr(AFMkitApp, "_set_status", lambda self, m: None)

    app._render_plot()

    assert len(render_calls) == 1
    call = render_calls[0]
    assert call["curve"] is curve
    # The reviewer's auto-detected peaks (not the ReviewedPeak wrappers)
    # must be passed through. Order matches the reviewer's list order.
    assert call["peaks"] == peaks
    assert call["fit"] is fit


def test_render_plot_skips_when_plot_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_render_plot`` is a no-op when the plot panel is hidden.

    The user only sees the plot when the ``P`` key has toggled
    the panel on. ``_render_plot`` is called from the fit action
    even when the panel is hidden (so the cached image is fresh
    the moment the user toggles the panel on), so it must
    short-circuit cleanly without touching the widget.
    """
    app = AFMkitApp()
    app._show_plot = False
    app._curves = []  # no data; would normally bail with status message

    query_calls: list[str] = []

    def fake_query_one(*_a: Any, **_kw: Any) -> Any:
        query_calls.append("called")
        raise AssertionError("query_one must not be called when plot is hidden")

    monkeypatch.setattr(app, "query_one", fake_query_one)
    app._render_plot()
    assert query_calls == []  # never called
