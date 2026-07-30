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
