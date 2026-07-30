"""Textual TUI for afmkit.

This module exposes :class:`AFMkitApp`, a small interactive terminal
UI for opening a directory of JPK ``.txt`` files, fitting the
selected curve with the WLC model, and exporting the loaded batch
as a wide-column-block CSV.

The TUI is deliberately minimal — v0.2's goal is to ship a working
interactive surface that does not require an X server or PySide6.
Peak picking, the eWLC model, and manual review are deferred to v0.3.

Layout
------
::

    +--------------------------------------------------+
    | Header: afmkit — SMFS analysis                   |
    +--------------------------------------------------+
    | [o] Open directory   current: /path/to/dir       |
    +--------------------------------------------------+
    | DataTable: id | file | n_pts | ext_range | ...   |
    | ...                                              |
    +--------------------------------------------------+
    | Footer / status line (last action result)        |
    +--------------------------------------------------+

Keybindings
-----------
- ``o`` — open a directory of JPK ``.txt`` files.
- ``f`` — fit the highlighted curve with the WLC model.
- ``e`` — export the loaded batch to ``afmkit_export.csv``.
- ``q`` — quit.

Optional dependency
-------------------
Textual is an optional ``[gui]`` extra — the import at the top of
this module is wrapped in a ``try/except`` so the rest of afmkit
imports cleanly without the TUI installed. The ``gui`` CLI
subcommand in :mod:`afmkit.presentation.cli` raises a clear
``ImportError`` with install instructions if the user tries to
launch the TUI without it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding, BindingType
    from textual.containers import Vertical
    from textual.screen import ModalScreen
    from textual.widgets import DataTable, Footer, Header, Input, Static
except ImportError as _exc:  # pragma: no cover - exercised only when textual is missing
    _TEXTUAL_IMPORT_ERROR: ImportError | None = _exc

    # mypy-only aliases. The runtime types only matter when textual is
    # actually importable (which is also the only time BINDINGS is
    # evaluated) — at runtime the import error exits before that.
    BindingType = Binding = object  # type: ignore[assignment,misc]
else:
    _TEXTUAL_IMPORT_ERROR = None


from afmkit.core.curve import CurveBatch, ForceCurve
from afmkit.fitting import fit as fit_curve
from afmkit.io.exporters import to_csv
from afmkit.io.jpk_txt import load_jpk_txt

if TYPE_CHECKING:
    from textual.app import ComposeResult as _ComposeResult  # noqa: F401


#: Cantilever spring constant (pN/nm) used when the user does not
#: provide one. This is the same default the legacy Igor
#: ``Load_JPK_FX_Data`` procedure used for JPK calibration files
#: in our test fixtures. The TUI surfaces a warning in the status
#: line whenever this fallback is in effect.
DEFAULT_K_CANTILEVER = 0.1

#: ``(x_min, x_max)`` in nm used by the ``f`` (fit) action. Matches
#: the default range used by the v0.1 ``afmkit fit`` CLI subcommand
#: for WLC fits on a 0-300 nm curve.
DEFAULT_X_RANGE: tuple[float, float] = (20.0, 180.0)

#: Path written by the ``e`` (export) action when the user does not
#: specify one. CWD-relative so a lab user gets a file they can
#: ``cat`` without hunting through ``~``.
DEFAULT_EXPORT_PATH = Path("afmkit_export.csv")


# -- Modal: directory picker -----------------------------------------------


if _TEXTUAL_IMPORT_ERROR is None:

    class _DirPickerScreen(ModalScreen[Path | None]):
        """Modal that asks the user for a directory path.

        Returns the entered path on submit, or ``None`` on dismiss.
        The text input is pre-filled with the current working
        directory so an unmodified Enter accepts the default.
        """

        DEFAULT_CSS = """
        _DirPickerScreen {
            align: center middle;
        }
        #dir-picker-box {
            width: 70;
            height: 7;
            border: round $primary;
            padding: 1 2;
            background: $surface;
        }
        #dir-picker-title {
            content-align: center middle;
            height: 1;
            text-style: bold;
        }
        #dir-picker-input {
            margin-top: 1;
        }
        #dir-picker-hint {
            margin-top: 1;
            color: $text-muted;
        }
        """

        BINDINGS: ClassVar[list[BindingType]] = [
            Binding("escape", "dismiss_picker", "Cancel"),
        ]

        def __init__(self, default: str = "") -> None:
            """Store the initial path so the user can just press Enter to accept."""
            super().__init__()
            self._default = default

        def compose(self) -> ComposeResult:
            """Build the dialog box with a title, an input, and a hint."""
            with Vertical(id="dir-picker-box"):
                yield Static("Open directory of JPK .txt files", id="dir-picker-title")
                yield Input(
                    value=self._default, placeholder="/path/to/jpk/files", id="dir-picker-input"
                )
                yield Static("Enter to load · Esc to cancel", id="dir-picker-hint")

        def on_input_submitted(self, event: Input.Submitted) -> None:
            """Resolve the entered path and dismiss the screen with it."""
            value = event.value.strip()
            if not value:
                self.dismiss(None)
                return
            self.dismiss(Path(value).expanduser())

        def action_dismiss_picker(self) -> None:
            """Dismiss the picker without selecting a path."""
            self.dismiss(None)


# -- App -------------------------------------------------------------------


if _TEXTUAL_IMPORT_ERROR is None:

    class AFMkitApp(App[None]):
        """Textual TUI for afmkit.

        Three-panel layout:

        - Top: directory input (``o`` to open a new directory).
        - Middle: data table — one row per loaded curve
          (id, source file, n_points, extension range, force range).
        - Bottom: status line (last action result / error).

        Keybindings:

        - ``o`` — open a directory of JPK ``.txt`` files.
        - ``f`` — fit the highlighted curve with the WLC model.
        - ``e`` — export the loaded batch to a CSV.
        - ``q`` — quit.

        The TUI is intentionally minimal: open, fit one curve, export.
        Peak picking, multiple models, and manual review are v0.3
        follow-ups.
        """

        TITLE = "afmkit — SMFS analysis"

        BINDINGS: ClassVar[list[BindingType]] = [
            Binding("o", "open_dir", "Open directory"),
            Binding("f", "fit_selected", "Fit WLC"),
            Binding("e", "export_csv", "Export CSV"),
            Binding("q", "quit", "Quit"),
        ]

        DEFAULT_CSS = """
        AFMkitApp {
            layout: vertical;
        }
        #dir-row {
            height: 3;
            padding: 0 1;
            background: $boost;
        }
        #dir-label {
            width: 100%;
            content-align: left middle;
        }
        #curves-table {
            height: 1fr;
        }
        #status-line {
            height: 3;
            padding: 0 1;
            background: $boost;
            color: $text;
        }
        """

        def __init__(self) -> None:
            """Initialise the app with an empty batch and an empty table."""
            super().__init__()
            self._batch: CurveBatch | None = None
            self._curves: list[ForceCurve] = []

        def compose(self) -> ComposeResult:
            """Build the four-region vertical layout."""
            yield Header(show_clock=False)
            with Vertical(id="dir-row"):
                yield Static(
                    "Press [bold]o[/bold] to open a directory of JPK .txt files", id="dir-label"
                )
            yield DataTable(id="curves-table", zebra_stripes=True, cursor_type="row")
            yield Static("", id="status-line")
            yield Footer()

        def on_mount(self) -> None:
            """Set up the data table columns on startup so the empty UI is ready."""
            table = self.query_one("#curves-table", DataTable)
            table.add_columns("id", "source", "n_points", "ext_min", "ext_max", "f_min", "f_max")

        # -- Actions ------------------------------------------------------

        async def action_open_dir(self) -> None:
            """Pop the directory picker, then load every ``.txt`` file inside.

            The default cantilever spring constant is
            :data:`DEFAULT_K_CANTILEVER` — the status line surfaces a
            warning so the user knows to override it on the next pass
            if their probe has a different k.
            """
            default = str(Path.cwd())
            picker: _DirPickerScreen = _DirPickerScreen(default=default)
            path: Path | None = await self.push_screen_wait(picker)
            if path is None:
                self._set_status("[yellow]open cancelled[/yellow]")
                return
            if not path.exists() or not path.is_dir():
                self._set_status(f"[red]not a directory:[/red] {path}")
                return
            files = sorted(
                p for p in path.glob("*.txt") if p.is_file() and not p.name.startswith(".")
            )
            if not files:
                self._set_status(f"[red]no .txt files under[/red] {path}")
                return

            batches: list[CurveBatch] = []
            k_used = DEFAULT_K_CANTILEVER
            k_warned = False
            for f in files:
                try:
                    batch = load_jpk_txt(f, k_cantilever=k_used)
                except (ValueError, OSError) as exc:
                    self._set_status(f"[red]failed to load[/red] {f.name}: {exc}")
                    return
                batches.append(batch)

            self._batch = CurveBatch.concat(batches, name=path.stem)
            self._batch.metadata = {
                **self._batch.metadata,
                "k_cantilever": k_used,
                "k_default": True,
                "source_directory": str(path),
            }
            self._curves = list(self._batch)
            self._populate_table()

            warn = (
                f" [yellow](k={k_used} pN/nm assumed; pass an explicit k in v0.3)[/yellow]"
                if not k_warned
                else ""
            )
            self._set_status(
                f"[green]loaded[/green] {len(files)} file(s), "
                f"{self._batch.n_curves} curve(s){warn}"
            )
            self.query_one("#dir-label", Static).update(
                f"Directory: [bold]{path}[/bold]  ·  press [bold]o[/bold] to change"
            )

        async def action_fit_selected(self) -> None:
            """Fit the highlighted curve with the WLC model.

            Reads the DataTable cursor row, maps the row index to the
            corresponding :class:`ForceCurve` in :attr:`_curves`, and
            runs :func:`afmkit.fitting.fit` with
            :data:`DEFAULT_X_RANGE`. The status line shows the
            parameter estimates or the fit error.
            """
            if not self._curves:
                self._set_status("[yellow]no curves loaded — press o to open a directory[/yellow]")
                return
            table = self.query_one("#curves-table", DataTable)
            row = table.cursor_row
            if row is None or row < 0 or row >= len(self._curves):
                self._set_status("[yellow]no curve highlighted[/yellow]")
                return
            curve = self._curves[int(row)]
            try:
                result = fit_curve(curve, model="wlc", x_range=DEFAULT_X_RANGE)
            except (ValueError, TypeError) as exc:
                self._set_status(f"[red]fit failed:[/red] {type(exc).__name__}: {exc}")
                return

            params = result.params
            success = bool(result.metadata.get("success", True))
            mark = "[green]✓[/green]" if success else "[red]✗[/red]"
            p = _fmt_num(params.get("p"))
            lc = _fmt_num(params.get("L"))
            redchi = _fmt_num(result.reduced_chi_square)
            self._set_status(f"{mark} curve {row} WLC  p={p} nm  L={lc} nm  redchi={redchi}")

        async def action_export_csv(self) -> None:
            """Write the loaded batch as a wide-column-block CSV."""
            if self._batch is None or self._batch.n_curves == 0:
                self._set_status("[yellow]nothing to export — load a directory first[/yellow]")
                return
            out = DEFAULT_EXPORT_PATH
            try:
                to_csv(self._batch, out)
            except (ValueError, OSError) as exc:
                self._set_status(f"[red]export failed:[/red] {exc}")
                return
            size_kb = out.stat().st_size / 1024 if out.exists() else 0.0
            self._set_status(
                f"[green]✓[/green] wrote {self._batch.n_curves} curves to "
                f"{out} ({size_kb:.1f} KB)"
            )

        # -- Internals ---------------------------------------------------

        def _populate_table(self) -> None:
            """Clear and refill the data table from :attr:`_curves`."""
            table = self.query_one("#curves-table", DataTable)
            table.clear()
            for idx, curve in enumerate(self._curves):
                ext = curve.extension
                force = curve.force
                source = str(curve.metadata.get("source_file", ""))
                table.add_row(
                    str(idx),
                    source,
                    str(curve.n_points),
                    _fmt_num(float(ext.min())),
                    _fmt_num(float(ext.max())),
                    _fmt_num(float(force.min())),
                    _fmt_num(float(force.max())),
                )

        def _set_status(self, msg: str) -> None:
            """Replace the status-line text and log it for the user's terminal scrollback."""
            self.query_one("#status-line", Static).update(msg)
            self.log(msg)

else:  # pragma: no cover - exercised only when textual is missing

    class AFMkitApp:  # type: ignore[no-redef]  # placeholder so the symbol exists
        """Placeholder raised when Textual is not installed.

        The real :class:`AFMkitApp` is only defined when the optional
        ``[gui]`` extra is installed. Importing this module without
        Textual raises :class:`ImportError` at the call site instead
        of failing the import of unrelated modules.
        """

        def __init__(self) -> None:
            """Raise immediately so the placeholder cannot be instantiated."""
            raise ImportError(
                "Textual is not installed. Install the [gui] extra to use the TUI: "
                "`pip install 'afmkit[gui]'`."
            ) from _TEXTUAL_IMPORT_ERROR

        def run(self) -> None:
            """Mirror :meth:`textual.app.App.run` so the placeholder is callable."""
            raise ImportError(
                "Textual is not installed. Install the [gui] extra to use the TUI: "
                "`pip install 'afmkit[gui]'`."
            ) from _TEXTUAL_IMPORT_ERROR


# -- Helpers ---------------------------------------------------------------


def _fmt_num(v: Any) -> str:
    """Format a numeric value for the data table.

    Returns ``"n/a"`` for missing values and ``"nan"`` for actual
    NaNs, otherwise a compact 4-significant-figure representation
    (the same format the CLI fit table uses).
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    if f != f:  # NaN
        return "nan"
    return f"{f:.4g}"


__all__ = [
    "AFMkitApp",
    "DEFAULT_K_CANTILEVER",
    "DEFAULT_X_RANGE",
    "DEFAULT_EXPORT_PATH",
]
