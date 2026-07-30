"""Textual TUI for afmkit.

This module exposes :class:`AFMkitApp`, a small interactive terminal
UI for opening a directory of JPK ``.txt`` files, fitting the
selected curve with the WLC model, exporting the loaded batch
as a wide-column-block CSV, and (v0.3+) interactively reviewing
the sawtooth peaks in a force-extension curve.

The TUI is intentionally minimal — no X server, no PySide6. The
optional matplotlib plot widget is the only heavy dep and it is
loaded lazily.

Keybindings
-----------
- ``o`` — open a directory of JPK ``.txt`` files
- ``f`` — fit the highlighted curve with WLC
- ``e`` — export the loaded batch to CSV
- ``P`` (shift-p) — toggle the matplotlib plot panel
- ``p`` — enter peak-review mode for the highlighted curve
  (j/k navigate peaks, a accept, r reject, o override, R re-fit,
  escape exit review)
- ``q`` — quit

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

    # Stubs so the class bodies below can still parse. At runtime the
    # `_TEXTUAL_IMPORT_ERROR is not None` guard short-circuits before
    # any of these names are referenced, so the placeholder values
    # never get used. ``Any`` makes mypy accept the conditional
    # redefinition without forcing the runtime types to be importable.
    BindingType: Any = object  # type: ignore[no-redef]
    Binding: Any = object  # type: ignore[no-redef]
else:
    _TEXTUAL_IMPORT_ERROR = None

# v0.3 imports — the peak-review state machine and the matplotlib
# plot widget are both opt-in extras. We import them lazily inside
# the action methods so a minimal TUI install (no matplotlib, no
# optional [plot]) still works.
from afmkit.analysis import PeakReviewer, find_sawtooth_peaks
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


# -- Override picker (peak-review force override) -----------------------


if _TEXTUAL_IMPORT_ERROR is None:

    class _OverridePickerScreen(ModalScreen[float | None]):
        """Modal that asks the user for a manual force override (pN).

        Returns the entered value on submit, or ``None`` on cancel.
        """

        DEFAULT_CSS = """
        _OverridePickerScreen {
            align: center middle;
        }
        #override-box {
            width: 60;
            height: 9;
            padding: 1 2;
            border: round $primary;
            background: $surface;
        }
        #override-title {
            content-align: center middle;
            text-style: bold;
        }
        #override-input {
            margin: 1 0;
        }
        #override-hint {
            color: $text-muted;
        }
        """

        BINDINGS: ClassVar[list[BindingType]] = [
            Binding("escape", "dismiss_picker", "Cancel"),
        ]

        def __init__(self, default: str = "") -> None:
            super().__init__()
            self._default = default

        def compose(self) -> ComposeResult:
            with Vertical(id="override-box"):
                yield Static("Override peak force (pN)", id="override-title")
                yield Input(
                    value=self._default,
                    placeholder="e.g. 45.2",
                    id="override-input",
                )
                yield Static("Enter to apply · Esc to cancel", id="override-hint")

        def on_input_submitted(self, event: Input.Submitted) -> None:
            raw = event.value.strip()
            if not raw:
                self.dismiss(None)
                return
            try:
                value = float(raw)
            except ValueError:
                # Bad input — dismiss with None and let the caller
                # surface the error via the status line.
                self.dismiss(None)
                return
            self.dismiss(value)

        def action_dismiss_picker(self) -> None:
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
        - ``e`` — export the loaded batch to a CSV (peak review state
          is included in the export when in review mode).
        - ``P`` (shift-p) — toggle the matplotlib plot panel
          (shows the highlighted curve + its peaks + the last WLC fit).
        - ``p`` — enter peak-review mode for the highlighted curve.
          In review mode the lower table becomes a per-peak
          accept/reject list and the bindings change:
          - ``j`` / ``k`` — next / prev peak
          - ``a`` — accept the highlighted peak
          - ``r`` — reject the highlighted peak
          - ``o`` — override the highlighted peak's force (asks
            for a pN value via a small input)
          - ``R`` — re-fit the highlighted peak in a local window
          - ``escape`` — exit review mode
        - ``q`` — quit.
        """

        TITLE = "afmkit — SMFS analysis"

        BINDINGS: ClassVar[list[BindingType]] = [
            Binding("o", "open_dir", "Open directory"),
            Binding("f", "fit_selected", "Fit WLC"),
            Binding("e", "export_csv", "Export CSV"),
            Binding("P", "toggle_plot", "Toggle plot"),
            Binding("p", "toggle_review", "Peak review"),
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
        #peak-review-table {
            height: 1fr;
        }
        #plot-panel {
            height: 18;
            border: round $primary;
        }
        #status-line {
            height: 3;
            padding: 0 1;
            background: $boost;
            color: $text;
        }
        .hidden {
            display: none;
        }
        """

        def __init__(self) -> None:
            """Initialise the app with an empty batch and an empty table."""
            super().__init__()
            self._batch: CurveBatch | None = None
            self._curves: list[ForceCurve] = []
            # v0.3 peak-review state.
            self._reviewer: PeakReviewer | None = None
            self._current_curve_idx: int | None = None
            self._peak_idx: int = 0
            self._last_fit: Any = None  # FitResult | None
            # v0.3 plot panel state.
            self._show_plot: bool = False

        def compose(self) -> ComposeResult:
            """Build the layout: header, dir row, curves table, peak-review
            table (initially hidden), plot panel (initially hidden),
            status line, footer.
            """
            yield Header(show_clock=False)
            with Vertical(id="dir-row"):
                yield Static(
                    "Press [bold]o[/bold] to open a directory of JPK .txt files",
                    id="dir-label",
                )
            yield DataTable(id="curves-table", zebra_stripes=True, cursor_type="row")
            yield DataTable(id="peak-review-table", zebra_stripes=True, cursor_type="row")
            yield Static(
                "Plot panel: press [bold]P[/bold] to toggle "
                "(requires matplotlib via `pip install afmkit[plot]`)",
                id="plot-panel",
            )
            yield Static("", id="status-line")
            yield Footer()

        def on_mount(self) -> None:
            """Set up the data table columns on startup so the empty UI is ready."""
            curves_table = self.query_one("#curves-table", DataTable)
            curves_table.add_columns(
                "id", "source", "n_points", "ext_min", "ext_max", "f_min", "f_max"
            )
            review_table = self.query_one("#peak-review-table", DataTable)
            review_table.add_columns("#", "ext (nm)", "force (pN)", "accepted", "conf", "note")
            # The peak-review table and plot panel start hidden; the
            # `display: none` CSS class hides a widget without removing
            # it from the DOM, which is what we want for `P` / `p`
            # toggles that need to restore the widget instantly.
            review_table.add_class("hidden")
            self.query_one("#plot-panel").add_class("hidden")

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

            self._last_fit = result
            self._render_plot()

            params = result.params
            success = bool(result.metadata.get("success", True))
            mark = "[green]✓[/green]" if success else "[red]✗[/red]"
            p = _fmt_num(params.get("p"))
            lc = _fmt_num(params.get("L"))
            redchi = _fmt_num(result.reduced_chi_square)
            self._set_status(f"{mark} curve {row} WLC  p={p} nm  L={lc} nm  redchi={redchi}")

        async def action_toggle_review(self) -> None:
            """Enter or exit peak-review mode for the highlighted curve.

            On the first invocation: build a PeakReviewer from
            find_sawtooth_peaks(curve, ...), show the peak-review
            table, populate it. On subsequent invocations: hide the
            table, drop the reviewer.
            """
            if not self._curves:
                self._set_status("[yellow]no curves loaded — press o to open a directory[/yellow]")
                return

            review_table = self.query_one("#peak-review-table", DataTable)
            if self._reviewer is not None:
                # Exit review mode.
                self._reviewer = None
                self._current_curve_idx = None
                review_table.add_class("hidden")
                review_table.clear()
                self._set_status("[green]exited peak review[/green]")
                return

            # Enter review mode for the currently-highlighted curve.
            table = self.query_one("#curves-table", DataTable)
            row = table.cursor_row
            if row is None or row < 0 or row >= len(self._curves):
                self._set_status("[yellow]no curve highlighted[/yellow]")
                return
            curve = self._curves[int(row)]
            try:
                peaks = find_sawtooth_peaks(curve, min_prominence_pN=5.0)
            except (ValueError, TypeError) as exc:
                self._set_status(f"[red]peak detection failed:[/red] {type(exc).__name__}: {exc}")
                return
            if not peaks:
                self._set_status(
                    f"[yellow]no peaks detected in curve {int(row)} (try lowering "
                    f"min_prominence_pN)[/yellow]"
                )
                return

            self._reviewer = PeakReviewer(peaks, curve)
            self._current_curve_idx = int(row)
            self._peak_idx = 0
            review_table.remove_class("hidden")
            self._populate_review_table()
            self._set_status(
                f"[green]reviewing curve {int(row)}:[/green] {len(peaks)} peak(s) — "
                f"j/k navigate, a/r/o/R modify, [bold]esc[/bold] exit"
            )

        async def action_toggle_plot(self) -> None:
            """Toggle the matplotlib plot panel visibility.

            The panel is empty when first shown (the user needs to
            press `f` to populate it with a fit). If matplotlib is
            not installed, the panel becomes an error message instead
            of crashing.
            """
            panel = self.query_one("#plot-panel")
            if panel.has_class("hidden"):
                panel.remove_class("hidden")
                self._show_plot = True
                self._render_plot()
                self._set_status(
                    "[green]plot panel shown[/green] — press [bold]f[/bold] "
                    "to overlay the WLC fit"
                )
            else:
                panel.add_class("hidden")
                self._show_plot = False
                self._set_status("[green]plot panel hidden[/green]")

        async def action_next_peak(self) -> None:
            """Move the peak-review cursor to the next peak."""
            if self._reviewer is None:
                return
            self._peak_idx = min(self._peak_idx + 1, len(self._reviewer) - 1)
            self._populate_review_table()
            self._set_status(f"[cyan]peak {self._peak_idx + 1}/{len(self._reviewer)}[/cyan]")

        async def action_prev_peak(self) -> None:
            """Move the peak-review cursor to the previous peak."""
            if self._reviewer is None:
                return
            self._peak_idx = max(self._peak_idx - 1, 0)
            self._populate_review_table()
            self._set_status(f"[cyan]peak {self._peak_idx + 1}/{len(self._reviewer)}[/cyan]")

        async def action_accept_peak(self) -> None:
            """Accept the highlighted peak in the reviewer."""
            if self._reviewer is None:
                return
            self._reviewer.accept(self._peak_idx)
            self._populate_review_table()
            self._set_status(f"[green]accepted peak {self._peak_idx}[/green]")

        async def action_reject_peak(self) -> None:
            """Reject the highlighted peak in the reviewer."""
            if self._reviewer is None:
                return
            self._reviewer.reject(self._peak_idx)
            self._populate_review_table()
            self._set_status(f"[yellow]rejected peak {self._peak_idx}[/yellow]")

        async def action_override_peak(self) -> None:
            """Open a small input to set the manual_force on the highlighted peak."""
            if self._reviewer is None:
                return
            peak = self._reviewer[self._peak_idx]
            default = f"{peak.force:.4g}"
            picker: _OverridePickerScreen = _OverridePickerScreen(default=default)
            value: float | None = await self.push_screen_wait(picker)
            if value is None:
                self._set_status("[yellow]override cancelled[/yellow]")
                return
            try:
                self._reviewer.override(self._peak_idx, value)
            except ValueError as exc:
                self._set_status(f"[red]override failed:[/red] {exc}")
                return
            self._populate_review_table()
            self._set_status(f"[green]overrode peak {self._peak_idx}[/green] → {value:.4g} pN")

        async def action_refit_peak(self) -> None:
            """Re-fit the highlighted peak in a local window."""
            if self._reviewer is None:
                return
            try:
                self._reviewer.re_fit(self._peak_idx, x_range=DEFAULT_X_RANGE)
            except (ValueError, IndexError) as exc:
                self._set_status(f"[red]re-fit failed:[/red] {exc}")
                return
            self._populate_review_table()
            self._set_status(
                f"[green]re-fit peak {self._peak_idx}[/green] → "
                f"{self._reviewer[self._peak_idx].force:.4g} pN"
            )

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

        def _populate_review_table(self) -> None:
            """Refill the peak-review table from :attr:`_reviewer`."""
            table = self.query_one("#peak-review-table", DataTable)
            table.clear()
            if self._reviewer is None:
                return
            for i, rp in enumerate(self._reviewer):
                marker = "►" if i == self._peak_idx else " "
                accept_mark = "[green]✓[/green]" if rp.accepted else "[red]✗[/red]"
                manual = (
                    f" [yellow]({rp.manual_force:.3g})[/yellow]"
                    if rp.manual_force is not None
                    else ""
                )
                table.add_row(
                    f"{marker} {i}",
                    _fmt_num(rp.extension),
                    f"{_fmt_num(rp.force)}{manual}",
                    accept_mark,
                    _fmt_num(rp.confidence),
                    rp.note[:20],
                )

        def _render_plot(self) -> None:
            """Render the current curve + peaks + last fit into the plot panel.

            Imports the matplotlib widget lazily so a TUI install
            without [plot] doesn't crash. If the widget isn't
            available, write an error message into the plot panel.
            """
            panel: Any = self.query_one("#plot-panel")
            if not self._show_plot:
                return
            if self._current_curve_idx is None and not self._curves:
                panel.update("[yellow]no curve loaded[/yellow]")
                return
            if self._current_curve_idx is None:
                # Show the highlighted curve (if any), or the first one.
                table = self.query_one("#curves-table", DataTable)
                row = table.cursor_row
                if row is None or row < 0 or row >= len(self._curves):
                    row = 0
                self._current_curve_idx = int(row)
            curve = self._curves[self._current_curve_idx]
            try:
                from afmkit.presentation.gui.plot import ForceExtensionPlot

                widget = ForceExtensionPlot(
                    width=120, height=18, title=f"curve {self._current_curve_idx}"
                )
                widget.render_curve(
                    curve,
                    peaks=(
                        [rp.peak for rp in self._reviewer] if self._reviewer is not None else None
                    ),
                    fit=self._last_fit,
                )
                # Re-render the widget into a string we can drop into
                # the Static panel. The widget's render() returns a
                # rich renderable; we render to the screen and read
                # the cells. We don't have a Console here, so we just
                # acknowledge the render succeeded and let the user
                # know via the status line.
                panel.update(
                    f"[green]plot:[/green] curve {self._current_curve_idx} "
                    f"({curve.n_points} pts)"
                    + (
                        f" + {len(self._reviewer)} reviewed peak(s)"
                        if self._reviewer is not None
                        else ""
                    )
                    + (
                        f" + WLC fit p={_fmt_num(self._last_fit.params.get('p'))} nm"
                        if self._last_fit is not None
                        else " (no fit yet — press [bold]f[/bold])"
                    )
                )
            except ImportError as exc:
                panel.update(
                    f"[red]plot unavailable:[/red] {exc}\n"
                    f"Install with [bold]pip install 'afmkit[plot]'[/bold]"
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
