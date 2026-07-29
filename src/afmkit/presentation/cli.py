"""Command-line interface for afmkit (typer-based).

This module is the entry point declared in ``pyproject.toml`` under
``[project.scripts]`` — installing afmkit makes the ``afmkit`` shell
command available.

v0.1 commands
-------------
- ``afmkit version`` — print the installed afmkit version.
- ``afmkit info FILE`` — describe a data file (HDF5 / JPK / IBW).
- ``afmkit import SOURCE --output H5 --k FLOAT [--recursive]`` —
  convert a folder of JPK ``.txt`` files into an afmkit HDF5 archive.
- ``afmkit fit SESSION --model wlc --output fits.csv [--x-range MIN MAX]`` —
  fit every curve in an HDF5 archive with the requested model and
  write a per-fit CSV plus a Markdown report.
- ``afmkit export SESSION --format csv|mat|parquet|md --output PATH`` —
  export an HDF5 session to one of the portable formats supported by
  :mod:`afmkit.io.exporters`.

Exit code convention
--------------------
- ``0`` — success.
- ``1`` — user error (file not found, bad format, bad argument).
- ``2`` — internal / unexpected failure (the underlying library
  raised something the user can't fix).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from afmkit import __version__
from afmkit._version import __version__ as _v  # noqa: F401  (sanity)

app = typer.Typer(
    name="afmkit",
    help="afmkit — single-molecule force spectroscopy analysis.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console(width=200)


# -- Exit codes -----------------------------------------------------------


#: User errors (file not found, bad format, missing required option).
EXIT_USER_ERROR = 1
#: Internal / unexpected failures — the user can't fix these from the CLI.
EXIT_INTERNAL = 2


# -- Small helpers --------------------------------------------------------


def _err(msg: str) -> None:
    """Print a red error message to stderr."""
    console.print(f"[red]error:[/red] {msg}")


def _abort_user(msg: str) -> None:
    """Print ``msg`` as a user error and exit with code 1."""
    _err(msg)
    raise typer.Exit(code=EXIT_USER_ERROR)


def _jpk_txt_row_count(path: Path) -> int:
    """Count the number of numeric data rows in a JPK ``.txt`` file.

    The file is allowed to have a 1-line text header (4 whitespace
    separated column names) — we just count the total number of
    non-empty lines after that header. Counting line-by-line is
    O(file size) but a 4-column JPK export is rarely more than a few
    MB; this avoids loading the whole file just to report a row count.
    """
    count = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def _iter_jpk_txts(directory: Path, *, recursive: bool) -> list[Path]:
    """Return a sorted list of ``.txt`` files under ``directory``.

    When ``recursive`` is True the search descends into sub-directories;
    otherwise only the top level is scanned. Hidden directories and
    files (dotfiles) are skipped so a ``.git`` or ``.venv`` next to the
    data folder never leaks in.
    """
    pattern = "**/*.txt" if recursive else "*.txt"
    return sorted(p for p in directory.glob(pattern) if p.is_file() and not p.name.startswith("."))


def _make_fit_table(fits: list[Any]) -> Table:
    """Build a rich :class:`Table` summarising the fit results."""
    table = Table(title="Fit results", show_lines=False)
    table.add_column("curve", justify="right", style="cyan")
    table.add_column("model", style="magenta")
    table.add_column("p", justify="right")
    table.add_column("p_stderr", justify="right", style="dim")
    table.add_column("L", justify="right")
    table.add_column("L_stderr", justify="right", style="dim")
    table.add_column("redchi", justify="right")
    table.add_column("ok", justify="center")

    def _fmt(v: Any) -> str:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return "n/a"
        if f != f:  # NaN
            return "nan"
        return f"{f:.4g}"

    for idx, fit in enumerate(fits):
        params = getattr(fit, "params", {}) or {}
        stderr = getattr(fit, "stderr", None) or getattr(fit, "param_stderr", {}) or {}
        # Some FitResult flavours expose `stderr` and some `param_stderr`
        # (see the dataclasses in fitting.report vs io.exporters). We
        # try both, and fall back to the old `FitResult` exporter
        # dataclass for the legacy io-side definition.
        if not stderr and hasattr(fit, "param_stderr"):
            stderr = fit.param_stderr
        redchi = getattr(fit, "reduced_chi_square", None)
        if redchi is None and hasattr(fit, "redchi"):
            redchi = fit.redchi
        success = bool(getattr(fit, "metadata", {}).get("success", True))
        table.add_row(
            str(idx),
            str(getattr(fit, "model_name", "")),
            _fmt(params.get("p")),
            _fmt(stderr.get("p")),
            _fmt(params.get("L")),
            _fmt(stderr.get("L")),
            _fmt(redchi),
            "[green]✓[/green]" if success else "[red]✗[/red]",
        )
    return table


# -- Version --------------------------------------------------------------


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"afmkit [bold cyan]{__version__}[/bold cyan]")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Print the installed afmkit version and exit.",
        ),
    ] = None,
) -> None:
    """afmkit — modern Python toolkit for single-molecule force spectroscopy."""


@app.command()
def version() -> None:
    """Print the installed afmkit version."""
    console.print(f"afmkit [bold cyan]{__version__}[/bold cyan]")


# -- info -----------------------------------------------------------------


@app.command()
def info(
    path: Annotated[Path, typer.Argument(help="Path to a data file (HDF5 / JPK .txt / IBW).")],
) -> None:
    """Describe a single data file.

    The output format depends on the file type:

    - ``.h5`` / ``.hdf5`` — opens the file and prints the number of
      curves, the batch name, and the batch-level metadata.
    - ``.txt`` (JPK 4-column shape) — peeks the first line and reports
      the row count plus a reminder that ``--k`` is required to
      actually import the data.
    - ``.ibw`` — prints a clear "not yet wired" message and exits 0.
    - Anything else — exits 1 with an "Unknown file format" error.
    """
    if not path.exists():
        _abort_user(f"File not found: {path}")

    suffix = path.suffix.lower()

    # -- HDF5 --
    if suffix in {".h5", ".hdf5"}:
        # Imported lazily so the JPK / IBW paths don't pay the
        # h5py import cost.
        from afmkit.io.hdf5_store import load_hdf5

        try:
            batch = load_hdf5(path)
        except (ValueError, OSError) as exc:
            _abort_user(f"Could not read HDF5 file {path}: {exc}")
        table = Table(title=f"{path} — afmkit HDF5 archive", show_lines=False)
        table.add_column("Field", style="cyan")
        table.add_column("Value")
        table.add_row("n_curves", str(batch.n_curves))
        table.add_row("batch_name", str(batch.name) if batch.name else "(unset)")
        table.add_row("batch_metadata", str(batch.metadata) if batch.metadata else "{}")
        console.print(table)
        return

    # -- JPK 4-column .txt --
    if suffix == ".txt":
        # Peek the first non-empty line to confirm the shape matches
        # the JPK 4-column export before reporting. If it does not, we
        # treat the file as "unknown" — the user can rename it or pass
        # the right loader.
        first = _first_nonempty_line(path)
        if not first:
            _abort_user(f"file {path} is empty")
        n_cols = len(first.split())
        if n_cols != 4:
            _abort_user(
                f"file {path} has {n_cols} columns on the first non-empty line; "
                "afmkit info only recognises JPK 4-column .txt exports"
            )
        n_rows = _jpk_txt_row_count(path)
        # NB: keep "k required" together — word-wrapping would split it
        # across lines and break substring-based test assertions.
        console.print(
            f"[bold]{path}[/bold] — JPK .txt, {n_rows} rows, [bold]k required[/bold] for import"
        )
        return

    # -- Igor .ibw --
    if suffix == ".ibw":
        console.print(
            f"[yellow]Igor .ibw support is not yet wired (v0.2).[/yellow] "
            f"File: {path} ({path.stat().st_size} bytes)"
        )
        raise typer.Exit()

    _abort_user(f"Unknown file format: {suffix or '(no extension)'}")


def _first_nonempty_line(path: Path) -> str:
    """Return the first non-empty line of ``path`` with whitespace stripped."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                return stripped
    return ""


# -- import --------------------------------------------------------------


@app.command(name="import")
def import_cmd(
    source: Annotated[
        Path,
        typer.Argument(help="A single .txt file or a directory of .txt files to import."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Destination HDF5 path."),
    ] = Path("afmkit_session.h5"),
    k: Annotated[
        float | None,
        typer.Option(
            "--k",
            help="Cantilever spring constant (pN/nm). Required when SOURCE is a directory.",
        ),
    ] = None,
    recursive: Annotated[
        bool,
        typer.Option(
            "--recursive",
            "-r",
            help="Recurse into sub-directories when SOURCE is a directory.",
        ),
    ] = False,
) -> None:
    """Import raw AFM data into an afmkit HDF5 archive.

    SOURCE may be either a single JPK ``.txt`` file or a directory of
    them. When SOURCE is a directory, ``--k`` is required (each loader
    call needs the same cantilever spring constant) and the directory
    walk is non-recursive by default — pass ``--recursive`` to descend
    into sub-folders.
    """
    if not source.exists():
        _abort_user(f"Source not found: {source}")

    # Decide the set of .txt files to load. For a single-file SOURCE
    # the file must end in .txt; a directory is searched with the
    # chosen recursion policy.
    if source.is_file():
        if source.suffix.lower() != ".txt":
            _abort_user(f"Single-file import expects a .txt source; got {source.suffix!r}")
        files: list[Path] = [source]
    else:
        files = _iter_jpk_txts(source, recursive=recursive)
        if not files:
            _abort_user(f"No .txt files found under {source}")
        if k is None:
            _abort_user(
                "Importing a directory of JPK .txt files requires --k "
                "(the cantilever spring constant is not stored in the file)."
            )

    # Sanity-check k up front; the loader also checks, but failing
    # fast here gives the user a cleaner message.
    if k is not None and (k <= 0 or k != k):  # NaN-safe
        _abort_user(f"--k must be a finite positive number; got {k!r}")

    # Lazy import — keeps `afmkit version` and `afmkit info` snappy.
    from afmkit.core.curve import CurveBatch
    from afmkit.io.hdf5_store import save_hdf5
    from afmkit.io.jpk_txt import load_jpk_txt

    batches: list[CurveBatch] = []
    for path in files:
        try:
            batch = load_jpk_txt(path, k_cantilever=k if k is not None else 0.0)
        except (ValueError, FileNotFoundError, OSError) as exc:
            _abort_user(f"Failed to import {path}: {exc}")
        # Per-file CurveBatch carries a single ``k`` and source name;
        # the merged batch keeps them in the per-curve metadata.
        batches.append(batch)

    merged = CurveBatch.concat(batches, name=source.stem if source.is_dir() else source.stem)
    # Top-level batch metadata: surface the k + a count of the
    # contributing files so the resulting HDF5 file is self-describing.
    merged.metadata = {
        "k_cantilever": k if k is not None else 0.0,
        "source_files": [str(p) for p in files],
        "n_source_files": len(files),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        save_hdf5(merged, output)
    except (ValueError, OSError) as exc:
        _abort_user(f"Failed to write {output}: {exc}")

    table = Table(title=f"Imported {len(files)} file(s) → {output}", show_lines=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("files_imported", str(len(files)))
    table.add_row("total_curves", str(merged.n_curves))
    table.add_row("k_cantilever", f"{k:.4g} pN/nm" if k is not None else "n/a")
    table.add_row("output", str(output))
    table.add_row("output_size_kb", f"{output.stat().st_size / 1024:.1f} KB")
    console.print(table)


# -- fit ------------------------------------------------------------------


@app.command()
def fit(
    session: Annotated[Path, typer.Argument(help="Path to an afmkit HDF5 session.")],
    model: Annotated[str, typer.Option(help="Polymer model name (default: wlc).")] = "wlc",
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Destination CSV path (one row per fit)."),
    ] = None,
    x_range: Annotated[
        tuple[float, float] | None,
        typer.Option(
            "--x-range",
            help="Restrict the fit to this extension range (nm), e.g. --x-range 20 180.",
        ),
    ] = None,
) -> None:
    """Fit every curve in a session with the given model.

    The fit results are written to a one-row-per-fit CSV (default
    ``fits.csv``) and a sibling Markdown report (default ``fits.md``).
    A rich table of the fits is printed to the console. The command
    exits with code 1 when no curve fits successfully, code 0
    otherwise.
    """
    if not session.exists():
        _abort_user(f"Session file not found: {session}")
    if output is None:
        output = session.with_name(f"{session.stem}_fits.csv")
    md_path = output.with_suffix(".md")

    # Lazy imports — keeps the version / info paths fast.
    from afmkit.fitting import fit as fit_curve
    from afmkit.io.exporters import to_csv_fits, to_markdown
    from afmkit.io.hdf5_store import load_hdf5
    from afmkit.models import get_model

    try:
        batch = load_hdf5(session)
    except (ValueError, OSError) as exc:
        _abort_user(f"Failed to load session {session}: {exc}")

    if batch.n_curves == 0:
        _abort_user(f"Session {session} contains no curves to fit")

    # Validate the model up front so an unknown name doesn't waste
    # time fitting N curves only to fail on the report step.
    try:
        get_model(model)
    except KeyError as exc:
        _abort_user(str(exc))

    # Run the fits. A failed curve is recorded but does not abort
    # the batch — the user gets a partial report and a clear summary.
    results: list[Any] = []
    for i, curve in enumerate(batch):
        try:
            result = fit_curve(curve, model=model, x_range=x_range)
        except (ValueError, TypeError) as exc:
            # Programming / data shape errors are surfaced per curve
            # but never abort the loop — the spec says "show how
            # many did" succeed, and the loop must continue.
            console.print(f"[yellow]curve {i}:[/yellow] fit raised {type(exc).__name__}: {exc}")
            continue
        results.append(result)

    n_ok = sum(1 for r in results if r.metadata.get("success", True))
    console.print(_make_fit_table(results))
    console.print(f"[bold]{n_ok}[/bold] / [bold]{len(results)}[/bold] curves fit successfully.")

    if not results:
        _abort_user("No curves could be fit — aborting")

    # -- CSV (one row per fit) --
    # The fitting layer returns ``FitResult`` objects with attribute
    # names ``chi_square`` / ``reduced_chi_square`` / ``stderr``; the
    # ``to_csv_fits`` helper in :mod:`afmkit.io.exporters` was written
    # against its own (slightly different) local dataclass and reads
    # ``chi2`` / ``redchi`` / ``param_stderr``. Adapt the field names
    # here so the spec'd exporter keeps working without modification.
    # The same adaptation is also needed for ``to_markdown``, which
    # reads ``chi2`` and ``redchi`` when rendering the fit table.
    exporter_fits = [_adapt_fit_for_csv(r) for r in results]
    try:
        to_csv_fits(exporter_fits, output)
    except (ValueError, OSError) as exc:
        _abort_user(f"Failed to write fit CSV {output}: {exc}")
    console.print(f"[green]✓[/green] wrote fit CSV: {output}")

    # -- Markdown report --
    try:
        to_markdown(batch, exporter_fits, md_path)
    except (ValueError, OSError) as exc:
        _abort_user(f"Failed to write Markdown report {md_path}: {exc}")
    console.print(f"[green]✓[/green] wrote Markdown report: {md_path}")

    # Non-zero exit when nothing succeeded — but we still wrote the
    # partial CSV / MD so the user can see what went wrong.
    if n_ok == 0:
        raise typer.Exit(code=EXIT_USER_ERROR)


def _adapt_fit_for_csv(fit: Any) -> Any:
    """Map a fitting-layer :class:`FitResult` to the CSV exporter's shape.

    The exporter in :mod:`afmkit.io.exporters` reads ``fit.chi2``,
    ``fit.redchi``, and ``fit.param_stderr``; the fitting layer's
    dataclass uses ``chi_square`` / ``reduced_chi_square`` / ``stderr``.
    Rather than fork the dataclass, this helper returns a fresh
    exporter-shaped :class:`afmkit.io.exporters.FitResult` constructed
    via :func:`dataclasses.replace` so the optional ``covariance`` /
    ``x`` / ``y`` fields are preserved. If the input is already
    exporter-shaped (e.g. in a test fixture), it is returned unchanged.
    """
    from dataclasses import is_dataclass, replace

    from afmkit.io.exporters import FitResult as ExporterFitResult

    if is_dataclass(fit) and type(fit) is ExporterFitResult:
        return fit
    # Fitting-layer shape → exporter shape.
    if not hasattr(fit, "chi2") and hasattr(fit, "chi_square"):
        return replace(
            ExporterFitResult(
                model_name=str(getattr(fit, "model_name", "")),
                params=dict(getattr(fit, "params", {}) or {}),
                param_stderr=dict(getattr(fit, "stderr", {}) or {}),
                chi2=float(getattr(fit, "chi_square", float("nan"))),
                redchi=float(getattr(fit, "reduced_chi_square", float("nan"))),
                n_data=int(getattr(fit, "n_data", 0)),
            ),
            covariance=getattr(fit, "covariance", None),
            x=getattr(fit, "x_fit", None),
            y=getattr(fit, "y_fit", None),
        )
    return fit


# -- export --------------------------------------------------------------


@app.command()
def export(
    session: Annotated[Path, typer.Argument(help="Path to an afmkit HDF5 session.")],
    fmt: Annotated[
        str | None,
        typer.Option(
            "--format",
            "-f",
            help="Output format: csv, mat, parquet, md, markdown. "
            "Inferred from the --output suffix if omitted.",
        ),
    ] = None,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Destination file path."),
    ] = Path("results.csv"),
) -> None:
    """Export an afmkit session to a portable format.

    Supported formats: ``csv`` (wide column-block), ``mat`` (Matlab v5),
    ``parquet`` (wide column-block), and ``md`` / ``markdown`` (human
    report). If ``--format`` is omitted, the format is inferred from
    the suffix of ``--output``.
    """
    if not session.exists():
        _abort_user(f"Session file not found: {session}")

    # Lazy imports.
    from afmkit.io.exporters import export as export_batch
    from afmkit.io.hdf5_store import load_hdf5

    try:
        batch = load_hdf5(session)
    except (ValueError, OSError) as exc:
        _abort_user(f"Failed to load session {session}: {exc}")

    if batch.n_curves == 0:
        _abort_user(f"Session {session} contains no curves to export")

    # Map the user-facing ``md`` shortcut to the canonical
    # ``markdown`` identifier understood by ``exporters.export``. The
    # mapping is only applied when the user passed an explicit
    # ``--format``; when ``fmt`` is None the suffix dispatch inside
    # ``exporters.export`` decides the format.
    fmt_normalized: str | None = None
    if fmt is not None:
        fmt_normalized = "markdown" if fmt.lower() == "md" else fmt

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        export_batch(batch, output, fmt=fmt_normalized)
    except (ValueError, ImportError, OSError) as exc:
        _abort_user(f"Failed to export to {output}: {exc}")

    size_kb = output.stat().st_size / 1024 if output.exists() else 0.0
    console.print(
        f"[green]✓[/green] Wrote {batch.n_curves} curves to {output} (size: {size_kb:.1f} KB)"
    )


# -- Self-check -----------------------------------------------------------


if __name__ == "__main__":
    app()
