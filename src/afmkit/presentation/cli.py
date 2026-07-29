"""Command-line interface for afmkit (typer-based).

This module is the entry point declared in ``pyproject.toml`` under
``[project.scripts]`` — installing afmkit makes the ``afmkit`` shell
command available.

v0.1 commands
-------------
- ``afmkit version`` — print the installed afmkit version.
- ``afmkit info FILE`` — describe a data file (HDF5 / JPK / IBW).
- ``afmkit import ...`` — convert a folder of JPK ``.txt`` files into
  an afmkit HDF5 archive.
- ``afmkit fit HDF5 ...`` — fit all curves in an HDF5 archive with the
  WLC model.
- ``afmkit export HDF5 ...`` — export results to CSV / Matlab / Parquet.

Subcommands are wired up as the corresponding features land. Until
then, running the unimplemented command prints a friendly message
instead of a stack trace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from afmkit import __version__
from afmkit._version import __version__ as _v  # noqa: F401  (sanity)

app = typer.Typer(
    name="afmkit",
    help="afmkit — single-molecule force spectroscopy analysis.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()


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


@app.command()
def info(
    path: Annotated[Path, typer.Argument(help="Path to a data file (HDF5 / JPK .txt / IBW).")],
) -> None:
    """Describe a single data file.

    [yellow]Coming with v0.1.[/yellow] Until then, use the Python API
    (e.g. ``afmkit.core.curve.ForceCurve.from_xarray(...)``) to inspect
    data manually.
    """
    if not path.exists():
        console.print(f"[red]File not found:[/red] {path}")
        raise typer.Exit(code=1)
    console.print(
        f"[yellow]'info' is not implemented yet (v0.1 in progress).[/yellow]\n"
        f"File: {path} ({path.stat().st_size} bytes)"
    )


@app.command()
def import_cmd(
    source: Annotated[
        Path,
        typer.Argument(help="A single file or a directory of files to import."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Destination HDF5 path."),
    ] = Path("afmkit_session.h5"),
    k: Annotated[
        float | None,
        typer.Option("--k", help="Cantilever spring constant (pN/nm)."),
    ] = None,
) -> None:
    """Import raw AFM data into an afmkit HDF5 archive.

    [yellow]Coming with v0.1.[/yellow] Will support JPK 4-column .txt
    exports from Nanowizzard / ForceRobot, plus legacy Igor .ibw files.
    """
    console.print(
        "[yellow]'import' is not implemented yet (v0.1 in progress).[/yellow]\n"
        f"Would import: {source} -> {output} (k = {k})"
    )


@app.command()
def fit(
    session: Annotated[Path, typer.Argument(help="Path to an afmkit HDF5 session.")],
    model: Annotated[str, typer.Option(help="Polymer model name (default: wlc).")] = "wlc",
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Destination CSV path."),
    ] = None,
) -> None:
    """Fit all curves in a session with the given model.

    [yellow]Coming with v0.1.[/yellow] The default model is the
    Marko-Siggia WLC; eWLC and FJC will be available as plugins.
    """
    console.print(
        f"[yellow]'fit' is not implemented yet (v0.1 in progress).[/yellow]\n"
        f"Would fit {session} with model={model!r}, output={output}"
    )


@app.command()
def export(
    session: Annotated[Path, typer.Argument(help="Path to an afmkit HDF5 session.")],
    fmt: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format: csv, mat, parquet, md, ibw.",
        ),
    ] = "csv",
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Destination file path."),
    ] = Path("results.csv"),
) -> None:
    """Export an afmkit session to a portable format (CSV/MAT/Parquet/MD/IBW).

    [yellow]Coming with v0.1.[/yellow]
    """
    console.print(
        f"[yellow]'export' is not implemented yet (v0.1 in progress).[/yellow]\n"
        f"Would export {session} to {output} as {fmt}"
    )


if __name__ == "__main__":
    app()
