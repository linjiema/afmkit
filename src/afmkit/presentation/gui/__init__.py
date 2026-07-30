"""Terminal UI for afmkit.

This subpackage hosts the interactive Textual TUI used by the
``afmkit gui`` subcommand. The actual :class:`AFMkitApp` lives in
:mod:`afmkit.presentation.gui.app` — this module is intentionally
empty so the package can be imported without paying the Textual
import cost (the CLI does the same trick with a lazy import inside
the ``gui`` command body).

Why a TUI and not a Qt GUI
---------------------------
PySide6 / PyQt6 is a 100 MB+ dependency and needs an X server (or
VNC) to run on Linux / macOS. Textual is pure Python, ~2 MB, and
runs in any terminal — so a researcher SSHed into a lab box can
still use afmkit interactively without any extra display setup.

The TUI is intentionally minimal in v0.2: it loads a directory of
JPK ``.txt`` files, lets the user fit one curve with the WLC
model, and exports the result. Peak picking, the eWLC model, and
manual review are deferred to v0.3.
"""
