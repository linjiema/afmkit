"""High-level analysis workflows.

These are the afmkit equivalents of the original Igor "macros":

- :mod:`~afmkit.analysis.single` — :func:`analyze_single_curve`
- :mod:`~afmkit.analysis.batch`  — :func:`analyze_batch`
- :mod:`~afmkit.analysis.peak_detection` — automated sawtooth peak finding
- :mod:`~afmkit.analysis.statistics` — folding statistics, ΔL histograms, etc.

A workflow composes IO + processing + fitting + IO-export into one call,
but is itself just a regular Python function — scriptable, testable,
GUI-replaceable.
"""

from __future__ import annotations

from afmkit.analysis.peak_detection import Peak, find_sawtooth_peaks

__all__ = ["Peak", "find_sawtooth_peaks"]
