"""Signal processing primitives used by the analysis workflows.

- :mod:`~afmkit.processing.smooth` — Savitzky-Golay and moving-average smoothing.
- :mod:`~afmkit.processing.baseline` — Baseline correction (mirror, linear, exponential).
- :mod:`~afmkit.processing.peaks` — Peak detection in force curves.
- :mod:`~afmkit.processing.calibration` — Cantilever calibration helpers
  (deflection → force, slope correction, …).

All functions are pure: they accept arrays and parameters, return arrays.
No hidden state, no global configuration.
"""

from __future__ import annotations
