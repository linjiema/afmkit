"""Core data model for afmkit.

This module defines the foundational types that everything else builds on:

- :class:`~afmkit.core.curve.ForceCurve` — a single force-extension measurement
  (one forward or backward sweep) with calibration metadata.
- :class:`~afmkit.core.curve.CurveBatch` — an ordered collection of curves
  (e.g. all curves from one experiment folder).
- :class:`~afmkit.core.session.Session` — a complete analysis context
  (input data + parameters + results) for serialization and reproduction.
- :class:`~afmkit.core.types` — shared Protocols and TypedDicts.
- :class:`~afmkit.core.units` — optional pint-based unit handling.

Nothing in this module depends on IO, models, fitting, or presentation —
it is the contract that all other layers consume.
"""

from __future__ import annotations
