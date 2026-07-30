"""Interactive review state machine for sawtooth unfolding peaks.

This module is the v0.3 answer to "auto-detection is good, but I need
to look at every peak by hand." It wraps the output of
:func:`~afmkit.analysis.peak_detection.find_sawtooth_peaks` in a small
state machine — a :class:`PeakReviewer` — that lets a researcher
**accept**, **reject**, **override**, **re-fit**, and **annotate**
each detected peak. The state machine is intentionally pure data +
behaviour, with no UI dependency, so the same logic can drive a TUI
keyboard handler, a Jupyter widget, or a batch CSV export.

Why a state machine and not just a flag on the :class:`Peak`?
-------------------------------------------------------------
The :class:`~afmkit.analysis.peak_detection.Peak` dataclass is a
frozen, read-only result of the auto-detection step. Mixing the
mutable "the user has decided to reject this one" flag into the same
class would conflate the immutable scientific result with the
mutable review workflow — and force every consumer (plotters,
statistics, exporters) to know about review semantics. Keeping the
review layer separate means :class:`Peak` stays a clean v0.2 output
and :class:`ReviewedPeak` adds the v0.3 review state on top.

Public surface
--------------
- :class:`ReviewedPeak` — frozen dataclass, the unit of the state machine.
- :class:`PeakReviewer` — list-like container + transition methods.

Conventions
-----------
- All extension values are in **nm**.
- All force values are in **pN**.
- Indexing is 0-based and contiguous; after a rejection call
  :meth:`PeakReviewer.reindex` to renumber.
- The :class:`ForceCurve` passed to the constructor is treated as
  read-only — :meth:`PeakReviewer.re_fit` never mutates it; it builds
  a fresh slice internally.

Examples
--------
>>> from afmkit.analysis import find_sawtooth_peaks, PeakReviewer
>>> # Setup: a synthetic curve with three known sawtooth peaks.
>>> # See tests/unit/test_peak_review.py for a runnable example.
>>> peaks = find_sawtooth_peaks(curve, min_prominence_pN=5.0)  # doctest: +SKIP
>>> reviewer = PeakReviewer(peaks, curve)  # doctest: +SKIP
>>> reviewer.reject(2)                       # doctest: +SKIP
>>> reviewer.override(0, 45.2)               # doctest: +SKIP
>>> result = reviewer.re_fit(1)              # doctest: +SKIP
>>> accepted = reviewer.accepted             # doctest: +SKIP
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from afmkit.analysis.peak_detection import Peak
from afmkit.fitting import fit
from afmkit.fitting.report import FitResult
from afmkit.models import get_model

if TYPE_CHECKING:
    from afmkit.core.curve import ForceCurve

__all__ = ["PeakReviewer", "ReviewedPeak"]


# -- Public dataclass -----------------------------------------------------


@dataclass(frozen=True)
class ReviewedPeak:
    """A :class:`~afmkit.analysis.peak_detection.Peak` tagged with manual review state.

    This is the unit of the :class:`PeakReviewer` state machine. It
    wraps an auto-detected :class:`Peak` and adds three pieces of
    mutable (in the reviewer's eyes) metadata:

    - ``accepted``      — soft-delete flag, default ``True``.
    - ``manual_force``  — user-override force in pN; ``None`` means
                          "use the auto-detected value".
    - ``note``          — free-form user annotation.

    The dataclass is :func:`frozen` so the review state can be hashed
    and compared for equality (useful in tests and exporters); the
    :class:`PeakReviewer` rebuilds a new :class:`ReviewedPeak` on
    every transition.

    Attributes
    ----------
    peak
        The original auto-detected :class:`Peak`. The reviewer never
        mutates this in place; if the peak's index needs to change
        (e.g. after a soft-delete + reindex) the reviewer produces a
        new :class:`Peak` with :func:`dataclasses.replace`.
    accepted
        ``True`` if the user has accepted this peak, ``False`` if
        rejected. Defaults to ``True`` — auto-detection passed the
        default filter, the user has to actively reject.
    manual_force
        Optional user-override force in pN. If set, this value
        replaces ``peak.force`` in the exported output. ``None`` (the
        default) means "no override, use the auto-detected value".
    note
        Free-form annotation the user attached to this peak.
        Empty string by default.
    """

    peak: Peak
    accepted: bool = True
    manual_force: float | None = None
    note: str = ""

    @property
    def force(self) -> float:
        """The force to use in output (pN).

        Returns ``manual_force`` if the user has set an override,
        otherwise the auto-detected ``peak.force``. This is the value
        the CSV / Markdown exporters should write into the ``force``
        column — it is the single source of truth for "what force to
        attribute to this peak after review".
        """
        if self.manual_force is not None:
            return float(self.manual_force)
        return float(self.peak.force)

    @property
    def extension(self) -> float:
        """The peak's extension in nm (always delegates to ``peak.extension``)."""
        return float(self.peak.extension)

    @property
    def confidence(self) -> float:
        """The auto-detected confidence in [0, 1] (always delegates to ``peak.confidence``)."""
        return float(self.peak.confidence)


# -- State machine --------------------------------------------------------


class PeakReviewer:
    """State machine over a list of auto-detected peaks.

    The reviewer is the v0.3 bridge between auto-detection and the
    CSV / Markdown export: it holds the immutable auto-detected
    :class:`Peak` list and the user's review decisions on top. It
    supports all four state transitions the user might want
    (``accept``, ``reject``, ``override``, ``clear_override``,
    ``set_note``) plus a re-fit path that updates the override to a
    fresh :class:`FitResult` evaluated at the peak's extension.

    Parameters
    ----------
    peaks
        The auto-detected peaks (output of
        :func:`~afmkit.analysis.peak_detection.find_sawtooth_peaks`).
        The list is **copied** — the caller may mutate the input
        without affecting the reviewer's state.
    curve
        The :class:`~afmkit.core.curve.ForceCurve` the peaks were
        detected on. Stored so :meth:`re_fit` can rebuild a local
        slice for the engine. The curve is treated as read-only.

    Examples
    --------
    >>> # See tests/unit/test_peak_review.py for a runnable setup.
    >>> peaks = find_sawtooth_peaks(curve, min_prominence_pN=5.0)  # doctest: +SKIP
    >>> reviewer = PeakReviewer(peaks, curve)  # doctest: +SKIP
    >>> reviewer.reject(2)            # doctest: +SKIP
    >>> reviewer.override(0, 45.2)    # doctest: +SKIP
    >>> reviewer.re_fit(1)            # doctest: +SKIP
    >>> len(reviewer.accepted)        # doctest: +SKIP
    ...                              # doctest: +SKIP
    """

    #: Default half-width of the local re-fit window (in nm), used
    #: when :meth:`re_fit` is called without an explicit ``x_range``.
    #: +/- 20 nm around the peak is wide enough to give the WLC solver
    #: a stable starting point on a 5000-point, 0-300 nm curve, while
    #: narrow enough to avoid pulling in adjacent peaks.
    DEFAULT_REFIT_HALF_WIDTH_NM: float = 20.0

    def __init__(self, peaks: list[Peak], curve: ForceCurve) -> None:
        # Copy the peak list so the caller's array is safe to mutate.
        # The `Peak` objects themselves are frozen, so a shallow copy
        # of the list is enough.
        self._items: list[ReviewedPeak] = [ReviewedPeak(peak=p) for p in peaks]
        self._curve: ForceCurve = curve

    # -- Container protocol ----------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[ReviewedPeak]:
        return iter(self._items)

    def __getitem__(self, idx: int) -> ReviewedPeak:
        return self._items[idx]

    # -- State transitions -----------------------------------------------

    def accept(self, idx: int) -> None:
        """Mark peak ``idx`` as accepted.

        The default state is already accepted, so this is mostly
        useful as the inverse of :meth:`reject` — a TUI keyboard
        handler can offer an explicit "un-reject" key without
        having to inspect the current state.

        Parameters
        ----------
        idx
            0-based index into the reviewer's peak list.

        Raises
        ------
        IndexError
            If ``idx`` is out of range.
        """
        current = self._items[idx]
        if not current.accepted:
            self._items[idx] = replace(current, accepted=True)

    def reject(self, idx: int) -> None:
        """Mark peak ``idx`` as rejected (soft delete).

        The peak stays in the reviewer's list with
        ``accepted=False`` so the user can see what they rejected
        in the export and undo the decision. Call
        :meth:`reindex` afterwards to keep the indices contiguous.

        Parameters
        ----------
        idx
            0-based index into the reviewer's peak list.

        Raises
        ------
        IndexError
            If ``idx`` is out of range.
        """
        current = self._items[idx]
        if current.accepted:
            self._items[idx] = replace(current, accepted=False)

    def override(self, idx: int, manual_force: float) -> None:
        """Set the user-override force (pN) for peak ``idx``.

        After this call, :attr:`ReviewedPeak.force` will return
        ``manual_force`` instead of the auto-detected value. To
        revert, use :meth:`clear_override`.

        Parameters
        ----------
        idx
            0-based index into the reviewer's peak list.
        manual_force
            The user-chosen force value, in pN. Must be a finite,
            strictly positive number — non-positive or non-finite
            values raise :class:`ValueError` so a typo in the TUI's
            override dialog cannot silently poison the export.

        Raises
        ------
        IndexError
            If ``idx`` is out of range.
        ValueError
            If ``manual_force`` is not a real number, is NaN, is
            ``±inf``, or is not strictly positive.
        """
        # `math.isfinite` rejects both NaN and ±inf in one call.
        if not isinstance(manual_force, int | float) or isinstance(manual_force, bool):
            raise ValueError(
                f"manual_force must be a real number, got {type(manual_force).__name__}"
            )
        value = float(manual_force)
        if not math.isfinite(value):
            raise ValueError(f"manual_force must be finite; got {value!r}")
        if value <= 0.0:
            raise ValueError(f"manual_force must be strictly positive (pN); got {value!r}")
        current = self._items[idx]
        self._items[idx] = replace(current, manual_force=value)

    def clear_override(self, idx: int) -> None:
        """Remove any user-override on peak ``idx``.

        After this call :attr:`ReviewedPeak.force` will return
        the auto-detected ``peak.force`` again. No-op if no
        override was set.

        Parameters
        ----------
        idx
            0-based index into the reviewer's peak list.

        Raises
        ------
        IndexError
            If ``idx`` is out of range.
        """
        current = self._items[idx]
        if current.manual_force is not None:
            self._items[idx] = replace(current, manual_force=None)

    def set_note(self, idx: int, note: str) -> None:
        """Attach a free-form note to peak ``idx``.

        Notes are surfaced in the Markdown / CSV export and are a
        convenient way to flag suspicious peaks for the next
        review pass (e.g. "spike near cantilever jump-off").

        Parameters
        ----------
        idx
            0-based index into the reviewer's peak list.
        note
            Any string. Empty string is allowed and means
            "no note".

        Raises
        ------
        IndexError
            If ``idx`` is out of range.
        """
        current = self._items[idx]
        if current.note != note:
            self._items[idx] = replace(current, note=note)

    # -- Re-fit ----------------------------------------------------------

    def re_fit(
        self,
        idx: int,
        *,
        model: str = "wlc",
        x_range: tuple[float, float] | None = None,
    ) -> FitResult:
        """Re-fit peak ``idx`` on a local slice of the curve.

        The fit uses the high-level :func:`afmkit.fitting.fit`
        helper, which resolves the model by name and returns a
        :class:`FitResult`. The new :attr:`ReviewedPeak.manual_force`
        is updated to the fit's ``y_fit`` evaluated at the peak's
        extension — that is, the *model's* value at the peak's x
        position, not the raw data. This keeps the "after review"
        force consistent with the WLC model and avoids the
        scenario where the user has a sharp spike but the WLC
        is much lower (the WLC value is the value the rest of the
        analysis pipeline trusts).

        Parameters
        ----------
        idx
            0-based index into the reviewer's peak list.
        model
            Registry key of the polymer model to fit (default
            ``"wlc"``). Forwarded to
            :func:`afmkit.fitting.fit`.
        x_range
            Optional ``(x_min, x_max)`` in nm, restricting the fit
            to a sub-range of the curve. Defaults to
            ``(peak.extension - 20, peak.extension + 20)``,
            clamped to the curve's actual extension bounds
            ``(0, max(curve.extension))``.

        Returns
        -------
        FitResult
            The new fit, also stored on the reviewer's internal
            state. The :attr:`ReviewedPeak.manual_force` for the
            peak is updated to the fit's ``y_fit`` value at
            ``peak.extension`` (interpolated if the peak's
            extension is not exactly on the fit's ``x_fit`` grid).

        Raises
        ------
        IndexError
            If ``idx`` is out of range.
        ValueError
            If the fit does not converge (``result.success is
            False``), or if the explicit ``x_range`` falls
            entirely outside the curve's extension axis.
        """
        _check_index(self._items, idx)
        current = self._items[idx]
        peak = current.peak

        # Build the slice range. We use `max(curve.extension)` as
        # the upper bound (the synthetic fixture goes 0→300 nm but
        # real JPK retracts can stop short), and 0.0 as the lower
        # bound. The peak's own extension is the centre of the
        # window.
        ext_max = float(np.max(np.asarray(self._curve.extension, dtype=np.float64)))
        if x_range is None:
            half = self.DEFAULT_REFIT_HALF_WIDTH_NM
            lo = float(peak.extension) - half
            hi = float(peak.extension) + half
            lo = max(0.0, lo)
            hi = min(ext_max, hi)
            if lo >= hi:
                # The peak is at (or beyond) the curve's right edge;
                # the default window collapses to a point. Fall back
                # to the full curve so the fit still has data.
                lo, hi = 0.0, ext_max
            x_range_local: tuple[float, float] = (lo, hi)
        else:
            x_range_local = x_range
            if x_range_local[0] > x_range_local[1]:
                raise ValueError(f"x_range must have x_min <= x_max; got {x_range_local}")
            # Catch the "range entirely outside the curve" case
            # early with a clear error message; the engine's own
            # x_range handling would raise something similar but
            # the message would not mention the unit.
            if x_range_local[1] < 0.0 or x_range_local[0] > ext_max:
                raise ValueError(
                    f"x_range {x_range_local} is entirely outside the curve's "
                    f"extension axis [0, {ext_max}]"
                )

        # Resolve the model and run the fit. We use the high-level
        # `fit()` helper rather than LmfitEngine directly so the
        # model registry stays the single point of truth for
        # "which models exist".
        _ = get_model(model)  # surface unknown-model KeyError early
        result = fit(self._curve, model=model, x_range=x_range_local)

        # A non-converging fit is a programming/usage error here —
        # the slice is small, the data is finite, and we are
        # passing an explicit x_range. Surface it as a ValueError
        # so the caller can decide whether to fall back to the
        # original auto-detected force.
        if not bool(result.metadata.get("success", False)):
            message = str(result.metadata.get("message", "unknown"))
            raise ValueError(f"re_fit[{idx}] did not converge: {message}")

        # Pull the model value at the peak's extension off the
        # fit's own x_fit / y_fit arrays. linear interpolation —
        # the peak's extension is in nm, the fit's x_fit is in nm.
        new_force = float(_interp_at(result.x_fit, result.y_fit, peak.extension))

        # Update the reviewer's state. The dataclass is frozen, so
        # `replace` is the supported mutation path.
        self._items[idx] = replace(current, manual_force=new_force)
        return result

    # -- Bulk views ------------------------------------------------------

    @property
    def accepted(self) -> list[ReviewedPeak]:
        """All peaks currently marked accepted (in list order)."""
        return [p for p in self._items if p.accepted]

    @property
    def rejected(self) -> list[ReviewedPeak]:
        """All peaks currently marked rejected (in list order)."""
        return [p for p in self._items if not p.accepted]

    # -- Export ----------------------------------------------------------

    def to_dict(self) -> list[dict[str, Any]]:
        """Export the reviewer state to a list of dicts, one per peak.

        The output is the contract the CSV / Markdown exporters
        consume in a later commit. The columns are:

        - ``index``         — 0-based position in the reviewer's list.
        - ``extension``     — nm, always from the auto-detected peak.
        - ``force``         — pN, the post-review force
          (``ReviewedPeak.force``: override if set, else peak.force).
        - ``manual_force``  — pN if the user has set an override, else
          ``None``. Surfaced separately from ``force`` so the export
          can flag which peaks were user-modified.
        - ``accepted``      — bool.
        - ``confidence``    — unitless, [0, 1].
        - ``prominence``    — pN.
        - ``width``         — points.
        - ``height_drop``   — pN.
        - ``note``          — free-form string.

        The list is in the same order as :attr:`__iter__`; rejected
        peaks are included with ``accepted=False`` so the export
        reflects the reviewer's full state.
        """
        out: list[dict[str, Any]] = []
        for i, rp in enumerate(self._items):
            out.append(
                {
                    "index": i,
                    "extension": float(rp.peak.extension),
                    "force": float(rp.force),
                    "manual_force": (
                        float(rp.manual_force) if rp.manual_force is not None else None
                    ),
                    "accepted": bool(rp.accepted),
                    "confidence": float(rp.peak.confidence),
                    "prominence": float(rp.peak.prominence),
                    "width": int(rp.peak.width),
                    "height_drop": float(rp.peak.height_drop),
                    "note": str(rp.note),
                }
            )
        return out

    # -- Reindex ---------------------------------------------------------

    def reindex(self) -> None:
        """Re-number the index column on every ReviewedPeak.

        Call this after rejecting a peak to keep the indices
        contiguous. Rejection is a soft delete — the peak stays in
        the list with ``accepted=False`` — but the ``peak.index``
        field on the auto-detected :class:`Peak` is now stale
        (it was set at detection time, not at review time).
        :meth:`reindex` updates the field in place by building a
        new :class:`Peak` with :func:`dataclasses.replace` (the
        auto-detected :class:`Peak` is a frozen dataclass, so
        in-place mutation is impossible).

        The new indices are assigned in the reviewer's current
        list order, skipping the rejected entries? **No** - the
        new index is the position in the *full* list, so the
        original position of a rejected peak is preserved as
        ``peak.index``. This matches the intuitive reading of the
        export: "peak 5 was the one I rejected", with the 5
        coming from the position at review time.
        """
        rebuilt: list[ReviewedPeak] = []
        for new_idx, rp in enumerate(self._items):
            if rp.peak.index != new_idx:
                new_peak = replace(rp.peak, index=new_idx)
                rebuilt.append(replace(rp, peak=new_peak))
            else:
                rebuilt.append(rp)
        self._items = rebuilt

    # -- Internals -------------------------------------------------------

    @property
    def curve(self) -> ForceCurve:
        """The :class:`ForceCurve` the reviewer is bound to (read-only)."""
        return self._curve


# -- Helpers --------------------------------------------------------------


def _check_index(items: list[Any], idx: int) -> None:
    """Validate ``idx`` against a list and raise :class:`IndexError`.

    Python's list indexing accepts negative values (``lst[-1]`` is
    the last element), but the :class:`PeakReviewer` API contract
    is that indices are non-negative, 0-based, contiguous, and
    in-range. A TUI keyboard handler that does ``idx -= 1`` to
    convert a 1-based display to 0-based storage, and then
    accidentally underflows past 0, should see an :class:`IndexError`,
    not a silent wrap to the last peak. This helper enforces that
    contract for the methods that explicitly require it.
    """
    n = len(items)
    if not isinstance(idx, int) or idx < 0 or idx >= n:
        raise IndexError(f"peak index {idx} out of range [0, {n})")


def _interp_at(x: np.ndarray, y: np.ndarray, x_query: float) -> float:
    """Linearly interpolate ``y(x_query)`` from the (x, y) arrays.

    Used by :meth:`PeakReviewer.re_fit` to read the model's
    y-value at the peak's extension. The peak's extension is in
    nm; the fit's ``x_fit`` is also in nm. The function clamps
    to the endpoints of ``x`` if the query falls outside the
    fit's range (which can happen for a peak near the curve's
    edge where the fit window is small).

    Raises
    ------
    ValueError
        If ``x`` is empty (the fit had no data — caller should
        have caught the failed fit earlier; this is a defensive
        guard).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size == 0 or y_arr.size == 0:
        raise ValueError("cannot interpolate on an empty fit result")
    if x_arr.size != y_arr.size:
        raise ValueError(f"x and y must have the same length; got {x_arr.size} vs {y_arr.size}")
    # `np.interp` clamps out-of-range queries to the endpoints, which
    # is the behaviour we want for a peak at the very edge of the
    # fit window.
    return float(np.interp(float(x_query), x_arr, y_arr))
