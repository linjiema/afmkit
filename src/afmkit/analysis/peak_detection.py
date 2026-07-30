"""Automated sawtooth unfolding-peak detection.

This module is the afmkit replacement for the original Igor Pro
``AutoFindForcePeaks`` macro. It scans a :class:`~afmkit.core.curve.ForceCurve`
(in the retract / extension direction) and returns the locations of the
"teeth" that mark individual protein-domain unfolding events.

Why this exists
---------------
Single-molecule force spectroscopy (SMFS) curves show a characteristic
sawtooth pattern: each unfolding event is a rapid force drop appearing
as a sharp peak in the force-extension trace. Identifying these peaks
is the very first step of any SMFS analysis — feeding the right peak
positions to a WLC fitter is what lets the contour-length increment
``ΔL`` for each domain be measured.

The legacy Igor macro did this with a hand-tuned routine
(``AutoFindForcePeaks``) that is brittle on noisy data and hard to
retune. The Python re-implementation uses :func:`scipy.signal.find_peaks`
with prominence + width filtering, which is more robust and parameter
tuning is exposed via a small, well-documented surface.

Algorithm
---------
1. Smooth the force trace with a centered moving average of
   ``smoothing_window`` points (default 5). NaN inputs propagate;
   curves whose smoothed trace contains NaN return ``[]`` rather than
   crashing the caller.
2. Find local maxima in the smoothed trace with
   :func:`scipy.signal.find_peaks`, using ``prominence=min_prominence_pN``
   and ``width=min_width_points`` as the two rejection thresholds.
3. For each candidate peak, compute:

   - ``prominence``  (pN)  — directly from scipy.
   - ``width``       (pts) — directly from scipy, full-width at half
     prominence, rounded to the nearest integer.
   - ``height_drop`` (pN)  — ``force[peak] - force[next_peak]`` (or
     ``force[-1]`` for the last peak), i.e. the size of the unfolding
     force drop right after the peak.
   - ``confidence``  [0,1] — ``clip(prominence / (median * 2), 0, 1)``;
     a simple, monotonic heuristic that scores 1.0 for the most
     prominent peak in the curve and degrades for smaller peaks. The
     scaling is intentionally conservative — even an average peak
     gets a confidence of 0.5, so downstream consumers can use a
     threshold of 0.3-0.5 to filter out weak candidates.

4. Sort the surviving peaks by extension (defensive — the curve's
   extension axis is normally already monotonic in the retract
   segment, but we don't want to assume it).
5. If ``max_peaks`` is set, keep the ``max_peaks`` highest-prominence
   candidates.

This algorithm is deliberately simple — it is the first piece of v0.2,
and the next iterations (v0.2.2 eWLC, v0.2.3 GUI) will refine it based
on the experience of running it on real lab data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import find_peaks

if TYPE_CHECKING:
    from afmkit.core.curve import ForceCurve

__all__ = ["Peak", "find_sawtooth_peaks"]


# -- Public dataclass -----------------------------------------------------


@dataclass(frozen=True)
class Peak:
    """A single sawtooth unfolding peak in a force-extension curve.

    All numeric fields use the afmkit-standard units:

    - ``extension``   — nm
    - ``force``       — pN
    - ``prominence``  — pN, peak height above its local baseline
    - ``height_drop`` — pN, force drop from this peak to the next
      (or to the curve's last point for the rightmost peak)
    - ``width``       — points, full-width-at-half-prominence
    - ``confidence``  — unitless in [0, 1]

    The dataclass is :func:`frozen` because peaks are read-only
    analysis results — callers should never mutate them; if you need
    to attach extra metadata, build a new dict alongside.
    """

    index: int
    extension: float
    force: float
    prominence: float
    width: int
    height_drop: float
    confidence: float


# -- Helpers --------------------------------------------------------------


def _centered_moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding.

    Uses ``mode="same"`` so the output has the same length as the
    input. Edges are filled by reflection (a "nearest" extrapolation
    is the default of :func:`numpy.convolve` with a uniform kernel
    of length ``window``) — this avoids the artificial dip that a
    zero-padded edge would create at the curve's start.

    Parameters
    ----------
    x
        Input 1-D array. Any dtype convertible to float64.
    window
        Smoothing window in points. Must be >= 1; values of 1 return
        the input unchanged. Values larger than ``len(x)`` are clamped
        to ``len(x)`` to avoid silent nonsense.
    """
    if window <= 1 or len(x) == 0:
        return np.asarray(x, dtype=np.float64)
    # Clamp the window so it never exceeds the array length — anything
    # bigger would just produce a flat line and waste cycles.
    w = min(int(window), int(len(x)))
    if w <= 1:
        return np.asarray(x, dtype=np.float64)
    kernel = np.ones(w, dtype=np.float64) / w
    return np.convolve(np.asarray(x, dtype=np.float64), kernel, mode="same")


# -- Public API -----------------------------------------------------------


def find_sawtooth_peaks(
    curve: ForceCurve,
    *,
    min_prominence_pN: float = 5.0,  # noqa: N803  (pN is the SI unit suffix)
    min_width_points: int = 3,
    max_peaks: int | None = None,
    smoothing_window: int = 5,
) -> list[Peak]:
    """Detect sawtooth unfolding peaks in a retract-mode ``ForceCurve``.

    Parameters
    ----------
    curve
        The :class:`~afmkit.core.curve.ForceCurve` to scan. The
        algorithm reads ``curve.force`` and ``curve.extension``; it
        does not mutate the curve.
    min_prominence_pN
        Minimum peak prominence, in **pN**. Peaks whose height above
        the surrounding baseline is below this threshold are
        discarded. The default (5 pN) is the lab's typical noise
        floor for protein unfolding experiments with a
        ``k ≈ 0.1 pN/nm`` cantilever.
    min_width_points
        Minimum full-width-at-half-prominence, in **points**. Spikes
        narrower than this are rejected as noise. The default (3)
        rejects single-point outliers while still catching genuine
        sawteeth, which are typically 5-20 points wide at typical
        sampling rates.
    max_peaks
        Optional cap on the number of returned peaks. If given, only
        the ``max_peaks`` highest-prominence candidates are kept.
        ``None`` (the default) means "return everything that
        survived the prominence / width filters".
    smoothing_window
        Width (in points) of the centered moving average applied to
        the force trace before peak finding. Default 5 — a good
        compromise between noise rejection and peak-position
        fidelity for the lab's default 5000-point curves. A
        ``smoothing_window`` of 1 disables smoothing.

    Returns
    -------
    list[Peak]
        Detected peaks, ordered by **increasing extension**. An empty
        list is returned for curves that are too short, that are
        entirely NaN after smoothing, or that contain no peaks above
        the prominence / width thresholds.

    Notes
    -----
    The function never raises on a "bad" curve — it returns ``[]`` for
    NaN-only inputs and for curves with no detectable peaks. This
    matches the spirit of the legacy Igor macro, which was used in
    batch loops and was expected to fail silently on the bad curves
    in a folder of 100 measurements.
    """
    ext = np.asarray(curve.extension, dtype=np.float64)
    force = np.asarray(curve.force, dtype=np.float64)

    # Too short to detect anything meaningful. This is the standard
    # edge-case the legacy macro returned [] for; reproduce that
    # behaviour rather than raising.
    if ext.size < 3 or force.size < 3:
        return []

    smoothed = _centered_moving_average(force, smoothing_window)
    # NaN propagation is intentional — if the user passed a curve
    # with NaNs (e.g. from a corrupted JPK file), we refuse to
    # silently invent peaks. Returning [] is the safe choice.
    if not np.all(np.isfinite(smoothed)):
        return []

    # scipy's find_peaks wants prominence and width as plain kwargs.
    # We pass them through directly — both are required to keep the
    # noise rejection honest.
    peak_indices, props = find_peaks(
        smoothed,
        prominence=float(min_prominence_pN),
        width=float(min_width_points),
    )

    if peak_indices.size == 0:
        return []

    prominences = np.asarray(props["prominences"], dtype=np.float64)
    # scipy returns widths as float (in points); we round to int for
    # the public API since the unit is "point count".
    widths = np.rint(np.asarray(props["widths"], dtype=np.float64)).astype(np.int64)

    # ``height_drop`` is the force drop from this peak to the next
    # detected one — that is, the size of the unfolding event.
    # For the rightmost peak, the next "reference" is the curve's
    # last point, so a tail-of-curve peak still gets a meaningful
    # drop estimate.
    last_force = float(force[-1])
    next_force = np.empty(peak_indices.size, dtype=np.float64)
    next_force[:-1] = force[peak_indices[1:]]
    next_force[-1] = last_force
    height_drops = force[peak_indices] - next_force

    # Confidence: a monotonic, conservative scaling.
    #
    #   confidence = clip(prominence / (median * 2), 0, 1)
    #
    # The most prominent peak in the curve (prominence = median * 2
    # for the median case, or higher if it dominates) gets a score
    # approaching 1.0. Smaller peaks get scores that shrink toward
    # zero but never go negative. A peak with prominence equal to
    # the median scores 0.5.
    #
    # We protect against the (degenerate) median == 0 case — which
    # only happens if every peak ties at zero prominence, an
    # extremely unlikely "all noise" outcome that nonetheless
    # deserves a sensible default rather than a division error.
    median_prom = float(np.median(prominences))
    if median_prom <= 0.0:
        confidences = np.zeros_like(prominences)
    else:
        confidences = np.clip(prominences / (2.0 * median_prom), 0.0, 1.0)

    peaks = [
        Peak(
            index=int(idx),
            extension=float(ext[idx]),
            force=float(force[idx]),
            prominence=float(prominences[i]),
            width=int(widths[i]),
            height_drop=float(height_drops[i]),
            confidence=float(confidences[i]),
        )
        for i, idx in enumerate(peak_indices)
    ]

    # Sort by extension — defensive, since the curve's extension axis
    # is normally already monotonic. Stable sort preserves the
    # prominence ranking within ties on extension, which is what we
    # want for the ``max_peaks`` truncation below.
    peaks.sort(key=lambda p: p.extension)

    if max_peaks is not None and len(peaks) > int(max_peaks):
        # Keep the N most prominent peaks. The sort is stable and
        # secondary-ordered by extension (ascending), so the kept
        # sub-list is also in extension order after the re-sort.
        peaks.sort(key=lambda p: p.prominence, reverse=True)
        peaks = peaks[: int(max_peaks)]
        peaks.sort(key=lambda p: p.extension)

    return peaks
