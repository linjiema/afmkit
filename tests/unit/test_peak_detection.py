"""Unit tests for :mod:`afmkit.analysis.peak_detection`.

These tests cover the public surface of the v0.2 sawtooth peak
detector:

- Empty / too-short curves return ``[]`` (no exceptions).
- A single synthetic peak is detected at the right position.
- Multi-peak curves return all peaks in extension order.
- ``max_peaks``, ``min_prominence_pN``, ``min_width_points`` all
  behave as documented.
- Smoothing doesn't shift a noise-free peak by more than 2 points.
- ``confidence`` is always in [0, 1].
- The integration test uses the ``synthetic_force_curve`` conftest
  fixture and validates that 3 injected sawtooth spikes are all
  recovered with extension error < 2 nm.

All tests run in well under a second — there is no IO, no
network, and no heavy computation.

Design note
-----------
The synthetic tests use a **flat baseline** (no WLC ramp). Real
SMFS curves have a rising WLC baseline, but the legacy
``AutoFindForcePeaks`` macro also saw a flat baseline in many
calibration / control experiments, and on a flat baseline the
detector's parameters (prominence, width, height_drop) are
unambiguous. The integration test at the bottom of this file is
the one that exercises the rising-baseline case via the
``synthetic_force_curve`` fixture.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from afmkit.analysis import Peak, find_sawtooth_peaks
from afmkit.core.curve import ForceCurve

# -- Helpers --------------------------------------------------------------


def _flat_force(n: int, *, baseline: float = 0.0) -> np.ndarray:
    """A flat zero-baseline force — used as the canvas for synthetic spikes."""
    return np.full(n, float(baseline), dtype=np.float64)


def _add_gaussian_peak(
    force: np.ndarray,
    index: int,
    *,
    amplitude: float,
    sigma_pts: float = 3.0,
) -> np.ndarray:
    """Add a Gaussian spike centered on ``index`` to ``force`` (returns a new array)."""
    out = np.array(force, dtype=np.float64, copy=True)
    x = np.arange(out.size)
    out += amplitude * np.exp(-0.5 * ((x - index) / sigma_pts) ** 2)
    return out


# -- Empty / pathological inputs ------------------------------------------


class TestEmptyAndPathological:
    """Curves that should produce zero peaks without raising."""

    def test_too_short_curve_returns_empty(self) -> None:
        ext = np.linspace(0.0, 10.0, 2)
        force = _flat_force(2)
        curve = ForceCurve(ext, force)
        assert find_sawtooth_peaks(curve) == []

    def test_constant_force_returns_empty(self) -> None:
        # A flat line has no local maxima.
        ext = np.linspace(0.0, 100.0, 200)
        force = _flat_force(200, baseline=10.0)
        curve = ForceCurve(ext, force)
        assert find_sawtooth_peaks(curve) == []


# -- Single peak ---------------------------------------------------------


class TestSinglePeak:
    """A single synthetic peak must be detected with correct metadata."""

    def test_single_peak_position(self) -> None:
        ext = np.linspace(0.0, 200.0, 1000)
        force = _flat_force(1000)
        # Place the spike at index 500 (= extension 100 nm).
        spike_idx = 500
        spike_amp = 30.0
        force = _add_gaussian_peak(force, spike_idx, amplitude=spike_amp, sigma_pts=5.0)
        curve = ForceCurve(ext, force)

        peaks = find_sawtooth_peaks(curve, min_prominence_pN=5.0)
        assert len(peaks) == 1

        peak = peaks[0]
        # The detected peak should be very close to the true center;
        # with smoothing=5 on a noise-free curve, sub-point accuracy
        # is achievable.
        assert abs(peak.index - spike_idx) <= 2
        assert abs(peak.extension - ext[spike_idx]) < 0.5  # nm
        # Force at the peak is the local max of the smoothed curve.
        # On a noise-free curve the smoothed maximum is slightly
        # below the true amplitude (because smoothing widens the
        # spike), so we use a generous lower bound.
        assert peak.force >= spike_amp * 0.85

    def test_single_peak_prominence_close_to_amplitude(self) -> None:
        ext = np.linspace(0.0, 200.0, 1000)
        force = _flat_force(1000)
        spike_idx = 500
        spike_amp = 30.0
        force = _add_gaussian_peak(force, spike_idx, amplitude=spike_amp, sigma_pts=5.0)
        curve = ForceCurve(ext, force)

        peaks = find_sawtooth_peaks(curve, min_prominence_pN=5.0)
        assert len(peaks) == 1
        # With a flat baseline, the prominence should match the
        # peak height within a few percent (smoothing eats a little).
        assert peaks[0].prominence == pytest.approx(spike_amp, rel=0.1)

    def test_single_peak_metadata_dataclass(self) -> None:
        # Verify the Peak dataclass exposes the documented fields
        # with the right types.
        ext = np.linspace(0.0, 200.0, 1000)
        force = _add_gaussian_peak(_flat_force(1000), 500, amplitude=30.0, sigma_pts=5.0)
        curve = ForceCurve(ext, force)
        peaks = find_sawtooth_peaks(curve, min_prominence_pN=5.0)
        assert len(peaks) == 1
        peak = peaks[0]
        assert isinstance(peak, Peak)
        assert isinstance(peak.index, int)
        assert isinstance(peak.extension, float)
        assert isinstance(peak.force, float)
        assert isinstance(peak.prominence, float)
        assert isinstance(peak.width, int)
        assert isinstance(peak.height_drop, float)
        assert isinstance(peak.confidence, float)


# -- Multiple peaks ------------------------------------------------------


class TestMultiplePeaks:
    """Curves with several peaks must return all of them, in order."""

    def test_three_peaks_in_extension_order(self) -> None:
        ext = np.linspace(0.0, 200.0, 1000)
        force = _flat_force(1000)
        spike_indices = [200, 500, 800]
        spike_amp = 30.0
        for idx in spike_indices:
            force = _add_gaussian_peak(force, idx, amplitude=spike_amp, sigma_pts=5.0)
        curve = ForceCurve(ext, force)

        peaks = find_sawtooth_peaks(curve, min_prominence_pN=5.0)
        assert len(peaks) == 3
        # The returned peaks are sorted by extension; verify the
        # ordering matches the input order.
        returned_idx = [p.index for p in peaks]
        assert returned_idx == sorted(returned_idx)
        # The leftmost peak should be near index 200, etc.
        for expected, got in zip(spike_indices, returned_idx, strict=True):
            assert abs(got - expected) <= 2

    def test_five_peaks_all_detected(self) -> None:
        ext = np.linspace(0.0, 200.0, 1000)
        force = _flat_force(1000)
        spike_indices = [100, 300, 500, 700, 900]
        for idx in spike_indices:
            force = _add_gaussian_peak(force, idx, amplitude=25.0, sigma_pts=4.0)
        curve = ForceCurve(ext, force)

        peaks = find_sawtooth_peaks(curve, min_prominence_pN=5.0)
        assert len(peaks) == 5

    def test_max_peaks_truncates_to_top_n(self) -> None:
        ext = np.linspace(0.0, 200.0, 1000)
        force = _flat_force(1000)
        # Five peaks with varied amplitudes: 10, 20, 50, 20, 10.
        amps = [10.0, 20.0, 50.0, 20.0, 10.0]
        spike_indices = [150, 350, 500, 650, 850]
        for idx, amp in zip(spike_indices, amps, strict=True):
            force = _add_gaussian_peak(force, idx, amplitude=amp, sigma_pts=5.0)
        curve = ForceCurve(ext, force)

        # Without the cap, all 5 are found.
        all_peaks = find_sawtooth_peaks(curve, min_prominence_pN=3.0)
        assert len(all_peaks) == 5

        # max_peaks=2 keeps the two highest-prominence ones.
        top2 = find_sawtooth_peaks(curve, min_prominence_pN=3.0, max_peaks=2)
        assert len(top2) == 2
        # The two kept peaks must be sorted by extension.
        assert top2[0].extension < top2[1].extension
        # The mid peak (amp 50) is the most prominent; the two side
        # peaks (amp 20) are next. The two kept peaks are the mid
        # peak and one of the side peaks — find which is which by
        # checking the prominence values.
        prom_values = sorted([p.prominence for p in top2], reverse=True)
        assert prom_values[0] == pytest.approx(50.0, rel=0.05)
        assert prom_values[1] == pytest.approx(20.0, rel=0.05)


# -- Thresholding --------------------------------------------------------


class TestThresholds:
    """``min_prominence_pN`` and ``min_width_points`` must filter properly."""

    def test_min_prominence_filters_noise(self) -> None:
        ext = np.linspace(0.0, 200.0, 1000)
        # Start with a flat baseline.
        force = _flat_force(1000)
        # One large peak (50 pN) and one small "noise" peak (2 pN,
        # below the default 5 pN threshold).
        force = _add_gaussian_peak(force, 300, amplitude=50.0, sigma_pts=5.0)
        force = _add_gaussian_peak(force, 700, amplitude=2.0, sigma_pts=5.0)
        curve = ForceCurve(ext, force)

        peaks = find_sawtooth_peaks(curve, min_prominence_pN=5.0)
        # Only the big peak survives.
        assert len(peaks) == 1
        assert abs(peaks[0].index - 300) <= 2

    def test_lower_min_prominence_keeps_both(self) -> None:
        ext = np.linspace(0.0, 200.0, 1000)
        force = _flat_force(1000)
        force = _add_gaussian_peak(force, 300, amplitude=50.0, sigma_pts=5.0)
        force = _add_gaussian_peak(force, 700, amplitude=4.0, sigma_pts=5.0)
        curve = ForceCurve(ext, force)

        # With a 1 pN threshold, both peaks survive. (The 4 pN
        # peak gets reduced to ~3.5 pN after the 5-point moving
        # average, but 3.5 > 1.)
        peaks = find_sawtooth_peaks(curve, min_prominence_pN=1.0)
        assert len(peaks) == 2

    def test_min_width_filters_narrow_artifacts(self) -> None:
        ext = np.linspace(0.0, 200.0, 1000)
        force = _flat_force(1000)
        # A genuine broad peak (sigma 5 pts → FWHM ~12 pts).
        force = _add_gaussian_peak(force, 300, amplitude=50.0, sigma_pts=5.0)
        # A narrower artifact: sigma 1.5 pts → FWHM ~3.5 pts. With
        # min_width_points=8 the artifact is rejected while the
        # broad peak (FWHM ~12 pts) survives.
        force = _add_gaussian_peak(force, 700, amplitude=50.0, sigma_pts=1.5)
        curve = ForceCurve(ext, force)

        peaks = find_sawtooth_peaks(curve, min_prominence_pN=5.0, min_width_points=8)
        # Only the broad peak survives; the narrow spike is rejected
        # by the width filter.
        assert len(peaks) == 1
        assert abs(peaks[0].index - 300) <= 2


# -- Output invariants ---------------------------------------------------


class TestOutputInvariants:
    """The returned Peak objects must always be well-formed."""

    def test_confidence_in_unit_interval(self) -> None:
        ext = np.linspace(0.0, 200.0, 1000)
        force = _flat_force(1000)
        for idx in (200, 500, 800):
            force = _add_gaussian_peak(force, idx, amplitude=30.0, sigma_pts=5.0)
        curve = ForceCurve(ext, force)

        peaks = find_sawtooth_peaks(curve, min_prominence_pN=5.0)
        assert len(peaks) >= 1
        for peak in peaks:
            assert 0.0 <= peak.confidence <= 1.0

    def test_peak_is_frozen_dataclass(self) -> None:
        # The contract is "Peak is immutable" — verify by trying to
        # assign an attribute and expecting FrozenInstanceError.
        ext = np.linspace(0.0, 200.0, 1000)
        force = _add_gaussian_peak(_flat_force(1000), 500, amplitude=30.0, sigma_pts=5.0)
        curve = ForceCurve(ext, force)
        peaks = find_sawtooth_peaks(curve, min_prominence_pN=5.0)
        assert len(peaks) == 1
        with pytest.raises(dataclasses.FrozenInstanceError):
            peaks[0].force = 999.0  # type: ignore[misc]

    def test_height_drop_nonneg_on_flat_baseline(self) -> None:
        # On a flat baseline, the force drop from each peak to the
        # next (or the curve's last point) is non-negative. (The
        # spec formula ``force[peak] - force[next_peak or last]``
        # is non-negative on a flat baseline; on a rising WLC
        # baseline it can be negative, which is a known limitation
        # documented in the module docstring.)
        ext = np.linspace(0.0, 200.0, 1000)
        force = _flat_force(1000)
        for idx in (200, 500, 800):
            force = _add_gaussian_peak(force, idx, amplitude=30.0, sigma_pts=5.0)
        curve = ForceCurve(ext, force)
        peaks = find_sawtooth_peaks(curve, min_prominence_pN=5.0)
        assert all(p.height_drop >= 0.0 for p in peaks)


# -- Smoothing behaviour -------------------------------------------------


class TestSmoothing:
    """Smoothing must not move a noise-free peak by more than 2 points."""

    def test_smoothing_does_not_shift_peak(self) -> None:
        ext = np.linspace(0.0, 200.0, 1000)
        force = _flat_force(1000)
        spike_idx = 500
        force = _add_gaussian_peak(force, spike_idx, amplitude=30.0, sigma_pts=5.0)
        curve = ForceCurve(ext, force)

        peaks = find_sawtooth_peaks(curve, min_prominence_pN=5.0)
        assert len(peaks) == 1
        # On a noise-free synthetic curve, smoothing of 5 points
        # should not shift the detected peak by more than 2 points.
        assert abs(peaks[0].index - spike_idx) <= 2

    def test_no_smoothing_window_one(self) -> None:
        # smoothing_window=1 disables smoothing — the peak should
        # still be found, possibly with a sharper (narrower) width.
        ext = np.linspace(0.0, 200.0, 1000)
        force = _flat_force(1000)
        force = _add_gaussian_peak(force, 500, amplitude=30.0, sigma_pts=5.0)
        curve = ForceCurve(ext, force)

        peaks = find_sawtooth_peaks(curve, min_prominence_pN=5.0, smoothing_window=1)
        assert len(peaks) == 1
        assert abs(peaks[0].index - 500) <= 1


# -- Integration: synthetic_force_curve + injected sawteeth -------------


class TestIntegrationWithSyntheticFixture:
    """End-to-end: 3 sawteeth injected into the conftest WLC fixture."""

    def test_finds_three_injected_sawteeth_on_wlc(
        self,
        synthetic_extension: np.ndarray,
        synthetic_force_curve: ForceCurve,
    ) -> None:
        # Use the conftest's WLC curve as the baseline, then inject
        # three sawtooth-like spikes. The WLC background is rising
        # (as in a real SMFS curve), so we need a more generous
        # prominence threshold than the flat-baseline tests.
        #
        # The full synthetic extension axis goes 0..300 nm with
        # L=200 nm — so the WLC diverges at extension=200 nm. We
        # restrict the analysis to the well-behaved region 20..150
        # nm to avoid the singularity dominating the peak search.
        # Within that window we inject three spikes at known
        # positions.
        ext_full = synthetic_extension
        wlc_force = synthetic_force_curve.force

        # Map spike target extensions (nm) to indices on the full
        # extension axis. The full axis is 0..300 nm over 5000
        # points, so index = extension_nm * 5000 / 300.
        target_extensions = np.array([40.0, 80.0, 120.0])
        spike_indices = np.round(target_extensions / 300.0 * 5000.0).astype(int)
        spike_amps = [25.0, 35.0, 20.0]

        # Inject spikes into the full WLC.
        new_force = np.array(wlc_force, dtype=np.float64, copy=True)
        for idx, amp in zip(spike_indices, spike_amps, strict=True):
            x = np.arange(new_force.size)
            new_force += amp * np.exp(-0.5 * ((x - idx) / 5.0) ** 2)

        # Restrict to the well-behaved extension range to keep the
        # WLC divergence from dominating the peak search. The
        # right edge is kept below 135 nm so the steep WLC rise
        # near L=200 nm doesn't produce a false "plateau" peak
        # after smoothing.
        curve = ForceCurve(ext_full, new_force, metadata=synthetic_force_curve.metadata)
        curve = curve.select_range(20.0, 135.0)

        peaks = find_sawtooth_peaks(curve, min_prominence_pN=10.0, min_width_points=3)
        assert len(peaks) == 3

        # Each returned peak should be within 2 nm of the true
        # target extension. (We compare to ``target_extensions``
        # directly since the spikes were placed on a grid in nm.)
        for target_ext, peak in zip(target_extensions, peaks, strict=True):
            assert abs(peak.extension - target_ext) < 2.0, (
                f"peak at extension {peak.extension:.2f} nm is more than 2 nm "
                f"away from the injected spike at {target_ext:.2f} nm"
            )
