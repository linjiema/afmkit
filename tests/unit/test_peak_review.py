"""Unit tests for :mod:`afmkit.analysis.peak_review`.

The peak-review state machine is the v0.3 bridge between auto-detection
and the CSV / Markdown export. These tests pin down the public
contract:

- :class:`ReviewedPeak` is a frozen dataclass with the documented
  defaults and the ``force`` / ``extension`` / ``confidence``
  properties that delegate correctly.
- :class:`PeakReviewer` is a list-like container over the auto-detected
  peaks, with the expected transition methods (``accept``, ``reject``,
  ``override``, ``clear_override``, ``set_note``).
- :meth:`PeakReviewer.re_fit` produces a converged :class:`FitResult`
  on a synthetic WLC curve and updates the peak's ``manual_force`` to
  the fit's y-value at the peak's extension.
- :meth:`PeakReviewer.to_dict` produces a row-per-peak dict list with
  the user-visible columns required by the future CSV exporter.
- :meth:`PeakReviewer.reindex` renumbers the ``peak.index`` column on
  every :class:`ReviewedPeak` after a soft-delete.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from afmkit.analysis import Peak, PeakReviewer, ReviewedPeak, find_sawtooth_peaks
from afmkit.core.curve import ForceCurve
from afmkit.fitting.report import FitResult

# -- Helpers --------------------------------------------------------------


def _peak(
    *,
    index: int = 0,
    extension: float = 50.0,
    force: float = 20.0,
    prominence: float = 10.0,
    width: int = 5,
    height_drop: float = 5.0,
    confidence: float = 0.6,
) -> Peak:
    """Build a :class:`Peak` with sensible defaults for the tests."""
    return Peak(
        index=index,
        extension=extension,
        force=force,
        prominence=prominence,
        width=width,
        height_drop=height_drop,
        confidence=confidence,
    )


def _flat_curve_with_spike(
    *,
    n: int = 401,
    ext_min: float = 0.0,
    ext_max: float = 400.0,
    baseline: float = 10.0,
    spike_index: int = 200,
    spike_height: float = 50.0,
) -> ForceCurve:
    """A flat-baseline curve with one tall spike — easy to fit a constant to.

    Used by the re-fit tests: the WLC model on a flat baseline is
    ill-defined (it has a singularity at ``x=L``), so for the
    re-fit round-trip we use a small local x_range and check that
    the fit converges to something close to the spike's height.
    """
    ext = np.linspace(ext_min, ext_max, n)
    force = np.full(n, baseline, dtype=np.float64)
    # Add a sharp, narrow spike at the requested index. The shape is
    # intentionally simple — a single tall point — so the fit's
    # y-value at the peak's x is exactly the baseline if the fit
    # ignores the spike (i.e. the WLC over the surrounding flat
    # baseline converges to the baseline value).
    force[spike_index] = baseline + spike_height
    return ForceCurve(ext, force, metadata={"k_cantilever": 0.1})


# -- ReviewedPeak ---------------------------------------------------------


class TestReviewedPeak:
    """Frozen dataclass contract: defaults + property delegation."""

    def test_defaults(self) -> None:
        p = _peak(force=20.0)
        rp = ReviewedPeak(peak=p)
        assert rp.accepted is True
        assert rp.manual_force is None
        assert rp.note == ""

    def test_force_property_returns_peak_force_when_no_override(self) -> None:
        p = _peak(force=20.0)
        rp = ReviewedPeak(peak=p)
        assert rp.force == 20.0
        assert rp.manual_force is None

    def test_force_property_returns_manual_override_when_set(self) -> None:
        p = _peak(force=20.0)
        rp = ReviewedPeak(peak=p, manual_force=45.2)
        assert rp.force == 45.2
        # The auto-detected peak.force is unchanged.
        assert rp.peak.force == 20.0

    def test_extension_and_confidence_delegate(self) -> None:
        p = _peak(extension=75.5, confidence=0.83)
        rp = ReviewedPeak(peak=p)
        assert rp.extension == 75.5
        assert rp.confidence == 0.83

    def test_is_frozen(self) -> None:
        rp = ReviewedPeak(peak=_peak())
        with pytest.raises(dataclasses.FrozenInstanceError):
            rp.accepted = False  # type: ignore[misc]


# -- PeakReviewerInit -----------------------------------------------------


class TestPeakReviewerInit:
    """Container protocol + initial state of the reviewer's items."""

    def test_len_matches_input_peaks(self) -> None:
        peaks = [_peak(index=i, extension=float(i * 10)) for i in range(5)]
        reviewer = PeakReviewer(peaks, _flat_curve_with_spike())
        assert len(reviewer) == 5

    def test_iter_yields_all_reviewed_peaks(self) -> None:
        peaks = [_peak(index=i, extension=float(i * 10)) for i in range(3)]
        reviewer = PeakReviewer(peaks, _flat_curve_with_spike())
        items = list(reviewer)
        assert len(items) == 3
        # All items start as accepted with no override.
        for item in items:
            assert isinstance(item, ReviewedPeak)
            assert item.accepted is True
            assert item.manual_force is None

    def test_getitem_returns_reviewed_peak(self) -> None:
        peaks = [_peak(index=i, extension=float(i * 10)) for i in range(3)]
        reviewer = PeakReviewer(peaks, _flat_curve_with_spike())
        rp = reviewer[1]
        assert isinstance(rp, ReviewedPeak)
        assert rp.peak.index == 1
        # Out-of-range raises IndexError (Python's list semantics).
        with pytest.raises(IndexError):
            _ = reviewer[99]

    def test_initial_accepted_contains_all_peaks(self) -> None:
        peaks = [_peak(index=i) for i in range(4)]
        reviewer = PeakReviewer(peaks, _flat_curve_with_spike())
        accepted = reviewer.accepted
        assert len(accepted) == 4

    def test_initial_rejected_is_empty(self) -> None:
        peaks = [_peak(index=i) for i in range(4)]
        reviewer = PeakReviewer(peaks, _flat_curve_with_spike())
        assert reviewer.rejected == []


# -- PeakReviewerStateTransitions -----------------------------------------


class TestPeakReviewerStateTransitions:
    """accept / reject / override / clear_override / set_note behaviour."""

    def test_accept_is_idempotent(self) -> None:
        peaks = [_peak(index=i) for i in range(3)]
        reviewer = PeakReviewer(peaks, _flat_curve_with_spike())
        # Already accepted; calling accept() again must not change state
        # and must not raise.
        reviewer.accept(0)
        reviewer.accept(0)
        reviewer.accept(0)
        assert reviewer[0].accepted is True

    def test_reject_flips_accepted(self) -> None:
        peaks = [_peak(index=i) for i in range(3)]
        reviewer = PeakReviewer(peaks, _flat_curve_with_spike())
        reviewer.reject(1)
        assert reviewer[1].accepted is False
        # The peak still sits in the list (soft delete).
        assert len(reviewer) == 3
        # The accepted / rejected views reflect the new state.
        assert len(reviewer.accepted) == 2
        assert len(reviewer.rejected) == 1
        assert reviewer.rejected[0].peak.index == 1

    def test_reject_then_accept_reverts(self) -> None:
        peaks = [_peak(index=i) for i in range(3)]
        reviewer = PeakReviewer(peaks, _flat_curve_with_spike())
        reviewer.reject(0)
        assert reviewer[0].accepted is False
        reviewer.accept(0)
        assert reviewer[0].accepted is True

    def test_override_sets_manual_force(self) -> None:
        peaks = [_peak(index=0, force=20.0)]
        reviewer = PeakReviewer(peaks, _flat_curve_with_spike())
        reviewer.override(0, 45.2)
        assert reviewer[0].manual_force == 45.2
        assert reviewer[0].force == 45.2

    def test_override_rejects_non_finite(self) -> None:
        peaks = [_peak(index=0)]
        reviewer = PeakReviewer(peaks, _flat_curve_with_spike())
        with pytest.raises(ValueError):
            reviewer.override(0, float("nan"))
        with pytest.raises(ValueError):
            reviewer.override(0, float("inf"))
        with pytest.raises(ValueError):
            reviewer.override(0, float("-inf"))

    def test_override_rejects_non_positive(self) -> None:
        peaks = [_peak(index=0)]
        reviewer = PeakReviewer(peaks, _flat_curve_with_spike())
        with pytest.raises(ValueError):
            reviewer.override(0, 0.0)
        with pytest.raises(ValueError):
            reviewer.override(0, -1.0)

    def test_clear_override_reverts_to_peak_force(self) -> None:
        peaks = [_peak(index=0, force=20.0)]
        reviewer = PeakReviewer(peaks, _flat_curve_with_spike())
        reviewer.override(0, 45.2)
        assert reviewer[0].force == 45.2
        reviewer.clear_override(0)
        assert reviewer[0].manual_force is None
        assert reviewer[0].force == 20.0

    def test_clear_override_on_unset_is_noop(self) -> None:
        peaks = [_peak(index=0, force=20.0)]
        reviewer = PeakReviewer(peaks, _flat_curve_with_spike())
        # No exception even though there's no override to clear.
        reviewer.clear_override(0)
        assert reviewer[0].manual_force is None
        assert reviewer[0].force == 20.0

    def test_set_note_attaches_annotation(self) -> None:
        peaks = [_peak(index=0)]
        reviewer = PeakReviewer(peaks, _flat_curve_with_spike())
        reviewer.set_note(0, "near cantilever jump-off, suspicious")
        assert reviewer[0].note == "near cantilever jump-off, suspicious"

    def test_state_transitions_out_of_range_raise(self) -> None:
        peaks = [_peak(index=0)]
        reviewer = PeakReviewer(peaks, _flat_curve_with_spike())
        with pytest.raises(IndexError):
            reviewer.accept(99)
        with pytest.raises(IndexError):
            reviewer.reject(99)
        with pytest.raises(IndexError):
            reviewer.override(99, 10.0)
        with pytest.raises(IndexError):
            reviewer.clear_override(99)
        with pytest.raises(IndexError):
            reviewer.set_note(99, "x")


# -- PeakReviewerRefit ----------------------------------------------------


class TestPeakReviewerRefit:
    """Re-fit round-trip: FitResult + manual_force update + error paths."""

    def test_re_fit_returns_fitresult_and_updates_manual_force(
        self, synthetic_force_curve: ForceCurve
    ) -> None:
        # Build a small peak list with one peak at ~50 nm on the
        # synthetic WLC curve. The WLC fit on a clean curve is
        # very stable, so the recovered force at the peak's x
        # is essentially the truth (modulo a tiny rounding error
        # from the line interpolation in `_interp_at`).
        peak = _peak(index=0, extension=50.0, force=15.0)
        reviewer = PeakReviewer([peak], synthetic_force_curve)
        result = reviewer.re_fit(0, x_range=(20.0, 80.0))
        assert isinstance(result, FitResult)
        assert bool(result.metadata.get("success", False))
        # The manual_force is updated — not the auto-detected force.
        assert reviewer[0].manual_force is not None
        assert reviewer[0].force == reviewer[0].manual_force
        # The force after re-fit should be close to the truth
        # (the WLC value at x=50 on a p=0.4, L=200 curve is well
        # within a few pN of the actual data — synthetic data is
        # noise-free).
        truth_at_50 = (4.1 / 0.4) * (0.25 * (1.0 - 50.0 / 200.0) ** -2 - 0.25 + 50.0 / 200.0)
        assert abs(reviewer[0].force - truth_at_50) < 2.0  # within 2 pN

    def test_re_fit_default_window_centred_on_peak(self, synthetic_force_curve: ForceCurve) -> None:
        peak = _peak(index=0, extension=100.0)
        reviewer = PeakReviewer([peak], synthetic_force_curve)
        # No x_range → default ±20 nm around the peak's extension.
        result = reviewer.re_fit(0)
        assert isinstance(result, FitResult)
        assert bool(result.metadata.get("success", False))
        # The fit's metadata should surface the default window.
        fit_range = result.metadata.get("fit_range")
        assert fit_range is not None
        # The default window is 100 ± 20 = (80, 120).
        assert abs(fit_range[0] - 80.0) < 1e-9
        assert abs(fit_range[1] - 120.0) < 1e-9

    def test_re_fit_bad_index_raises_index_error(self, synthetic_force_curve: ForceCurve) -> None:
        peaks = [_peak(index=0, extension=50.0)]
        reviewer = PeakReviewer(peaks, synthetic_force_curve)
        with pytest.raises(IndexError):
            reviewer.re_fit(5)
        with pytest.raises(IndexError):
            reviewer.re_fit(-1)

    def test_re_fit_x_range_outside_curve_raises_value_error(
        self, synthetic_force_curve: ForceCurve
    ) -> None:
        # The synthetic curve's extension axis is 0–300 nm. A
        # window entirely above 300 nm must raise — the engine
        # would also raise ("x_range selects no data points"),
        # but the message here is friendlier.
        peak = _peak(index=0, extension=50.0)
        reviewer = PeakReviewer([peak], synthetic_force_curve)
        with pytest.raises(ValueError):
            reviewer.re_fit(0, x_range=(500.0, 600.0))

    def test_re_fit_then_reindex_keeps_indices_contiguous(
        self, synthetic_force_curve: ForceCurve
    ) -> None:
        # Three peaks; reject the middle one and reindex.
        peaks = [
            _peak(index=10, extension=50.0),
            _peak(index=20, extension=100.0),
            _peak(index=30, extension=150.0),
        ]
        reviewer = PeakReviewer(peaks, synthetic_force_curve)
        reviewer.reject(1)
        reviewer.reindex()
        # After reindex, every peak.index equals its list position.
        for i, rp in enumerate(reviewer):
            assert rp.peak.index == i
        # The previously-rejected peak still has accepted=False.
        assert reviewer[1].accepted is False

    def test_to_dict_after_re_fit_carries_manual_force(
        self, synthetic_force_curve: ForceCurve
    ) -> None:
        peak = _peak(index=0, extension=50.0, force=15.0)
        reviewer = PeakReviewer([peak], synthetic_force_curve)
        reviewer.re_fit(0, x_range=(20.0, 80.0))
        rows = reviewer.to_dict()
        assert len(rows) == 1
        assert rows[0]["manual_force"] is not None
        assert rows[0]["force"] == rows[0]["manual_force"]


# -- PeakReviewerToDict ---------------------------------------------------


class TestPeakReviewerToDict:
    """Dict export: columns, types, and round-trip stability."""

    def test_to_dict_columns_and_types(self) -> None:
        # Build a small, diverse set of peaks so the export has a
        # mix of accepted / rejected / override / note / no-override.
        peaks = [
            _peak(
                index=0,
                extension=20.0,
                force=12.0,
                prominence=8.0,
                width=4,
                height_drop=3.0,
                confidence=0.55,
            ),
            _peak(
                index=1,
                extension=60.0,
                force=18.0,
                prominence=12.0,
                width=6,
                height_drop=5.0,
                confidence=0.78,
            ),
            _peak(
                index=2,
                extension=100.0,
                force=22.0,
                prominence=10.0,
                width=5,
                height_drop=4.0,
                confidence=0.66,
            ),
        ]
        reviewer = PeakReviewer(peaks, _flat_curve_with_spike())
        reviewer.reject(1)
        reviewer.override(0, 13.5)
        reviewer.set_note(2, "looks like a doublet")

        rows = reviewer.to_dict()
        assert len(rows) == 3

        # Every row has the documented column set, in any order.
        expected_keys = {
            "index",
            "extension",
            "force",
            "manual_force",
            "accepted",
            "confidence",
            "prominence",
            "width",
            "height_drop",
            "note",
        }
        for row in rows:
            assert set(row.keys()) == expected_keys

        # Row 0: accepted, with override.
        assert rows[0]["index"] == 0
        assert rows[0]["extension"] == 20.0
        assert rows[0]["force"] == 13.5  # override wins
        assert rows[0]["manual_force"] == 13.5
        assert rows[0]["accepted"] is True
        assert isinstance(rows[0]["accepted"], bool)
        assert isinstance(rows[0]["manual_force"], float)
        assert rows[0]["note"] == ""

        # Row 1: rejected, no override.
        assert rows[1]["index"] == 1
        assert rows[1]["accepted"] is False
        assert isinstance(rows[1]["accepted"], bool)
        assert rows[1]["manual_force"] is None
        assert rows[1]["force"] == 18.0  # falls back to peak.force

        # Row 2: accepted, no override, with note.
        assert rows[2]["index"] == 2
        assert rows[2]["accepted"] is True
        assert rows[2]["manual_force"] is None
        assert rows[2]["force"] == 22.0
        assert rows[2]["note"] == "looks like a doublet"

    def test_to_dict_empty_reviewer(self) -> None:
        reviewer = PeakReviewer([], _flat_curve_with_spike())
        assert reviewer.to_dict() == []

    def test_to_dict_is_jsonable(self) -> None:
        # A smoke test: the dict can be round-tripped through
        # json.dumps. numpy scalars in the values would otherwise
        # raise a TypeError — this catches accidental regressions.
        import json

        peaks = [
            _peak(
                index=0,
                extension=20.0,
                force=12.0,
                prominence=8.0,
                width=4,
                height_drop=3.0,
                confidence=0.55,
            )
        ]
        reviewer = PeakReviewer(peaks, _flat_curve_with_spike())
        rows = reviewer.to_dict()
        # Should not raise.
        encoded = json.dumps(rows)
        assert isinstance(encoded, str)
        assert len(encoded) > 0


# -- Integration with the real peak detector ------------------------------


class TestIntegrationWithFindSawtoothPeaks:
    """Round-trip: build a peak list from the real detector, then review it."""

    def test_real_detector_peeds_into_reviewer(self) -> None:
        # Build a flat-baseline curve with three injected Gaussian
        # spikes (the same pattern used in test_peak_detection) and
        # confirm find_sawtooth_peaks() + PeakReviewer() cooperate.
        ext = np.linspace(0.0, 300.0, 2000)
        force = np.zeros_like(ext)
        centers = (400, 800, 1200)
        sigma = 3.0
        amp = 30.0
        for c in centers:
            force += amp * np.exp(-0.5 * ((np.arange(ext.size) - c) / sigma) ** 2)
        curve = ForceCurve(ext, force, metadata={"k_cantilever": 0.1})
        peaks = find_sawtooth_peaks(curve, min_prominence_pN=5.0, min_width_points=3)
        # All three spikes should be detected on this noise-free data.
        assert len(peaks) == 3
        reviewer = PeakReviewer(peaks, curve)
        assert len(reviewer) == 3
        # Reject the middle peak; the export reflects the decision.
        reviewer.reject(1)
        rows = reviewer.to_dict()
        assert rows[0]["accepted"] is True
        assert rows[1]["accepted"] is False
        assert rows[2]["accepted"] is True
