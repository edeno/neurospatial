"""Tests for theta-phase extraction and the ripple-detection scope-out docs.

Covers the three validation-slice checks:

* ``test_theta_phase_monotonic`` — on a pure theta sinusoid the extracted
  phase advances ~linearly mod 2*pi and feeds ``phase_precession`` unchanged.
* ``test_replay_docstrings_point_to_ripple_detection`` — the decoding replay
  (trajectory) docstrings direct users to the external ``ripple_detection``
  package.
* ``test_ripple_detection_example_feeds_psth`` — skippable end-to-end check
  that ``ripple_detection`` intervals feed ``peri_event_histogram``.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest


def _theta_sinusoid(
    frequency: float = 8.0,
    sampling_rate: float = 1000.0,
    duration: float = 5.0,
) -> tuple[np.ndarray, float]:
    """Return (lfp, sampling_rate) for a pure-theta sinusoid."""
    t = np.arange(0, duration, 1 / sampling_rate)
    lfp = np.sin(2 * np.pi * frequency * t)
    return lfp, sampling_rate


class TestThetaPhase:
    """Direct checks of ``encoding.theta_phase``."""

    def test_exported_from_encoding(self) -> None:
        """theta_phase is exported from the encoding package namespace."""
        import neurospatial.encoding as enc

        assert hasattr(enc, "theta_phase")
        assert "theta_phase" in enc.__all__

    def test_theta_phase_monotonic(self) -> None:
        """Phase advances ~linearly mod 2*pi on a pure sinusoid.

        On a clean ``frequency`` Hz sinusoid the instantaneous phase should
        complete exactly one cycle per period, so the *unwrapped* phase is a
        straight line with slope ``2*pi*frequency`` rad/s.
        """
        from neurospatial.encoding import theta_phase

        frequency = 8.0
        sampling_rate = 1000.0
        lfp, _ = _theta_sinusoid(frequency, sampling_rate, duration=5.0)

        phase = theta_phase(lfp, sampling_rate, band=(6, 10))

        # Shape and range: one value per sample, wrapped to [0, 2*pi).
        assert phase.shape == lfp.shape
        assert phase.min() >= 0.0
        assert phase.max() < 2 * np.pi

        # Drop filter edge transients before checking linearity.
        edge = int(0.25 * sampling_rate)
        unwrapped = np.unwrap(phase)[edge:-edge]
        t = np.arange(unwrapped.size) / sampling_rate

        # Slope of the unwrapped phase matches 2*pi*frequency (monotonic,
        # ~linear increase). Fit a line and compare the slope.
        slope = np.polyfit(t, unwrapped, 1)[0]
        assert slope == pytest.approx(2 * np.pi * frequency, rel=0.02)

        # Strictly increasing (monotonic) interior.
        assert np.all(np.diff(unwrapped) > 0)

    def test_feeds_phase_precession_without_reshaping(self) -> None:
        """theta_phase output is drop-in for phase_precession (no reshaping)."""
        from neurospatial.encoding import phase_precession, theta_phase

        sampling_rate = 1000.0
        lfp, _ = _theta_sinusoid(8.0, sampling_rate, duration=2.0)
        phase = theta_phase(lfp, sampling_rate)

        positions = np.linspace(0.0, 50.0, phase.size)
        result = phase_precession(positions, phase, rng=0)

        # Consumed without error and returns the standard result type.
        assert isinstance(result.slope, float)
        assert 0.0 <= result.pval <= 1.0

    def test_rejects_invalid_inputs(self) -> None:
        """Bad shape / sampling_rate / band raise clear ValueErrors."""
        from neurospatial.encoding import theta_phase

        lfp, sr = _theta_sinusoid()

        with pytest.raises(ValueError, match="1-D"):
            theta_phase(lfp.reshape(-1, 1), sr)
        with pytest.raises(ValueError, match="positive"):
            theta_phase(lfp, -1.0)
        with pytest.raises(ValueError, match="low < high"):
            theta_phase(lfp, sr, band=(10.0, 6.0))
        with pytest.raises(ValueError, match="Nyquist"):
            theta_phase(lfp, sr, band=(6.0, sr))

    def test_theta_phase_rejects_nonfinite_lfp(self) -> None:
        """NaN or Inf in lfp raises (else filtfilt silently returns all-NaN)."""
        from neurospatial.encoding import theta_phase

        lfp, sr = _theta_sinusoid()

        lfp_nan = lfp.copy()
        lfp_nan[100] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            theta_phase(lfp_nan, sr)

        lfp_inf = lfp.copy()
        lfp_inf[100] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            theta_phase(lfp_inf, sr)

    def test_theta_phase_rejects_too_short_lfp(self) -> None:
        """An lfp shorter than the filter padlen raises a clear ValueError.

        The message must talk about length / too short, not leak scipy's
        opaque ``padlen`` message as the primary explanation.
        """
        from neurospatial.encoding import theta_phase

        sampling_rate = 1000.0
        # 4th-order Butterworth bandpass -> len(b)=len(a)=9, padlen=27.
        short_lfp = np.sin(2 * np.pi * 8.0 * np.arange(20) / sampling_rate)
        with pytest.raises(ValueError, match="too short"):
            theta_phase(short_lfp, sampling_rate)


class TestRippleDetectionScopeOut:
    """The replay path documents the external ripple_detection package."""

    def test_replay_docstrings_point_to_ripple_detection(self) -> None:
        """Decoding replay (trajectory) docstrings mention ripple_detection."""
        import neurospatial.decoding.trajectory as trajectory

        module_doc = inspect.getdoc(trajectory) or ""
        assert "ripple_detection" in module_doc
        # And it ties the intervals to the PSTH entry point.
        assert "peri_event_histogram" in module_doc

    def test_ripple_detection_example_feeds_psth(self) -> None:
        """ripple_detection intervals feed peri_event_histogram end-to-end."""
        pytest.importorskip("ripple_detection")

        import numpy as np
        from ripple_detection import Kay_ripple_detector

        from neurospatial.events import peri_event_histogram

        # Synthesize a band-limited LFP with an injected high-frequency burst
        # so the detector returns at least one interval.
        sampling_frequency = 1500.0
        duration = 4.0
        time = np.arange(0, duration, 1 / sampling_frequency)
        rng = np.random.default_rng(0)
        lfp = 0.05 * rng.standard_normal(time.size)
        burst = (time > 2.0) & (time < 2.06)
        lfp[burst] += np.sin(2 * np.pi * 200.0 * time[burst])
        lfps = lfp[:, np.newaxis]
        speed = np.zeros_like(time)  # animal still -> ripples allowed

        ripple_times = Kay_ripple_detector(time, lfps, speed, sampling_frequency)
        ripple_times = np.asarray(ripple_times)

        if ripple_times.size == 0:
            pytest.skip("ripple_detection returned no intervals on this fixture")

        # ripple_detection returns (start, end) pairs in seconds; the start
        # times are the events fed to the PSTH.
        ripple_starts = ripple_times[:, 0].astype(np.float64)
        spike_times = np.sort(rng.uniform(0, duration, 200)).astype(np.float64)

        result = peri_event_histogram(
            spike_times, ripple_starts, window=(-0.5, 0.5), bin_size=0.05
        )
        # Standard PSTH result consumed without reshaping.
        assert result.firing_rate.ndim == 1
        assert result.bin_centers.shape == result.firing_rate.shape
