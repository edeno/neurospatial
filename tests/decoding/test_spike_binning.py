"""Tests for the public temporal spike-binner ``bin_spikes_in_time``."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from neurospatial import Environment
from neurospatial.decoding import bin_spikes_in_time, decode_position


def test_bin_spikes_counts_hand_computed() -> None:
    """A 2-neuron fixture yields the exact count matrix and bin_centers."""
    # Edges at [0.0, 0.025, 0.05, 0.075, 0.1] -> 4 half-open bins.
    spike_trains = [
        np.array([0.01, 0.06, 0.07]),  # neuron 0: bin0=1, bin2=2
        np.array([0.03, 0.09]),  # neuron 1: bin1=1, bin3=1
    ]
    counts, bin_centers = bin_spikes_in_time(
        spike_trains, dt=0.025, t_start=0.0, t_stop=0.1
    )

    expected = np.array(
        [[1, 0], [0, 1], [2, 0], [0, 1]],
        dtype=np.int64,
    )
    assert_array_equal(counts, expected)
    assert counts.dtype == np.int64

    edges = np.arange(0.0, 0.1 + 0.025, 0.025)
    assert_allclose(bin_centers, edges[:-1] + 0.025 / 2.0)
    assert_allclose(bin_centers, [0.0125, 0.0375, 0.0625, 0.0875])


def test_bin_spikes_orient_transposes() -> None:
    """neuron_x_time is the exact transpose of time_x_neuron with swapped shape."""
    spike_trains = [
        np.array([0.01, 0.06, 0.07]),
        np.array([0.03, 0.09]),
    ]
    counts_tn, centers_tn = bin_spikes_in_time(
        spike_trains, dt=0.025, t_start=0.0, t_stop=0.1, orient="time_x_neuron"
    )
    counts_nt, centers_nt = bin_spikes_in_time(
        spike_trains, dt=0.025, t_start=0.0, t_stop=0.1, orient="neuron_x_time"
    )

    n_time, n_neuron = 4, 2
    assert counts_tn.shape == (n_time, n_neuron)
    assert counts_nt.shape == (n_neuron, n_time)
    assert_array_equal(counts_nt, counts_tn.T)
    # bin_centers do not depend on orientation.
    assert_allclose(centers_tn, centers_nt)


def test_bin_spikes_rejects_bad_dt() -> None:
    """dt <= 0, non-finite dt, and t_stop <= t_start raise ValueError."""
    spike_trains = [np.array([0.01, 0.05])]

    with pytest.raises(ValueError, match="dt must be finite and > 0"):
        bin_spikes_in_time(spike_trains, dt=0.0)

    with pytest.raises(ValueError, match="dt must be finite and > 0"):
        bin_spikes_in_time(spike_trains, dt=-0.025)

    with pytest.raises(ValueError, match="dt must be finite and > 0"):
        bin_spikes_in_time(spike_trains, dt=np.inf)

    with pytest.raises(ValueError, match="dt must be finite and > 0"):
        bin_spikes_in_time(spike_trains, dt=np.nan)

    with pytest.raises(ValueError, match="must be > t_start"):
        bin_spikes_in_time(spike_trains, dt=0.025, t_start=1.0, t_stop=0.5)

    with pytest.raises(ValueError, match="must be > t_start"):
        bin_spikes_in_time(spike_trains, dt=0.025, t_start=1.0, t_stop=1.0)


def test_bin_spikes_rejects_non_numeric_and_bool_dt() -> None:
    """Non-numeric (incl. numeric string) and bool dt raise a clean ValueError.

    Before the shared ``validate_dt`` guard, a numeric STRING leaked a raw
    ``TypeError`` from ``"0.1" <= 0`` and ``dt=True`` was silently accepted
    (``True <= 0`` is False; ``isfinite(True)`` is True) and used as a chunk
    size of 1. Both now raise the same ``ValueError`` as the other decoding
    entry points.
    """
    spike_trains = [np.array([0.01, 0.05])]

    # Numeric string (would otherwise be silently float()-able or raw TypeError).
    with pytest.raises(ValueError, match="dt must"):
        bin_spikes_in_time(spike_trains, dt="0.1")  # type: ignore[arg-type]

    # Non-numeric string.
    with pytest.raises(ValueError, match="dt must"):
        bin_spikes_in_time(spike_trains, dt="abc")  # type: ignore[arg-type]

    # bool: NOT silently accepted as chunk size 1.
    with pytest.raises(ValueError, match="dt must"):
        bin_spikes_in_time(spike_trains, dt=True)  # type: ignore[arg-type]

    # Numeric-but-invalid values still raise (regression on the shared helper).
    for bad in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError, match="dt must"):
            bin_spikes_in_time(spike_trains, dt=bad)


def test_bin_spikes_valid_dt_still_works() -> None:
    """A valid dt still bins normally (regression after the shared guard)."""
    spike_trains = [np.array([0.01, 0.06, 0.07]), np.array([0.03, 0.09])]
    counts, bin_centers = bin_spikes_in_time(
        spike_trains, dt=0.025, t_start=0.0, t_stop=0.1
    )
    assert counts.shape == (4, 2)
    assert_allclose(bin_centers, [0.0125, 0.0375, 0.0625, 0.0875])


def test_bin_spikes_rejects_bad_orient() -> None:
    """An unknown orient value raises ValueError naming the allowed options."""
    spike_trains = [np.array([0.01, 0.05])]
    with pytest.raises(ValueError, match="orient must be"):
        bin_spikes_in_time(
            spike_trains,
            dt=0.025,
            t_start=0.0,
            t_stop=0.1,
            orient="bogus",  # type: ignore[arg-type]
        )


def test_bin_spikes_feeds_decode_position() -> None:
    """Default-orient output is accepted by decode_position without reshaping."""
    positions = np.array(
        [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]],
        dtype=np.float64,
    )
    env = Environment.from_samples(positions, bin_size=5.0)
    # Two neurons with positive, finite place fields over every bin.
    encoding_models = np.array(
        [np.full(env.n_bins, 5.0), np.full(env.n_bins, 3.0)],
        dtype=np.float64,
    )

    # Two neurons spiking within a 0.1 s window; default orient is
    # time_x_neuron, exactly what decode_position consumes.
    spike_trains = [
        np.array([0.01, 0.06, 0.07]),
        np.array([0.03, 0.09]),
    ]
    counts, bin_centers = bin_spikes_in_time(
        spike_trains, dt=0.025, t_start=0.0, t_stop=0.1
    )
    assert counts.shape == (4, 2)
    assert encoding_models.shape[0] == counts.shape[1]

    result = decode_position(
        env,
        counts,
        encoding_models,
        dt=0.025,
        times=bin_centers,
    )
    # Posterior is one normalized distribution per time bin.
    assert result.posterior.shape == (counts.shape[0], env.n_bins)
    assert_allclose(result.posterior.sum(axis=1), 1.0, atol=1e-10)


def test_bin_spikes_empty_neuron() -> None:
    """A neuron with no spikes yields an all-zero column, not a crash."""
    spike_trains = [
        np.array([0.01, 0.06]),  # neuron 0 has spikes
        np.array([]),  # neuron 1 is silent
    ]
    counts, bin_centers = bin_spikes_in_time(
        spike_trains, dt=0.025, t_start=0.0, t_stop=0.1
    )
    assert counts.shape == (4, 2)
    # The silent neuron contributes an all-zero column.
    assert_array_equal(counts[:, 1], np.zeros(4, dtype=np.int64))
    # The active neuron is counted correctly: bin0 and bin2.
    assert_array_equal(counts[:, 0], np.array([1, 0, 1, 0], dtype=np.int64))
    assert bin_centers.shape == (4,)


def test_bin_spikes_default_bounds_span_all_spikes() -> None:
    """Default t_start/t_stop span the min/max spike time across neurons."""
    spike_trains = [
        np.array([0.10, 0.40]),
        np.array([0.20, 0.55]),
    ]
    counts, bin_centers = bin_spikes_in_time(spike_trains, dt=0.1)
    # t_start defaults to 0.10 (global min); t_stop defaults to global max
    # (0.55) + dt = 0.65, giving floor((0.65 - 0.10) / 0.1) = 5 bins with
    # edges [0.10, 0.20, 0.30, 0.40, 0.50, 0.60].
    assert counts.shape == (5, 2)
    assert_allclose(bin_centers[0], 0.10 + 0.1 / 2.0)
    # Every spike is accounted for in the totals (the +dt margin ensures the
    # last spike at 0.55 lands in bin 4 rather than being dropped).
    assert int(counts.sum()) == 4


def test_bin_spikes_no_bin_past_t_stop() -> None:
    """A non-multiple span drops the trailing partial bin; no edge exceeds t_stop."""
    # t_start=0, t_stop=0.09, dt=0.025 -> floor(0.09 / 0.025) = 3 bins.
    # Edges are [0.0, 0.025, 0.05, 0.075]; grid covers [0.0, 0.075], which is
    # strictly inside t_stop=0.09. A spike at 0.095 is past the grid entirely.
    spike_trains = [
        np.array([0.01, 0.06, 0.095]),  # 0.095 is beyond the covered grid
    ]
    counts, bin_centers = bin_spikes_in_time(
        spike_trains, dt=0.025, t_start=0.0, t_stop=0.09
    )

    # Exactly 3 whole bins; no bin extends past t_stop.
    assert counts.shape == (3, 1)
    # 0.01 -> bin0, 0.06 -> bin2, 0.095 -> excluded (past final edge 0.075).
    assert_array_equal(counts[:, 0], np.array([1, 0, 1], dtype=np.int64))
    assert int(counts.sum()) == 2

    edges = 0.0 + 0.025 * np.arange(4)
    assert_allclose(edges, [0.0, 0.025, 0.05, 0.075])
    # Final edge is <= t_stop (here strictly less, since 0.09 is not a multiple).
    assert edges[-1] <= 0.09
    assert_allclose(bin_centers, edges[:-1] + 0.025 / 2.0)
    assert_allclose(bin_centers, [0.0125, 0.0375, 0.0625])


def test_bin_spikes_single_spike_auto_bounds() -> None:
    """A single neuron with one spike under auto bounds yields a valid 1-bin result."""
    spike_trains = [np.array([0.42])]
    # t_start defaults to 0.42, t_stop to 0.42 + dt = 0.52 -> exactly 1 bin.
    counts, bin_centers = bin_spikes_in_time(spike_trains, dt=0.1)

    assert counts.shape == (1, 1)
    # The lone spike lands in the single bin (edges [0.42, 0.52]).
    assert_array_equal(counts[:, 0], np.array([1], dtype=np.int64))
    assert_allclose(bin_centers, [0.42 + 0.1 / 2.0])


def test_bin_spikes_span_smaller_than_dt_raises() -> None:
    """An explicit span narrower than one bin raises a clear ValueError."""
    spike_trains = [np.array([0.01])]
    with pytest.raises(ValueError, match="smaller than one"):
        bin_spikes_in_time(spike_trains, dt=0.1, t_start=0.0, t_stop=0.05)
