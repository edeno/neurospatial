"""Circular spike shifts preserve counts and stay on the analyzed clock."""

import numpy as np
import pytest

from neurospatial import stats


def test_count_preserved_and_inside_windows():
    rng = np.random.default_rng(3)
    spikes = np.sort(np.r_[rng.uniform(0, 100, 250), rng.uniform(200, 300, 250)])
    for shifted in stats.shuffle_spike_times_circular(
        spikes, [[0, 100], [200, 300]], n_shuffles=20, rng=7
    ):
        assert len(shifted) == 500
        assert np.all(np.diff(shifted) >= 0)
        assert np.all(
            ((shifted >= 0) & (shifted < 100)) | ((shifted >= 200) & (shifted < 300))
        )


def test_shift_respects_min_shift():
    draws = list(
        stats.shuffle_spike_times_circular(
            np.array([10.0]), [[0, 100]], n_shuffles=500, min_shift=20, rng=7
        )
    )
    assert np.all((np.asarray(draws) >= 30) & (np.asarray(draws) <= 90))


def test_drops_spikes_outside_windows():
    for shifted in stats.shuffle_spike_times_circular(
        [0, 99, 100, 150, 200, 299, 300], [[0, 100], [200, 300]], n_shuffles=5, rng=7
    ):
        assert len(shifted) == 4
        assert np.all((shifted < 100) | ((shifted >= 200) & (shifted < 300)))


def test_seeded_reproducible():
    def draw(seed):
        return np.asarray(
            list(
                stats.shuffle_spike_times_circular(
                    [1, 5, 15, 80], [[0, 100]], n_shuffles=10, rng=seed
                )
            )
        )

    np.testing.assert_array_equal(draw(7), draw(7))
    assert not np.array_equal(draw(7), draw(8))


def test_rejects_too_short():
    with pytest.raises(ValueError, match=r"Analyzed time is") as caught:
        list(stats.shuffle_spike_times_circular([10], [[0, 30]], min_shift=20))
    assert "min_shift" in str(caught.value)
    assert "Why:" in str(caught.value)
    assert "Fix:" in str(caught.value)


@pytest.mark.parametrize(
    "kwargs",
    [{"n_shuffles": 0}, {"n_shuffles": 1.5}, {"min_shift": -1}, {"min_shift": np.nan}],
)
def test_rejects_invalid_shuffle_settings(kwargs):
    with pytest.raises(ValueError, match="Fix:"):
        list(stats.shuffle_spike_times_circular([10], [[0, 100]], **kwargs))


def test_one_offset_preserves_compressed_circular_spacings():
    spikes = np.array([5, 20, 70, 205, 220])
    compressed = np.where(spikes < 100, spikes, spikes - 100)
    expected = np.sort(np.diff(np.r_[compressed, compressed[0] + 200]))
    for shifted in stats.shuffle_spike_times_circular(
        spikes, [[0, 100], [200, 300]], n_shuffles=10, rng=7
    ):
        axis = np.where(shifted < 100, shifted, shifted - 100)
        np.testing.assert_allclose(
            np.sort(np.diff(np.r_[axis, axis[0] + 200])), expected, atol=1e-12
        )


def test_large_clock_rounding_stays_below_excluded_stop():
    class LargestShift(np.random.Generator):
        def uniform(self, *_):
            return np.nextafter(1.0, 0.0)

    t0 = 1e9
    train = next(
        stats.shuffle_spike_times_circular(
            np.array([t0]),
            np.array([[t0, t0 + 1.0]]),
            n_shuffles=1,
            min_shift=0,
            rng=LargestShift(np.random.PCG64(0)),
        )
    )
    assert len(train) == 1
    assert t0 <= train[0] < t0 + 1


@pytest.mark.parametrize("windows", [[[5, 1], [0, np.nan]], [[1, 2, 3]]])
def test_invalid_windows_fix_names_primitive_argument(windows):
    with pytest.raises(
        ValueError, match=r"Why: circular shifting requires finite half-open"
    ) as exc:
        list(stats.shuffle_spike_times_circular([10], windows))
    message = str(exc.value)
    assert "windows" in message
    assert "Why:" in message
    fix = message.split("Fix:", 1)[1]
    assert "windows=" in fix
    assert "epochs" not in fix
    assert "spike_window" not in fix
    assert "None" not in fix
    if len(windows) == 2:
        assert "stop <= start" in message
        assert "NaN" in message
    else:
        assert "shape" in message


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_shuffles": -5}, "must be a positive integer"),
        ({"windows": np.array([[10.0, 5.0]])}, "have stop <= start"),
        ({"min_shift": 1e6}, "not longer than 2 \\* min_shift"),
    ],
)
def test_invalid_arguments_raise_at_call_not_first_draw(kwargs, message):
    """Errors must surface where the shuffle is set up, before any draw."""
    options = {
        "windows": np.array([[0.0, 100.0]]),
        "n_shuffles": 3,
        "min_shift": 5.0,
        "rng": 0,
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        stats.shuffle_spike_times_circular(np.array([1.0, 2.0]), **options)


def test_single_spike_shift_is_uniform_between_min_shift_and_rest():
    """One spike moves by an offset uniform on [min_shift, T - min_shift].

    The spike at 70 s of a 100 s window wraps for offsets above 30 s, so the
    offset is read back modulo T.
    """
    from scipy.stats import kstest

    draws = np.concatenate(
        list(
            stats.shuffle_spike_times_circular(
                np.array([70.0]), [[0, 100]], n_shuffles=2000, min_shift=20, rng=7
            )
        )
    )
    offsets = np.mod(draws - 70.0, 100.0)
    assert np.all((offsets >= 20) & (offsets <= 80))
    # KS against Uniform(20, 80); a false failure has probability 0.001.
    assert kstest(offsets, "uniform", args=(20, 60)).pvalue > 1e-3


def _reflected_ou_path(rng, duration=600.0, fs=30.0, length=100.0):
    """1-D path on [0, length] cm: OU velocity (tau 1 s, SD 20 cm/s), walls reflect."""
    from scipy.signal import lfilter

    n = int(duration * fs)
    a = np.exp(-1 / (fs * 1.0))
    velocity = lfilter([20.0 * np.sqrt(1 - a**2)], [1, -a], rng.standard_normal(n))
    unfolded = length / 2 + np.cumsum(velocity) / fs
    folded = np.mod(unfolded, 2 * length)
    return np.arange(n) / fs, length - np.abs(folded - length)


def _spatial_information(trains, times, positions, edges, occupancy):
    """Skaggs information (bits/spike) of each row of ``trains``, binned by hand."""
    n_bins = len(edges) - 1
    bins = np.clip(
        np.searchsorted(edges, np.interp(trains, times, positions), side="right") - 1,
        0,
        n_bins - 1,
    )
    rows = np.repeat(np.arange(len(trains)), trains.shape[1])
    counts = np.bincount(
        rows * n_bins + bins.ravel(), minlength=len(trains) * n_bins
    ).reshape(len(trains), n_bins)
    relative = counts / counts.sum(axis=1, keepdims=True) / occupancy
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(relative > 0, occupancy * relative * np.log2(relative), 0.0)
    return terms.sum(axis=1)


@pytest.mark.slow
def test_place_shuffle_p_is_calibrated_for_untuned_cells():
    """Untuned cells give uniform shuffle p-values: neither too many nor too few.

    400 homogeneous 2 Hz Poisson cells on a 10-min reflected OU path. Each
    cell's spatial information is ranked among 99 circular shifts; p =
    (1 + #null >= observed) / 100. With p <= 0.05 meaning rank <= 4, the
    rejection rate is exactly 0.05 when the shifted trains are exchangeable
    with the observed one. (A 2000-cell run of this simulation gave 0.0525.)
    """
    from scipy.stats import binom, chisquare

    rng = np.random.default_rng(0)
    times, positions = _reflected_ou_path(rng)
    duration = times[-1] + times[1]
    edges = np.linspace(0, 100, 11)
    occupancy = np.histogram(positions, edges)[0] / len(positions)
    n_cells, n_shuffles = 400, 99
    ranks = np.empty(n_cells, dtype=int)
    for cell in range(n_cells):
        spikes = np.sort(rng.uniform(0, duration, rng.poisson(2.0 * duration)))
        null = np.asarray(
            list(
                stats.shuffle_spike_times_circular(
                    spikes, [[0, duration]], n_shuffles=n_shuffles, rng=rng
                )
            )
        )
        observed = _spatial_information(
            spikes[None], times, positions, edges, occupancy
        )
        null_info = _spatial_information(null, times, positions, edges, occupancy)
        ranks[cell] = np.sum(null_info >= observed)
    p_values = (1 + ranks) / (n_shuffles + 1)

    # Two-sided 99.9% binomial intervals: a null that never rejects (or always
    # does) fails, as does one whose median p is off.
    for level in (0.05, 0.5):
        low, high = binom.ppf([0.0005, 0.9995], n_cells, level)
        assert low <= np.sum(p_values <= level) <= high, level
    # Ranks 0..99 are uniform under exchangeability: chi-square over deciles.
    assert chisquare(np.bincount(ranks // 10, minlength=10)).pvalue > 1e-3
