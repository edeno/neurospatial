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
    with pytest.raises(ValueError) as caught:
        list(stats.shuffle_spike_times_circular([10], [[0, 30]], min_shift=20))
    assert "min_shift" in str(caught.value)
    assert "Why:" in str(caught.value) and "Fix:" in str(caught.value)


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


def test_large_clock_rounding_stays_below_excluded_stop(monkeypatch):
    from types import SimpleNamespace

    from neurospatial.stats import shuffle

    t0 = 1e9
    monkeypatch.setattr(
        shuffle,
        "_ensure_rng",
        lambda _: SimpleNamespace(uniform=lambda *_: np.nextafter(1.0, 0.0)),
    )
    train = next(
        stats.shuffle_spike_times_circular(
            np.array([t0]), np.array([[t0, t0 + 1.0]]), n_shuffles=1, min_shift=0
        )
    )
    assert len(train) == 1
    assert t0 <= train[0] < t0 + 1


@pytest.mark.parametrize("windows", [[[5, 1], [0, np.nan]], [[1, 2, 3]]])
def test_invalid_windows_fix_names_primitive_argument(windows):
    with pytest.raises(ValueError) as exc:
        list(stats.shuffle_spike_times_circular([10], windows))
    message = str(exc.value)
    assert "windows" in message and "Why:" in message
    fix = message.split("Fix:", 1)[1]
    assert "windows=" in fix
    assert "epochs" not in fix and "spike_window" not in fix and "None" not in fix
    if len(windows) == 2:
        assert "stop <= start" in message and "NaN" in message
    else:
        assert "shape" in message


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_shuffles": -5},
        {"windows": np.array([[10.0, 5.0]])},
        {"min_shift": 1e6},
    ],
)
def test_invalid_arguments_raise_at_call_not_first_draw(kwargs):
    """Errors must surface where the shuffle is set up, before any draw."""
    options = {
        "windows": np.array([[0.0, 100.0]]),
        "n_shuffles": 3,
        "min_shift": 5.0,
        "rng": 0,
        **kwargs,
    }
    with pytest.raises(ValueError):
        stats.shuffle_spike_times_circular(np.array([1.0, 2.0]), **options)
