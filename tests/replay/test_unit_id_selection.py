"""``unit_ids`` selects units by id in ``null_events`` and ``spike_words``."""

import numpy as np
import pytest

from neurospatial.replay import null_events, spike_words

MASK = np.array([True, False, True, False, True, True])


@pytest.fixture(scope="module")
def trains():
    rng = np.random.default_rng(0)
    return [np.sort(rng.uniform(0, 60, 80)) for _ in range(6)]


def test_spike_words_ids_give_the_words_the_boolean_mask_gave_before(trains):
    """Values recorded with ``cells=mask`` before the argument took ids."""
    words = spike_words(
        trains, max_isi=0.05, word_gap=0.3, unit_ids=np.flatnonzero(MASK)
    ).intervals
    assert words.shape == (69, 2)
    np.testing.assert_allclose(words.sum(), 3972.7761311707345, rtol=1e-12)
    shuffled = spike_words(
        trains, max_isi=0.05, word_gap=0.3, unit_ids=np.array([5, 0, 4, 2])
    ).intervals
    np.testing.assert_array_equal(shuffled, words)


@pytest.mark.parametrize(
    ("kind", "total"),
    [
        ("cell_identity", 15434.76354495811),
        ("spike_shift", 15390.412664798168),
        ("spike_time_uniform", 15519.16539181476),
    ],
)
def test_null_events_ids_give_the_nulls_the_boolean_mask_gave_before(
    trains, kind, total
):
    windows = np.array([[5.0, 15.0], [20.0, 30.0], [35.0, 45.0]])
    null = null_events(
        trains,
        windows,
        kind=kind,
        n_copies=2,
        rng=0,
        unit_ids=np.flatnonzero(MASK),
    )
    assert sum(len(t) for t in null.spike_times) == 464
    np.testing.assert_allclose(
        sum(np.asarray(t).sum() for t in null.spike_times), total, rtol=1e-12
    )


@pytest.mark.parametrize(
    ("unit_ids", "message"),
    [
        (MASK, "np.flatnonzero"),
        (np.array([0, 6]), "available ids"),
        (np.array([-1, 2]), "available ids"),
        (np.array([1, 1]), "repeats a unit"),
        (np.array([0.5]), "integer unit ids"),
        (np.zeros((2, 2), dtype=int), "1-D"),
    ],
)
def test_bad_unit_ids_are_refused(trains, unit_ids, message):
    with pytest.raises(ValueError, match=message):
        spike_words(trains, max_isi=0.05, word_gap=0.3, unit_ids=unit_ids)
    with pytest.raises(ValueError, match=message):
        null_events(
            trains, np.array([[5.0, 15.0]]), kind="cell_identity", unit_ids=unit_ids
        )


def test_ids_are_the_labels_a_spike_group_carries():
    """``{1: a, 2: b, 3: c}`` with ``unit_ids=[1, 2]`` selects labels 1 and 2."""
    rng = np.random.default_rng(1)
    group = {label: np.sort(rng.uniform(0, 10, 50)) for label in (1, 2, 3)}
    by_label = spike_words(group, max_isi=0.05, word_gap=0.2, unit_ids=[1, 2])
    by_position = spike_words(
        list(group.values()), max_isi=0.05, word_gap=0.2, unit_ids=[0, 1]
    )
    np.testing.assert_array_equal(by_label.intervals, by_position.intervals)
    with pytest.raises(ValueError, match=r"available ids: \[1, 2, 3\]"):
        spike_words(group, max_isi=0.05, word_gap=0.2, unit_ids=[0, 1])
    with pytest.raises(ValueError, match=r"available ids: \[1, 2, 3\]"):
        null_events(group, np.array([[1.0, 5.0]]), kind="cell_identity", unit_ids=[0])
