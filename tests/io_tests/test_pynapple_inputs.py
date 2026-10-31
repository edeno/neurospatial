"""pynapple objects passed straight to analysis functions.

A ``TsGroup`` keeps its unit labels and metadata, a ``TsdFrame`` or ``Tsd``
passed as ``positions`` or ``headings`` keeps its time support and column
order, and ``IntervalSet`` intervals keep their inclusive ends. Each test
compares the pynapple route with the plain-array route it must equal.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from neurospatial import Environment

pytestmark = [
    pytest.mark.pynapple,
    pytest.mark.skipif(
        importlib.util.find_spec("pynapple") is None,
        reason="requires the pynapple extra",
    ),
]


@pytest.fixture
def nap():
    import pynapple

    return pynapple


@pytest.fixture
def session():
    """A 40 s open-field walk at 100 Hz with three units."""
    rng = np.random.default_rng(0)
    times = np.round(np.arange(4000) * 0.01, 10)
    positions = np.column_stack(
        [
            50.0 + 40.0 * np.sin(times / 3.0),
            20.0 + 15.0 * np.cos(times / 2.0),
        ]
    )
    env = Environment.from_samples(positions, bin_size=10.0)
    trains = [np.sort(rng.uniform(0.0, 39.99, n)) for n in (80, 60, 70)]
    return env, times, positions, trains


@pytest.fixture
def group(nap, session):
    """The three units labelled 5, 6, 7 with a brain-region column."""
    _, _, _, trains = session
    return nap.TsGroup(
        {5: nap.Ts(trains[0]), 6: nap.Ts(trains[1]), 7: nap.Ts(trains[2])},
        metadata={"region": ["CA1", "CA3", "CA1"], "depth": [1.0, 2.0, 3.0]},
    )


# ---------------------------------------------------------------------------
# TsGroup metadata becomes the unit table
# ---------------------------------------------------------------------------


def _expected_table(pd):
    return pd.DataFrame(
        {"region": ["CA1", "CA3", "CA1"], "depth": [1.0, 2.0, 3.0]},
        index=pd.Index([5, 6, 7], name="unit_id"),
    )


def test_spike_trains_accepts_a_tsgroup_with_labels_and_metadata(group, session):
    import pandas as pd

    from neurospatial.encoding import SpikeTrains

    st = SpikeTrains(group)

    assert st.unit_ids.tolist() == [5, 6, 7]
    for got, want in zip(st.trains, session[3], strict=True):
        np.testing.assert_array_equal(got, want)
    # pynapple's computed ``rate`` column is not unit metadata.
    pd.testing.assert_frame_equal(
        st.unit_table, _expected_table(pd), check_index_type=False
    )
    assert st.filter("region == 'CA1'").unit_ids.tolist() == [5, 7]


def test_spike_trains_accepts_a_dict_of_trains() -> None:
    from neurospatial.encoding import SpikeTrains

    st = SpikeTrains({"a": [0.1, 0.2], "b": [0.3]})

    assert st.unit_ids.tolist() == ["a", "b"]
    np.testing.assert_array_equal(st.sel(unit_id="b"), [0.3])


def test_spike_trains_from_tsgroup_refuses_other_unit_ids(group):
    from neurospatial.encoding import SpikeTrains

    with pytest.raises(ValueError, match=r"already labelled \[5, 6, 7\]"):
        SpikeTrains(group, unit_ids=[0, 1, 2])


def test_spike_trains_from_tsgroup_with_explicit_table(group):
    import pandas as pd

    from neurospatial.encoding import SpikeTrains

    table = pd.DataFrame({"quality": [0.9, 0.5, 0.7]}, index=pd.Index([5, 6, 7]))
    st = SpikeTrains(group, unit_table=table)

    assert st.unit_table.columns.tolist() == ["quality"]


def test_compute_spatial_rates_keeps_tsgroup_metadata(group, session):
    import pandas as pd

    from neurospatial import compute_spatial_rates

    env, times, positions, trains = session
    from_group = compute_spatial_rates(env, group, times, positions, method="binned")
    from_arrays = compute_spatial_rates(env, trains, times, positions, method="binned")

    np.testing.assert_array_equal(from_group.firing_rates, from_arrays.firing_rates)
    pd.testing.assert_frame_equal(
        from_group.unit_table, _expected_table(pd), check_index_type=False
    )
    assert from_group.sel(unit_id=6).unit_id == 6


def test_population_encoders_keep_tsgroup_metadata(group, session):
    from neurospatial.encoding import (
        compute_directional_rates,
        compute_object_vector_rates,
    )

    env, times, positions, _ = session
    headings = np.arctan2(np.gradient(positions[:, 1]), np.gradient(positions[:, 0]))
    results = [
        compute_directional_rates(group, times, headings),
        compute_object_vector_rates(
            env,
            group,
            times,
            positions,
            np.array([[50.0, 20.0]]),
            distance_bin_size=5.0,
        ),
    ]
    for result in results:
        assert result.unit_table["region"].tolist() == ["CA1", "CA3", "CA1"]
        assert result.unit_table.index.tolist() == [5, 6, 7]


def test_population_peri_event_histogram_keeps_unit_tables(group):
    from neurospatial.encoding import SpikeTrains
    from neurospatial.events import population_peri_event_histogram

    events = np.array([10.0, 20.0, 30.0])
    for spikes in (group, SpikeTrains(group)):
        result = population_peri_event_histogram(
            spikes, events, (-0.5, 0.5), bin_size=0.1
        )
        assert result.unit_ids.tolist() == [5, 6, 7]
        assert result.unit_table["region"].tolist() == ["CA1", "CA3", "CA1"]


def test_tsgroup_with_only_rate_metadata_gives_no_unit_table(nap, session):
    from neurospatial import compute_spatial_rates

    env, times, positions, trains = session
    bare = nap.TsGroup({5: nap.Ts(trains[0]), 6: nap.Ts(trains[1])})

    result = compute_spatial_rates(env, bare, times, positions, method="binned")

    assert result.unit_table is None
    assert result.unit_ids.tolist() == [5, 6]
