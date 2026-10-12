"""Scientific meaning survives common user choices and operation handoffs."""

import numpy as np
import pandas as pd
import pytest

from neurospatial import Environment, compute_spatial_rates
from neurospatial.encoding import SpikeTrains


@pytest.fixture
def partial_recording():
    env = Environment.from_grid_mask(
        np.ones(10, dtype=bool), grid_edges=(np.arange(11.0) * 10,)
    )
    env.units = "cm"
    times = np.arange(101) / 10
    positions = np.full((len(times), 1), 5.0)
    spikes = [np.arange(0.05, 10.0, 0.1)]
    return env, spikes, times, positions


@pytest.mark.parametrize("identity", ["index", "column"])
def test_metadata_filtering_and_encoding_preserve_identity(partial_recording, identity):
    env, spikes, times, positions = partial_recording
    table = pd.DataFrame(
        {"region": ["CA3", "CA1"]}, index=pd.Index([20, 10], name="unit_id")
    )
    if identity == "column":
        table = table.reset_index()
    original = table.copy()
    trains = SpikeTrains(
        [spikes[0], spikes[0][::2]], unit_ids=[10, 20], unit_table=table
    )
    assert trains.filter("region == 'CA1'").unit_ids.tolist() == [10]
    if identity == "index":
        assert trains.filter("region == 'CA1'").unit_table.index.tolist() == [10]
    rates = compute_spatial_rates(env, trains, times, positions)
    assert rates.unit_table.iloc[0]["region"] == "CA1"
    pd.testing.assert_frame_equal(table, original)
    with pytest.raises(ValueError, match="labels do not match"):
        SpikeTrains([spikes[0], spikes[0]], unit_ids=[1, 2], unit_table=table)
