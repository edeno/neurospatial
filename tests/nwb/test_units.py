"""
Tests for reading spike times from an NWB ``units`` table.

Tests the read_units() function, which extracts per-neuron spike-time arrays
from the ragged ``units`` DynamicTable of an NWB file.
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import numpy as np
import pytest

# Skip all tests if pynwb is not installed
pynwb = pytest.importorskip("pynwb")

# Known spike trains keyed by unit id, used across the round-trip tests.
_SPIKE_TIMES_BY_ID = {
    10: np.array([0.1, 0.5, 1.2, 3.4]),
    20: np.array([0.3, 0.9]),
    30: np.array([2.0, 2.5, 2.7, 4.1, 5.0]),
}


@pytest.fixture
def sample_nwb_with_units():
    """
    Create an NWB file with a units table of 3 neurons.

    Each unit has a distinct ragged spike-time array and an explicit ``id``
    (10, 20, 30) so that subset-by-id behavior can be exercised.

    Returns
    -------
    NWBFile
        NWB file with a populated ``units`` table.
    """
    from pynwb import NWBFile

    nwbfile = NWBFile(
        session_description="Test session for units reading",
        identifier=str(uuid4()),
        session_start_time=datetime.now().astimezone(),
    )
    for unit_id, spike_times in _SPIKE_TIMES_BY_ID.items():
        nwbfile.add_unit(spike_times=spike_times, id=unit_id)
    return nwbfile


class TestReadUnits:
    """Tests for read_units()."""

    def test_read_units_roundtrip(self, sample_nwb_with_units):
        """A 3-unit table reads back the exact ragged arrays and ids."""
        from neurospatial.io.nwb import read_units

        unit_data = read_units(sample_nwb_with_units)
        spike_trains = unit_data.spike_times
        unit_ids = unit_data.unit_ids

        assert list(unit_ids) == [10, 20, 30]
        assert len(spike_trains) == 3
        for unit_id, train in zip(unit_ids, spike_trains, strict=True):
            assert train.dtype == np.float64
            np.testing.assert_array_equal(train, _SPIKE_TIMES_BY_ID[unit_id])

    def test_read_units_subset(self, sample_nwb_with_units):
        """unit_ids selects those units by id value, in request order."""
        from neurospatial.io.nwb import read_units

        unit_data = read_units(sample_nwb_with_units, unit_ids=[10, 30])
        spike_trains = unit_data.spike_times
        unit_ids = unit_data.unit_ids

        assert list(unit_ids) == [10, 30]
        assert len(spike_trains) == 2
        np.testing.assert_array_equal(spike_trains[0], _SPIKE_TIMES_BY_ID[10])
        np.testing.assert_array_equal(spike_trains[1], _SPIKE_TIMES_BY_ID[30])

    def test_read_units_subset_preserves_request_order(self, sample_nwb_with_units):
        """Subset ordering follows the request, not the table order."""
        from neurospatial.io.nwb import read_units

        unit_data = read_units(sample_nwb_with_units, unit_ids=[30, 10])
        unit_ids = unit_data.unit_ids
        assert list(unit_ids) == [30, 10]

    def test_read_units_unknown_id_raises(self, sample_nwb_with_units):
        """An id absent from units.id raises ValueError naming it."""
        from neurospatial.io.nwb import read_units

        with pytest.raises(ValueError, match="not found in the units table"):
            read_units(sample_nwb_with_units, unit_ids=[10, 999])

        # The missing id is named; the present one is not flagged.
        with pytest.raises(ValueError, match="999"):
            read_units(sample_nwb_with_units, unit_ids=[10, 999])

    def test_read_units_sorts_spike_trains(self):
        """Out-of-order spike_times are returned sorted ascending."""
        from pynwb import NWBFile

        from neurospatial.io.nwb import read_units

        nwbfile = NWBFile(
            session_description="Test session for unsorted spikes",
            identifier=str(uuid4()),
            session_start_time=datetime.now().astimezone(),
        )
        unsorted = np.array([3.4, 0.1, 1.2, 0.5])
        nwbfile.add_unit(spike_times=unsorted, id=42)

        unit_data = read_units(nwbfile)
        spike_trains = unit_data.spike_times
        unit_ids = unit_data.unit_ids

        assert list(unit_ids) == [42]
        train = spike_trains[0]
        assert np.all(np.diff(train) >= 0)
        np.testing.assert_array_equal(train, np.sort(unsorted))

    def test_read_units_no_table_raises(self, empty_nwb):
        """An NWBFile without a units table raises a clear ValueError."""
        from neurospatial.io.nwb import read_units

        with pytest.raises(ValueError, match="no `units` table"):
            read_units(empty_nwb)

    def test_read_units_lazy_export(self):
        """read_units is reachable through the lazy package __getattr__."""
        from neurospatial.io.nwb import read_units

        assert callable(read_units)


class TestLazyUnitSpikeTrainCaching:
    """_LazyUnitSpikeTrain memoizes: the ragged slice is read at most once."""

    def test_materialize_reads_ragged_slice_once(self):
        """Multiple accesses read+sort the unit's slice exactly one time."""
        from neurospatial.io.nwb._units import _LazyUnitSpikeTrain

        class _CountingUnits:
            """Stand-in for the NWB units table counting ragged reads."""

            def __init__(self, per_row):
                self._per_row = per_row
                self.reads = 0

            def __getitem__(self, key):
                # pynwb ragged access is units[row, "spike_times"].
                row, col = key
                assert col == "spike_times"
                self.reads += 1
                return self._per_row[row]

        units = _CountingUnits({0: np.array([3.0, 1.0, 2.0])})
        train = _LazyUnitSpikeTrain(units, 0)

        # Access the handle several different ways.
        first = np.asarray(train)
        _ = train[0]
        _ = len(train)
        second = np.asarray(train)

        # The underlying ragged slice was read exactly once, not per-access.
        assert units.reads == 1
        # And it returns the same sorted array the eager path would.
        np.testing.assert_array_equal(first, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(second, [1.0, 2.0, 3.0])


def _file_with_tetrodes():
    """Electrodes 0-3 on tetrode1 in CA1, 4-7 on tetrode2 in PFC; no units."""
    from pynwb import NWBFile

    nwbfile = NWBFile(
        session_description="Units with electrodes and metadata",
        identifier=str(uuid4()),
        session_start_time=datetime.now().astimezone(),
    )
    device = nwbfile.create_device("drive")
    groups = [
        nwbfile.create_electrode_group("tetrode1", "tetrode 1", "CA1", device),
        nwbfile.create_electrode_group("tetrode2", "tetrode 2", "PFC", device),
    ]
    for channel in range(8):
        group = groups[channel // 4]
        nwbfile.add_electrode(group=group, location=group.location)
    return nwbfile


def _units_with_metadata():
    """Two tetrodes in different regions, DANDI-style unit columns, 30 kHz."""
    from pynwb.misc import Units

    nwbfile = _file_with_tetrodes()
    nwbfile.units = Units(
        name="units", resolution=1 / 30000, electrode_table=nwbfile.electrodes
    )
    nwbfile.add_unit_column("cell_type", "Putative cell type")
    nwbfile.add_unit_column("shank_id", "Shank of the unit's peak channel")
    nwbfile.add_unit_column("waveform_mean", "Mean waveform")
    nwbfile.add_unit_column("peak_channels", "Channels with a peak", index=True)
    rows = [
        (10, [0.1, 0.5], [0, 1, 2, 3], "excitatory", 0),
        (20, [0.3], [4, 5, 6, 7], "inhibitory", 1),
        (30, [0.2, 0.9, 1.4], [0, 1, 2, 3], "inhibitory", 0),
    ]
    for unit_id, spikes, electrodes, cell_type, shank_id in rows:
        nwbfile.add_unit(
            spike_times=spikes,
            id=unit_id,
            electrodes=electrodes,
            cell_type=cell_type,
            shank_id=shank_id,
            waveform_mean=np.zeros(5),
            peak_channels=electrodes[:2],
        )
    return nwbfile


class TestUnitTable:
    """read_units returns per-unit metadata aligned to unit_ids."""

    def test_electrode_regions_and_unit_columns(self):
        """Region and group come from the electrodes; scalar columns are kept."""
        from neurospatial.io.nwb import read_units

        units = read_units(_units_with_metadata())
        table = units.unit_table

        assert table.index.name == "unit_id"
        assert table.index.tolist() == [10, 20, 30]
        assert table["electrode_location"].tolist() == ["CA1", "PFC", "CA1"]
        assert table["electrode_group"].tolist() == [
            "tetrode1",
            "tetrode2",
            "tetrode1",
        ]
        assert table["cell_type"].tolist() == [
            "excitatory",
            "inhibitory",
            "inhibitory",
        ]
        assert table["shank_id"].tolist() == [0, 1, 0]
        # Spike times, ragged and multi-valued columns are not table columns.
        assert set(table.columns) == {
            "cell_type",
            "shank_id",
            "electrode_location",
            "electrode_group",
        }
        assert units.resolution == pytest.approx(1 / 30000)

    def test_selection_order_and_spiketrains_filter(self):
        """Rows follow the requested ids and select units by region."""
        from neurospatial.encoding import SpikeTrains
        from neurospatial.io.nwb import read_units

        units = read_units(_units_with_metadata(), unit_ids=[30, 20])
        assert units.unit_table.index.tolist() == [30, 20]
        assert units.unit_table["electrode_location"].tolist() == ["CA1", "PFC"]

        trains = SpikeTrains(
            units.spike_times, unit_ids=units.unit_ids, unit_table=units.unit_table
        )
        ca1 = trains.filter("electrode_location == 'CA1'")
        assert ca1.unit_ids.tolist() == [30]
        np.testing.assert_array_equal(ca1.trains[0], [0.2, 0.9, 1.4])

    def test_unit_location_column_without_electrodes(self):
        """A Units ``location`` column (DANDI 000044 style) is kept verbatim."""
        from pynwb import NWBFile

        from neurospatial.io.nwb import read_units

        nwbfile = NWBFile(
            session_description="Units with a location column",
            identifier=str(uuid4()),
            session_start_time=datetime.now().astimezone(),
        )
        nwbfile.add_unit_column("location", "Brain region")
        nwbfile.add_unit_column("cell_type", "Putative cell type")
        nwbfile.add_unit(spike_times=[0.1], id=1, location="lCA1", cell_type="pyr")
        nwbfile.add_unit(spike_times=[0.2], id=2, location="lCA3", cell_type="int")

        units = read_units(nwbfile)
        assert units.unit_table["location"].tolist() == ["lCA1", "lCA3"]
        assert "electrode_location" not in units.unit_table.columns
        assert units.resolution is None

    def test_units_spanning_regions_list_each_region_once(self):
        """A unit on electrodes in two regions names both, in electrode order."""
        from neurospatial.io.nwb import read_units

        nwbfile = _units_with_metadata()
        nwbfile.add_unit(
            spike_times=[2.0],
            id=40,
            electrodes=[2, 3, 4],
            cell_type="excitatory",
            shank_id=0,
            waveform_mean=np.zeros(5),
            peak_channels=[3, 4],
        )
        table = read_units(nwbfile, unit_ids=[40]).unit_table
        assert table["electrode_location"].tolist() == ["CA1, PFC"]
        assert table["electrode_group"].tolist() == ["tetrode1, tetrode2"]

    def test_table_without_metadata_has_no_columns(self, sample_nwb_with_units):
        """Units with spike times only give an empty, id-indexed table."""
        from neurospatial.io.nwb import read_units

        table = read_units(sample_nwb_with_units).unit_table
        assert table.shape == (3, 0)
        assert table.index.tolist() == [10, 20, 30]

    def test_on_disk_lazy_matches_eager(self, tmp_path):
        """On disk, lazy spike handles keep the same eager metadata table."""
        import pandas as pd
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_units
        from neurospatial.io.nwb._units import _LazyUnitSpikeTrain

        path = tmp_path / "units.nwb"
        with NWBHDF5IO(path, "w") as io:
            io.write(_units_with_metadata())
        with NWBHDF5IO(path, "r") as io:
            nwbfile = io.read()
            eager = read_units(nwbfile)
            lazy = read_units(nwbfile, lazy=True)
            assert all(isinstance(t, _LazyUnitSpikeTrain) for t in lazy.spike_times)
            assert all(t._cache is None for t in lazy.spike_times)
            np.testing.assert_array_equal(
                np.asarray(lazy.spike_times[2]), [0.2, 0.9, 1.4]
            )
        pd.testing.assert_frame_equal(lazy.unit_table, eager.unit_table)
        assert eager.unit_table["electrode_location"].tolist() == ["CA1", "PFC", "CA1"]
        assert eager.unit_table["cell_type"].tolist() == [
            "excitatory",
            "inhibitory",
            "inhibitory",
        ]
        assert eager.resolution == lazy.resolution == pytest.approx(1 / 30000)

    def test_unit_table_edits_do_not_reach_the_holder(self):
        """The holder hands out copies, keeping rows aligned with unit_ids."""
        from neurospatial.io.nwb import read_units

        units = read_units(_units_with_metadata())
        table = units.unit_table
        table.drop(index=10, inplace=True)
        assert units.unit_table.index.tolist() == [10, 20, 30]

    def test_electrode_group_column_without_electrodes(self):
        """A Units ``electrode_group`` column gives the group and its region."""
        from pynwb import NWBFile

        from neurospatial.io.nwb import read_units

        nwbfile = NWBFile(
            session_description="Units with electrode groups only",
            identifier=str(uuid4()),
            session_start_time=datetime.now().astimezone(),
        )
        device = nwbfile.create_device("drive")
        ca1 = nwbfile.create_electrode_group("shank0", "shank 0", "CA1", device)
        ca3 = nwbfile.create_electrode_group("shank1", "shank 1", "CA3", device)
        nwbfile.add_unit(spike_times=[0.1], id=1, electrode_group=ca3)
        nwbfile.add_unit(spike_times=[0.2], id=2, electrode_group=ca1)

        table = read_units(nwbfile).unit_table
        assert table["electrode_location"].tolist() == ["CA3", "CA1"]
        assert table["electrode_group"].tolist() == ["shank1", "shank0"]

    def test_invalid_resolution_is_refused(self):
        """A non-positive spike-time resolution is not a precision."""
        from neurospatial.io.nwb import NWBUnits

        with pytest.raises(ValueError, match="resolution must be a finite positive"):
            NWBUnits([np.array([0.1])], np.array([7]), None, resolution=0.0)

    def test_one_electrode_per_unit_without_index(self):
        """An unindexed electrodes column (one electrode per unit) is read.

        Older pynwb wrote this layout, for example DANDI 000447; current
        pynwb no longer writes it, so the table is built in memory.
        """
        from hdmf.common.table import DynamicTableRegion, VectorData, VectorIndex
        from pynwb.misc import Units

        from neurospatial.io.nwb import read_units

        nwbfile = _file_with_tetrodes()
        spikes = VectorData(name="spike_times", description="s", data=[0.1, 0.2])
        nwbfile.units = Units(
            name="units",
            id=[5, 6],
            columns=[
                spikes,
                VectorIndex(name="spike_times_index", data=[1, 2], target=spikes),
                DynamicTableRegion(
                    name="electrodes",
                    description="Electrode of each unit",
                    data=[5, 1],
                    table=nwbfile.electrodes,
                ),
            ],
        )
        table = read_units(nwbfile).unit_table
        assert table["electrode_location"].tolist() == ["PFC", "CA1"]
        assert table["electrode_group"].tolist() == ["tetrode2", "tetrode1"]
