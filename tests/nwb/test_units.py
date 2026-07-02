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

        spike_trains, unit_ids = read_units(sample_nwb_with_units)

        assert list(unit_ids) == [10, 20, 30]
        assert len(spike_trains) == 3
        for unit_id, train in zip(unit_ids, spike_trains, strict=True):
            assert train.dtype == np.float64
            np.testing.assert_array_equal(train, _SPIKE_TIMES_BY_ID[unit_id])

    def test_read_units_subset(self, sample_nwb_with_units):
        """unit_ids selects those units by id value, in request order."""
        from neurospatial.io.nwb import read_units

        spike_trains, unit_ids = read_units(sample_nwb_with_units, unit_ids=[10, 30])

        assert list(unit_ids) == [10, 30]
        assert len(spike_trains) == 2
        np.testing.assert_array_equal(spike_trains[0], _SPIKE_TIMES_BY_ID[10])
        np.testing.assert_array_equal(spike_trains[1], _SPIKE_TIMES_BY_ID[30])

    def test_read_units_subset_preserves_request_order(self, sample_nwb_with_units):
        """Subset ordering follows the request, not the table order."""
        from neurospatial.io.nwb import read_units

        _, unit_ids = read_units(sample_nwb_with_units, unit_ids=[30, 10])
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

        spike_trains, unit_ids = read_units(nwbfile)

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
