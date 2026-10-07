"""NWB reader errors identify the reader argument and available data."""

from datetime import datetime, timezone

import pytest

from neurospatial.io.nwb import read_position, read_units

pynwb = pytest.importorskip("pynwb")


@pytest.fixture
def nwbfile():
    return pynwb.NWBFile(
        session_description="Reader error fixture",
        identifier="reader-errors",
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


@pytest.mark.parametrize(
    "site", ["module_missing", "container_missing", "position_missing", "unit_missing"]
)
def test_reader_errors_teach(site, nwbfile):
    if site == "module_missing" or site == "container_missing":
        nwbfile.create_processing_module("behavior", "Behavior data")
    elif site == "unit_missing":
        nwbfile.add_unit(id=7, spike_times=[0.1, 0.5])
    with pytest.raises((KeyError, ValueError)) as caught:
        if site == "unit_missing":
            read_units(nwbfile, unit_ids=[99])
        else:
            module = {
                "module_missing": "missing",
                "container_missing": "behavior",
                "position_missing": None,
            }[site]
            read_position(nwbfile, processing_module=module)
    message = str(caught.value)
    assert "Fix:" in message
    if site == "unit_missing":
        assert "unit_ids=" in message
        assert "7" in message
        assert message.splitlines()[-1].startswith("Fix: ")
    else:
        assert "processing_module=" in message
        assert "\\n" not in message
