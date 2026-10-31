"""Files written by neurospatial follow the NWB best practices for tables.

NWB best practice is not to write empty tables (nwbinspector
``check_empty_table``) or columns whose values are all NaN
(``check_col_not_nan``). These tests write zero trials, laps, crossings and
events, and laps found without a template, then check the file read back.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pynwb")
pytest.importorskip("ndx_events")

from neurospatial.behavior.segmentation import Lap

from ._nwb_helpers import create_roundtrip_nwb

REGION_LAPS = [
    Lap(start_time=1.0, end_time=5.0, direction="unknown", overlap_score=np.nan),
    Lap(start_time=6.0, end_time=9.0, direction="clockwise", overlap_score=np.nan),
]


@pytest.fixture
def edge_case_file(tmp_path: Path) -> Path:
    """Write every record writer with zero records, plus region-method laps."""
    from pynwb import NWBHDF5IO

    from neurospatial.io.nwb import (
        write_events,
        write_laps,
        write_region_crossings,
        write_trials,
    )

    nwbfile = create_roundtrip_nwb()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "write_.* wrote nothing", UserWarning)
        write_trials(nwbfile, [])
        write_laps(nwbfile, [])
        write_region_crossings(nwbfile, [])
        write_events(nwbfile, pd.DataFrame({"timestamp": []}), name="rewards")
    write_laps(nwbfile, REGION_LAPS, name="region_laps")
    path = tmp_path / "edge_cases.nwb"
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)
    return path


def test_no_empty_tables_or_all_nan_columns(edge_case_file: Path) -> None:
    from hdmf.common import DynamicTable, VectorData
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(edge_case_file), "r") as io:
        nwbfile = io.read()
        tables = {
            obj.name: obj
            for obj in nwbfile.all_children()
            if isinstance(obj, DynamicTable)
        }
        assert set(tables) == {"region_laps"}
        for name, table in tables.items():
            assert len(table) > 0, f"{name} is empty"
            for column in table.columns:
                if not isinstance(column, VectorData):
                    continue
                values = np.asarray(column.data[:])
                if values.dtype.kind == "f":
                    assert not np.all(np.isnan(values)), (
                        f"{name}.{column.name} is all NaN"
                    )
        assert "behavior" not in nwbfile.processing


def test_nwbinspector_finds_no_table_violations(edge_case_file: Path) -> None:
    nwbinspector = pytest.importorskip("nwbinspector")

    messages = list(
        nwbinspector.inspect_nwbfile(
            nwbfile_path=edge_case_file,
            config=nwbinspector.load_config(filepath_or_keyword="dandi"),
        )
    )

    table_checks = {"check_empty_table", "check_col_not_nan"}
    flagged = [m for m in messages if m.check_function_name in table_checks]
    assert flagged == []
    errors = [m for m in messages if m.importance.name == "ERROR"]
    assert errors == []
