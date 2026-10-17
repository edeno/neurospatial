"""Clip mappings are interned by value while legacy duplicates stay valid."""

import copy

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.ops.transforms import translate


def _clipped():
    return Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    ).clip(box(0.25, 0, 2, 2))


@pytest.mark.parametrize(
    "operation,expected_count", [("native", 1), ("transform", 1), ("subset", 2)]
)
def test_equal_owned_clip_mappings_use_one_stored_snapshot(operation, expected_count):
    env = _clipped()
    if operation == "transform":
        env = env.apply_transform(translate(5, -2))
    elif operation == "subset":
        env = env.subset(mask=np.array([True, False, True, False]))
    record = env.to_dict()
    assert len(record["binning"]["snapshots"]["clip_mappings"]) == expected_count
    assert sum("/clip_mappings/" in key for key in record["arrays"]) == expected_count
    restored = Environment.from_dict(record)
    np.testing.assert_array_equal(
        restored.clip_mapping.parent_bin_indices, env.clip_mapping.parent_bin_indices
    )
    np.testing.assert_array_equal(
        restored.bin_at(env.bin_centers), np.arange(env.n_bins)
    )


@pytest.mark.parametrize("corrupt", [False, True])
def test_legacy_duplicate_mapping_tables_remain_validated(corrupt):
    env = _clipped()
    record = env.to_dict()
    tables = record["binning"]["snapshots"]["clip_mappings"]
    tables[:] = [tables[0], copy.deepcopy(tables[0])]
    record["binning"]["locator"]["mapping_ref"] = 1
    source = "binning/snapshots/clip_mappings/0/parent_bin_indices"
    duplicate = "binning/snapshots/clip_mappings/1/parent_bin_indices"
    record["arrays"][duplicate] = copy.deepcopy(record["arrays"][source])
    record["array_specs"][duplicate] = copy.deepcopy(record["array_specs"][source])
    if corrupt:
        record["arrays"][duplicate][0], record["arrays"][duplicate][1] = (
            record["arrays"][duplicate][1],
            record["arrays"][duplicate][0],
        )
        # The wrong parent can be detected by mapping consistency or by the
        # resulting disagreement between cell ownership and stored centers.
        with pytest.raises(ValueError, match=r"mapping|lookup|parent|clipped centers"):
            Environment.from_dict(record)
    else:
        restored = Environment.from_dict(record)
        np.testing.assert_array_equal(
            restored.bin_at(env.bin_centers), np.arange(env.n_bins)
        )
