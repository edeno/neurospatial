"""Saved environments load with canonical layout labels, whoever wrote them.

Earlier versions wrote other spellings: the user's layout name
(``"regular-grid"``), one ``"_transformed"`` suffix per ``apply_transform``,
``"subset"``, ``"PolygonCells"`` and ``"combined"`` for derived bins,
``"MaskedGrid"`` for polar bins, a ``layout_category`` field, derivation
markers among the layout parameters and the input data there too. Each test
writes a record with this version, rewrites it in the older spelling, and
reads it back through a dict, a file and NWB.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.io import from_dict, from_file, to_dict, to_file
from neurospatial.layout import create_layout
from neurospatial.ops.transforms import translate


def _grid():
    positions = np.random.default_rng(0).uniform(0, 20, (800, 2))
    return Environment.from_samples(positions, bin_size=2.0)


def _hexagons():
    return Environment.from_layout(
        create_layout(
            "Hexagonal", hexagon_width=3.0, dimension_ranges=[(0, 20), (0, 20)]
        )
    )


def _half(env):
    return env.subset(mask=np.arange(env.n_bins) < env.n_bins // 2)


def _polar():
    return Environment.from_polar_egocentric(
        distance_range=(0.0, 10.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=2.0,
        angle_bin_size=np.pi / 4,
    )


def _combined():
    return Environment.combine(
        {
            "a": Environment.from_polygon(box(0, 0, 10, 10), bin_size=2.0),
            "b": Environment.from_polygon(box(12, 0, 22, 10), bin_size=2.0),
        }
    )


# name: (build, saved layout_type, saved binning kind, category,
#        saved layout_parameters update, expected label, expected history)
OLD_RECORDS = {
    "user spelling": (
        _grid,
        "regular-grid",
        "RegularGrid",
        "grid",
        {"positions": [[0.0, 1.0], [2.0, 3.0]], "dimension_ranges": [[0, 20]] * 2},
        "RegularGrid",
        (),
    ),
    "hexagonal spelling": (
        _hexagons,
        "hexagonal",
        "Hexagonal",
        "hexagonal",
        {},
        "Hexagonal",
        (),
    ),
    "subset of hexagons": (
        lambda: _half(_hexagons()),
        "subset",
        "subset",
        "subset",
        {"source": "subset", "original_n_bins": 99},
        "Hexagonal",
        ("subset",),
    ),
    "two transforms": (
        lambda: _hexagons().apply_transform(translate(1.0, 0.0)),
        "hexagonal_transformed_transformed",
        "Hexagonal_transformed_transformed",
        "hexagonal",
        {"transformed": True},
        "Hexagonal",
        ("apply_transform", "apply_transform"),
    ),
    "subset then transform": (
        lambda: _half(_hexagons()).apply_transform(translate(1.0, 0.0)),
        "subset_transformed",
        "subset_transformed",
        "subset",
        {"source": "subset", "original_n_bins": 99, "transformed": True},
        "Hexagonal",
        ("subset", "apply_transform"),
    ),
    "clip of hexagons": (
        lambda: _hexagons().clip(box(4.0, 4.0, 15.0, 15.0)),
        "PolygonCells",
        "PolygonCells",
        "polygon",
        {"source": "clip", "original_n_bins": 99},
        "Hexagonal",
        ("clip",),
    ),
    "grid subset": (
        lambda: _half(_grid()),
        "MaskedGrid",
        "MaskedGrid",
        "mask",
        {"source": "subset", "original_n_bins": 99},
        "MaskedGrid",
        ("subset",),
    ),
    "combined": (
        _combined,
        "combined",
        "combined",
        "polygon",
        {},
        "Combined",
        ("combine",),
    ),
    "polar": (
        _polar,
        "MaskedGrid",
        "MaskedGrid",
        "mask",
        {"active_mask": [[True, True]], "grid_edges": [[0.0, 1.0]]},
        "Polar",
        (),
    ),
}


def _old_record(record, layout_type, kind, category, params):
    """Rewrite a saved record in the spelling of an earlier version."""
    record = json.loads(json.dumps(record))
    record["layout_type"] = layout_type
    record["layout_category"] = category
    record["binning"]["kind"] = kind
    del record["derivations"]
    for key in ("source", "original_n_bins", "transformed"):
        if key in params:
            # Earlier derivations replaced the build choices with markers.
            record["layout_parameters"] = {}
    record["layout_parameters"].update(params)
    return record


def _read_back(record, how, tmp_path: Path):
    if how == "dict":
        return from_dict(record)
    if how == "file":
        env = from_dict(record)
        to_file(env, tmp_path / "env")
        payload = tmp_path / "env.json"
        saved = json.loads(payload.read_text())
        for key in ("layout_type", "layout_category", "binning", "layout_parameters"):
            saved[key] = record[key]
        saved.pop("derivations")
        payload.write_text(json.dumps(saved))
        return from_file(tmp_path / "env")
    pynwb = pytest.importorskip("pynwb")
    from neurospatial.io.nwb import read_environment, write_environment

    nwbfile = pynwb.NWBFile(
        session_description="old labels",
        identifier="old labels",
        session_start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    write_environment(nwbfile, from_dict(record))
    table = nwbfile.scratch["spatial_environment"]
    saved = json.loads(table.description)
    for key in ("layout_type", "layout_category", "binning", "layout_parameters"):
        saved[key] = record[key]
    saved.pop("derivations")
    table.fields["description"] = json.dumps(saved)
    assert json.loads(table.description)["layout_type"] == record["layout_type"]
    return read_environment(nwbfile)


@pytest.mark.parametrize("how", ["dict", "file", "nwb"])
@pytest.mark.parametrize("case", sorted(OLD_RECORDS))
def test_older_spellings_load_with_canonical_labels(case, how, tmp_path):
    build, layout_type, kind, category, params, label, history = OLD_RECORDS[case]
    env = build()
    record = _old_record(to_dict(env), layout_type, kind, category, params)
    restored = _read_back(record, how, tmp_path)
    assert restored.layout_type == label == restored.binning.kind
    assert restored.derivations == history
    # Input data and derivation markers are not build choices.
    assert not {"positions", "dimension_ranges", "active_mask", "grid_edges"} & set(
        restored.layout_parameters
    )
    assert not {"source", "original_n_bins", "transformed"} & set(
        restored.layout_parameters
    )
    np.testing.assert_array_equal(restored.bin_centers, env.bin_centers)
    np.testing.assert_array_equal(restored.edge_pairs, env.edge_pairs)
    np.testing.assert_array_equal(restored.bin_sizes, env.bin_sizes)


def test_current_records_keep_their_labels_history_and_choices(tmp_path):
    env = _half(_grid()).apply_transform(translate(1.0, 2.0))
    to_file(env, tmp_path / "env")
    saved = json.loads((tmp_path / "env.json").read_text())
    assert saved["layout_type"] == "RegularGrid"
    assert saved["binning"]["kind"] == "RegularGrid"
    assert saved["derivations"] == ["subset", "apply_transform"]
    assert "layout_category" not in saved
    restored = from_file(tmp_path / "env")
    assert restored.derivations == env.derivations
    assert restored.layout_parameters == env.layout_parameters


def test_malformed_derivations_are_refused():
    record = to_dict(_grid())
    record["derivations"] = "subset"
    with pytest.raises(ValueError, match="derivations must be a list"):
        from_dict(record)
