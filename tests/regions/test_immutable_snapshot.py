"""Ownership and mutation contracts for region values and published snapshots."""

from __future__ import annotations

import copy
import json
import pickle
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from shapely.geometry import Point, box

from neurospatial.regions import Region, Regions


def _restore(value: Any, method: str) -> Any:
    if method == "copy":
        return copy.copy(value)
    if method == "deepcopy":
        return copy.deepcopy(value)
    if method == "pickle":
        return pickle.loads(pickle.dumps(value))
    return value


@pytest.mark.parametrize("method", ["original", "copy", "deepcopy", "pickle", "dict"])
@pytest.mark.parametrize("source_kind", ["strided", "point", "empty"])
def test_point_buffers_are_independent_and_irreversibly_read_only(method, source_kind):
    source = np.arange(8.0)[::2]
    coordinates = source if source_kind == "strided" else Point(1, 2)
    if source_kind == "empty":
        coordinates = np.array([], dtype=float)
    region = Region("goal", "point", coordinates)
    expected = np.asarray(region.data).copy()
    restored = (
        Region.from_dict(region.to_dict())
        if method == "dict"
        else _restore(region, method)
    )

    source[:] = -99
    np.testing.assert_array_equal(restored.data, expected)
    assert source.flags.writeable
    assert not np.shares_memory(restored.data, source)
    assert restored.data.dtype == np.float64
    assert not restored.data.flags.writeable
    with pytest.raises(ValueError):
        restored.data.setflags(write=True)
    if expected.size:
        with pytest.raises(ValueError):
            restored.data[0] = 123


@pytest.mark.parametrize("method", ["original", "copy", "deepcopy", "pickle"])
@pytest.mark.parametrize("change", ["shape", "dtype"])
def test_point_export_headers_cannot_poison_shared_regions(method, change):
    region = _restore(Region("goal", "point", [1.0, 2.0]), method)
    snapshot = Regions([region])._snapshot()
    exported = region.data
    center = snapshot.region_center("goal")
    if change == "shape":
        exported.shape = (1, 2)
        center.shape = (2, 1)
    else:
        exported.dtype = np.uint8
        center.dtype = np.uint8
    assert region.n_dims == 2
    assert region.data.shape == (2,)
    assert region.data.dtype == np.float64
    np.testing.assert_array_equal(region.data, [1, 2])
    np.testing.assert_array_equal(snapshot["goal"].data, [1, 2])
    np.testing.assert_array_equal(snapshot.region_center("goal"), [1, 2])
    assert isinstance(region.data.base, bytes)


@pytest.mark.parametrize("method", ["original", "copy", "deepcopy", "pickle", "dict"])
def test_nested_metadata_and_export_ownership(method):
    source = {"nested": ({"values": [1, {"label": "goal"}]},)}
    region = Region("goal", "point", [1, 2], metadata=source)
    restored = (
        Region.from_dict(region.to_dict())
        if method == "dict"
        else _restore(region, method)
    )
    source["nested"][0]["values"][1]["label"] = "changed"
    assert restored.metadata["nested"][0]["values"][1]["label"] == "goal"
    with pytest.raises(TypeError):
        restored.metadata["nested"][0]["values"][1]["label"] = "changed"
    exported = restored.to_dict()
    exported["metadata"]["nested"][0]["values"][1]["label"] = "changed"
    exported["geom"][0] = -99
    assert restored.metadata["nested"][0]["values"][1]["label"] == "goal"
    np.testing.assert_array_equal(restored.data, [1, 2])


@pytest.mark.parametrize("value", [np.array([1, 2]), {1, 2}, np.int64(1), object()])
def test_non_json_metadata_is_rejected_at_construction(value):
    with pytest.raises(TypeError, match=r"metadata.*JSON"):
        Region("goal", "point", [1, 2], metadata={"nested": [value]})


def test_json_validation_preserves_accepted_scalars_and_distinct_mapping_keys():
    metadata = {"number": np.float64(1.25), "nan": float("nan"), 1: "int", "1": "str"}
    region = Region("goal", "point", [1, 2], metadata=metadata)
    assert region.metadata["number"] == 1.25
    assert np.isnan(region.metadata["nan"])
    assert region.metadata[1] == "int"
    assert region.metadata["1"] == "str"
    json.dumps(region.to_dict())


def test_nested_tuple_inside_list_metadata_is_frozen():
    source = {"nested": [({"label": ["goal"]},)]}
    region = Region("goal", "point", [1, 2], metadata=source)
    source["nested"][0][0]["label"].append("changed")
    assert region.metadata["nested"][0][0]["label"] == ("goal",)
    with pytest.raises(TypeError):
        region.metadata["nested"][0][0]["label"] = ("changed",)


def test_circular_metadata_is_rejected_as_non_json():
    metadata = {}
    metadata["self"] = metadata
    with pytest.raises(TypeError, match=r"metadata.*JSON"):
        Region("goal", "point", [1, 2], metadata=metadata)


def _builder() -> Regions:
    return Regions(
        [
            Region("goal", "point", [1, 2], metadata={"nested": {"color": "red"}}),
            Region("arena", "polygon", box(0, 0, 3, 3)),
        ]
    )


_MUTATIONS: list[tuple[str, Callable[[Regions], Any]]] = [
    ("assignment", lambda r: r.__setitem__("new", Region("new", "point", [0, 0]))),
    ("deletion", lambda r: r.__delitem__("goal")),
    ("add", lambda r: r.add("new", point=[0, 0])),
    ("set", lambda r: r.set("goal", Region("goal", "point", [0, 0]))),
    ("update_region", lambda r: r.update_region("goal", point=[0, 0])),
    ("remove", lambda r: r.remove("goal")),
    ("buffer", lambda r: r.buffer(np.array([0, 0]), 1, "new")),
    ("clear", lambda r: r.clear()),
    ("pop", lambda r: r.pop("goal")),
    ("pop_missing_default", lambda r: r.pop("missing", None)),
    ("popitem", lambda r: r.popitem()),
    ("setdefault", lambda r: r.setdefault("goal", Region("goal", "point", [0, 0]))),
    ("update", lambda r: r.update({"new": Region("new", "point", [0, 0])})),
    ("update_noop", lambda r: r.update()),
]


@pytest.mark.parametrize("method", ["original", "copy", "deepcopy", "pickle"])
@pytest.mark.parametrize("mutation", _MUTATIONS, ids=[name for name, _ in _MUTATIONS])
@pytest.mark.parametrize("empty", [False, True])
def test_snapshot_rejects_every_mutator_including_noops(method, mutation, empty):
    builder = Regions() if empty else _builder()
    snapshot = _restore(builder._snapshot(), method)
    assert type(snapshot) is Regions
    before = snapshot.list_names()
    with pytest.raises(TypeError, match=r"read-only.*with_regions"):
        mutation[1](snapshot)
    assert snapshot.list_names() == before


@pytest.mark.parametrize("method", ["original", "copy", "deepcopy", "pickle"])
def test_snapshot_read_helpers_and_mutable_builder_reconstruction(method, tmp_path):
    builder = _builder()
    snapshot = _restore(builder._snapshot(), method)
    assert list(snapshot) == ["goal", "arena"]
    assert list(snapshot.keys()) == ["goal", "arena"]
    assert len(snapshot) == 2
    assert snapshot.get("missing") is None
    assert snapshot.area("arena") == 9
    np.testing.assert_array_equal(snapshot.region_center("goal"), [1, 2])
    np.testing.assert_array_equal(snapshot.region_center("arena"), [1.5, 1.5])

    builder.update_region("goal", point=[9, 9])
    builder.remove("arena")
    assert snapshot.list_names() == ["goal", "arena"]
    np.testing.assert_array_equal(snapshot["goal"].data, [1, 2])
    editable = Regions(snapshot.values())
    editable.update_region("goal", point=[3, 4])
    assert editable["goal"] is not snapshot["goal"]
    np.testing.assert_array_equal(snapshot["goal"].data, [1, 2])

    output = tmp_path / "regions.json"
    snapshot.to_json(output)
    loaded = Regions.from_json(output)
    loaded.remove("goal")
    assert loaded.list_names() == ["arena"]


@pytest.mark.parametrize("method", ["copy", "deepcopy", "pickle"])
def test_builder_restoration_remains_mutable_with_independent_store(method):
    builder = _builder()
    restored = _restore(builder, method)
    restored.remove("goal")
    assert builder.list_names() == ["goal", "arena"]
    restored.add("new", point=[3, 4])
    assert "new" not in builder


@pytest.mark.parametrize("invalid", ["point", "metadata"])
def test_failed_region_replacement_preserves_original_value_and_order(invalid):
    regions = _builder()
    original = regions["goal"]
    kwargs = (
        {"point": [[1, 2]]}
        if invalid == "point"
        else {"point": [3, 4], "metadata": {"bad": {1, 2}}}
    )
    with pytest.raises((TypeError, ValueError)):
        regions.update_region("goal", **kwargs)
    assert regions.list_names() == ["goal", "arena"]
    assert regions["goal"] is original
