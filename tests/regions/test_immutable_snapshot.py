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
@pytest.mark.parametrize("source_kind", ["strided", "point"])
def test_point_data_is_an_independent_read_only_copy(method, source_kind):
    source = np.arange(8.0)[::2]
    coordinates = source if source_kind == "strided" else Point(1, 2)
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
        restored.data[0] = 123


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


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        (np.array([1, 2]), "ndarray"),
        ({1, 2}, "set"),
        (np.int64(1), "int64"),
        (object(), "object"),
    ],
)
def test_non_json_metadata_names_the_region_path_and_conversion(value, kind):
    with pytest.raises(ValueError, match="JSON cannot store") as raised:
        Region("goal", "point", [1, 2], metadata={"nested": [value]})
    message = str(raised.value)
    assert "Region 'goal' metadata value metadata['nested'][0]" in message
    assert kind in message
    assert "int(...), float(...) or str(...)" in message


@pytest.mark.parametrize(
    "metadata",
    [{1: "int"}, {1: "int", "1": "str"}, {"nested": {2.5: "float"}}],
    ids=["int-key", "colliding-keys", "nested-float-key"],
)
def test_non_string_keys_are_refused_because_json_would_rename_them(metadata):
    # json.dumps turns 1 into "1": {1: "x"} would reload as {"1": "x"} and
    # {1: ..., "1": ...} would lose an entry.
    with pytest.raises(ValueError, match="not a str") as raised:
        Region("goal", "point", [1, 2], metadata=metadata)
    assert "Region 'goal' metadata key" in str(raised.value)
    assert "str(" in str(raised.value)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_metadata_numbers_are_refused(value):
    with pytest.raises(ValueError, match="must be finite") as raised:
        Region("goal", "point", [1, 2], metadata={"limits": [0.0, value]})
    assert "metadata['limits'][1]" in str(raised.value)
    assert "None" in str(raised.value)


@pytest.mark.parametrize("metadata", [None, ["color", "red"], "red"])
def test_metadata_must_be_a_mapping(metadata):
    with pytest.raises(TypeError, match="Region 'goal' metadata must be a mapping"):
        Region("goal", "point", [1, 2], metadata=metadata)


def test_accepted_metadata_round_trips_through_standard_json():
    metadata = {"number": np.float64(1.25), "flag": True, "missing": None}
    region = Region("goal", "point", [1, 2], metadata=metadata)
    assert region.metadata["number"] == 1.25
    reloaded = Region.from_dict(
        json.loads(json.dumps(region.to_dict(), allow_nan=False))
    )
    assert reloaded == region


def test_landmark_metadata_follows_the_region_rules():
    from neurospatial.landmarks import Landmark

    with pytest.raises(ValueError, match="Landmark 'cue' metadata key 1"):
        Landmark("cue", "point", [0.0, 0.0], {1: "x"})


def test_nested_tuple_inside_list_metadata_is_frozen():
    source = {"nested": [({"label": ["goal"]},)]}
    region = Region("goal", "point", [1, 2], metadata=source)
    source["nested"][0][0]["label"].append("changed")
    assert region.metadata["nested"][0][0]["label"] == ("goal",)
    with pytest.raises(TypeError):
        region.metadata["nested"][0][0]["label"] = ("changed",)


def test_circular_metadata_is_refused():
    metadata = {}
    metadata["self"] = metadata
    with pytest.raises(ValueError, match="Region 'goal' metadata contains itself"):
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
    before = list(snapshot)
    with pytest.raises(TypeError, match=r"read-only.*with_regions"):
        mutation[1](snapshot)
    assert list(snapshot) == before


@pytest.mark.parametrize("method", ["original", "copy", "deepcopy", "pickle"])
def test_snapshot_read_helpers_and_mutable_builder_reconstruction(method, tmp_path):
    builder = _builder()
    snapshot = _restore(builder._snapshot(), method)
    assert list(snapshot) == ["goal", "arena"]
    assert list(snapshot.keys()) == ["goal", "arena"]
    assert len(snapshot) == 2
    assert snapshot.get("missing") is None
    assert snapshot["arena"].area == 9
    np.testing.assert_array_equal(snapshot["goal"].center, [1, 2])
    np.testing.assert_array_equal(snapshot["arena"].center, [1.5, 1.5])

    builder.update_region("goal", point=[9, 9])
    builder.remove("arena")
    assert list(snapshot) == ["goal", "arena"]
    np.testing.assert_array_equal(snapshot["goal"].data, [1, 2])
    editable = Regions(snapshot.values())
    editable.update_region("goal", point=[3, 4])
    assert editable["goal"] is not snapshot["goal"]
    np.testing.assert_array_equal(snapshot["goal"].data, [1, 2])

    output = tmp_path / "regions.json"
    snapshot.to_json(output)
    loaded = Regions.from_json(output)
    loaded.remove("goal")
    assert list(loaded) == ["arena"]


@pytest.mark.parametrize("method", ["copy", "deepcopy", "pickle"])
def test_builder_restoration_remains_mutable_with_independent_store(method):
    builder = _builder()
    restored = _restore(builder, method)
    restored.remove("goal")
    assert list(builder) == ["goal", "arena"]
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
    assert list(regions) == ["goal", "arena"]
    assert regions["goal"] is original


@pytest.mark.parametrize("kind", ["mapping", "list"])
def test_metadata_thaw_rejects_cycles_before_exhausting_stack(kind):
    from neurospatial.regions.core import _thaw_metadata

    cyclic = {} if kind == "mapping" else []
    if kind == "mapping":
        cyclic["self"] = cyclic
    else:
        cyclic.append(cyclic)
    with pytest.raises(ValueError, match="Circular"):
        _thaw_metadata(cyclic)


def test_metadata_shared_containers_are_not_cycles():
    nested = {"colors": ["red", "blue"]}
    region = Region("goal", "point", [1, 2], metadata={"a": nested, "b": nested})
    assert region.to_dict()["metadata"] == {"a": nested, "b": nested}
