"""Layout and imported coordinate units survive annotation workflows."""

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from neurospatial.annotation._track_helpers import build_track_graph_result
from neurospatial.annotation._track_layout import resolve_track_layout
from neurospatial.annotation._track_state import TrackBuilderState
from neurospatial.annotation.converters import env_from_boundary_region
from neurospatial.annotation.io import regions_from_cvat, regions_from_labelme
from neurospatial.ops.transforms import AffineND, VideoCalibration


def test_manual_order_infers_gaps_from_effective_endpoints() -> None:
    state = TrackBuilderState()
    for x, y in [(0, 0), (100, 0), (200, 0), (100, 100)]:
        state.add_node(x, y)
    for edge in [(0, 1), (1, 2), (1, 3)]:
        state.add_edge(*edge)
    state.set_edge_order([(1, 2), (0, 1), (1, 3)])
    result = build_track_graph_result(state, None)
    assert result.edge_order == [(1, 2), (0, 1), (1, 3)]
    np.testing.assert_array_equal(result.edge_spacing, [15.0, 0.0])


@pytest.mark.parametrize("start_node", [0, 2, 4])
def test_automatic_layout_covers_disconnected_graph(start_node: int) -> None:
    graph = nx.Graph([(0, 1), (2, 3)])
    graph.add_node(4)
    order, spacing = resolve_track_layout(graph, start_node=start_node)
    assert {frozenset(edge) for edge in order} == {frozenset((0, 1)), frozenset((2, 3))}
    np.testing.assert_array_equal(spacing, [15.0])


@pytest.mark.parametrize("source", ["labelme", "cvat"])
@pytest.mark.parametrize("calibrated", [False, True])
def test_import_units_survive_environment(
    tmp_path: Path, source: str, calibrated: bool
) -> None:
    calibration = (
        VideoCalibration(AffineND(np.diag([0.5, 0.5, 1.0])), (640, 480))
        if calibrated
        else None
    )
    if source == "labelme":
        path = tmp_path / "arena.json"
        path.write_text(
            '{"shapes":[{"label":"arena","points":[[0,0],[100,0],[100,100],[0,100]]}]}'
        )
        regions = regions_from_labelme(path, calibration=calibration)
    else:
        path = tmp_path / "arena.xml"
        path.write_text(
            '<annotations><image id="0" name="arena.png" width="640" height="480">'
            '<polygon label="arena" points="0,0;100,0;100,100;0,100"/>'
            "</image></annotations>"
        )
        regions = regions_from_cvat(path, calibration=calibration)
    boundary = regions["arena"]
    expected_units = "cm" if calibrated else "pixels"
    assert boundary.metadata["coord_system"] == expected_units
    assert boundary.name == "arena"
    assert boundary.kind == "polygon"
    assert boundary.data.area == pytest.approx(2500 if calibrated else 10000)
    env = env_from_boundary_region(boundary, 10.0)
    assert env.units == expected_units


@pytest.mark.parametrize("source", ["labelme", "cvat"])
def test_unit_stamp_preserves_nested_metadata_and_source(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    from shapely.geometry import Polygon

    from neurospatial.regions import Region, Regions

    original = Region(
        "arena",
        "polygon",
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        {"nested": {"tags": ["original"]}, "source": "external"},
    )
    loader_name = "load_labelme_json" if source == "labelme" else "load_cvat_xml"
    monkeypatch.setattr(
        f"neurospatial.regions.io.{loader_name}",
        lambda *args, **kwargs: Regions([original]),
    )
    importer = regions_from_labelme if source == "labelme" else regions_from_cvat
    stamped = importer("unused")["arena"]
    assert stamped.data.equals(original.data)
    assert stamped.metadata["nested"] == original.metadata["nested"]
    assert stamped.metadata["source"] == "external"
    assert "coord_system" not in original.metadata
    with pytest.raises(TypeError):
        stamped.metadata["coord_system"] = "changed"  # type: ignore[index]


@pytest.mark.parametrize(
    "coordinates", [[(0.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (np.nan, 1.0)]]
)
def test_builder_rejects_unrepresentable_geometry(
    coordinates: list[tuple[float, float]],
) -> None:
    state = TrackBuilderState(nodes=coordinates, edges=[(0, 1)])
    with pytest.raises(ValueError, match=r"finite|positive"):
        build_track_graph_result(state, None)
