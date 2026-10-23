"""Scientific contracts for complete, unit-aware track annotation results."""

import numpy as np
import pytest

from neurospatial.annotation._track_helpers import build_track_graph_result
from neurospatial.annotation._track_state import TrackBuilderState
from neurospatial.ops.transforms import VideoCalibration, calibrate_from_scale_bar


def _state():
    state = TrackBuilderState()
    for x in (0.0, 100.0, 200.0):
        state.add_node(x, 480.0)
    state.add_edge(0, 1)
    state.add_edge(1, 2)
    return state


@pytest.mark.parametrize("calibrated", [False, True])
def test_units_and_track_length_survive_environment(calibrated):
    calibration = None
    if calibrated:
        calibration = VideoCalibration(
            transform_px_to_cm=calibrate_from_scale_bar(
                p1_px=(0.0, 0.0),
                p2_px=(100.0, 0.0),
                known_length_cm=50.0,
                frame_size_px=(640, 480),
            ),
            frame_size_px=(640, 480),
        )
    result = build_track_graph_result(_state(), calibration)
    assert result.units == ("cm" if calibrated else "pixels")
    env = result.to_environment(bin_size=5.0)
    assert env.units == result.units
    expected_length = 100.0 if calibrated else 200.0
    assert sum(
        data["distance"] for _, _, data in result.track_graph.edges(data=True)
    ) == pytest.approx(expected_length)
    assert env.dimension_ranges[0][1] - env.dimension_ranges[0][0] == pytest.approx(
        expected_length
    )


def test_result_owns_uncalibrated_positions():
    state = _state()
    result = build_track_graph_result(state, None)
    state.nodes[0] = (999.0, 999.0)
    assert result.node_positions[0] == result.pixel_positions[0] == (0.0, 480.0)


@pytest.mark.parametrize("order", [[(0, 1)], [(0, 1), (1, 0)], [(0, 1), (0, 2)]])
def test_builder_rejects_stale_manual_order(order):
    state = _state()
    state.edge_order_override = order
    with pytest.raises(
        ValueError, match=r"edge_order.*every track graph edge exactly once"
    ):
        build_track_graph_result(state, None)


@pytest.mark.parametrize("field", ["edge_order", "edges"])
@pytest.mark.parametrize("edges", [[(0, 1)], [(0, 1), (1, 0)], [(0, 1), (0, 2)]])
def test_manually_tampered_result_rejects_incomplete_coverage(field, edges):
    result = build_track_graph_result(_state(), None)._replace(**{field: edges})
    with pytest.raises(
        ValueError, match=rf"{field}.*every track graph edge exactly once"
    ):
        result.to_environment(bin_size=5.0)


def test_graph_mutation_requires_new_complete_result():
    result = build_track_graph_result(_state(), None)
    result.track_graph.add_edge(0, 2, distance=200.0)
    with pytest.raises(ValueError, match="every track graph edge exactly once"):
        result.to_environment(bin_size=5.0)


def test_edge_addition_retains_every_segment_and_oriented_order():
    state = _state()
    state.set_edge_order([(2, 1), (1, 0)])
    state.set_edge_spacing([0.0])
    result = build_track_graph_result(state, None)
    assert result.edge_order == [(2, 1), (1, 0)]
    state.add_node(300.0, 480.0)
    state.add_edge(2, 3)
    result = build_track_graph_result(state, None)
    assert {frozenset(edge) for edge in result.edge_order} == {
        frozenset(edge) for edge in state.edges
    }
    assert result.to_environment(bin_size=5.0).dimension_ranges[0][1] == pytest.approx(
        300.0
    )


@pytest.mark.parametrize("spacing", [[np.nan], [-1.0], [], [[0.0]]])
def test_builder_rejects_invalid_spacing(spacing):
    state = _state()
    state.edge_spacing_override = spacing
    with pytest.raises(ValueError, match="edge_spacing"):
        build_track_graph_result(state, None)
