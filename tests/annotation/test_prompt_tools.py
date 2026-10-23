"""Exercise assisted clicks through napari's mouse callback dispatcher."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal


@pytest.fixture
def prompts():
    pytest.importorskip("napari")
    from napari.components import ViewerModel
    from napari.layers import Points

    from neurospatial.annotation._prompt_tools import PromptTools

    viewer = ViewerModel()
    inside = Points(np.empty((0, 2)), ndim=2)
    outside = Points(np.empty((0, 2)), ndim=2)
    tools = PromptTools(viewer, inside, outside, (20, 30))
    yield SimpleNamespace(viewer=viewer, inside=inside, outside=outside, tools=tools)
    tools.dispose()


def _event(kind, position=(4, 6), *, pos=(40, 60), button=1, modifiers=()):
    return SimpleNamespace(
        type=kind,
        position=position,
        pos=np.asarray(pos, dtype=float),
        button=button,
        modifiers=modifiers,
        camera_zoom=1.0,
        handled=False,
        is_dragging=kind != "mouse_press",
        view_direction=None,
        dims_displayed=[0, 1],
    )


def _gesture(layer, *, moves=(), **kwargs):
    from napari.utils._proxies import ReadOnlyWrapper
    from napari.utils.interactions import (
        mouse_move_callbacks,
        mouse_press_callbacks,
        mouse_release_callbacks,
    )

    event = ReadOnlyWrapper(_event("mouse_press", **kwargs), exceptions=("handled",))
    mouse_press_callbacks(layer, event)
    for position in moves:
        mouse_move_callbacks(
            layer, _event("mouse_move", **(kwargs | {"pos": position}))
        )
    release = _event("mouse_release", **kwargs)
    mouse_release_callbacks(layer, release)
    return release


def test_left_and_right_add_to_separate_sets_without_switching(prompts):
    p = prompts
    p.inside.mode = "add"
    p.tools.activate()
    assert p.inside.mode == "pan_zoom"
    assert p.inside.mouse_pan
    assert p.inside.mouse_zoom
    assert _gesture(p.inside).handled
    assert _gesture(p.inside, button=2, position=(7, 8)).handled
    assert p.viewer.layers.selection.active is p.inside
    assert_array_equal(p.inside.data, [[4, 6]])
    assert_array_equal(p.outside.data, [[7, 8]])
    assert not p.inside.selected_data
    assert p.outside.selected_data == {0}


def test_either_active_prompt_layer_uses_same_button_meanings(prompts):
    p = prompts
    p.tools.activate(p.outside)
    _gesture(p.outside)
    _gesture(p.outside, button=2)
    assert_array_equal(p.inside.data, [[4, 6]])
    assert_array_equal(p.outside.data, [[4, 6]])


@pytest.mark.parametrize("button", [1, 2])
def test_world_coordinates_are_transformed_once_and_bounded(
    prompts, monkeypatch, button
):
    p = prompts
    target = p.inside if button == 1 else p.outside
    target.scale = (2, 3)
    target.translate = (10, -5)
    convert = target.world_to_data
    calls = []

    def world_to_data(position):
        calls.append(position)
        return convert(position)

    monkeypatch.setattr(target, "world_to_data", world_to_data)
    p.tools.activate()
    _gesture(p.inside, button=button, position=(18, 13))
    assert_array_equal(target.data, [[4, 6]])
    assert len(calls) == 1
    # Integer data coordinates are pixel centers, so the 20 by 30 image covers
    # rows [-0.5, 19.5) and columns [-0.5, 29.5).
    for position in [(8.8, 0), (49, 0), (20, 83.5), (np.nan, 4)]:
        _gesture(p.inside, button=button, position=position)
    assert len(target.data) == 1
    for position in [(9.2, 0), (48.8, 83.3)]:
        _gesture(p.inside, button=button, position=position)
    assert_allclose(target.data[1:], [[-0.4, 5 / 3], [19.4, 88.3 / 3]])


def test_alt_click_removes_nearest_of_either_set_and_respects_distance(prompts):
    p = prompts
    p.tools.activate()
    _gesture(p.inside, position=(2, 2))
    _gesture(p.inside, button=2, position=(8, 8))
    _gesture(p.inside, position=(8, 9), modifiers=("Alt",))
    assert len(p.outside.data) == 0
    assert_array_equal(p.inside.data, [[2, 2]])
    _gesture(p.inside, position=(19, 29), modifiers=("Alt",))
    assert len(p.inside.data) == 1


def test_nearest_uses_canvas_distance_and_layer_transforms(prompts):
    p = prompts
    p.outside.add([2, 2])
    p.outside.scale = (2, 3)
    p.outside.translate = (1, 1)
    assert not p.tools.remove_nearest((5, 14), zoom=2)
    assert p.tools.remove_nearest((5, 8), zoom=2)
    assert len(p.outside.data) == 0


def test_drag_does_not_add_even_when_cursor_returns_to_press(prompts):
    p = prompts
    p.tools.activate()
    for button in (1, 2):
        event = _gesture(p.inside, button=button, moves=[(60, 60), (40, 60)])
        assert not event.handled
    assert not len(p.inside.data)
    assert not len(p.outside.data)
    assert p.inside.mouse_pan and p.inside.mouse_zoom


@pytest.mark.parametrize("modifiers", [("Shift",), ("Control",), ("Meta",)])
def test_modified_clicks_preserve_camera_shortcuts(prompts, modifiers):
    p = prompts
    p.tools.activate()
    assert not _gesture(p.inside, modifiers=modifiers).handled
    assert not len(p.inside.data)


def test_selected_deletion_and_undo_follow_manual_moves_and_removals(prompts):
    p = prompts
    p.tools.activate()
    _gesture(p.inside, position=(1, 1))
    _gesture(p.inside, button=2, position=(2, 2))
    _gesture(p.inside, position=(3, 3))
    # A native point move emits CHANGED and must retain its addition order.
    p.inside.data[1] = [5, 5]
    p.inside.events.data(action="changed", data_indices=(1,))
    p.inside.remove([0])
    assert p.tools.undo_last()
    assert not len(p.inside.data)
    assert_array_equal(p.outside.data, [[2, 2]])
    p.outside.selected_data = {0}
    assert p.tools.remove_selected() == 1
    assert not p.tools.undo_last()


def test_undo_survives_direct_reorder_and_replacement(prompts):
    p = prompts
    p.inside.add([[1, 1], [2, 2], [3, 3]])
    p.inside.data = p.inside.data[[2, 0, 1]]
    assert p.tools.undo_last()
    assert_array_equal(p.inside.data, [[1, 1], [2, 2]])
    p.inside.data = np.array([[2, 2]])
    assert p.tools.undo_last()
    assert not len(p.inside.data)
    assert not p.tools.undo_last()


def test_moving_an_earlier_point_onto_later_point_preserves_undo_order(prompts):
    p = prompts
    p.inside.add([[1, 1], [2, 2], [3, 3]])
    p.inside.data[0] = [3, 3]
    p.inside.events.data(action="changed", data_indices=(0,))
    assert p.tools.undo_last()
    assert_array_equal(p.inside.data, [[3, 3], [2, 2]])


def test_native_point_mode_deactivates_and_does_not_duplicate_additions(prompts):
    p = prompts
    p.tools.activate()
    p.inside.mode = "add"
    assert not p.tools.active
    _gesture(p.inside)
    assert_array_equal(p.inside.data, [[4, 6]])
    assert not len(p.outside.data)


def test_selecting_ordinary_layer_disables_until_explicit_activation(prompts):
    p = prompts
    p.tools.activate()
    shapes = p.viewer.add_shapes(ndim=2)
    shapes.mode = "add_polygon"
    assert not p.tools.active
    p.viewer.layers.selection.active = p.inside
    assert not _gesture(p.inside).handled
    assert not len(p.inside.data)
    assert shapes.mode == "add_polygon"
    assert shapes.mouse_drag_callbacks != p.inside.mouse_drag_callbacks
    p.tools.activate()
    _gesture(p.inside)
    assert len(p.inside.data) == 1


def test_dispose_keeps_other_callbacks_and_is_safe_during_gesture(prompts):
    from napari.utils._proxies import ReadOnlyWrapper
    from napari.utils.interactions import mouse_press_callbacks, mouse_release_callbacks

    p = prompts
    seen = []

    def unrelated(layer, event):
        seen.append(event.type)

    p.inside.mouse_drag_callbacks.append(unrelated)
    p.tools.activate()
    event = ReadOnlyWrapper(_event("mouse_press"), exceptions=("handled",))
    mouse_press_callbacks(p.inside, event)
    p.tools.dispose()
    p.tools.dispose()
    mouse_release_callbacks(p.inside, _event("mouse_release"))
    assert not len(p.inside.data)
    assert unrelated in p.inside.mouse_drag_callbacks
    assert p.tools._mouse_drag not in p.inside.mouse_drag_callbacks
    assert not p.tools._connections
    assert not p.inside._mouse_drag_gen
    assert seen == ["mouse_press"]
    assert not p.tools.undo_last()
    assert p.tools.remove_selected() == 0
    p.tools.activate()
    assert not p.tools.active


def test_left_click_can_add_outside_points_without_a_right_button(prompts):
    p = prompts
    p.tools.activate()
    p.tools.left_click_outside = True
    _gesture(p.inside)
    _gesture(p.inside, button=2, position=(8, 9))
    assert_array_equal(p.outside.data, [[4, 6]])
    assert_array_equal(p.inside.data, [[8, 9]])
