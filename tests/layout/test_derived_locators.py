"""Locator derivations preserve cell ownership and own their wrapper data."""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

from neurospatial.geometry._codecs import decode, encode
from neurospatial.geometry._locators import (
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
)
from neurospatial.ops.transforms import Affine2D


@pytest.fixture
def full_locator():
    return RectilinearLocator((np.arange(3.0),) * 2, (2, 2), None, None)


def test_subset_locator_masks_outside_parent_before_mapping(full_locator):
    mapping = np.array([-1, 0, -1, 1])
    locator = SubsetLocator(full_locator, mapping)
    points = np.array(
        [[-1.0, -1.0], [2.0, 2.0], [0.5, 0.5], [0.5, 1.5], [1.5, 1.5], [np.nan, 0.0]]
    )
    # (2, 2), the parent grid's outer corner, belongs to its kept corner cell.
    np.testing.assert_array_equal(locator.locate(points), [-1, 1, -1, 0, 1, -1])
    assert mapping[-1] == 1


@pytest.fixture(params=["subset", "transformed"])
def wrapper_and_inputs(full_locator, request):
    if request.param == "subset":
        backing = np.array([-1, -9, 0, -9, -1, -9, 1, -9])
        caller = backing[::2]
        return SubsetLocator(full_locator, caller), "parent_to_child", caller
    backing = np.zeros((6, 6))
    backing[::2, ::2] = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    caller = backing[::2, ::2]
    return TransformedLocator(full_locator, caller), "inverse", caller


def test_wrapper_owns_caller_view_without_freezing_it(wrapper_and_inputs):
    locator, field, caller = wrapper_and_inputs
    expected = caller.copy()
    stored = getattr(locator, field)
    assert caller.flags.writeable
    assert not stored.flags.writeable
    assert not np.shares_memory(stored, caller)
    caller[:] = 0
    np.testing.assert_array_equal(stored, expected)


@pytest.mark.parametrize(
    "restore",
    [copy.deepcopy, lambda obj: pickle.loads(pickle.dumps(obj))],
    ids=["deepcopy", "pickle"],
)
def test_wrapper_serialization_restores_array_protection(wrapper_and_inputs, restore):
    locator, field, _ = wrapper_and_inputs
    restored = restore(locator)
    np.testing.assert_array_equal(getattr(restored, field), getattr(locator, field))
    assert not getattr(restored, field).flags.writeable
    points = np.array([[0.5, 0.5], [-0.5, 0.5], [3.0, 3.0]])
    np.testing.assert_array_equal(restored.locate(points), locator.locate(points))


def test_wrapper_codecs_nest_base_data_and_preserve_lookup(wrapper_and_inputs):
    locator, field, _ = wrapper_and_inputs
    meta, arrays = encode(locator, "locator")
    assert meta["base"]["codec"] == "RectilinearLocator"
    assert "locator/base/edges_0" in arrays
    decoded = decode(meta, arrays, "locator")
    np.testing.assert_array_equal(getattr(decoded, field), getattr(locator, field))
    assert not getattr(decoded, field).flags.writeable
    assert all(
        not np.shares_memory(getattr(decoded, field), value)
        for value in arrays.values()
    )
    points = np.array([[0.5, 0.5], [-0.5, 0.5], [3.0, 3.0]])
    np.testing.assert_array_equal(decoded.locate(points), locator.locate(points))


def test_nested_wrappers_round_trip_without_layout_objects(full_locator):
    reflected = TransformedLocator(full_locator, np.diag([-1.0, 1.0, 1.0]))
    selected = SubsetLocator(reflected, np.array([-1, 0, -1, 1]))
    translated = TransformedLocator(
        selected, np.array([[1.0, 0.0, -2.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]])
    )
    meta, arrays = encode(translated, "locator")
    assert "locator/base/base/base/edges_0" in arrays
    decoded = decode(meta, arrays, "locator")
    points = np.array([[1.5, 4.5], [0.5, 4.5], [10.0, 10.0]])
    np.testing.assert_array_equal(decoded.locate(points), [0, 1, -1])


@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize("n_rows,fill", [(0, 0.0), (2, 0.0), (2, np.nan), (2, np.inf)])
def test_wrappers_validate_coordinate_width(wrapper_and_inputs, width, n_rows, fill):
    locator, _, _ = wrapper_and_inputs
    points = np.full((n_rows, width), fill)
    with pytest.raises(ValueError):
        locator.locate(points)


def test_transformed_locator_nonfinite_rows_do_not_poison_finite_rows(full_locator):
    locator = TransformedLocator(full_locator, np.diag([-1.0, 1.0, 1.0]))
    points = np.array([[-0.5, 0.5], [np.nan, 0.0], [np.inf, 1.0], [-1.5, 1.5]])
    expected = points.copy()
    np.testing.assert_array_equal(locator.locate(points), [0, -1, -1, 3])
    np.testing.assert_array_equal(points, expected)
    assert points.flags.writeable


def test_rectilinear_subset_keeps_full_grid_and_changes_active_map():
    mask = np.array([[True, False], [True, True]])
    parent = RectilinearLocator(
        (np.arange(3.0),) * 2, (2, 2), mask, np.array([0, -1, 1, 2])
    )
    keep = np.array([True, False, True])
    child = parent.subset(keep)
    assert isinstance(child, RectilinearLocator)
    np.testing.assert_array_equal(child.active_mask, [[True, False], [False, True]])
    np.testing.assert_array_equal(child.flat_to_active, [0, -1, -1, 1])
    np.testing.assert_array_equal(parent.active_mask, mask)
    np.testing.assert_array_equal(
        child.locate(np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5]])), [0, -1, 1]
    )
    assert keep.flags.writeable


def test_positive_diagonal_locator_maps_edges_exactly(full_locator):
    matrix = np.array([[2.0, 0.0, 0.1], [0.0, 0.5, 0.2], [0.0, 0.0, 1.0]])
    transform = Affine2D(matrix)
    child = full_locator.transformed(transform)
    assert isinstance(child, RectilinearLocator)
    for axis, scale in enumerate([2.0, 0.5]):
        np.testing.assert_array_equal(
            child.edges[axis], full_locator.edges[axis] * scale + matrix[axis, 2]
        )
    points = np.array(np.meshgrid(*full_locator.edges, indexing="ij")).reshape(2, -1).T
    np.testing.assert_array_equal(
        child.locate(transform(points)), full_locator.locate(points)
    )
    assert matrix.flags.writeable


@pytest.mark.parametrize(
    "linear",
    [
        np.diag([-1.0, 1.0]),
        np.diag([-1.0, -1.0]),
        np.array([[0.0, -1.0], [1.0, 0.0]]),
        np.array([[1.0, 0.2], [0.0, 1.0]]),
    ],
)
def test_rectilinear_fast_path_refuses_reflections_and_axis_mixing(
    full_locator, linear
):
    matrix = np.eye(3)
    matrix[:2, :2] = linear
    assert full_locator.transformed(Affine2D(matrix)) is None
