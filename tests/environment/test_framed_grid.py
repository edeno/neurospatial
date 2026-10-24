"""Retained source grids and transformed cells keep identities and ownership."""

from __future__ import annotations

import pickle
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pytest
from shapely.affinity import affine_transform
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.binning import FramedGrid, RectilinearGrid
from neurospatial.geometry._codecs import encode
from neurospatial.geometry._polygon import materialize_cells
from neurospatial.geometry._polygon_cells import PolygonCells
from neurospatial.io import from_dict, from_file, to_dict, to_file
from neurospatial.layout import create_layout
from neurospatial.ops.transforms import Affine2D, AffineND


def grid(n_dims=2, hole=True):
    mask = np.ones((3,) * n_dims, dtype=bool)
    if hole:
        mask[(1,) * n_dims] = False
    return Environment.from_grid_mask(mask, grid_edges=(np.arange(4.0),) * n_dims)


def matrix(n_dims=2, kind="shear", offset=0.0):
    value = np.eye(n_dims + 1)
    value[0, -1] = offset
    if kind == "shear":
        value[0, 1] = 0.3
    elif kind == "rotation":
        value[:2, :2] = [[0.0, -1.0], [1.0, 0.0]]
    elif kind == "reflection":
        value[0, 0] = -1.0
    return value


def moved(env, value):
    return env.apply_transform(AffineND(value))


@pytest.mark.parametrize("n_dims", [2, 3])
@pytest.mark.parametrize("kind", ["shear", "rotation", "reflection"])
@pytest.mark.parametrize("offset", [0.0, 1e9])
def test_frame_retains_source_mask_order_and_world_centers(n_dims, kind, offset):
    env = grid(n_dims)
    value = matrix(n_dims, kind, offset)
    child = moved(env, value)
    frame = child._binning.grid_frame
    assert isinstance(frame, FramedGrid)
    assert child._binning.rectilinear is None
    np.testing.assert_array_equal(frame.source.active_mask, env.active_mask)
    for actual, expected in zip(frame.source.edges, env.grid_edges, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(frame.forward, value)
    np.testing.assert_array_equal(child.edge_pairs, env.edge_pairs)
    np.testing.assert_array_equal(
        child.bin_at(child.bin_centers), np.arange(env.n_bins)
    )
    np.testing.assert_allclose(frame.to_world(env.bin_centers), child.bin_centers)
    np.testing.assert_allclose(
        frame.to_source(child.bin_centers), env.bin_centers, atol=1e-6
    )


def oblique(n_dims, angle, shear, offset):
    """Rotate the first two axes by ``angle``, then shear and translate."""
    value = np.eye(n_dims + 1)
    value[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    value[0, :-1] += shear * value[1, :-1]
    value[:-1, -1] = offset
    return value


def half_step_lattice(n_dims):
    """Cell centers, shared faces, edges and corners, inside and just outside."""
    steps = np.arange(-0.5, 4.0, 0.5)
    return np.stack(np.meshgrid(*[steps] * n_dims, indexing="ij"), -1).reshape(
        -1, n_dims
    )


@pytest.mark.parametrize("n_dims", [2, 3])
@pytest.mark.parametrize("angle,shear,offset", [(0.123, 0.3, 13.7), (2.0, 0.0, 1e4)])
def test_rotating_the_grid_and_positions_together_keeps_each_bin(
    n_dims, angle, shear, offset
):
    # A position on a face shared by two cells belongs to the cell above it on
    # any rectangular grid. Rotating the grid together with the positions must
    # not move it into the other cell, whatever the rounding of the rotation.
    env = grid(n_dims)
    child = moved(env, oblique(n_dims, angle, shear, offset))
    source = half_step_lattice(n_dims)
    world = child._binning.grid_frame.to_world(source)
    expected = env.bin_at(source)
    np.testing.assert_array_equal(child.bin_at(world), expected)

    # The binning's own locator applies the same rule, also after a subset:
    # the kept cells close their outline, as on the axis-aligned grid, so a
    # point on a face shared with a dropped cell belongs to the kept cell.
    np.testing.assert_array_equal(child._binning.locator.locate(world), expected)
    keep = np.arange(child.n_bins) % 3 != 1
    subset = child.subset(mask=keep)
    selected_first = env.subset(mask=keep).bin_at(source)
    renumber = np.full(child.n_bins + 1, -1)
    renumber[np.flatnonzero(keep)] = np.arange(keep.sum())
    owned = expected >= 0
    assert np.any(selected_first[owned] != renumber[expected[owned]])
    np.testing.assert_array_equal(subset.bin_at(world), selected_first)
    np.testing.assert_array_equal(subset._binning.locator.locate(world), selected_first)

    # A combined environment finds the same bins in its rotated component.
    far = moved(grid(n_dims), oblique(n_dims, angle, shear, offset + 100.0))
    joined = Environment.combine({"near": child, "far": far})
    np.testing.assert_array_equal(joined.bin_at(world), expected)

    # The bins of start allocation are the bins of bin_at.
    times = np.arange(len(source), dtype=np.float64)
    np.testing.assert_array_equal(
        child.occupancy(times, world, time_allocation="start"),
        env.occupancy(times, source, time_allocation="start"),
    )


@pytest.mark.parametrize("n_dims", [2, 3])
def test_composed_frames_and_subsets_keep_logical_ids(n_dims):
    env = grid(n_dims)
    first, second = matrix(n_dims, "rotation"), matrix(n_dims, "shear")
    child = moved(moved(env, first), second)
    np.testing.assert_allclose(child._binning.grid_frame.forward, second @ first)
    keep = np.arange(child.n_bins) % 3 != 1
    subset = child.subset(mask=keep)
    frame = subset._binning.grid_frame
    expected = env.active_mask.copy()
    expected.flat[np.flatnonzero(expected)] = keep
    np.testing.assert_array_equal(frame.source.active_mask, expected)
    np.testing.assert_array_equal(
        subset.bin_at(child.bin_centers[keep]), np.arange(keep.sum())
    )
    np.testing.assert_array_equal(subset.bin_at(child.bin_centers[~keep]), -1)
    original = moved(child, np.linalg.inv(second @ first))
    np.testing.assert_allclose(original.bin_centers, env.bin_centers, atol=1e-12)


@pytest.mark.parametrize(
    "transport",
    ["dict", "file", pytest.param("nwb", marks=pytest.mark.nwb), "pickle"],
)
@pytest.mark.parametrize("n_dims", [2, 3])
def test_frame_round_trip_owns_all_arrays(
    transport, n_dims, tmp_path, forbid_layout_builds
):
    # Create before forbidding builders (the fixture patches classes at setup).
    from neurospatial.binning import (
        BinConnections,
        Binning,
        MissingFaces,
        RectilinearLocator,
        SpatialBins,
    )

    shape = (2,) * n_dims
    edges = (np.arange(3.0),) * n_dims
    centers = (
        np.array(np.meshgrid(*([np.arange(2.0) + 0.5] * n_dims), indexing="ij"))
        .reshape(n_dims, -1)
        .T
    )
    locator = RectilinearLocator(edges, shape, np.ones(shape, bool))
    bins = SpatialBins(centers, np.ones(len(centers)), n_dims)
    native = Environment.from_bins(
        Binning(
            "explicit",
            bins,
            BinConnections.from_arrays(np.empty((0, 2), dtype=np.intp), np.empty(0)),
            locator,
            MissingFaces("No edges."),
            grid=RectilinearGrid(edges, shape, np.ones(shape, dtype=bool)),
        )
    )
    env = moved(native, oblique(n_dims, 0.123, 0.3, 13.7))
    if transport == "dict":
        restored = from_dict(to_dict(env))
    elif transport == "file":
        to_file(env, tmp_path / "frame")
        restored = from_file(tmp_path / "frame")
    elif transport == "pickle":
        restored = pickle.loads(pickle.dumps(env))
    else:
        pynwb = pytest.importorskip("pynwb")
        from neurospatial.io.nwb import read_environment, write_environment

        record = pynwb.NWBFile(
            "frame", "frame", datetime(2020, 1, 1, tzinfo=timezone.utc)
        )
        write_environment(record, env)
        path = tmp_path / "frame.nwb"
        with pynwb.NWBHDF5IO(str(path), "w") as io:
            io.write(record)
        with pynwb.NWBHDF5IO(str(path), "r") as io:
            restored = read_environment(io.read())
    actual, expected = restored._binning.grid_frame, env._binning.grid_frame
    for a, b in zip(
        (
            actual.forward,
            actual.inverse,
            actual.source.active_mask,
            *actual.source.edges,
        ),
        (
            expected.forward,
            expected.inverse,
            expected.source.active_mask,
            *expected.source.edges,
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)
        assert not a.flags.writeable
        assert not np.shares_memory(a, b)
    # The restored locator still assigns points on cell faces like the grid
    # before the transform.
    source = half_step_lattice(n_dims)
    world = expected.to_world(source)
    np.testing.assert_array_equal(
        restored._binning.locator.locate(world), native.bin_at(source)
    )
    np.testing.assert_array_equal(
        restored.bin_at(env.bin_centers), np.arange(env.n_bins)
    )


@pytest.mark.parametrize("corruption", ["shift", "mask", "edges", "inverse"])
def test_publication_rejects_inconsistent_frame(corruption):
    env = moved(grid(), matrix())
    binning = env._binning
    frame = binning.grid_frame
    forward = frame.forward.copy()
    source = frame.source
    if corruption == "shift":
        forward[0, -1] += 0.1
    elif corruption == "mask":
        mask = source.active_mask.copy()
        mask.flat[0], mask.flat[4] = False, True
        source = RectilinearGrid(source.edges, source.shape, mask)
    elif corruption == "edges":
        edges = tuple(edge + 0.1 for edge in source.edges)
        source = RectilinearGrid(edges, source.shape, source.active_mask)
    frame = FramedGrid(source, forward)
    if corruption == "inverse":
        frame.inverse.setflags(write=True)
        frame.inverse[0, -1] += 0.1
    candidate = replace(binning, grid=frame)
    with pytest.raises(ValueError, match=r"[Ff]rame"):
        Environment.from_bins(candidate)


def test_partial_clipping_drops_tensor_grid_and_keeps_parent_map():
    env = moved(grid(hole=False), matrix())
    child = env.clip(box(0.25, 0.25, 2.75, 2.75))
    assert child._binning.grid_frame is None
    assert isinstance(child._binning.shapes, PolygonCells)
    assert child.clip_mapping is not None


@pytest.mark.parametrize("kind", ["reflection", "shear"])
def test_hexagon_conversion_retains_exact_vertices(kind):
    env = Environment.from_layout(
        create_layout(
            "Hexagonal", hexagon_width=4.0, dimension_ranges=[(0.0, 12.0), (0.0, 12.0)]
        )
    )
    value = matrix(kind=kind)
    child = env.apply_transform(Affine2D(value))
    assert isinstance(child._binning.shapes, PolygonCells)
    coefficients = [*value[0, :2], *value[1, :2], value[0, 2], value[1, 2]]
    for original, actual in zip(
        materialize_cells(env._binning), materialize_cells(child._binning), strict=True
    ):
        expected = affine_transform(original, coefficients)
        assert actual.normalize().equals_exact(expected.normalize(), tolerance=1e-12)
    loaded = from_dict(to_dict(child))
    np.testing.assert_array_equal(
        loaded.bin_at(child.bin_centers), np.arange(env.n_bins)
    )


def test_binning_and_frame_codec_versions():
    meta, _ = encode(moved(grid(), matrix())._binning, "binning")
    assert meta["codec_version"] == 6
    assert meta["grid"]["codec"] == "FramedGrid"
    assert meta["grid"]["codec_version"] == 1


@pytest.mark.parametrize("kind", ["grid", "hexagonal"])
@pytest.mark.parametrize("reflect", [False, True])
@pytest.mark.parametrize(
    "angle,scale", [(1.0, 1.0), (1.0, 0.25), (2.2250738585e-313, 1.9375)]
)
def test_similarity_inverse_after_cancelling_translations_keeps_lookup_and_storage(
    kind, reflect, angle, scale
):
    from neurospatial.ops.transforms import translate

    env = (
        grid()
        if kind == "grid"
        else Environment.from_samples(
            np.array([[0.0, 0.0], [4.0, 4.0]]),
            bin_size=1.0,
            layout="hexagonal",
            infer_active_bins=False,
        )
    )
    value = np.array(
        [
            [np.cos(angle), -np.sin(angle), 10.0],
            [np.sin(angle), np.cos(angle), -7.0],
            [0.0, 0.0, 1.0],
        ]
    )
    value[:2, :2] *= scale
    if reflect:
        value[:2, 0] *= -1
    mapping = Affine2D(value)
    child = (
        env.apply_transform(mapping)
        .apply_transform(translate(3.0, 2.0))
        .apply_transform(translate(-3.0, -2.0))
        .apply_transform(mapping.inverse())
    )
    np.testing.assert_allclose(child.bin_centers, env.bin_centers, atol=1e-12)
    np.testing.assert_array_equal(child.bin_at(env.bin_centers), np.arange(env.n_bins))
    loaded = from_dict(to_dict(child))
    np.testing.assert_array_equal(loaded.bin_at(env.bin_centers), np.arange(env.n_bins))
    if kind == "grid":
        bad = child._binning.grid_frame.forward.copy()
        bad[0, -1] += 0.01
        with pytest.raises(ValueError, match=r"grid_frame\.forward"):
            Environment.from_bins(
                replace(
                    child._binning,
                    grid=FramedGrid(child._binning.grid_frame.source, bad),
                ),
                arena=child.arena,
            )


@pytest.mark.parametrize("angle,reflect", [(0.3, True), (1.7, False)])
def test_hole_ring_order_does_not_change_polygon_identity_after_inverse(angle, reflect):
    from neurospatial.ops.transforms import translate

    polygon = (
        box(0, 0, 1, 1)
        .difference(box(0.1, 0.1, 0.2, 0.2))
        .difference(box(0.1, 0.7, 0.2, 0.8))
    )
    env = Environment.from_grid_mask(
        np.ones((1, 1), dtype=bool), grid_edges=(np.arange(2.0),) * 2
    ).clip(polygon)
    value = np.array(
        [
            [np.cos(angle), -np.sin(angle), 10.0],
            [np.sin(angle), np.cos(angle), -7.0],
            [0.0, 0.0, 1.0],
        ]
    )
    if reflect:
        value[:2, 0] *= -1
    mapping = Affine2D(value)
    child = (
        env.apply_transform(mapping)
        .apply_transform(translate(3.0, 2.0))
        .apply_transform(translate(-3.0, -2.0))
        .apply_transform(mapping.inverse())
    )
    loaded = from_dict(to_dict(child))
    points = np.array([[0.05, 0.15], [0.15, 0.15], [0.15, 0.75], [0.5, 0.5]])
    np.testing.assert_array_equal(loaded.bin_at(points), env.bin_at(points))
    corrupted = loaded._binning.shapes.polygons[0].difference(box(0.6, 0.6, 0.7, 0.7))
    with pytest.raises(ValueError, match="polygon cell 0"):
        Environment.from_bins(
            replace(loaded._binning, shapes=PolygonCells((corrupted,))),
            arena=loaded.arena,
        )


def test_a_replaced_framed_locator_keeps_its_frame():
    env = Environment.from_grid_mask(
        np.ones((4, 4), bool), grid_edges=(np.arange(5.0),) * 2
    ).apply_transform(
        AffineND(np.array([[0.6, -0.8, 1.0], [0.8, 0.6, 2.0], [0, 0, 1]]))
    )
    locator = env._binning.locator
    assert locator.frame is not None
    assert replace(locator).frame is locator.frame
