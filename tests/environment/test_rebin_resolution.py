"""New-resolution re-binning uses the arena and owned construction data."""

import copy
import pickle
import re
import warnings
from dataclasses import replace

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, box

from neurospatial import Environment
from neurospatial.arena import OpenArena, UnknownArena
from neurospatial.binning import KnownFaces
from neurospatial.layout import create_layout
from neurospatial.ops.transforms import Affine2D, AffineND


def grid():
    return Environment.from_grid_mask(
        np.ones((6, 6), dtype=bool),
        grid_edges=(np.arange(7.0),) * 2,
        units="cm",
        frame="lab",
    )


def test_rebin_new_resolution_keeps_the_selection_after_an_analysis_crop():
    env = grid().subset(mask=grid().bin_centers[:, 0] < 2)
    coarse = env.rebin(bin_size=2.0)
    expected = np.array([[1, y] for y in [1, 3, 5]])
    np.testing.assert_array_equal(coarse.bin_centers, expected)
    np.testing.assert_allclose(coarse.bin_sizes, 4.0)
    assert coarse.units == "cm" and coarse.frame == "lab"
    np.testing.assert_array_equal(coarse.rebin(bin_size=1).bin_centers, env.bin_centers)


def test_rebin_reports_original_area_left_without_a_new_bin():
    # 4-unit cells over a 6-unit arena: only the cell centered at (2, 2) lies
    # strictly inside, so 20 of the 36 original bins lose their bin.
    env = Environment.from_polygon(box(0, 0, 6, 6), bin_size=1.0)
    with pytest.warns(UserWarning, match="without a new bin") as record:
        coarse = env.rebin(bin_size=4.0)
    lost = coarse.bin_at(env.bin_centers) < 0
    assert np.count_nonzero(lost) == 20
    (message,) = [str(item.message) for item in record]
    assert re.search(r"20 of 36 original bins \(55\.6% of their area\)", message)
    assert "divisor" in message and "-1" in message


@pytest.mark.parametrize("size", [1.0, 2.0, 3.0, 0.5])
def test_rebin_covering_the_arena_does_not_warn(size):
    env = Environment.from_polygon(box(0, 0, 6, 6), bin_size=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        coarse = env.rebin(bin_size=size)
    assert np.all(coarse.bin_at(env.bin_centers) >= 0)
    assert coarse.bin_sizes.sum() == pytest.approx(36.0)


def _u_maze():
    """Fine U-maze: a 2-wide barrier separates two arms joined below y = 5."""
    footprint = box(0, 0, 20, 20).difference(box(9, 5, 11, 20))
    return Environment.from_polygon(
        footprint, bin_size=1.0, connect_diagonal_neighbors=False
    )


def _arm_to_arm_path(env):
    """Shortest navigation path between the tops of the two arms."""
    start, end = env.bin_at(np.array([[5.5, 18.5], [14.5, 18.5]]))
    return nx.shortest_path_length(env.connectivity, start, end, weight="distance")


@pytest.mark.parametrize(
    "rebin",
    [lambda env: env.rebin(bin_size=5.0), lambda env: env.rebin(5)],
    ids=["bin_size", "factor"],
)
def test_rebin_reports_connections_across_a_barrier_in_the_outline(rebin):
    # 5-unit cells on both sides of the 2-wide barrier have centers inside the
    # arms, so their grid connection jumps the barrier: the path between the
    # arm tops shrinks from 37 to 5. The warning names the crossing.
    fine = _u_maze()
    assert _arm_to_arm_path(fine) == pytest.approx(37.0)
    with pytest.warns(UserWarning, match="only outside the arena") as record:
        coarse = rebin(fine)
    assert _arm_to_arm_path(coarse) == pytest.approx(5.0)
    (message,) = [
        str(item.message)
        for item in record
        if "only outside the arena" in str(item.message)
    ]
    # Count independently: neighbors whose shared face lies in the barrier.
    domain = coarse.arena.polygonal_domain()
    half = coarse.bin_sizes[0] ** 0.5 / 2
    crossing = 0
    for u, v in coarse.connectivity.edges:
        a, b = (box(*(c - half), *(c + half)) for c in coarse.bin_centers[[u, v]])
        face = a.boundary.intersection(b.boundary)
        crossing += face.length > 0 and face.intersection(domain).length == 0
    assert crossing > 0
    assert re.search(rf"\b{crossing} connections\b", message)
    assert "walls" in message


def test_rebin_within_the_arms_does_not_report_crossings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        coarse = _u_maze().rebin(bin_size=0.5)
    assert _arm_to_arm_path(coarse) > 36.0


@pytest.mark.parametrize(
    "restore",
    [
        copy.deepcopy,
        lambda x: pickle.loads(pickle.dumps(x)),
        lambda x: Environment.from_dict(x.to_dict()),
    ],
)
def test_spec_survives_restore_and_is_independent_of_provenance(restore):
    env = restore(grid())
    env.layout_parameters.clear()
    coarse = env.rebin(bin_size=(2.0, 3.0))
    np.testing.assert_array_equal(
        coarse.bin_centers, [[x, y] for x in [1, 3, 5] for y in [1.5, 4.5]]
    )
    assert env._binning.grid_spec is not None
    assert not env._binning.grid_spec.forward.flags.writeable


def test_rebin_retains_grid_frame_and_uses_physical_axis_lengths():
    from neurospatial.ops.transforms import Affine2D

    angle = 0.4
    matrix = np.array(
        [
            [2 * np.cos(angle), -3 * np.sin(angle), 7],
            [2 * np.sin(angle), 3 * np.cos(angle), -5],
            [0, 0, 1],
        ]
    )
    transform = Affine2D(matrix)
    parent = grid().apply_transform(transform, units="cm")
    coarse = parent.rebin(bin_size=(4.0, 9.0))
    local = np.array([[x, y] for x in [1, 3, 5] for y in [1.5, 4.5]])
    np.testing.assert_allclose(
        coarse.bin_centers, local @ matrix[:2, :2].T + matrix[:2, 2]
    )
    np.testing.assert_allclose(coarse.bin_sizes, 36.0)
    restored = Environment.from_dict(coarse.to_dict()).rebin(bin_size=(4.0, 9.0))
    np.testing.assert_allclose(restored.bin_centers, coarse.bin_centers)


def test_clip_keeps_spec_and_rebinning_keeps_the_crop():
    clipped = grid().clip(box(0.2, 0.2, 1.8, 1.8))
    assert clipped._binning.grid_spec is not None
    coarse = clipped.rebin(bin_size=2)
    np.testing.assert_array_equal(coarse.bin_centers, [[1.0, 1.0]])


def test_rebin_preserves_complete_face_walls_and_refuses_splitting_cells():
    env = Environment.from_grid_mask(
        np.ones((6, 6), dtype=bool),
        grid_edges=(np.arange(7.0),) * 2,
        arena=OpenArena(box(0, 0, 6, 6), walls=(LineString([(2, 0), (2, 6)]),)),
    )
    coarse = env.rebin(bin_size=2)
    assert all(
        (coarse.bin_centers[u, 0] < 2) == (coarse.bin_centers[v, 0] < 2)
        for u, v in coarse.edge_pairs
    )
    with pytest.raises(ValueError, match=r"wall 'wall_0' cuts through bin"):
        env.rebin(bin_size=3)


@pytest.mark.parametrize("size", [0, -1, True, np.nan, np.inf, (1,), (1, 0), "2"])
def test_invalid_new_size_refuses(size):
    with pytest.raises((TypeError, ValueError), match="bin_size"):
        grid().rebin(bin_size=size)


def test_missing_spec_unknown_arena_and_mutually_exclusive_arguments():
    parent = grid()
    with pytest.raises(ValueError, match="exactly one"):
        parent.rebin(2, bin_size=2)
    with pytest.raises(ValueError, match="exactly one"):
        parent.rebin()
    explicit = replace(parent._binning, grid_spec=None)
    explicit = replace(explicit, faces=KnownFaces(parent._binning.faces.build()))
    with pytest.raises(NotImplementedError, match="do not record"):
        Environment.from_bins(explicit, arena=parent.arena).rebin(bin_size=2)
    with pytest.raises(NotImplementedError, match="known arena"):
        Environment.from_bins(
            replace(parent._binning, faces=explicit.faces), arena=UnknownArena(2)
        ).rebin(bin_size=2)


@pytest.mark.parametrize("dims", [1, 3])
def test_new_resolution_nd_and_file_round_trip(tmp_path, dims):
    original = Environment.from_grid_mask(
        np.ones((4,) * dims, dtype=bool), grid_edges=(np.arange(5.0),) * dims
    )
    path = tmp_path / "arena"
    original.to_file(path)
    coarse = Environment.from_file(path).rebin(bin_size=2.0)
    assert coarse.n_bins == 2**dims
    np.testing.assert_allclose(coarse.bin_sizes, 2**dims)
    np.testing.assert_allclose(
        coarse.diffuse(np.ones(coarse.n_bins), bandwidth=1, mode="average"), 1
    )


def test_grid_spec_owns_inputs_and_survives_copy_restore():
    from neurospatial.binning import GridBinningSpec

    origin, forward = np.zeros(2), np.eye(3)
    spec = GridBinningSpec(origin, forward, True)
    origin[:] = 100
    forward[:] = 0
    for owned in (spec, copy.deepcopy(spec), pickle.loads(pickle.dumps(spec))):
        np.testing.assert_array_equal(owned.origin, [0, 0])
        np.testing.assert_array_equal(owned.forward, np.eye(3))
        assert not owned.origin.flags.writeable and not owned.forward.flags.writeable


def test_payload_without_spec_does_not_infer_one_from_provenance():
    data = grid().to_dict()
    data["binning"]["grid_spec"] = None
    restored = Environment.from_dict(data)
    assert restored._binning.grid_spec is None
    with pytest.raises(NotImplementedError, match="do not record"):
        restored.rebin(bin_size=2)


def test_polar_unit_radius_cannot_be_misclassified_as_rectangular():
    polar = Environment.from_polar_egocentric(
        distance_range=(0.5, 1.5),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=1.0,
        angle_bin_size=np.pi / 2,
    )
    assert polar._binning.grid_spec is None


def test_grid_spec_is_available_after_nwb_reader_closes(tmp_path):
    from datetime import datetime, timezone

    pynwb = pytest.importorskip("pynwb")
    from neurospatial.io.nwb import read_environment, write_environment

    nwb = pynwb.NWBFile(
        session_description="rebin",
        identifier="grid-spec",
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    write_environment(nwb, grid())
    path = tmp_path / "grid.nwb"
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    with pynwb.NWBHDF5IO(str(path), "r") as io:
        restored = read_environment(io.read())
    np.testing.assert_array_equal(
        restored.rebin(bin_size=2).bin_centers,
        [[x, y] for x in [1, 3, 5] for y in [1, 3, 5]],
    )


@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("family", ["mask", "regular"])
def test_thin_grid_preserves_construction_choice_before_diagonals_exist(
    diagonal, family
):
    if family == "mask":
        env = Environment.from_grid_mask(
            np.ones((1, 2), dtype=bool),
            grid_edges=(np.arange(2.0), np.arange(3.0)),
            connect_diagonal_neighbors=diagonal,
        )
    else:
        env = Environment.from_layout(
            create_layout(
                "RegularGrid",
                bin_size=1.0,
                dimension_ranges=[(0, 1), (0, 2)],
                connect_diagonal_neighbors=diagonal,
            )
        )
    assert env._binning.grid_spec.connect_diagonal is diagonal
    # Make the regular-family result exercise legacy identity coarsening too.
    for variant in (env, env.rebin(bin_size=1).rebin(1)):
        refined = variant.rebin(bin_size=0.5)
        actual = any(
            np.count_nonzero(refined.bin_centers[u] != refined.bin_centers[v]) > 1
            for u, v in refined.edge_pairs
        )
        assert actual is diagonal


def test_factor_rebin_without_spec_records_the_choice_it_built():
    """A coarse grid records the diagonal choice used to connect it.

    The coarse grid here is one cell wide, so it has no diagonal pair to show
    the choice; later new-resolution re-binning must still add diagonals.
    """
    env = Environment.from_grid_mask(
        np.ones((4, 2), dtype=bool), grid_edges=(np.arange(5.0), np.arange(3.0))
    )
    binning = replace(
        env.binning, faces=KnownFaces(env.binning.faces.build()), grid_spec=None
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        coarse = Environment.from_bins(binning).rebin((2, 2))
    assert coarse.grid_shape == (2, 1)
    assert coarse.layout_parameters["connect_diagonal_neighbors"] is True
    assert coarse._binning.grid_spec.connect_diagonal is True
    np.testing.assert_array_equal(coarse._binning.grid_spec.origin, [0.0, 0.0])
    refined = coarse.rebin(bin_size=1.0)
    assert any(
        np.count_nonzero(refined.bin_centers[u] != refined.bin_centers[v]) > 1
        for u, v in refined.edge_pairs
    )


@pytest.mark.parametrize(
    ("options", "n_bins"),
    [
        ({"factor": 1}, 18),
        ({"factor": 2}, 6),  # coarse columns 0-1 and 2-3 hold retained bins
        ({"bin_size": 2.0}, 6),
        ({"bin_size": 0.5}, 72),
    ],
)
def test_rebinning_a_subset_never_restores_dropped_space(options, n_bins):
    full = grid()
    west = full.subset(mask=full.bin_centers[:, 0] < 3)  # 3 of 6 columns
    coarse = west.rebin(**options)
    assert coarse.n_bins == n_bins
    # Every new bin holds a retained bin, or lies within one.
    holds = set(coarse.bin_at(west.bin_centers).tolist())
    within = west.bin_at(coarse.bin_centers) >= 0
    assert all(index in holds or within[index] for index in range(coarse.n_bins))


def test_rebinning_after_a_rotation_keeps_the_same_cells():
    # A bin centered on new cell corners overlaps all four cells; rounding from
    # the rotation must not move which cells it claims.
    full = Environment.from_grid_mask(
        np.ones((3, 3), dtype=bool), grid_edges=(np.arange(4.0),) * 2
    )
    one = full.subset(mask=np.all(full.bin_centers == [1.5, 1.5], axis=1))
    expected = one.rebin(bin_size=1.5).bin_centers
    np.testing.assert_array_equal(
        expected, [[0.75, 0.75], [0.75, 2.25], [2.25, 0.75], [2.25, 2.25]]
    )
    for angle in np.linspace(0.01, 6.2, 50):
        rotation = np.eye(3)
        rotation[:2, :2] = [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
        rotated = one.apply_transform(Affine2D(rotation)).rebin(bin_size=1.5)
        back = rotated.bin_centers @ rotation[:2, :2]  # inverse of a rotation
        np.testing.assert_allclose(
            np.sort(back, axis=0), np.sort(expected, axis=0), atol=1e-9
        )


@pytest.mark.parametrize("n_dims", [1, 3, 4])
def test_rebinning_a_reflected_grid_matches_rebinning_before_the_reflection(n_dims):
    # Reflection leaves 1-D and 4-D grids without rectangular or framed
    # cells; their boxes come from the transformed lookup instead.
    shape = (6,) if n_dims == 1 else (2,) * n_dims
    edges = tuple(np.arange(size + 1.0) for size in shape)
    env = Environment.from_grid_mask(np.ones(shape, dtype=bool), grid_edges=edges)
    reflection = np.eye(n_dims + 1)
    reflection[0, 0], reflection[0, -1] = -1.0, float(shape[0])
    reflected = env.apply_transform(AffineND(reflection))
    expected = env.rebin(bin_size=2).bin_centers
    actual = reflected.rebin(bin_size=2).bin_centers
    mirrored = actual.copy()
    mirrored[:, 0] = shape[0] - mirrored[:, 0]
    np.testing.assert_allclose(
        mirrored[np.lexsort(mirrored.T[::-1])], expected[np.lexsort(expected.T[::-1])]
    )
