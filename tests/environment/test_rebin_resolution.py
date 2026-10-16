"""New-resolution re-binning uses the arena and owned construction data."""

import copy
import pickle
import re
import warnings
from dataclasses import replace

import numpy as np
import pytest
from shapely.geometry import LineString, box

from neurospatial import Environment
from neurospatial.arena import OpenArena, UnknownArena
from neurospatial.binning import KnownFaces


def grid():
    return Environment.from_grid_mask(
        np.ones((6, 6), dtype=bool),
        grid_edges=(np.arange(7.0),) * 2,
        units="cm",
        frame="lab",
    )


def test_rebin_new_resolution_restores_arena_after_analysis_crop():
    env = grid().subset(bins=grid().bin_centers[:, 0] < 2)
    coarse = env.rebin(bin_size=2.0)
    expected = np.array([[x, y] for x in [1, 3, 5] for y in [1, 3, 5]])
    np.testing.assert_array_equal(coarse.bin_centers, expected)
    np.testing.assert_allclose(coarse.bin_sizes, 4.0)
    assert coarse.units == "cm" and coarse.frame == "lab"
    np.testing.assert_array_equal(
        coarse.rebin(bin_size=1).bin_centers, grid().bin_centers
    )


def test_rebin_reports_original_area_left_without_a_new_bin():
    # 4-unit cells over a 6-unit arena: only the cell centered at (2, 2) lies
    # strictly inside, so 20 of the 36 original bins lose their bin.
    env = Environment.from_polygon(box(0, 0, 6, 6), 1.0)
    with pytest.warns(UserWarning, match="without a new bin") as record:
        coarse = env.rebin(bin_size=4.0)
    lost = coarse.bin_at(env.bin_centers) < 0
    assert np.count_nonzero(lost) == 20
    (message,) = [str(item.message) for item in record]
    assert re.search(r"20 of 36 original bins \(55\.6% of their area\)", message)
    assert "divisor" in message and "-1" in message


@pytest.mark.parametrize("size", [1.0, 2.0, 3.0, 0.5])
def test_rebin_covering_the_arena_does_not_warn(size):
    env = Environment.from_polygon(box(0, 0, 6, 6), 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        coarse = env.rebin(bin_size=size)
    assert np.all(coarse.bin_at(env.bin_centers) >= 0)
    assert coarse.bin_sizes.sum() == pytest.approx(36.0)


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
    parent = grid().apply_transform(transform)
    coarse = parent.rebin(bin_size=(4.0, 9.0))
    local = np.array([[x, y] for x in [1, 3, 5] for y in [1.5, 4.5]])
    np.testing.assert_allclose(
        coarse.bin_centers, local @ matrix[:2, :2].T + matrix[:2, 2]
    )
    np.testing.assert_allclose(coarse.bin_sizes, 36.0)
    restored = Environment.from_dict(coarse.to_dict()).rebin(bin_size=(4.0, 9.0))
    np.testing.assert_allclose(restored.bin_centers, coarse.bin_centers)


def test_clip_keeps_spec_but_does_not_turn_crop_into_new_arena():
    clipped = grid().clip(box(0.2, 0.2, 1.8, 1.8))
    assert clipped._binning.grid_spec is not None
    assert clipped.rebin(bin_size=2).n_bins == 9


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
    with pytest.raises(ValueError, match="wall cuts"):
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
    with pytest.raises(NotImplementedError, match="specification"):
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
        coarse.diffuse(np.ones(coarse.n_bins), 1, mode="average"), 1
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


def test_legacy_payload_does_not_infer_spec_from_provenance():
    data = grid().to_dict()
    data["binning"].pop("grid_spec")
    restored = Environment.from_dict(data)
    assert restored._binning.grid_spec is None
    with pytest.raises(NotImplementedError, match="specification"):
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
            "RegularGrid",
            {
                "bin_size": 1.0,
                "dimension_ranges": [(0, 1), (0, 2)],
                "connect_diagonal_neighbors": diagonal,
            },
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
