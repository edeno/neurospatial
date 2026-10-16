"""Arrays an environment hands out are read-only, so an accidental write raises."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from neurospatial import Environment


def _grid():
    return Environment.from_grid_mask(
        np.ones((3, 4), dtype=bool), grid_edges=(np.arange(4.0), np.arange(5.0))
    )


def _track():
    return Environment.linear_track(endpoints=[(0.0, 0.0), (8.0, 0.0)], bin_size=2.0)


def _hexagonal():
    positions = np.random.default_rng(0).uniform(0, 12, (500, 2))
    return Environment.from_samples(positions, bin_size=3.0, layout="Hexagonal")


def _handed_out(env):
    """Every array the environment returns from its geometry and cached analyses."""
    arrays = {
        "bin_centers": env.bin_centers,
        "active_mask": env.active_mask,
        "edge_pairs": env.edge_pairs,
        "bin_sizes": env.bin_sizes,
        **{f"grid_edges[{i}]": edge for i, edge in enumerate(env.grid_edges)},
    }
    binning = env.binning
    arrays.update(
        {
            "binning.bins.centers": binning.bins.centers,
            "binning.bins.sizes": binning.bins.sizes,
            "binning.bins.pairs": binning.bins.pairs,
            "binning.bins.pair_distances": binning.bins.pair_distances,
        }
    )
    if binning.rectilinear is not None:
        arrays["binning.rectilinear.active_mask"] = binning.rectilinear.active_mask
        for i, edge in enumerate(binning.rectilinear.edges):
            arrays[f"binning.rectilinear.edges[{i}]"] = edge
    if binning.linearization is not None:
        arrays["binning.linearization.active_mask"] = binning.linearization.active_mask
    faces = binning.faces.build()
    arrays.update({f"faces.{name}": value for name, value in faces._asdict().items()})
    geometry = env.get_diffusion_geometry()
    arrays.update(
        {
            "diffusion.W.data": geometry.W.data,
            "diffusion.volumes": geometry.volumes,
            "diffusion.labels": geometry.labels,
            "gaussian_kernel": env.compute_gaussian_kernel(1.0),
        }
    )
    return arrays


@pytest.mark.parametrize("build", [_grid, _track, _hexagonal])
@pytest.mark.parametrize("restore", ["fresh", "pickle"])
def test_an_accidental_write_to_any_handed_out_array_raises(build, restore):
    env = build()
    if restore == "pickle":
        env = pickle.loads(pickle.dumps(env))
    for name, array in _handed_out(env).items():
        assert not array.flags.writeable, name
        with pytest.raises(ValueError, match="read-only"):
            array[...] = 0
    centers = env.bin_centers.copy()
    np.testing.assert_array_equal(env.bin_centers, centers)


def test_graph_attributes_refuse_edits():
    env = _grid()
    with pytest.raises(TypeError, match="read-only"):
        env.connectivity.nodes[0]["pos"] = (9.0, 9.0)
    edge = next(iter(env.connectivity.edges))
    with pytest.raises(TypeError, match="read-only"):
        env.connectivity.edges[edge]["distance"] = 9.0


def test_bin_centers_are_views_of_the_one_published_copy():
    """No duplicate centers: each access is a new header over the binning's array."""
    env = _grid()
    first, second = env.bin_centers, env.bin_centers
    assert first is not second
    assert np.shares_memory(first, env._binning.bins.centers)
    first.shape = (first.size,)
    assert env.bin_centers.shape == (env.n_bins, 2)


@pytest.mark.parametrize(
    "name", ["bin_centers", "connectivity", "grid_edges", "name", "regions", "extra"]
)
def test_public_attributes_cannot_be_set_or_deleted(name):
    env = _grid()
    with pytest.raises(AttributeError, match=f"Environment.{name} is read-only"):
        setattr(env, name, None)
    with pytest.raises(AttributeError, match=f"Environment.{name} is read-only"):
        delattr(env, name)
