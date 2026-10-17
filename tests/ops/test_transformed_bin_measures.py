"""Affine transformations preserve physical cell measures, including track lengths."""

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.ops.transforms import AffineND, scale_2d


@pytest.mark.parametrize("n_dims", [1, 2, 3])
@pytest.mark.parametrize("n_cells", [1, 2])
@pytest.mark.parametrize(
    "kind", ["identity", "translation", "rotation", "scale", "shear"]
)
def test_transformed_rectangular_cells_keep_physical_measures(n_dims, n_cells, kind):
    """Lengths, areas and volumes obey rigid invariance and analytical scaling."""
    widths = np.arange(1.0, n_dims + 1) + 1
    env = Environment.from_grid_mask(
        np.ones((n_cells,) * n_dims, dtype=bool),
        grid_edges=tuple(np.arange(n_cells + 1) * width for width in widths),
    )
    matrix = np.eye(n_dims + 1)
    expected_scale = 1.0
    if kind == "translation":
        matrix[:n_dims, -1] = 10
    elif kind == "rotation":
        if n_dims == 1:
            matrix[0, 0] = -1
        else:
            matrix[:2, :2] = [[0, -1], [1, 0]]
    elif kind == "scale":
        factors = np.arange(2, n_dims + 2)
        matrix[:n_dims, :n_dims] = np.diag(factors)
        expected_scale = float(np.prod(factors))
    elif kind == "shear" and n_dims > 1:
        matrix[0, 1] = 2

    transformed = env.apply_transform(AffineND(matrix))
    expected = np.full(env.n_bins, np.prod(widths) * expected_scale)
    np.testing.assert_allclose(transformed.bin_sizes, expected)
    np.testing.assert_allclose(env.bin_sizes, np.prod(widths))
    np.testing.assert_allclose(transformed.with_metadata().bin_sizes, expected)


def test_transformed_bent_track_keeps_similarity_lengths_and_refuses_anisotropy():
    """Uniform scales preserve lengths; directional scales require rebuilding."""
    graph = nx.Graph()
    graph.add_nodes_from(
        [(0, {"pos": (0, 0)}), (1, {"pos": (6, 0)}), (2, {"pos": (6, 6)})]
    )
    graph.add_edges_from([(0, 1, {"distance": 6.0}), (1, 2, {"distance": 6.0})])
    env = Environment.from_graph(
        graph, edge_order=[(0, 1), (1, 2)], edge_spacing=0.0, bin_size=2.0
    )
    transformed = env.apply_transform(scale_2d(2))
    expected = env.bin_sizes * 2
    np.testing.assert_allclose(transformed.bin_sizes, expected)
    rotation = AffineND(np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]]))
    rotated = transformed.apply_transform(rotation)
    np.testing.assert_allclose(rotated.bin_sizes, expected)
    twice = rotated.apply_transform(scale_2d(3))
    np.testing.assert_allclose(twice.bin_sizes, env.bin_sizes * 6)
    for candidate in (env, rotated):
        with pytest.raises(ValueError) as caught:
            candidate.apply_transform(scale_2d(2, 3))
        assert "Environment.from_graph" in str(caught.value)
        assert "Why:" in str(caught.value)
        assert "Fix:" in str(caught.value)


def test_transformed_bin_measures_stay_read_only_after_copy_and_cache_clear():
    """A public measure array cannot overwrite a transformed layout's geometry."""
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool),
        grid_edges=(np.arange(3.0) * 2, np.arange(3.0)),
    ).apply_transform(scale_2d(2, 3))
    expected = np.full(env.n_bins, 12.0)
    for candidate in (env, env.with_metadata()):
        with pytest.raises(ValueError, match="read-only"):
            candidate.bin_sizes[:] = 0
        candidate.clear_cache()
        np.testing.assert_array_equal(candidate.bin_sizes, expected)
        assert not candidate.bin_sizes.flags.writeable
