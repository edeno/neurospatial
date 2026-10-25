"""Behavioral tests for HexagonalLayout driven directly through the engine."""

import numpy as np

from neurospatial.layout.engines.hexagonal import HexagonalLayout


def _build_hex_layout(hexagon_width=1.0):
    """Build a fully-active hexagonal lattice over a 5x5 area."""
    layout = HexagonalLayout()
    layout.build(
        hexagon_width=hexagon_width,
        dimension_ranges=((0.0, 5.0), (0.0, 5.0)),
        infer_active_bins=False,
    )
    return layout


def _degrees(layout):
    n = layout.bin_centers.shape[0]
    degree_dict = dict(layout.connectivity.degree())
    return np.array([degree_dict[i] for i in range(n)])


def test_degree_histogram_of_a_four_by_four_lattice():
    """A 4x4 lattice has the neighbor counts of a hexagonal tiling.

    The rows sit at y = 0, 0.87, 1.73, 2.60 with the odd rows shifted half a
    cell right. Counting each cell's neighbors by hand: the two acute corners
    have 2, the two obtuse corners and the ends of the inner rows 3, the
    inner cells of the outer rows 4, the inner rows' other ends 5, and the
    four interior cells 6.
    """
    layout = HexagonalLayout()
    layout.build(
        hexagon_width=1.0,
        dimension_ranges=((0.0, 3.0), (0.0, 2.0)),
        infer_active_bins=False,
    )
    assert layout.bin_centers.shape == (16, 2)
    degrees = _degrees(layout)
    np.testing.assert_array_equal(np.bincount(degrees), [0, 0, 2, 4, 4, 2, 4])
    # Neighbors are exactly the cells one hexagon width away.
    gaps = np.linalg.norm(
        layout.bin_centers[:, None] - layout.bin_centers[None], axis=2
    )
    np.testing.assert_array_equal(degrees, np.sum(np.isclose(gaps, 1.0), axis=1))


def test_point_to_bin_round_trip():
    """Querying each bin center returns that bin's own index."""
    layout = _build_hex_layout()
    n = layout.bin_centers.shape[0]

    result = layout.binning().locator.locate(layout.bin_centers)
    np.testing.assert_array_equal(result, np.arange(n))


def test_hexagonal_spacing():
    """Distance between adjacent bin centers equals the hexagon width."""
    hexagon_width = 1.0
    layout = _build_hex_layout(hexagon_width=hexagon_width)

    edge_distances = np.array(
        [data["distance"] for _, _, data in layout.connectivity.edges(data=True)]
    )
    assert edge_distances.size > 0
    np.testing.assert_allclose(edge_distances, hexagon_width, atol=1e-9)
