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


def test_neighbors_have_six_for_interior_cells():
    """At least one interior hexagon has exactly six neighbors."""
    layout = _build_hex_layout()
    degrees = _degrees(layout)

    interior = np.flatnonzero(degrees == 6)
    assert interior.size > 0
    assert degrees[interior[0]] == 6


def test_neighbors_have_fewer_at_boundary():
    """Boundary hexagons have fewer than six neighbors."""
    layout = _build_hex_layout()
    degrees = _degrees(layout)

    boundary = np.flatnonzero(degrees < 6)
    assert boundary.size > 0
    assert degrees[boundary[0]] < 6


def test_point_to_bin_round_trip():
    """Querying each bin center returns that bin's own index."""
    layout = _build_hex_layout()
    n = layout.bin_centers.shape[0]

    result = layout.point_to_bin_index(layout.bin_centers)
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
