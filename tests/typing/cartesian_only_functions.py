"""Static contracts: Cartesian-only domain functions reject polar environments.

These functions read bin centers as Cartesian ``(x, y)`` coordinates, so a
polar environment must be a type error. Functions that refuse only one mode
keep accepting polar environments for their graph-based modes.

Run with ``uv run mypy tests/typing/cartesian_only_functions.py``.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from neurospatial import Environment
    from neurospatial.behavior import (
        detect_laps,
        distance_to_region,
        graph_turn_sequence,
        path_progress,
        trajectory_similarity,
    )
    from neurospatial.decoding import decode_position_summary, mean_position
    from neurospatial.encoding import (
        compute_field_emd,
        field_shape_metrics,
        field_shift_distance,
        rate_map_centroid,
        spatial_autocorrelation,
    )
    from neurospatial.geometry._coordinates import (
        CartesianSystem,
        EgocentricPolarSystem,
    )
    from neurospatial.ops import (
        map_probabilities,
        resample_field,
        select_basis_centers,
        spatial_basis,
    )

    def _cartesian_only(
        cartesian: Environment[CartesianSystem],
        polar: Environment[EgocentricPolarSystem],
        values: NDArray[np.float64],
        bins: NDArray[np.int64],
    ) -> None:
        field_shape_metrics(cartesian, values, bins)
        field_shape_metrics(polar, values, bins)  # type: ignore[arg-type]
        field_shift_distance(cartesian, values, bins, cartesian, values, bins)
        field_shift_distance(polar, values, bins, cartesian, values, bins)  # type: ignore[arg-type]
        field_shift_distance(cartesian, values, bins, polar, values, bins)  # type: ignore[arg-type]
        spatial_autocorrelation(cartesian, values)
        spatial_autocorrelation(polar, values)  # type: ignore[arg-type]
        resample_field(values, cartesian, cartesian)
        resample_field(values, polar, cartesian)  # type: ignore[arg-type]
        resample_field(values, cartesian, polar)  # type: ignore[arg-type]
        map_probabilities(cartesian, cartesian, values)
        map_probabilities(polar, cartesian, values)  # type: ignore[arg-type]
        map_probabilities(cartesian, polar, values)  # type: ignore[arg-type]
        spatial_basis(cartesian)
        spatial_basis(polar)  # type: ignore[arg-type]
        graph_turn_sequence(cartesian, bins)
        graph_turn_sequence(polar, bins)  # type: ignore[arg-type]
        detect_laps(bins, values, cartesian)
        detect_laps(bins, values, polar)  # type: ignore[arg-type]

    def _graph_modes_accept_polar(
        polar: Environment[EgocentricPolarSystem],
        values: NDArray[np.float64],
        bins: NDArray[np.int64],
    ) -> None:
        # Only the Euclidean or coordinate-based modes refuse polar at runtime.
        rate_map_centroid(polar, values, bins, method="geodesic")
        compute_field_emd(polar, values, values, metric="geodesic")
        select_basis_centers(polar, 5, method="farthest_point")
        path_progress(bins, polar, start_bins=bins, goal_bins=bins)
        distance_to_region(bins, polar, target_bins=bins)
        trajectory_similarity(bins, bins, polar, method="jaccard")
        # Posterior means are egocentric-plane centroids on polar environments.
        mean_position(polar, values)
        decode_position_summary(polar, bins, values, 0.02)
