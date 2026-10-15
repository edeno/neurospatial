"""Capture existing composition and overlap cases before Stage C changes.

Run with ``uv run --no-sync python
docs/plans/environment-combine/probe_composition.py``. Observations describe
the baseline, including defects; they are not assertions of desired behavior.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np

from neurospatial import CompositeEnvironment, Environment
from neurospatial.arena import TrackArena
from neurospatial.binning import NearestBinLocator, SubsetLocator


def attempt(operation: Callable[[], Any]) -> dict[str, Any]:
    """Record a value or a refusal without hiding unrelated failures."""
    try:
        return {"value": operation()}
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        return {"error": type(error).__name__, "message": str(error)}


def grid(offset: float, *, n_dims: int = 2, units: str = "cm") -> Environment:
    """Construct a single exact unit cell with known coordinate labels."""
    return Environment.from_grid_mask(
        np.ones((1,) * n_dims, dtype=bool),
        grid_edges=tuple(np.array([offset, offset + 1.0]) for _ in range(n_dims)),
        units=units,
        frame="lab",
    )


def record() -> dict[str, Any]:
    """Record deterministic legacy behavior and broader support witnesses."""
    left, right = grid(0), grid(3)
    joined = CompositeEnvironment([left, right])
    separated = CompositeEnvironment([left, right], auto_bridge=False)
    overlap = CompositeEnvironment([left, grid(0.5)], auto_bridge=False)
    nearest_left = NearestBinLocator(np.array([[0.0, 0.0]]))
    nearest_right = NearestBinLocator(np.array([[10.0, 0.0]]))
    bounded_left = NearestBinLocator(np.array([[0.0, 0.0]]), max_distance=1.0)
    bounded_right = NearestBinLocator(np.array([[1.5, 0.0]]), max_distance=1.0)
    crossing_a = TrackArena(np.array([[[-1.0, 0.0], [1.0, 0.0]]]), width=0.5)
    crossing_b = TrackArena(np.array([[[0.0, -1.0], [0.0, 1.0]]]), width=0.5)
    parent_nearest = NearestBinLocator(np.array([[0.0, 0.0], [10.0, 0.0]]))
    left_subset = SubsetLocator(parent_nearest, np.array([0, -1], dtype=np.intp))
    right_subset = SubsetLocator(parent_nearest, np.array([-1, 0], dtype=np.intp))
    return {
        "automatic_bridge_count": joined.connectivity.number_of_edges(),
        "automatic_bridge_distance": joined.connectivity.edges[0, 1]["distance"],
        "disconnected_edge_count": separated.connectivity.number_of_edges(),
        "lookup_inside_gap_outside": joined.bin_at(
            np.array([[0.5, 0.5], [2.0, 2.0], [3.5, 3.5], [20.0, 20.0]])
        ).tolist(),
        "overlap_first_component_wins": overlap.bin_at(
            np.array([[0.75, 0.75]])
        ).tolist(),
        "overlapping_3d_bins_accepted": CompositeEnvironment(
            [grid(0, n_dims=3), grid(0.5, n_dims=3)], auto_bridge=False
        ).n_bins,
        "mismatched_units": attempt(
            lambda: CompositeEnvironment([left, grid(3, units="m")]).n_bins
        ),
        "edge_table": attempt(lambda: joined.edge_attributes().columns.tolist()),
        "ordinary_file_api": hasattr(joined, "to_file"),
        "unbounded_nearest_both_claim_midpoint": [
            locator.locate(np.array([[5.0, 0.0]])).tolist()
            for locator in (nearest_left, nearest_right)
        ],
        "bounded_nearest_both_claim_midpoint": [
            locator.locate(np.array([[0.75, 0.0]])).tolist()
            for locator in (bounded_left, bounded_right)
        ],
        "crossing_track_footprints_both_contain_origin": [
            arena.contains(np.array([[0.0, 0.0]])).tolist()
            for arena in (crossing_a, crossing_b)
        ],
        "complementary_unbounded_subsets": [
            locator.locate(np.array([[-100.0, 0.0], [0.0, 0.0], [100.0, 0.0]])).tolist()
            for locator in (left_subset, right_subset)
        ],
    }


if __name__ == "__main__":
    print(json.dumps(record(), indent=2, sort_keys=True))
