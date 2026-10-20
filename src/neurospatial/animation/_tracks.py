"""Shared track coordinates and field projection for animation renderers."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._stacklevel import _external_stacklevel
from neurospatial._units import normalize_position_units
from neurospatial.geometry._coordinates import require_cartesian

if TYPE_CHECKING:
    from neurospatial.environment import Environment


def _typical_bin_spacing(env: Environment) -> float:
    """Typical neighboring-center distance for a useful projection warning."""
    centers = env.bin_centers
    if len(centers) < 2:
        return 1.0
    sample = centers[:: max(1, len(centers) // 1000)]
    distances, _ = env.bin_center_tree().query(sample, k=2)
    spacing = float(np.median(distances[:, 1]))
    return spacing if spacing > 0 else 1.0


def track_extents(
    env: Environment,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Where each bin starts and ends along a track; None off a track.

    Linearized tracks (``linear_track``, ``from_graph``) measure distance
    along the track, with the spacing between segments left empty; 1-D
    environments use their own coordinate.
    """
    if env.is_linearized_track:
        linearization = env._binning.linearization
        assert linearization is not None
        edges = np.asarray(linearization.grid_edges[0], dtype=np.float64)
        active = linearization.active_mask
        return edges[:-1][active], edges[1:][active]
    if env.n_dims == 1:
        centers, half = env.bin_centers[:, 0], env.bin_sizes / 2
        return centers - half, centers + half
    return None


def track_segments(env: Environment) -> list[tuple[str, float, float]]:
    """Each segment of a linearized track: its name, start and stop along it.

    A segment is named by its nodes, such as ``"center–left"``, and spans
    its bins. Empty off a linearized track.
    """
    from neurospatial.geometry._graph import _track_cell_segments

    linearization = env._binning.linearization
    cells = None if linearization is None else _track_cell_segments(linearization)
    if linearization is None or cells is None:
        return []
    edges = np.asarray(linearization.grid_edges[0], dtype=np.float64)
    segments = []
    for u, v in linearization.edge_order:
        own = np.flatnonzero(cells == linearization.graph.edges[u, v]["edge_id"])
        segments.append(
            (f"{u}\u2013{v}", float(edges[own[0]]), float(edges[own[-1] + 1]))
        )
    return segments


def track_position(
    env: Environment, positions: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Positions as distances along the track, shape (n,); NaN stays NaN."""
    if env.is_linearized_track:
        return np.asarray(env.to_linear(positions), dtype=np.float64)
    return np.asarray(positions, dtype=np.float64)[:, 0]


def _checked_track(env: Environment, track: Environment) -> Environment:
    """``track``, if a 2-D environment's fields can be drawn along it."""
    if not track.is_linearized_track:
        raise ValueError(
            "track must be a linearized track (Environment.from_graph or "
            "linear_track): its bins give position along the maze. Why: the "
            "2-D fields are drawn against position along it."
        )
    if env.is_linearized_track or env.n_dims != 2 or track.n_dims != 2:
        raise ValueError(
            "track= draws a 2-D environment's fields along a track of the same "
            f"maze; the environment has {env.n_dims} dimensions"
            + (" and is a track itself" if env.is_linearized_track else "")
            + ". Fix: animate a 2-D environment, or leave track out."
        )
    for coordinates in (env.coordinate_system, track.coordinate_system):
        require_cartesian(
            coordinates,
            "animate_fields(track=...)",
            why="each bin's probability goes to the track bin nearest its center, "
            "a distance in shared Cartesian coordinates",
            fix="use an allocentric Cartesian environment and track",
        )
    for name in ("units", "frame"):
        expected, declared = getattr(env, name), getattr(track, name)
        if name == "units":
            expected = None if expected is None else normalize_position_units(expected)
            declared = None if declared is None else normalize_position_units(declared)
        # Two unspecified declarations preserve the legacy caller-established
        # correspondence; one-sided metadata cannot establish correspondence.
        if expected != declared:
            raise ValueError(
                f"track {name} {declared!r} does not match environment {name} "
                f"{expected!r}. Why: nearest-bin projection requires shared "
                "Cartesian coordinates. Fix: explicitly convert both environments "
                "to the same units and frame and declare matching metadata; "
                "coordinates are never silently rescaled or transformed."
            )
    return track


def _projection(env: Environment, track: Environment) -> Any:
    """Sparse matrix, shape (env.n_bins, track.n_bins), sending each of
    ``env``'s bins to the track bin nearest its center.

    Warns when many track bins receive no 2-D bin, as when the track's bins
    are finer than the 2-D ones: those bins are drawn as missing.
    """
    from scipy.sparse import csr_matrix

    _, nearest = track.bin_center_tree().query(env.bin_centers)
    empty = 1.0 - len(np.unique(nearest)) / track.n_bins
    if empty > 0.1:
        warnings.warn(
            f"{empty:.0%} of the track's bins are nearest to no 2-D bin, so "
            "they are drawn as missing along the track. Why: each 2-D bin's "
            "probability goes to one track bin. Fix: build the track with a "
            f"bin_size of at least the 2-D bins' ({_typical_bin_spacing(env):.3g}).",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )
    ones = np.ones(env.n_bins)
    return csr_matrix(
        (ones, (np.arange(env.n_bins), nearest)), shape=(env.n_bins, track.n_bins)
    )


def _project(rows: NDArray[np.float64], projection: Any) -> NDArray[np.float64]:
    """Each row summed into the track bins; NaN where no finite value maps."""
    finite = np.isfinite(rows)
    total = np.asarray(np.where(finite, rows, 0.0) @ projection)
    seen = np.asarray(finite.astype(np.float64) @ projection)
    result: NDArray[np.float64] = np.where(seen > 0, total, np.nan)
    return result
