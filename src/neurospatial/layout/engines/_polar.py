"""Internal masked polar grid with physical sector geometry and connectivity."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from neurospatial.geometry import Binning
from neurospatial.layout._assembly import assemble_binning
from neurospatial.layout._labels import LayoutType
from neurospatial.layout.base import capture_build_params
from neurospatial.layout.engines.masked_grid import MaskedGridLayout
from neurospatial.layout.helpers.neighbor_pairs import order_neighbor_pairs


class _PolarGridLayout(MaskedGridLayout):
    """Masked grid over ``(distance, angle)`` whose cells are annular sectors.

    Storage, indexing and neighbor pairs are the masked grid's, with the
    polar pair lengths and seam pairs added by this layout. Cell areas and
    shared faces are the grid's ``Δr · Δθ`` measures scaled by radius, so
    diffusion and finite-volume operators read physical sector geometry from
    the layout.

    Attributes
    ----------
    circular_angle : bool
        Whether the first and last angular bins of each ring are joined,
        set by ``build()``.
    """

    layout_type: ClassVar[str] = LayoutType.POLAR.value
    circular_angle: bool

    def __init__(self) -> None:
        """Initialize an unbuilt polar grid."""
        super().__init__()
        self.circular_angle = False

    @capture_build_params(data=("active_mask", "grid_edges"))
    def build(
        self,
        *,
        active_mask: NDArray[np.bool_],
        grid_edges: tuple[NDArray[np.float64], ...],
        connect_diagonal_neighbors: bool = True,
        circular_angle: bool = True,
    ) -> None:
        """Build radial, angular and diagonal edges with physical polar lengths.

        Parameters
        ----------
        active_mask : ndarray, shape (n_distance, n_angle)
            Active annular sectors, in radial then angular order.
        grid_edges : tuple of ndarray
            Uniform radial and angular bin edges, in physical units and radians.
        connect_diagonal_neighbors : bool, default=True
            Include diagonal neighbors, including across the angular seam.
        circular_angle : bool, default=True
            Join the first and last angular bins of each ring.

        Raises
        ------
        ValueError
            If the grid is not two-dimensional or the masked grid is invalid.
        """
        if np.ndim(active_mask) != 2 or len(grid_edges) != 2:
            raise ValueError(
                "Polar layouts require a 2-D mask and two grid edge arrays."
            )
        super().build(
            active_mask=active_mask,
            grid_edges=grid_edges,
            connect_diagonal_neighbors=connect_diagonal_neighbors,
        )
        self.circular_angle = bool(circular_angle)
        grid_pairs, _ = self._neighbor_pairs()
        first, second = grid_pairs[:, 0], grid_pairs[:, 1]
        distances = _polar_pair_lengths(self.bin_centers, grid_pairs)
        n_distance, n_angle = active_mask.shape
        if circular_angle and n_angle > 1:
            assert self.grid_edges is not None
            distance_edges, angle_edges = self.grid_edges
            distance_centers = 0.5 * (distance_edges[:-1] + distance_edges[1:])
            seams, seam_lengths = _circular_seam_pairs(
                self.locator().flat_to_active,
                n_distance,
                n_angle,
                distance_centers,
                float(np.diff(angle_edges).mean()),
                connect_diagonal_neighbors=connect_diagonal_neighbors,
            )
            # A seam pair the grid already connects (two angular bins) keeps
            # its grid length.
            pairs, distances = order_neighbor_pairs(
                np.concatenate((first, seams[:, 0])),
                np.concatenate((second, seams[:, 1])),
                np.concatenate((distances, seam_lengths)),
            )
        else:
            pairs = grid_pairs
        self._set_neighbor_pairs(pairs, distances)

    def bin_sizes(self) -> NDArray[np.float64]:
        """Annular-sector area ``r̄ · Δr · Δθ`` of each active bin.

        The masked grid reports ``Δr · Δθ``, treating the angular extent
        (radians) as a length. The sector spanning ``[r0, r1]`` has area
        ``0.5 · (r1² - r0²) · Δθ``, which equals ``r̄ · Δr · Δθ`` with ``r̄`` the
        bin's center radius.

        Returns
        -------
        ndarray of shape (n_bins,)
            Annular-sector area of each active bin, in (length unit)².
        """
        return super().bin_sizes() * self.bin_centers[:, 0]

    def binning(self) -> Binning:
        """Return owned sectors, lookup and lazy polar faces.

        Faces scale radial interfaces by their arc radius and join the angular
        seam when the grid is circular. Face directions and a Cartesian
        re-binning specification do not apply to ``(distance, angle)`` cells.

        Raises
        ------
        LayoutNotBuiltError
            If the layout has not been built.
        """
        pairs, distances = self._neighbor_pairs()
        assert self.grid_shape is not None
        return assemble_binning(
            pairs,
            distances,
            kind=self.layout_type,
            centers=self.bin_centers,
            sizes=self.bin_sizes(),
            measure_dim=self.bin_centers.shape[1],
            locator=self.locator(),
            faces="polar",
            face_arrays=(self._axis_widths(),),
            n_angle=self.grid_shape[1],
            circular=self.circular_angle,
            face_normals=False,
            rectilinear=self.rectilinear_grid(),
            dimension_ranges=self.dimension_ranges,
        )


def _circular_seam_pairs(
    node_of_flat: NDArray[np.intp],
    n_distance: int,
    n_angle: int,
    distance_centers: NDArray[np.float64],
    angle_step: float,
    connect_diagonal_neighbors: bool = False,
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Pair the first and last angular bins across the angular seam.

    Each distance ring proposes a pair between angle index 0 and angle index
    ``n_angle - 1``. Bins are numbered in active-bin order; ``node_of_flat``
    maps full-grid indices to bins, and seams touching an inactive sector are
    skipped.

    The seam pair's length is the arc length of a single angular step at that
    ring's radius (``distance_center * angle_step``), not the Euclidean chord
    between the first/last angle-bin centers (which span the full angular
    range). ``angle_step`` must be the *actual* realized angular bin step
    (``np.diff`` of the realized angle edges), not the requested
    ``angle_bin_size``, so the seam matches the regular angular pairs (which
    also use the realized step) when ``angle_bin_size`` does not evenly divide
    the angular range.

    When ``connect_diagonal_neighbors`` is True, the *diagonal* seam pairs
    between adjacent distance rings follow, ring by ring: ``(r_i,
    last_angle)`` with ``(r_{i+1}, first_angle)``, then ``(r_i, first_angle)``
    with ``(r_{i+1}, last_angle)``. Without these the seam would have fewer
    diagonal connections than every interior angular boundary, making the ±π
    topology anisotropic. Each seam diagonal's length is the physical polar
    diagonal ``sqrt(Δr² + (r̄ · Δθ_actual)²)`` with ``r̄`` the mean radius of
    the two rings, matching :func:`_polar_pair_lengths`.

    Parameters
    ----------
    node_of_flat : ndarray of shape (n_distance * n_angle,)
        Active-bin ID of each full-grid flat index, or -1 if inactive.
    n_distance : int
        Number of distance bins.
    n_angle : int
        Number of angle bins.
    distance_centers : ndarray of shape (n_distance,)
        Center radius of each distance ring, in physical units.
    angle_step : float
        Actual (realized) angular step between adjacent angle bins, in
        radians.
    connect_diagonal_neighbors : bool, default=False
        If True, also pair diagonal neighbors across the seam between adjacent
        distance rings, mirroring the interior diagonal connectivity so the
        seam is isotropic.

    Returns
    -------
    pairs : ndarray of intp, shape (n_seams, 2)
        Seam pairs ``(lower, higher)`` in proposal order.
    lengths : ndarray of float64, shape (n_seams,)
        Physical length of each seam pair.
    """
    rings = np.arange(n_distance)
    first = rings * n_angle
    last = first + (n_angle - 1)
    flat = [np.column_stack((first, last))]
    lengths = [distance_centers * angle_step]
    if connect_diagonal_neighbors and n_distance > 1:
        inner, outer = distance_centers[:-1], distance_centers[1:]
        diagonal = np.hypot(outer - inner, 0.5 * (inner + outer) * angle_step)
        # Per ring: inner last with outer first, then inner first with outer last.
        flat.append(
            np.column_stack((last[:-1], first[1:], first[:-1], last[1:])).reshape(-1, 2)
        )
        lengths.append(np.repeat(diagonal, 2))
    pairs = node_of_flat[np.concatenate(flat)]
    seam_lengths = np.concatenate(lengths)
    active = np.all(pairs >= 0, axis=1)
    return pairs[active], seam_lengths[active]


def _polar_pair_lengths(
    centers: NDArray[np.float64], pairs: NDArray[np.intp]
) -> NDArray[np.float64]:
    """Physical polar length of each grid pair of ``(distance, angle)`` bins.

    The Euclidean norm of a displacement in ``(distance, angle)`` coordinates
    treats an angular step ``Δθ`` (radians) as if it were a length and so
    collapses cm and radians into one scalar. The physical polar length is:

    - pure angular step (same distance ring): arc length ``r̄ · Δθ`` where
      ``r̄`` is the mean of the two endpoints' radii;
    - pure radial step (same angle): ``|Δr|``;
    - diagonal step: ``sqrt(Δr² + (r̄ · Δθ)²)``.

    It applies to the grid pairs (single-step radial/angular/diagonal moves),
    not to the seam pairs, whose angular step crosses the ±π seam.

    Parameters
    ----------
    centers : ndarray of float64, shape (n_bins, 2)
        Bin centers as ``(distance, angle)``.
    pairs : ndarray of int, shape (n_pairs, 2)
        Grid neighbor pairs.

    Returns
    -------
    ndarray of float64, shape (n_pairs,)
        Physical length of each pair.
    """
    start, end = centers[pairs[:, 0]], centers[pairs[:, 1]]
    r_mean = 0.5 * (start[:, 0] + end[:, 0])
    return np.hypot(end[:, 0] - start[:, 0], r_mean * (end[:, 1] - start[:, 1]))
