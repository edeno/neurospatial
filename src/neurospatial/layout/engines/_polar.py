"""Internal masked polar grid with physical sector geometry and connectivity."""

from __future__ import annotations

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import LayoutNotBuiltError
from neurospatial.geometry._binning import Binning
from neurospatial.layout._assembly import assemble_binning
from neurospatial.layout.engines.masked_grid import MaskedGridLayout


class _PolarGridLayout(MaskedGridLayout):
    """Masked grid over ``(distance, angle)`` whose cells are annular sectors.

    Storage, indexing and connectivity are the masked grid's, carrying the
    polar edge lengths and seam edges built by this layout. Cell areas and
    shared faces are the grid's ``Δr · Δθ`` measures scaled by radius, so
    diffusion and finite-volume operators read physical sector geometry from
    the layout.

    Attributes
    ----------
    circular_angle : bool
        Whether the first and last angular bins of each ring are joined,
        set by ``build()``.
    """

    circular_angle: bool

    def __init__(self) -> None:
        """Initialize an unbuilt polar grid."""
        super().__init__()
        self._layout_type_tag = "Polar"
        self.circular_angle = False

    @property
    def layout_type(self) -> str:
        """Return the standardized category for egocentric polar sectors."""
        return "polar"

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
        # A rebuild changes active-bin numbering; both seams and point lookup
        # must derive their inverse mapping from the new mask.
        self._active_mask_inverse_map = None
        self.circular_angle = bool(circular_angle)
        # The masked-grid build captures the shared parameters; record the
        # polar-specific option alongside them as provenance.
        self._build_params_used["circular_angle"] = circular_angle
        assert self.connectivity is not None
        _fix_polar_edge_geometry(self.connectivity)
        n_distance, n_angle = active_mask.shape
        if circular_angle and n_angle > 1:
            assert self.grid_edges is not None
            distance_edges, angle_edges = self.grid_edges
            distance_centers = 0.5 * (distance_edges[:-1] + distance_edges[1:])
            assert self.active_mask_inverse_map is not None
            _add_circular_connectivity(
                self.connectivity,
                self.active_mask_inverse_map,
                n_distance,
                n_angle,
                distance_centers,
                float(np.diff(angle_edges).mean()),
                connect_diagonal_neighbors=connect_diagonal_neighbors,
            )

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
        if self.connectivity is None or self.grid_shape is None:
            raise LayoutNotBuiltError(type(self).__name__, "connectivity")
        return assemble_binning(
            self.connectivity,
            kind=self._layout_type_tag,
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
        )

    def finite_volume_geometry(self) -> tuple[nx.Graph, NDArray[np.float64]]:
        """Grid faces with each radial face lengthened to its arc ``r_face · Δθ``.

        On the ``(distance, angle)`` grid the masked grid already gives angular
        (and seam) neighbors the radial face ``Δr``, radial neighbors ``Δθ``, and
        diagonal neighbors no face (``A = 0``). Only a radial face's arc length
        depends on radius. Polar sectors are K-orthogonal, so this is exact.
        """
        from neurospatial.geometry._face_builders import polar_geometry

        assert self.connectivity is not None
        assert self.grid_shape is not None
        return polar_geometry(
            self.connectivity,
            self.bin_centers,
            self.bin_sizes(),
            self._axis_widths(),
            self.grid_shape[1],
            self.circular_angle,
        )


def _add_circular_connectivity(
    connectivity: nx.Graph,
    node_of_flat: NDArray[np.intp],
    n_distance: int,
    n_angle: int,
    distance_centers: NDArray[np.float64],
    angle_step: float,
    connect_diagonal_neighbors: bool = False,
) -> None:
    """Add circular wrap edges between first and last angle bins.

    Modifies ``connectivity`` in-place, adding an edge between angle index 0
    and angle index ``n_angle - 1`` for each distance ring. Nodes are indexed
    in active-bin order; ``node_of_flat`` maps full-grid indices to graph
    nodes, and seams touching an inactive sector are skipped.

    The wrap edge's ``distance`` attribute is the arc length of a single
    angular step at that ring's radius (``distance_center * angle_step``), not
    the Euclidean chord between the first/last angle-bin centers (which span
    the full angular range). ``angle_step`` must be the *actual* realized
    angular bin step (``np.diff`` of the realized angle edges), not the
    requested ``angle_bin_size``, so the seam edge matches the regular angular
    edges (which also use the realized step) when ``angle_bin_size`` does not
    evenly divide the angular range.

    When ``connect_diagonal_neighbors`` is True, also add the *diagonal* wrap
    edges that cross the ±π seam between adjacent distance rings: connecting
    ``(r_i, last_angle)`` to ``(r_{i+1}, first_angle)`` and ``(r_i,
    first_angle)`` to ``(r_{i+1}, last_angle)``. Without these the seam would
    have fewer diagonal connections than every interior angular boundary,
    making the ±π topology anisotropic. Interior diagonals (added by the
    masked-grid layout) and these seam diagonals together give every angular
    step the same diagonal connectivity. Each seam diagonal's ``distance`` is
    the physical polar diagonal length ``sqrt(Δr² + (r̄ · Δθ_actual)²)`` with
    ``r̄`` the mean radius of the two rings, matching
    :func:`_fix_polar_edge_geometry`.

    Parameters
    ----------
    connectivity : nx.Graph
        Connectivity graph to modify.
    node_of_flat : ndarray of shape (n_distance * n_angle,)
        Active-bin node of each full-grid flat index, or -1 if inactive.
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
        If True, also add diagonal wrap edges across the seam between adjacent
        distance rings, mirroring the interior diagonal connectivity so the
        seam is isotropic.
    """
    max_edge_id = max(
        (data.get("edge_id", -1) for _, _, data in connectivity.edges(data=True)),
        default=-1,
    )

    def _add_seam_edge(flat_a: int, flat_b: int, distance: float) -> None:
        nonlocal max_edge_id
        node_a, node_b = int(node_of_flat[flat_a]), int(node_of_flat[flat_b])
        if node_a < 0 or node_b < 0:
            return
        if connectivity.has_edge(node_a, node_b):
            return
        pos_a = connectivity.nodes[node_a]["pos"]
        pos_b = connectivity.nodes[node_b]["pos"]
        vector = np.array(pos_b) - np.array(pos_a)
        angle_2d = np.arctan2(vector[1], vector[0]) if len(vector) >= 2 else 0.0
        max_edge_id += 1
        connectivity.add_edge(
            node_a,
            node_b,
            distance=float(distance),
            vector=vector.tolist(),
            angle_2d=float(angle_2d),
            edge_id=int(max_edge_id),
        )

    # Same-ring seam edges (pure angular step across ±π).
    for d_idx in range(n_distance):
        first_angle_node = d_idx * n_angle + 0
        last_angle_node = d_idx * n_angle + (n_angle - 1)
        distance = float(distance_centers[d_idx]) * float(angle_step)
        _add_seam_edge(first_angle_node, last_angle_node, distance)

    # Diagonal seam edges (radial + angular step across ±π) between adjacent
    # rings, mirroring the interior diagonal connectivity so the seam matches
    # every other angular boundary.
    if connect_diagonal_neighbors:
        for d_idx in range(n_distance - 1):
            r_inner = float(distance_centers[d_idx])
            r_outer = float(distance_centers[d_idx + 1])
            d_r = r_outer - r_inner
            r_mean = 0.5 * (r_inner + r_outer)
            arc = r_mean * float(angle_step)
            diag = float(np.hypot(d_r, arc))

            inner_first = d_idx * n_angle + 0
            inner_last = d_idx * n_angle + (n_angle - 1)
            outer_first = (d_idx + 1) * n_angle + 0
            outer_last = (d_idx + 1) * n_angle + (n_angle - 1)

            # (r_i, last_angle) <-> (r_{i+1}, first_angle)
            _add_seam_edge(inner_last, outer_first, diag)
            # (r_i, first_angle) <-> (r_{i+1}, last_angle)
            _add_seam_edge(inner_first, outer_last, diag)


def _fix_polar_edge_geometry(connectivity: nx.Graph) -> None:
    """Rewrite all edge ``distance`` weights with physical polar lengths.

    The masked-grid layout computes each edge ``distance`` as the Euclidean
    norm of the displacement in ``(distance, angle)`` coordinates, which
    treats an angular step ``Δθ`` (radians) as if it were a length and so
    collapses cm and radians into one scalar. Replace each edge weight with
    the correct physical polar length:

    - pure angular step (same distance ring): arc length ``r̄ · Δθ`` where
      ``r̄`` is the mean of the two endpoints' radii;
    - pure radial step (same angle): ``|Δr|``;
    - diagonal step: ``sqrt(Δr² + (r̄ · Δθ)²)``.

    This is called BEFORE the circular wrap edges are added, so it only sees
    the grid edges (single-step radial/angular/diagonal moves). Node positions
    are read from the ``"pos"`` node attribute, where ``pos[0]`` is distance
    and ``pos[1]`` is angle.

    Parameters
    ----------
    connectivity : nx.Graph
        Connectivity graph to modify in-place.
    """
    for u, v, data in connectivity.edges(data=True):
        pos_u = np.asarray(connectivity.nodes[u]["pos"], dtype=np.float64)
        pos_v = np.asarray(connectivity.nodes[v]["pos"], dtype=np.float64)

        r_u, theta_u = float(pos_u[0]), float(pos_u[1])
        r_v, theta_v = float(pos_v[0]), float(pos_v[1])

        d_r = r_v - r_u
        d_theta = theta_v - theta_u
        r_mean = 0.5 * (r_u + r_v)

        arc = r_mean * d_theta
        distance = float(np.hypot(d_r, arc))
        data["distance"] = distance
