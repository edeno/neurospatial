"""Deterministic floating-geometry evidence for the Stage A clipping design.

Run with ``uv run --no-sync python docs/plans/environment-arena/probe_polygon_clipping.py``.
This is a design probe, not a clipping implementation or a real-arithmetic claim.
No precision grid, minimum-area cutoff, triangulation or geometry repair is used.
"""

from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.ops import polygonize, unary_union

from neurospatial.layout._diffusion import DiffusionGeometry, FaceGeometry
from neurospatial.ops.diffusion import heat_kernel_from_W


def parts(geometry: shapely.Geometry) -> list[Polygon]:
    """Keep positive-area connected polygon pieces, including their holes."""
    if isinstance(geometry, Polygon):
        return [geometry] if geometry.area > 0 else []
    if hasattr(geometry, "geoms"):
        return [part for child in geometry.geoms for part in parts(child)]
    return []


def key(polygon: Polygon) -> tuple:
    """Canonical ordering is explicit; GEOS collection order is not an ID."""
    return (*polygon.bounds, polygon.area, shapely.normalize(polygon).wkb)


def anchor(polygon: Polygon) -> tuple[float, float]:
    """Use the area centroid when interior, otherwise an interior representative."""
    point = polygon.centroid
    if not polygon.contains(point):
        point = shapely.normalize(polygon).representative_point()
    assert polygon.contains(point), "No representable strict interior anchor"
    return point.x, point.y


def overlay(
    cells: list[Polygon], region: Polygon | MultiPolygon, roots: list[int] | None = None
) -> list[tuple[int, int, Polygon]]:
    """Node boundaries once, classify faces, and merge atoms within each parent."""
    roots = list(range(len(cells))) if roots is None else roots
    atoms: dict[int, list[Polygon]] = defaultdict(list)
    lines = unary_union([*(cell.boundary for cell in cells), region.boundary])
    for face in polygonize(lines):
        point = face.representative_point()
        if not region.contains(point):
            continue
        owners = [index for index, cell in enumerate(cells) if cell.contains(point)]
        assert len(owners) <= 1, "Source cells overlap in their interiors"
        if owners:
            atoms[owners[0]].append(face)
    result = [
        (parent, roots[parent], shapely.normalize(piece))
        for parent, faces in atoms.items()
        for piece in parts(unary_union(faces))
    ]
    result.sort(key=lambda entry: (entry[0], key(entry[2])))
    for _, _, piece in result:
        assert piece.is_valid and np.isfinite(piece.area) and piece.area > 0
        anchor(piece)
    return result


def assert_same_support(actual: shapely.Geometry, expected: shapely.Geometry) -> float:
    """The fixed simulated examples use an explicit float64 area-error bound."""
    error = actual.symmetric_difference(expected).area
    assert error <= 256 * np.finfo(float).eps * max(1.0, expected.area)
    return error


def vertical_face_flux(
    cells: list[Polygon], centers: np.ndarray
) -> tuple[float, float, float, DiffusionGeometry]:
    """Evaluate today's scalar operator for u=y across a vertical shared face."""
    shared = cells[0].boundary.intersection(cells[1].boundary)
    vertices = np.asarray(shared.coords)
    assert np.all(vertices[:, 0] == vertices[0, 0])
    area = shared.length
    distance = float(np.linalg.norm(centers[1] - centers[0]))
    faces = FaceGeometry(
        np.array([[0, 1]], dtype=np.intp),
        np.array([area]),
        np.array([distance]),
        np.array([cell.area for cell in cells]),
    )
    geometry = DiffusionGeometry.from_faces(faces)
    flux = float(geometry.W[0, 1] * (centers[1, 1] - centers[0, 1]))
    return area, distance, flux, geometry


def numerical_evidence() -> None:
    """Compare analytic normal flux with existing TPFA and center choices."""
    sloping = Polygon([(-2, -2), (4, -2), (4, 1.25), (-2, -0.25)])
    cells = [box(x, 0, x + 1, 1).intersection(sloping) for x in (0, 1)]
    centers = np.array([anchor(cell) for cell in cells])
    np.testing.assert_allclose([cell.area for cell in cells], [3 / 8, 5 / 8])
    np.testing.assert_allclose(centers, [[5 / 9, 7 / 36], [23 / 15, 19 / 60]])
    area, distance, flux, geometry = vertical_face_flux(cells, centers)
    assert area == 0.5 and flux > 0.06  # Analytic normal derivative of u=y is zero.
    print(
        f"grid trapezoids: areas (3/8, 5/8), A={area:g}, d={distance:.12g}; "
        f"u=y numerical normal flux {flux:.12g}, analytic 0"
    )
    for mode in ("average", "transition", "density"):
        kernel = heat_kernel_from_W(geometry.W, geometry.volumes, 1.0, mode=mode)
        sums = (
            kernel.sum(axis=1)
            if mode == "average"
            else kernel.sum(axis=0)
            if mode == "transition"
            else geometry.volumes @ kernel
        )
        assert np.all(kernel >= 0)
        np.testing.assert_allclose(sums, 1.0)
    print("wrong flux still passes positivity and all three kernel normalizations")

    half_width = np.sqrt(3) / 2
    left = Polygon(
        [
            (half_width, -0.5),
            (half_width, 0.5),
            (0, 1),
            (-half_width, 0.5),
            (-half_width, -0.5),
            (0, -1),
        ]
    )
    right = Polygon(
        [
            (3 * half_width, -0.5),
            (3 * half_width, 0.5),
            (2 * half_width, 1),
            (half_width, 0.5),
            (half_width, -0.5),
            (2 * half_width, -1),
        ]
    )
    cells = [cell.intersection(sloping) for cell in (left, right)]
    centroids = np.array([anchor(cell) for cell in cells])
    _, _, centroid_flux, _ = vertical_face_flux(cells, centroids)
    parents = np.array([[0.0, 0.0], [2 * half_width, 0.0]])
    assert all(
        cell.contains(shapely.Point(point))
        for cell, point in zip(cells, parents, strict=True)
    )
    _, _, parent_flux, _ = vertical_face_flux(cells, parents)
    assert centroid_flux > 0.1 and parent_flux == 0
    print(
        f"canonical hex cut: u=y centroid flux {centroid_flux:.12g}; parent-anchor flux 0"
    )

    cells = [box(0.25, 0, 1, 1), box(1, 0, 2, 1)]
    area, distance, flux, _ = vertical_face_flux(
        cells, np.array([anchor(c) for c in cells])
    )
    assert area == 1 and distance == 0.875 and flux == 0
    print("valid rectangle clip: areas (0.75, 1), A=1, d=0.875; u=y flux 0")

    first, final = box(0, 0, 0.4, 1), box(0, 0, 0.4, 0.8)
    sequential_parent_anchor = anchor(first)
    direct_parent_anchor = anchor(final)
    assert final.contains(shapely.Point(sequential_parent_anchor))
    assert sequential_parent_anchor != direct_parent_anchor
    print(
        f"parent-anchor history: x<=0.4 then y<=0.8 gives {sequential_parent_anchor}; "
        f"direct same rectangle gives {direct_parent_anchor}; centroid rule agrees geometrically"
    )


def main() -> None:
    """Exercise splits, holes, degeneracies, shared faces and repeated overlays."""
    print(
        f"versions: Python {sys.version.split()[0]}, NumPy {np.__version__}, "
        f"Shapely {shapely.__version__}, GEOS {shapely.geos_version_string}"
    )
    cell = box(0, 0, 4, 4)
    lobes = MultiPolygon([box(3, 0, 4, 4), box(0, 0, 1, 4)])
    split = overlay([cell], lobes)
    assert [parent for parent, _, _ in split] == [0, 0]
    assert [piece.bounds[0] for _, _, piece in split] == [0, 3]
    reversed_split = overlay([cell], MultiPolygon(list(reversed(lobes.geoms))))
    assert [piece.wkb for _, _, piece in split] == [p.wkb for _, _, p in reversed_split]
    print("split: 2 connected bins; parent mapping [0, 0]; stable reversed-input order")

    hole = Polygon(cell.exterior.coords, holes=[box(1, 1, 3, 3).exterior.coords])
    concave = Polygon([(0, 0), (4, 0), (4, 4), (3, 4), (3, 1), (1, 1), (1, 4), (0, 4)])
    for label, polygon in [("hole", hole), ("concave", concave)]:
        assert not polygon.contains(polygon.centroid)
        print(f"{label}: centroid outside; interior anchor {anchor(polygon)}")

    for touching in [box(4, 0, 5, 4), box(4, 4, 5, 5)]:
        assert not overlay([cell], touching)
    thin = overlay([cell], box(0, 0, 2.0**-40, 4))
    assert len(thin) == 1 and thin[0][2].area == 2.0**-38
    print(
        f"degeneracies: line/point touches discarded; positive sliver area {thin[0][2].area:.12g}"
    )

    cells = [box(x, y, x + 2, y + 2) for x in (0, 2) for y in (0, 2)]
    region = Polygon(
        [(0.25, 0.15), (3.8, 0.45), (3.45, 3.8), (0.2, 3.2)],
        holes=[[(1, 1), (2.6, 1.1), (2.4, 2.8), (0.9, 2.7)]],
    )
    clipped = overlay(cells, region)
    union = unary_union([piece for _, _, piece in clipped])
    error = assert_same_support(union, unary_union(cells).intersection(region))
    area = sum(piece.area for _, _, piece in clipped)
    assert abs(area - union.area) <= 256 * np.finfo(float).eps * max(1, union.area)
    print(
        f"coverage: {len(clipped)} bins; area {area:.12g}; symmetric-difference area {error:.3g}"
    )

    second = MultiPolygon([box(0, 0, 1.9, 4), box(2.1, 0, 4, 4)])
    repeated = overlay([p for _, _, p in clipped], second, [r for _, r, _ in clipped])
    direct = overlay(cells, region.intersection(second))
    errors = []
    for root in range(len(cells)):
        left = unary_union([p for _, r, p in repeated if r == root])
        right = unary_union([p for _, r, p in direct if r == root])
        errors.append(assert_same_support(left, right))
    idempotent = overlay([p for _, _, p in clipped], region)
    assert_same_support(unary_union([p for _, _, p in idempotent]), union)
    print(
        f"repeated clips: per-root direct/intersection oracle passes; maximum area error {max(errors):.3g}"
    )

    shared = (
        box(0, 0, 2, 4)
        .intersection(hole)
        .boundary.intersection(box(2, 0, 4, 4).intersection(hole).boundary)
    )
    assert shared.geom_type == "MultiLineString" and shared.length == 2
    print("shared face: two disjoint collinear interface segments; total length 2")
    left, right = [box(x, 0, x + 2, 4).intersection(hole) for x in (0, 2)]
    left_center, right_center = anchor(left), anchor(right)
    center_link = LineString([left_center, right_center])
    assert left_center[1] == right_center[1] == 2
    assert not unary_union([left, right]).covers(center_link)
    print(
        "flux hazard: orthogonal center link crosses the hole and misses both openings"
    )

    radius, width = 1 / np.sqrt(3.0), 1.0
    centers = np.array(
        [(c + (r % 2) / 2, r * np.sqrt(3) / 2) for r in range(3) for c in range(3)]
    )
    angles = np.pi / 2 + np.arange(6) * np.pi / 3
    offsets = radius * np.column_stack((np.cos(angles), np.sin(angles)))
    naive = [Polygon(center + offsets) for center in centers]
    vertex_offsets = [(0, 2), (1, 1), (1, -1), (0, -2), (-1, -1), (-1, 1)]
    vertices: dict[tuple[int, int], tuple[float, float]] = {}
    canonical = []
    for row in range(3):
        for column in range(3):
            keys = [
                (2 * column + row % 2 + dx, 3 * row + dy) for dx, dy in vertex_offsets
            ]
            for x, y in keys:
                vertices.setdefault((x, y), (x * width / 2, y * radius / 2))
            canonical.append(Polygon([vertices[index] for index in keys]))
    pairs = [
        (u, v)
        for u in range(9)
        for v in range(u + 1, 9)
        if np.isclose(np.linalg.norm(centers[u] - centers[v]), width)
    ]
    missing = sum(
        naive[u].boundary.intersection(naive[v].boundary).length == 0 for u, v in pairs
    )
    overlaps = sum(naive[u].intersection(naive[v]).area > 0 for u, v in pairs)
    assert all(
        canonical[u].boundary.intersection(canonical[v].boundary).length > 0
        for u, v in pairs
    )
    assert all(canonical[u].intersection(canonical[v]).area == 0 for u, v in pairs)
    print(
        f"hex topology: independent trig misses {missing}/{len(pairs)} interfaces, creates {overlaps} tiny overlaps; shared integer-key vertices fix both"
    )
    numerical_evidence()


if __name__ == "__main__":
    main()
