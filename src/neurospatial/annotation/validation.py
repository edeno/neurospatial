"""Validation functions for annotation quality checks.

This module provides validation functions to detect common issues with
annotated polygons and regions:

- Self-intersecting polygons
- Very small polygons (below threshold)
- Regions overlapping heavily
- Regions extending outside the environment boundary

All validation functions emit warnings rather than raising errors,
allowing users to proceed with awareness of potential issues.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from shapely import Polygon
from shapely.validation import explain_validity

if TYPE_CHECKING:
    from neurospatial.regions import Region, Regions


# Default thresholds for validation
DEFAULT_MIN_AREA_THRESHOLD = 1e-6  # Minimum polygon area (in coordinate units²)
DEFAULT_OVERLAP_THRESHOLD = 0.5  # Fraction of region area overlapping another


def validate_polygon_geometry(
    polygon: Polygon,
    name: str,
    *,
    min_area: float = DEFAULT_MIN_AREA_THRESHOLD,
    warn_self_intersecting: bool = True,
    warn_small_area: bool = True,
) -> list[str]:
    """Validate polygon geometry and emit warnings for issues.

    Parameters
    ----------
    polygon : Polygon
        Shapely Polygon to validate.
    name : str
        Name of the polygon (for warning messages).
    min_area : float, default=1e-6
        Minimum area threshold. Polygons with area below this are considered
        degenerate and will trigger a warning.
    warn_self_intersecting : bool, default=True
        Whether to warn about self-intersecting polygons.
    warn_small_area : bool, default=True
        Whether to warn about small polygons.

    Returns
    -------
    list of str
        List of validation issue descriptions (empty if no issues).

    Notes
    -----
    Self-intersecting polygons are automatically fixed using ``buffer(0)``
    in the conversion pipeline, but this warning alerts users that their
    hand-drawn polygon was self-intersecting.

    Examples
    --------
    >>> from shapely import Polygon
    >>> # Valid polygon
    >>> valid_poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    >>> issues = validate_polygon_geometry(valid_poly, "test")
    >>> len(issues)
    0

    >>> # Self-intersecting (bowtie) polygon
    >>> bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])
    >>> issues = validate_polygon_geometry(bowtie, "bowtie")
    >>> "self-intersect" in issues[0].lower()
    True

    """
    issues = []

    # Check for self-intersection
    if warn_self_intersecting and not polygon.is_valid:
        reason = explain_validity(polygon)
        issues.append(f"Polygon '{name}' is self-intersecting: {reason}")
        warnings.warn(
            f"Polygon '{name}' is self-intersecting ({reason}). "
            "It will be automatically fixed using buffer(0), but you may want "
            "to redraw it for cleaner boundaries.",
            UserWarning,
            stacklevel=3,
        )

    # Check for very small area
    area = polygon.area
    if warn_small_area and area < min_area:
        issues.append(
            f"Polygon '{name}' has very small area ({area:.2e} < {min_area:.2e})",
        )
        warnings.warn(
            f"Polygon '{name}' has very small area ({area:.2e}). "
            "This may indicate a degenerate or collapsed polygon. "
            "Consider redrawing with larger extent.",
            UserWarning,
            stacklevel=3,
        )

    # Check for empty polygon
    if polygon.is_empty:
        issues.append(f"Polygon '{name}' is empty")
        warnings.warn(
            f"Polygon '{name}' is empty (no area). This polygon will be skipped.",
            UserWarning,
            stacklevel=3,
        )

    return issues


def validate_region_within_boundary(
    region: Region,
    boundary: Region,
    *,
    warn_outside: bool = True,
    tolerance: float = 0.01,
) -> list[str]:
    """Check if a region extends outside the environment boundary.

    Parameters
    ----------
    region : Region
        Region to check.
    boundary : Region
        Environment boundary polygon.
    warn_outside : bool, default=True
        Whether to emit warning if region extends outside boundary.
    tolerance : float, default=0.01
        Fraction of region area that can be outside boundary without warning.
        Set to 0.0 for strict containment checking.

    Returns
    -------
    list of str
        List of validation issue descriptions (empty if no issues).

    Notes
    -----
    Regions that extend slightly outside the boundary (< tolerance fraction)
    are acceptable due to drawing imprecision. Larger extensions trigger warnings.

    """
    issues: list[str] = []

    if region.kind != "polygon" or boundary.kind != "polygon":
        return issues  # Only check polygon regions

    # Type narrowing: we know these are Polygons after the kind check
    region_poly: Polygon = region.data
    boundary_poly: Polygon = boundary.data

    # Check if region is completely within boundary
    if not boundary_poly.contains(region_poly):
        # Calculate what fraction is outside
        intersection = region_poly.intersection(boundary_poly)
        region_area = region_poly.area

        if region_area > 0:
            fraction_inside = intersection.area / region_area
            fraction_outside = 1.0 - fraction_inside

            if fraction_outside > tolerance:
                issues.append(
                    f"Region '{region.name}' extends {fraction_outside:.1%} "
                    f"outside the environment boundary",
                )
                if warn_outside:
                    warnings.warn(
                        f"Region '{region.name}' extends {fraction_outside:.1%} "
                        f"outside the environment boundary. "
                        "Consider redrawing the region within the boundary.",
                        UserWarning,
                        stacklevel=3,
                    )

    return issues


def validate_region_overlap(
    regions: Regions,
    *,
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
    warn_overlap: bool = True,
) -> list[str]:
    """Check for heavy overlap between regions.

    Parameters
    ----------
    regions : Regions
        Collection of regions to check for overlap.
    overlap_threshold : float, default=0.5
        Fraction of region area overlapping another region that triggers
        a warning. Set to 1.0 to disable overlap warnings.
    warn_overlap : bool, default=True
        Whether to emit warnings for overlapping regions.

    Returns
    -------
    list of str
        List of validation issue descriptions (empty if no issues).

    Notes
    -----
    Some overlap between regions may be intentional (e.g., nested regions).
    This validation warns about significant overlap (>50% by default) that
    may indicate drawing errors.

    The overlap is checked bidirectionally: if A overlaps 60% of B's area
    OR B overlaps 60% of A's area, a warning is issued.

    """
    issues: list[str] = []

    # Get all polygon regions
    polygon_regions = [r for r in regions.values() if r.kind == "polygon"]

    if len(polygon_regions) < 2:
        return issues  # Need at least 2 regions to check overlap

    # Check pairwise overlap
    checked_pairs: set[tuple[str, str]] = set()

    for r1 in polygon_regions:
        for r2 in polygon_regions:
            if r1.name == r2.name:
                continue

            # Create canonical pair key to avoid checking both (A,B) and (B,A)
            sorted_names = sorted([r1.name, r2.name])
            pair_key: tuple[str, str] = (sorted_names[0], sorted_names[1])
            if pair_key in checked_pairs:
                continue
            checked_pairs.add(pair_key)

            # Calculate overlap (we know these are Polygons after kind check)
            poly1: Polygon = r1.data
            poly2: Polygon = r2.data

            if not poly1.intersects(poly2):
                continue

            intersection = poly1.intersection(poly2)
            intersection_area = intersection.area

            # Check overlap fraction for both regions
            overlap_frac_1 = intersection_area / poly1.area if poly1.area > 0 else 0
            overlap_frac_2 = intersection_area / poly2.area if poly2.area > 0 else 0
            max_overlap = max(overlap_frac_1, overlap_frac_2)

            if max_overlap > overlap_threshold:
                issues.append(
                    f"Regions '{r1.name}' and '{r2.name}' overlap heavily "
                    f"({max_overlap:.1%} of smaller region)",
                )
                if warn_overlap:
                    warnings.warn(
                        f"Regions '{r1.name}' and '{r2.name}' overlap heavily "
                        f"({max_overlap:.1%} of the smaller region's area). "
                        "This may be intentional, but could indicate a drawing error.",
                        UserWarning,
                        stacklevel=3,
                    )

    return issues


def validate_environment_boundary(
    boundary: Region,
    *,
    min_area: float = DEFAULT_MIN_AREA_THRESHOLD,
    warn_issues: bool = True,
) -> list[str]:
    """Validate environment boundary polygon for common issues.

    Parameters
    ----------
    boundary : Region
        Environment boundary region.
    min_area : float, default=1e-6
        Minimum area threshold for the boundary.
    warn_issues : bool, default=True
        Whether to emit warnings for issues found.

    Returns
    -------
    list of str
        List of validation issue descriptions (empty if no issues).

    Notes
    -----
    This is a convenience function that combines polygon geometry validation
    with environment-specific checks.

    """
    issues: list[str] = []

    if boundary.kind != "polygon":
        return issues

    polygon: Polygon = boundary.data

    # Validate basic polygon geometry
    geometry_issues = validate_polygon_geometry(
        polygon,
        boundary.name,
        min_area=min_area,
        warn_self_intersecting=warn_issues,
        warn_small_area=warn_issues,
    )
    issues.extend(geometry_issues)

    return issues


def validate_annotations(
    regions: Regions,
    boundary: Region | None = None,
    *,
    min_area: float = DEFAULT_MIN_AREA_THRESHOLD,
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
    containment_tolerance: float = 0.01,
    warn_issues: bool = True,
) -> list[str]:
    """Comprehensive validation of annotated regions and boundary.

    This is the main validation entry point that checks all annotation
    quality issues:

    1. Polygon geometry (self-intersection, small area)
    2. Region-boundary containment
    3. Region-region overlap

    Parameters
    ----------
    regions : Regions
        Collection of annotated regions.
    boundary : Region, optional
        Environment boundary polygon. If provided, regions are checked
        for containment within the boundary.
    min_area : float, default=1e-6
        Minimum polygon area threshold.
    overlap_threshold : float, default=0.5
        Fraction of region area overlap that triggers warnings.
    containment_tolerance : float, default=0.01
        Fraction of region area that can be outside boundary without warning.
    warn_issues : bool, default=True
        Whether to emit warnings for issues found.

    Returns
    -------
    list of str
        List of all validation issue descriptions.

    Examples
    --------
    >>> from neurospatial.regions import Region, Regions
    >>> from shapely import Polygon
    >>> boundary = Region(
    ...     name="arena",
    ...     kind="polygon",
    ...     data=Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
    ... )
    >>> r1 = Region(
    ...     name="goal",
    ...     kind="polygon",
    ...     data=Polygon([(10, 10), (30, 10), (30, 30), (10, 30)]),
    ... )
    >>> regions = Regions([r1])
    >>> issues = validate_annotations(regions, boundary)
    >>> len(issues)  # No issues with valid regions
    0

    """
    all_issues = []

    # Validate boundary if provided
    if boundary is not None:
        boundary_issues = validate_environment_boundary(
            boundary,
            min_area=min_area,
            warn_issues=warn_issues,
        )
        all_issues.extend(boundary_issues)

    # Validate each region's geometry
    for region in regions.values():
        if region.kind != "polygon":
            continue

        geometry_issues = validate_polygon_geometry(
            region.data,
            region.name,
            min_area=min_area,
            warn_self_intersecting=warn_issues,
            warn_small_area=warn_issues,
        )
        all_issues.extend(geometry_issues)

        # Check containment within boundary
        if boundary is not None:
            containment_issues = validate_region_within_boundary(
                region,
                boundary,
                warn_outside=warn_issues,
                tolerance=containment_tolerance,
            )
            all_issues.extend(containment_issues)

    # Check region overlap
    overlap_issues = validate_region_overlap(
        regions,
        overlap_threshold=overlap_threshold,
        warn_overlap=warn_issues,
    )
    all_issues.extend(overlap_issues)

    return all_issues
