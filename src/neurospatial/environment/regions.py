"""Region-related operations for Environment.

This module provides methods for querying bins and masks within named regions.
Regions can be points or polygons (2D only) defined in the Environment's regions
container.

"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from neurospatial.environment._protocols import SelfEnv
from neurospatial.environment.decorators import check_fitted
from neurospatial.regions import Regions

# Conditional import for shapely (optional dependency)
try:
    import shapely  # noqa: F401

    _HAS_SHAPELY = True
except ModuleNotFoundError:
    _HAS_SHAPELY = False


class EnvironmentRegions:
    """Mixin class providing region-related operations for Environment.

    This class provides methods for querying which bins fall within named regions
    and generating boolean masks for region membership. All methods require
    a fitted Environment instance.

    Methods
    -------
    bins_in_region(region_name)
        Get active bin indices that fall within a specified named region.
    mask_for_region(region_name)
        Get a boolean mask over active bins indicating membership in a region.
    region_mask(regions, *, include_boundary=True)
        Get boolean mask for one or more regions (accepts names, Region, or Regions).
    region_membership(regions=None, *, include_boundary=True)
        Check which bins belong to which regions (returns 2D array).

    Notes
    -----
    This is a mixin class designed to be used with the Environment class.
    It is NOT a dataclass and should not be instantiated directly.

    See Also
    --------
    neurospatial.regions.Region : Region data structure
    neurospatial.regions.Regions : Container for multiple regions

    """

    @check_fitted
    def bins_in_region(self: SelfEnv, region_name: str) -> NDArray[np.int_]:
        """Get active bin indices that fall within a specified named region.

        This method identifies all active bins whose centers fall within the
        geometric bounds of a named region. For point regions, returns the
        bin containing that point. For polygon regions (2D only), returns all
        bins whose centers are contained by the polygon.

        Parameters
        ----------
        region_name : str
            The name of a defined region in `self.regions`.

        Returns
        -------
        NDArray[np.int_]
            Array of active bin indices (0 to n_active_bins - 1)
            that are part of the region. Returns empty array if no bins
            fall within the region.

        Raises
        ------
        KeyError
            If `region_name` is not found in `self.regions`.
        ValueError
            If region point dimension does not match environment dimension,
            if polygon regions are used in non-2D environments, or if
            region kind is unsupported.
        RuntimeError
            If polygon region is requested but shapely is not installed.

        Notes
        -----
        - Point regions return at most one bin (the bin containing the point,
          or empty array if point is outside environment)
        - Polygon regions require shapely library and 2D environments
        - Bin membership is determined by whether the bin center falls within
          the region geometry

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(100, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> env.regions.add("center", point=(5.0, 5.0))
        >>> bins = env.bins_in_region("center")
        >>> bins.shape
        (1,)

        For polygon regions (requires shapely):

        >>> from shapely.geometry import box
        >>> env.regions.add("roi", polygon=box(2, 2, 8, 8))
        >>> bins = env.bins_in_region("roi")
        >>> len(bins)  # Number of bins in the 6x6 square
        9

        See Also
        --------
        mask_for_region : Get boolean mask for region membership
        neurospatial.regions.Region : Region data structure

        """
        region = self.regions[region_name]

        if region.kind == "point":
            point_nd = np.asarray(region.data).reshape(1, -1)
            if point_nd.shape[1] != self.n_dims:
                raise ValueError(
                    f"Region point dimension {point_nd.shape[1]} "
                    f"does not match environment dimension {self.n_dims}.",
                )
            bin_idx = self.bin_at(point_nd)
            return np.asarray(bin_idx[bin_idx != -1], dtype=int)

        if region.kind == "polygon":
            # Check shapely availability from parent module for test compatibility
            import neurospatial.environment as _env_module

            if not _env_module._HAS_SHAPELY:  # pragma: no cover
                raise RuntimeError("Polygon region queries require 'shapely'.")
            if self.n_dims != 2:  # pragma: no cover
                raise ValueError(
                    "Polygon regions are only supported for 2D environments.",
                )

            import shapely

            polygon = region.data
            contained_mask = shapely.contains_xy(
                polygon,
                self.bin_centers[:, 0],
                self.bin_centers[:, 1],
            )

            return np.flatnonzero(contained_mask)

        # pragma: no cover
        raise ValueError(f"Unsupported region kind: {region.kind}")

    @check_fitted
    def mask_for_region(self: SelfEnv, region_name: str) -> NDArray[np.bool_]:
        """Get a boolean mask over active bins indicating membership in a region.

        This method creates a boolean array of length n_active_bins where True
        indicates that the corresponding bin falls within the named region.
        This is useful for filtering data arrays indexed by bin ID.

        Parameters
        ----------
        region_name : str
            Name of region to query.

        Returns
        -------
        NDArray[np.bool_]
            Boolean array of shape (n_active_bins,). True if an active bin
            is part of the region, False otherwise. All False if no bins
            fall within the region.

        Raises
        ------
        KeyError
            If `region_name` is not found in `self.regions`.
        ValueError
            If region geometry is incompatible with environment (see
            `bins_in_region` for details).
        RuntimeError
            If polygon region is requested but shapely is not installed.

        Notes
        -----
        This method is implemented by calling `bins_in_region()` and converting
        the result to a boolean mask. It provides a convenient way to filter
        data arrays by region membership.

        The mask length equals the number of active bins (`n_bins`), making it
        suitable for indexing arrays that store per-bin data.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(100, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> env.regions.add("center", point=(5.0, 5.0))
        >>> mask = env.mask_for_region("center")
        >>> mask.shape
        (25,)  # Number of active bins
        >>> mask.sum()  # Number of bins in region
        1

        Use the mask to filter per-bin data:

        >>> occupancy = np.random.rand(env.n_bins)
        >>> region_occupancy = occupancy[mask]
        >>> region_occupancy.shape
        (1,)

        See Also
        --------
        bins_in_region : Get array of bin indices in region
        neurospatial.regions.Region : Region data structure

        """
        active_bins_for_mask = self.bins_in_region(region_name)
        mask = np.zeros(self.bin_centers.shape[0], dtype=bool)
        if active_bins_for_mask.size > 0:
            mask[active_bins_for_mask] = True
        return mask

    def region_membership(
        self: SelfEnv,
        regions: Regions | None = None,
        *,
        include_boundary: bool = True,
    ) -> NDArray[np.bool_]:
        """Check which bins belong to which regions.

        This method performs vectorized containment checks to determine which
        bins are inside each region. Useful for:
        - Filtering bins by region
        - Computing region-specific statistics
        - Identifying spatial distributions across regions
        - Selecting bins for subset operations

        Parameters
        ----------
        regions : Regions, optional
            Regions to test against. If None (default), uses self.regions.
            Allows testing against external region sets without modifying
            the environment.
        include_boundary : bool, default=True
            How to handle bins on region boundaries:
            - True: Bins on boundary count as inside (shapely.covers).
            - False: Only bins strictly inside count (shapely.contains).

        Returns
        -------
        membership : NDArray[np.bool_], shape (n_bins, n_regions)
            Boolean array where membership[i, j] = True if bin i is in region j.
            Columns are ordered according to region iteration order.
            If regions is empty, returns array with shape (n_bins, 0).

        Raises
        ------
        TypeError
            If regions parameter is not a Regions instance or None.
            If include_boundary is not a boolean.

        See Also
        --------
        subset : Create new environment from bin selection.
        bins_in_region : Get bin indices for a specific region.

        Notes
        -----
        **Region Types**:
        - Polygon regions: Uses Shapely containment (covers/contains).
        - Point regions: Always return False (points have no area).

        **Performance**:
        This method uses vectorized Shapely operations for efficiency.
        For N bins and R regions, complexity is O(N * R), but vectorized
        operations make it fast even for thousands of bins.

        **Boundary Semantics**:
        The include_boundary parameter controls the Shapely predicate used:
        - include_boundary=True: Uses shapely.covers(region, point)
          Returns True if point is inside region OR on its boundary.
        - include_boundary=False: Uses shapely.contains(region, point)
          Returns True only if point is strictly inside region.

        For most applications, include_boundary=True is appropriate, as it
        avoids ambiguity for bins whose centers lie exactly on region edges.

        **Region Order**:
        The column order in the output array matches the iteration order of
        the regions mapping. For self.regions, this is insertion order.

        Examples
        --------
        >>> import numpy as np
        >>> from shapely.geometry import box
        >>> # Create 10x10 grid
        >>> data = np.array([[i, j] for i in range(11) for j in range(11)])
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> _ = env.regions.add("left", polygon=box(0, 0, 5, 10))
        >>> _ = env.regions.add("right", polygon=box(5, 0, 10, 10))
        >>> membership = env.region_membership()
        >>> membership.shape[1]  # Number of regions
        2
        >>> membership.dtype
        dtype('bool')

        >>> # Find bins in specific region
        >>> left_bins = np.where(membership[:, 0])[0]
        >>> len(left_bins) > 0
        True

        >>> # Bins in multiple regions (overlapping)
        >>> both = np.all(membership, axis=1)
        >>> overlapping_bins = np.where(both)[0]
        >>> len(overlapping_bins) >= 0
        True

        >>> # Use external regions without modifying environment
        >>> from neurospatial.regions import Regions
        >>> external = Regions()
        >>> _ = external.add("test", polygon=box(2, 2, 8, 8))
        >>> test_membership = env.region_membership(regions=external)
        >>> test_membership.shape[1]
        1

        >>> # Strict interior only (exclude boundary)
        >>> interior = env.region_membership(include_boundary=False)
        >>> bool(interior.sum() <= membership.sum())  # Fewer or equal bins
        True

        """
        # Import here to avoid circular dependency
        from neurospatial.regions import Regions

        # Input validation
        if regions is None:
            regions = self.regions
        elif not isinstance(regions, Regions):
            raise TypeError(
                f"regions must be a Regions instance or None, "
                f"got {type(regions).__name__}"
            )

        if not isinstance(include_boundary, bool):
            raise TypeError(
                f"include_boundary must be a bool, got {type(include_boundary).__name__}"
            )

        # Handle empty regions case
        if len(regions) == 0:
            return np.zeros((self.n_bins, 0), dtype=bool)

        # Get bin centers as points
        bin_centers = self.bin_centers  # shape (n_bins, n_dims)

        # Initialize membership array
        n_regions = len(regions)
        membership = np.zeros((self.n_bins, n_regions), dtype=bool)

        # Import shapely functions for vectorized operations
        from shapely import contains, covers
        from shapely import points as shapely_points

        # Check if any polygon regions exist and validate dimensionality
        has_polygon_regions = any(r.kind == "polygon" for r in regions.values())
        if has_polygon_regions:
            # Only supports 2D for now
            if bin_centers.shape[1] != 2:
                raise NotImplementedError(
                    f"region_membership currently only supports 2D environments "
                    f"for polygon regions. Environment has {bin_centers.shape[1]} dimensions."
                )
            # Create shapely Points array ONCE before loop (major optimization)
            points = shapely_points(bin_centers[:, 0], bin_centers[:, 1])

        # Iterate over regions and check containment
        for region_idx, (_region_name, region) in enumerate(regions.items()):
            # Handle point regions - points have no area, so no bins can be inside
            if region.kind == "point":
                # Leave column as all False (no bin can be "inside" a point)
                continue

            # Handle polygon regions
            if region.kind == "polygon":
                # Use pre-created points array (created once before loop)
                # Vectorized containment check
                if include_boundary:
                    # covers: True if point is inside or on boundary
                    mask = covers(region.data, points)
                else:
                    # contains: True only if strictly inside
                    mask = contains(region.data, points)

                membership[:, region_idx] = mask

        return membership

    @check_fitted
    def region_mask(
        self: SelfEnv,
        regions: str | list[str] | object,  # Will be Region | Regions after import
        *,
        include_boundary: bool = True,
    ) -> NDArray[np.bool_]:
        """Get boolean mask over bins for one or more regions.

        This method provides a convenient interface to rasterize continuous regions
        onto discrete environment bins. It accepts multiple input types and returns
        a boolean mask indicating which bins fall within the region(s).

        This is the method-based interface to the `regions_to_mask()` free function,
        providing better discoverability and ergonomics when working with a single
        environment.

        Parameters
        ----------
        regions : str, list[str], Region, or Regions
            Region(s) to rasterize. Can be:
            - A single region name (str) from self.regions
            - A list of region names (list[str]) from self.regions
            - A single Region object
            - A Regions collection
        include_boundary : bool, default=True
            Whether to include bins whose centers lie on region boundaries:
            - True: Include boundary bins (uses shapely.covers)
            - False: Exclude boundary bins (uses shapely.contains)

        Returns
        -------
        mask : NDArray[np.bool_], shape (n_bins,)
            Boolean mask where True indicates bin center is inside region(s).
            For multiple regions, returns the union (logical OR).
            Point regions always return all False (points have no area).

        Raises
        ------
        KeyError
            If a region name is not found in self.regions.
        TypeError
            If regions parameter has invalid type or include_boundary is not bool.

        Notes
        -----
        **Region Types**:
        - Polygon regions: Rasterized using Shapely containment checks
        - Point regions: Always return all False (points have no area)

        **Multiple Regions**:

        When multiple regions are provided, the result is their **union** (logical OR).
        Bins are included if they fall inside ANY of the regions. Overlapping
        regions are counted only once.

        For intersection or difference, use boolean operations on separate masks:

        >>> left_mask = env.region_mask("left")  # doctest: +SKIP
        >>> right_mask = env.region_mask("right")  # doctest: +SKIP
        >>> intersection = left_mask & right_mask  # doctest: +SKIP
        >>> difference = left_mask & ~right_mask  # doctest: +SKIP

        **Boundary Semantics**:

        - include_boundary=True: Uses shapely.covers (includes boundary)
        - include_boundary=False: Uses shapely.contains (excludes boundary)

        Examples
        --------
        >>> import numpy as np
        >>> from shapely.geometry import box
        >>> from neurospatial import Environment

        Create environment with regions:

        >>> data = np.array([[i, j] for i in range(11) for j in range(11)])
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> env.regions.add("center", polygon=box(3, 3, 7, 7))
        >>> env.regions.add("left", polygon=box(0, 0, 4, 10))
        >>> env.regions.add("right", polygon=box(6, 0, 10, 10))

        Single region by name:

        >>> mask = env.region_mask("center")
        >>> mask.shape
        (36,)
        >>> mask.dtype
        dtype('bool')

        Multiple regions (union):

        >>> mask = env.region_mask(["left", "right"])
        >>> np.any(mask)
        True

        All regions in environment:

        >>> mask = env.region_mask(env.regions)
        >>> mask.shape
        (36,)

        Exclude boundary bins:

        >>> mask_no_boundary = env.region_mask("center", include_boundary=False)
        >>> mask_with_boundary = env.region_mask("center", include_boundary=True)
        >>> bool(mask_no_boundary.sum() <= mask_with_boundary.sum())
        True

        Use with external Region object:

        >>> from neurospatial.regions import Region
        >>> external_region = Region(name="ext", kind="polygon", data=box(4, 4, 6, 6))
        >>> mask = env.region_mask(external_region)
        >>> mask.shape
        (36,)

        See Also
        --------
        region_membership : 2D membership array for all regions
        bins_in_region : Get bin indices for a specific region
        mask_for_region : Get boolean mask for a single region name
        neurospatial.spatial.regions_to_mask : Free function version

        """
        # Import here to avoid circular dependency
        from typing import cast

        from neurospatial.environment.core import Environment
        from neurospatial.ops.binning import regions_to_mask

        # Delegate to the free function (cast for mypy compatibility)
        return regions_to_mask(
            cast("Environment", self), regions, include_boundary=include_boundary
        )
