"""Region-related operations for Environment.

This module provides methods for querying bins and masks within named regions.
Regions can be points or polygons (2D only) defined in the Environment's regions
container.

"""

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from neurospatial.environment.decorators import check_fitted

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment

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
    def bins_in_region(self: "Environment", region_name: str) -> NDArray[np.int_]:
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
            if not _HAS_SHAPELY:  # pragma: no cover
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
    def mask_for_region(self: "Environment", region_name: str) -> NDArray[np.bool_]:
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
