"""Egocentric rate computation for object-vector cells.

This module provides result classes and compute functions for egocentric
firing rate analysis, specifically for object-vector cells (OVCs). OVCs fire
when an animal is at a specific distance and direction from an object.

Unlike spatial encoding which uses allocentric coordinates (world-centered),
egocentric encoding uses animal-centered coordinates where:
- Distance = Euclidean or geodesic distance to object
- Direction = Bearing to object relative to animal's heading (0=ahead)

Result Classes
--------------
EgocentricRateResult
    Single-neuron egocentric rate map with convenience methods
EgocentricRatesResult
    Multi-neuron egocentric rate maps with batch methods and iteration

Compute Functions (to be implemented in Tasks 5.7-5.8)
------------------------------------------------------
compute_egocentric_rate
    Compute egocentric firing rate for one neuron
compute_egocentric_rates
    Compute egocentric firing rates for multiple neurons

Coordinate Convention
---------------------
**Egocentric direction** (animal-centered):
- 0 radians = object is directly ahead of animal
- +pi/2 radians = object is to the left
- -pi/2 radians = object is to the right
- +/-pi radians = object is behind

This matches the convention in ``neurospatial.ops.egocentric``.

Examples
--------
>>> import numpy as np
>>> from neurospatial.encoding.egocentric import EgocentricRateResult

>>> # Create environment representing egocentric polar space
>>> from neurospatial import Environment
>>> positions = np.random.rand(100, 2) * 50
>>> ego_env = Environment.from_samples(positions, bin_size=5.0)

>>> # Create result (typically from compute_egocentric_rate)
>>> firing_rate = np.random.rand(ego_env.n_bins) * 10
>>> occupancy = np.ones(ego_env.n_bins)
>>> result = EgocentricRateResult(
...     firing_rate=firing_rate,
...     occupancy=occupancy,
...     ego_env=ego_env,
...     distance_range=(0.0, 50.0),
...     n_distance_bins=10,
...     n_direction_bins=12,
... )

References
----------
Hoydal, O. A., et al. (2019). Object-vector coding in the medial entorhinal
    cortex. Nature, 568(7752), 400-404.
Deshmukh, S. S., & Knierim, J. J. (2011). Representation of non-spatial and
    spatial information in the lateral entorhinal cortex. Frontiers in
    Behavioral Neuroscience, 5, 69.

See Also
--------
neurospatial.encoding.spatial : Spatial rate computation for place cells
neurospatial.encoding.object_vector : Legacy object-vector module
neurospatial.ops.egocentric : Egocentric coordinate transforms
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial.encoding._base import _to_numpy

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from neurospatial import Environment

__all__ = [
    "EgocentricRateResult",
    "EgocentricRatesResult",
]


@dataclass(frozen=True)
class EgocentricRateResult:
    """Result of egocentric rate computation for a single neuron.

    This class wraps an egocentric firing rate map (firing rate by distance
    and direction to object) with its associated metadata. Object-vector cells
    fire when the animal is at a specific distance and direction from an object.

    Parameters
    ----------
    firing_rate : ArrayLike
        Firing rate in egocentric polar coordinates in Hz. Shape is (n_bins,)
        where n_bins is the number of active bins in the egocentric environment.
        The egocentric environment represents a polar grid with distance on one
        axis and direction on another. Can contain NaN for bins with insufficient
        occupancy.
    occupancy : ArrayLike
        Time spent in each egocentric bin in seconds. Shape is (n_bins,).
    ego_env : Environment
        The egocentric polar environment used for the computation. This is
        typically created via ``Environment.from_polar_egocentric()`` and
        represents the (distance, direction) space centered on the animal.
    distance_range : tuple[float, float]
        Range of distances (min, max) covered by the egocentric environment.
    n_distance_bins : int
        Number of distance bins in the egocentric grid.
    n_direction_bins : int
        Number of direction bins in the egocentric grid.

    Attributes
    ----------
    firing_rate : ArrayLike
        Firing rate by egocentric coordinates in Hz. Shape is (n_bins,).
    occupancy : ArrayLike
        Time in each bin in seconds. Shape is (n_bins,).
    ego_env : Environment
        The egocentric polar environment.
    distance_range : tuple[float, float]
        Distance range (min, max).
    n_distance_bins : int
        Number of distance bins.
    n_direction_bins : int
        Number of direction bins.

    Notes
    -----
    This is a frozen dataclass (immutable). All fields are set at construction
    and cannot be modified afterward.

    **Egocentric polar environment**: The ``ego_env`` represents a polar
    coordinate system centered on the animal. Each bin corresponds to a
    (distance, direction) combination relative to the animal's heading.

    Convenience methods (plot, preferred_distance, preferred_direction, etc.)
    are implemented in Tasks 5.2-5.3.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.egocentric import EgocentricRateResult

    >>> # Create a simple egocentric environment
    >>> positions = np.random.rand(100, 2) * 50
    >>> ego_env = Environment.from_samples(positions, bin_size=5.0)

    >>> # Create result
    >>> firing_rate = np.random.rand(ego_env.n_bins) * 10
    >>> occupancy = np.ones(ego_env.n_bins)
    >>> result = EgocentricRateResult(
    ...     firing_rate=firing_rate,
    ...     occupancy=occupancy,
    ...     ego_env=ego_env,
    ...     distance_range=(0.0, 50.0),
    ...     n_distance_bins=10,
    ...     n_direction_bins=12,
    ... )

    >>> # Access fields
    >>> result.firing_rate.shape
    (n_bins,)
    >>> result.distance_range
    (0.0, 50.0)

    See Also
    --------
    EgocentricRatesResult : Batch version for multiple neurons
    compute_egocentric_rate : Function to compute this result
    """

    firing_rate: ArrayLike
    occupancy: ArrayLike
    ego_env: Environment
    distance_range: tuple[float, float]
    n_distance_bins: int
    n_direction_bins: int

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot the egocentric rate map (firing rate by distance/direction).

        Delegates to the egocentric environment's plot_field method for
        consistent visualization across the codebase.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure and axes.
        **kwargs
            Additional keyword arguments passed to ego_env.plot_field().
            Common options include:
            - cmap : str or Colormap, default="viridis"
            - vmin, vmax : float, colorbar limits
            - add_colorbar : bool, default=True

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.

        Notes
        -----
        The egocentric rate map shows firing rate indexed by (distance,
        direction) relative to the object. Distance is the first dimension,
        direction is the second dimension.

        Examples
        --------
        >>> result = EgocentricRateResult(...)
        >>> ax = result.plot()
        >>> plt.show()

        >>> fig, ax = plt.subplots()
        >>> result.plot(ax=ax, cmap="hot", vmax=20.0)

        See Also
        --------
        preferred_distance : Get distance component of peak response
        preferred_direction : Get direction component of peak response
        """
        return self.ego_env.plot_field(_to_numpy(self.firing_rate), ax=ax, **kwargs)

    def preferred_distance(self) -> float:
        """Distance to object at peak firing rate.

        Returns the distance component (first dimension) of the egocentric
        bin where the neuron shows maximum firing rate.

        Returns
        -------
        float
            Distance to object at peak firing rate, in the same units as
            the environment (typically cm). Uses nanargmax to handle NaN
            values in the firing rate map.

        Notes
        -----
        For object-vector cells, this represents the preferred distance to
        the object. A cell with preferred_distance=20 fires most when the
        object is 20 cm away from the animal.

        The distance is extracted from the egocentric environment's bin
        centers. The first component (index 0) represents distance.

        Examples
        --------
        >>> result = EgocentricRateResult(...)
        >>> dist = result.preferred_distance()
        >>> print(f"Preferred distance: {dist:.1f} cm")

        See Also
        --------
        preferred_direction : Get direction component of peak response
        plot : Visualize the egocentric rate map
        """
        firing_rate = _to_numpy(self.firing_rate)
        peak_bin = np.nanargmax(firing_rate)
        bin_centers: NDArray[np.float64] = self.ego_env.bin_centers
        return float(bin_centers[peak_bin, 0])

    def preferred_direction(self) -> float:
        """Direction to object at peak firing rate.

        Returns the direction component (second dimension) of the egocentric
        bin where the neuron shows maximum firing rate. Direction is in
        radians using the egocentric coordinate convention.

        Returns
        -------
        float
            Direction to object at peak firing rate, in radians.
            - 0 = object is directly ahead of animal
            - +π/2 = object is to the left
            - -π/2 = object is to the right
            - ±π = object is behind

            Uses nanargmax to handle NaN values in the firing rate map.

        Notes
        -----
        For object-vector cells, this represents the preferred direction to
        the object relative to the animal's heading. A cell with
        preferred_direction=π/2 fires most when the object is to the left.

        The direction is extracted from the egocentric environment's bin
        centers. The second component (index 1) represents direction.

        **Coordinate convention**: This uses egocentric (animal-centered)
        coordinates, NOT allocentric (world-centered) coordinates:
        - Egocentric: 0 = ahead of animal, +π/2 = left
        - Allocentric: 0 = East, +π/2 = North

        Examples
        --------
        >>> result = EgocentricRateResult(...)
        >>> direction = result.preferred_direction()
        >>> print(f"Preferred direction: {np.degrees(direction):.1f}°")

        See Also
        --------
        preferred_distance : Get distance component of peak response
        plot : Visualize the egocentric rate map
        """
        firing_rate = _to_numpy(self.firing_rate)
        peak_bin = np.nanargmax(firing_rate)
        bin_centers: NDArray[np.float64] = self.ego_env.bin_centers
        return float(bin_centers[peak_bin, 1])


@dataclass(frozen=True)
class EgocentricRatesResult:
    """Result of egocentric rate computation for multiple neurons.

    This class wraps egocentric firing rate maps for a population of neurons
    with shared metadata (occupancy, egocentric environment, bin parameters).
    It supports iteration and indexing to access individual neuron results.

    Parameters
    ----------
    firing_rates : ArrayLike
        Firing rates in egocentric polar coordinates for all neurons in Hz.
        Shape is (n_neurons, n_bins) where n_bins is the number of active
        bins in the egocentric environment.
    occupancy : ArrayLike
        Time spent in each egocentric bin in seconds. Shape is (n_bins,).
        This is shared across all neurons since the animal's trajectory
        (and thus egocentric occupancy) is the same for all neurons.
    ego_env : Environment
        The egocentric polar environment used for the computation.
    distance_range : tuple[float, float]
        Range of distances (min, max) covered by the egocentric environment.
    n_distance_bins : int
        Number of distance bins in the egocentric grid.
    n_direction_bins : int
        Number of direction bins in the egocentric grid.

    Attributes
    ----------
    firing_rates : ArrayLike
        Firing rates for all neurons. Shape is (n_neurons, n_bins).
    occupancy : ArrayLike
        Time in each bin in seconds. Shape is (n_bins,). Shared.
    ego_env : Environment
        The egocentric polar environment.
    distance_range : tuple[float, float]
        Distance range (min, max).
    n_distance_bins : int
        Number of distance bins.
    n_direction_bins : int
        Number of direction bins.

    Notes
    -----
    This is a frozen dataclass (immutable). All fields are set at construction
    and cannot be modified afterward.

    **Iteration interface**: Supports ``len()``, indexing with ``[]``, and
    iteration with ``for``. Each element is an ``EgocentricRateResult`` for
    one neuron.

    Batch methods (preferred_distances, preferred_directions, detect_ovcs,
    to_dataframe) are implemented in Tasks 5.4-5.5.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.egocentric import EgocentricRatesResult

    >>> # Create a simple egocentric environment
    >>> positions = np.random.rand(100, 2) * 50
    >>> ego_env = Environment.from_samples(positions, bin_size=5.0)

    >>> # Create batch result for 3 neurons
    >>> firing_rates = np.random.rand(3, ego_env.n_bins) * 10
    >>> occupancy = np.ones(ego_env.n_bins)
    >>> result = EgocentricRatesResult(
    ...     firing_rates=firing_rates,
    ...     occupancy=occupancy,
    ...     ego_env=ego_env,
    ...     distance_range=(0.0, 50.0),
    ...     n_distance_bins=10,
    ...     n_direction_bins=12,
    ... )

    >>> # Access fields
    >>> len(result)
    3
    >>> result[0]  # Get first neuron as EgocentricRateResult
    EgocentricRateResult(...)

    >>> # Iterate over neurons
    >>> for single in result:
    ...     print(single.firing_rate.max())

    See Also
    --------
    EgocentricRateResult : Single-neuron version
    compute_egocentric_rates : Function to compute this result
    """

    firing_rates: ArrayLike
    occupancy: ArrayLike
    ego_env: Environment
    distance_range: tuple[float, float]
    n_distance_bins: int
    n_direction_bins: int

    def __len__(self) -> int:
        """Return the number of neurons.

        Returns
        -------
        int
            Number of neurons in the batch.

        Examples
        --------
        >>> len(result)
        5
        """
        return len(self.firing_rates)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> EgocentricRateResult:
        """Get single-neuron result by index.

        Parameters
        ----------
        idx : int
            Index of the neuron (0-based).

        Returns
        -------
        EgocentricRateResult
            Egocentric rate result for the specified neuron.

        Examples
        --------
        >>> single = result[0]
        >>> isinstance(single, EgocentricRateResult)
        True
        """
        return EgocentricRateResult(
            firing_rate=self.firing_rates[idx],  # type: ignore[index]
            occupancy=self.occupancy,
            ego_env=self.ego_env,
            distance_range=self.distance_range,
            n_distance_bins=self.n_distance_bins,
            n_direction_bins=self.n_direction_bins,
        )

    def __iter__(self) -> Iterator[EgocentricRateResult]:
        """Iterate over single-neuron results.

        Yields
        ------
        EgocentricRateResult
            Egocentric rate result for each neuron in order.

        Examples
        --------
        >>> for single in result:
        ...     print(single.firing_rate.max())
        """
        for i in range(len(self)):
            yield self[i]
