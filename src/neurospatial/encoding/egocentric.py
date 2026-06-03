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

Compute Functions
-----------------
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
>>> env = Environment.from_samples(positions, bin_size=5.0)

>>> # Create result (typically from compute_egocentric_rate)
>>> firing_rate = np.random.rand(env.n_bins) * 10
>>> occupancy = np.ones(env.n_bins)
>>> result = EgocentricRateResult(
...     firing_rate=firing_rate,
...     occupancy=occupancy,
...     env=env,
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
neurospatial.ops.egocentric : Egocentric coordinate transforms
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial.encoding._base import SpatialResultMixin, _to_numpy

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.projections.polar import PolarAxes

    from neurospatial import Environment
    from neurospatial.environment.polar import EgocentricPolarEnvironment


__all__ = [
    # Result classes
    "EgocentricRateResult",
    "EgocentricRatesResult",
    # Compute functions
    "compute_egocentric_rate",
    "compute_egocentric_rates",
    # Convenience functions
    "is_object_vector_cell",
    "object_vector_score",
    "plot_object_vector_tuning",
]


@dataclass(frozen=True)
class EgocentricRateResult(SpatialResultMixin):
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
    env : Environment
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
    env : Environment
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

    **Egocentric polar environment**: The ``env`` represents a polar
    coordinate system centered on the animal. Each bin corresponds to a
    (distance, direction) combination relative to the animal's heading.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.egocentric import compute_egocentric_rate

    >>> # Build a result from a small, seeded trajectory + spike train
    >>> rng = np.random.default_rng(0)
    >>> times = np.linspace(0, 100, 1000)
    >>> positions = rng.uniform(10, 90, (1000, 2))
    >>> headings = rng.uniform(-np.pi, np.pi, 1000)
    >>> object_positions = np.array([[50.0, 50.0]])
    >>> spike_times = np.sort(rng.uniform(0, 100, 100))
    >>> result = compute_egocentric_rate(
    ...     None, spike_times, times, positions, headings, object_positions
    ... )

    >>> # Access fields
    >>> result.firing_rate.shape
    (120,)
    >>> result.distance_range
    (0.0, 50.0)

    See Also
    --------
    EgocentricRatesResult : Batch version for multiple neurons
    compute_egocentric_rate : Function to compute this result
    """

    firing_rate: ArrayLike
    occupancy: ArrayLike
    env: EgocentricPolarEnvironment
    distance_range: tuple[float, float]
    n_distance_bins: int
    n_direction_bins: int

    @property
    def _bin_centers(self) -> NDArray[np.float64]:
        # Override SpatialResultMixin: egocentric results index polar bins
        # via env, not a world-coordinate Environment.
        bin_centers: NDArray[np.float64] = self.env.bin_centers
        return bin_centers

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot the egocentric rate map (firing rate by distance/direction).

        Delegates to the egocentric environment's plot_field method for
        consistent visualization across the codebase.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure and axes.
        **kwargs
            Additional keyword arguments passed to env.plot_field().
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
        >>> ax = result.plot()  # doctest: +SKIP
        >>> plt.show()  # doctest: +SKIP

        >>> fig, ax = plt.subplots()  # doctest: +SKIP
        >>> result.plot(ax=ax, cmap="hot", vmax=20.0)  # doctest: +SKIP

        See Also
        --------
        preferred_distance : Get distance component of peak response
        preferred_direction : Get direction component of peak response
        """
        return self.env.plot_field(_to_numpy(self.firing_rate), ax=ax, **kwargs)

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
        >>> import numpy as np
        >>> from neurospatial.encoding.egocentric import compute_egocentric_rate
        >>> rng = np.random.default_rng(0)
        >>> times = np.linspace(0, 100, 1000)
        >>> positions = rng.uniform(10, 90, (1000, 2))
        >>> headings = rng.uniform(-np.pi, np.pi, 1000)
        >>> object_positions = np.array([[50.0, 50.0]])
        >>> spike_times = np.sort(rng.uniform(0, 100, 100))
        >>> result = compute_egocentric_rate(
        ...     None, spike_times, times, positions, headings, object_positions
        ... )
        >>> dist = result.preferred_distance()
        >>> print(f"Preferred distance: {dist:.1f} cm")
        Preferred distance: 2.5 cm

        See Also
        --------
        preferred_direction : Get direction component of peak response
        plot : Visualize the egocentric rate map
        """
        firing_rate = _to_numpy(self.firing_rate)
        peak_bin = np.nanargmax(firing_rate)
        bin_centers: NDArray[np.float64] = self.env.bin_centers
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
        >>> import numpy as np
        >>> from neurospatial.encoding.egocentric import compute_egocentric_rate
        >>> rng = np.random.default_rng(0)
        >>> times = np.linspace(0, 100, 1000)
        >>> positions = rng.uniform(10, 90, (1000, 2))
        >>> headings = rng.uniform(-np.pi, np.pi, 1000)
        >>> object_positions = np.array([[50.0, 50.0]])
        >>> spike_times = np.sort(rng.uniform(0, 100, 100))
        >>> result = compute_egocentric_rate(
        ...     None, spike_times, times, positions, headings, object_positions
        ... )
        >>> direction = result.preferred_direction()
        >>> print(f"Preferred direction: {np.degrees(direction):.1f}")
        Preferred direction: -45.0

        See Also
        --------
        preferred_distance : Get distance component of peak response
        plot : Visualize the egocentric rate map
        """
        firing_rate = _to_numpy(self.firing_rate)
        peak_bin = np.nanargmax(firing_rate)
        bin_centers: NDArray[np.float64] = self.env.bin_centers
        return float(bin_centers[peak_bin, 1])

    def egocentric_spatial_information(self) -> float:
        """Compute egocentric spatial information (bits per spike).

        Egocentric spatial information quantifies how much information each
        spike conveys about the animal's egocentric position relative to an
        object. This uses the Skaggs spatial information formula with the
        egocentric occupancy.

        Returns
        -------
        float
            Egocentric spatial information in bits per spike. Returns 0.0
            for uniform firing (no spatial selectivity).

        Notes
        -----
        **Formula (Skaggs et al. 1993)**:

        .. math::

            I = \\sum_i p_i \\frac{r_i}{\\bar{r}} \\log_2 \\left( \\frac{r_i}{\\bar{r}} \\right)

        where :math:`p_i` is occupancy probability in egocentric bin :math:`i`,
        :math:`r_i` is firing rate in that bin, and :math:`\\bar{r}` is mean
        firing rate.

        **Interpretation**:

        - Object-vector cells typically have 0.5-2.0+ bits/spike
        - Higher values indicate more selective tuning to distance/direction
        - Zero means uniform firing (no egocentric selectivity)

        This metric uses the egocentric occupancy (time spent at each
        distance/direction combination), which differs from standard spatial
        information that uses allocentric position occupancy.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.egocentric import compute_egocentric_rate
        >>> rng = np.random.default_rng(0)
        >>> times = np.linspace(0, 100, 1000)
        >>> positions = rng.uniform(10, 90, (1000, 2))
        >>> headings = rng.uniform(-np.pi, np.pi, 1000)
        >>> object_positions = np.array([[50.0, 50.0]])
        >>> spike_times = np.sort(rng.uniform(0, 100, 100))
        >>> result = compute_egocentric_rate(
        ...     None, spike_times, times, positions, headings, object_positions
        ... )
        >>> info = result.egocentric_spatial_information()
        >>> print(f"Egocentric spatial info: {info:.2f} bits/spike")
        Egocentric spatial info: 0.83 bits/spike

        See Also
        --------
        is_object_vector_cell : Classify as object-vector cell based on this metric
        """
        from neurospatial.encoding._metrics import spatial_information

        firing_rate = _to_numpy(self.firing_rate)
        occupancy = _to_numpy(self.occupancy)
        return spatial_information(firing_rate, occupancy)

    def is_object_vector_cell(self, min_info: float = 0.3) -> bool:
        """Classify as object-vector cell based on egocentric spatial information.

        A neuron is classified as an object-vector cell (OVC) if its egocentric
        spatial information exceeds the minimum threshold. OVCs fire when the
        animal is at a specific distance and direction from an object.

        Parameters
        ----------
        min_info : float, default=0.3
            Minimum egocentric spatial information threshold in bits/spike.

            **How was 0.3 chosen?**

            This threshold is based on values reported in rodent entorhinal
            cortex studies (Hoydal et al., 2019). The lower threshold compared
            to spatial view cells (0.5) reflects that:

            - Egocentric polar coordinates have sparser sampling
            - Object-vector fields can be broader than place fields
            - The information calculation is sensitive to bin count

            Empirically:

            - Strong OVCs: 0.5-1.5+ bits/spike
            - Moderate OVCs: 0.3-0.5 bits/spike
            - Weak/non-OVCs: < 0.3 bits/spike

            **When to adjust:**

            - Different brain regions: May need 0.2-0.5
            - Different bin counts: Fewer bins → higher info, adjust accordingly
            - Noisy recordings: Consider 0.2 (more permissive)
            - Publication quality: Use 0.5 or higher (more conservative)

        Returns
        -------
        bool
            True if egocentric_spatial_information() > min_info, False otherwise.

        Notes
        -----
        **Object-vector vs place cells**: Both may show high spatial
        information, but OVCs have higher *egocentric* spatial information
        (using egocentric occupancy relative to objects) than *allocentric*
        spatial information (using standard position occupancy).

        For more rigorous classification, consider also using:

        - Stability across sessions
        - Multiple objects (OVCs should generalize)
        - Shuffling controls

        References
        ----------
        .. [1] Hoydal, O. A., et al. (2019). Object-vector coding in the medial
               entorhinal cortex. Nature, 568(7752), 400-404.
        .. [2] Deshmukh, S. S., & Knierim, J. J. (2011). Representation of
               non-spatial and spatial information in the lateral entorhinal
               cortex. Frontiers in Behavioral Neuroscience, 5, 69.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.egocentric import compute_egocentric_rate
        >>> rng = np.random.default_rng(0)
        >>> times = np.linspace(0, 100, 1000)
        >>> positions = rng.uniform(10, 90, (1000, 2))
        >>> headings = rng.uniform(-np.pi, np.pi, 1000)
        >>> object_positions = np.array([[50.0, 50.0]])
        >>> spike_times = np.sort(rng.uniform(0, 100, 100))
        >>> result = compute_egocentric_rate(
        ...     None, spike_times, times, positions, headings, object_positions
        ... )
        >>> result.is_object_vector_cell()
        True
        >>> result.is_object_vector_cell(min_info=0.5)
        True

        See Also
        --------
        egocentric_spatial_information : Compute the metric used for classification
        """
        return self.egocentric_spatial_information() > min_info


@dataclass(frozen=True)
class EgocentricRatesResult(SpatialResultMixin):
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
    env : Environment
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
    env : Environment
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

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.egocentric import (
    ...     EgocentricRateResult,
    ...     compute_egocentric_rates,
    ... )

    >>> # Build a batch result from a small, seeded trajectory
    >>> rng = np.random.default_rng(0)
    >>> times = np.linspace(0, 100, 1000)
    >>> positions = rng.uniform(10, 90, (1000, 2))
    >>> headings = rng.uniform(-np.pi, np.pi, 1000)
    >>> object_positions = np.array([[50.0, 50.0]])
    >>> spike_times = [
    ...     np.sort(rng.uniform(0, 100, 100)),
    ...     np.sort(rng.uniform(0, 100, 150)),
    ...     np.sort(rng.uniform(0, 100, 50)),
    ... ]
    >>> result = compute_egocentric_rates(
    ...     None, spike_times, times, positions, headings, object_positions
    ... )

    >>> # Access fields
    >>> len(result)
    3
    >>> isinstance(result[0], EgocentricRateResult)  # First neuron
    True

    >>> # Iterate over neurons
    >>> rates = [float(single.firing_rate.max()) for single in result]
    >>> len(rates)
    3

    See Also
    --------
    EgocentricRateResult : Single-neuron version
    compute_egocentric_rates : Function to compute this result
    """

    firing_rates: ArrayLike
    occupancy: ArrayLike
    env: EgocentricPolarEnvironment
    distance_range: tuple[float, float]
    n_distance_bins: int
    n_direction_bins: int

    @property
    def _bin_centers(self) -> NDArray[np.float64]:
        # Override SpatialResultMixin: egocentric results index polar bins
        # via env, not a world-coordinate Environment.
        bin_centers: NDArray[np.float64] = self.env.bin_centers
        return bin_centers

    def __len__(self) -> int:
        """Return the number of neurons.

        Returns
        -------
        int
            Number of neurons in the batch.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.egocentric import compute_egocentric_rates
        >>> rng = np.random.default_rng(0)
        >>> times = np.linspace(0, 100, 1000)
        >>> positions = rng.uniform(10, 90, (1000, 2))
        >>> headings = rng.uniform(-np.pi, np.pi, 1000)
        >>> object_positions = np.array([[50.0, 50.0]])
        >>> spike_times = [
        ...     np.sort(rng.uniform(0, 100, 100)),
        ...     np.sort(rng.uniform(0, 100, 150)),
        ...     np.sort(rng.uniform(0, 100, 50)),
        ... ]
        >>> result = compute_egocentric_rates(
        ...     None, spike_times, times, positions, headings, object_positions
        ... )
        >>> len(result)
        3
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
        >>> import numpy as np
        >>> from neurospatial.encoding.egocentric import compute_egocentric_rates
        >>> rng = np.random.default_rng(0)
        >>> times = np.linspace(0, 100, 1000)
        >>> positions = rng.uniform(10, 90, (1000, 2))
        >>> headings = rng.uniform(-np.pi, np.pi, 1000)
        >>> object_positions = np.array([[50.0, 50.0]])
        >>> spike_times = [
        ...     np.sort(rng.uniform(0, 100, 100)),
        ...     np.sort(rng.uniform(0, 100, 150)),
        ... ]
        >>> result = compute_egocentric_rates(
        ...     None, spike_times, times, positions, headings, object_positions
        ... )
        >>> single = result[0]
        >>> isinstance(single, EgocentricRateResult)
        True
        """
        return EgocentricRateResult(
            firing_rate=self.firing_rates[idx],  # type: ignore[index]
            occupancy=self.occupancy,
            env=self.env,
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
        >>> import numpy as np
        >>> from neurospatial.encoding.egocentric import compute_egocentric_rates
        >>> rng = np.random.default_rng(0)
        >>> times = np.linspace(0, 100, 1000)
        >>> positions = rng.uniform(10, 90, (1000, 2))
        >>> headings = rng.uniform(-np.pi, np.pi, 1000)
        >>> object_positions = np.array([[50.0, 50.0]])
        >>> spike_times = [
        ...     np.sort(rng.uniform(0, 100, 100)),
        ...     np.sort(rng.uniform(0, 100, 150)),
        ... ]
        >>> result = compute_egocentric_rates(
        ...     None, spike_times, times, positions, headings, object_positions
        ... )
        >>> peaks = [float(single.firing_rate.max()) for single in result]
        >>> len(peaks)
        2
        """
        for i in range(len(self)):
            yield self[i]

    def plot(self, idx: int, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot the egocentric rate map for a specific neuron.

        Delegates to the egocentric environment's plot_field method for
        consistent visualization across the codebase.

        Parameters
        ----------
        idx : int
            Index of the neuron to plot (0-indexed).
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure and axes.
        **kwargs
            Additional keyword arguments passed to env.plot_field().
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
        >>> # Plot the first neuron's egocentric rate map
        >>> ax = result.plot(idx=0)  # doctest: +SKIP
        >>> plt.show()  # doctest: +SKIP

        >>> # Plot neuron 3 with custom colormap
        >>> fig, ax = plt.subplots()  # doctest: +SKIP
        >>> result.plot(idx=3, ax=ax, cmap="hot", vmax=20.0)  # doctest: +SKIP

        See Also
        --------
        preferred_distances : Get distance preferences for all neurons
        EgocentricRateResult.plot : Plot for single-neuron result
        """
        return self.env.plot_field(
            _to_numpy(self.firing_rates[idx]),  # type: ignore[index]
            ax=ax,
            **kwargs,
        )

    def preferred_distances(self) -> NDArray[np.float64]:
        """Preferred distances to object for all neurons.

        Returns the distance component (first dimension) of the egocentric
        bin where each neuron shows maximum firing rate.

        Returns
        -------
        ndarray, shape (n_neurons,)
            Distance to object at peak firing rate for each neuron, in the
            same units as the environment (typically cm).

        Notes
        -----
        For object-vector cells, this represents the preferred distance to
        the object. A cell with preferred_distance=20 fires most when the
        object is 20 cm away from the animal.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.egocentric import compute_egocentric_rates
        >>> rng = np.random.default_rng(0)
        >>> times = np.linspace(0, 100, 1000)
        >>> positions = rng.uniform(10, 90, (1000, 2))
        >>> headings = rng.uniform(-np.pi, np.pi, 1000)
        >>> object_positions = np.array([[50.0, 50.0]])
        >>> spike_times = [
        ...     np.sort(rng.uniform(0, 100, 100)),
        ...     np.sort(rng.uniform(0, 100, 150)),
        ...     np.sort(rng.uniform(0, 100, 50)),
        ... ]
        >>> result = compute_egocentric_rates(
        ...     None, spike_times, times, positions, headings, object_positions
        ... )
        >>> distances = result.preferred_distances()
        >>> distances.shape
        (3,)
        >>> print(f"Neuron 0 prefers distance: {distances[0]:.1f} cm")
        Neuron 0 prefers distance: 2.5 cm

        See Also
        --------
        EgocentricRateResult.preferred_distance : Single-neuron version
        preferred_directions : Get direction preferences for all neurons
        """
        firing_rates = _to_numpy(self.firing_rates)
        bin_centers: NDArray[np.float64] = self.env.bin_centers

        # Peak (max-firing) bin per neuron, ignoring NaNs; then read off the
        # distance component (column 0 of bin_centers).
        peak_idx = np.nanargmax(firing_rates, axis=1)
        distances: NDArray[np.float64] = bin_centers[peak_idx, 0]
        return distances

    def preferred_directions(self) -> NDArray[np.float64]:
        """Preferred directions to object for all neurons.

        Returns the direction component (second dimension) of the egocentric
        bin where each neuron shows maximum firing rate. Direction is in
        radians using the egocentric coordinate convention.

        Returns
        -------
        ndarray, shape (n_neurons,)
            Direction to object at peak firing rate for each neuron, in radians.
            - 0 = object is directly ahead of animal
            - +π/2 = object is to the left
            - -π/2 = object is to the right
            - ±π = object is behind

        Notes
        -----
        For object-vector cells, this represents the preferred direction to
        the object relative to the animal's heading.

        **Coordinate convention**: This uses egocentric (animal-centered)
        coordinates, NOT allocentric (world-centered) coordinates.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.egocentric import compute_egocentric_rates
        >>> rng = np.random.default_rng(0)
        >>> times = np.linspace(0, 100, 1000)
        >>> positions = rng.uniform(10, 90, (1000, 2))
        >>> headings = rng.uniform(-np.pi, np.pi, 1000)
        >>> object_positions = np.array([[50.0, 50.0]])
        >>> spike_times = [
        ...     np.sort(rng.uniform(0, 100, 100)),
        ...     np.sort(rng.uniform(0, 100, 150)),
        ...     np.sort(rng.uniform(0, 100, 50)),
        ... ]
        >>> result = compute_egocentric_rates(
        ...     None, spike_times, times, positions, headings, object_positions
        ... )
        >>> directions = result.preferred_directions()
        >>> directions.shape
        (3,)
        >>> print(f"Neuron 0 prefers direction: {np.degrees(directions[0]):.1f}")
        Neuron 0 prefers direction: -45.0

        See Also
        --------
        EgocentricRateResult.preferred_direction : Single-neuron version
        preferred_distances : Get distance preferences for all neurons
        """
        firing_rates = _to_numpy(self.firing_rates)
        bin_centers: NDArray[np.float64] = self.env.bin_centers

        # Peak (max-firing) bin per neuron, ignoring NaNs; then read off the
        # direction component (column 1 of bin_centers).
        peak_idx = np.nanargmax(firing_rates, axis=1)
        directions: NDArray[np.float64] = bin_centers[peak_idx, 1]
        return directions

    def egocentric_spatial_information(self) -> NDArray[np.float64]:
        """Egocentric spatial information for all neurons (bits per spike).

        Quantifies how much information each spike conveys about the animal's
        egocentric position relative to an object for each neuron.

        Returns
        -------
        ndarray, shape (n_neurons,)
            Egocentric spatial information in bits/spike for each neuron.
            Always non-negative. Returns 0.0 for uniform firing.

        Notes
        -----
        Uses the Skaggs et al. (1993) formula with **egocentric occupancy**.
        This is computed by delegating to the batch spatial information
        function in ``_metrics.py``.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.egocentric import compute_egocentric_rates
        >>> rng = np.random.default_rng(0)
        >>> times = np.linspace(0, 100, 1000)
        >>> positions = rng.uniform(10, 90, (1000, 2))
        >>> headings = rng.uniform(-np.pi, np.pi, 1000)
        >>> object_positions = np.array([[50.0, 50.0]])
        >>> spike_times = [
        ...     np.sort(rng.uniform(0, 100, 100)),
        ...     np.sort(rng.uniform(0, 100, 150)),
        ...     np.sort(rng.uniform(0, 100, 50)),
        ... ]
        >>> result = compute_egocentric_rates(
        ...     None, spike_times, times, positions, headings, object_positions
        ... )
        >>> info = result.egocentric_spatial_information()
        >>> info.shape
        (3,)
        >>> print(f"Neuron with highest info: {np.argmax(info)}")
        Neuron with highest info: 2

        See Also
        --------
        EgocentricRateResult.egocentric_spatial_information : Single-neuron version
        detect_ovcs : Classify neurons based on this metric
        """
        from neurospatial.encoding._metrics import batch_spatial_information

        return batch_spatial_information(
            _to_numpy(self.firing_rates), _to_numpy(self.occupancy)
        )

    def detect_ovcs(self, min_info: float = 0.3) -> NDArray[np.bool_]:
        """Classify neurons as object-vector cells.

        A neuron is classified as an object-vector cell (OVC) if its egocentric
        spatial information exceeds the minimum threshold.

        Parameters
        ----------
        min_info : float, default=0.3
            Minimum egocentric spatial information threshold in bits/spike.
            See EgocentricRateResult.is_object_vector_cell() for threshold rationale.

        Returns
        -------
        ndarray, shape (n_neurons,)
            Boolean array where True indicates the neuron is classified
            as an object-vector cell.

        Notes
        -----
        Uses vectorized computation of egocentric_spatial_information() for
        efficiency with large populations.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.egocentric import compute_egocentric_rates
        >>> rng = np.random.default_rng(0)
        >>> times = np.linspace(0, 100, 1000)
        >>> positions = rng.uniform(10, 90, (1000, 2))
        >>> headings = rng.uniform(-np.pi, np.pi, 1000)
        >>> object_positions = np.array([[50.0, 50.0]])
        >>> spike_times = [
        ...     np.sort(rng.uniform(0, 100, 100)),
        ...     np.sort(rng.uniform(0, 100, 150)),
        ...     np.sort(rng.uniform(0, 100, 50)),
        ... ]
        >>> result = compute_egocentric_rates(
        ...     None, spike_times, times, positions, headings, object_positions
        ... )
        >>> is_object_vector_cell = result.detect_ovcs()
        >>> print(f"Found {is_object_vector_cell.sum()} OVCs")
        Found 3 OVCs

        >>> # Use stricter threshold
        >>> is_object_vector_cell = result.detect_ovcs(min_info=0.5)

        See Also
        --------
        EgocentricRateResult.is_object_vector_cell : Single-neuron classification
        egocentric_spatial_information : The metric used for classification
        """
        info = self.egocentric_spatial_information()
        return info > min_info

    def to_dataframe(
        self,
        neuron_ids: Sequence[str | int] | None = None,
    ) -> pd.DataFrame:
        """Export metrics to DataFrame for exploratory analysis.

        Computes all egocentric metrics and exports them to a pandas DataFrame
        for easy filtering, sorting, and analysis.

        Parameters
        ----------
        neuron_ids : sequence of str or int, optional
            Identifiers for each neuron. If None, uses integer indices
            (0, 1, 2, ..., n_neurons-1).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:

            - neuron_id: identifier for each neuron
            - preferred_distance: preferred distance to object (cm)
            - preferred_direction: preferred direction to object (radians, 0=ahead)
            - preferred_direction_deg: preferred direction (degrees)
            - peak_rate: maximum firing rate (Hz)
            - is_object_vector_cell: whether classified as OVC (using default threshold)

        Raises
        ------
        ValueError
            If neuron_ids has a different length than the number of neurons.

        Notes
        -----
        This method computes all metrics at once, which may be slow for
        large populations. For selective metric computation, use the
        individual methods (``preferred_distances()``, ``detect_ovcs()``, etc.).

        **Common pandas workflows**:

        - Filter: ``df[df["is_object_vector_cell"] == True]``
        - Sort: ``df.sort_values("preferred_distance")``
        - Top-N: ``df.nlargest(10, "peak_rate")``

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.egocentric import compute_egocentric_rates
        >>> rng = np.random.default_rng(0)
        >>> times = np.linspace(0, 100, 1000)
        >>> positions = rng.uniform(10, 90, (1000, 2))
        >>> headings = rng.uniform(-np.pi, np.pi, 1000)
        >>> object_positions = np.array([[50.0, 50.0]])
        >>> spike_times = [
        ...     np.sort(rng.uniform(0, 100, 100)),
        ...     np.sort(rng.uniform(0, 100, 150)),
        ...     np.sort(rng.uniform(0, 100, 50)),
        ... ]
        >>> result = compute_egocentric_rates(
        ...     None, spike_times, times, positions, headings, object_positions
        ... )
        >>> df = result.to_dataframe()
        >>> list(df.columns)
        ['neuron_id', 'preferred_distance', 'preferred_direction', 'preferred_direction_deg', 'peak_rate', 'is_object_vector_cell']
        >>> len(df)
        3

        >>> # Filter for OVCs only
        >>> ovcs = df[df["is_object_vector_cell"]]

        >>> # Sort by preferred distance
        >>> sorted_df = df.sort_values("preferred_distance")

        >>> # Custom neuron identifiers
        >>> df = result.to_dataframe(neuron_ids=["unit_0", "unit_1", "unit_2"])
        >>> list(df["neuron_id"])
        ['unit_0', 'unit_1', 'unit_2']

        See Also
        --------
        detect_ovcs : OVC classification
        preferred_distances : Batch preferred distance computation
        preferred_directions : Batch preferred direction computation
        """
        import pandas as pd

        n_neurons = len(self)

        # Validate and convert neuron_ids
        if neuron_ids is None:
            neuron_ids_list: list[str | int] = list(range(n_neurons))
        else:
            neuron_ids_list = list(neuron_ids)
            if len(neuron_ids_list) != n_neurons:
                raise ValueError(
                    f"neuron_ids has {len(neuron_ids_list)} elements but "
                    f"result contains {n_neurons} neurons"
                )

        # Compute all metrics
        pref_dists = self.preferred_distances()
        pref_dirs = self.preferred_directions()
        peak_rates = self.peak_firing_rate()
        is_object_vector_cell = self.detect_ovcs()

        # Build DataFrame
        data: dict[str, Any] = {
            "neuron_id": neuron_ids_list,
            "preferred_distance": pref_dists,
            "preferred_direction": pref_dirs,
            "preferred_direction_deg": np.degrees(pref_dirs),
            "peak_rate": peak_rates,
            "is_object_vector_cell": is_object_vector_cell,
        }

        return pd.DataFrame(data)


def _raw_polar_rate(
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    min_occupancy: float,
) -> NDArray[np.float64]:
    """Raw firing rate (spikes / occupancy) for an egocentric polar grid.

    Graph-diffusion smoothing on the polar environment bleeds rate across
    *distance* rings (the env connects adjacent distance bins radially), which
    erases the distance tuning object-vector cells encode. The ``binned``
    method therefore computes the bin rate directly, with no graph smoothing.

    Parameters
    ----------
    spike_counts : ndarray of shape (n_bins,), dtype float64
        Spike counts per polar bin.
    occupancy : ndarray of shape (n_bins,), dtype float64
        Time spent in each polar bin, in seconds.
    min_occupancy : float
        Bins with occupancy below this value are treated as unvisited.

    Returns
    -------
    ndarray of shape (n_bins,), dtype float64
        Firing rate per bin in Hz. Bins whose occupancy does not exceed the
        threshold are NaN (undefined, not zero).

    Notes
    -----
    Masking convention (shared across the encoding smoothing paths): a bin is
    valid iff the occupancy quantity used as the firing-rate denominator is
    *strictly greater than* ``max(min_occupancy, 0.0)``. Here the denominator
    is the raw per-bin occupancy (this is the unsmoothed ``binned`` polar
    path), so the raw occupancy is thresholded. When ``min_occupancy`` is 0
    (the default) this reduces to "valid iff ``occupancy > 0``", matching the
    smoothed-density threshold used by the KDE paths in ``_smoothing.py``.
    """
    occ = np.asarray(occupancy, dtype=np.float64)
    counts = np.asarray(spike_counts, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = counts / occ
    occupancy_threshold = max(min_occupancy, 0.0)
    valid = occ > occupancy_threshold
    return np.where(valid, rate, np.nan)


def _egocentric_firing_rate(
    polar_env: EgocentricPolarEnvironment,
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    *,
    smoothing_method: Literal["diffusion_kde", "gaussian_kde", "binned"],
    bandwidth: float,
    min_occupancy: float,
    backend: Literal["numpy", "jax"],
) -> ArrayLike:
    """Egocentric polar firing rate: raw for ``binned``, smoothed otherwise.

    Parameters
    ----------
    polar_env : Environment
        The egocentric polar grid the rate is computed over.
    spike_counts : ndarray of shape (n_bins,), dtype float64
        Spike counts per polar bin.
    occupancy : ndarray of shape (n_bins,), dtype float64
        Time spent in each polar bin, in seconds.
    smoothing_method : {"diffusion_kde", "gaussian_kde", "binned"}
        ``"binned"`` returns the raw bin rate (see ``_raw_polar_rate``); the
        kernel methods smooth via ``smooth_rate_map``.
    bandwidth : float
        Smoothing bandwidth, in environment units. Unused for ``"binned"``.
    min_occupancy : float
        Bins with occupancy below this value are treated as unvisited.
    backend : {"numpy", "jax"}
        Array backend for the returned rate.

    Returns
    -------
    ArrayLike of shape (n_bins,)
        Firing rate per bin in Hz, as a NumPy or JAX array per ``backend``.
    """
    from neurospatial.encoding._backend import is_jax_available

    if smoothing_method == "binned":
        rate = _raw_polar_rate(spike_counts, occupancy, min_occupancy)
        if backend == "jax" and is_jax_available():
            import jax.numpy as jnp

            return jnp.asarray(rate, dtype=jnp.float64)
        return rate

    from neurospatial.encoding._smoothing import smooth_rate_map

    return smooth_rate_map(
        polar_env,
        spike_counts,
        occupancy,
        method=smoothing_method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        backend=backend,
    )


def compute_egocentric_rate(
    env: Environment | None,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    object_positions: NDArray[np.float64],
    *,
    distance_range: tuple[float, float] = (0.0, 50.0),
    n_distance_bins: int = 10,
    n_direction_bins: int = 12,
    metric: Literal["euclidean", "geodesic"] = "euclidean",
    smoothing_method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "binned",
    bandwidth: float = 5.0,
    min_occupancy: float = 0.0,
    backend: Literal["numpy", "jax", "auto"] = "numpy",
) -> EgocentricRateResult:
    """Compute egocentric firing rate for one neuron.

    This function computes a smoothed firing rate map in egocentric polar
    coordinates (distance and direction to nearest object). This is the key
    metric for identifying object-vector cells (OVCs).

    Parameters
    ----------
    env : Environment or None
        The allocentric environment. Required when
        ``metric="geodesic"`` (used to compute distances around
        obstacles). May be ``None`` when ``metric="euclidean"``.
    spike_times : ndarray, shape (n_spikes,)
        Times of spike events in seconds. Can be empty.
    times : ndarray, shape (n_samples,)
        Timestamps of trajectory samples in seconds.
    positions : ndarray, shape (n_samples, 2)
        Animal position coordinates at each time sample. NaN values (in
        positions or headings) are treated as missing data and excluded from
        occupancy and firing-rate computation; callers do not need to
        pre-filter tracking dropouts.
    headings : ndarray, shape (n_samples,)
        Head direction at each time sample (radians, **allocentric
        world-frame convention**: 0 = East, π/2 = North, π = West,
        -π/2 = South, wrapped to [-π, π]). The allocentric→egocentric
        transform is applied internally; pass world-frame headings, not
        animal-frame angles.
    object_positions : ndarray, shape (n_objects, 2)
        Object positions in allocentric coordinates. The firing rate is
        computed relative to the *nearest* object at each timepoint.
    distance_range : tuple of float, default=(0.0, 50.0)
        (min_distance, max_distance) for egocentric binning. Distances outside
        this range are not included in the rate map.
    n_distance_bins : int, default=10
        Number of distance bins in the egocentric polar grid.
    n_direction_bins : int, default=12
        Number of direction bins in the egocentric polar grid. Covers the
        full circle (-π to π).
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric for computing distance to objects:

        - **euclidean**: Straight-line distance.
        - **geodesic**: Path distance respecting environment boundaries.
          Requires ``env`` parameter.

    smoothing_method : {"diffusion_kde", "gaussian_kde", "binned"}, default="binned"
        Smoothing method to use:

        - **binned** (default): Raw rate computation without smoothing.
          Appropriate for egocentric polar grids where boundary-aware
          smoothing may not apply.
        - **diffusion_kde**: Graph-based boundary-aware KDE.
        - **gaussian_kde**: Standard Euclidean KDE.

    bandwidth : float, default=5.0
        Smoothing bandwidth for gaussian_kde and diffusion_kde methods.
    min_occupancy : float, default=0.0
        Minimum occupancy (seconds) for a bin to be included. Bins with
        occupancy below this threshold are set to NaN.
    backend : {'numpy', 'jax', 'auto'}, default='numpy'
        Computation backend.

        - 'numpy': Use NumPy (always available)
        - 'jax': Use JAX for rate computation (requires JAX installation)
        - 'auto': Use JAX if available, otherwise NumPy

    Returns
    -------
    EgocentricRateResult
        Result object containing:

        - ``firing_rate``: Firing rate by egocentric coordinates in Hz,
          shape (n_bins,)
        - ``occupancy``: Time in each egocentric bin in seconds,
          shape (n_bins,)
        - ``env``: The egocentric polar environment
        - ``distance_range``: Distance range used
        - ``n_distance_bins``: Number of distance bins
        - ``n_direction_bins``: Number of direction bins

    Raises
    ------
    ValueError
        If ``metric="geodesic"`` but ``env`` is None.
        If ``metric`` is not one of the valid options.
        If inputs have mismatched lengths.

    See Also
    --------
    compute_egocentric_rates : Batch version for multiple neurons
    EgocentricRateResult : Result class with convenience methods
    compute_spatial_rate : Standard spatial rate (by animal position)

    Notes
    -----
    The function uses the egocentric binning layer (``_egocentric_binning.py``)
    to convert spike times to spike counts based on distance and direction to
    the nearest object, then the smoothing layer (``_smoothing.py``) to compute
    the smoothed firing rate.

    **Algorithm**:

    1. Compute egocentric coordinates (distance, bearing) to nearest object
       at each behavioral frame
    2. Bin spikes by egocentric coordinates at spike time
    3. Compute egocentric occupancy (time spent at each distance/direction)
    4. Apply smoothing (method-dependent, see ``_smoothing.py``)

    **Coordinate convention**: Direction uses egocentric (animal-centered)
    coordinates where 0=ahead, +π/2=left, -π/2=right.

    **Place cells vs object-vector cells**: For place cells, firing is
    determined by allocentric position. For OVCs, firing is determined by
    egocentric relationship to objects. Computing place field using
    ``compute_spatial_rate`` and OVC field using this function, then comparing
    spatial information, can help distinguish cell types.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.egocentric import compute_egocentric_rate

    >>> # Create trajectory and objects
    >>> rng = np.random.default_rng(42)
    >>> times = np.linspace(0, 100, 1000)
    >>> positions = rng.uniform(10, 90, (1000, 2))
    >>> headings = rng.uniform(-np.pi, np.pi, 1000)
    >>> object_positions = np.array([[50.0, 50.0], [25.0, 75.0]])
    >>> spike_times = np.sort(rng.uniform(0, 100, 100))

    >>> # Compute egocentric rate
    >>> result = compute_egocentric_rate(
    ...     None,
    ...     spike_times,
    ...     times,
    ...     positions,
    ...     headings,
    ...     object_positions,
    ... )

    >>> # Access results
    >>> result.firing_rate.shape
    (120,)
    >>> pref_dist = result.preferred_distance()
    >>> pref_dir = result.preferred_direction()
    >>> info = result.egocentric_spatial_information()
    >>> is_object_vector_cell = result.is_object_vector_cell()

    >>> # Plot the egocentric rate map
    >>> ax = result.plot()  # doctest: +SKIP

    References
    ----------
    .. [1] Hoydal, O. A., et al. (2019). Object-vector coding in the medial
           entorhinal cortex. Nature, 568(7752), 400-404.
    """
    from neurospatial.encoding._backend import (
        SUPPORTED_BACKENDS,
        get_backend_name,
        is_jax_available,
    )
    from neurospatial.encoding._egocentric_binning import (
        bin_egocentric_spike_trains,
        normalize_object_positions,
    )
    from neurospatial.encoding._smoothing import (
        _validate_smoothing_parameters,
    )
    from neurospatial.encoding._validation import (
        validate_env_fitted,
        validate_spike_times,
        validate_trajectory,
    )

    # `env` is optional in this function (None is permitted with the
    # euclidean distance metric, since geodesic distance is the only
    # path that needs an env-derived graph). Only validate fitted-state
    # if the user supplied an env; the geodesic path raises its own
    # explicit error a few lines below if env is None.
    if env is not None:
        validate_env_fitted(env, context="compute_egocentric_rate")

    # Validate backend
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            f"Supported backends are: {', '.join(repr(b) for b in SUPPORTED_BACKENDS)}"
        )

    # Resolve backend (handles "auto" → "numpy" or "jax")
    # This raises ImportError if backend="jax" and JAX is unavailable
    resolved_backend = get_backend_name(backend)

    # Validate metric
    valid_metrics = {"euclidean", "geodesic"}
    if metric not in valid_metrics:
        raise ValueError(
            f"Invalid metric: '{metric}'. Must be one of {sorted(valid_metrics)}"
        )

    # Validate env requirement for geodesic
    if metric == "geodesic" and env is None:
        raise ValueError(
            "metric='geodesic' requires env parameter.\n"
            "Pass the allocentric environment to compute geodesic distances."
        )

    _validate_smoothing_parameters(smoothing_method, bandwidth)

    # Convert inputs to arrays (1D required for spike_times/times/headings)
    spike_times = np.asarray(spike_times, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)
    # Normalize object_positions: [x, y] -> [[x, y]] for single object
    object_positions = normalize_object_positions(object_positions)

    validate_trajectory(
        times, positions=positions, headings=headings, context="compute_egocentric_rate"
    )
    validate_spike_times(spike_times, context="compute_egocentric_rate")

    # Reuse the batch binning path for the single-neuron API so egocentric
    # coordinates are computed once and shared by spike counts and occupancy.
    # The third return is the *polar* env that indexes the (distance,
    # direction) bins; keep it under a distinct name so the cartesian
    # ``env`` parameter (used for geodesic distance, validated above)
    # is not shadowed mid-function.
    spike_counts_batch, occupancy, polar_env = bin_egocentric_spike_trains(
        [spike_times],
        times,
        positions,
        headings,
        object_positions,
        distance_range=distance_range,
        n_distance_bins=n_distance_bins,
        n_direction_bins=n_direction_bins,
        metric=metric,
        env=env,
        n_jobs=1,
    )
    spike_counts = spike_counts_batch[0]

    firing_rate = _egocentric_firing_rate(
        polar_env,
        spike_counts,
        occupancy,
        smoothing_method=smoothing_method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        backend=resolved_backend,
    )

    # Convert occupancy to JAX if JAX backend is selected
    # (firing_rate is already JAX from smooth_rate_map)
    if resolved_backend == "jax" and is_jax_available():
        import jax.numpy as jnp

        occupancy = jnp.asarray(occupancy, dtype=jnp.float64)  # type: ignore[assignment]

    # Return result
    return EgocentricRateResult(
        firing_rate=firing_rate,
        occupancy=occupancy,
        env=polar_env,
        distance_range=distance_range,
        n_distance_bins=n_distance_bins,
        n_direction_bins=n_direction_bins,
    )


def compute_egocentric_rates(
    env: Environment | None,
    spike_times: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    object_positions: NDArray[np.float64],
    *,
    distance_range: tuple[float, float] = (0.0, 50.0),
    n_distance_bins: int = 10,
    n_direction_bins: int = 12,
    metric: Literal["euclidean", "geodesic"] = "euclidean",
    smoothing_method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "binned",
    bandwidth: float = 5.0,
    min_occupancy: float = 0.0,
    n_jobs: int = 1,
    backend: Literal["numpy", "jax", "auto"] = "numpy",
) -> EgocentricRatesResult:
    """Compute egocentric firing rates for multiple neurons.

    This is the batch version of ``compute_egocentric_rate(None)`` that efficiently
    processes multiple neurons with shared trajectory data. It precomputes
    shared quantities (egocentric coordinates, occupancy) once and optionally
    parallelizes spike counting with joblib.

    Parameters
    ----------
    env : Environment or None
        The allocentric environment. Required when
        ``metric="geodesic"`` (used to compute distances around
        obstacles). May be ``None`` when ``metric="euclidean"``.
    spike_times : sequence of arrays or 2D array
        Spike times for each neuron. Accepted formats:

        - List/tuple of 1D arrays: ``[spikes_0, spikes_1, ...]`` (canonical)
        - 2D array with NaN padding: shape ``(n_neurons, max_spikes)``
        - 1D array (single neuron): wrapped in list automatically

        All formats are normalized via ``normalize_spike_times()``.
    times : ndarray, shape (n_samples,)
        Timestamps of trajectory samples in seconds.
    positions : ndarray, shape (n_samples, 2)
        Animal position coordinates at each time sample. NaN values (in
        positions or headings) are treated as missing data and excluded from
        occupancy and firing-rate computation; callers do not need to
        pre-filter tracking dropouts.
    headings : ndarray, shape (n_samples,)
        Head direction at each time sample (radians, **allocentric
        world-frame convention**: 0 = East, π/2 = North, π = West,
        -π/2 = South, wrapped to [-π, π]). The allocentric→egocentric
        transform is applied internally; pass world-frame headings, not
        animal-frame angles.
    object_positions : ndarray, shape (n_objects, 2)
        Object positions in allocentric coordinates. The firing rate is
        computed relative to the *nearest* object at each timepoint.
    distance_range : tuple of float, default=(0.0, 50.0)
        (min_distance, max_distance) for egocentric binning. Distances outside
        this range are not included in the rate map.
    n_distance_bins : int, default=10
        Number of distance bins in the egocentric polar grid.
    n_direction_bins : int, default=12
        Number of direction bins in the egocentric polar grid. Covers the
        full circle (-pi to pi).
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric for computing distance to objects:

        - **euclidean**: Straight-line distance.
        - **geodesic**: Path distance respecting environment boundaries.
          Requires ``env`` parameter.

    smoothing_method : {"diffusion_kde", "gaussian_kde", "binned"}, default="binned"
        Smoothing method to use:

        - **binned** (default): Raw rate computation without smoothing.
          Appropriate for egocentric polar grids where boundary-aware
          smoothing may not apply.
        - **diffusion_kde**: Graph-based boundary-aware KDE.
        - **gaussian_kde**: Standard Euclidean KDE.

    bandwidth : float, default=5.0
        Smoothing bandwidth for gaussian_kde and diffusion_kde methods.
    min_occupancy : float, default=0.0
        Minimum occupancy (seconds) for a bin to be included. Bins with
        occupancy below this threshold are set to NaN.
    n_jobs : int, default=1
        Number of parallel jobs for spike counting. Use -1 for all CPUs.
        1 means sequential processing (no parallelization overhead).
    backend : {'numpy', 'jax', 'auto'}, default='numpy'
        Computation backend.

        - 'numpy': Use NumPy (always available)
        - 'jax': Use JAX for rate computation (requires JAX installation)
        - 'auto': Use JAX if available, otherwise NumPy

    Returns
    -------
    EgocentricRatesResult
        Result object containing:

        - ``firing_rates``: Firing rate maps, shape ``(n_neurons, n_bins)``
        - ``occupancy``: Time in each egocentric bin in seconds, shape ``(n_bins,)``
        - ``env``: The egocentric polar environment
        - ``distance_range``: Distance range used
        - ``n_distance_bins``: Number of distance bins
        - ``n_direction_bins``: Number of direction bins

        The result supports iteration: ``for single in result: ...``
        and indexing: ``single = result[0]``.

    Raises
    ------
    ValueError
        If ``metric="geodesic"`` but ``env`` is None.
        If ``metric`` is not one of the valid options.
        If inputs have mismatched lengths.

    See Also
    --------
    compute_egocentric_rate : Single-neuron version
    EgocentricRatesResult : Result class with batch methods
    compute_spatial_rates : Standard spatial rates (by animal position)

    Notes
    -----
    **Efficiency advantages over calling ``compute_egocentric_rate(None)`` in a loop**:

    1. Egocentric coordinates (distance, bearing to nearest object) are
       computed once and shared across all neurons
    2. Occupancy is computed once and shared
    3. Diffusion kernel (for ``diffusion_kde`` method) is computed once
    4. Spike binning can be parallelized with joblib

    **When to use batch vs single**:

    - **Batch** (this function): Processing 3+ neurons, or any case where
      efficiency matters. The overhead of precomputing shared quantities
      is amortized over multiple neurons.
    - **Single** (``compute_egocentric_rate``): Processing 1-2 neurons, or when
      you need fine-grained control over individual neurons.

    **Coordinate convention**: Direction uses egocentric (animal-centered)
    coordinates where 0=ahead, +pi/2=left, -pi/2=right.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.egocentric import compute_egocentric_rates

    >>> # Create trajectory and objects
    >>> rng = np.random.default_rng(42)
    >>> times = np.linspace(0, 100, 1000)
    >>> positions = rng.uniform(10, 90, (1000, 2))
    >>> headings = rng.uniform(-np.pi, np.pi, 1000)
    >>> object_positions = np.array([[50.0, 50.0], [25.0, 75.0]])

    >>> # Spike times for 3 neurons
    >>> spike_times = [
    ...     np.sort(rng.uniform(0, 100, 100)),  # Neuron 0
    ...     np.sort(rng.uniform(0, 100, 150)),  # Neuron 1
    ...     np.sort(rng.uniform(0, 100, 50)),  # Neuron 2
    ... ]

    >>> # Compute egocentric rates for all neurons
    >>> result = compute_egocentric_rates(
    ...     None,
    ...     spike_times,
    ...     times,
    ...     positions,
    ...     headings,
    ...     object_positions,
    ...     n_jobs=2,  # Parallel spike binning
    ... )

    >>> # Access results
    >>> print(f"Number of neurons: {len(result)}")
    Number of neurons: 3
    >>> print(f"Firing rates shape: {result.firing_rates.shape}")
    Firing rates shape: (3, 120)

    >>> # Iterate over neurons
    >>> for i, single in enumerate(result):
    ...     pref_dist = single.preferred_distance()
    ...     pref_dir = single.preferred_direction()
    ...     print(f"Neuron {i}: {pref_dist:.1f} cm at {np.degrees(pref_dir):.0f} deg")
    Neuron 0: 2.5 cm at 15 deg
    Neuron 1: 7.5 cm at 45 deg
    Neuron 2: 42.5 cm at -75 deg

    >>> # Get metrics for all neurons
    >>> df = result.to_dataframe()
    >>> len(df)
    3

    >>> # Use 2D array with NaN padding
    >>> spike_times_2d = np.array(
    ...     [
    ...         [0.1, 0.5, 1.0, np.nan],
    ...         [0.2, 0.3, 0.8, 1.2],
    ...     ]
    ... )
    >>> result2 = compute_egocentric_rates(
    ...     None, spike_times_2d, times, positions, headings, object_positions
    ... )
    >>> len(result2)
    2

    References
    ----------
    .. [1] Hoydal, O. A., et al. (2019). Object-vector coding in the medial
           entorhinal cortex. Nature, 568(7752), 400-404.
    """
    from neurospatial.encoding._backend import (
        SUPPORTED_BACKENDS,
        get_backend_name,
        is_jax_available,
    )
    from neurospatial.encoding._egocentric_binning import (
        bin_egocentric_spike_trains,
        normalize_object_positions,
    )
    from neurospatial.encoding._smoothing import (
        _validate_smoothing_parameters,
        smooth_rate_maps_batch,
    )
    from neurospatial.encoding._spikes import normalize_spike_times
    from neurospatial.encoding._validation import (
        validate_env_fitted,
        validate_spike_times,
        validate_trajectory,
    )

    # `env` is optional in this function (None is permitted with the
    # euclidean distance metric); only validate fitted-state when supplied.
    if env is not None:
        validate_env_fitted(env, context="compute_egocentric_rates")

    # Validate backend
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            f"Supported backends are: {', '.join(repr(b) for b in SUPPORTED_BACKENDS)}"
        )

    # Resolve backend (handles "auto" → "numpy" or "jax")
    # This raises ImportError if backend="jax" and JAX is unavailable
    resolved_backend = get_backend_name(backend)

    _validate_smoothing_parameters(smoothing_method, bandwidth)

    # Validate metric
    valid_metrics = {"euclidean", "geodesic"}
    if metric not in valid_metrics:
        raise ValueError(
            f"Invalid metric: '{metric}'. Must be one of {sorted(valid_metrics)}"
        )

    # Validate env requirement for geodesic
    if metric == "geodesic" and env is None:
        raise ValueError(
            "metric='geodesic' requires env parameter.\n"
            "Pass the allocentric environment to compute geodesic distances."
        )

    # Normalize spike times to canonical list-of-arrays format
    spike_times_list = normalize_spike_times(spike_times)
    n_neurons = len(spike_times_list)

    # Convert inputs to arrays (1D required for times/headings)
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)
    # Normalize object_positions: [x, y] -> [[x, y]] for single object
    object_positions = normalize_object_positions(object_positions)

    validate_trajectory(
        times,
        positions=positions,
        headings=headings,
        context="compute_egocentric_rates",
    )
    for i, st in enumerate(spike_times_list):
        validate_spike_times(st, context=f"compute_egocentric_rates (neuron {i})")

    # Handle edge case: no neurons
    if n_neurons == 0:
        # Still need to compute occupancy for consistency
        from neurospatial.encoding._egocentric_binning import (
            compute_egocentric_occupancy,
        )

        # The third return is the polar (distance, direction) env;
        # bind it under a distinct name to avoid shadowing the
        # cartesian ``env`` parameter (used for geodesic distance).
        occupancy, polar_env = compute_egocentric_occupancy(
            times,
            positions,
            headings,
            object_positions,
            distance_range=distance_range,
            n_distance_bins=n_distance_bins,
            n_direction_bins=n_direction_bins,
            metric=metric,
            env=env,
        )
        firing_rates_result: ArrayLike = np.empty(
            (0, polar_env.n_bins), dtype=np.float64
        )
        if resolved_backend == "jax" and is_jax_available():
            import jax.numpy as jnp

            firing_rates_result = jnp.asarray(firing_rates_result)
            occupancy = jnp.asarray(occupancy, dtype=jnp.float64)  # type: ignore[assignment]
        return EgocentricRatesResult(
            firing_rates=firing_rates_result,
            occupancy=occupancy,
            env=polar_env,
            distance_range=distance_range,
            n_distance_bins=n_distance_bins,
            n_direction_bins=n_direction_bins,
        )

    # Bin spike trains by egocentric coordinates and compute occupancy.
    # Third return is the polar env (see above); rebind to ``polar_env``
    # so the cartesian ``env`` parameter remains accessible.
    spike_counts, occupancy, polar_env = bin_egocentric_spike_trains(
        spike_times_list,
        times,
        positions,
        headings,
        object_positions,
        distance_range=distance_range,
        n_distance_bins=n_distance_bins,
        n_direction_bins=n_direction_bins,
        metric=metric,
        env=env,
        n_jobs=n_jobs,
    )

    # Compute firing rates. The "binned" method uses the raw bin rate (no graph
    # smoothing): diffusion over the polar env bleeds rate across distance rings
    # and erases distance tuning (see _raw_polar_rate). Other methods smooth.
    firing_rates: ArrayLike
    if smoothing_method == "binned":
        firing_rates = np.stack(
            [
                _raw_polar_rate(counts, occupancy, min_occupancy)
                for counts in spike_counts
            ]
        )
        if resolved_backend == "jax" and is_jax_available():
            import jax.numpy as jnp

            firing_rates = jnp.asarray(firing_rates, dtype=jnp.float64)
    else:
        # smooth_rate_maps_batch dispatches to JAX or NumPy based on backend
        firing_rates = smooth_rate_maps_batch(
            polar_env,
            spike_counts,
            occupancy,
            method=smoothing_method,
            bandwidth=bandwidth,
            min_occupancy=min_occupancy,
            backend=resolved_backend,
        )

    # Convert occupancy to JAX if JAX backend is selected
    # (firing_rates is already JAX from smooth_rate_maps_batch)
    if resolved_backend == "jax" and is_jax_available():
        import jax.numpy as jnp

        occupancy = jnp.asarray(occupancy, dtype=jnp.float64)  # type: ignore[assignment]

    # Return result
    return EgocentricRatesResult(
        firing_rates=firing_rates,
        occupancy=occupancy,
        env=polar_env,
        distance_range=distance_range,
        n_distance_bins=n_distance_bins,
        n_direction_bins=n_direction_bins,
    )


# ==============================================================================
# Convenience Functions for Object-Vector Cell Analysis
# ==============================================================================


def _mean_resultant_length(
    angles: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
) -> float:
    """Compute mean resultant length of circular data.

    Parameters
    ----------
    angles : array of float
        Angles in radians.
    weights : array of float, optional
        Weights for each angle. If None, uniform weights.

    Returns
    -------
    float
        Mean resultant length in [0, 1].
    """
    if weights is None:
        weights = np.ones_like(angles)

    weights = weights / np.sum(weights)
    x = np.sum(weights * np.cos(angles))
    y = np.sum(weights * np.sin(angles))

    return float(np.sqrt(x**2 + y**2))


def object_vector_score(
    tuning_curve: NDArray[np.float64],
    *,
    max_distance_selectivity: float = 10.0,
) -> float:
    """Compute combined object-vector selectivity score.

    The score combines distance selectivity and direction selectivity
    following the formula:

        s_OV = ((s_d - 1) / (s_d* - 1)) * s_theta

    where:
    - s_d = peak / mean (distance selectivity)
    - s_d* = max_distance_selectivity (normalization constant)
    - s_theta = mean resultant length (direction selectivity)

    Parameters
    ----------
    tuning_curve : NDArray[np.float64], shape (n_dist, n_dir)
        2D firing rate tuning curve in egocentric polar coordinates.
    max_distance_selectivity : float, default=10.0
        Maximum expected distance selectivity for normalization.
        Must be > 1.

    Returns
    -------
    float
        Object-vector score in [0, 1]. Higher scores indicate sharper
        tuning to a specific distance and direction.

    Raises
    ------
    ValueError
        If max_distance_selectivity <= 1.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.egocentric import object_vector_score
    >>> # Sharp tuning at one location
    >>> tc = np.zeros((10, 12)) + 0.1
    >>> tc[5, 6] = 20.0
    >>> score = object_vector_score(tc)
    >>> score > 0.5
    True

    See Also
    --------
    is_object_vector_cell : Classify neuron as OVC
    EgocentricRateResult.is_object_vector_cell : Classifier method on result object
    """
    if max_distance_selectivity <= 1.0:
        raise ValueError(
            f"max_distance_selectivity must be > 1, got {max_distance_selectivity}"
        )

    tuning_curve = np.asarray(tuning_curve, dtype=np.float64)

    # Handle NaN values
    valid_mask = np.isfinite(tuning_curve)
    if not np.any(valid_mask):
        return float(np.nan)

    valid_rates = tuning_curve[valid_mask]

    # Compute distance selectivity
    peak_rate = float(np.max(valid_rates))
    mean_rate = float(np.mean(valid_rates))

    if mean_rate == 0:
        return 0.0

    distance_selectivity = peak_rate / mean_rate

    # Normalize distance selectivity to [0, 1]
    normalized_dist_sel = (distance_selectivity - 1.0) / (
        max_distance_selectivity - 1.0
    )
    normalized_dist_sel = float(np.clip(normalized_dist_sel, 0.0, 1.0))

    # Compute direction selectivity (mean resultant length)
    n_dir = tuning_curve.shape[1]
    direction_bins = np.linspace(-np.pi, np.pi, n_dir + 1)
    dir_bin_centers = (direction_bins[:-1] + direction_bins[1:]) / 2

    # Marginalize over distance
    direction_tuning = np.nansum(tuning_curve, axis=0)
    total = np.sum(direction_tuning)

    if total == 0:
        direction_selectivity = 0.0
    else:
        direction_selectivity = _mean_resultant_length(
            dir_bin_centers, weights=direction_tuning
        )

    # Combined score
    score = normalized_dist_sel * direction_selectivity

    return float(np.clip(score, 0.0, 1.0))


def is_object_vector_cell(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    object_positions: NDArray[np.float64],
    *,
    distance_range: tuple[float, float] = (0.0, 50.0),
    n_distance_bins: int = 10,
    n_direction_bins: int = 12,
    metric: Literal["euclidean", "geodesic"] = "euclidean",
    min_info: float = 0.3,
) -> bool:
    """Quick check: Is this an object-vector cell?

    Convenience function for fast screening of neurons. Computes the egocentric
    rate map for the supplied trajectory + spikes and classifies the cell as an
    object-vector cell (OVC) using the egocentric-spatial-information criterion.

    This function delegates classification to
    :meth:`EgocentricRateResult.is_object_vector_cell`, so the quick-check and
    the result-object classification always agree: a neuron is an OVC when its
    egocentric spatial information exceeds ``min_info`` (bits/spike).

    For detailed metrics, use :func:`compute_egocentric_rate` and inspect
    the result's methods (``is_object_vector_cell()``, ``preferred_distance()``,
    etc.).

    Parameters
    ----------
    env : Environment
        Allocentric environment (used for geodesic metric and visualization).
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Spike times in seconds. Can be empty.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps of trajectory samples in seconds.
    positions : NDArray[np.float64], shape (n_samples, 2)
        Animal position coordinates at each time sample.
    headings : NDArray[np.float64], shape (n_samples,)
        Head direction at each time sample (radians, **allocentric
        world-frame convention**: 0 = East, π/2 = North, π = West,
        -π/2 = South, wrapped to [-π, π]). The allocentric→egocentric
        transform is applied internally.
    object_positions : NDArray[np.float64], shape (n_objects, 2)
        Object positions in allocentric coordinates.
    distance_range : tuple of float, default=(0.0, 50.0)
        (min_distance, max_distance) for egocentric binning.
    n_distance_bins : int, default=10
        Number of distance bins.
    n_direction_bins : int, default=12
        Number of direction bins (covers full circle).
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric for computing distance to objects.
    min_info : float, default=0.3
        Minimum egocentric spatial information threshold in bits/spike.
        Matches the default of
        :meth:`EgocentricRateResult.is_object_vector_cell`.

    Returns
    -------
    bool
        True if the neuron's egocentric spatial information exceeds
        ``min_info``.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.egocentric import is_object_vector_cell
    >>> positions = np.random.rand(1000, 2) * 100
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>> times = np.linspace(0, 60, 1000)
    >>> headings = np.random.uniform(-np.pi, np.pi, 1000)
    >>> objects = np.array([[50.0, 50.0]])
    >>> spike_times = np.random.uniform(0, 60, 100)
    >>> result = is_object_vector_cell(
    ...     env, spike_times, times, positions, headings, objects
    ... )
    >>> type(result)
    <class 'bool'>

    See Also
    --------
    compute_egocentric_rate : Full egocentric rate computation
    object_vector_score : Compute OVC score from a tuning curve
    EgocentricRateResult.is_object_vector_cell : OVC classification on result object
    """
    try:
        result = compute_egocentric_rate(
            env,
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            distance_range=distance_range,
            n_distance_bins=n_distance_bins,
            n_direction_bins=n_direction_bins,
            metric=metric,
        )
    except (ValueError, RuntimeError):
        return False

    return result.is_object_vector_cell(min_info=min_info)


def plot_object_vector_tuning(
    result: EgocentricRateResult,
    ax: Axes | PolarAxes | None = None,
    *,
    show_peak: bool = True,
    add_colorbar: bool = False,
    cmap: str = "viridis",
    **kwargs: Any,
) -> Axes | PolarAxes:
    """Plot object-vector tuning curve as polar heatmap.

    Creates a polar plot where:
    - Radial axis = distance from object
    - Angular axis = egocentric direction to object

    Parameters
    ----------
    result : EgocentricRateResult
        Result from ``compute_egocentric_rate(None)``.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure with polar projection.
    show_peak : bool, default=True
        If True, mark the peak location with a marker.
    add_colorbar : bool, default=False
        If True, add a colorbar.
    cmap : str, default='viridis'
        Colormap name.
    **kwargs : dict
        Additional keyword arguments passed to pcolormesh.

    Returns
    -------
    matplotlib.axes.Axes or matplotlib.projections.polar.PolarAxes
        The axes object with the plot.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.egocentric import (
    ...     compute_egocentric_rate,
    ...     plot_object_vector_tuning,
    ... )
    >>> # Compute egocentric rate field
    >>> result = compute_egocentric_rate(None, ...)  # doctest: +SKIP
    >>> ax = plot_object_vector_tuning(result)  # doctest: +SKIP

    See Also
    --------
    EgocentricRateResult.plot : Basic plotting on result object
    """
    import matplotlib.pyplot as plt
    from matplotlib.projections.polar import PolarAxes as MPLPolarAxes

    # Reshape firing rate to 2D grid (distance x direction)
    firing_rate = np.asarray(result.firing_rate, dtype=np.float64)
    tuning_curve = firing_rate.reshape(result.n_distance_bins, result.n_direction_bins)

    # Create bin edges
    dist_min, dist_max = result.distance_range
    distance_bins = np.linspace(dist_min, dist_max, result.n_distance_bins + 1)
    direction_bins = np.linspace(-np.pi, np.pi, result.n_direction_bins + 1)

    # Create figure if needed
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})

    # Create mesh grid for polar plot
    theta, r = np.meshgrid(direction_bins, distance_bins)

    # Plot heatmap
    mesh = ax.pcolormesh(theta, r, tuning_curve, cmap=cmap, shading="flat", **kwargs)

    # Configure polar plot
    if isinstance(ax, MPLPolarAxes):
        ax.set_theta_zero_location("N")  # 0 degrees at top (ahead)
        ax.set_theta_direction(-1)  # Clockwise

    # Mark peak if requested
    if show_peak:
        valid_mask = np.isfinite(tuning_curve)
        if np.any(valid_mask):
            # Find peak
            peak_idx = np.unravel_index(np.nanargmax(tuning_curve), tuning_curve.shape)
            dist_centers = (distance_bins[:-1] + distance_bins[1:]) / 2
            dir_centers = (direction_bins[:-1] + direction_bins[1:]) / 2

            peak_r = dist_centers[peak_idx[0]]
            peak_theta = dir_centers[peak_idx[1]]

            ax.scatter(
                [peak_theta],
                [peak_r],
                color="red",
                s=100,
                marker="*",
                zorder=5,
                label="Peak",
            )

    # Add colorbar if requested
    if add_colorbar:
        plt.colorbar(mesh, ax=ax, label="Firing rate (Hz)")

    return ax
