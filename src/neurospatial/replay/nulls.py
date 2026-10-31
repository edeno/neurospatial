"""Shuffles that build null distributions for replay and decoded-trajectory tests.

Each shuffle yields one shuffled copy at a time and takes ``rng`` for
reproducibility. They break one structure of a decoded event while keeping
the rest:

| Shuffle | Acts on | Breaks |
|---------|---------|--------|
| ``shuffle_cell_identity`` | spike counts vs. rate maps | the place code |
| ``shuffle_spike_counts_circular`` | each cell's counts in time | co-firing order |
| ``shuffle_spike_counts_uniform`` | each cell's spikes in the event | all spike timing |
| ``shuffle_place_fields_circular`` | 1-D rate maps | field positions |
| ``shuffle_place_fields_circular_2d`` | 2-D rate maps | field positions |
| ``shuffle_posterior_circular`` | posterior rows | alignment along a trajectory |
| ``shuffle_posterior_weighted_circular`` | posterior rows | alignment, with peaks drawn from the session's posterior |

Temporal shuffles (``shuffle_time_bins``), spike-train surrogates and the
Monte Carlo p-value helpers live in :mod:`neurospatial.stats`.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error

if TYPE_CHECKING:
    from neurospatial.environment import Environment

# =============================================================================
# Cell identity and place-field shuffles: test spatial code coherence
# =============================================================================


def shuffle_cell_identity(
    spike_counts: NDArray[np.int64],
    encoding_models: NDArray[np.float64],
    *,
    active_only: bool = False,
    n_shuffles: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Generator[tuple[NDArray[np.int64], NDArray[np.float64]], None, None]:
    """Shuffle mapping between spike trains and place fields.

    **Primary test for spatial code coherence.** Disrupts the learned
    relationship between a neuron's activity and its encoded spatial location
    by randomly permuting which spike train is associated with which place
    field.

    This shuffle randomly permutes columns (neuron axis) of spike_counts,
    effectively reassigning which place field goes with which spike train.
    The encoding models are returned unchanged.

    Parameters
    ----------
    spike_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Spike counts per neuron per time bin.
    encoding_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Firing rate maps (place fields) for each neuron.
    active_only : bool, default=False
        Permute only the neurons that fire in the event among themselves,
        as Bush et al. (2022) "shuffled the identity of active cells within
        each event": silent neurons keep their (empty) columns, so a spike
        train is only ever paired with the field of another active cell.
        With ``k`` active neurons there are ``k!`` distinct shuffles.
    n_shuffles : int, default=1000
        Number of shuffled versions to generate.
    rng : np.random.Generator | int | None, default=None
        Random number generator for reproducibility.

        - If Generator: Use directly
        - If int: Seed for ``np.random.default_rng()``
        - If None: Use default RNG (not reproducible)

    Yields
    ------
    shuffled_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Spike counts with neuron identities permuted (columns shuffled).
    encoding_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Original encoding models (unchanged, same object).

    Raises
    ------
    ValueError
        If the number of neurons in ``spike_counts`` (columns) does not
        match the number of rows in ``encoding_models``. Row i of
        ``encoding_models`` is the place field for neuron i (column i of
        ``spike_counts``); a mismatch would silently yield wrong decodes.

    Notes
    -----
    - Randomly permutes columns of spike_counts (neuron axis)
    - Encoding models remain fixed (same object returned each iteration)
    - Equivalent to randomly reassigning which place field goes with which
      spike train
    - Preserves spike counts per neuron per time bin
    - Preserves total spikes per time bin
    - Caution: Can introduce noise if firing rates differ greatly between
      cells

    Alternative implementation (equivalent):
        Instead of shuffling spike_counts columns, shuffle encoding_models
        rows. This yields the same decoded posteriors.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.replay.nulls import shuffle_cell_identity

    >>> spike_counts = np.array([[0, 1, 2], [2, 0, 1]], dtype=np.int64)
    >>> encoding_models = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> for i, (shuffled, models) in enumerate(
    ...     shuffle_cell_identity(spike_counts, encoding_models, n_shuffles=3, rng=42)
    ... ):
    ...     print(
    ...         f"Shuffle {i}: sum={shuffled.sum()}, models unchanged={models is encoding_models}"
    ...     )
    Shuffle 0: sum=6, models unchanged=True
    Shuffle 1: sum=6, models unchanged=True
    Shuffle 2: sum=6, models unchanged=True

    See Also
    --------
    shuffle_place_fields_circular : Circular shift of place fields
    shuffle_time_bins : Temporal order shuffle
    """
    generator = np.random.default_rng(rng)
    n_neurons = spike_counts.shape[1]

    if encoding_models.shape[0] != n_neurons:
        raise ValueError(
            f"spike_counts has {n_neurons} neuron(s) (columns) but "
            f"encoding_models has {encoding_models.shape[0]} (rows). These must "
            f"match: row i of encoding_models is the place field for neuron i "
            f"(column i of spike_counts).\n"
            f"Fix: pass spike_counts shaped (n_time_bins, n_neurons) and "
            f"encoding_models shaped (n_neurons, n_bins)."
        )

    active = (
        np.flatnonzero(np.asarray(spike_counts).sum(axis=0) > 0)
        if active_only
        else np.arange(n_neurons)
    )
    for _ in range(n_shuffles):
        # Permute the chosen columns (neuron axis) among themselves.
        perm = np.arange(n_neurons)
        perm[active] = generator.permutation(active)
        yield spike_counts[:, perm].copy(), encoding_models


def _track_order(
    env: Environment, values: NDArray[np.float64], operation: str, alternative: str
) -> NDArray[np.intp]:
    """Bin numbers in order along a track, after checking ``values`` fits ``env``.

    Parameters
    ----------
    env : Environment
        A 1-D environment or a linearized track.
    values : NDArray[np.float64], shape (n_rows, n_bins)
        Per-bin values, one column per bin of ``env``.
    operation : str
        The calling function, for error messages.
    alternative : str
        What works on other environments.

    Returns
    -------
    NDArray[np.intp], shape (n_bins,)
        ``order[k]`` is the bin ``k``-th along the track. A stable sort keeps
        bins already numbered along the track in place.

    Raises
    ------
    ValueError
        If ``env`` is neither 1-D nor a linearized track, or ``values`` is not
        2-D with one column per bin.
    """
    from neurospatial.environment import _require_track_extents

    starts, _ = _require_track_extents(env, operation, alternative=alternative)
    if values.ndim != 2 or values.shape[1] != env.n_bins:
        raise ValueError(
            _format_error(
                f"{operation} got an array of shape {values.shape}, but the "
                f"environment has {env.n_bins} bins.",
                why="each row is shifted along the track, one value per bin",
                fix="pass an array shaped (n_rows, env.n_bins), computed on "
                "this environment",
            )
        )
    order: NDArray[np.intp] = np.argsort(starts, kind="stable")
    return order


def shuffle_place_fields_circular(
    env: Environment,
    encoding_models: NDArray[np.float64],
    *,
    n_shuffles: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Generator[NDArray[np.float64], None, None]:
    """Circularly shift each place field by a random amount.

    **Conservative null hypothesis.** Preserves individual cell spiking
    properties and local place field structure while disrupting spatial
    relationships between neurons.

    Each neuron's place field is independently shifted by a random number
    of bins along the track, wrapping from one end to the other. This
    preserves the shape of each place field but destroys the spatial
    relationships between neurons.

    Bins where any cell's rate is NaN (no training data) are left as they
    are, and each field is shifted only over the other bins, in track order,
    so the shuffled maps keep the same training support.

    Parameters
    ----------
    env : Environment
        The environment the fields were computed on: a 1-D environment or a
        linearized track.
    encoding_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Firing rate maps (place fields) for each neuron.
    n_shuffles : int, default=1000
        Number of shuffled versions to generate.
    rng : np.random.Generator | int | None, default=None
        Random number generator for reproducibility.

        - If Generator: Use directly
        - If int: Seed for ``np.random.default_rng()``
        - If None: Use default RNG (not reproducible)

    Yields
    ------
    shuffled_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Place fields with each row circularly shifted by a random amount.

    Raises
    ------
    ValueError
        If ``env`` is neither 1-D nor a linearized track, or
        ``encoding_models`` does not have one column per bin.

    Notes
    -----
    - Each neuron's place field is shifted independently
    - Preserves the shape of each place field
    - Destroys spatial relationships between neurons
    - More conservative than cell identity shuffle
    - A 2-D arena has no single axis to shift along: on a full rectangular
      grid use ``shuffle_place_fields_circular_2d``; otherwise shift spike
      times with ``shuffle_spike_times_circular``

    A shift moves values by a whole number of bins in track order: bins are
    ordered by their position along the track (linear position on a
    linearized track, the coordinate in a 1-D environment), not by bin
    number, and the gap between two track segments is skipped. With bins of
    one length this moves a field rigidly. With bins of unequal length each
    value keeps its own bin's worth of track, so a field is stretched or
    compressed where it lands on bins of a different length; use bins of
    one length when a field's extent in centimeters must be kept.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.replay.nulls import shuffle_place_fields_circular

    >>> env = Environment.linear_track(endpoints=[[0.0, 0.0], [8.0, 0.0]], bin_size=2.0)
    >>> encoding_models = np.array(
    ...     [
    ...         [1.0, 2.0, 3.0, 4.0],
    ...         [4.0, 3.0, 2.0, 1.0],
    ...     ]
    ... )
    >>> for i, shuffled in enumerate(
    ...     shuffle_place_fields_circular(env, encoding_models, n_shuffles=3, rng=42)
    ... ):
    ...     print(f"Shuffle {i}: shape={shuffled.shape}")
    Shuffle 0: shape=(2, 4)
    Shuffle 1: shape=(2, 4)
    Shuffle 2: shape=(2, 4)

    See Also
    --------
    shuffle_place_fields_circular_2d : 2D circular shift for 2D environments
    shuffle_cell_identity : Shuffle neuron-to-place-field mapping
    """
    encoding_models = np.asarray(encoding_models)
    order = _track_order(
        env,
        encoding_models,
        "shuffle_place_fields_circular()",
        "on a full rectangular 2-D grid use shuffle_place_fields_circular_2d; "
        "otherwise shift spike times with shuffle_spike_times_circular",
    )
    # Bins where any cell's rate is NaN (no training data) stay as they are;
    # each field is shifted over the remaining bins, in track order.
    supported: NDArray[np.bool_] = np.asarray(
        np.isfinite(np.asarray(encoding_models)).all(axis=0)
    )
    order = order[supported[order]]
    return _roll_rows_along_track(
        encoding_models, order, n_shuffles, np.random.default_rng(rng)
    )


def _roll_rows_along_track(
    values: NDArray[np.float64],
    order: NDArray[np.intp],
    n_shuffles: int,
    generator: np.random.Generator,
    *,
    exclude_zero_shift: bool = False,
) -> Generator[NDArray[np.float64], None, None]:
    """Roll each row by its own random number of bins, in track order.

    Only the bins in ``order`` move; any other column is copied unchanged.
    Shifts are uniform on 0 to ``n_bins - 1`` bins, or on 1 to ``n_bins - 1``
    with ``exclude_zero_shift``.
    """
    n_rows = values.shape[0]
    n_bins = len(order)
    along = values[:, order]
    smallest = 1 if exclude_zero_shift else 0

    for _ in range(n_shuffles):
        shifts = np.asarray(generator.integers(smallest, n_bins, size=n_rows))
        shuffled = values.copy()
        shuffled[:, order] = _roll_along(along, shifts)
        yield shuffled


def _roll_along(
    along: NDArray[np.float64], shifts: NDArray[np.integer[Any]]
) -> NDArray[np.float64]:
    """Each row rolled circularly by its own number of bins.

    Parameters
    ----------
    along : NDArray[np.float64], shape (n_rows, n_bins)
        Rows in track order.
    shifts : NDArray[np.integer], shape (n_rows,)
        Bins to roll each row by, toward the end of the track; any integer.

    Returns
    -------
    NDArray[np.float64], shape (n_rows, n_bins)
        Row ``i`` is ``np.roll(along[i], shifts[i])``.
    """
    n_rows, n_bins = along.shape
    columns = (np.arange(n_bins) - shifts[:, np.newaxis]) % n_bins
    rolled: NDArray[np.float64] = along[np.arange(n_rows)[:, np.newaxis], columns]
    return rolled


_NO_2D_FIELD_SHUFFLE = (
    "there is no field-preserving shuffle for this arena; shift each "
    "neuron's spike times with shuffle_spike_times_circular, the null used "
    "by place_cell_significance and the other *_significance functions, and "
    "recompute the rate maps"
)


def shuffle_place_fields_circular_2d(
    env: Environment,
    encoding_models: NDArray[np.float64],
    *,
    n_shuffles: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Generator[NDArray[np.float64], None, None]:
    """Circularly shift 2D place fields in both dimensions.

    For 2D environments, shifts place fields in both x and y dimensions.
    This is the 2D analog of ``shuffle_place_fields_circular``.

    Each neuron's place field is reshaped to a 2D grid, shifted by random
    amounts in both dimensions, and flattened back to 1D.

    Parameters
    ----------
    env : Environment
        2D environment with grid layout (provides ``grid_shape``).
    encoding_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Firing rate maps (place fields) for each neuron.
    n_shuffles : int, default=1000
        Number of shuffled versions to generate.
    rng : np.random.Generator | int | None, default=None
        Random number generator for reproducibility.

        - If Generator: Use directly
        - If int: Seed for ``np.random.default_rng()``
        - If None: Use default RNG (not reproducible)

    Yields
    ------
    shuffled_models : NDArray[np.float64], shape (n_neurons, n_bins)
        Place fields with 2D circular shifts applied.

    Raises
    ------
    ValueError
        If environment is not 2D or doesn't have grid layout, or if the grid
        has cells outside the arena (a masked grid): no field-preserving
        shuffle exists there, so shift spike times with
        ``shuffle_spike_times_circular`` instead.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.replay.nulls import shuffle_place_fields_circular_2d

    >>> positions = np.random.default_rng(42).uniform(0, 10, (100, 2))  # doctest: +SKIP
    >>> env = Environment.from_samples(positions, bin_size=2.0)  # doctest: +SKIP
    >>> encoding_models = np.random.default_rng(42).random(
    ...     (3, env.n_bins)
    ... )  # doctest: +SKIP
    >>> for i, shuffled in enumerate(  # doctest: +SKIP
    ...     shuffle_place_fields_circular_2d(env, encoding_models, n_shuffles=3, rng=42)
    ... ):
    ...     print(f"Shuffle {i}: shape={shuffled.shape}")
    Shuffle 0: shape=(3, 36)
    Shuffle 1: shape=(3, 36)
    Shuffle 2: shape=(3, 36)

    See Also
    --------
    shuffle_place_fields_circular : Circular shift along a 1-D track
    shuffle_spike_times_circular : Spike-time shift, for any arena
    shuffle_cell_identity : Shuffle neuron-to-place-field mapping
    """
    # Validate 2D environment
    if env.n_dims != 2:
        raise ValueError(
            f"shuffle_place_fields_circular_2d requires a 2D environment, "
            f"got {env.n_dims}D"
        )

    # Get grid shape from environment
    if env.grid_shape is None:
        raise ValueError(
            "shuffle_place_fields_circular_2d requires environment with grid layout "
            "(grid_shape attribute)"
        )

    grid_shape = env.grid_shape
    if len(grid_shape) != 2:
        raise ValueError(
            _format_error(
                "shuffle_place_fields_circular_2d() needs a world-aligned 2-D grid.",
                why="it rolls each field along the grid's rows and columns",
                fix=_NO_2D_FIELD_SHUFFLE,
            )
        )

    # Verify encoding models match expected size
    expected_bins = int(np.prod(grid_shape))
    actual_bins = encoding_models.shape[1]
    if actual_bins != expected_bins:
        raise ValueError(
            _format_error(
                f"shuffle_place_fields_circular_2d() needs one value per cell of "
                f"the full {grid_shape} grid ({expected_bins} bins), but got "
                f"{actual_bins} bins, so the grid has cells outside the arena.",
                why="rolling a field across a masked grid would move rates into "
                "cells outside the arena and break fields at the mask edges",
                fix=_NO_2D_FIELD_SHUFFLE,
            )
        )
    generator = np.random.default_rng(rng)
    n_neurons = encoding_models.shape[0]

    # Pre-compute indices for vectorized 2D roll
    # Create meshgrid of indices
    rows, cols = np.arange(grid_shape[0]), np.arange(grid_shape[1])

    for _ in range(n_shuffles):
        # Generate random shift amounts for each neuron in each dimension
        shifts_x = np.asarray(generator.integers(0, grid_shape[0], size=n_neurons))
        shifts_y = np.asarray(generator.integers(0, grid_shape[1], size=n_neurons))

        # Reshape all fields to 2D grids: (n_neurons, grid_shape[0], grid_shape[1])
        fields_2d = encoding_models.reshape(n_neurons, grid_shape[0], grid_shape[1])

        # Vectorized 2D circular shift using advanced indexing
        # For each neuron i, we want:
        #   row_indices = (rows - shifts_x[i]) % grid_shape[0]
        #   col_indices = (cols - shifts_y[i]) % grid_shape[1]
        # Shape: (n_neurons, grid_shape[0])
        shifted_row_indices = (rows - shifts_x[:, np.newaxis]) % grid_shape[0]
        # Shape: (n_neurons, grid_shape[1])
        shifted_col_indices = (cols - shifts_y[:, np.newaxis]) % grid_shape[1]

        # Apply shifts using advanced indexing
        # fields_2d[n, shifted_row_indices[n, :, None], shifted_col_indices[n, None, :]]
        neuron_indices = np.arange(n_neurons)[:, np.newaxis, np.newaxis]
        row_idx = shifted_row_indices[:, :, np.newaxis]
        col_idx = shifted_col_indices[:, np.newaxis, :]

        shifted_2d = fields_2d[neuron_indices, row_idx, col_idx]

        # Flatten back to 1D: (n_neurons, n_bins)
        shuffled = shifted_2d.reshape(n_neurons, -1)

        yield shuffled


# =============================================================================
# Posterior shuffles: test trajectory detection
# =============================================================================


_POSTERIOR_SHUFFLE_ALTERNATIVE = (
    "a 2-D arena has no single axis to shift along; shuffle the data that "
    "produced the posterior instead, with shuffle_time_bins or "
    "shuffle_cell_identity"
)


def shuffle_posterior_circular(
    env: Environment,
    posterior: NDArray[np.float64],
    *,
    n_shuffles: int = 1000,
    rng: np.random.Generator | int | None = None,
    exclude_zero_shift: bool = False,
) -> Generator[NDArray[np.float64], None, None]:
    """Circularly shift posterior at each time bin independently.

    Controls for chance linear alignment of position estimates by disrupting
    trajectory progression while preserving local smoothness. Each time bin's
    posterior is independently shifted by a random number of bins along the
    track, wrapping from one end to the other.

    Parameters
    ----------
    env : Environment
        The environment the posterior was decoded on: a 1-D environment or a
        linearized track.
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability distribution from decoding. Each row should
        sum to 1.0.
    n_shuffles : int, default=1000
        Number of shuffled versions to generate.
    rng : np.random.Generator | int | None, default=None
        Random number generator for reproducibility.

        - If Generator: Use directly
        - If int: Seed for ``np.random.default_rng()``
        - If None: Use default RNG (not reproducible)
    exclude_zero_shift : bool, default=False
        Draw each row's shift from 1 to ``n_bins - 1`` bins, so no row stays
        in place, as nelpy does (Maboudi et al. 2018). The default draws
        from 0 to ``n_bins - 1``, every circular shift equally likely, the
        exact permutation null of the column cycle.

    Yields
    ------
    shuffled_posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior with each row circularly shifted by a random amount.

    Notes
    -----
    - Each time bin is shifted independently
    - Preserves the shape of each instantaneous posterior
    - Destroys temporal continuity of decoded positions
    - Caution: Can generate position representations that don't exist in
      original data (edge effects near track boundaries)
    - Normalization is preserved (each row still sums to 1.0)

    A shift moves values by a whole number of bins in track order: bins are
    ordered by their position along the track (linear position on a
    linearized track, the coordinate in a 1-D environment), not by bin
    number, and the gap between two track segments is skipped. With bins of
    one length this moves a field rigidly. With bins of unequal length each
    value keeps its own bin's worth of track, so a field is stretched or
    compressed where it lands on bins of a different length; use bins of
    one length when a field's extent in centimeters must be kept.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.replay.nulls import shuffle_posterior_circular

    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [10.0, 0.0]], bin_size=2.0
    ... )

    >>> # Create a normalized posterior (3 time bins, 5 spatial bins)
    >>> raw = np.array(
    ...     [
    ...         [0.1, 0.2, 0.4, 0.2, 0.1],
    ...         [0.5, 0.3, 0.1, 0.05, 0.05],
    ...         [0.05, 0.1, 0.2, 0.4, 0.25],
    ...     ]
    ... )
    >>> for i, shuffled in enumerate(
    ...     shuffle_posterior_circular(env, raw, n_shuffles=3, rng=42)
    ... ):
    ...     print(
    ...         f"Shuffle {i}: shape={shuffled.shape}, sums to 1={np.allclose(shuffled.sum(axis=1), 1.0)}"
    ...     )
    Shuffle 0: shape=(3, 5), sums to 1=True
    Shuffle 1: shape=(3, 5), sums to 1=True
    Shuffle 2: shape=(3, 5), sums to 1=True

    See Also
    --------
    shuffle_posterior_weighted_circular : Shift each peak to a position drawn
        from the session's mean posterior (Farooq et al. 2019)
    shuffle_place_fields_circular : Circular shift of place fields
    """
    posterior = np.asarray(posterior)
    order = _track_order(
        env, posterior, "shuffle_posterior_circular()", _POSTERIOR_SHUFFLE_ALTERNATIVE
    )
    if not isinstance(exclude_zero_shift, (bool, np.bool_)):
        raise TypeError(
            "shuffle_posterior_circular() needs exclude_zero_shift=True or "
            f"False, got {exclude_zero_shift!r}."
        )
    return _roll_rows_along_track(
        posterior,
        order,
        n_shuffles,
        np.random.default_rng(rng),
        exclude_zero_shift=bool(exclude_zero_shift),
    )


def shuffle_posterior_weighted_circular(
    env: Environment,
    posterior: NDArray[np.float64],
    peak_weights: NDArray[np.float64],
    *,
    edge_exclusion: float = 0.0,
    n_shuffles: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Generator[NDArray[np.float64], None, None]:
    """Shift each time bin's posterior so its peak lands at a weighted position.

    The weighted circular shift of Farooq et al. (2019). For each time bin
    independently, a new position for the row's peak (its MAP bin) is
    drawn with probability proportional to ``peak_weights``, and the whole
    row is shifted circularly along the track so that its peak lands there.
    Each row keeps its shape, but the decoded positions follow where the
    decoder places mass across the session rather than a uniform
    distribution, as the plain column cycle
    (:func:`shuffle_posterior_circular`) assumes. Peaks never land within
    ``edge_exclusion`` of either end of the track, so a decoded
    distribution is not split across the track's two ends.

    Parameters
    ----------
    env : Environment
        The environment the posterior was decoded on: a 1-D environment or a
        linearized track.
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior mass of each time bin of the event.
    peak_weights : NDArray[np.float64], shape (n_bins,)
        Non-negative weight of each bin as a new peak position. Farooq et al.
        used the mean decoded posterior over every time bin of the
        session's candidate events: ``np.vstack(posteriors).mean(axis=0)``.
    edge_exclusion : float, default=0.0
        Distance along the track, in the environment's units, within which
        no peak lands at either end: bins whose center is closer than this
        to the first or last position of the track get no weight. Farooq et
        al. used 10 cm, set from the smoothing of their rate maps. The
        default excludes nothing.
    n_shuffles : int, default=1000
        Number of shuffled posteriors to yield.
    rng : np.random.Generator, int, or None, default=None
        Random number generator or seed.

    Yields
    ------
    NDArray[np.float64], shape (n_time_bins, n_bins)
        The posterior with each row shifted circularly along the track.

    Raises
    ------
    ValueError
        If ``env`` is neither 1-D nor a linearized track; if ``posterior``
        does not have one column per bin; if ``peak_weights`` is not one
        finite, non-negative weight per bin; if ``edge_exclusion`` is not a
        finite distance >= 0; or if no bin outside the excluded ends has
        weight.

    See Also
    --------
    shuffle_posterior_circular : Shift by a uniformly random distance.
    neurospatial.replay.replay_significance : Runs this shuffle as the
        ``"column_cycle_weighted"`` null.

    Notes
    -----
    Farooq et al. (2019, STAR Methods p. e4): "the maximum decoded location
    in each bin within the frames ... was shifted, weighted by the mean
    decoded posteriors across the track, to conserve non-uniformity in these
    shuffles. Second, the decoding field surrounding the maximum decoded
    location was prevented from being divided at the ends of the track to
    mitigate the edge effect ... this peak was not shifted to within 10 cm
    of the ends of the virtual track". Figure S4B: "the average decoded
    probabilities across bins for the whole sleep were calculated, and the
    peak decoded location was shifted to a location weighed by that
    distribution ... the whole bin was circularly shifted". The methods
    text also says "a pseudorandom function of the average peak decoded
    locations", which could mean a histogram of MAP positions; the legend's
    mean posterior is used here.

    Only the peak is kept away from the ends: the rest of the row is still
    shifted circularly, so mass far from the peak can wrap from one end to
    the other. A row's peak is its first largest bin along the track, and a
    row without mass (a time bin whose posterior was set to zero) stays
    zero. The new peak may be the old one, so a shift of zero is possible.
    Shifts move values by whole bins in track order, as in
    :func:`shuffle_posterior_circular`; with bins of unequal length a
    row is stretched or compressed where it lands.

    References
    ----------
    Farooq, U., Sibille, J., Liu, K., & Dragoi, G. (2019). Strengthened
    temporal coordination within pre-existing sequential cell assemblies
    supports trajectory replay. *Neuron*, 103(4), 719-733.

    Examples
    --------
    Peaks land only where the weights put them: here on the middle three of
    ten bins.

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.replay import shuffle_posterior_weighted_circular
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [20.0, 0.0]], bin_size=2.0, units="cm"
    ... )
    >>> posterior = np.full((4, env.n_bins), 0.05)
    >>> posterior[np.arange(4), [0, 3, 6, 9]] = 0.55
    >>> weights = np.zeros(env.n_bins)
    >>> weights[4:7] = 1.0
    >>> peaks = set()
    >>> for shuffled in shuffle_posterior_weighted_circular(
    ...     env, posterior, weights, n_shuffles=50, rng=0
    ... ):
    ...     peaks |= set(np.argmax(shuffled, axis=1).tolist())
    >>> sorted(peaks)
    [4, 5, 6]
    """
    operation = "shuffle_posterior_weighted_circular()"
    posterior = np.asarray(posterior, dtype=np.float64)
    order = _track_order(env, posterior, operation, _POSTERIOR_SHUFFLE_ALTERNATIVE)
    probabilities = _peak_probabilities(
        env, order, peak_weights, edge_exclusion, operation
    )
    return _peak_weighted_rolls(
        posterior, order, probabilities, n_shuffles, np.random.default_rng(rng)
    )


def _peak_probabilities(
    env: Environment,
    order: NDArray[np.intp],
    peak_weights: NDArray[np.float64],
    edge_exclusion: float,
    operation: str,
) -> NDArray[np.float64]:
    """Probability of each bin, in track order, to receive a shifted peak.

    The weights with bins within ``edge_exclusion`` of either end set to
    zero, normalized to sum to 1; refused when no bin keeps weight.
    """
    from neurospatial.environment import _require_track_extents

    weights = np.asarray(peak_weights, dtype=np.float64)
    if (
        weights.shape != (env.n_bins,)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        raise ValueError(
            _format_error(
                f"{operation}: peak_weights must be one finite, non-negative "
                f"weight per bin, shape ({env.n_bins},); got shape "
                f"{weights.shape}.",
                why="each time bin's new peak position is drawn in proportion "
                "to these weights",
                fix="pass the mean posterior of the session's candidate "
                "events, np.vstack(posteriors).mean(axis=0)",
            )
        )
    if (
        isinstance(edge_exclusion, bool)
        or not isinstance(edge_exclusion, (int, float, np.integer, np.floating))
        or not np.isfinite(edge_exclusion)
        or edge_exclusion < 0
    ):
        raise ValueError(
            f"{operation}: edge_exclusion must be a finite distance >= 0 in "
            f"the environment's units, got {edge_exclusion!r}."
        )
    starts, stops = _require_track_extents(
        env, operation, alternative=_POSTERIOR_SHUFFLE_ALTERNATIVE
    )
    centers = (np.asarray(starts) + np.asarray(stops)) / 2.0
    allowed = (centers - np.min(starts) >= edge_exclusion) & (
        np.max(stops) - centers >= edge_exclusion
    )
    weights = np.where(allowed, weights, 0.0)[order]
    total = weights.sum()
    if not total > 0:
        raise ValueError(
            _format_error(
                f"{operation}: no bin farther than edge_exclusion="
                f"{edge_exclusion} from the track's ends has weight.",
                why="each time bin's peak must land on a bin with weight, "
                "away from the ends",
                fix="use a smaller edge_exclusion, or weights from more "
                "candidate events",
            )
        )
    probabilities: NDArray[np.float64] = weights / total
    return probabilities


def _peak_weighted_rolls(
    posterior: NDArray[np.float64],
    order: NDArray[np.intp],
    probabilities: NDArray[np.float64],
    n_shuffles: int,
    generator: np.random.Generator,
) -> Generator[NDArray[np.float64], None, None]:
    """Roll each row, in track order, so its peak lands at a drawn bin.

    ``probabilities`` are in track order and sum to 1.
    """
    n_rows = len(posterior)
    n_bins = len(order)
    along = posterior[:, order]
    source = np.argmax(along, axis=1)
    for _ in range(n_shuffles):
        destination = generator.choice(n_bins, size=n_rows, p=probabilities)
        shuffled = np.empty_like(posterior)
        shuffled[:, order] = _roll_along(along, destination - source)
        yield shuffled


def shuffle_spike_counts_circular(
    spike_counts: NDArray[np.int64],
    *,
    n_shuffles: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Generator[NDArray[np.int64], None, None]:
    """Circularly shift each cell's spike counts in time, within one event.

    Each cell's binned spike counts are rotated along the event's time bins by
    its own random number of bins, from 0 to ``n_time_bins - 1``, wrapping the
    end of the event to its start. Each cell keeps its spike count and burst
    structure, but the order in which cells fire is broken. Decoding the
    shifted counts gives the "spike train circular shift" null of Tirole et
    al. (2022) and the "temporal shuffle" of Maboudi et al. (2018).

    Parameters
    ----------
    spike_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Spike counts of one event.
    n_shuffles : int, default=1000
        Number of shuffles to yield.
    rng : np.random.Generator, int, or None, default=None
        Random number generator or seed.

    Yields
    ------
    NDArray[np.int64], shape (n_time_bins, n_neurons)
        Counts with each column rotated along time.

    Raises
    ------
    ValueError
        If ``spike_counts`` is not 2-D.

    See Also
    --------
    neurospatial.stats.shuffle_spike_times_circular : Shift spike times across
        a whole session, for tuning tests.
    neurospatial.stats.shuffle_time_bins : Permute time bins, all cells together.

    Notes
    -----
    The shift of each cell is drawn uniformly from all ``n_time_bins`` shifts.
    The Bendor-lab code draws ``ceil(rand * n_time_bins)``, which is the same
    distribution, since a shift by the whole event leaves a cell unchanged.

    References
    ----------
    Tirole, M., Huelin Gorriz, M., Takigawa, M., Kukovska, L., & Bendor, D.
    (2022). Experience-driven rate modulation is reinstated during
    hippocampal replay. *eLife*, 11, e79031.

    Maboudi, K., Ackermann, E., de Jong, L. W., Pfeiffer, B. E., Foster, D.,
    Diba, K., & Kemere, C. (2018). Uncovering temporal structure in
    hippocampal output patterns. *eLife*, 7, e34467.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.replay import shuffle_spike_counts_circular
    >>> counts = np.array([[1, 0], [2, 0], [0, 3]])
    >>> shuffled = next(shuffle_spike_counts_circular(counts, n_shuffles=1, rng=0))
    >>> shuffled.sum(axis=0).tolist()  # each cell keeps its spikes
    [3, 3]
    """
    counts = np.asarray(spike_counts)
    if counts.ndim != 2:
        raise ValueError(
            f"spike_counts must be 2-D (n_time_bins, n_neurons), got shape "
            f"{counts.shape}."
        )
    generator = np.random.default_rng(rng)
    n_time, n_neurons = counts.shape
    rows = np.arange(n_time)[:, np.newaxis]
    columns = np.arange(n_neurons)[np.newaxis, :]
    for _ in range(n_shuffles):
        shifts = generator.integers(0, max(n_time, 1), size=n_neurons)
        yield counts[(rows - shifts[np.newaxis, :]) % max(n_time, 1), columns]


def shuffle_spike_counts_uniform(
    spike_counts: NDArray[np.int64],
    *,
    n_shuffles: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Generator[NDArray[np.int64], None, None]:
    """Place each cell's spikes uniformly at random within one event.

    Each cell keeps its number of spikes in the event, and each spike is put
    in a time bin drawn uniformly and independently, so a cell with ``n``
    spikes gets counts ``Multinomial(n, 1 / n_time_bins)``. This is binning
    spike times redrawn uniformly over the event window, the "spike
    jittering" of Stella et al. (2019), when the time bins are equal and do
    not overlap. All timing is broken: the order of cells, each cell's
    spike intervals and bursts, and which cells fire together.

    Parameters
    ----------
    spike_counts : NDArray[np.int64], shape (n_time_bins, n_neurons)
        Spike counts of one event, in equal time bins that do not overlap
        and that together cover the event.
    n_shuffles : int, default=1000
        Number of shuffles to yield.
    rng : np.random.Generator, int, or None, default=None
        Random number generator or seed.

    Yields
    ------
    NDArray[np.int64], shape (n_time_bins, n_neurons)
        Counts with each cell's spikes spread over the time bins.

    Raises
    ------
    ValueError
        If ``spike_counts`` is not 2-D or holds negative counts.

    See Also
    --------
    shuffle_spike_counts_circular : Shift each cell's counts in time,
        keeping its spike intervals.
    neurospatial.replay.replay_significance : Runs this shuffle as the
        ``"spike_time_uniform"`` null, and refuses time bins that overlap
        or leave gaps.
    neurospatial.replay.null_events : ``kind="spike_time_uniform"`` makes
        the same null events from spike times.

    Notes
    -----
    Stella et al. (2019, STAR Methods p. e4): "Within each reactivation
    event, the decoding procedure was performed after each spike's
    occurrence time was randomly and independently reassigned by drawing it
    from a uniform distribution within the event window." The spikes keep
    their cells, so each cell keeps its count. In spike-train analyses
    "jitter" usually means a small local shift of each spike; this null
    moves each spike anywhere in the event.

    Counts of overlapping windows (a time step shorter than the window)
    share spikes, and windows whose edges follow the spikes (Stella et al.'s
    15-spike windows) change with them, so neither can be redrawn from
    counts: redraw spike times with :func:`~neurospatial.replay.null_events`
    and bin them again instead.

    References
    ----------
    Stella, F., Baracskay, P., O'Neill, J., & Csicsvari, J. (2019).
    Hippocampal reactivation of random trajectories resembling Brownian
    diffusion. *Neuron*, 102(2), 450-461.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.replay import shuffle_spike_counts_uniform
    >>> counts = np.array([[3, 0], [0, 1], [0, 0], [0, 0]])
    >>> shuffled = next(shuffle_spike_counts_uniform(counts, n_shuffles=1, rng=0))
    >>> shuffled.sum(axis=0).tolist()  # each cell keeps its spikes
    [3, 1]
    """
    counts = np.asarray(spike_counts)
    if counts.ndim != 2 or np.any(counts < 0):
        raise ValueError(
            "shuffle_spike_counts_uniform(): spike_counts must be 2-D "
            "(n_time_bins, n_neurons) non-negative counts, got shape "
            f"{counts.shape}."
        )
    generator = np.random.default_rng(rng)
    n_time = len(counts)
    totals = counts.sum(axis=0)
    share = np.full(n_time, 1.0 / max(n_time, 1))
    for _ in range(n_shuffles):
        if n_time == 0:
            yield counts.copy()
            continue
        drawn = generator.multinomial(totals, share)  # (n_neurons, n_time_bins)
        yield drawn.T.astype(counts.dtype, copy=False)
