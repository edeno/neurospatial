"""Shuffles that build null distributions for replay and decoded-trajectory tests.

Each shuffle yields one shuffled copy at a time and takes ``rng`` for
reproducibility. They break one structure of a decoded event while keeping
the rest:

| Shuffle | Acts on | Breaks |
|---------|---------|--------|
| ``shuffle_cell_identity`` | spike counts vs. rate maps | the place code |
| ``shuffle_place_fields_circular`` | 1-D rate maps | field positions |
| ``shuffle_place_fields_circular_2d`` | 2-D rate maps | field positions |
| ``shuffle_posterior_circular`` | posterior rows | alignment along a trajectory |
| ``shuffle_posterior_weighted_circular`` | posterior rows | alignment, with edge control |

Temporal shuffles (``shuffle_time_bins``), spike-train surrogates and the
Monte Carlo p-value helpers live in :mod:`neurospatial.stats`.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial.stats import _ensure_rng

if TYPE_CHECKING:
    from neurospatial.environment import Environment

# =============================================================================
# Cell identity and place-field shuffles: test spatial code coherence
# =============================================================================


def shuffle_cell_identity(
    spike_counts: NDArray[np.int64],
    encoding_models: NDArray[np.float64],
    *,
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
    generator = _ensure_rng(rng)
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

    for _ in range(n_shuffles):
        # Generate a random permutation of column indices
        perm = generator.permutation(n_neurons)
        # Apply permutation to columns (neuron axis)
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
    from neurospatial.environment._queries import require_track_extents

    starts, _ = require_track_extents(env, operation, alternative=alternative)
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
    return _roll_rows_along_track(encoding_models, order, n_shuffles, _ensure_rng(rng))


def _roll_rows_along_track(
    values: NDArray[np.float64],
    order: NDArray[np.intp],
    n_shuffles: int,
    generator: np.random.Generator,
) -> Generator[NDArray[np.float64], None, None]:
    """Roll each row by its own random number of bins, in track order."""
    n_rows, n_bins = values.shape
    along = values[:, order]
    col_indices = np.arange(n_bins)
    rows = np.arange(n_rows)[:, np.newaxis]

    for _ in range(n_shuffles):
        # Generate random shift amounts for each row
        shifts = np.asarray(generator.integers(0, n_bins, size=n_rows))

        # For each row i, take columns (col_indices - shifts[i]) % n_bins
        shifted_indices = (col_indices - shifts[:, np.newaxis]) % n_bins
        shuffled = np.empty_like(values)
        shuffled[:, order] = along[rows, shifted_indices]

        yield shuffled


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
    generator = _ensure_rng(rng)
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
    shuffle_posterior_weighted_circular : Weighted circular shift with edge
        effect mitigation
    shuffle_place_fields_circular : Circular shift of place fields
    """
    posterior = np.asarray(posterior)
    order = _track_order(
        env, posterior, "shuffle_posterior_circular()", _POSTERIOR_SHUFFLE_ALTERNATIVE
    )
    return _roll_rows_along_track(posterior, order, n_shuffles, _ensure_rng(rng))


def shuffle_posterior_weighted_circular(
    env: Environment,
    posterior: NDArray[np.float64],
    *,
    edge_buffer: int = 5,
    n_shuffles: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Generator[NDArray[np.float64], None, None]:
    """Weighted circular shift with edge effect mitigation.

    Refined version of posterior shuffle that maintains non-uniformity and
    reduces edge effects by restricting shifts when the MAP position is near
    track boundaries. This is more conservative than standard circular shuffle.

    Parameters
    ----------
    env : Environment
        The environment the posterior was decoded on: a 1-D environment or a
        linearized track. Track ends are the first and last bins along the
        track.
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability distribution from decoding. Each row should
        sum to 1.0.
    edge_buffer : int, default=5
        Number of bins from track ends where shifts are restricted.
        When the MAP position is within ``edge_buffer`` bins of either edge,
        the shift is restricted to keep the MAP position within bounds
        (not wrapping to the other end).

        **Default justification (5 bins)**: For typical 1D track decoding with
        2-5 cm spatial bins, 5 bins = 10-25 cm buffer region. This prevents
        unrealistic position jumps (e.g., rat at start wrapping to end) while
        allowing sufficient shift variability. Recommended values:

        - Short tracks (<1m): edge_buffer = 3-5 bins
        - Medium tracks (1-2m): edge_buffer = 5-10 bins
        - Long tracks (>2m): edge_buffer = 10-15 bins
        - Adaptive: edge_buffer = max(5, n_bins // 20)

        For circular tracks, use ``shuffle_posterior_circular`` instead (no
        edge restriction needed).
    n_shuffles : int, default=1000
        Number of shuffled versions to generate.
    rng : np.random.Generator | int | None, default=None
        Random number generator for reproducibility.

        - If Generator: Use directly
        - If int: Seed for ``np.random.default_rng()``
        - If None: Use default RNG (not reproducible)

    Yields
    ------
    shuffled_posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior with weighted circular shifts applied.

    Notes
    -----
    - More conservative than standard circular shuffle
    - Shifts are restricted near track ends to mitigate edge effects
    - When MAP position is within ``edge_buffer`` of an edge, shift is
      constrained to avoid wrapping probability mass to the other end
    - Preserves row normalization (each row still sums to 1.0)

    The edge restriction works as follows:
    - For each time bin, compute MAP position (argmax)
    - If MAP is within ``edge_buffer`` of left edge (bin < edge_buffer):
      restrict shift to [-MAP, n_bins - 2*edge_buffer] to keep MAP in center
    - If MAP is within ``edge_buffer`` of right edge (bin >= n_bins - edge_buffer):
      restrict shift to [-(MAP - center), n_bins - MAP] to keep MAP in center
    - Otherwise: allow full circular shift [0, n_bins)

    Example with n_bins=20 and edge_buffer=5:
    - MAP at bin 2 (near left): shift limited to [-2, 10) so MAP stays in [0, 11]
    - MAP at bin 17 (near right): shift limited to [-7, 3) so MAP stays in [10, 19]
    - MAP at bin 10 (center): full circular shift [0, 20)

    Bin positions here count along the track. A shift moves values by a whole number of bins in track order: bins are
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
    >>> from neurospatial.replay.nulls import shuffle_posterior_weighted_circular

    >>> # A 1-D track of 10 bins and a normalized posterior (3 time bins)
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [20.0, 0.0]], bin_size=2.0
    ... )
    >>> raw = np.random.default_rng(42).random((3, 10))
    >>> posterior = raw / raw.sum(axis=1, keepdims=True)
    >>> for i, shuffled in enumerate(
    ...     shuffle_posterior_weighted_circular(
    ...         env, posterior, edge_buffer=2, n_shuffles=3, rng=42
    ...     )
    ... ):
    ...     print(f"Shuffle {i}: sums to 1={np.allclose(shuffled.sum(axis=1), 1.0)}")
    Shuffle 0: sums to 1=True
    Shuffle 1: sums to 1=True
    Shuffle 2: sums to 1=True

    See Also
    --------
    shuffle_posterior_circular : Standard circular shift without edge restriction
    shuffle_place_fields_circular : Circular shift of place fields
    """
    posterior = np.asarray(posterior)
    order = _track_order(
        env,
        posterior,
        "shuffle_posterior_weighted_circular()",
        _POSTERIOR_SHUFFLE_ALTERNATIVE,
    )
    return _weighted_rolls(posterior, order, edge_buffer, n_shuffles, _ensure_rng(rng))


def _weighted_rolls(
    posterior: NDArray[np.float64],
    order: NDArray[np.intp],
    edge_buffer: int,
    n_shuffles: int,
    generator: np.random.Generator,
) -> Generator[NDArray[np.float64], None, None]:
    """Edge-restricted rolls of each posterior row, in track order."""
    n_time_bins, n_bins = posterior.shape

    # Handle empty posterior
    if n_time_bins == 0:
        for _ in range(n_shuffles):
            yield posterior.copy()
        return

    along = posterior[:, order]
    # MAP positions along the track (used to determine shift restrictions)
    map_positions = np.argmax(along, axis=1)

    # NOTE: the per-time-bin loop below is intentionally not vectorized. Each
    # iteration draws exactly one integer from ``generator`` in time order,
    # and the *range* of that draw depends on the bin's MAP position (edge
    # bins use a narrower, position-dependent range). Drawing all shifts at
    # once would change the RNG consumption pattern and therefore the output
    # for a fixed seed, breaking the reproducible-shuffle contract that
    # callers rely on for null distributions. The cost is also bounded by
    # n_time_bins (a single candidate event, typically tens of bins), so
    # vectorizing buys little. Left as a loop deliberately.
    for _ in range(n_shuffles):
        shuffled_along = np.empty_like(along)
        for i in range(n_time_bins):
            map_pos = map_positions[i]

            # Determine allowed shift range based on MAP position
            # The key principle: restrict shifts to prevent probability mass from
            # wrapping to the opposite end of the track when near edges.
            if edge_buffer == 0 or n_bins <= 2 * edge_buffer:
                # No edge restriction or buffer spans entire track
                # Allow any shift in [0, n_bins)
                shift_amount = generator.integers(0, n_bins)
            elif map_pos < edge_buffer:
                # Near left edge: restrict shift range to avoid wrapping to far right
                # Allow negative shifts (move left) but limit positive shifts
                min_shift = -map_pos  # Don't shift MAP below 0
                # Limit positive shift: keep shifted MAP within center region
                max_shift = min(n_bins - map_pos, n_bins - 2 * edge_buffer)
                if max_shift <= min_shift:
                    max_shift = min_shift + 1  # Ensure valid range
                shift_amount = generator.integers(min_shift, max_shift)
            elif map_pos >= n_bins - edge_buffer:
                # Near right edge: restrict shift range to avoid wrapping to far left
                # Allow positive shifts (move right) but limit negative shifts
                max_shift = n_bins - map_pos  # Don't shift MAP beyond last bin
                # Limit negative shift: keep shifted MAP within center region
                min_shift = max(-(map_pos - (n_bins - 2 * edge_buffer)), -(map_pos))
                if max_shift <= min_shift:
                    min_shift = max_shift - 1  # Ensure valid range
                shift_amount = generator.integers(min_shift, max_shift)
            else:
                # Not near edge: allow full circular shift
                shift_amount = generator.integers(0, n_bins)

            shuffled_along[i, :] = np.roll(along[i, :], shift_amount)
        shuffled = np.empty_like(posterior)
        shuffled[:, order] = shuffled_along
        yield shuffled
