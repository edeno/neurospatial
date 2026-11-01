"""One check of the per-sample ``position_bins`` behavior functions take."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error

if TYPE_CHECKING:
    from neurospatial.environment import Environment


def validated_position_bins(
    position_bins: ArrayLike,
    env: Environment | None = None,
    *,
    call: str,
    times: ArrayLike | None = None,
    n_bins: int | None = None,
) -> NDArray[np.int64]:
    """Check per-sample bin indices: integers in ``[0, n_bins)`` or ``-1``.

    ``-1`` is ``env.bin_at``'s marker for a position outside the environment
    (a missing sample). Any other negative index would silently count from
    the end of the bin arrays, an index at or past ``n_bins`` names no bin,
    and a float bin would be truncated to an unrelated bin, so all are
    refused.

    Parameters
    ----------
    position_bins : array-like of int, shape (n_samples,)
        Bin index at each sample.
    env : Environment, optional
        Environment the bins belong to. Give ``env`` or ``n_bins``.
    call : str
        Public function name, used in messages.
    times : array-like, shape (n_samples,), optional
        Timestamps of the same samples; when given, both must be 1-D with
        one timestamp per bin. Only their shape is checked here.
    n_bins : int, optional
        Number of bins, for a function without an environment (such as one
        taking a per-bin label array).

    Returns
    -------
    ndarray of int64, shape (n_samples,)

    Raises
    ------
    ValueError
        If ``position_bins`` is not 1-D, or does not match ``times``.
    TypeError
        If ``position_bins`` holds booleans, floats or non-numbers.
    BinIndexOutOfRangeError
        If a bin index is outside ``[0, n_bins)`` and is not ``-1``.
    """
    from neurospatial.environment import _validate_target_bins
    from neurospatial.geometry import CartesianSystem

    shape = np.shape(position_bins)
    if times is not None:
        times_shape = np.shape(times)
        if len(shape) != 1 or len(times_shape) != 1 or shape != times_shape:
            raise ValueError(
                _format_error(
                    f"{call}() needs position_bins and times as 1-D arrays of "
                    f"the same length; got shapes {shape} and {times_shape}.",
                    why="each sample's bin is matched to its timestamp.",
                    fix="pass env.bin_at(positions) for the same samples as times.",
                )
            )
    elif len(shape) != 1:
        raise ValueError(
            _format_error(
                f"{call}() got position_bins with shape {shape}.",
                why="position_bins holds one bin index per sample.",
                fix="pass a 1-D integer array, for example env.bin_at(positions).",
            )
        )
    if env is not None:
        n_bins, coordinates = env.n_bins, env.coordinate_system
    else:
        assert n_bins is not None, "give env or n_bins"
        coordinates = CartesianSystem()
    bins = _validate_target_bins(
        position_bins,
        n_bins=n_bins,
        coordinates=coordinates,
        operation=f"{call}()",
        argument="position_bins",
        allow_missing=True,
    )
    return bins.astype(np.int64, copy=False)


def sample_bins(
    env: Environment, times: ArrayLike, positions: ArrayLike, *, call: str
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Validated ``times`` and each position's bin, ``-1`` outside or missing.

    The behavior detectors take ``(env, times, positions)`` and bin the
    positions here, so users never handle bin indices. A NaN position (lost
    tracking) or one outside the environment is bin ``-1``, an unobserved
    sample.

    Parameters
    ----------
    env : Environment
        Allocentric environment the positions were recorded in.
    times : array-like, shape (n_samples,)
        Finite, strictly increasing timestamps in seconds.
    positions : array-like, shape (n_samples, n_dims)
        Positions in the environment's coordinates; NaN where missing.
    call : str
        Public function name, used in messages.

    Returns
    -------
    times : ndarray of float64, shape (n_samples,)
    bins : ndarray of int64, shape (n_samples,)

    Raises
    ------
    ValueError
        If times are invalid, the arrays do not match, or the positions do
        not have the environment's number of coordinates.
    """
    from neurospatial._validation import validate_times_positions

    t, p = validate_times_positions(times, positions, call=call)
    if p.ndim == 1:  # a 1-D track: one coordinate per sample
        p = p[:, np.newaxis]
    if p.shape[1] != env.n_dims:
        raise ValueError(
            _format_error(
                f"{call}(): positions have {p.shape[1]} coordinate(s) per sample; "
                f"the environment has {env.n_dims}.",
                why="positions are located in the environment's bins.",
                fix="pass positions in the environment's coordinates, shape "
                "(n_samples, n_dims).",
            )
        )
    bins = np.asarray(env.bin_at(p), dtype=np.int64)
    return t, bins
