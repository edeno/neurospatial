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
