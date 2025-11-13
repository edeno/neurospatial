"""Base protocol for neural models."""

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class NeuralModel(Protocol):
    """Protocol for neural models that generate firing rates.

    All neural models must implement:

    - ``firing_rate(positions, times)`` → rates
    - ``ground_truth`` property (returns dict with model parameters)

    The structure of ground_truth dict depends on model type:

    **PlaceCell**

        - ``'center'``: NDArray[np.float64] - field center coordinates
        - ``'width'``: float | NDArray[np.float64] - field width (isotropic or per-dimension)
        - ``'max_rate'``: float - peak firing rate in Hz
        - ``'baseline_rate'``: float - baseline firing rate in Hz

    **BoundaryCell**

        - ``'preferred_distance'``: float - preferred distance from boundary
        - ``'distance_tolerance'``: float - tuning width
        - ``'preferred_direction'``: float | None - direction (radians, or None for omnidirectional)
        - ``'max_rate'``: float - peak firing rate in Hz
        - ``'baseline_rate'``: float - baseline firing rate in Hz

    **GridCell**

        - ``'grid_spacing'``: float - distance between grid peaks
        - ``'grid_orientation'``: float - grid rotation in radians
        - ``'phase_offset'``: NDArray[np.float64] - spatial phase offset
        - ``'max_rate'``: float - peak firing rate in Hz
        - ``'baseline_rate'``: float - baseline firing rate in Hz

    Examples
    --------
    Creating a custom neural model:

    >>> from neurospatial.simulation.models.base import NeuralModel
    >>> import numpy as np
    >>>
    >>> class CustomModel:
    ...     '''Custom neural model implementation.'''
    ...
    ...     def __init__(self, max_rate):
    ...         self.max_rate = max_rate
    ...
    ...     def firing_rate(self, positions, times=None):
    ...         # Constant firing rate everywhere
    ...         return np.full(len(positions), self.max_rate)
    ...
    ...     @property
    ...     def ground_truth(self):
    ...         return {"max_rate": self.max_rate}
    >>>
    >>> # Verify it implements the protocol
    >>> model = CustomModel(max_rate=10.0)
    >>> isinstance(model, NeuralModel)  # True
    True

    See Also
    --------
    PlaceCellModel : Gaussian place field model
    BoundaryCellModel : Boundary-distance tuned model
    GridCellModel : Hexagonal grid pattern model

    Notes
    -----
    This is a Protocol (PEP 544), not a base class. Models do not need to inherit
    from NeuralModel; they only need to implement the required methods and properties.

    The ``@runtime_checkable`` decorator allows isinstance() checks at runtime,
    which is useful for validation and type checking.
    """

    def firing_rate(
        self,
        positions: NDArray[np.float64],
        times: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute firing rate at given positions.

        Parameters
        ----------
        positions : NDArray[np.float64], shape (n_time, n_dims)
            Position coordinates in the same units as the environment.
        times : NDArray[np.float64], shape (n_time,), optional
            Time points in seconds. Used for time-varying models (e.g., theta phase
            precession, speed modulation). If None, assumes stationary firing rate.

        Returns
        -------
        firing_rate : NDArray[np.float64], shape (n_time,)
            Firing rate in Hz at each position/time point.

        Notes
        -----
        Firing rates should be non-negative. Implementations should handle edge
        cases gracefully (e.g., positions outside environment bounds).
        """
        ...

    @property
    def ground_truth(self) -> dict[str, Any]:
        """Return ground truth parameters for validation.

        Returns
        -------
        parameters : dict[str, Any]
            Model-specific parameters used to generate firing rates.
            Structure varies by model type (see class docstring for details).

        Notes
        -----
        The ground truth dictionary is used for:

        - Validation: comparing detected fields to true parameters
        - Reproducibility: documenting simulation parameters
        - Analysis: accessing model parameters without implementation details

        Examples
        --------
        >>> from neurospatial.simulation import PlaceCellModel
        >>> from neurospatial import Environment
        >>> import numpy as np
        >>>
        >>> # Create environment
        >>> samples = np.random.uniform(0, 100, (1000, 2))
        >>> env = Environment.from_samples(samples, bin_size=2.0)
        >>>
        >>> # Create place cell
        >>> pc = PlaceCellModel(env, center=[50, 50], width=10.0, max_rate=20.0)
        >>>
        >>> # Access ground truth
        >>> pc.ground_truth["center"]  # array([50., 50.])
        array([50., 50.])
        >>> pc.ground_truth["width"]  # 10.0
        10.0
        """
        ...
