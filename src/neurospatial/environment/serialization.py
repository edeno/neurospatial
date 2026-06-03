"""Serialization methods for Environment.

This module provides serialization and deserialization methods for the
Environment class, including:

- JSON + npz format (the on-disk persistence path)
- In-memory dictionary format

The methods in this mixin delegate to the implementations in `neurospatial.io`
to maintain separation of concerns and avoid code duplication.

Classes
-------
EnvironmentSerialization
    Mixin class providing serialization methods.

Notes
-----
This is a plain mixin class (NOT a dataclass). It is designed to be mixed
into the Environment dataclass defined in `neurospatial.environment.core`.

The TYPE_CHECKING guard is used to prevent circular imports while still
providing type hints for IDEs and type checkers.

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from neurospatial.environment._protocols import EnvironmentProtocol, SelfEnv

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment


class EnvironmentSerialization:
    """Mixin providing serialization and deserialization methods.

    This mixin provides methods for saving and loading Environment instances
    using two complementary formats:

    - **JSON + npz**: Stable, cross-platform, no arbitrary code execution.
      The disk persistence path.
    - **Dictionary**: In-memory representation for programmatic access.

    Methods
    -------
    to_file(path)
        Save environment to JSON + npz files.
    from_file(path)
        Load environment from JSON + npz files.
    to_dict()
        Convert environment to dictionary representation.
    from_dict(data)
        Reconstruct environment from dictionary.

    Notes
    -----
    This is a plain class (NOT a dataclass). Only the Environment class itself
    should be a dataclass.

    All methods delegate to implementations in `neurospatial.io` to maintain
    separation of concerns and avoid code duplication.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> data = np.random.rand(100, 2) * 10
    >>> env = Environment.from_samples(data, bin_size=2.0)
    >>> tmp = Path(tempfile.mkdtemp()) / "my_environment"
    >>> env.to_file(tmp)  # Creates .json and .npz files
    >>> loaded = Environment.from_file(tmp)
    >>> loaded.n_bins == env.n_bins
    True

    See Also
    --------
    neurospatial.io : Implementation module for serialization logic.

    """

    def to_file(self: SelfEnv, path: str | Path) -> None:
        """Save Environment to versioned JSON + npz files.

        This method provides stable, reproducible serialization compatible
        across Python versions. Creates two files:
        `{path}.json` (metadata) and `{path}.npz` (arrays).

        Parameters
        ----------
        path : str or Path
            Base path for output files (without extension).
            Will create `{path}.json` and `{path}.npz`.

        Examples
        --------
        >>> env = Environment.from_samples(data, bin_size=2.0)  # doctest: +SKIP
        >>> env.to_file("my_environment")  # doctest: +SKIP

        See Also
        --------
        from_file : Load environment from saved files
        to_dict : Convert to dictionary for in-memory use

        Notes
        -----
        The JSON file contains metadata, regions, and the serialization
        version, while the npz file holds numpy arrays in efficient binary
        form. No arbitrary code is executed at load time.

        """
        from neurospatial.io import to_file as _to_file

        _to_file(cast("Environment", self), path)

    @classmethod
    def from_file(cls: type[EnvironmentProtocol], path: str | Path) -> Environment:
        """Load Environment from versioned JSON + npz files.

        Parameters
        ----------
        path : str or Path
            Base path to load from (without extension).
            Will read `{path}.json` and `{path}.npz`.

        Returns
        -------
        Environment
            Reconstructed Environment instance.

        Raises
        ------
        FileNotFoundError
            If JSON or npz files are missing.

        Examples
        --------
        >>> env = Environment.from_file("my_environment")  # doctest: +SKIP

        See Also
        --------
        to_file : Save environment to files
        from_dict : Reconstruct from dictionary

        Notes
        -----
        This method can load environments saved with any serialization version,
        providing forward and backward compatibility.

        """
        from neurospatial.io import from_file as _from_file

        return _from_file(path)

    def to_dict(self: SelfEnv) -> dict[str, Any]:
        """Convert Environment to dictionary for in-memory handoff.

        Returns a dictionary representation with all numpy arrays converted
        to lists for JSON compatibility. Useful for programmatic access,
        testing, or passing to other libraries.

        Returns
        -------
        dict[str, Any]
            Dictionary representation with all arrays as lists.

        Examples
        --------
        >>> env = Environment.from_samples(data, bin_size=2.0)  # doctest: +SKIP
        >>> env_dict = env.to_dict()  # doctest: +SKIP
        >>> # Dictionary can be converted to JSON
        >>> import json  # doctest: +SKIP
        >>> json_str = json.dumps(env_dict)  # doctest: +SKIP

        See Also
        --------
        from_dict : Reconstruct from dictionary
        to_file : Save to disk with efficient binary format

        Notes
        -----
        This method creates an in-memory representation. For disk storage,
        use `to_file()` instead, which stores arrays in efficient binary
        format (npz) rather than converting to lists.

        """
        from neurospatial.io import to_dict as _to_dict

        return _to_dict(cast("Environment", self))

    @classmethod
    def from_dict(cls: type[EnvironmentProtocol], data: dict[str, Any]) -> Environment:
        """Reconstruct Environment from dictionary representation.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary from `to_dict()` or equivalent structure.

        Returns
        -------
        Environment
            Reconstructed instance.

        Examples
        --------
        >>> env = Environment.from_samples(data, bin_size=2.0)  # doctest: +SKIP
        >>> env_dict = env.to_dict()  # doctest: +SKIP
        >>> reconstructed = Environment.from_dict(env_dict)  # doctest: +SKIP
        >>> assert env == reconstructed  # doctest: +SKIP

        See Also
        --------
        to_dict : Convert to dictionary
        from_file : Load from disk files

        """
        from neurospatial.io import from_dict as _from_dict

        return _from_dict(data)

    def to_nwb(
        self: EnvironmentProtocol,
        nwbfile: Any,
        name: str = "spatial_environment",
        *,
        overwrite: bool = False,
    ) -> None:
        """Write Environment to NWB file scratch space.

        Serializes the Environment to the NWB file's scratch space, allowing
        it to be stored alongside neural data and retrieved later using
        ``Environment.from_nwb()``.

        The stored data includes:

        - bin_centers and connectivity graph structure
        - dimension_ranges and edge weights
        - regions (points and polygons)
        - metadata (units, frame, name, layout_type)

        Parameters
        ----------
        nwbfile : NWBFile
            The NWB file to write to. Must be a pynwb.NWBFile instance.
        name : str, default "spatial_environment"
            Name for the environment in scratch/. Use this name when loading
            with ``Environment.from_nwb(nwbfile, scratch_name=name)``.
        overwrite : bool, default False
            If True, replace existing environment with the same name.
            If False, raise ValueError on duplicate name.

        Raises
        ------
        ValueError
            If environment with same name exists and overwrite=False.
        RuntimeError
            If Environment is not fitted (must be created with factory methods).
        ImportError
            If pynwb is not installed.

        See Also
        --------
        from_nwb : Load Environment from NWB file.
        neurospatial.io.nwb.write_environment : Low-level NWB writing function.

        Examples
        --------
        Save environment to NWB file:

        >>> from pynwb import NWBHDF5IO, NWBFile  # doctest: +SKIP
        >>> from datetime import datetime  # doctest: +SKIP
        >>> from neurospatial import Environment  # doctest: +SKIP
        >>> import numpy as np  # doctest: +SKIP
        >>> import tempfile  # doctest: +SKIP
        >>> from pathlib import Path  # doctest: +SKIP
        >>>
        >>> # Create environment
        >>> positions = np.random.rand(1000, 2) * 100  # doctest: +SKIP
        >>> env = Environment.from_samples(positions, bin_size=5.0)  # doctest: +SKIP
        >>> env.units = "cm"  # doctest: +SKIP
        >>>
        >>> # Save to NWB
        >>> nwbfile = NWBFile(  # doctest: +SKIP
        ...     session_description="Test session",
        ...     identifier="test_001",
        ...     session_start_time=datetime.now().astimezone(),
        ... )
        >>> env.to_nwb(nwbfile, name="linear_track")  # doctest: +SKIP
        >>>
        >>> # Write to disk (use a temp directory, not the CWD)
        >>> path = Path(tempfile.mkdtemp()) / "session.nwb"  # doctest: +SKIP
        >>> with NWBHDF5IO(path, "w") as io:  # doctest: +SKIP
        ...     io.write(nwbfile)

        Load environment back:

        >>> with NWBHDF5IO(path, "r") as io:  # doctest: +SKIP
        ...     nwbfile = io.read()
        ...     loaded_env = Environment.from_nwb(nwbfile, scratch_name="linear_track")

        """
        # Lazy import to keep pynwb optional
        try:
            from neurospatial.io.nwb import write_environment
        except ImportError as e:
            raise ImportError(
                "pynwb is required for NWB integration. Install with: pip install pynwb"
            ) from e

        write_environment(nwbfile, self, name=name, overwrite=overwrite)
