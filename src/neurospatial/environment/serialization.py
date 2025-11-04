"""Serialization methods for Environment.

This module provides serialization and deserialization methods for the
Environment class, including:

- Modern JSON + npz format (recommended)
- Legacy pickle format (deprecated, for backward compatibility)
- In-memory dictionary format

The methods in this mixin delegate to the implementations in `neurospatial.io`
to maintain separation of concerns and avoid code duplication.

Classes
-------
EnvironmentSerialization
    Mixin class providing save/load and serialization methods.

Notes
-----
This is a plain mixin class (NOT a dataclass). It is designed to be mixed
into the Environment dataclass defined in `neurospatial.environment.core`.

The TYPE_CHECKING guard is used to prevent circular imports while still
providing type hints for IDEs and type checkers.

"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment

logger = logging.getLogger(__name__)


class EnvironmentSerialization:
    """Mixin providing serialization and deserialization methods.

    This mixin provides methods for saving and loading Environment instances
    using various formats:

    - **JSON + npz** (recommended): Stable, cross-platform, no arbitrary code execution
    - **Pickle** (deprecated): Legacy format, security risk, use only for trusted files
    - **Dictionary**: In-memory representation for programmatic access

    Methods
    -------
    to_file(path)
        Save environment to JSON + npz files (recommended).
    from_file(path)
        Load environment from JSON + npz files (recommended).
    to_dict()
        Convert environment to dictionary representation.
    from_dict(data)
        Reconstruct environment from dictionary.
    save(filename)
        Save environment using pickle (deprecated, security risk).
    load(filename)
        Load environment from pickle file (deprecated, security risk).

    Notes
    -----
    This is a plain class (NOT a dataclass). Only the Environment class itself
    should be a dataclass.

    All methods delegate to implementations in `neurospatial.io` to maintain
    separation of concerns and avoid code duplication.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> data = np.random.rand(100, 2) * 10
    >>> env = Environment.from_samples(data, bin_size=2.0)
    >>> env.to_file("my_environment")  # Creates .json and .npz files
    >>> loaded = Environment.from_file("my_environment")
    >>> assert env == loaded

    See Also
    --------
    neurospatial.io : Implementation module for serialization logic.

    """

    def to_file(self: Environment, path: str | Path) -> None:
        """Save Environment to versioned JSON + npz files.

        This method provides stable, reproducible serialization that is safer
        than pickle and compatible across Python versions. Creates two files:
        `{path}.json` (metadata) and `{path}.npz` (arrays).

        Parameters
        ----------
        path : str or Path
            Base path for output files (without extension).
            Will create `{path}.json` and `{path}.npz`.

        Examples
        --------
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> env.to_file("my_environment")

        See Also
        --------
        from_file : Load environment from saved files
        save : Legacy pickle-based serialization
        to_dict : Convert to dictionary for in-memory use

        Notes
        -----
        This format is safer than pickle (no arbitrary code execution) and
        more portable across Python versions and platforms.

        The JSON file contains metadata, regions, and serialization version,
        while the npz file contains numpy arrays for efficiency.

        """
        from neurospatial.io import to_file as _to_file

        _to_file(self, path)

    @classmethod
    def from_file(cls: type[Environment], path: str | Path) -> Environment:
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
        >>> env = Environment.from_file("my_environment")

        See Also
        --------
        to_file : Save environment to files
        load : Legacy pickle-based deserialization
        from_dict : Reconstruct from dictionary

        Notes
        -----
        This method can load environments saved with any serialization version,
        providing forward and backward compatibility.

        """
        from neurospatial.io import from_file as _from_file

        return _from_file(path)

    def to_dict(self: Environment) -> dict[str, Any]:
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
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> env_dict = env.to_dict()
        >>> # Dictionary can be converted to JSON
        >>> import json
        >>> json_str = json.dumps(env_dict)

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

        return _to_dict(self)

    @classmethod
    def from_dict(cls: type[Environment], data: dict[str, Any]) -> Environment:
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
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> env_dict = env.to_dict()
        >>> reconstructed = Environment.from_dict(env_dict)
        >>> assert env == reconstructed

        See Also
        --------
        to_dict : Convert to dictionary
        from_file : Load from disk files

        """
        from neurospatial.io import from_dict as _from_dict

        return _from_dict(data)

    def save(self: Environment, filename: str = "environment.pkl") -> None:
        """Save the Environment object to a file using pickle.

        .. deprecated:: 0.1.0
            Use `to_file()` instead. Pickle is less secure and less portable.

        Parameters
        ----------
        filename : str, optional
            The name of the file to save the environment to.
            Defaults to "environment.pkl".

        Warnings
        --------
        This method uses pickle for serialization. Pickle files can execute
        arbitrary code during deserialization. Only share pickle files with
        trusted users and only load files from trusted sources.

        **Security Risk**: Do not use pickle files from untrusted sources.

        See Also
        --------
        load : Load an Environment from a pickle file.
        to_file : Recommended alternative using JSON + npz format.

        Notes
        -----
        This method is retained for backward compatibility but is deprecated.
        New code should use `to_file()` which is safer and more portable.

        """
        with Path(filename).open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Environment saved to %s", filename)

    @classmethod
    def load(cls: type[Environment], filename: str) -> Environment:
        """Load an Environment object from a pickled file.

        .. deprecated:: 0.1.0
            Use `from_file()` instead. Pickle is less secure and less portable.

        Parameters
        ----------
        filename : str
            The name of the file to load the environment from.

        Returns
        -------
        Environment
            The loaded Environment object.

        Raises
        ------
        TypeError
            If the loaded object is not an instance of the Environment class.

        Warnings
        --------
        This method uses pickle for deserialization. **Only load files from
        trusted sources**, as pickle can execute arbitrary code during
        deserialization. Do not load pickle files from untrusted or
        unknown sources.

        **Security Risk**: Pickle can execute arbitrary code during loading.

        See Also
        --------
        save : Save an Environment to a pickle file.
        from_file : Recommended alternative using JSON + npz format.

        Notes
        -----
        This method is retained for backward compatibility but is deprecated.
        New code should use `from_file()` which is safer and more portable.

        """
        with Path(filename).open("rb") as fh:
            environment = pickle.load(fh)
        if not isinstance(environment, cls):
            raise TypeError(f"Loaded object is not type {cls.__name__}")
        return environment
