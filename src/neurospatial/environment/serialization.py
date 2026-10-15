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

from neurospatial.environment._protocols import SerializationConsumer

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment


def _to_file(
    env: SerializationConsumer, path: str | Path, *, overwrite: bool = False
) -> None:
    """Implement Environment.to_file using explicit serialization inputs."""
    from neurospatial.io import to_file as _to_file

    _to_file(cast("Environment", env), path, overwrite=overwrite)


def _from_file(cls: type[Environment[Any]], path: str | Path) -> Environment:
    """Implement Environment.from_file using explicit construction inputs."""
    from neurospatial.io import from_file as _from_file

    return _from_file(path)


def _to_dict(env: SerializationConsumer) -> dict[str, Any]:
    """Implement Environment.to_dict using explicit serialization inputs."""
    from neurospatial.io import to_dict as _to_dict

    return _to_dict(cast("Environment", env))


def _from_dict(cls: type[Environment[Any]], data: dict[str, Any]) -> Environment:
    """Implement Environment.from_dict using explicit construction inputs."""
    from neurospatial.io import from_dict as _from_dict

    return _from_dict(data)


def _to_nwb(
    env: SerializationConsumer,
    nwbfile: Any,
    name: str = "spatial_environment",
    *,
    overwrite: bool = False,
) -> None:
    """Implement Environment.to_nwb using explicit serialization inputs."""
    try:
        from neurospatial.io.nwb import write_environment
    except ImportError as e:
        raise ImportError(
            "pynwb is required for NWB integration. Install with: pip install pynwb"
        ) from e

    write_environment(nwbfile, env, name=name, overwrite=overwrite)
