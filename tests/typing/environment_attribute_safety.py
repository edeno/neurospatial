"""Runtime coordinate filtering must not admit missing static capabilities."""

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from neurospatial import Environment
    from neurospatial.geometry._coordinates import EgocentricPolarSystem

    class MissingOperation(Protocol):
        def nonexistent_operation(self) -> int: ...

    def _check(env: Environment, polar: Environment[EgocentricPolarSystem]) -> None:
        env.occpuancy("misspelled method")  # type: ignore[attr-defined]
        _ = env.nonexistent_attribute  # type: ignore[attr-defined]
        _missing: MissingOperation = env  # type: ignore[assignment]
        _polar_missing: MissingOperation = polar  # type: ignore[assignment]
