"""The names of the checks a dataset session must pass to load."""

from __future__ import annotations

__all__ = ["REFUSING_CHECKS"]

#: The checks whose failure refuses a load unless it is accepted with
#: ``accept_failed_checks``.
REFUSING_CHECKS = ("clock", "units")
