"""Shared ``Literal`` aliases for argument choices used across the package.

Kept in a dependency-free leaf module (imports only ``typing``) so the ``ops``
implementations and the ``environment``, ``encoding``, ``behavior`` and
``decoding`` layers can name the same alias without triggering import cycles.
Validation lives next to each consumer; where a check restates a whole
alias, it reads the members with ``typing.get_args`` so the two cannot drift.
"""

from __future__ import annotations

from typing import Literal

# Kernel normalization mode, shared by the kernel cache, ``compute_kernel``,
# ``smooth``, and ``diffuse`` across the ops and environment layers.
KernelMode = Literal["transition", "density", "average"]

# How distance between two places is measured: straight line, or shortest path
# through the environment's bins.
DistanceMetric = Literal["euclidean", "geodesic"]

# Array backend a caller may request; "auto" picks JAX when it is available.
BackendType = Literal["numpy", "jax", "auto"]

# Array backend after "auto" has been resolved.
ResolvedBackend = Literal["numpy", "jax"]
