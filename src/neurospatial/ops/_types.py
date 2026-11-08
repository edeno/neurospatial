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

# How distance between two places is measured: the straight line, the
# shortest path inside the environment at any angle ("geodesic"), or the walk
# along the connections between bin centers ("bin_graph").
DistanceMetric = Literal["euclidean", "geodesic", "bin_graph"]

# The two choices of the functions that have no bin-graph walk.
StraightOrGeodesic = Literal["euclidean", "geodesic"]

# The two path lengths, for functions that measure no straight lines.
PathMetric = Literal["geodesic", "bin_graph"]

# Array backend a caller may request; "auto" picks JAX when it is available.
BackendType = Literal["numpy", "jax", "auto"]

# Array backend after "auto" has been resolved.
ResolvedBackend = Literal["numpy", "jax"]
