"""Shared low-level type aliases for kernel operations.

Kept in a dependency-free leaf module (imports only ``typing``) so both the
``ops`` implementations and the ``environment`` methods/Protocol can name the
same alias without triggering import cycles.
"""

from typing import Literal

# Kernel normalization mode, shared by the kernel cache, ``compute_kernel``,
# ``smooth``, and ``diffuse`` across the ops and environment layers. One alias so
# the accepted set cannot drift between the type annotations. (Runtime validation
# lives next to each consumer as an explicit membership check.)
KernelMode = Literal["transition", "density", "average"]
