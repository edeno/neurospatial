"""Installed adapters from external spatial data to built-in owned snapshots.

Adapters are objects declared in the ``neurospatial.snapshot_adapters.v1``
entry-point group. Publication retains their normalized data, never their
provider objects or callbacks. Saved normalized records use built-in codecs.

The implementation lives in ``neurospatial.geometry``; this module is the
public import path.
"""

from neurospatial.geometry import SnapshotAdapter

__all__ = ["SnapshotAdapter"]
