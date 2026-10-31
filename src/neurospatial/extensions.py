"""Let neurospatial accept spatial objects defined by another package.

A lab or tool may have its own class for a track, an arena, a bin layout or
a position-to-bin lookup. An extension package teaches neurospatial to use
those objects by installing a :class:`SnapshotAdapter`. The adapter converts
one external type into neurospatial's own records (a ``Binning``, a
``TrackLinearization``, an arena or a bin locator from
:mod:`neurospatial.binning` and :mod:`neurospatial.arena`) and reads files
saved in that package's format. The environment keeps only the converted
records, so an environment saved after conversion loads without the
extension package.

Declare the adapter object in the extension package's ``pyproject.toml``
under the ``neurospatial.snapshot_adapters.v1`` entry-point group, then
install the package::

    [project.entry-points."neurospatial.snapshot_adapters.v1"]
    recorded_sites = "my_lab.adapters:adapter"

Here ``my_lab/adapters.py`` defines ``adapter = SnapshotAdapter(...)``.
Check environments built through the adapter with
:func:`neurospatial.testing.assert_environment_contract`.
"""

from neurospatial.geometry import SnapshotAdapter

__all__ = ["SnapshotAdapter"]
