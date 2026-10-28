"""Region membership tables for Environment.

Which bins a region covers is decided by
:func:`neurospatial.geometry.region_bin_mask`; this module only
arranges that rule's answers for the region methods.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import RegionNotFoundError, _format_error
from neurospatial.geometry import Binning, region_bin_mask, region_bin_masks
from neurospatial.regions import Region, Regions, _snapshot_regions

if TYPE_CHECKING:
    from neurospatial.environment._cache import AnalysisCache
    from neurospatial.geometry import Arena


def _region_values(regions: Mapping[object, object]) -> list[Region]:
    """The Region values of ``regions``, refusing anything else by name."""
    values = []
    for key, region in regions.items():
        if not isinstance(region, Region):
            raise TypeError(
                _format_error(
                    f"regions[{key!r}] is a {type(region).__name__}, not a Region.",
                    fix="make it with Region(name, 'point' or 'polygon', data); "
                    "attach landmarks with env.with_landmarks(Landmarks([...]))",
                )
            )
        if key != region.name:
            raise ValueError(
                _format_error(
                    f"regions[{key!r}] holds the region named {region.name!r}.",
                    fix="key each region by its name, as Regions does",
                )
            )
        values.append(region)
    return values


def _relabeled_region(
    region: Region, old_units: str | None, new_units: str | None
) -> Region:
    """Return ``region`` with its unit label moved from ``old_units`` to ``new_units``.

    Used when the environment's unit label changes but its coordinates keep
    their lengths. A region's recorded ``metadata["coord_system"]`` follows
    the environment's only when the two agreed; a region with its own label
    on an unlabelled environment keeps it, and labelling such an
    environment in other units is refused because nothing converted the
    region's coordinates.

    Raises
    ------
    ValueError
        If the region records units other than ``old_units`` and
        ``new_units`` is set and differs from them.
    """
    label = region.metadata.get("coord_system")
    if label is None or label == new_units:
        return region
    metadata = dict(region.metadata)
    if label == old_units:
        if new_units is None:
            del metadata["coord_system"]
        else:
            metadata["coord_system"] = new_units
        return Region(region.name, region.kind, region.data, metadata)
    if new_units is None:
        return region
    raise ValueError(
        _format_error(
            f"Region {region.name!r} is in {label!r} (its "
            f"metadata['coord_system']), but the environment is being labelled "
            f"{new_units!r} without converting its coordinates.",
            why="Why: a unit label describes coordinates and never rescales "
            f"them, so the region would stay in {label!r} on an environment "
            f"in {new_units!r} and cover the wrong bins.",
            fix=f"convert the coordinates with apply_transform(transform, "
            f"units={new_units!r}) using a {label}-to-{new_units} transform, "
            "or load the annotation with its calibration",
        )
    )


def _relabeled_regions(
    regions: Regions, old_units: str | None, new_units: str | None
) -> Regions:
    """Return ``regions`` relabelled from ``old_units`` to ``new_units``."""
    return _snapshot_regions(
        _relabeled_region(region, old_units, new_units) for region in regions.values()
    )


def _owned_regions(regions: object, n_dims: int, units: str | None) -> Regions:
    """Return an immutable snapshot of ``regions`` for an ``n_dims`` environment.

    ``units`` is the environment's unit label. A region whose
    ``metadata["coord_system"]`` (set by the annotation readers to
    ``"pixels"`` or ``"cm"``) names other units is refused.

    Raises
    ------
    TypeError
        If ``regions`` is not a mapping of names to Region values.
    ValueError
        If a key differs from its region's name, a region's coordinates
        have a different dimension, or its coordinate units differ.
    """
    if not isinstance(regions, Mapping):
        raise TypeError(
            _format_error(
                f"regions must map names to Region values, got "
                f"{type(regions).__name__}.",
                fix="build them with Regions([...]) and pass the builder",
            )
        )
    values = _region_values(regions)
    for region in values:
        if region.n_dims != n_dims:
            raise ValueError(
                _format_error(
                    f"Region {region.name!r} has {region.n_dims} coordinates; "
                    f"the environment has {n_dims}.",
                    fix="give regions coordinates in the environment's frame",
                )
            )
        region_units = region.metadata.get("coord_system")
        if units is not None and region_units is not None and region_units != units:
            raise ValueError(
                _format_error(
                    f"Region {region.name!r} is in {region_units!r} (its "
                    f"metadata['coord_system']), but the environment is in "
                    f"{units!r}.",
                    why="Why: a region covers the bins whose centers lie in "
                    "it, so a region drawn in video pixels would cover the "
                    "wrong bins of an environment in centimeters.",
                    fix="load the annotation with its calibration "
                    "(calibration= for the annotation readers, pixel_to_world= "
                    "for load_cvat_xml or load_labelme_json) so its coordinates "
                    f"are in {units!r}; if they already are, rebuild the region "
                    f"with metadata['coord_system'] = {units!r}",
                )
            )
    return _snapshot_regions(values)


def _bins_in_region(
    region_name: str,
    regions: Mapping[str, Region],
    *,
    binning: Binning,
    arena: Arena,
    cache: AnalysisCache,
    include_boundary: bool,
) -> NDArray[np.intp]:
    """Indices of the bins the named region covers, in increasing order."""
    if region_name not in regions:
        raise RegionNotFoundError(
            region_name, available=list(regions), argument="region_name"
        )
    mask = region_bin_mask(
        regions[region_name],
        binning=binning,
        arena=arena,
        include_boundary=include_boundary,
        indexes=cache,
    )
    return np.flatnonzero(mask).astype(np.intp, copy=False)


def _region_membership(
    regions: object,
    *,
    binning: Binning,
    arena: Arena,
    cache: AnalysisCache,
    include_boundary: bool = True,
) -> NDArray[np.bool_]:
    """Membership of every bin in every region, ``(n_bins, n_regions)``."""
    if not isinstance(regions, Mapping):
        names = list(regions) if isinstance(regions, (list, tuple)) else ["goal"]
        raise TypeError(
            _format_error(
                f"region_membership() takes a mapping of names to regions, such "
                f"as env.regions or a Regions builder, or None; got "
                f"{type(regions).__name__}.",
                why="Why: it returns one column per region in the mapping; "
                "a list of names says which regions but not their shapes.",
                fix="select the regions from env.regions: "
                f"env.region_membership({{name: env.regions[name] for name in "
                f"{names!r}}}), or env.region_mask({names!r}) for one mask over "
                "all of them",
            )
        )
    values = _region_values(regions)
    if not isinstance(include_boundary, bool):
        raise TypeError(
            f"include_boundary must be a bool, got {type(include_boundary).__name__}"
        )
    return region_bin_masks(
        values,
        binning=binning,
        arena=arena,
        include_boundary=include_boundary,
        indexes=cache,
    )
