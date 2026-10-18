"""Text and notebook presentation of published environment metadata."""

from __future__ import annotations

import html
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error


def _html_table_row(label: str, value: str, highlight: bool = False) -> str:
    """Generate a single HTML table row.

    Parameters
    ----------
    label : str
        Row label (left column).
    value : str
        Row value (right column).
    highlight : bool, default=False
        If True, use highlighted background color.

    Returns
    -------
    str
        HTML string for the table row.

    """
    bg_color = "#fffacd" if highlight else "#fff"
    return (
        f'<tr style="background-color: {bg_color};">'
        f'<td style="padding: 6px 12px; border-top: 1px solid #ddd; '
        f'font-weight: bold; color: #555;">{label}</td>'
        f'<td style="padding: 6px 12px; border-top: 1px solid #ddd; '
        f'color: #000;">{value}</td>'
        "</tr>"
    )


def _html_table_header(title: str) -> str:
    """Generate HTML table header row.

    Parameters
    ----------
    title : str
        Title text for the header (already HTML-escaped).

    Returns
    -------
    str
        HTML string for the header row.

    """
    return (
        '<tr style="background-color: #f0f0f0; border-bottom: 2px solid #999;">'
        '<th colspan="2" style="padding: 8px; text-align: left; '
        'font-weight: bold; font-size: 14px;">'
        f"{title}"
        "</th></tr>"
    )


def _repr(
    *, class_name: str, name: str, n_dims: int, n_bins: int, layout_type: str
) -> str:
    """Return the single-line diagnostic representation."""
    # Use ``repr(name)`` so the empty-string case ``name=""`` is visibly
    # distinct in the output. Truncate very long names.
    if len(name) > 40:
        name = name[:37] + "..."

    if layout_type.endswith("Layout"):
        layout_type = layout_type[:-6]  # Remove 'Layout' suffix

    return f"{class_name}(name={name!r}, {n_dims}D, {n_bins} bins, {layout_type})"


def _repr_html(
    *,
    name: str,
    layout_type: str,
    n_dims: int,
    n_bins: int,
    dimension_ranges: Sequence[tuple[float, float]] | None,
    n_regions: int,
    is_linearized_track: bool,
) -> str:
    """Format the notebook table from published metadata."""
    # Escape HTML special characters in name
    escaped_name = html.escape(name if name else "None")

    # Build table rows
    rows = []
    rows.append(_html_table_header(f"Environment: {escaped_name}"))

    rows.append(_html_table_row("Layout Type", layout_type))
    rows.append(_html_table_row("Dimensions", str(n_dims)))
    rows.append(_html_table_row("Number of Bins", str(n_bins)))

    # Spatial extent
    if dimension_ranges:
        extent_parts = []
        for dim_idx, (min_val, max_val) in enumerate(dimension_ranges):
            extent_parts.append(f"dim{dim_idx}: [{min_val:.2f}, {max_val:.2f}]")
        extent_str = "<br>".join(extent_parts)
        rows.append(_html_table_row("Spatial Extent", extent_str))

    # Regions
    if n_regions > 0:
        rows.append(_html_table_row("Regions", f"{n_regions} defined"))
    else:
        rows.append(_html_table_row("Regions", "None"))

    # 1D-specific info
    if n_dims == 1 and is_linearized_track:
        rows.append(_html_table_row("Linearization", "Available (1D environment)"))

    return (
        '<div style="margin: 10px;">'
        '<table style="border-collapse: collapse; border: 1px solid #ddd; '
        'font-family: monospace; font-size: 12px;">'
        f"{''.join(rows)}"
        "</table></div>"
    )


_MEASURE_NAMES = {1: "Length", 2: "Area", 3: "Volume"}


def _mixed_measure_error(measure_dims: NDArray[np.intp], units: str | None) -> str:
    """Explain why one array cannot hold lengths and areas together."""
    unit = units or "units"
    counts = ", ".join(
        f"{int(np.count_nonzero(measure_dims == dim))} by "
        f"{_MEASURE_NAMES.get(int(dim), 'volume').lower()} ({unit}"
        f"{'' if dim == 1 else f'^{int(dim)}'})"
        for dim in np.unique(measure_dims)
    )
    return _format_error(
        f"bin_sizes would mix measures: {counts}.",
        why="Why: these bins come from components of different dimension, such "
        "as a track combined with an open field, so a sum, ratio or density "
        "over all bin sizes has no unit.",
        fix="analyze one component at a time with env.subset(mask=...), using "
        "env.bin_sources to find each component's bins, or read each bin's "
        "measure and its dimension from env.binning.bins.sizes and "
        "env.binning.bins.measure_dims",
    )


def _info(
    *,
    name: str,
    layout_type: str,
    n_dims: int,
    bin_sizes: NDArray[np.float64],
    measure_dims: NDArray[np.intp],
    dimension_ranges: Sequence[tuple[float, float]] | None,
    grid_edges: tuple[NDArray[np.float64], ...] | None,
    region_names: Sequence[str],
    is_linearized_track: bool,
) -> str:
    """Format the multi-line geometry and region summary."""
    # Build output line by line
    lines = []

    # Header
    lines.append("Environment Information")
    lines.append("=" * 23)
    lines.append("")

    # Basic information
    name_display = name if name else "(unnamed)"
    lines.append(f"Name: {name_display}")
    lines.append(f"Layout Type: {layout_type}")
    lines.append(f"Dimensions: {n_dims}")
    lines.append(f"Number of Bins: {len(bin_sizes)}")
    lines.append("")

    # Spatial extent
    if dimension_ranges is not None:
        lines.append("Spatial Extent:")
        for dim_idx, (dim_min, dim_max) in enumerate(dimension_ranges):
            dim_range = dim_max - dim_min
            lines.append(
                f"  Dimension {dim_idx}: [{dim_min:.2f}, {dim_max:.2f}] "
                f"(range: {dim_range:.2f})"
            )
        lines.append("")
    else:
        lines.append("Spatial Extent: Not available")
        lines.append("")

    # Bin sizes
    lines.append("Bin Sizes:")
    dims = np.unique(measure_dims)
    if len(dims) > 1:
        # Lengths and areas have no common summary statistic.
        for dim in dims:
            selected = bin_sizes[measure_dims == dim]
            lines.append(
                f"  {_MEASURE_NAMES.get(int(dim), 'Volume')} of {len(selected)} bins: "
                f"mean {np.mean(selected):.2f}, "
                f"range [{np.min(selected):.2f}, {np.max(selected):.2f}]"
            )
    elif np.allclose(bin_sizes, bin_sizes[0]):
        # Uniform bin size - for grids, extract per-dimension from grid_edges
        if grid_edges and all(len(e) > 1 for e in grid_edges):
            for dim_idx, edges in enumerate(grid_edges):
                dim_sizes = np.diff(edges)
                if np.allclose(dim_sizes, dim_sizes[0]):
                    lines.append(f"  Dimension {dim_idx}: {dim_sizes[0]:.2f}")
                else:
                    lines.append(
                        f"  Dimension {dim_idx}: variable "
                        f"(mean: {np.mean(dim_sizes):.2f}, "
                        f"std: {np.std(dim_sizes):.2f})"
                    )
        else:
            # Non-grid layout or 1D - show the uniform measure
            measure_name = (
                "Size" if n_dims == 1 else "Area" if n_dims == 2 else "Volume"
            )
            lines.append(f"  {measure_name}: {bin_sizes[0]:.2f}")
    else:
        # Variable bin sizes
        lines.append(
            f"  Variable (mean: {np.mean(bin_sizes):.2f}, "
            f"std: {np.std(bin_sizes):.2f}, "
            f"range: [{np.min(bin_sizes):.2f}, {np.max(bin_sizes):.2f}])"
        )
    lines.append("")

    # Regions
    n_regions = len(region_names)
    if n_regions > 0:
        lines.append(f"Regions: {n_regions} defined")
        # Show region names if not too many
        if n_regions <= 5:
            for region_name in region_names:
                lines.append(f"  - {region_name}")
        else:
            lines.append("  (use env.regions to inspect all regions)")
    else:
        lines.append("Regions: None")
    lines.append("")

    # 1D-specific information
    if is_linearized_track:
        lines.append("Linearization: Available (1D environment)")
        lines.append("")

    return "\n".join(lines)
