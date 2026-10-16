"""Text and notebook presentation of published environment metadata."""

from __future__ import annotations

import numpy as np

from neurospatial.environment._protocols import SummaryConsumer


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


def _repr(env: SummaryConsumer) -> str:
    """Return the existing single-line diagnostic representation."""
    # Use ``repr(env.name)`` so the empty-string case ``name=""``
    # is visibly distinct from the unset case ``name=None``.
    # The previous ``f"'{name}'" if name else 'None'`` collapsed the
    # two into the same string and hid the difference from users
    # debugging an Environment they thought they had named.
    cls_name = type(env).__name__
    # Show name, dims, bins, layout. Truncate very long names but keep
    # ``None`` distinct from ``""``.
    name = env.name
    if isinstance(name, str) and len(name) > 40:
        name = name[:37] + "..."

    try:
        dims_str = f"{env.n_dims}D"
    except (RuntimeError, AttributeError):
        dims_str = "?D"

    n_bins = env.bin_centers.shape[0] if hasattr(env, "bin_centers") else 0

    layout_type = env._layout_type_used or "Unknown"
    if layout_type.endswith("Layout"):
        layout_type = layout_type[:-6]  # Remove 'Layout' suffix

    return f"{cls_name}(name={name!r}, {dims_str}, {n_bins} bins, {layout_type})"


def _str(env: SummaryConsumer) -> str:
    """Return the diagnostic summary."""
    return env.info()


def _repr_html(env: SummaryConsumer) -> str:
    """Format the existing notebook table from published metadata."""
    import html

    # Escape HTML special characters in name
    name = html.escape(str(env.name) if env.name else "None")

    # Build table rows
    rows = []
    rows.append(_html_table_header(f"Environment: {name}"))

    rows.append(_html_table_row("Layout Type", env._layout_type_used or "Unknown"))

    # Dimensions and bins
    try:
        n_dims = env.n_dims
        rows.append(_html_table_row("Dimensions", str(n_dims)))
    except (RuntimeError, AttributeError):
        rows.append(_html_table_row("Dimensions", "Unknown"))
        n_dims = None

    n_bins = env.bin_centers.shape[0] if hasattr(env, "bin_centers") else 0
    rows.append(_html_table_row("Number of Bins", str(n_bins)))

    # Spatial extent
    if hasattr(env, "dimension_ranges") and env.dimension_ranges:
        extent_parts = []
        for dim_idx, (min_val, max_val) in enumerate(env.dimension_ranges):
            extent_parts.append(f"dim{dim_idx}: [{min_val:.2f}, {max_val:.2f}]")
        extent_str = "<br>".join(extent_parts)
        rows.append(_html_table_row("Spatial Extent", extent_str))

    # Regions
    n_regions = len(env.regions) if hasattr(env, "regions") else 0
    if n_regions > 0:
        rows.append(_html_table_row("Regions", f"{n_regions} defined"))
    else:
        rows.append(_html_table_row("Regions", "None"))

    # 1D-specific info
    if n_dims == 1 and hasattr(env, "is_linearized_track") and env.is_linearized_track:
        rows.append(_html_table_row("Linearization", "Available (1D environment)"))

    return (
        '<div style="margin: 10px;">'
        '<table style="border-collapse: collapse; border: 1px solid #ddd; '
        'font-family: monospace; font-size: 12px;">'
        f"{''.join(rows)}"
        "</table></div>"
    )


def _info(env: SummaryConsumer) -> str:
    """Format the existing multi-line geometry and region summary."""
    # Build output line by line
    lines = []

    # Header
    lines.append("Environment Information")
    lines.append("=" * 23)
    lines.append("")

    # Basic information
    name_display = env.name if env.name else "(unnamed)"
    lines.append(f"Name: {name_display}")
    lines.append(f"Layout Type: {env.layout_type}")
    lines.append(f"Dimensions: {env.n_dims}")
    lines.append(f"Number of Bins: {env.n_bins}")
    lines.append("")

    # Spatial extent
    if env.dimension_ranges is not None:
        lines.append("Spatial Extent:")
        for dim_idx, (dim_min, dim_max) in enumerate(env.dimension_ranges):
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
    try:
        bin_sizes_array = env.bin_sizes

        # Check if all bins have the same size (uniform)
        if np.allclose(bin_sizes_array, bin_sizes_array[0]):
            # Uniform bin size - for grids, extract per-dimension from grid_edges
            if env.grid_edges and all(len(e) > 1 for e in env.grid_edges):
                for dim_idx, edges in enumerate(env.grid_edges):
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
                    "Size"
                    if env.n_dims == 1
                    else "Area"
                    if env.n_dims == 2
                    else "Volume"
                )
                lines.append(f"  {measure_name}: {bin_sizes_array[0]:.2f}")
        else:
            # Variable bin sizes
            lines.append(
                f"  Variable (mean: {np.mean(bin_sizes_array):.2f}, "
                f"std: {np.std(bin_sizes_array):.2f}, "
                f"range: [{np.min(bin_sizes_array):.2f}, {np.max(bin_sizes_array):.2f}])"
            )
    except (AttributeError, RuntimeError, ValueError):
        lines.append("  (not available)")
    lines.append("")

    # Regions
    n_regions = len(env.regions) if env.regions else 0
    if n_regions > 0:
        lines.append(f"Regions: {n_regions} defined")
        # Show region names if not too many
        if n_regions <= 5:
            for region_name in env.regions:
                lines.append(f"  - {region_name}")
        else:
            lines.append("  (use env.regions to inspect all regions)")
    else:
        lines.append("Regions: None")
    lines.append("")

    # 1D-specific information
    if hasattr(env, "is_linearized_track") and env.is_linearized_track:
        lines.append("Linearization: Available (1D environment)")
        lines.append("")

    return "\n".join(lines)
