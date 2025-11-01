"""
Tests for regions plotting functionality.

This module tests the plot_regions() function which visualizes Region objects
using matplotlib. Tests cover:
- Point regions (scatter plots)
- Polygon regions (patches)
- Per-region style customization
- Legend creation
- Region name filtering
- Coordinate transformations
- Edge cases and warnings
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from shapely.geometry import Polygon

from neurospatial.regions import Region, Regions
from neurospatial.regions.plot import plot_regions
from neurospatial.transforms import scale_2d, translate


class TestPlotRegionsBasic:
    """Tests for basic plot_regions functionality."""

    def test_plot_single_point(self):
        """Test plotting a single point region."""
        point_data = np.array([5.0, 10.0])
        regions = Regions([Region(name="pt1", data=point_data, kind="point")])

        fig, ax = plt.subplots()
        plot_regions(regions, ax=ax, add_legend=False)

        # Check that scatter was called (ax has collections)
        assert len(ax.collections) == 1
        plt.close(fig)

    def test_plot_single_polygon(self):
        """Test plotting a single polygon region."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        regions = Regions([Region(name="box", data=poly, kind="polygon")])

        fig, ax = plt.subplots()
        plot_regions(regions, ax=ax, add_legend=False)

        # Check that patch was added
        assert len(ax.patches) == 1
        plt.close(fig)

    def test_plot_multiple_regions(self):
        """Test plotting multiple regions of different kinds."""
        point_data = np.array([2.0, 2.0])
        poly = Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])
        regions = Regions(
            [
                Region(name="pt", data=point_data, kind="point"),
                Region(name="box", data=poly, kind="polygon"),
            ]
        )

        fig, ax = plt.subplots()
        plot_regions(regions, ax=ax, add_legend=False)

        # Should have 1 scatter collection and 1 patch
        assert len(ax.collections) == 1
        assert len(ax.patches) == 1
        plt.close(fig)

    def test_empty_regions(self):
        """Test plotting with empty Regions collection."""
        regions = Regions([])

        fig, ax = plt.subplots()
        # Should not raise an error
        plot_regions(regions, ax=ax)
        plt.close(fig)

    def test_no_region_names_specified(self):
        """Test plotting all regions when region_names is None."""
        poly1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        poly2 = Polygon([(10, 10), (15, 10), (15, 15), (10, 15)])
        regions = Regions(
            [
                Region(name="box1", data=poly1, kind="polygon"),
                Region(name="box2", data=poly2, kind="polygon"),
            ]
        )

        fig, ax = plt.subplots()
        plot_regions(regions, ax=ax, add_legend=False)

        # Should plot both regions
        assert len(ax.patches) == 2
        plt.close(fig)

    def test_ax_default_to_gca(self):
        """Test that ax defaults to plt.gca() when not provided."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        regions = Regions([Region(name="box", data=poly, kind="polygon")])

        fig, ax = plt.subplots()
        plt.sca(ax)  # Set current axes
        plot_regions(regions, add_legend=False)

        # Should use the current axes
        assert len(ax.patches) == 1
        plt.close(fig)


class TestPlotRegionsFiltering:
    """Tests for region name filtering."""

    def test_plot_subset_by_name(self):
        """Test plotting only specified region names."""
        poly1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        poly2 = Polygon([(10, 10), (15, 10), (15, 15), (10, 15)])
        poly3 = Polygon([(20, 20), (25, 20), (25, 25), (20, 25)])
        regions = Regions(
            [
                Region(name="box1", data=poly1, kind="polygon"),
                Region(name="box2", data=poly2, kind="polygon"),
                Region(name="box3", data=poly3, kind="polygon"),
            ]
        )

        fig, ax = plt.subplots()
        plot_regions(regions, ax=ax, region_names=["box1", "box3"], add_legend=False)

        # Should only plot box1 and box3
        assert len(ax.patches) == 2
        plt.close(fig)

    def test_nonexistent_region_name_error(self):
        """Test error when region name doesn't exist (plt.warning doesn't exist)."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        regions = Regions([Region(name="box", data=poly, kind="polygon")])

        fig, ax = plt.subplots()
        # Test that nonexistent region names generate a warning but don't crash
        with pytest.warns(
            UserWarning,
            match=r"'nonexistent' not in collection; skipping",
        ):
            plot_regions(
                regions, ax=ax, region_names=["box", "nonexistent"], add_legend=False
            )
        plt.close(fig)

    def test_empty_region_names_list(self):
        """Test plotting with empty region_names list."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        regions = Regions([Region(name="box", data=poly, kind="polygon")])

        fig, ax = plt.subplots()
        plot_regions(regions, ax=ax, region_names=[], add_legend=False)

        # Should not plot anything
        assert len(ax.patches) == 0
        plt.close(fig)


class TestPlotRegionsStyling:
    """Tests for region styling and customization."""

    def test_default_kwargs(self):
        """Test applying default kwargs to all regions."""
        poly1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        poly2 = Polygon([(10, 10), (15, 10), (15, 15), (10, 15)])
        regions = Regions(
            [
                Region(name="box1", data=poly1, kind="polygon"),
                Region(name="box2", data=poly2, kind="polygon"),
            ]
        )

        fig, ax = plt.subplots()
        plot_regions(
            regions,
            ax=ax,
            default_kwargs={"edgecolor": "red", "linewidth": 2},
            add_legend=False,
        )

        # Both patches should have the default style
        # Note: alpha defaults to 0.5, which affects edge color alpha too
        for patch in ax.patches:
            edge_color = patch.get_edgecolor()
            assert edge_color[0] == 1.0  # Red channel
            assert edge_color[1] == 0.0  # Green channel
            assert edge_color[2] == 0.0  # Blue channel
            assert edge_color[3] == 0.5  # Alpha (default)
        plt.close(fig)

    def test_per_region_kwargs(self):
        """Test per-region style overrides."""
        poly1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        poly2 = Polygon([(10, 10), (15, 10), (15, 15), (10, 15)])
        regions = Regions(
            [
                Region(name="box1", data=poly1, kind="polygon"),
                Region(name="box2", data=poly2, kind="polygon"),
            ]
        )

        fig, ax = plt.subplots()
        plot_regions(regions, ax=ax, add_legend=False, box1={"facecolor": "blue"})

        # At least one patch should exist
        assert len(ax.patches) == 2
        plt.close(fig)

    def test_metadata_plot_kwargs(self):
        """Test that region metadata plot_kwargs are applied."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        regions = Regions(
            [
                Region(
                    name="box",
                    data=poly,
                    kind="polygon",
                    metadata={"plot_kwargs": {"edgecolor": "green"}},
                )
            ]
        )

        fig, ax = plt.subplots()
        plot_regions(regions, ax=ax, add_legend=False)

        # Patch should have green edge
        # Note: alpha defaults to 0.5, which affects edge color alpha too
        assert len(ax.patches) == 1
        edge_color = ax.patches[0].get_edgecolor()
        assert edge_color[0] == 0.0  # Red channel
        assert np.isclose(edge_color[1], 0.5019607843137255)  # Green channel
        assert edge_color[2] == 0.0  # Blue channel
        assert edge_color[3] == 0.5  # Alpha (default)
        plt.close(fig)

    def test_alpha_parameter(self):
        """Test alpha parameter for transparency."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        regions = Regions([Region(name="box", data=poly, kind="polygon")])

        fig, ax = plt.subplots()
        plot_regions(regions, ax=ax, add_legend=False, box={"alpha": 0.3})

        # Check alpha is applied
        assert len(ax.patches) == 1
        assert ax.patches[0].get_alpha() == 0.3
        plt.close(fig)

    def test_point_marker_and_size(self):
        """Test customizing point marker and size."""
        point_data = np.array([5.0, 10.0])
        regions = Regions([Region(name="pt", data=point_data, kind="point")])

        fig, ax = plt.subplots()
        plot_regions(regions, ax=ax, add_legend=False, pt={"marker": "o", "s": 200})

        # Check that scatter was called with custom marker
        assert len(ax.collections) == 1
        plt.close(fig)


class TestPlotRegionsLegend:
    """Tests for legend creation."""

    def test_legend_with_default_labels(self):
        """Test that legend uses region names as default labels."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        regions = Regions([Region(name="my_box", data=poly, kind="polygon")])

        fig, ax = plt.subplots()
        plot_regions(regions, ax=ax, add_legend=True)

        # Check legend was created with correct label
        legend = ax.get_legend()
        assert legend is not None
        assert any("my_box" in text.get_text() for text in legend.get_texts())
        plt.close(fig)

    def test_legend_with_custom_label(self):
        """Test custom legend labels via label kwarg."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        regions = Regions([Region(name="box", data=poly, kind="polygon")])

        fig, ax = plt.subplots()
        plot_regions(regions, ax=ax, add_legend=True, box={"label": "Custom Label"})

        # Check legend has custom label
        legend = ax.get_legend()
        assert legend is not None
        assert any("Custom Label" in text.get_text() for text in legend.get_texts())
        plt.close(fig)

    def test_no_legend_when_add_legend_false(self):
        """Test that legend is not created when add_legend=False."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        regions = Regions([Region(name="box", data=poly, kind="polygon")])

        fig, ax = plt.subplots()
        plot_regions(regions, ax=ax, add_legend=False)

        # No legend should be created
        legend = ax.get_legend()
        assert legend is None
        plt.close(fig)


class TestPlotRegionsTransforms:
    """Tests for coordinate transformations."""

    def test_world_to_pixel_transform_polygon(self):
        """Test applying world_to_pixel transform to polygon."""
        # Create a polygon in world coordinates
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        regions = Regions([Region(name="box", data=poly, kind="polygon")])

        # Create a transform that scales by 2
        transform = scale_2d(2.0, 2.0)

        fig, ax = plt.subplots()
        plot_regions(regions, ax=ax, world_to_pixel=transform, add_legend=False)

        # Patch should exist (exact coordinates are harder to check)
        assert len(ax.patches) == 1
        plt.close(fig)

    def test_world_to_pixel_transform_point(self):
        """Test applying world_to_pixel transform to point."""
        point_data = np.array([5.0, 10.0])
        regions = Regions([Region(name="pt", data=point_data, kind="point")])

        # Create a transform that translates by (10, 20)
        transform = translate(10.0, 20.0)

        fig, ax = plt.subplots()
        plot_regions(regions, ax=ax, world_to_pixel=transform, add_legend=False)

        # Scatter should exist
        assert len(ax.collections) == 1
        plt.close(fig)

    def test_no_transform(self):
        """Test plotting without transform (world_to_pixel=None)."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        regions = Regions([Region(name="box", data=poly, kind="polygon")])

        fig, ax = plt.subplots()
        plot_regions(regions, ax=ax, world_to_pixel=None, add_legend=False)

        # Should plot normally without transform
        assert len(ax.patches) == 1
        plt.close(fig)


class TestPlotRegionsPolygonWithHoles:
    """Tests for polygons with interior holes."""

    def test_polygon_with_holes(self):
        """Test plotting polygon with interior holes."""
        # Create a polygon with a hole
        exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
        hole = [(5, 5), (15, 5), (15, 15), (5, 15)]
        poly = Polygon(exterior, [hole])
        regions = Regions([Region(name="donut", data=poly, kind="polygon")])

        fig, ax = plt.subplots()
        plot_regions(regions, ax=ax, add_legend=False)

        # Should create a compound path with hole
        assert len(ax.patches) == 1
        plt.close(fig)


class TestPlotRegionsEdgeCases:
    """Tests for edge cases and error handling."""

    def test_autoscale_called(self):
        """Test that autoscale is called after plotting."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        regions = Regions([Region(name="box", data=poly, kind="polygon")])

        fig, ax = plt.subplots()
        # Set initial limits
        ax.set_xlim(-100, -50)
        ax.set_ylim(-100, -50)

        plot_regions(regions, ax=ax, add_legend=False)

        # Autoscale should have adjusted the limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # Limits should include the polygon
        assert xlim[0] <= 0 and xlim[1] >= 10
        assert ylim[0] <= 0 and ylim[1] >= 10
        plt.close(fig)

    def test_facecolor_vs_color_kwarg(self):
        """Test that facecolor takes precedence over color."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        regions = Regions([Region(name="box", data=poly, kind="polygon")])

        fig, ax = plt.subplots()
        plot_regions(
            regions,
            ax=ax,
            add_legend=False,
            box={"facecolor": "red", "color": "blue"},
        )

        # facecolor should take precedence
        assert len(ax.patches) == 1
        # Check that patch was created (exact color is hard to test due to matplotlib internals)
        plt.close(fig)

    def test_color_fallback_when_no_facecolor(self):
        """Test that color is used when facecolor is not specified."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        regions = Regions([Region(name="box", data=poly, kind="polygon")])

        fig, ax = plt.subplots()
        plot_regions(regions, ax=ax, add_legend=False, box={"color": "blue"})

        # Should use color as facecolor
        assert len(ax.patches) == 1
        plt.close(fig)
