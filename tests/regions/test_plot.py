"""Tests for drawing regions with plot_regions."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from numpy.testing import assert_allclose, assert_array_equal
from shapely.geometry import Polygon, box

from neurospatial import RegionNotFoundError
from neurospatial.ops.transforms import scale_2d, translate
from neurospatial.regions import Region, Regions
from neurospatial.regions.plot import plot_regions


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def square():
    """One 10 x 10 square named ``box``."""
    return Regions([Region(name="box", data=box(0, 0, 10, 10), kind="polygon")])


@pytest.fixture(scope="module")
def two_squares():
    return Regions(
        [
            Region(name="box1", data=box(0, 0, 5, 5), kind="polygon"),
            Region(name="box2", data=box(10, 10, 15, 15), kind="polygon"),
        ]
    )


def _ring_vertices(patch):
    """The distinct vertices of a patch's path, as a sorted list of tuples."""
    return sorted({tuple(v) for v in patch.get_path().vertices.tolist()})


class TestDrawing:
    def test_point_is_a_marker_at_its_coordinates(self):
        regions = Regions([Region(name="pt", data=np.array([5.0, 10.0]), kind="point")])
        _, ax = plt.subplots()

        plot_regions(regions, ax=ax, add_legend=False)

        assert len(ax.collections) == 1
        assert_array_equal(ax.collections[0].get_offsets(), [[5.0, 10.0]])

    def test_polygon_is_a_patch_with_its_vertices(self, square):
        _, ax = plt.subplots()

        plot_regions(square, ax=ax, add_legend=False)

        assert len(ax.patches) == 1
        assert _ring_vertices(ax.patches[0]) == [(0, 0), (0, 10), (10, 0), (10, 10)]

    def test_points_and_polygons_together(self):
        regions = Regions(
            [
                Region(name="pt", data=np.array([2.0, 2.0]), kind="point"),
                Region(name="box", data=box(5, 5, 15, 15), kind="polygon"),
            ]
        )
        _, ax = plt.subplots()

        plot_regions(regions, ax=ax, add_legend=False)

        assert_array_equal(ax.collections[0].get_offsets(), [[2.0, 2.0]])
        assert _ring_vertices(ax.patches[0]) == [(5, 5), (5, 15), (15, 5), (15, 15)]

    def test_returns_the_axes_it_drew_on(self, square):
        _, ax = plt.subplots()
        assert plot_regions(square, ax=ax, add_legend=False) is ax

    def test_defaults_to_the_current_axes(self, square):
        _, ax = plt.subplots()
        plt.sca(ax)

        assert plot_regions(square, add_legend=False) is ax
        assert len(ax.patches) == 1

    def test_empty_collection_returns_axes(self):
        _, ax = plt.subplots()
        assert plot_regions(Regions([]), ax=ax) is ax
        assert not ax.patches
        assert ax.get_legend() is None


class TestPolygonHoles:
    """A hole is drawn empty whatever direction its ring runs."""

    @staticmethod
    def _pixel(fig, ax, xy):
        """RGBA of the rendered pixel at data coordinates ``xy``."""
        image = np.asarray(fig.canvas.buffer_rgba())
        x, y = ax.transData.transform(xy)
        return image[round(image.shape[0] - y), round(x)]

    @pytest.mark.parametrize("y_sign", [1.0, -1.0], ids=["as_is", "image_y_flip"])
    @pytest.mark.parametrize(
        "hole",
        [
            [
                (5, 5),
                (15, 5),
                (15, 15),
                (5, 15),
            ],  # counter-clockwise, like the exterior
            [(5, 5), (5, 15), (15, 15), (15, 5)],  # clockwise
        ],
        ids=["same_winding", "opposite_winding"],
    )
    def test_hole_is_background_and_ring_is_filled(self, hole, y_sign):
        exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]  # counter-clockwise
        donut = Polygon(exterior, [hole])
        regions = Regions([Region(name="donut", data=donut, kind="polygon")])
        fig, ax = plt.subplots(figsize=(4, 4), dpi=50)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.set_xlim(-1, 21)
        ax.set_ylim(sorted([-1 * y_sign, 21 * y_sign]))

        # A reflection, such as flipping y for image rows, reverses both rings.
        plot_regions(
            regions,
            ax=ax,
            add_legend=False,
            world_to_pixel=scale_2d(1.0, y_sign),
            donut={"facecolor": "black", "edgecolor": "none", "alpha": 1.0},
        )
        fig.canvas.draw()

        white, black = [255, 255, 255, 255], [0, 0, 0, 255]
        assert_array_equal(self._pixel(fig, ax, (10, 10 * y_sign)), white)  # hole
        assert_array_equal(self._pixel(fig, ax, (2.5, 10 * y_sign)), black)  # ring
        assert_array_equal(self._pixel(fig, ax, (10, 2.5 * y_sign)), black)  # ring


class TestSelection:
    def test_subset_by_name(self, two_squares):
        _, ax = plt.subplots()

        plot_regions(two_squares, ax=ax, region_names=["box2"], add_legend=False)

        assert len(ax.patches) == 1
        assert _ring_vertices(ax.patches[0])[0] == (10, 10)

    def test_single_name_string(self, square):
        _, ax = plt.subplots()
        plot_regions(square, ax=ax, region_names="box", add_legend=False)
        assert len(ax.patches) == 1

    def test_misspelled_name_is_refused_before_drawing(self, square):
        _, ax = plt.subplots()
        with pytest.raises(RegionNotFoundError, match="Region 'bx' not found"):
            plot_regions(square, ax=ax, region_names=["box", "bx"], add_legend=False)
        assert not ax.patches

    def test_misspelled_name_is_refused_before_a_figure_is_made(self, square):
        plt.close("all")
        with pytest.raises(RegionNotFoundError):
            plot_regions(square, region_names=["bx"])
        assert plt.get_fignums() == []

    def test_misspelled_style_name_is_refused(self, square):
        _, ax = plt.subplots()
        with pytest.raises(RegionNotFoundError) as caught:
            plot_regions(square, ax=ax, bx={"facecolor": "red"})
        assert str(caught.value).endswith("Fix: use 'box'")
        assert not ax.patches

    def test_style_for_an_undrawn_region_is_allowed(self, two_squares):
        _, ax = plt.subplots()
        plot_regions(
            two_squares,
            ax=ax,
            region_names=["box1"],
            add_legend=False,
            box2={"facecolor": "red"},
        )
        assert len(ax.patches) == 1

    @pytest.mark.parametrize("coords", [[1.0, 2.0, 3.0], [4.0]], ids=["3d", "1d"])
    def test_non_2d_point_is_refused(self, coords):
        regions = Regions([Region(name="cue", data=np.array(coords), kind="point")])
        _, ax = plt.subplots()
        with pytest.raises(ValueError, match=f"'cue' is a {len(coords)}-D point"):
            plot_regions(regions, ax=ax)
        assert not ax.collections


class TestStyling:
    def test_default_kwargs_apply_to_every_region(self, two_squares):
        _, ax = plt.subplots()

        plot_regions(
            two_squares,
            ax=ax,
            default_kwargs={"edgecolor": "red", "linewidth": 2},
            add_legend=False,
        )

        for patch in ax.patches:
            assert_allclose(patch.get_edgecolor(), (1.0, 0.0, 0.0, 0.5))  # alpha 0.5
            assert patch.get_linewidth() == 2

    def test_per_region_kwargs_style_only_that_region(self, two_squares):
        _, ax = plt.subplots()

        plot_regions(two_squares, ax=ax, add_legend=False, box1={"facecolor": "blue"})

        first, second = ax.patches
        assert_allclose(first.get_facecolor(), to_rgba("blue", 0.5))
        assert not np.allclose(second.get_facecolor(), to_rgba("blue", 0.5))

    @pytest.mark.parametrize(
        ("per_region", "expected"),
        [(None, "green"), ({"edgecolor": "blue"}, "blue")],
        ids=["metadata_over_default", "per_region_over_metadata"],
    )
    def test_option_precedence(self, per_region, expected):
        regions = Regions(
            [
                Region(
                    name="box",
                    data=box(0, 0, 10, 10),
                    kind="polygon",
                    metadata={"plot_kwargs": {"edgecolor": "green"}},
                )
            ]
        )
        _, ax = plt.subplots()
        overrides = {"box": per_region} if per_region else {}

        plot_regions(
            regions,
            ax=ax,
            add_legend=False,
            default_kwargs={"edgecolor": "red"},
            **overrides,
        )

        assert_allclose(ax.patches[0].get_edgecolor(), to_rgba(expected, 0.5))

    def test_alpha(self, square):
        _, ax = plt.subplots()
        plot_regions(square, ax=ax, add_legend=False, box={"alpha": 0.3})
        assert ax.patches[0].get_alpha() == 0.3

    def test_point_marker_size(self):
        regions = Regions([Region(name="pt", data=np.array([5.0, 10.0]), kind="point")])
        _, ax = plt.subplots()

        plot_regions(regions, ax=ax, add_legend=False, pt={"marker": "o", "s": 200})

        assert_array_equal(ax.collections[0].get_sizes(), [200])

    def test_facecolor_takes_precedence_over_color(self, square):
        _, ax = plt.subplots()
        plot_regions(
            square, ax=ax, add_legend=False, box={"facecolor": "red", "color": "blue"}
        )
        assert_allclose(ax.patches[0].get_facecolor(), to_rgba("red", 0.5))

    def test_color_sets_facecolor(self, square):
        _, ax = plt.subplots()
        plot_regions(square, ax=ax, add_legend=False, box={"color": "blue"})
        assert_allclose(ax.patches[0].get_facecolor(), to_rgba("blue", 0.5))


class TestLegend:
    def test_labels_default_to_region_names(self, two_squares):
        _, ax = plt.subplots()
        plot_regions(two_squares, ax=ax)
        texts = [text.get_text() for text in ax.get_legend().get_texts()]
        assert texts == ["box1", "box2"]

    def test_custom_label(self, square):
        _, ax = plt.subplots()
        plot_regions(square, ax=ax, box={"label": "Custom Label"})
        texts = [text.get_text() for text in ax.get_legend().get_texts()]
        assert texts == ["Custom Label"]

    def test_no_legend_when_disabled(self, square):
        _, ax = plt.subplots()
        plot_regions(square, ax=ax, add_legend=False)
        assert ax.get_legend() is None


class TestTransforms:
    def test_world_to_pixel_maps_polygon_vertices(self, square):
        _, ax = plt.subplots()

        plot_regions(square, ax=ax, world_to_pixel=scale_2d(2.0, 2.0), add_legend=False)

        assert _ring_vertices(ax.patches[0]) == [(0, 0), (0, 20), (20, 0), (20, 20)]

    def test_world_to_pixel_maps_point_offsets(self):
        regions = Regions([Region(name="pt", data=np.array([5.0, 10.0]), kind="point")])
        _, ax = plt.subplots()

        plot_regions(
            regions, ax=ax, world_to_pixel=translate(10.0, 20.0), add_legend=False
        )

        assert_allclose(ax.collections[0].get_offsets(), [[15.0, 30.0]])


class TestAxisLimits:
    def test_autoscaling_axes_grow_to_show_the_regions(self, two_squares):
        _, ax = plt.subplots()

        plot_regions(two_squares, ax=ax, add_legend=False)

        (x_low, x_high), (y_low, y_high) = ax.get_xlim(), ax.get_ylim()
        assert x_low <= 0 and x_high >= 15
        assert y_low <= 0 and y_high >= 15

    def test_limits_set_by_the_caller_are_kept(self, square):
        _, ax = plt.subplots()
        ax.set_xlim(-100, -50)
        ax.set_ylim(200, 0)  # an image frame: y increases downward

        plot_regions(square, ax=ax, add_legend=False)

        assert ax.get_xlim() == (-100, -50)
        assert ax.get_ylim() == (200, 0)
