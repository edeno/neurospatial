"""Tests for region membership queries on 2-D positions."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from shapely.geometry import Polygon, box

from neurospatial import RegionNotFoundError
from neurospatial.regions import Region, Regions
from neurospatial.regions.ops import (
    _prepare_positions,
    points_in_any_region,
    regions_containing_points,
)


@pytest.fixture(scope="module")
def square():
    """One 10 x 10 square region named ``box``."""
    return Regions([Region(name="box", data=box(0, 0, 10, 10), kind="polygon")])


@pytest.fixture(scope="module")
def overlapping():
    """``box1`` (0-10) and ``box2`` (5-15) overlap on 5-10."""
    return Regions(
        [
            Region(name="box1", data=box(0, 0, 10, 10), kind="polygon"),
            Region(name="box2", data=box(5, 5, 15, 15), kind="polygon"),
        ]
    )


@pytest.fixture(scope="module")
def three_boxes():
    """Three overlapping squares on the diagonal."""
    return Regions(
        [
            Region(name="box1", data=box(0, 0, 10, 10), kind="polygon"),
            Region(name="box2", data=box(5, 5, 15, 15), kind="polygon"),
            Region(name="box3", data=box(10, 10, 20, 20), kind="polygon"),
        ]
    )


@pytest.fixture(scope="module")
def well():
    """A point region (reward well) at (50, 50), in cm."""
    return Regions([Region(name="well", data=np.array([50.0, 50.0]), kind="point")])


@pytest.fixture(scope="module")
def box_and_well(square, well):
    """The 10 x 10 square plus the reward well."""
    return Regions([square["box"], well["well"]])


# Interior, two edge points and an exterior point of the 10 x 10 square.
EDGE_PROBES = np.array([[5.0, 5.0], [0.0, 5.0], [10.0, 5.0], [15.0, 15.0]])
EDGE_EXPECTED = {True: [True, True, True, False], False: [True, False, False, False]}


class PixelsToCm:
    """Divide by 10: 10 px per cm."""

    def __call__(self, positions):
        return positions / 10


class TestPreparePositions:
    """Shape, finiteness and transform checks shared by both queries."""

    def test_returns_float64_array_from_list_input(self):
        result = _prepare_positions([[1, 2], [3, 4]], None)
        assert result.dtype == np.float64
        assert_array_equal(result, [[1.0, 2.0], [3.0, 4.0]])

    def test_empty_positions_keep_their_shape(self):
        assert _prepare_positions(np.empty((0, 2)), None).shape == (0, 2)

    def test_bare_pair_is_refused_as_ambiguous(self):
        with pytest.raises(ValueError, match=r"got \(2,\)") as caught:
            _prepare_positions(np.array([5.0, 10.0]), None)
        assert "positions[None]" in str(caught.value)
        assert "np.atleast_2d" in str(caught.value)

    @pytest.mark.parametrize("shape", [(3,), (1, 2, 2), (2, 3), (4, 1)])
    def test_wrong_shapes_are_refused(self, shape):
        with pytest.raises(ValueError, match=r"\(n_positions, 2\)"):
            _prepare_positions(np.zeros(shape), None)

    def test_non_numeric_input_is_refused(self):
        with pytest.raises(ValueError, match="numeric"):
            _prepare_positions("not positions", None)

    def test_transform_maps_into_region_coordinates(self):
        result = _prepare_positions(np.array([[10.0, 20.0]]), PixelsToCm())
        assert_array_equal(result, [[1.0, 2.0]])

    def test_transform_that_adds_a_column_is_refused(self):
        def to_3d(positions):
            return np.column_stack([positions, np.zeros(len(positions))])

        with pytest.raises(ValueError, match=r"transform returned shape \(2, 3\)"):
            _prepare_positions(np.ones((2, 2)), to_3d)

    def test_transform_that_drops_rows_is_refused(self):
        with pytest.raises(ValueError, match=r"transform returned shape \(1, 2\)"):
            _prepare_positions(np.ones((2, 2)), lambda positions: positions[:1])

    def test_transform_with_non_finite_output_is_refused(self):
        def blow_up(positions):
            out = positions.copy()
            out[1, 0] = np.inf
            return out

        with pytest.raises(
            ValueError, match=r"transform returned non-finite coordinates for 1 of 3"
        ):
            _prepare_positions(np.ones((3, 2)), blow_up)


class TestMissingPositions:
    """A NaN position is missing, not outside every region."""

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    @pytest.mark.parametrize(
        "query",
        [
            points_in_any_region,
            regions_containing_points,
            lambda p, r: regions_containing_points(p, r, return_dataframe=False),
        ],
        ids=["any", "dataframe", "list"],
    )
    def test_non_finite_rows_are_refused(self, square, query, bad):
        positions = np.array([[5.0, 5.0], [1.0, 1.0], [bad, 2.0], [3.0, bad]])

        with pytest.raises(ValueError) as caught:
            query(positions, square)

        message = str(caught.value)
        assert "2 row(s)" in message
        assert "first at row 2" in message
        assert "Fix: drop or interpolate missing samples" in message


class TestPolygonMembership:
    def test_inside_outside_and_edge(self, square):
        mask = points_in_any_region(
            np.array([[5.0, 5.0], [15.0, 15.0], [0.0, 0.0]]), square
        )
        assert_array_equal(mask, [True, False, True])

    def test_union_of_overlapping_regions(self, overlapping):
        positions = np.array([[2.0, 2.0], [7.0, 7.0], [12.0, 12.0], [20.0, 20.0]])
        assert_array_equal(
            points_in_any_region(positions, overlapping), [True, True, True, False]
        )

    def test_hole_is_outside(self):
        donut = Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            holes=[[(4, 4), (6, 4), (6, 6), (4, 6)]],
        )
        regions = Regions([Region(name="donut", data=donut, kind="polygon")])
        mask = points_in_any_region(np.array([[5.0, 5.0], [2.0, 2.0]]), regions)
        assert_array_equal(mask, [False, True])

    def test_region_after_partial_cover_still_counts(self, three_boxes):
        # (1, 1) is only in box1, (18, 18) only in box3, the last region.
        positions = np.array([[1.0, 1.0], [18.0, 18.0], [30.0, 30.0]])
        assert_array_equal(
            points_in_any_region(positions, three_boxes), [True, True, False]
        )

    def test_empty_positions(self, square):
        mask = points_in_any_region(np.empty((0, 2)), square)
        assert mask.shape == (0,)
        assert mask.dtype == bool

    def test_no_regions(self):
        mask = points_in_any_region(np.array([[5.0, 5.0], [1.0, 1.0]]), Regions([]))
        assert_array_equal(mask, [False, False])

    def test_transform(self):
        unit = Regions([Region(name="box", data=box(0, 0, 1, 1), kind="polygon")])
        mask = points_in_any_region(
            np.array([[5.0, 5.0], [15.0, 15.0]]), unit, transform=PixelsToCm()
        )
        assert_array_equal(mask, [True, False])


@pytest.mark.parametrize("include_boundary", [True, False])
class TestIncludeBoundary:
    """Edge positions count only with ``include_boundary=True``, in every output."""

    def test_any(self, square, include_boundary):
        mask = points_in_any_region(
            EDGE_PROBES, square, include_boundary=include_boundary
        )
        assert_array_equal(mask, EDGE_EXPECTED[include_boundary])

    def test_dataframe(self, square, include_boundary):
        df = regions_containing_points(
            EDGE_PROBES, square, include_boundary=include_boundary
        )
        assert_array_equal(df["box"].to_numpy(), EDGE_EXPECTED[include_boundary])

    def test_list(self, square, include_boundary):
        result = regions_containing_points(
            EDGE_PROBES,
            square,
            return_dataframe=False,
            include_boundary=include_boundary,
        )
        expected = [["box"] if hit else [] for hit in EDGE_EXPECTED[include_boundary]]
        assert result == expected

    def test_point_region_rim(self, well, include_boundary):
        # (53, 54) is exactly 5 cm from (50, 50): a 3-4-5 triangle.
        mask = points_in_any_region(
            np.array([[53.0, 54.0]]),
            well,
            point_radius=5.0,
            include_boundary=include_boundary,
        )
        assert_array_equal(mask, [include_boundary])


class TestPointRegions:
    """A point region is a disc of radius ``point_radius`` in the regions' units."""

    def test_euclidean_disc_not_square(self, well):
        # (53.6, 53.6) is within 5 cm on each axis but 5.09 cm away.
        positions = np.array([[50.0, 50.0], [54.9, 50.0], [53.6, 53.6], [53.5, 53.5]])
        mask = points_in_any_region(positions, well, point_radius=5.0)
        assert_array_equal(mask, [True, True, False, True])

    def test_radius_sets_the_hit_distance(self, well):
        # The same 3 cm offset is in a 5 cm well and out of a 2 cm one.
        positions = np.array([[53.0, 50.0]])
        assert points_in_any_region(positions, well, point_radius=5.0)[0]
        assert not points_in_any_region(positions, well, point_radius=2.0)[0]

    def test_zero_radius_matches_only_the_point(self, well):
        mask = points_in_any_region(
            np.array([[50.0, 50.0], [50.0001, 50.0]]), well, point_radius=0.0
        )
        assert_array_equal(mask, [True, False])

    @pytest.mark.parametrize(
        "query",
        [points_in_any_region, regions_containing_points],
        ids=["any", "dataframe"],
    )
    def test_missing_radius_names_the_point_region(self, box_and_well, query):
        with pytest.raises(ValueError) as caught:
            query(np.array([[50.0, 50.0]]), box_and_well)

        message = str(caught.value)
        assert "'well'" in message
        assert "'box'" not in message
        assert "no area" in message
        assert "point_radius=" in message
        assert "env.bins_in_region('well')" in message

    @pytest.mark.parametrize("radius", [-1.0, np.nan, np.inf, True, "5"])
    def test_invalid_radius_is_refused(self, well, radius):
        with pytest.raises(ValueError, match="point_radius must be finite"):
            points_in_any_region(np.array([[50.0, 50.0]]), well, point_radius=radius)

    def test_polygon_only_selection_needs_no_radius(self, box_and_well):
        df = regions_containing_points(
            np.array([[5.0, 5.0], [50.0, 50.0]]), box_and_well, region_names=["box"]
        )
        assert_array_equal(df["box"].to_numpy(), [True, False])

    def test_mixed_regions(self, box_and_well):
        result = regions_containing_points(
            np.array([[5.0, 5.0], [51.0, 51.0], [30.0, 30.0]]),
            box_and_well,
            return_dataframe=False,
            point_radius=2.0,
        )
        assert result == [["box"], ["well"], []]

    @pytest.mark.parametrize("coords", [[1.0, 2.0, 3.0], [4.0]], ids=["3d", "1d"])
    @pytest.mark.parametrize(
        "query",
        [points_in_any_region, regions_containing_points],
        ids=["any", "dataframe"],
    )
    def test_non_2d_point_region_is_refused(self, coords, query):
        regions = Regions([Region(name="cue", data=np.array(coords), kind="point")])
        with pytest.raises(ValueError, match=f"'cue' is a {len(coords)}-D point"):
            query(np.array([[1.0, 2.0]]), regions, point_radius=1.0)


class TestRegionsContainingPoints:
    def test_list_output_names_every_containing_region(self, overlapping):
        positions = np.array([[2.0, 2.0], [7.0, 7.0], [12.0, 12.0], [20.0, 20.0]])
        result = regions_containing_points(
            positions, overlapping, return_dataframe=False
        )
        assert result == [["box1"], ["box1", "box2"], ["box2"], []]

    def test_dataframe_output(self, overlapping):
        positions = np.array([[2.0, 2.0], [7.0, 7.0], [12.0, 12.0], [20.0, 20.0]])

        df = regions_containing_points(positions, overlapping)

        assert list(df.columns) == ["box1", "box2"]
        assert isinstance(df.index, pd.RangeIndex)
        assert (df.dtypes == np.dtype(bool)).all()
        assert_array_equal(df["box1"].to_numpy(), [True, True, False, False])
        assert_array_equal(df["box2"].to_numpy(), [False, True, True, False])

    def test_list_and_dataframe_agree(self, three_boxes):
        rng = np.random.default_rng(0)
        positions = rng.uniform(-2, 22, size=(200, 2))

        df = regions_containing_points(positions, three_boxes)
        result = regions_containing_points(
            positions, three_boxes, return_dataframe=False
        )

        assert result == [list(df.columns[row]) for row in df.to_numpy()]
        assert_array_equal(
            df.any(axis=1).to_numpy(), points_in_any_region(positions, three_boxes)
        )

    def test_region_names_select_and_order_columns(self, three_boxes):
        df = regions_containing_points(
            np.array([[5.0, 5.0]]), three_boxes, region_names=["box3", "box1"]
        )
        assert list(df.columns) == ["box3", "box1"]
        assert_array_equal(df.to_numpy(), [[False, True]])

    def test_single_name_string(self, overlapping):
        df = regions_containing_points(
            np.array([[5.0, 5.0]]), overlapping, region_names="box1"
        )
        assert list(df.columns) == ["box1"]
        assert bool(df.loc[0, "box1"])

    def test_region_names_not_found(self, square):
        with pytest.raises(RegionNotFoundError) as caught:
            regions_containing_points(
                np.array([[5.0, 5.0]]), square, region_names=["box", "bx"]
            )
        assert str(caught.value).endswith("Fix: pass region_names=['box']")

    def test_duplicate_region_names_are_refused(self, overlapping):
        with pytest.raises(ValueError, match="lists 'box1' more than once"):
            regions_containing_points(
                np.array([[5.0, 5.0]]),
                overlapping,
                region_names=["box1", "box2", "box1"],
            )

    def test_empty_positions_dataframe_has_bool_columns(self, overlapping):
        df = regions_containing_points(np.empty((0, 2)), overlapping)

        assert df.shape == (0, 2)
        assert list(df.columns) == ["box1", "box2"]
        assert isinstance(df.index, pd.RangeIndex)
        assert (df.dtypes == np.dtype(bool)).all()

    def test_empty_positions_list(self, square):
        result = regions_containing_points(
            np.empty((0, 2)), square, return_dataframe=False
        )
        assert result == []

    def test_no_regions(self):
        positions = np.array([[5.0, 5.0], [1.0, 1.0]])
        df = regions_containing_points(positions, Regions([]))
        assert df.shape == (2, 0)
        assert isinstance(df.index, pd.RangeIndex)
        result = regions_containing_points(
            positions, Regions([]), return_dataframe=False
        )
        assert result == [[], []]

    def test_transform(self):
        unit = Regions([Region(name="box", data=box(0, 0, 1, 1), kind="polygon")])
        df = regions_containing_points(
            np.array([[5.0, 5.0], [15.0, 15.0]]), unit, transform=PixelsToCm()
        )
        assert_array_equal(df["box"].to_numpy(), [True, False])
