"""
Tests for boundary cell metrics.

References
----------
Solstad et al. (2008). Representation of geometric borders in the entorhinal cortex.
    Science, 322(5909), 1865-1868.
"""

from __future__ import annotations

import numpy as np
import pytest
import shapely

from neurospatial import Environment, Regions


def _solstad_reference(rates, width, height, bin_size, threshold=0.3):
    """Solstad et al. (2008) border score on a rectangular box of square bins.

    Written from the paper's definition, independently of the package's wall
    geometry: rows and columns of a ``(n_y, n_x)`` rate image, one field, the
    best of four walls, rate-weighted center-to-wall distance and half the
    shorter side.
    """
    field = rates >= threshold * rates.max()
    coverage = max(
        field[:, 0].mean(), field[:, -1].mean(), field[0].mean(), field[-1].mean()
    )
    n_y, n_x = rates.shape
    x = (np.arange(n_x) + 0.5) * bin_size
    y = (np.arange(n_y) + 0.5) * bin_size
    to_wall = np.minimum(
        np.minimum(x, width - x)[None, :], np.minimum(y, height - y)[:, None]
    )
    weights = np.where(field, rates, 0.0)
    distance = np.sum(weights * to_wall) / np.sum(weights) / (min(width, height) / 2)
    return (coverage - distance) / (coverage + distance)


def _box(width, height, bin_size=2.5):
    return Environment.from_polygon(
        shapely.box(0, 0, width, height), bin_size=bin_size, units="cm"
    )


def _rate_image(env, rates, bin_size=2.5):
    """Rates as a ``(n_y, n_x)`` image indexed by bin row and column."""
    columns = np.floor(env.bin_centers[:, 0] / bin_size).astype(int)
    rows = np.floor(env.bin_centers[:, 1] / bin_size).astype(int)
    image = np.full((rows.max() + 1, columns.max() + 1), np.nan)
    image[rows, columns] = rates
    assert np.isfinite(image).all()
    return image


def _maps(env, width, height):
    x, y = env.bin_centers.T

    def bump(x0, y0, sigma):
        return 0.2 + 10.0 * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))

    return {
        "west strip": np.where(x < 7.5, 10.0, 0.2),
        "field on west wall": bump(5.0, height / 2, 10.0),
        "corner field": bump(5.0, 5.0, 12.0),
        "central field": bump(width / 2, height / 2, 10.0),
        "broad central field": bump(width / 2, height / 2, 35.0),
        "uniform with noise": 5.0 + np.random.default_rng(0).normal(0, 1, env.n_bins),
    }


MAP_NAMES = list(_maps(_box(20, 20), 20, 20))


class TestBorderScore:
    """border_score is the Solstad et al. (2008) score on rectangular boxes."""

    @pytest.mark.parametrize("size", [(100.0, 100.0), (150.0, 60.0)])
    @pytest.mark.parametrize("name", MAP_NAMES)
    def test_rectangles_match_independent_solstad_reference(self, size, name):
        from neurospatial.encoding import border_score

        width, height = size
        env = _box(width, height)
        rates = _maps(env, width, height)[name]
        expected = _solstad_reference(_rate_image(env, rates), width, height, 2.5)
        for metric in ("euclidean", "geodesic"):
            for source in ("physical", "analysis"):
                np.testing.assert_allclose(
                    border_score(env, rates, metric=metric, boundary_source=source),
                    expected,
                    rtol=1e-9,
                    err_msg=f"{name}, {metric}, {source}",
                )

    def test_one_wall_cell_scores_well_above_a_uniform_map(self):
        """A uniform map's one field covers every wall, so it scores 0.5 on a
        square: the spatial-information gate, not the score, rejects it."""
        from neurospatial.encoding import border_score

        env = _box(100, 100)
        maps = _maps(env, 100, 100)
        uniform = border_score(env, maps["uniform with noise"])
        wall = border_score(env, maps["west strip"])
        assert uniform == pytest.approx(0.5, abs=0.01)
        assert wall - uniform > 0.3
        assert border_score(env, maps["central field"]) == -1.0

    def test_fields_smaller_than_min_area_are_ignored(self):
        """Solstad's 200 cm^2 field minimum: a single bright bin in the middle
        of the box is not a field, so the wall field alone is scored."""
        from neurospatial.encoding import border_score

        env = _box(100, 100)
        wall = _maps(env, 100, 100)["west strip"]
        with_speck = wall.copy()
        with_speck[np.argmin(np.hypot(*(env.bin_centers - 50.0).T))] = 10.0
        np.testing.assert_allclose(
            border_score(env, with_speck), border_score(env, wall)
        )
        assert border_score(env, with_speck, min_area=0) < border_score(env, wall)
        assert np.isnan(border_score(env, wall, min_area=10_000.0))

    def test_default_min_area_follows_units(self):
        """200 cm^2 converts with the units: meters score like centimeters."""
        from neurospatial.encoding import border_score

        env = _box(100, 100)
        rates = _maps(env, 100, 100)["corner field"]
        meters = Environment.from_polygon(
            shapely.box(0, 0, 1.0, 1.0), bin_size=0.025, units="m"
        )
        np.testing.assert_allclose(
            border_score(meters, rates), border_score(env, rates)
        )

    @pytest.mark.parametrize(
        ("blob_bins", "kept_without_units"), [(16, True), (8, False)]
    )
    def test_default_field_minimum_is_200_cm2_or_9_bins(
        self, blob_bins, kept_without_units
    ):
        """With metric units a field needs 200 cm^2 (32 bins of 2.5 cm); without
        them it needs 9 bins, the BNT/opexebo default."""
        from neurospatial.encoding import border_score

        env = _box(100, 100)
        unitless = Environment.from_polygon(shapely.box(0, 0, 100, 100), bin_size=2.5)
        x, y = env.bin_centers.T
        wall = np.where(x < 7.5, 10.0, 0.2)
        # A central blob of 4 x 4 or 2 x 4 bins at the peak rate.
        rows = 4 if blob_bins == 16 else 2
        blob = (np.abs(x - 50) < 5) & (y > 50) & (y < 50 + rows * 2.5)
        assert blob.sum() == blob_bins
        rates = np.where(blob, 10.0, wall)

        # Centimeters: the blob is under 200 cm^2, so only the wall field counts.
        np.testing.assert_allclose(border_score(env, rates), border_score(env, wall))
        expected = (
            border_score(env, rates, min_area=0)
            if kept_without_units
            else border_score(env, wall)
        )
        np.testing.assert_allclose(border_score(unitless, rates), expected)

    def test_summary_records_the_field_minimum_it_applied(self):
        from neurospatial.encoding.spatial import SpatialRatesResult

        for units, rule in [("cm", "200 cm²"), (None, "9 bins")]:
            env = Environment.from_polygon(
                shapely.box(0, 0, 100, 100), bin_size=2.5, units=units
            )
            wall = _maps(env, 100, 100)["west strip"]
            # 1000 s per bin: enough spikes for the information gate's bias
            # correction on 1600 unsmoothed bins.
            result = SpatialRatesResult(
                wall[None],
                np.full(env.n_bins, 1000.0),
                env,
                method="binned",
                bandwidth=0.0,
            )
            table = result.summary_table()
            assert table.attrs["border_field_minimum"].startswith(rule)
            assert np.isfinite(table["border_score"]).all()
            assert "unavailable_metrics" not in table.attrs
            assert result.label_cell_types().tolist() == ["border"]

    def test_wall_strip_coverage_and_distance_by_hand(self):
        """A one-bin strip along the west wall of a 10 x 10 box covers that
        whole wall; every center is half a bin from the wall."""
        from neurospatial.encoding import border_score

        env = _box(10, 10, bin_size=1.0)
        strip = (env.bin_centers[:, 0] < 1).astype(float)
        distance = 0.5 / 5.0
        expected = (1 - distance) / (1 + distance)
        np.testing.assert_allclose(border_score(env, strip, min_area=0), expected)

    def test_unvisited_or_silent_maps_are_nan(self):
        from neurospatial.encoding import border_score

        env = _box(20, 20)
        assert np.isnan(
            border_score(env, np.zeros(env.n_bins), boundary_source="analysis")
        )
        assert np.isnan(
            border_score(env, np.full(env.n_bins, np.nan), boundary_source="analysis")
        )

    def test_argument_validation(self):
        from neurospatial.encoding import border_score

        env = _box(20, 20)
        rates = np.ones(env.n_bins)
        with pytest.raises(ValueError, match=r"firing_rate\.shape"):
            border_score(env, np.ones((env.n_bins, 2)))
        for threshold in (0.0, 1.5):
            with pytest.raises(ValueError, match="threshold must be in"):
                border_score(env, rates, threshold=threshold)
        with pytest.raises(ValueError, match="min_area must be non-negative"):
            border_score(env, rates, min_area=-10.0)
        with pytest.raises(ValueError, match="metric must be"):
            border_score(env, rates, metric="invalid")


class TestBorderWalls:
    """How outlines split into the walls that coverage is measured on."""

    @pytest.mark.parametrize(
        ("polygon", "n_walls"),
        [
            (shapely.box(0, 0, 100, 60), 4),
            (shapely.Polygon([(0, 0), (100, 0), (50, 80)]), 3),
            (
                shapely.Polygon(
                    [
                        (50 + 50 * np.cos(a), 50 + 50 * np.sin(a))
                        for a in np.arange(6) * np.pi / 3
                    ]
                ),
                6,
            ),
            (shapely.Point(50, 50).buffer(50, quad_segs=2), 1),
            (shapely.Point(50, 50).buffer(50, quad_segs=32), 1),
        ],
        ids=["rectangle", "triangle", "hexagon", "octagon", "circle"],
    )
    def test_corners_split_outlines_into_walls(self, polygon, n_walls):
        from neurospatial.encoding._physical_border import border_geometry

        env = Environment.from_polygon(polygon, bin_size=2.5, units="cm")
        assert len(border_geometry(env, "physical").wall_lengths) == n_walls

    def test_pixel_staircase_of_a_circle_is_one_analysis_wall(self):
        from neurospatial.encoding._physical_border import border_geometry

        rng = np.random.default_rng(1)
        angle, radius = (
            rng.uniform(0, 2 * np.pi, 20_000),
            50 * np.sqrt(rng.uniform(0, 1, 20_000)),
        )
        positions = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
        env = Environment.from_samples(positions, bin_size=2.5, units="cm")
        assert len(border_geometry(env, "analysis").wall_lengths) == 1

    def test_named_internal_walls_are_separate_walls(self):
        from neurospatial.arena import OpenArena
        from neurospatial.encoding import border_score
        from neurospatial.encoding._physical_border import border_geometry

        outline = shapely.box(0, 0, 100, 100)
        divider = shapely.LineString([(50, 20), (50, 80)])
        env = Environment.from_polygon(
            outline,
            bin_size=2.5,
            units="cm",
            arena=OpenArena(outline, walls={"divider": divider}),
        )
        x, y = env.bin_centers.T
        # One column against the divider's west face covers that whole wall,
        # with every center half a bin from it.
        along = (x > 47.5) & (x < 50) & (y > 20) & (y < 80)
        rates = np.where(along, 10.0, 0.2)
        radius = border_geometry(env, "physical").radius
        # The largest circle clear of every wall, by brute force on a grid.
        grid = np.stack(np.meshgrid(*[np.linspace(0, 100, 401)] * 2), -1).reshape(-1, 2)
        clearance = shapely.distance(
            shapely.points(grid), shapely.union_all([outline.boundary, divider])
        )
        assert radius == pytest.approx(clearance.max(), abs=0.25)
        distance = 1.25 / radius
        np.testing.assert_allclose(
            border_score(env, rates, min_area=0, metric="euclidean"),
            (1 - distance) / (1 + distance),
        )


class TestComputeRegionCoverage:
    """Test compute_region_coverage function.

    Note: These tests require inline Environment.from_samples() calls because they
    add regions to the environment. Session-scoped fixtures cannot be used since
    region additions would persist and conflict between tests.
    """

    def test_region_coverage_basic(self) -> None:
        """Test basic region coverage computation with wall regions."""
        from shapely.geometry import box

        # Inline environment required: adds wall regions (north, south, east, west)
        x = np.linspace(0, 40, 400)
        y = np.linspace(0, 40, 400)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])

        env = Environment.from_samples(positions, bin_size=4.0)

        # Add wall regions
        region_builder = Regions(env.regions.values())
        region_builder.add("north", polygon=box(0, 30, 40, 40))
        region_builder.add("south", polygon=box(0, 0, 40, 10))
        region_builder.add("east", polygon=box(30, 0, 40, 40))
        region_builder.add("west", polygon=box(0, 0, 10, 40))
        env = env.with_regions(region_builder)

        # Create field along north wall
        firing_rate = np.zeros(env.n_bins)
        north_bins = np.where(env.region_mask("north"))[0]
        firing_rate[north_bins] = 5.0
        field_bins = np.where(firing_rate > 0)[0]

        from neurospatial.encoding.border import compute_region_coverage

        coverage = compute_region_coverage(env, field_bins)

        # North wall should have high coverage, others should be low
        # Note: walls overlap at corners, so east/west have ~0.22 coverage from corner overlap
        assert coverage["north"] > 0.8, (
            f"Expected north coverage > 0.8, got {coverage['north']}"
        )
        assert coverage["south"] < 0.3, (
            f"Expected south coverage < 0.3, got {coverage['south']}"
        )
        assert coverage["east"] < 0.3, (
            f"Expected east coverage < 0.3, got {coverage['east']}"
        )
        assert coverage["west"] < 0.3, (
            f"Expected west coverage < 0.3, got {coverage['west']}"
        )

    def test_region_coverage_specific_regions(self) -> None:
        """Test region coverage with specific regions parameter."""
        from shapely.geometry import box

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((2000, 2)) * 20
        env = Environment.from_samples(positions, bin_size=4.0)

        region_builder = Regions(env.regions.values())
        region_builder.add("region1", polygon=box(-20, -20, 0, 0))
        region_builder.add("region2", polygon=box(0, 0, 20, 20))
        region_builder.add("region3", polygon=box(-20, 0, 0, 20))
        env = env.with_regions(region_builder)

        # Create field in region1
        firing_rate = np.zeros(env.n_bins)
        region1_bins = np.where(env.region_mask("region1"))[0]
        firing_rate[region1_bins] = 5.0
        field_bins = np.where(firing_rate > 0)[0]

        from neurospatial.encoding.border import compute_region_coverage

        # Only compute for region1 and region2
        coverage = compute_region_coverage(
            env, field_bins, regions=["region1", "region2"]
        )

        # Should only return specified regions
        assert set(coverage.keys()) == {"region1", "region2"}
        assert coverage["region1"] > 0.8
        assert coverage["region2"] < 0.2

    def test_region_coverage_none_regions(self) -> None:
        """Test region coverage with regions=None uses all regions."""
        from shapely.geometry import box

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((2000, 2)) * 20
        env = Environment.from_samples(positions, bin_size=4.0)

        region_builder = Regions(env.regions.values())
        region_builder.add("region1", polygon=box(-20, -20, 0, 0))
        region_builder.add("region2", polygon=box(0, 0, 20, 20))
        env = env.with_regions(region_builder)

        firing_rate = np.zeros(env.n_bins)
        region1_bins = np.where(env.region_mask("region1"))[0]
        firing_rate[region1_bins] = 5.0
        field_bins = np.where(firing_rate > 0)[0]

        from neurospatial.encoding.border import compute_region_coverage

        coverage = compute_region_coverage(env, field_bins, regions=None)

        # Should return all regions
        assert set(coverage.keys()) == {"region1", "region2"}

    def test_region_coverage_empty_region(self) -> None:
        """Test region coverage with empty region."""
        from shapely.geometry import box

        rng = np.random.default_rng(42)
        positions = rng.standard_normal((2000, 2)) * 20
        env = Environment.from_samples(positions, bin_size=4.0)

        # Add region that doesn't overlap with any bins
        region_builder = Regions(env.regions.values())
        region_builder.add("far_away", polygon=box(1000, 1000, 1100, 1100))
        env = env.with_regions(region_builder)

        firing_rate = np.zeros(env.n_bins)
        firing_rate[0] = 5.0
        field_bins = np.where(firing_rate > 0)[0]

        from neurospatial.encoding.border import compute_region_coverage

        coverage = compute_region_coverage(env, field_bins)

        # A region with no bins has no coverage to measure: 0/0 is NaN.
        assert np.isnan(coverage["far_away"])

    def test_region_coverage_nonexistent_region(
        self, small_2d_env: Environment
    ) -> None:
        """Test region coverage with non-existent region raises error."""
        env = small_2d_env

        firing_rate = np.zeros(env.n_bins)
        firing_rate[0] = 5.0
        field_bins = np.where(firing_rate > 0)[0]

        from neurospatial.encoding.border import compute_region_coverage

        with pytest.raises(ValueError, match="Region 'nonexistent' not found"):
            compute_region_coverage(env, field_bins, regions=["nonexistent"])

    def test_region_coverage_full_coverage(self) -> None:
        """Test region coverage when field covers entire region."""
        from shapely.geometry import box

        # Create rectangular environment using vectorized meshgrid
        x = np.linspace(0, 40, 400)
        y = np.linspace(0, 40, 400)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])

        env = Environment.from_samples(positions, bin_size=4.0)

        # Add region
        region_builder = Regions(env.regions.values())
        region_builder.add("center", polygon=box(15, 15, 25, 25))
        env = env.with_regions(region_builder)

        # Create field that covers all of center region
        center_bins = np.where(env.region_mask("center"))[0]
        field_bins = center_bins

        from neurospatial.encoding.border import compute_region_coverage

        coverage = compute_region_coverage(env, field_bins)

        # Should have perfect coverage
        assert coverage["center"] == 1.0

    def test_region_coverage_no_coverage(self) -> None:
        """Test region coverage when field doesn't overlap region."""
        from shapely.geometry import box

        # Create rectangular environment using vectorized meshgrid
        x = np.linspace(0, 40, 400)
        y = np.linspace(0, 40, 400)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])

        env = Environment.from_samples(positions, bin_size=4.0)

        # Add two disjoint regions
        region_builder = Regions(env.regions.values())
        region_builder.add("north", polygon=box(0, 30, 40, 40))
        region_builder.add("south", polygon=box(0, 0, 40, 10))
        env = env.with_regions(region_builder)

        # Create field only in north
        north_bins = np.where(env.region_mask("north"))[0]
        field_bins = north_bins

        from neurospatial.encoding.border import compute_region_coverage

        coverage = compute_region_coverage(env, field_bins)

        # North should have full coverage, south should have zero
        assert coverage["north"] == 1.0
        assert coverage["south"] == 0.0
