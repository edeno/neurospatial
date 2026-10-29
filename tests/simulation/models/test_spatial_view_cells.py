"""Tests for SpatialViewCellModel."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from shapely.geometry import box

from neurospatial import Environment


class TestSpatialViewCellModelCreation:
    """Test SpatialViewCellModel dataclass creation."""

    @pytest.fixture
    def env(self):
        """Create a simple 2D environment."""
        return Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0, units="cm")

    def test_basic_creation(self, env):
        """Model can be created with required parameters."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=np.array([50.0, 50.0]),
            boundary_source="analysis",
        )

        assert model.env is env
        assert_allclose(model.preferred_view_location, [50.0, 50.0])

    def test_default_parameters(self, env):
        """Model uses sensible defaults for optional parameters."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=np.array([50.0, 50.0]),
            boundary_source="analysis",
        )

        # Default view field width: 10 cm in the environment's units
        assert env.units == "cm"
        assert model.view_field_width == pytest.approx(10.0)

        # Default rates
        assert model.max_rate > 0
        assert model.baseline_rate >= 0
        assert model.baseline_rate < model.max_rate

        # The view analysis's default gaze: the first wall along the gaze.
        import inspect

        from neurospatial.encoding.view import compute_view_rate

        default = inspect.signature(compute_view_rate).parameters["gaze_model"]
        assert model.gaze_model == default.default == "ray_cast"
        assert model.view_distance is None

        # Default visibility requirement
        assert isinstance(model.require_visibility, bool)

        # Default fov
        # Can be None (full 360°) or FieldOfView instance
        assert model.fov is None or hasattr(model.fov, "contains_angle")

    def test_custom_parameters(self, env):
        """Custom parameters are stored correctly."""
        from neurospatial.ops.visibility import FieldOfView
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        fov = FieldOfView.rat()

        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=np.array([50.0, 50.0]),
            view_field_width=15.0,
            view_distance=25.0,
            gaze_model="fixed_distance",
            max_rate=30.0,
            baseline_rate=0.5,
            require_visibility=True,
            fov=fov,
            boundary_source="analysis",
        )

        assert model.view_field_width == 15.0
        assert model.view_distance == 25.0
        assert model.gaze_model == "fixed_distance"
        assert model.max_rate == 30.0
        assert model.baseline_rate == 0.5
        assert model.require_visibility is True
        assert model.fov is fov


class TestParameterValidation:
    """Test parameter validation in __init__."""

    @pytest.fixture
    def env(self):
        """Create a simple 2D environment."""
        return Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0, units="cm")

    def test_invalid_preferred_view_location_shape_raises(self, env):
        """Invalid preferred_view_location shape raises ValueError."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        # 1D array with wrong length
        with pytest.raises(ValueError, match=r"preferred_view_location.*shape"):
            SpatialViewCellModel(
                env=env,
                preferred_view_location=np.array([50.0]),
                boundary_source="analysis",  # Wrong shape
            )

        # 2D array (should be 1D with 2 elements)
        with pytest.raises(ValueError, match=r"preferred_view_location.*shape"):
            SpatialViewCellModel(
                env=env,
                preferred_view_location=np.array([[50.0, 50.0]]),
                boundary_source="analysis",
            )

    def test_non_positive_view_field_width_raises(self, env):
        """Non-positive view_field_width raises ValueError."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        with pytest.raises(ValueError, match=r"view_field_width.*positive"):
            SpatialViewCellModel(
                env=env,
                preferred_view_location=np.array([50.0, 50.0]),
                view_field_width=0.0,
                boundary_source="analysis",
            )

        with pytest.raises(ValueError, match=r"view_field_width.*positive"):
            SpatialViewCellModel(
                env=env,
                preferred_view_location=np.array([50.0, 50.0]),
                view_field_width=-5.0,
                boundary_source="analysis",
            )

    def test_non_positive_view_distance_raises(self, env):
        """Non-positive view_distance raises ValueError."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        with pytest.raises(ValueError, match=r"view_distance.*positive"):
            SpatialViewCellModel(
                env=env,
                preferred_view_location=np.array([50.0, 50.0]),
                view_distance=0.0,
                gaze_model="fixed_distance",
                boundary_source="analysis",
            )

    def test_non_positive_max_rate_raises(self, env):
        """Non-positive max_rate raises ValueError."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        with pytest.raises(ValueError, match=r"max_rate.*positive"):
            SpatialViewCellModel(
                env=env,
                preferred_view_location=np.array([50.0, 50.0]),
                max_rate=0.0,
                boundary_source="analysis",
            )

    def test_negative_baseline_rate_raises(self, env):
        """Negative baseline_rate raises ValueError."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        with pytest.raises(ValueError, match=r"baseline_rate.*non-negative"):
            SpatialViewCellModel(
                env=env,
                preferred_view_location=np.array([50.0, 50.0]),
                baseline_rate=-1.0,
                boundary_source="analysis",
            )

    def test_baseline_rate_exceeds_max_rate_raises(self, env):
        """A baseline above the peak raises ValueError."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        with pytest.raises(ValueError, match=r"baseline_rate .* is above max_rate"):
            SpatialViewCellModel(
                env=env,
                preferred_view_location=np.array([50.0, 50.0]),
                max_rate=10.0,
                baseline_rate=15.0,
                boundary_source="analysis",
            )

    def test_invalid_gaze_model_raises(self, env):
        """Invalid gaze_model raises ValueError."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        with pytest.raises(ValueError, match=r"gaze_model.*fixed_distance.*ray_cast"):
            SpatialViewCellModel(
                env=env,
                preferred_view_location=np.array([50.0, 50.0]),
                gaze_model="invalid",
                boundary_source="analysis",
            )

    def test_preferred_view_location_outside_warns(self, env):
        """preferred_view_location outside environment emits warning."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        with pytest.warns(UserWarning, match=r"outside.*environment"):
            SpatialViewCellModel(
                env=env,
                preferred_view_location=np.array([200.0, 200.0]),
                boundary_source="analysis",
            )


class TestFiringRateComputation:
    """Test firing_rate() method."""

    @pytest.fixture
    def env(self):
        """Create a simple 2D environment."""
        return Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0, units="cm")

    def test_firing_rate_shape(self, env):
        """Firing rate has correct shape."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=np.array([50.0, 50.0]),
            boundary_source="analysis",
        )

        positions = np.random.default_rng(42).uniform(0, 100, (100, 2))
        headings = np.random.default_rng(42).uniform(-np.pi, np.pi, 100)

        rates = model.firing_rate(
            np.zeros(len(positions)), positions, headings=headings
        )

        assert rates.shape == (100,)

    def test_firing_rate_peaks_at_preferred_view_location(self, env):
        """Firing rate is highest when viewing preferred location."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        preferred_location = np.array([75.0, 50.0])  # Target on the right

        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=preferred_location,
            view_field_width=10.0,
            view_distance=25.0,  # View distance matches distance to target
            gaze_model="fixed_distance",
            max_rate=20.0,
            baseline_rate=0.0,
            boundary_source="analysis",
        )

        # Position to the West of the target, looking East
        position_viewing = np.array([[50.0, 50.0]])
        heading_viewing = np.array([0.0])  # Facing East, toward preferred location

        # Same position but looking away (West)
        heading_away = np.array([np.pi])  # Facing West, away from preferred location

        rate_viewing = model.firing_rate(
            np.zeros(len(position_viewing)), position_viewing, headings=heading_viewing
        )[0]
        rate_away = model.firing_rate(
            np.zeros(len(position_viewing)), position_viewing, headings=heading_away
        )[0]

        # Rate when viewing preferred location should be higher
        assert rate_viewing > rate_away

        # Rate when viewing should be near max_rate
        assert rate_viewing > 0.5 * model.max_rate

    def test_gaussian_tuning_shape(self, env):
        """Spatial tuning follows Gaussian shape around preferred view location."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        preferred_location = np.array([50.0, 50.0])
        view_field_width = 10.0
        view_distance = 20.0

        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=preferred_location,
            view_field_width=view_field_width,
            view_distance=view_distance,
            gaze_model="fixed_distance",
            max_rate=20.0,
            baseline_rate=0.0,
            boundary_source="analysis",
        )

        # Animal at position looking at various locations along x-axis
        animal_position = np.array([30.0, 50.0])  # 20 units West of preferred
        heading = 0.0  # Looking East

        # At this distance, viewed location is 30 + 20 = 50, 50 (exactly at preferred)
        rate_exact = model.firing_rate(
            np.zeros(1), animal_position.reshape(1, -1), headings=np.array([heading])
        )[0]

        # Move animal 10 units North - now viewed location is at (50, 60)
        animal_north = np.array([30.0, 60.0])
        rate_north = model.firing_rate(
            np.zeros(1), animal_north.reshape(1, -1), headings=np.array([heading])
        )[0]

        # Viewed location is 10 units from preferred
        # Expected Gaussian response
        expected_ratio = np.exp(-0.5 * (10.0 / view_field_width) ** 2)

        # rate_north should be approximately expected_ratio * rate_exact
        assert rate_exact > rate_north
        assert_allclose(rate_north / rate_exact, expected_ratio, rtol=0.1)

    def test_headings_required(self, env):
        """firing_rate() raises ValueError if headings not provided."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=np.array([50.0, 50.0]),
            boundary_source="analysis",
        )

        positions = np.array([[60.0, 50.0]])

        with pytest.raises(ValueError, match="needs headings"):
            model.firing_rate(np.zeros(len(positions)), positions)


class TestGazeModels:
    """Test different gaze models."""

    @pytest.fixture
    def env(self):
        """Create a simple 2D environment."""
        return Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0, units="cm")

    def test_fixed_distance_gaze_model(self, env):
        """fixed_distance gaze model works correctly."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=np.array([70.0, 50.0]),
            view_distance=20.0,
            gaze_model="fixed_distance",
            max_rate=20.0,
            baseline_rate=0.0,
            boundary_source="analysis",
        )

        # Position at (50, 50), heading East
        # Viewed location should be at (70, 50) = exactly preferred
        position = np.array([[50.0, 50.0]])
        heading = np.array([0.0])

        rate = model.firing_rate(np.zeros(len(position)), position, headings=heading)[0]

        # Should be very high (close to max)
        assert rate > 0.8 * model.max_rate

    def test_ray_cast_gaze_model(self, env):
        """The ray-cast gaze lands on the first wall along the heading."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=np.array([100.0, 50.0]),
            view_field_width=10.0,
            gaze_model="ray_cast",
            max_rate=20.0,
            baseline_rate=0.0,
            boundary_source="analysis",
        )

        # From (30, 50) the gaze East lands on (100, 50), the gaze North on
        # (30, 100), 86 cm (8.6 widths) from the preferred location.
        positions = np.array([[30.0, 50.0], [30.0, 50.0]])
        rates = model.firing_rate(
            np.zeros(2), positions, headings=np.array([0.0, np.pi / 2])
        )
        assert_allclose(
            rates, [20.0, 20.0 * np.exp(-0.5 * (np.hypot(70, 50) / 10) ** 2)]
        )

    def test_ray_cast_from_wall_views_the_opposite_wall(self):
        """An animal tracked on the wall views the wall it faces, not itself."""
        from shapely.geometry import box

        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        env = Environment.from_polygon(box(0, 0, 10, 10), bin_size=1.0)
        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=np.array([9.75, 4.5]),
            view_field_width=1.0,
            gaze_model="ray_cast",
            require_visibility=True,
            max_rate=20.0,
            baseline_rate=0.0,
        )
        rates = model.firing_rate(
            np.zeros(2),
            np.array([[0.0, 4.5], [0.0, 4.5]]),
            headings=np.array([0.0, np.pi]),
        )
        # Facing east views (10, 4.5), 0.25 from the preferred location; facing
        # west views the animal's own wall contact, 9.75 away.
        assert_allclose(rates, [20.0 * np.exp(-0.5 * 0.25**2), 0.0], atol=1e-9)


class TestVisibilityRequirement:
    """require_visibility needs a clear line of sight to the preferred location."""

    @pytest.fixture(scope="class")
    def divided(self):
        """A 1 m box with a 4 cm thick divider from the south wall to y = 60."""
        footprint = box(0, 0, 100, 100).difference(box(48, 0, 52, 60))
        return Environment.from_polygon(footprint, bin_size=2.0, units="cm")

    @pytest.mark.parametrize(("require", "expected"), [(False, 0.726), (True, 0.0)])
    def test_divider_blocks_the_view_only_when_visibility_is_required(
        self, divided, require, expected
    ):
        """The gaze stops on the divider 12 cm from the location behind it."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        model = SpatialViewCellModel(
            divided,
            preferred_view_location=np.array([60.0, 30.0]),
            view_field_width=15.0,
            view_distance=20.0,
            gaze_model="fixed_distance",
            require_visibility=require,
            max_rate=20.0,
            baseline_rate=0.0,
        )
        rate = model.firing_rate(
            np.zeros(1), np.array([[40.0, 30.0]]), headings=np.array([0.0])
        )
        assert rate[0] == pytest.approx(20.0 * expected, abs=0.01)

    def test_clear_line_of_sight_keeps_the_rate(self, divided):
        """North of the divider the location is in plain view."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        kwargs = {
            "preferred_view_location": np.array([60.0, 80.0]),
            "view_field_width": 15.0,
            "view_distance": 20.0,
            "gaze_model": "fixed_distance",
            "max_rate": 20.0,
            "baseline_rate": 0.0,
        }
        args = (np.zeros(1), np.array([[40.0, 80.0]]))
        headings = np.array([0.0])
        visible = SpatialViewCellModel(divided, require_visibility=True, **kwargs)
        free = SpatialViewCellModel(divided, **kwargs)
        assert visible.firing_rate(*args, headings=headings)[0] == pytest.approx(20.0)
        assert free.firing_rate(*args, headings=headings)[0] == pytest.approx(20.0)


class TestFieldOfViewIntegration:
    """fov: the preferred location must lie inside the animal's field of view."""

    @pytest.fixture
    def env(self):
        """Create a simple 2D environment."""
        return Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0, units="cm")

    @pytest.mark.parametrize("require_visibility", [False, True])
    @pytest.mark.parametrize(
        ("half_angle_deg", "in_view"), [(None, True), (80.0, True), (30.0, False)]
    )
    def test_location_off_to_the_side_needs_a_wide_field_of_view(
        self, env, require_visibility, half_angle_deg, in_view
    ):
        """Facing East 5 cm from the east wall, the location 10 cm North on the
        wall is 63 degrees to the left: the gaze lands near it, but a 30-degree
        field of view does not hold it."""
        from neurospatial.ops.visibility import FieldOfView
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        fov = (
            None
            if half_angle_deg is None
            else FieldOfView.symmetric(half_angle=np.radians(half_angle_deg))
        )
        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=np.array([100.0, 50.0]),
            view_field_width=30.0,
            fov=fov,
            require_visibility=require_visibility,
            max_rate=20.0,
            baseline_rate=0.5,
        )
        rate = model.firing_rate(
            np.zeros(1), np.array([[95.0, 40.0]]), headings=np.array([0.0])
        )[0]
        # The gaze lands on (100, 40), 10 cm from the preferred location.
        gaze_rate = 0.5 + 19.5 * np.exp(-0.5 * (10.0 / 30.0) ** 2)
        assert rate == pytest.approx(gaze_rate if in_view else 0.5)


class TestGroundTruth:
    """Test ground_truth property."""

    @pytest.fixture
    def env(self):
        """Create a simple 2D environment."""
        return Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0, units="cm")

    def test_ground_truth_contains_all_parameters(self, env):
        """ground_truth contains all model parameters."""
        from neurospatial.ops.visibility import FieldOfView
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=np.array([50.0, 50.0]),
            view_field_width=15.0,
            view_distance=25.0,
            gaze_model="fixed_distance",
            max_rate=30.0,
            baseline_rate=0.5,
            require_visibility=True,
            fov=FieldOfView.rat(),
            boundary_source="analysis",
        )

        gt = model.ground_truth

        # Check all expected keys
        assert "preferred_view_location" in gt
        assert "view_field_width" in gt
        assert "view_distance" in gt
        assert "gaze_model" in gt
        assert "max_rate" in gt
        assert "baseline_rate" in gt
        assert "require_visibility" in gt
        assert "fov" in gt

        # Check values
        assert_allclose(gt["preferred_view_location"], [50.0, 50.0])
        assert gt["view_field_width"] == 15.0
        assert gt["view_distance"] == 25.0
        assert gt["gaze_model"] == "fixed_distance"
        assert gt["max_rate"] == 30.0
        assert gt["baseline_rate"] == 0.5
        assert gt["require_visibility"] is True

    def test_ground_truth_immutable(self, env):
        """ground_truth returns a copy, not original values."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=np.array([50.0, 50.0]),
            boundary_source="analysis",
        )

        gt = model.ground_truth

        # Modify the ground truth dict
        gt["view_field_width"] = 999.0

        # Original model should be unchanged
        assert model.view_field_width != 999.0


class TestVectorizationEquivalence:
    """Equivalence guards for the vectorized firing_rate FOV/visibility paths."""

    @pytest.fixture
    def env(self):
        """Create a simple 2D environment."""
        return Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0, units="cm")

    @staticmethod
    def _reference_firing_rate(model, positions, headings):
        """Reference firing rate using the original per-timestep Python loops."""
        from neurospatial.geometry._boundaries import line_of_sight_clear
        from neurospatial.ops.egocentric import compute_egocentric_bearing
        from neurospatial.ops.visibility import compute_viewed_location

        positions = np.asarray(positions, dtype=np.float64)
        headings = np.asarray(headings, dtype=np.float64)
        n_time = len(positions)

        viewed_locations = compute_viewed_location(
            positions,
            headings,
            method=model.gaze_model,
            view_distance=model.view_distance,
            env=model.env,
            boundary_source="analysis",
        )

        distances = np.linalg.norm(
            viewed_locations - model.preferred_view_location, axis=1
        )
        response = np.exp(-0.5 * (distances / model.view_field_width) ** 2)
        nan_mask = np.isnan(distances)
        response[nan_mask] = 0.0

        for i in range(n_time):
            if nan_mask[i]:
                continue
            if model.fov is not None:
                bearing = compute_egocentric_bearing(
                    positions[i : i + 1],
                    headings[i : i + 1],
                    model.preferred_view_location[None, :],
                )[0, 0]
                if not model.fov.contains_angle(bearing):
                    response[i] = 0.0
                    continue
            if model.require_visibility and not line_of_sight_clear(
                model.env.binning,
                model.env.arena,
                positions[i],
                model.preferred_view_location,
                boundary_source="analysis",
            ):
                response[i] = 0.0

        return model.baseline_rate + (model.max_rate - model.baseline_rate) * response

    def test_fov_only_path_matches_reference_loop(self, env):
        """FOV-only path (require_visibility=False) matches the loop reference."""
        from neurospatial.ops.visibility import FieldOfView
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=np.array([50.0, 50.0]),
            view_field_width=12.0,
            view_distance=20.0,
            gaze_model="fixed_distance",
            fov=FieldOfView.symmetric(half_angle=np.pi / 4),
            require_visibility=False,
            max_rate=20.0,
            baseline_rate=0.3,
            boundary_source="analysis",
        )

        rng = np.random.default_rng(7)
        positions = rng.uniform(0, 100, (300, 2))
        headings = rng.uniform(-np.pi, np.pi, 300)

        actual = model.firing_rate(
            np.zeros(len(positions)), positions, headings=headings
        )
        expected = self._reference_firing_rate(model, positions, headings)
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_visibility_path_matches_reference_loop(self, env):
        """require_visibility=True path (FOV + line of sight) matches reference."""
        from neurospatial.ops.visibility import FieldOfView
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=np.array([50.0, 50.0]),
            view_field_width=12.0,
            view_distance=15.0,
            gaze_model="fixed_distance",
            fov=FieldOfView.symmetric(half_angle=np.pi / 3),
            require_visibility=True,
            max_rate=20.0,
            baseline_rate=0.3,
            boundary_source="analysis",
        )

        rng = np.random.default_rng(11)
        positions = rng.uniform(0, 100, (60, 2))
        headings = rng.uniform(-np.pi, np.pi, 60)

        actual = model.firing_rate(
            np.zeros(len(positions)), positions, headings=headings
        )
        expected = self._reference_firing_rate(model, positions, headings)
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_visibility_path_no_fov_matches_reference_loop(self, env):
        """require_visibility=True with fov=None matches reference (LOS only)."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=np.array([50.0, 50.0]),
            view_field_width=12.0,
            view_distance=15.0,
            gaze_model="fixed_distance",
            fov=None,
            require_visibility=True,
            max_rate=20.0,
            baseline_rate=0.3,
            boundary_source="analysis",
        )

        rng = np.random.default_rng(13)
        positions = rng.uniform(0, 100, (60, 2))
        headings = rng.uniform(-np.pi, np.pi, 60)

        actual = model.firing_rate(
            np.zeros(len(positions)), positions, headings=headings
        )
        expected = self._reference_firing_rate(model, positions, headings)
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


class TestProtocolCompliance:
    """Test compliance with NeuralModel protocol."""

    @pytest.fixture
    def env(self):
        """Create a simple 2D environment."""
        return Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0, units="cm")

    def test_implements_neural_model_protocol(self, env):
        """SpatialViewCellModel implements NeuralModel protocol."""
        from neurospatial.simulation.models.base import NeuralModel
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=np.array([50.0, 50.0]),
            boundary_source="analysis",
        )

        assert isinstance(model, NeuralModel)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.fixture
    def env(self):
        """Create a simple 2D environment."""
        return Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0, units="cm")

    def test_missing_input_or_outside_position_is_nan(self, env):
        """Missing is not zero: nothing defines what an unobserved animal sees."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=np.array([100.0, 50.0]),
            max_rate=20.0,
            baseline_rate=0.5,
            require_visibility=True,
        )
        positions = np.array(
            [[50.0, 50.0], [np.nan, 50.0], [50.0, 50.0], [150.0, 50.0], [50.0, 50.0]]
        )
        headings = np.array([0.0, 0.0, np.nan, 0.0, np.pi])

        rates = model.firing_rate(np.zeros(5), positions, headings=headings)

        assert rates[0] == pytest.approx(20.0)
        np.testing.assert_array_equal(np.isnan(rates), [False, True, True, True, False])
        # Looking away from the location is observed: the baseline, not NaN.
        assert rates[4] == pytest.approx(0.5, abs=1e-6)

    def test_single_position(self, env):
        """Model works with single position."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=np.array([50.0, 50.0]),
            boundary_source="analysis",
        )

        position = np.array([[60.0, 50.0]])
        heading = np.array([np.pi])

        rate = model.firing_rate(np.zeros(len(position)), position, headings=heading)

        assert rate.shape == (1,)


@pytest.mark.parametrize("gaze_model", ["fixed_distance", "ray_cast"])
def test_firing_rate_builds_the_mesh_lookup_once_per_call(gaze_model, monkeypatch):
    """The triangle index is built a fixed number of times, not per timestep."""
    from shapely.geometry import box

    from neurospatial import Environment
    from neurospatial.geometry import _locators
    from neurospatial.layout import create_layout
    from neurospatial.simulation.models.spatial_view_cells import SpatialViewCellModel

    env = Environment.from_layout(
        create_layout(
            "TriangularMesh", boundary_polygon=box(0, 0, 20, 20), point_spacing=2.0
        )
    )
    model = SpatialViewCellModel(
        env,
        preferred_view_location=np.array([10.0, 10.0]),
        view_field_width=10.0,
        gaze_model=gaze_model,
        view_distance=5.0 if gaze_model == "fixed_distance" else None,
        require_visibility=True,
        boundary_source="analysis",
    )
    builds: list[int] = []
    build = _locators._triangle_index

    def counted(*args):
        builds.append(1)
        return build(*args)

    monkeypatch.setattr(_locators, "_triangle_index", counted)
    rng = np.random.default_rng(0)
    counts = []
    for n_time in (12, 48):
        builds.clear()
        rates = model.firing_rate(
            np.arange(n_time) / 30,
            rng.uniform(3, 17, (n_time, 2)),
            headings=rng.uniform(-np.pi, np.pi, n_time),
        )
        assert np.all(np.isfinite(rates))
        counts.append(len(builds))
    assert counts[0] == counts[1] < 12


class TestGazeStopsAtWalls:
    """The simulated gaze is the view analysis's gaze: it never crosses a wall."""

    @pytest.fixture(scope="class")
    def divided(self):
        """A 1 m box with a 4 cm thick divider from the south wall to y = 60."""
        footprint = box(0, 0, 100, 100).difference(box(48, 0, 52, 60))
        return Environment.from_polygon(footprint, bin_size=2.0, units="cm")

    def test_a_location_behind_a_wall_is_not_seen(self, divided):
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        # 20 cm east of the observer, behind the divider 8 cm away.
        model = SpatialViewCellModel(
            divided,
            preferred_view_location=np.array([60.0, 30.0]),
            view_field_width=3.0,
            view_distance=20.0,
            gaze_model="fixed_distance",
            max_rate=20.0,
            baseline_rate=0.1,
        )
        rate = model.firing_rate(
            np.zeros(1), np.array([[40.0, 30.0]]), headings=np.array([0.0])
        )
        # The gaze stops at the divider (48, 30), 12 cm (4 widths) away.
        assert rate[0] == pytest.approx(0.1 + 19.9 * np.exp(-0.5 * 4.0**2))

    @pytest.mark.parametrize("gaze_model", ["ray_cast", "fixed_distance"])
    def test_rates_follow_the_analysis_gaze(self, divided, gaze_model):
        from neurospatial.ops.visibility import compute_viewed_location
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        model = SpatialViewCellModel(
            divided,
            preferred_view_location=np.array([48.0, 30.0]),
            view_field_width=10.0,
            gaze_model=gaze_model,
            view_distance=20.0 if gaze_model == "fixed_distance" else None,
            max_rate=20.0,
            baseline_rate=0.0,
        )
        rng = np.random.default_rng(5)
        positions = divided.bin_centers[rng.integers(0, divided.n_bins, 500)]
        headings = rng.uniform(-np.pi, np.pi, 500)
        viewed = compute_viewed_location(
            positions,
            headings,
            method=gaze_model,
            view_distance=model.view_distance,
            env=divided,
        )
        distance = np.linalg.norm(viewed - [48.0, 30.0], axis=1)
        assert_allclose(
            model.firing_rate(np.zeros(len(positions)), positions, headings=headings),
            20.0 * np.exp(-0.5 * (distance / 10.0) ** 2),
        )
