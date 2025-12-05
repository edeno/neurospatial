"""Tests for SpatialViewCellModel."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial import Environment


class TestModuleSetup:
    """Test module imports and structure."""

    def test_import_module(self):
        """Module can be imported."""
        from neurospatial.simulation.models import spatial_view_cells

        assert spatial_view_cells is not None

    def test_import_class(self):
        """SpatialViewCellModel can be imported."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        assert SpatialViewCellModel is not None

    def test_module_docstring_exists(self):
        """Module has a docstring."""
        from neurospatial.simulation.models import spatial_view_cells

        assert spatial_view_cells.__doc__ is not None
        assert len(spatial_view_cells.__doc__) > 50


class TestSpatialViewCellModelCreation:
    """Test SpatialViewCellModel dataclass creation."""

    @pytest.fixture
    def env(self):
        """Create a simple 2D environment."""
        samples = np.random.default_rng(42).uniform(0, 100, (500, 2))
        return Environment.from_samples(samples, bin_size=2.0)

    def test_basic_creation(self, env):
        """Model can be created with required parameters."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=np.array([50.0, 50.0]),
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
        )

        # Default view field width
        assert model.view_field_width > 0

        # Default rates
        assert model.max_rate > 0
        assert model.baseline_rate >= 0
        assert model.baseline_rate < model.max_rate

        # Default view distance
        assert model.view_distance > 0

        # Default gaze model
        assert model.gaze_model in ("fixed_distance", "ray_cast", "boundary")

        # Default visibility requirement
        assert isinstance(model.require_visibility, bool)

        # Default fov
        # Can be None (full 360°) or FieldOfView instance
        assert model.fov is None or hasattr(model.fov, "contains_angle")

    def test_custom_parameters(self, env):
        """Custom parameters are stored correctly."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )
        from neurospatial.visibility import FieldOfView

        fov = FieldOfView.rat()

        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=np.array([50.0, 50.0]),
            view_field_width=15.0,
            view_distance=25.0,
            gaze_model="ray_cast",
            max_rate=30.0,
            baseline_rate=0.5,
            require_visibility=True,
            fov=fov,
        )

        assert model.view_field_width == 15.0
        assert model.view_distance == 25.0
        assert model.gaze_model == "ray_cast"
        assert model.max_rate == 30.0
        assert model.baseline_rate == 0.5
        assert model.require_visibility is True
        assert model.fov is fov


class TestParameterValidation:
    """Test parameter validation in __init__."""

    @pytest.fixture
    def env(self):
        """Create a simple 2D environment."""
        samples = np.random.default_rng(42).uniform(0, 100, (500, 2))
        return Environment.from_samples(samples, bin_size=2.0)

    def test_invalid_preferred_view_location_shape_raises(self, env):
        """Invalid preferred_view_location shape raises ValueError."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        # 1D array with wrong length
        with pytest.raises(ValueError, match=r"preferred_view_location.*shape"):
            SpatialViewCellModel(
                env=env,
                preferred_view_location=np.array([50.0]),  # Wrong shape
            )

        # 2D array (should be 1D with 2 elements)
        with pytest.raises(ValueError, match=r"preferred_view_location.*shape"):
            SpatialViewCellModel(
                env=env,
                preferred_view_location=np.array([[50.0, 50.0]]),
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
            )

        with pytest.raises(ValueError, match=r"view_field_width.*positive"):
            SpatialViewCellModel(
                env=env,
                preferred_view_location=np.array([50.0, 50.0]),
                view_field_width=-5.0,
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
            )

    def test_baseline_rate_exceeds_max_rate_raises(self, env):
        """baseline_rate >= max_rate raises ValueError."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        with pytest.raises(ValueError, match=r"baseline_rate.*less than.*max_rate"):
            SpatialViewCellModel(
                env=env,
                preferred_view_location=np.array([50.0, 50.0]),
                max_rate=10.0,
                baseline_rate=15.0,
            )

    def test_invalid_gaze_model_raises(self, env):
        """Invalid gaze_model raises ValueError."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        with pytest.raises(
            ValueError, match=r"gaze_model.*fixed_distance.*ray_cast.*boundary"
        ):
            SpatialViewCellModel(
                env=env,
                preferred_view_location=np.array([50.0, 50.0]),
                gaze_model="invalid",
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
            )


class TestFiringRateComputation:
    """Test firing_rate() method."""

    @pytest.fixture
    def env(self):
        """Create a simple 2D environment."""
        samples = np.random.default_rng(42).uniform(0, 100, (500, 2))
        return Environment.from_samples(samples, bin_size=2.0)

    def test_firing_rate_shape(self, env):
        """Firing rate has correct shape."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=np.array([50.0, 50.0]),
        )

        positions = np.random.default_rng(42).uniform(0, 100, (100, 2))
        headings = np.random.default_rng(42).uniform(-np.pi, np.pi, 100)

        rates = model.firing_rate(positions, headings=headings)

        assert rates.shape == (100,)

    def test_firing_rate_non_negative(self, env):
        """Firing rates are always non-negative."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=np.array([50.0, 50.0]),
        )

        positions = np.random.default_rng(42).uniform(0, 100, (1000, 2))
        headings = np.random.default_rng(42).uniform(-np.pi, np.pi, 1000)

        rates = model.firing_rate(positions, headings=headings)

        assert np.all(rates >= 0)

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
        )

        # Position to the West of the target, looking East
        position_viewing = np.array([[50.0, 50.0]])
        heading_viewing = np.array([0.0])  # Facing East, toward preferred location

        # Same position but looking away (West)
        heading_away = np.array([np.pi])  # Facing West, away from preferred location

        rate_viewing = model.firing_rate(position_viewing, headings=heading_viewing)[0]
        rate_away = model.firing_rate(position_viewing, headings=heading_away)[0]

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
        )

        # Animal at position looking at various locations along x-axis
        animal_position = np.array([30.0, 50.0])  # 20 units West of preferred
        heading = 0.0  # Looking East

        # At this distance, viewed location is 30 + 20 = 50, 50 (exactly at preferred)
        rate_exact = model.firing_rate(
            animal_position.reshape(1, -1), headings=np.array([heading])
        )[0]

        # Move animal 10 units North - now viewed location is at (50, 60)
        animal_north = np.array([30.0, 60.0])
        rate_north = model.firing_rate(
            animal_north.reshape(1, -1), headings=np.array([heading])
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
        )

        positions = np.array([[60.0, 50.0]])

        with pytest.raises(ValueError, match=r"headings.*required"):
            model.firing_rate(positions)


class TestGazeModels:
    """Test different gaze models."""

    @pytest.fixture
    def env(self):
        """Create a simple 2D environment."""
        samples = np.random.default_rng(42).uniform(0, 100, (500, 2))
        return Environment.from_samples(samples, bin_size=2.0)

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
        )

        # Position at (50, 50), heading East
        # Viewed location should be at (70, 50) = exactly preferred
        position = np.array([[50.0, 50.0]])
        heading = np.array([0.0])

        rate = model.firing_rate(position, headings=heading)[0]

        # Should be very high (close to max)
        assert rate > 0.8 * model.max_rate

    def test_ray_cast_gaze_model(self, env):
        """ray_cast gaze model works correctly."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=np.array([50.0, 50.0]),
            gaze_model="ray_cast",
            max_rate=20.0,
            baseline_rate=0.0,
        )

        # Ray cast should work without errors
        position = np.array([[30.0, 50.0]])
        heading = np.array([0.0])

        rate = model.firing_rate(position, headings=heading)
        assert len(rate) == 1
        assert rate[0] >= 0


class TestVisibilityRequirement:
    """Test require_visibility flag behavior."""

    @pytest.fixture
    def env(self):
        """Create a simple 2D environment."""
        samples = np.random.default_rng(42).uniform(0, 100, (500, 2))
        return Environment.from_samples(samples, bin_size=2.0)

    def test_require_visibility_false(self, env):
        """With require_visibility=False, fires even if view blocked."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=np.array([70.0, 50.0]),
            view_distance=20.0,
            gaze_model="fixed_distance",
            require_visibility=False,
            max_rate=20.0,
        )

        # Set up position and heading to look at preferred location
        position = np.array([[50.0, 50.0]])
        heading = np.array([0.0])

        rate = model.firing_rate(position, headings=heading)[0]

        # Should fire even without visibility check
        assert rate > 0

    def test_require_visibility_true_checks_line_of_sight(self, env):
        """With require_visibility=True, checks line of sight."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=np.array([50.0, 50.0]),
            view_distance=10.0,
            gaze_model="fixed_distance",
            require_visibility=True,
            max_rate=20.0,
        )

        # Inside environment looking at preferred location
        position = np.array([[40.0, 50.0]])
        heading = np.array([0.0])

        rate = model.firing_rate(position, headings=heading)[0]

        # Should fire if view is clear
        # (In a filled environment, view should generally be clear)
        assert rate >= 0  # At least doesn't crash


class TestFieldOfViewIntegration:
    """Test field of view integration."""

    @pytest.fixture
    def env(self):
        """Create a simple 2D environment."""
        samples = np.random.default_rng(42).uniform(0, 100, (500, 2))
        return Environment.from_samples(samples, bin_size=2.0)

    def test_fov_restricts_viewing(self, env):
        """FOV restricts what can be viewed."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )
        from neurospatial.visibility import FieldOfView

        # Narrow FOV (only forward 60 degrees)
        narrow_fov = FieldOfView.symmetric(half_angle=np.pi / 6)  # 30 degrees each side

        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=np.array([50.0, 75.0]),  # North of center
            view_distance=25.0,
            gaze_model="fixed_distance",
            fov=narrow_fov,
            max_rate=20.0,
            baseline_rate=0.0,
        )

        # Position at center, facing East
        position = np.array([[50.0, 50.0]])
        heading_east = np.array([0.0])  # North is 90° away from view direction

        # Position at center, facing North (toward preferred)
        heading_north = np.array([np.pi / 2])

        rate_east = model.firing_rate(position, headings=heading_east)[0]
        rate_north = model.firing_rate(position, headings=heading_north)[0]

        # When facing North (toward preferred), rate should be higher
        # When facing East, preferred is outside narrow FOV
        assert rate_north > rate_east


class TestGroundTruth:
    """Test ground_truth property."""

    @pytest.fixture
    def env(self):
        """Create a simple 2D environment."""
        samples = np.random.default_rng(42).uniform(0, 100, (500, 2))
        return Environment.from_samples(samples, bin_size=2.0)

    def test_ground_truth_contains_all_parameters(self, env):
        """ground_truth contains all model parameters."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )
        from neurospatial.visibility import FieldOfView

        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=np.array([50.0, 50.0]),
            view_field_width=15.0,
            view_distance=25.0,
            gaze_model="ray_cast",
            max_rate=30.0,
            baseline_rate=0.5,
            require_visibility=True,
            fov=FieldOfView.rat(),
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
        assert gt["gaze_model"] == "ray_cast"
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
        )

        gt = model.ground_truth

        # Modify the ground truth dict
        gt["view_field_width"] = 999.0

        # Original model should be unchanged
        assert model.view_field_width != 999.0


class TestProtocolCompliance:
    """Test compliance with NeuralModel protocol."""

    @pytest.fixture
    def env(self):
        """Create a simple 2D environment."""
        samples = np.random.default_rng(42).uniform(0, 100, (500, 2))
        return Environment.from_samples(samples, bin_size=2.0)

    def test_implements_neural_model_protocol(self, env):
        """SpatialViewCellModel implements NeuralModel protocol."""
        from neurospatial.simulation.models.base import NeuralModel
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=np.array([50.0, 50.0]),
        )

        assert isinstance(model, NeuralModel)

    def test_firing_rate_signature(self, env):
        """firing_rate() has correct signature."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=np.array([50.0, 50.0]),
        )

        positions = np.array([[60.0, 50.0]])
        headings = np.array([0.0])

        # With positions and headings
        rates = model.firing_rate(positions, headings=headings)
        assert isinstance(rates, np.ndarray)

        # With positions, times, and headings
        times = np.array([0.0])
        rates = model.firing_rate(positions, times, headings=headings)
        assert isinstance(rates, np.ndarray)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.fixture
    def env(self):
        """Create a simple 2D environment."""
        samples = np.random.default_rng(42).uniform(0, 100, (500, 2))
        return Environment.from_samples(samples, bin_size=2.0)

    def test_nan_viewed_location_returns_baseline(self, env):
        """When viewed location is NaN (outside env), returns baseline rate."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        # Put preferred outside environment
        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=np.array([50.0, 50.0]),
            view_distance=200.0,  # Very far - likely outside env
            gaze_model="fixed_distance",
            max_rate=20.0,
            baseline_rate=0.5,
        )

        # Position and heading that would look outside environment
        position = np.array([[10.0, 50.0]])
        heading = np.array([np.pi])  # Looking West, away from environment

        rate = model.firing_rate(position, headings=heading)[0]

        # Should return baseline rate when viewing outside
        assert rate == pytest.approx(model.baseline_rate, rel=0.1)

    def test_single_position(self, env):
        """Model works with single position."""
        from neurospatial.simulation.models.spatial_view_cells import (
            SpatialViewCellModel,
        )

        model = SpatialViewCellModel(
            env=env,
            preferred_view_location=np.array([50.0, 50.0]),
        )

        position = np.array([[60.0, 50.0]])
        heading = np.array([np.pi])

        rate = model.firing_rate(position, headings=heading)

        assert rate.shape == (1,)
