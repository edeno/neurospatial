"""Tests for ObjectVectorCellModel."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial import Environment


class TestModuleSetup:
    """Test module imports and structure."""

    def test_import_module(self):
        """Module can be imported."""
        from neurospatial.simulation.models import object_vector_cells

        assert object_vector_cells is not None

    def test_import_class(self):
        """ObjectVectorCellModel can be imported."""
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        assert ObjectVectorCellModel is not None

    def test_module_docstring_exists(self):
        """Module has a docstring."""
        from neurospatial.simulation.models import object_vector_cells

        assert object_vector_cells.__doc__ is not None
        assert len(object_vector_cells.__doc__) > 50


class TestObjectVectorCellModelCreation:
    """Test ObjectVectorCellModel dataclass creation."""

    @pytest.fixture
    def env(self):
        """Create a simple 2D environment."""
        samples = np.random.default_rng(42).uniform(0, 100, (500, 2))
        return Environment.from_samples(samples, bin_size=2.0)

    @pytest.fixture
    def object_positions(self):
        """Sample object positions."""
        return np.array([[25.0, 25.0], [75.0, 75.0]])

    def test_basic_creation(self, env, object_positions):
        """Model can be created with required parameters."""
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        model = ObjectVectorCellModel(
            env=env,
            object_positions=object_positions,
            preferred_distance=10.0,
            distance_width=5.0,
        )

        assert model.env is env
        assert_allclose(model.object_positions, object_positions)
        assert model.preferred_distance == 10.0
        assert model.distance_width == 5.0

    def test_default_parameters(self, env, object_positions):
        """Model uses sensible defaults for optional parameters."""
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        model = ObjectVectorCellModel(
            env=env,
            object_positions=object_positions,
            preferred_distance=10.0,
            distance_width=5.0,
        )

        # Default direction tuning (None = omnidirectional)
        assert model.preferred_direction is None

        # Default rates
        assert model.max_rate > 0
        assert model.baseline_rate >= 0
        assert model.baseline_rate < model.max_rate

        # Default object selectivity
        assert model.object_selectivity == "nearest"

        # Default distance metric
        assert model.distance_metric == "euclidean"

    def test_directional_tuning_parameters(self, env, object_positions):
        """Directional tuning parameters are stored correctly."""
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        model = ObjectVectorCellModel(
            env=env,
            object_positions=object_positions,
            preferred_distance=10.0,
            distance_width=5.0,
            preferred_direction=np.pi / 4,  # 45 degrees
            direction_kappa=4.0,  # ~30 degree half-width
        )

        assert model.preferred_direction == pytest.approx(np.pi / 4)
        assert model.direction_kappa == 4.0

    def test_single_object(self, env):
        """Model works with a single object."""
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        single_object = np.array([[50.0, 50.0]])

        model = ObjectVectorCellModel(
            env=env,
            object_positions=single_object,
            preferred_distance=10.0,
            distance_width=5.0,
        )

        assert model.object_positions.shape == (1, 2)


class TestParameterValidation:
    """Test parameter validation in __post_init__."""

    @pytest.fixture
    def env(self):
        """Create a simple 2D environment."""
        samples = np.random.default_rng(42).uniform(0, 100, (500, 2))
        return Environment.from_samples(samples, bin_size=2.0)

    @pytest.fixture
    def object_positions(self):
        """Sample object positions."""
        return np.array([[25.0, 25.0], [75.0, 75.0]])

    def test_negative_preferred_distance_raises(self, env, object_positions):
        """Negative preferred_distance raises ValueError."""
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        with pytest.raises(ValueError, match=r"preferred_distance.*non-negative"):
            ObjectVectorCellModel(
                env=env,
                object_positions=object_positions,
                preferred_distance=-5.0,
                distance_width=5.0,
            )

    def test_non_positive_distance_width_raises(self, env, object_positions):
        """Non-positive distance_width raises ValueError."""
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        with pytest.raises(ValueError, match=r"distance_width.*positive"):
            ObjectVectorCellModel(
                env=env,
                object_positions=object_positions,
                preferred_distance=10.0,
                distance_width=0.0,
            )

        with pytest.raises(ValueError, match=r"distance_width.*positive"):
            ObjectVectorCellModel(
                env=env,
                object_positions=object_positions,
                preferred_distance=10.0,
                distance_width=-1.0,
            )

    def test_non_positive_max_rate_raises(self, env, object_positions):
        """Non-positive max_rate raises ValueError."""
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        with pytest.raises(ValueError, match=r"max_rate.*positive"):
            ObjectVectorCellModel(
                env=env,
                object_positions=object_positions,
                preferred_distance=10.0,
                distance_width=5.0,
                max_rate=0.0,
            )

    def test_negative_baseline_rate_raises(self, env, object_positions):
        """Negative baseline_rate raises ValueError."""
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        with pytest.raises(ValueError, match=r"baseline_rate.*non-negative"):
            ObjectVectorCellModel(
                env=env,
                object_positions=object_positions,
                preferred_distance=10.0,
                distance_width=5.0,
                baseline_rate=-1.0,
            )

    def test_baseline_rate_exceeds_max_rate_raises(self, env, object_positions):
        """baseline_rate >= max_rate raises ValueError."""
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        with pytest.raises(ValueError, match=r"baseline_rate.*less than.*max_rate"):
            ObjectVectorCellModel(
                env=env,
                object_positions=object_positions,
                preferred_distance=10.0,
                distance_width=5.0,
                max_rate=10.0,
                baseline_rate=15.0,
            )

    def test_non_positive_direction_kappa_raises(self, env, object_positions):
        """Non-positive direction_kappa raises ValueError."""
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        with pytest.raises(ValueError, match=r"direction_kappa.*positive"):
            ObjectVectorCellModel(
                env=env,
                object_positions=object_positions,
                preferred_distance=10.0,
                distance_width=5.0,
                preferred_direction=0.0,
                direction_kappa=0.0,
            )

    def test_invalid_object_selectivity_raises(self, env, object_positions):
        """Invalid object_selectivity raises ValueError."""
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        with pytest.raises(
            ValueError, match=r"object_selectivity.*any.*nearest.*specific"
        ):
            ObjectVectorCellModel(
                env=env,
                object_positions=object_positions,
                preferred_distance=10.0,
                distance_width=5.0,
                object_selectivity="invalid",
            )

    def test_invalid_distance_metric_raises(self, env, object_positions):
        """Invalid distance_metric raises ValueError."""
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        with pytest.raises(ValueError, match=r"distance_metric.*euclidean.*geodesic"):
            ObjectVectorCellModel(
                env=env,
                object_positions=object_positions,
                preferred_distance=10.0,
                distance_width=5.0,
                distance_metric="invalid",
            )

    def test_1d_object_positions_raises(self, env):
        """1D object_positions raises ValueError."""
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        with pytest.raises(ValueError, match=r"object_positions.*2D"):
            ObjectVectorCellModel(
                env=env,
                object_positions=np.array([25.0, 25.0]),  # 1D, not 2D
                preferred_distance=10.0,
                distance_width=5.0,
            )

    def test_objects_outside_environment_warns(self, env):
        """Objects outside environment bounds emit warning."""
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        outside_objects = np.array([[150.0, 150.0], [50.0, 50.0]])

        with pytest.warns(UserWarning, match=r"outside.*environment"):
            ObjectVectorCellModel(
                env=env,
                object_positions=outside_objects,
                preferred_distance=10.0,
                distance_width=5.0,
            )


class TestFiringRateDistanceTuning:
    """Test firing_rate() method with distance tuning only."""

    @pytest.fixture
    def env(self):
        """Create a simple 2D environment."""
        samples = np.random.default_rng(42).uniform(0, 100, (500, 2))
        return Environment.from_samples(samples, bin_size=2.0)

    def test_firing_rate_shape(self, env):
        """Firing rate has correct shape."""
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        model = ObjectVectorCellModel(
            env=env,
            object_positions=np.array([[50.0, 50.0]]),
            preferred_distance=10.0,
            distance_width=5.0,
        )

        positions = np.random.default_rng(42).uniform(0, 100, (100, 2))
        rates = model.firing_rate(positions)

        assert rates.shape == (100,)

    def test_firing_rate_non_negative(self, env):
        """Firing rates are always non-negative."""
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        model = ObjectVectorCellModel(
            env=env,
            object_positions=np.array([[50.0, 50.0]]),
            preferred_distance=10.0,
            distance_width=5.0,
        )

        positions = np.random.default_rng(42).uniform(0, 100, (1000, 2))
        rates = model.firing_rate(positions)

        assert np.all(rates >= 0)

    def test_firing_rate_peaks_at_preferred_distance(self, env):
        """Firing rate is highest at preferred distance from object."""
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        object_pos = np.array([[50.0, 50.0]])
        preferred_distance = 10.0

        model = ObjectVectorCellModel(
            env=env,
            object_positions=object_pos,
            preferred_distance=preferred_distance,
            distance_width=5.0,
            max_rate=20.0,
            baseline_rate=0.1,
        )

        # Position at preferred distance (East of object)
        at_preferred = np.array([[50.0 + preferred_distance, 50.0]])
        # Position too close
        too_close = np.array([[50.0 + 2.0, 50.0]])
        # Position too far
        too_far = np.array([[50.0 + 25.0, 50.0]])

        rate_preferred = model.firing_rate(at_preferred)[0]
        rate_close = model.firing_rate(too_close)[0]
        rate_far = model.firing_rate(too_far)[0]

        # Rate at preferred distance should be highest
        assert rate_preferred > rate_close
        assert rate_preferred > rate_far

        # Rate at preferred should be close to max_rate
        assert rate_preferred > 0.9 * model.max_rate

    def test_gaussian_distance_tuning_shape(self, env):
        """Distance tuning follows Gaussian shape."""
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        object_pos = np.array([[50.0, 50.0]])
        preferred_distance = 15.0
        distance_width = 5.0

        model = ObjectVectorCellModel(
            env=env,
            object_positions=object_pos,
            preferred_distance=preferred_distance,
            distance_width=distance_width,
            max_rate=20.0,
            baseline_rate=0.0,
        )

        # Test at various distances
        distances = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
        positions = np.column_stack(
            [
                50.0 + distances,  # x positions
                np.full_like(distances, 50.0),  # y positions
            ]
        )

        rates = model.firing_rate(positions)

        # Expected Gaussian response
        expected_gaussian = np.exp(
            -0.5 * ((distances - preferred_distance) / distance_width) ** 2
        )
        expected_rates = (
            model.baseline_rate
            + (model.max_rate - model.baseline_rate) * expected_gaussian
        )

        assert_allclose(rates, expected_rates, rtol=0.01)


class TestFiringRateDirectionTuning:
    """Test firing_rate() method with direction tuning."""

    @pytest.fixture
    def env(self):
        """Create a simple 2D environment."""
        samples = np.random.default_rng(42).uniform(0, 100, (500, 2))
        return Environment.from_samples(samples, bin_size=2.0)

    def test_directional_tuning_preferred_direction(self, env):
        """Firing rate is highest when object is in preferred direction."""
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        object_pos = np.array([[50.0, 50.0]])
        preferred_distance = 10.0
        preferred_direction = 0.0  # Object should be ahead (egocentric)

        model = ObjectVectorCellModel(
            env=env,
            object_positions=object_pos,
            preferred_distance=preferred_distance,
            distance_width=5.0,
            preferred_direction=preferred_direction,
            direction_kappa=4.0,
            max_rate=20.0,
            baseline_rate=0.0,
        )

        # Animal at preferred distance, object ahead (requires heading to be toward object)
        # Position: West of object, heading East (toward object)
        position_ahead = np.array([[40.0, 50.0]])  # 10 units West of object
        heading_ahead = np.array([0.0])  # Facing East (toward object)

        # Position: East of object, heading West (away from object)
        position_behind = np.array([[60.0, 50.0]])  # 10 units East of object
        heading_behind = np.array([np.pi])  # Facing West (toward object)

        rate_ahead = model.firing_rate(position_ahead, headings=heading_ahead)[0]
        rate_behind = model.firing_rate(position_behind, headings=heading_behind)[0]

        # Both positions should have similar rates since object is ahead in egocentric frame
        # (both cases: object is in preferred direction relative to heading)
        assert rate_ahead > 0.5 * model.max_rate
        assert rate_behind > 0.5 * model.max_rate

    def test_von_mises_direction_tuning(self, env):
        """Direction tuning follows von Mises shape."""
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        object_pos = np.array([[50.0, 50.0]])
        preferred_direction = 0.0  # Object ahead
        direction_kappa = 4.0  # ~30 degree half-width

        model = ObjectVectorCellModel(
            env=env,
            object_positions=object_pos,
            preferred_distance=10.0,
            distance_width=100.0,  # Wide distance tuning to isolate direction effect
            preferred_direction=preferred_direction,
            direction_kappa=direction_kappa,
            max_rate=20.0,
            baseline_rate=0.0,
        )

        # Animal at fixed position (10 units West of object)
        position = np.array([[40.0, 50.0]])

        # Test headings from various angles
        # When heading=0 (East), object is ahead
        # When heading=pi/2 (North), object is to the right
        # When heading=pi (West), object is behind
        headings = np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])

        rates = []
        for h in headings:
            r = model.firing_rate(position, headings=np.array([h]))[0]
            rates.append(r)

        rates = np.array(rates)

        # Rate should be highest when heading=0 (object ahead)
        assert rates[0] == rates.max()

        # Rate should decrease as heading turns away from object
        assert rates[0] > rates[1] > rates[2]


class TestObjectSelectivity:
    """Test different object selectivity modes."""

    @pytest.fixture
    def env(self):
        """Create a simple 2D environment."""
        samples = np.random.default_rng(42).uniform(0, 100, (500, 2))
        return Environment.from_samples(samples, bin_size=2.0)

    @pytest.fixture
    def two_objects(self):
        """Two objects at different positions."""
        return np.array([[25.0, 50.0], [75.0, 50.0]])

    def test_selectivity_nearest(self, env, two_objects):
        """selectivity='nearest' responds to nearest object only."""
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        model = ObjectVectorCellModel(
            env=env,
            object_positions=two_objects,
            preferred_distance=10.0,
            distance_width=5.0,
            object_selectivity="nearest",
        )

        # Position near first object
        pos_near_first = np.array(
            [[25.0 + 10.0, 50.0]]
        )  # At preferred distance from obj 0
        # Position near second object
        pos_near_second = np.array(
            [[75.0 + 10.0, 50.0]]
        )  # At preferred distance from obj 1

        rate_first = model.firing_rate(pos_near_first)[0]
        rate_second = model.firing_rate(pos_near_second)[0]

        # Both should have high rates (responding to their nearest object)
        assert rate_first > 0.5 * model.max_rate
        assert rate_second > 0.5 * model.max_rate

    def test_selectivity_any(self, env, two_objects):
        """selectivity='any' responds to any object at preferred distance."""
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        model = ObjectVectorCellModel(
            env=env,
            object_positions=two_objects,
            preferred_distance=10.0,
            distance_width=5.0,
            object_selectivity="any",
        )

        # Position at preferred distance from both objects
        # Objects at x=25 and x=75, so x=50 is equidistant (25 units from each)
        # For preferred_distance=10, we need to be 10 units from at least one object
        pos_near_first = np.array([[25.0 + 10.0, 50.0]])

        rate = model.firing_rate(pos_near_first)[0]

        # Should have high rate
        assert rate > 0.5 * model.max_rate

    def test_selectivity_specific(self, env, two_objects):
        """selectivity='specific' responds to specific object index."""
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        # Respond only to object 1 (at x=75)
        model = ObjectVectorCellModel(
            env=env,
            object_positions=two_objects,
            preferred_distance=10.0,
            distance_width=5.0,
            object_selectivity="specific",
            specific_object_index=1,
        )

        # Position at preferred distance from object 0
        pos_near_first = np.array([[25.0 + 10.0, 50.0]])
        # Position at preferred distance from object 1
        pos_near_second = np.array([[75.0 + 10.0, 50.0]])

        rate_first = model.firing_rate(pos_near_first)[0]
        rate_second = model.firing_rate(pos_near_second)[0]

        # Rate near object 1 should be high
        assert rate_second > 0.5 * model.max_rate

        # Rate near object 0 should be low (baseline-ish)
        assert rate_first < rate_second


class TestGeodesicDistance:
    """Test geodesic vs Euclidean distance modes."""

    @pytest.fixture
    def env(self):
        """Create a simple 2D environment."""
        samples = np.random.default_rng(42).uniform(0, 100, (500, 2))
        return Environment.from_samples(samples, bin_size=2.0)

    def test_euclidean_distance_used_by_default(self, env):
        """Euclidean distance is used when distance_metric='euclidean'."""
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        model = ObjectVectorCellModel(
            env=env,
            object_positions=np.array([[50.0, 50.0]]),
            preferred_distance=10.0,
            distance_width=5.0,
            distance_metric="euclidean",
        )

        assert model.distance_metric == "euclidean"

        # Should run without errors
        positions = np.array([[60.0, 50.0], [70.0, 50.0]])
        rates = model.firing_rate(positions)
        assert len(rates) == 2

    def test_geodesic_distance_precomputes_fields(self, env):
        """Geodesic distance mode precomputes distance fields."""
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        model = ObjectVectorCellModel(
            env=env,
            object_positions=np.array([[50.0, 50.0]]),
            preferred_distance=10.0,
            distance_width=5.0,
            distance_metric="geodesic",
        )

        assert model.distance_metric == "geodesic"

        # Model should have precomputed distance fields
        assert hasattr(model, "_distance_fields")
        assert model._distance_fields is not None


class TestGroundTruth:
    """Test ground_truth property."""

    @pytest.fixture
    def env(self):
        """Create a simple 2D environment."""
        samples = np.random.default_rng(42).uniform(0, 100, (500, 2))
        return Environment.from_samples(samples, bin_size=2.0)

    def test_ground_truth_contains_all_parameters(self, env):
        """ground_truth contains all model parameters."""
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        model = ObjectVectorCellModel(
            env=env,
            object_positions=np.array([[50.0, 50.0]]),
            preferred_distance=10.0,
            distance_width=5.0,
            preferred_direction=np.pi / 4,
            direction_kappa=4.0,
            max_rate=25.0,
            baseline_rate=0.5,
            object_selectivity="nearest",
        )

        gt = model.ground_truth

        # Check all expected keys
        assert "object_positions" in gt
        assert "preferred_distance" in gt
        assert "distance_width" in gt
        assert "preferred_direction" in gt
        assert "direction_kappa" in gt
        assert "max_rate" in gt
        assert "baseline_rate" in gt
        assert "object_selectivity" in gt

        # Check values
        assert_allclose(gt["object_positions"], np.array([[50.0, 50.0]]))
        assert gt["preferred_distance"] == 10.0
        assert gt["distance_width"] == 5.0
        assert gt["preferred_direction"] == pytest.approx(np.pi / 4)
        assert gt["direction_kappa"] == 4.0
        assert gt["max_rate"] == 25.0
        assert gt["baseline_rate"] == 0.5
        assert gt["object_selectivity"] == "nearest"

    def test_ground_truth_immutable(self, env):
        """ground_truth returns a copy, not original values."""
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        model = ObjectVectorCellModel(
            env=env,
            object_positions=np.array([[50.0, 50.0]]),
            preferred_distance=10.0,
            distance_width=5.0,
        )

        gt = model.ground_truth

        # Modify the ground truth dict
        gt["preferred_distance"] = 999.0

        # Original model should be unchanged
        assert model.preferred_distance == 10.0


class TestProtocolCompliance:
    """Test compliance with NeuralModel protocol."""

    @pytest.fixture
    def env(self):
        """Create a simple 2D environment."""
        samples = np.random.default_rng(42).uniform(0, 100, (500, 2))
        return Environment.from_samples(samples, bin_size=2.0)

    def test_implements_neural_model_protocol(self, env):
        """ObjectVectorCellModel implements NeuralModel protocol."""
        from neurospatial.simulation.models.base import NeuralModel
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        model = ObjectVectorCellModel(
            env=env,
            object_positions=np.array([[50.0, 50.0]]),
            preferred_distance=10.0,
            distance_width=5.0,
        )

        assert isinstance(model, NeuralModel)

    def test_firing_rate_signature(self, env):
        """firing_rate() has correct signature."""
        from neurospatial.simulation.models.object_vector_cells import (
            ObjectVectorCellModel,
        )

        model = ObjectVectorCellModel(
            env=env,
            object_positions=np.array([[50.0, 50.0]]),
            preferred_distance=10.0,
            distance_width=5.0,
        )

        positions = np.array([[60.0, 50.0]])

        # With positions only
        rates = model.firing_rate(positions)
        assert isinstance(rates, np.ndarray)

        # With positions and times
        times = np.array([0.0])
        rates = model.firing_rate(positions, times)
        assert isinstance(rates, np.ndarray)
