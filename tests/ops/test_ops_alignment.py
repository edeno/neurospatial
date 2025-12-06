"""Tests for neurospatial.ops.alignment module.

These tests verify the new import path `neurospatial.ops.alignment` works correctly.
The module provides probability mapping and similarity transforms between
spatial environments.
"""

import numpy as np
import pytest

# Test imports from the new location
from neurospatial.ops.alignment import (
    ProbabilityMappingParams,
    apply_similarity_transform,
    get_2d_rotation_matrix,
    map_probabilities,
)


# Minimal mock Environment class for testing
class MockEnvironment:
    """Mock Environment class for testing alignment functions."""

    def __init__(self, bin_centers, n_dims, is_fitted=True):
        self.bin_centers = bin_centers
        self.n_dims = n_dims
        self._is_fitted = is_fitted
        self.n_bins = len(bin_centers)


class TestGet2dRotationMatrix:
    """Tests for get_2d_rotation_matrix function."""

    def test_identity(self):
        """Test 0 degree rotation returns identity matrix."""
        R = get_2d_rotation_matrix(0)
        np.testing.assert_allclose(R, np.eye(2), atol=1e-8)

    def test_90_degrees(self):
        """Test 90 degree counter-clockwise rotation."""
        R = get_2d_rotation_matrix(90)
        expected = np.array([[0, -1], [1, 0]])
        np.testing.assert_allclose(R, expected, atol=1e-8)

    def test_180_degrees(self):
        """Test 180 degree rotation."""
        R = get_2d_rotation_matrix(180)
        expected = np.array([[-1, 0], [0, -1]])
        np.testing.assert_allclose(R, expected, atol=1e-8)

    def test_negative_angle(self):
        """Test negative angle (clockwise rotation)."""
        R = get_2d_rotation_matrix(-90)
        expected = np.array([[0, 1], [-1, 0]])
        np.testing.assert_allclose(R, expected, atol=1e-8)


class TestApplySimilarityTransform:
    """Tests for apply_similarity_transform function."""

    def test_identity_transform(self):
        """Test identity transformation."""
        points = np.array([[1, 2], [3, 4]])
        R = np.eye(2)
        s = 1.0
        t = np.zeros(2)
        transformed = apply_similarity_transform(points, R, s, t)
        np.testing.assert_allclose(transformed, points)

    def test_rotation_scale_translate(self):
        """Test combined rotation, scaling, and translation."""
        points = np.array([[1, 0], [0, 1]])
        R = get_2d_rotation_matrix(90)
        s = 2.0
        t = np.array([1, 1])
        transformed = apply_similarity_transform(points, R, s, t)
        expected = np.array([[1, 3], [-1, 1]])
        np.testing.assert_allclose(transformed, expected, atol=1e-8)

    def test_empty_points(self):
        """Test with empty points array."""
        points = np.empty((0, 2))
        R = np.eye(2)
        s = 1.0
        t = np.zeros(2)
        transformed = apply_similarity_transform(points, R, s, t)
        assert transformed.size == 0

    def test_invalid_shapes(self):
        """Test error on invalid shapes."""
        points = np.array([[1, 2, 3]])  # 3D point
        R = np.eye(2)  # 2D rotation
        s = 1.0
        t = np.zeros(2)  # 2D translation
        with pytest.raises(ValueError):
            apply_similarity_transform(points, R, s, t)

    def test_scale_only(self):
        """Test scaling only."""
        points = np.array([[1, 2], [3, 4]])
        R = np.eye(2)
        t = np.zeros(2)
        transformed = apply_similarity_transform(
            points, R, scale=2.0, translation_vector=t
        )
        expected = np.array([[2, 4], [6, 8]])
        np.testing.assert_allclose(transformed, expected)


class TestMapProbabilities:
    """Tests for map_probabilities function."""

    def test_basic_mapping(self):
        """Test basic probability mapping with identical environments."""
        src_bins = np.array([[0, 0], [1, 0], [0, 1]])
        tgt_bins = np.array([[0, 0], [1, 0], [0, 1]])
        src_probs = np.array([0.2, 0.5, 0.3])
        src_env = MockEnvironment(src_bins, 2)
        tgt_env = MockEnvironment(tgt_bins, 2)
        tgt_probs = map_probabilities(src_env, tgt_env, src_probs)
        np.testing.assert_allclose(tgt_probs, src_probs)

    def test_with_rotation(self):
        """Test mapping with rotation transform."""
        src_bins = np.array([[1, 0], [0, 1]])
        tgt_bins = np.array([[0, 1], [-1, 0]])
        src_probs = np.array([0.7, 0.3])
        src_env = MockEnvironment(src_bins, 2)
        tgt_env = MockEnvironment(tgt_bins, 2)
        R = get_2d_rotation_matrix(90)
        tgt_probs = map_probabilities(
            src_env, tgt_env, src_probs, source_rotation_matrix=R
        )
        np.testing.assert_allclose(tgt_probs, [0.7, 0.3])

    def test_duplicate_mapping(self):
        """Test multiple source bins mapping to same target bin."""
        src_bins = np.array([[0, 0], [0.1, 0]])
        tgt_bins = np.array([[0, 0]])
        src_probs = np.array([0.4, 0.6])
        src_env = MockEnvironment(src_bins, 2)
        tgt_env = MockEnvironment(tgt_bins, 2)
        tgt_probs = map_probabilities(src_env, tgt_env, src_probs)
        np.testing.assert_allclose(tgt_probs, [1.0])

    def test_empty_source(self):
        """Test with empty source environment."""
        src_bins = np.empty((0, 2))
        tgt_bins = np.array([[0, 0], [1, 1]])
        src_probs = np.array([])
        src_env = MockEnvironment(src_bins, 2)
        tgt_env = MockEnvironment(tgt_bins, 2)
        tgt_probs = map_probabilities(src_env, tgt_env, src_probs)
        np.testing.assert_allclose(tgt_probs, [0, 0])

    def test_empty_target(self):
        """Test with empty target environment."""
        src_bins = np.array([[0, 0]])
        tgt_bins = np.empty((0, 2))
        src_probs = np.array([1.0])
        src_env = MockEnvironment(src_bins, 2)
        tgt_env = MockEnvironment(tgt_bins, 2)
        tgt_probs = map_probabilities(src_env, tgt_env, src_probs)
        assert tgt_probs.size == 0

    def test_source_not_fitted(self):
        """Test error when source environment is not fitted."""
        src_bins = np.array([[0, 0]])
        tgt_bins = np.array([[0, 0]])
        src_probs = np.array([1.0])
        src_env = MockEnvironment(src_bins, 2, is_fitted=False)
        tgt_env = MockEnvironment(tgt_bins, 2)
        with pytest.raises(ValueError):
            map_probabilities(src_env, tgt_env, src_probs)

    def test_shape_mismatch(self):
        """Test error when probability shape doesn't match."""
        src_bins = np.array([[0, 0], [1, 1]])
        tgt_bins = np.array([[0, 0], [1, 1]])
        src_probs = np.array([1.0])  # Wrong size
        src_env = MockEnvironment(src_bins, 2)
        tgt_env = MockEnvironment(tgt_bins, 2)
        with pytest.raises(ValueError):
            map_probabilities(src_env, tgt_env, src_probs)

    def test_dimension_mismatch(self):
        """Test error when environments have different dimensions."""
        src_bins = np.array([[0, 0]])
        tgt_bins = np.array([[0, 0, 0]])
        src_probs = np.array([1.0])
        src_env = MockEnvironment(src_bins, 2)
        tgt_env = MockEnvironment(tgt_bins, 3)
        with pytest.raises(ValueError):
            map_probabilities(src_env, tgt_env, src_probs)

    def test_with_scale(self):
        """Test mapping with source scaling."""
        src_bins = np.array([[1, 0], [0, 1]])
        tgt_bins = np.array([[2, 0], [0, 2]])
        src_probs = np.array([0.6, 0.4])
        src_env = MockEnvironment(src_bins, 2)
        tgt_env = MockEnvironment(tgt_bins, 2)
        tgt_probs = map_probabilities(src_env, tgt_env, src_probs, source_scale=2.0)
        np.testing.assert_allclose(tgt_probs, [0.6, 0.4])


class TestProbabilityMappingParams:
    """Tests for ProbabilityMappingParams dataclass."""

    def test_valid_params(self):
        """Test valid parameter creation."""
        src_bins = np.array([[0, 0], [1, 0]])
        tgt_bins = np.array([[0, 0], [1, 0]])
        src_probs = np.array([0.5, 0.5])
        src_env = MockEnvironment(src_bins, 2)
        tgt_env = MockEnvironment(tgt_bins, 2)
        params = ProbabilityMappingParams(src_env, tgt_env, src_probs)
        assert params.n_source_bins == 2
        assert params.n_target_bins == 2

    def test_invalid_mode(self):
        """Test error for invalid mapping mode."""
        src_bins = np.array([[0, 0]])
        tgt_bins = np.array([[0, 0]])
        src_probs = np.array([1.0])
        src_env = MockEnvironment(src_bins, 2)
        tgt_env = MockEnvironment(tgt_bins, 2)
        with pytest.raises(ValueError, match="Unrecognized mode"):
            ProbabilityMappingParams(src_env, tgt_env, src_probs, mode="invalid")
