"""Tests for new transform estimation and application features."""

import numpy as np
import pytest

from neurospatial import Environment, apply_transform_to_environment, estimate_transform
from neurospatial.transforms import Affine2D, translate


class TestEstimateTransform:
    """Test estimate_transform function."""

    def test_rigid_transform_identity(self):
        """Test that identical points give identity transform."""
        src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        dst = src.copy()

        T = estimate_transform(src, dst, kind="rigid")
        transformed = T(src)

        assert np.allclose(transformed, dst, atol=1e-10)

    def test_rigid_transform_translation(self):
        """Test rigid transform with pure translation."""
        src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        dst = src + np.array([5, 10])

        T = estimate_transform(src, dst, kind="rigid")
        transformed = T(src)

        assert np.allclose(transformed, dst, atol=1e-10)

    def test_rigid_transform_rotation(self):
        """Test rigid transform with rotation."""
        src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        angle = np.pi / 4
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        dst = src @ R.T

        T = estimate_transform(src, dst, kind="rigid")
        transformed = T(src)

        assert np.allclose(transformed, dst, atol=1e-10)

    def test_similarity_transform_with_scale(self):
        """Test similarity transform includes scaling."""
        src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        scale = 2.5
        dst = src * scale

        T = estimate_transform(src, dst, kind="similarity")
        transformed = T(src)

        assert np.allclose(transformed, dst, atol=1e-10)

    def test_affine_transform(self):
        """Test general affine transform."""
        src = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        # Affine matrix with shear
        A_affine = np.array([[1.5, 0.3], [0.2, 2.0]])
        dst = src @ A_affine.T + np.array([1, 2])

        T = estimate_transform(src, dst, kind="affine")
        transformed = T(src)

        assert np.allclose(transformed, dst, atol=1e-10)

    def test_insufficient_points_rigid(self):
        """Test that insufficient points raise error."""
        src = np.array([[0, 0]], dtype=float)
        dst = np.array([[1, 1]], dtype=float)

        with pytest.raises(ValueError, match="at least 2 point pairs"):
            estimate_transform(src, dst, kind="rigid")

    def test_insufficient_points_affine(self):
        """Test that affine requires at least 3 points."""
        src = np.array([[0, 0], [1, 0]], dtype=float)
        dst = np.array([[1, 1], [2, 1]], dtype=float)

        with pytest.raises(ValueError, match="at least 3 point pairs"):
            estimate_transform(src, dst, kind="affine")

    def test_shape_mismatch_raises_error(self):
        """Test that mismatched shapes raise error."""
        src = np.array([[0, 0], [1, 0]], dtype=float)
        dst = np.array([[1, 1]], dtype=float)

        with pytest.raises(ValueError, match="same shape"):
            estimate_transform(src, dst, kind="rigid")

    def test_invalid_kind_raises_error(self):
        """Test that invalid kind raises error."""
        src = np.array([[0, 0], [1, 0]], dtype=float)
        dst = np.array([[1, 1], [2, 1]], dtype=float)

        with pytest.raises(ValueError, match="Invalid kind"):
            estimate_transform(src, dst, kind="invalid")


class TestApplyTransformToEnvironment:
    """Test apply_transform_to_environment function."""

    @pytest.fixture
    def simple_2d_env(self):
        """Create a simple 2D environment."""
        np.random.seed(42)
        data = np.random.randn(200, 2) * 5
        env = Environment.from_samples(data, bin_size=2.0, name="test")
        env.units = "cm"
        env.frame = "session1"
        return env

    def test_apply_identity_transform(self, simple_2d_env):
        """Test applying identity transform."""
        T = Affine2D(np.eye(3))
        transformed_env = apply_transform_to_environment(simple_2d_env, T)

        assert np.allclose(transformed_env.bin_centers, simple_2d_env.bin_centers)
        assert transformed_env.n_bins == simple_2d_env.n_bins

    def test_apply_translation(self, simple_2d_env):
        """Test applying translation transform."""
        T = translate(10, 20)
        transformed_env = apply_transform_to_environment(simple_2d_env, T)

        # Centers should be translated
        expected = simple_2d_env.bin_centers + np.array([10, 20])
        assert np.allclose(transformed_env.bin_centers, expected)

    def test_transform_preserves_n_bins(self, simple_2d_env):
        """Test that transformation preserves number of bins."""
        T = translate(5, 5)
        transformed_env = apply_transform_to_environment(simple_2d_env, T)

        assert transformed_env.n_bins == simple_2d_env.n_bins

    def test_transform_preserves_connectivity(self, simple_2d_env):
        """Test that transformation preserves connectivity structure."""
        T = translate(5, 5)
        transformed_env = apply_transform_to_environment(simple_2d_env, T)

        assert (
            transformed_env.connectivity.number_of_edges()
            == simple_2d_env.connectivity.number_of_edges()
        )

    def test_transform_updates_edge_distances(self, simple_2d_env):
        """Test that edge distances are recomputed after rotation."""
        # Rotation should preserve distances
        angle = np.pi / 6
        R = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        T = Affine2D(R)

        transformed_env = apply_transform_to_environment(simple_2d_env, T)

        # Get edge distances from original
        orig_dists = [
            simple_2d_env.connectivity.edges[u, v]["distance"]
            for u, v in simple_2d_env.connectivity.edges
        ]
        new_dists = [
            transformed_env.connectivity.edges[u, v]["distance"]
            for u, v in transformed_env.connectivity.edges
        ]

        # Distances should be approximately preserved under rotation
        assert np.allclose(sorted(new_dists), sorted(orig_dists), atol=1e-10)

    def test_transform_copies_units(self, simple_2d_env):
        """Test that units are preserved."""
        T = translate(5, 5)
        transformed_env = apply_transform_to_environment(simple_2d_env, T)

        assert transformed_env.units == simple_2d_env.units

    def test_transform_updates_frame(self, simple_2d_env):
        """Test that frame name is updated."""
        T = translate(5, 5)
        transformed_env = apply_transform_to_environment(simple_2d_env, T)

        assert "transformed" in transformed_env.frame

    def test_transform_with_custom_name(self, simple_2d_env):
        """Test custom name for transformed environment."""
        T = translate(5, 5)
        transformed_env = apply_transform_to_environment(
            simple_2d_env, T, name="aligned"
        )

        assert transformed_env.name == "aligned"

    def test_transform_with_regions(self, simple_2d_env):
        """Test that regions are transformed."""
        simple_2d_env.regions.add("goal", point=np.array([5.0, 5.0]))

        T = translate(10, 20)
        transformed_env = apply_transform_to_environment(simple_2d_env, T)

        assert "goal" in transformed_env.regions
        expected_point = np.array([15.0, 25.0])
        assert np.allclose(transformed_env.regions["goal"].data, expected_point)

    def test_3d_environment_raises_error(self):
        """Test that 3D environments raise error."""
        np.random.seed(42)
        data = np.random.randn(100, 3) * 5
        env_3d = Environment.from_samples(data, bin_size=2.0)

        T = translate(5, 5)
        with pytest.raises(ValueError, match="only supports 2D"):
            apply_transform_to_environment(env_3d, T)

    def test_unfitted_environment_raises_error(self):
        """Test that unfitted environment raises error."""
        # Create a minimal layout that hasn't been built
        from neurospatial.layout.engines.regular_grid import RegularGridLayout

        layout = RegularGridLayout()
        # Don't call build(), so it remains unfitted
        env = Environment(name="unfitted", layout=layout)
        T = translate(5, 5)

        with pytest.raises(RuntimeError, match="must be fitted"):
            apply_transform_to_environment(env, T)
