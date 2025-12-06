"""Tests for 3D transform functionality (AffineND, Affine3D)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from neurospatial import Environment, apply_transform_to_environment, estimate_transform
from neurospatial.ops.transforms import (
    Affine3D,
    AffineND,
    from_rotation_matrix,
    identity_nd,
    scale_3d,
    translate_3d,
)


class TestAffineND:
    """Test AffineND class for N-dimensional affine transforms."""

    def test_affine_nd_creation(self):
        """Test creating AffineND from matrix."""
        A = np.eye(4)  # 3D identity
        transform = AffineND(A)
        assert transform.n_dims == 3
        assert np.allclose(transform.A, A)

    def test_affine_nd_n_dims_property(self):
        """Test n_dims property for different dimensions."""
        # 2D
        transform_2d = AffineND(np.eye(3))
        assert transform_2d.n_dims == 2

        # 3D
        transform_3d = AffineND(np.eye(4))
        assert transform_3d.n_dims == 3

        # 4D
        transform_4d = AffineND(np.eye(5))
        assert transform_4d.n_dims == 4

    def test_affine_nd_call_3d(self):
        """Test applying 3D affine transform to points."""
        # Translation
        A = np.eye(4)
        A[:3, 3] = [10, 20, 30]
        transform = AffineND(A)

        points = np.array([[0, 0, 0], [1, 1, 1]])
        transformed = transform(points)

        expected = np.array([[10, 20, 30], [11, 21, 31]])
        assert np.allclose(transformed, expected)

    def test_affine_nd_dimension_validation(self):
        """Test that dimension mismatch raises error."""
        transform_3d = AffineND(np.eye(4))
        points_2d = np.array([[0, 0], [1, 1]])

        with pytest.raises(ValueError, match="3D but points have 2 dimensions"):
            transform_3d(points_2d)

    def test_affine_nd_inverse(self):
        """Test inverse transformation."""
        # 3D translation
        A = np.eye(4)
        A[:3, 3] = [10, 20, 30]
        transform = AffineND(A)

        points = np.array([[10, 20, 30]])
        inv = transform.inverse()
        result = inv(points)

        assert np.allclose(result, [[0, 0, 0]])

    def test_affine_nd_compose(self):
        """Test composing transformations."""
        # Translation
        A1 = np.eye(4)
        A1[:3, 3] = [10, 0, 0]
        t1 = AffineND(A1)

        # Scaling
        A2 = np.diag([2.0, 2.0, 2.0, 1.0])
        t2 = AffineND(A2)

        # Compose
        combined = t1.compose(t2)

        points = np.array([[1, 1, 1]])
        result = combined(points)

        # t2 first (scale by 2), then t1 (translate by 10 in x)
        expected = [[12, 2, 2]]
        assert np.allclose(result, expected)

    def test_affine_nd_matmul_operator(self):
        """Test @ operator for composition."""
        t1 = translate_3d(10, 0, 0)
        t2 = scale_3d(2.0)

        combined = t1 @ t2

        points = np.array([[1, 1, 1]])
        result = combined(points)

        expected = [[12, 2, 2]]
        assert np.allclose(result, expected)

    def test_affine_nd_invalid_matrix_shape(self):
        """Test that non-square matrix raises error."""
        A = np.ones((3, 4))  # Not square
        with pytest.raises(ValueError, match="must be square"):
            AffineND(A)


class Test3DFactoryFunctions:
    """Test 3D transform factory functions."""

    def test_translate_3d(self):
        """Test 3D translation."""
        transform = translate_3d(10, 20, 30)
        assert transform.n_dims == 3

        points = np.array([[0, 0, 0], [1, 1, 1]])
        result = transform(points)

        expected = np.array([[10, 20, 30], [11, 21, 31]])
        assert np.allclose(result, expected)

    def test_translate_3d_defaults(self):
        """Test 3D translation with default parameters."""
        transform = translate_3d()  # All zeros
        points = np.array([[1, 2, 3]])
        result = transform(points)

        assert np.allclose(result, points)

    def test_scale_3d_uniform(self):
        """Test uniform 3D scaling."""
        transform = scale_3d(2.0)
        points = np.array([[1, 2, 3]])
        result = transform(points)

        expected = [[2, 4, 6]]
        assert np.allclose(result, expected)

    def test_scale_3d_anisotropic(self):
        """Test anisotropic 3D scaling."""
        transform = scale_3d(sx=2.0, sy=0.5, sz=3.0)
        points = np.array([[1, 2, 3]])
        result = transform(points)

        expected = [[2, 1, 9]]
        assert np.allclose(result, expected)

    def test_from_rotation_matrix_3d(self):
        """Test creating transform from 3D rotation matrix."""
        # 90-degree rotation around z-axis
        R = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        transform = from_rotation_matrix(R)

        points = np.array([[1, 0, 0], [0, 1, 0]])
        result = transform(points)

        expected = [[0, 1, 0], [-1, 0, 0]]
        assert np.allclose(result, expected, atol=1e-10)

    def test_from_rotation_matrix_with_translation(self):
        """Test rotation matrix with translation."""
        R = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        translation = np.array([10, 20, 30])

        transform = from_rotation_matrix(R, translation=translation)

        points = np.array([[1, 0, 0]])
        result = transform(points)

        # Rotate [1, 0, 0] -> [0, 1, 0], then translate
        expected = [[10, 21, 30]]
        assert np.allclose(result, expected, atol=1e-10)

    def test_from_rotation_matrix_2d(self):
        """Test that from_rotation_matrix works for 2D."""
        angle = np.pi / 4
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        transform = from_rotation_matrix(R)
        assert transform.n_dims == 2

        points = np.array([[1, 0]])
        result = transform(points)

        expected = [[np.cos(angle), np.sin(angle)]]
        assert np.allclose(result, expected)

    def test_from_rotation_matrix_invalid_shape(self):
        """Test that non-square matrix raises error."""
        R = np.ones((3, 4))
        with pytest.raises(ValueError, match="must be square"):
            from_rotation_matrix(R)

    def test_from_rotation_matrix_translation_shape_mismatch(self):
        """Test that translation shape mismatch raises error."""
        R = np.eye(3)
        translation = np.array([1, 2])  # Wrong dimension

        with pytest.raises(ValueError, match="must have shape"):
            from_rotation_matrix(R, translation=translation)

    def test_identity_nd_3d(self):
        """Test 3D identity transform."""
        transform = identity_nd(n_dims=3)
        assert transform.n_dims == 3

        points = np.array([[1, 2, 3], [4, 5, 6]])
        result = transform(points)

        assert np.allclose(result, points)

    def test_identity_nd_various_dimensions(self):
        """Test identity for various dimensions."""
        rng = np.random.default_rng(42)
        for n_dims in [2, 3, 4, 5]:
            transform = identity_nd(n_dims=n_dims)
            assert transform.n_dims == n_dims

            points = rng.standard_normal((10, n_dims))
            result = transform(points)
            assert np.allclose(result, points)

    def test_affine3d_type_alias(self):
        """Test that Affine3D is an alias for AffineND."""
        assert Affine3D is AffineND


class TestEstimateTransform3D:
    """Test estimate_transform function with 3D points."""

    def test_rigid_transform_3d_identity(self):
        """Test that identical 3D points give identity transform."""
        src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        dst = src.copy()

        transform = estimate_transform(src, dst, kind="rigid")
        assert transform.n_dims == 3

        transformed = transform(src)
        assert np.allclose(transformed, dst, atol=1e-10)

    def test_rigid_transform_3d_translation(self):
        """Test 3D rigid transform with pure translation."""
        src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        dst = src + np.array([10, 20, 30])

        transform = estimate_transform(src, dst, kind="rigid")
        transformed = transform(src)

        assert np.allclose(transformed, dst, atol=1e-10)

    def test_rigid_transform_3d_rotation(self):
        """Test 3D rigid transform with rotation."""
        src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)

        # 45-degree rotation around z-axis
        R = Rotation.from_euler("z", 45, degrees=True).as_matrix()
        dst = src @ R.T

        transform = estimate_transform(src, dst, kind="rigid")
        transformed = transform(src)

        assert np.allclose(transformed, dst, atol=1e-8)

    def test_rigid_transform_3d_rotation_and_translation(self):
        """Test 3D rigid transform with rotation and translation."""
        src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)

        # Rotation + translation
        R = Rotation.from_euler("xyz", [45, 30, 60], degrees=True).as_matrix()
        translation = np.array([10, 20, 30])
        dst = src @ R.T + translation

        transform = estimate_transform(src, dst, kind="rigid")
        transformed = transform(src)

        assert np.allclose(transformed, dst, atol=1e-8)

    def test_similarity_transform_3d_with_scale(self):
        """Test 3D similarity transform with scaling."""
        src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        scale = 2.5
        dst = src * scale

        transform = estimate_transform(src, dst, kind="similarity")
        transformed = transform(src)

        assert np.allclose(transformed, dst, atol=1e-8)

    def test_similarity_transform_3d_rotation_scale_translation(self):
        """Test 3D similarity transform with rotation, scale, and translation."""
        src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)

        # Scale + rotation + translation
        R = Rotation.from_euler("z", 45, degrees=True).as_matrix()
        scale = 2.0
        translation = np.array([10, 20, 30])
        dst = (src @ R.T) * scale + translation

        transform = estimate_transform(src, dst, kind="similarity")
        transformed = transform(src)

        assert np.allclose(transformed, dst, atol=1e-8)

    def test_affine_transform_3d(self):
        """Test general 3D affine transform."""
        src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)

        # Affine matrix with shear
        A_affine = np.array([[1.5, 0.3, 0.1], [0.2, 2.0, 0.4], [0.1, 0.2, 1.8]])
        dst = src @ A_affine.T + np.array([1, 2, 3])

        transform = estimate_transform(src, dst, kind="affine")
        transformed = transform(src)

        assert np.allclose(transformed, dst, atol=1e-8)

    def test_insufficient_points_3d_affine(self):
        """Test that 3D affine requires at least 4 points."""
        src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        dst = src + 1

        with pytest.raises(ValueError, match="at least 4 point pairs"):
            estimate_transform(src, dst, kind="affine")


class TestApplyTransformToEnvironment3D:
    """Test apply_transform_to_environment with 3D environments.

    Uses session-scoped `simple_3d_env` fixture from conftest.py for most tests.
    Tests that modify environment state create their own inline environments.
    """

    def test_apply_3d_identity_transform(self, simple_3d_env):
        """Test applying 3D identity transform."""
        transform = identity_nd(n_dims=3)
        transformed_env = apply_transform_to_environment(simple_3d_env, transform)

        assert np.allclose(transformed_env.bin_centers, simple_3d_env.bin_centers)
        assert transformed_env.n_bins == simple_3d_env.n_bins

    def test_apply_3d_translation(self, simple_3d_env):
        """Test applying 3D translation."""
        transform = translate_3d(10, 20, 30)
        transformed_env = apply_transform_to_environment(simple_3d_env, transform)

        # Centers should be translated
        expected = simple_3d_env.bin_centers + np.array([10, 20, 30])
        assert np.allclose(transformed_env.bin_centers, expected)

    def test_apply_3d_scaling(self, simple_3d_env):
        """Test applying 3D scaling."""
        transform = scale_3d(2.0)
        transformed_env = apply_transform_to_environment(simple_3d_env, transform)

        # Centers should be scaled
        expected = simple_3d_env.bin_centers * 2.0
        assert np.allclose(transformed_env.bin_centers, expected)

    def test_apply_3d_rotation(self, simple_3d_env):
        """Test applying 3D rotation."""
        R = Rotation.from_euler("z", 45, degrees=True).as_matrix()
        transform = from_rotation_matrix(R)
        transformed_env = apply_transform_to_environment(simple_3d_env, transform)

        # Centers should be rotated
        expected = simple_3d_env.bin_centers @ R.T
        assert np.allclose(transformed_env.bin_centers, expected, atol=1e-10)

    def test_3d_transform_preserves_n_bins(self, simple_3d_env):
        """Test that 3D transformation preserves number of bins."""
        transform = translate_3d(5, 5, 5)
        transformed_env = apply_transform_to_environment(simple_3d_env, transform)

        assert transformed_env.n_bins == simple_3d_env.n_bins

    def test_3d_transform_preserves_connectivity(self, simple_3d_env):
        """Test that 3D transformation preserves connectivity structure."""
        transform = translate_3d(5, 5, 5)
        transformed_env = apply_transform_to_environment(simple_3d_env, transform)

        assert (
            transformed_env.connectivity.number_of_edges()
            == simple_3d_env.connectivity.number_of_edges()
        )

    def test_3d_transform_updates_edge_distances(self, simple_3d_env):
        """Test that edge distances are recomputed correctly after rotation."""
        # Rotation should preserve distances
        R = Rotation.from_euler("xyz", [30, 45, 60], degrees=True).as_matrix()
        transform = from_rotation_matrix(R)

        transformed_env = apply_transform_to_environment(simple_3d_env, transform)

        # Get edge distances from original
        orig_dists = [
            simple_3d_env.connectivity.edges[u, v]["distance"]
            for u, v in simple_3d_env.connectivity.edges
        ]
        new_dists = [
            transformed_env.connectivity.edges[u, v]["distance"]
            for u, v in transformed_env.connectivity.edges
        ]

        # Distances should be approximately preserved under rotation
        assert np.allclose(sorted(new_dists), sorted(orig_dists), atol=1e-8)

    def test_3d_transform_no_angle_2d_attribute(self, simple_3d_env):
        """Test that 3D transforms don't have angle_2d edge attribute."""
        transform = translate_3d(5, 5, 5)
        transformed_env = apply_transform_to_environment(simple_3d_env, transform)

        # Check that angle_2d is not in any edge
        for u, v in transformed_env.connectivity.edges:
            # Should not have angle_2d for 3D environment
            # The implementation only updates angle_2d for 2D environments
            assert "angle_2d" not in transformed_env.connectivity.edges[u, v]

    def test_3d_transform_copies_units(self, simple_3d_env):
        """Test that units are preserved."""
        transform = translate_3d(5, 5, 5)
        transformed_env = apply_transform_to_environment(simple_3d_env, transform)

        assert transformed_env.units == simple_3d_env.units

    def test_3d_transform_updates_frame(self, simple_3d_env):
        """Test that frame name is updated."""
        transform = translate_3d(5, 5, 5)
        transformed_env = apply_transform_to_environment(simple_3d_env, transform)

        assert "transformed" in transformed_env.frame

    def test_3d_transform_with_custom_name(self, simple_3d_env):
        """Test custom name for 3D transformed environment."""
        transform = translate_3d(5, 5, 5)
        transformed_env = apply_transform_to_environment(
            simple_3d_env, transform, name="aligned_3d"
        )

        assert transformed_env.name == "aligned_3d"

    def test_3d_transform_with_regions(self):
        """Test that 3D regions are transformed.

        Creates inline environment since test modifies state by adding regions.
        """
        # Create a fresh 3D environment (deterministic 5x5x5 grid)
        # Bin edges at [0, 2, 4, 6, 8, 10] → 5 bins per dimension
        mask = np.ones((5, 5, 5), dtype=bool)
        edges = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        env = Environment.from_mask(
            active_mask=mask,
            grid_edges=(edges, edges, edges),
            name="test_3d_regions",
        )

        # Add region (modifies environment)
        env.regions.add("goal", point=np.array([5.0, 5.0, 5.0]))

        transform = translate_3d(10, 20, 30)
        transformed_env = apply_transform_to_environment(env, transform)

        assert "goal" in transformed_env.regions
        expected_point = np.array([15.0, 25.0, 35.0])
        assert np.allclose(transformed_env.regions["goal"].data, expected_point)

    def test_3d_dimension_ranges_transformed(self, simple_3d_env):
        """Test that dimension_ranges are correctly transformed."""
        transform = translate_3d(100, 200, 300)
        transformed_env = apply_transform_to_environment(simple_3d_env, transform)

        # Original ranges
        orig_ranges = simple_3d_env.dimension_ranges
        new_ranges = transformed_env.dimension_ranges

        # Each dimension should be shifted by the translation
        assert len(new_ranges) == 3
        for dim, (orig_lo, orig_hi) in enumerate(orig_ranges):
            translation = [100, 200, 300][dim]
            new_lo, new_hi = new_ranges[dim]
            assert np.allclose(new_lo, orig_lo + translation, atol=1e-8)
            assert np.allclose(new_hi, orig_hi + translation, atol=1e-8)


class TestScipyIntegration:
    """Test integration with scipy.spatial.transform.Rotation."""

    def test_rotation_from_euler(self):
        """Test creating transform from Euler angles."""
        rng = np.random.default_rng(42)
        # XYZ Euler angles
        R = Rotation.from_euler("xyz", [30, 45, 60], degrees=True).as_matrix()
        transform = from_rotation_matrix(R)

        # Test that it's a valid rotation (preserves distances)
        points = rng.standard_normal((10, 3))
        transformed = transform(points)

        orig_dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
        new_dists = np.linalg.norm(np.diff(transformed, axis=0), axis=1)

        assert np.allclose(orig_dists, new_dists)

    def test_rotation_from_rotvec(self):
        """Test creating transform from rotation vector."""
        # Rotation vector (axis-angle representation)
        rotvec = np.array([0.5, 0.3, 0.2])
        R = Rotation.from_rotvec(rotvec).as_matrix()
        transform = from_rotation_matrix(R)

        points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        transformed = transform(points)

        # Check that rotation preserves origin distance
        orig_norms = np.linalg.norm(points, axis=1)
        new_norms = np.linalg.norm(transformed, axis=1)

        assert np.allclose(orig_norms, new_norms)

    def test_rotation_from_quaternion(self):
        """Test creating transform from quaternion."""
        # Quaternion (w, x, y, z)
        quat = np.array([0.9238795, 0, 0, 0.3826834])  # ~45 degrees around z
        R = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        transform = from_rotation_matrix(R)

        points = np.array([[1, 0, 0]])
        transformed = transform(points)

        # Should rotate roughly 45 degrees in xy-plane
        expected_angle = 2 * np.arccos(quat[0])
        actual_angle = np.arctan2(transformed[0, 1], transformed[0, 0])

        assert np.allclose(actual_angle, expected_angle, atol=1e-6)

    def test_full_3d_alignment_workflow(self):
        """Test complete 3D alignment workflow with scipy.Rotation."""
        # Create source and target 3D environments
        rng = np.random.default_rng(42)
        positions = rng.standard_normal((500, 3)) * 15
        env1 = Environment.from_samples(positions, bin_size=5.0)

        # Apply known transformation
        R = Rotation.from_euler("xyz", [20, 30, 40], degrees=True).as_matrix()
        translation = np.array([50, 60, 70])
        scale = 1.2

        # Use actual bin centers as landmarks
        src_landmarks = env1.bin_centers[:4]

        # Transform landmarks with known transform
        dst_landmarks = (src_landmarks @ R.T) * scale + translation

        # Estimate transformation
        transform = estimate_transform(src_landmarks, dst_landmarks, kind="similarity")

        # Check that estimated transform recovers landmarks
        aligned_landmarks = transform(src_landmarks)
        assert np.allclose(aligned_landmarks, dst_landmarks, atol=1e-6)

        # Check that applied transform works on full environment
        aligned_centers = transform(env1.bin_centers)
        expected_centers = (env1.bin_centers @ R.T) * scale + translation
        assert np.allclose(aligned_centers, expected_centers, atol=1e-6)
