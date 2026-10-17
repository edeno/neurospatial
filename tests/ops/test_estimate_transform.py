"""Tests for new transform estimation and application features."""

import numpy as np
import pytest

from neurospatial import Environment, Regions
from neurospatial.ops.transforms import (
    Affine2D,
    AffineND,
    estimate_transform,
    translate,
)


class TestEstimateTransform:
    """Test estimate_transform function."""

    def test_rigid_transform_identity(self):
        """Test that identical points give identity transform."""
        src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        dst = src.copy()

        transform = estimate_transform(src, dst, kind="rigid")
        transformed = transform(src)

        assert np.allclose(transformed, dst, atol=1e-10)

    def test_rigid_transform_translation(self):
        """Test rigid transform with pure translation."""
        src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        dst = src + np.array([5, 10])

        transform = estimate_transform(src, dst, kind="rigid")
        transformed = transform(src)

        assert np.allclose(transformed, dst, atol=1e-10)

    def test_rigid_transform_rotation(self):
        """Test rigid transform with rotation."""
        src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        angle = np.pi / 4
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        dst = src @ R.T

        transform = estimate_transform(src, dst, kind="rigid")
        transformed = transform(src)

        assert np.allclose(transformed, dst, atol=1e-10)

    def test_similarity_transform_with_scale(self):
        """Test similarity transform includes scaling."""
        src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        scale = 2.5
        dst = src * scale

        transform = estimate_transform(src, dst, kind="similarity")
        transformed = transform(src)

        assert np.allclose(transformed, dst, atol=1e-10)

    def test_affine_transform(self):
        """Test general affine transform."""
        src = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        # Affine matrix with shear
        A_affine = np.array([[1.5, 0.3], [0.2, 2.0]])
        dst = src @ A_affine.T + np.array([1, 2])

        transform = estimate_transform(src, dst, kind="affine")
        transformed = transform(src)

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


class TestEstimateTransformKabsch:
    """Rigid/similarity fits must return a proper rotation (det = +1)."""

    @staticmethod
    def _linear_block(transform, n_dims):
        return transform.A[:n_dims, :n_dims]

    def test_estimate_transform_reflected_points_proper_rotation(self):
        """A reflected dst recovers the OPTIMAL proper rotation, not a reflection.

        ``src``->``dst`` here is itself improper (it involves a reflection), so
        no proper rotation can match ``dst`` exactly. The naive column-flip on
        the finished orthogonal matrix returned a proper-but-NON-OPTIMAL fit
        with a larger residual; the Kabsch SVD sign-correction returns the
        minimal-residual proper rotation (det = +1).
        """
        rng = np.random.default_rng(7)
        src = rng.normal(size=(8, 2))

        # Reflect across the x-axis (negate x), then rotate slightly.
        reflected = src * np.array([-1.0, 1.0])
        angle = 0.3
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        dst = reflected @ R.T + np.array([2.0, -1.0])

        transform = estimate_transform(src, dst, kind="rigid")
        linear = self._linear_block(transform, 2)

        # Proper rotation.
        assert np.linalg.det(linear) == pytest.approx(1.0, abs=1e-8)

        # Independently compute the optimal proper-rotation residual via the
        # closed-form Kabsch solution and confirm the returned transform attains
        # it. The old column-flip path produced a strictly larger residual.
        src_c = src - src.mean(axis=0)
        dst_c = dst - dst.mean(axis=0)
        h_mat = src_c.T @ dst_c
        u_mat, _s, vt_mat = np.linalg.svd(h_mat)
        v_mat = vt_mat.T
        d_sign = np.sign(np.linalg.det(v_mat @ u_mat.T))
        d_diag = np.eye(2)
        d_diag[-1, -1] = d_sign
        r_opt = v_mat @ d_diag @ u_mat.T
        optimal_resid = np.linalg.norm(src_c @ r_opt.T - dst_c)

        achieved_resid = np.linalg.norm(transform(src) - dst)
        assert achieved_resid == pytest.approx(optimal_resid, abs=1e-8)

    def test_estimate_transform_coplanar_3d_proper_rotation(self):
        """3D coplanar landmarks under a known rotation recover det(R) ~= +1."""
        rng = np.random.default_rng(11)
        # Coplanar points: all z = 0.
        src = np.column_stack([rng.normal(size=6), rng.normal(size=6), np.zeros(6)])

        angle = 0.5
        R_true = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        t_true = np.array([1.0, -2.0, 0.5])
        dst = src @ R_true.T + t_true

        transform = estimate_transform(src, dst, kind="rigid")
        linear = self._linear_block(transform, 3)

        assert np.linalg.det(linear) == pytest.approx(1.0, abs=1e-8)
        assert np.allclose(transform(src), dst, atol=1e-8)

    def test_estimate_transform_pure_rotation_unchanged(self):
        """Proper-rotation cases still recover the planted transform exactly."""
        # 2D 45-degree rotation.
        src_2d = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        angle = np.pi / 4
        R2 = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        dst_2d = src_2d @ R2.T
        transform_2d = estimate_transform(src_2d, dst_2d, kind="rigid")
        assert np.allclose(transform_2d(src_2d), dst_2d, atol=1e-10)
        assert np.linalg.det(transform_2d.A[:2, :2]) == pytest.approx(1.0, abs=1e-10)

        # 3D z-rotation.
        rng = np.random.default_rng(3)
        src_3d = rng.normal(size=(6, 3))
        z_angle = 0.7
        R3 = np.array(
            [
                [np.cos(z_angle), -np.sin(z_angle), 0.0],
                [np.sin(z_angle), np.cos(z_angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        dst_3d = src_3d @ R3.T
        transform_3d = estimate_transform(src_3d, dst_3d, kind="rigid")
        assert np.allclose(transform_3d(src_3d), dst_3d, atol=1e-10)
        assert np.linalg.det(transform_3d.A[:3, :3]) == pytest.approx(1.0, abs=1e-10)


class TestApplyTransformToEnvironment:
    """Test apply_transform function."""

    @pytest.fixture
    def simple_2d_env(self):
        """Create a simple 2D environment."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((200, 2)) * 5
        env = Environment.from_samples(data, bin_size=2.0, name="test")
        env = env.with_metadata(units="cm")
        env = env.with_metadata(frame="session1")
        return env

    def test_apply_identity_transform(self, simple_2d_env):
        """Test applying identity transform."""
        transform = Affine2D(np.eye(3))
        transformed_env = simple_2d_env.apply_transform(transform)

        assert np.allclose(transformed_env.bin_centers, simple_2d_env.bin_centers)
        assert transformed_env.n_bins == simple_2d_env.n_bins

    def test_apply_translation(self, simple_2d_env):
        """Test applying translation transform."""
        transform = translate(10, 20)
        transformed_env = simple_2d_env.apply_transform(transform)

        # Centers should be translated
        expected = simple_2d_env.bin_centers + np.array([10, 20])
        assert np.allclose(transformed_env.bin_centers, expected)

    @pytest.mark.parametrize(
        "transform",
        [
            translate(10, 20),
            Affine2D(np.array([[0.0, -2.0, 10.0], [0.5, 0.0, 20.0], [0, 0, 1]])),
            Affine2D(np.array([[1.0, 2.0, 0.0], [0.0, -1.0, 0.0], [0, 0, 1]])),
        ],
        ids=["translation", "rotation_and_scaling", "shear_and_reflection"],
    )
    def test_transform_preserves_containment_and_occupancy(self, transform):
        """Outside points and inactive cells stay excluded after alignment."""
        mask = np.ones((3, 3), dtype=bool)
        mask[1, 1] = False
        env = Environment.from_grid_mask(mask, grid_edges=(np.arange(4.0),) * 2)
        points = np.array([[0.1, 0.9], [1.5, 1.5], [8.0, 8.0], [2.5, 2.5]])
        expected = env.bin_at(points)
        assert expected.tolist() == [0, -1, -1, 7]

        transformed = env.apply_transform(transform)
        mapped = transform(points)
        np.testing.assert_array_equal(transformed.bin_at(mapped), expected)
        np.testing.assert_array_equal(transformed.contains(mapped), expected >= 0)

        times = np.arange(len(points)) / 10
        np.testing.assert_allclose(
            transformed.occupancy(times, mapped), env.occupancy(times, points)
        )
        # Applying another transform must preserve the same containment rules.
        again = transformed.apply_transform(translate(-3, 4))
        np.testing.assert_array_equal(
            again.bin_at(mapped + np.array([-3, 4])), expected
        )

    @pytest.mark.parametrize("n_dims", [1, 3])
    def test_transform_preserves_nd_containment(self, n_dims):
        """Inverse mapping also preserves inactive bins for AffineND layouts."""
        mask = np.ones((2,) * n_dims, dtype=bool)
        mask[(1,) * n_dims] = False
        env = Environment.from_grid_mask(mask, grid_edges=(np.arange(3.0),) * n_dims)
        matrix = np.eye(n_dims + 1)
        matrix[:n_dims, :n_dims] *= 2
        matrix[:n_dims, -1] = 10
        transform = AffineND(matrix)
        points = np.array([[0.5] * n_dims, [1.5] * n_dims, [8.0] * n_dims])

        transformed = env.apply_transform(transform)
        np.testing.assert_array_equal(
            transformed.bin_at(transform(points)), [0, -1, -1]
        )

    def test_singular_environment_transform_raises(self, simple_2d_env):
        """Collapsed cells cannot support an inverse containment query."""
        with pytest.raises(ValueError, match="invertible"):
            simple_2d_env.apply_transform(Affine2D(np.diag([0.0, 1.0, 1.0])))

    def test_transform_preserves_polygon_holes(self):
        """Excluded regions stay excluded under both translation and reflection."""
        from shapely.geometry import Point, Polygon

        env = Environment.from_grid_mask(
            np.ones((4, 4), dtype=bool), grid_edges=(np.arange(5.0),) * 2
        )
        polygon = Polygon(
            [(0, 0), (4, 0), (4, 4), (0, 4)],
            holes=[
                [(1, 1), (2, 1), (2, 2), (1, 2)],
                [(3, 1), (3.5, 1), (3.5, 2), (3, 2)],
            ],
        )
        region_builder = Regions(env.regions.values())
        region_builder.add("arena", polygon=polygon)
        env = env.with_regions(region_builder)
        transform = Affine2D(np.array([[-2.0, 0, 10], [0, 3.0, 20], [0, 0, 1]]))
        transformed = env.apply_transform(transform)
        region = transformed.regions["arena"].data

        assert len(region.interiors) == 2
        assert region.area == pytest.approx(polygon.area * 6)
        for point in ([0.5, 0.5], [1.5, 1.5], [3.25, 1.5], [8.0, 8.0]):
            assert region.contains(
                Point(transform(np.array(point)))
            ) == polygon.contains(Point(point))
        assert env.regions["arena"].data.equals(polygon)

    def test_transform_preserves_n_bins(self, simple_2d_env):
        """Test that transformation preserves number of bins."""
        transform = translate(5, 5)
        transformed_env = simple_2d_env.apply_transform(transform)

        assert transformed_env.n_bins == simple_2d_env.n_bins

    def test_transform_preserves_connectivity(self, simple_2d_env):
        """Test that transformation preserves connectivity structure."""
        transform = translate(5, 5)
        transformed_env = simple_2d_env.apply_transform(transform)

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
        transform = Affine2D(R)

        transformed_env = simple_2d_env.apply_transform(transform)

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
        transform = translate(5, 5)
        transformed_env = simple_2d_env.apply_transform(transform)

        assert transformed_env.units == simple_2d_env.units

    def test_transform_frame_comes_from_the_caller(self, simple_2d_env):
        """The transformed frame is unlabeled unless the caller names it."""
        transform = translate(5, 5)
        unlabeled = simple_2d_env.apply_transform(transform)
        labeled = simple_2d_env.apply_transform(transform, frame="session2")

        assert unlabeled.frame is None
        assert labeled.frame == "session2"

    def test_transform_with_custom_name(self, simple_2d_env):
        """Test custom name for transformed environment."""
        transform = translate(5, 5)
        transformed_env = simple_2d_env.apply_transform(transform, name="aligned")

        assert transformed_env.name == "aligned"

    def test_transform_with_regions(self, simple_2d_env):
        """Test that regions are transformed."""
        region_builder = Regions(simple_2d_env.regions.values())
        region_builder.add("goal", point=np.array([5.0, 5.0]))
        simple_2d_env = simple_2d_env.with_regions(region_builder)

        transform = translate(10, 20)
        transformed_env = simple_2d_env.apply_transform(transform)

        assert "goal" in transformed_env.regions
        expected_point = np.array([15.0, 25.0])
        assert np.allclose(transformed_env.regions["goal"].data, expected_point)

    def test_3d_environment_dimension_mismatch_raises_error(self):
        """Test that 3D environment with 2D transform raises error."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 3)) * 5
        env_3d = Environment.from_samples(data, bin_size=2.0)

        # 2D transform on 3D environment should raise error
        transform = translate(5, 5)
        with pytest.raises(ValueError, match="-D transform for"):
            env_3d.apply_transform(transform)
