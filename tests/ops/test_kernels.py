"""Tests for the low-level diffusion-kernel primitive ``compute_diffusion_kernels``.

These exercise the graph-level seam: given ``"A"`` (shared-face measure) and
``"distance"`` on each edge plus a node-ordered ``volumes`` array, it assembles
the finite-volume weight matrix ``W[i, j] = A / d`` and returns the requested
normalized mode. The finite-volume operator + per-geometry face measures + the
grid-independence / sigma-recovery guarantees are covered in ``test_diffusion.py``.
"""

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.ops.smoothing import compute_diffusion_kernels


def _line_graph(n, *, distance=1.0, A=1.0):
    """1D chain of ``n`` nodes with uniform ``distance`` and face measure ``A``."""
    graph = nx.Graph()
    for i in range(n):
        graph.add_node(i, pos=(float(i),))
    for i in range(n - 1):
        graph.add_edge(i, i + 1, distance=distance, A=A)
    return graph


def _grid_graph_with_A(side=3):
    """``side x side`` orthogonal grid with unit distance and face measure."""
    graph = nx.grid_2d_graph(side, side)
    mapping = {node: i for i, node in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, mapping)
    for i in range(side * side):
        row, col = divmod(i, side)
        graph.nodes[i]["pos"] = (float(col), float(row))
    for u, v in graph.edges():
        pos_u = np.array(graph.nodes[u]["pos"])
        pos_v = np.array(graph.nodes[v]["pos"])
        graph.edges[u, v]["distance"] = float(np.linalg.norm(pos_v - pos_u))
        graph.edges[u, v]["A"] = 1.0
    return graph


class TestComputeDiffusionKernels:
    """Structural / invariant tests for the primitive on the new signature."""

    def test_kernel_shape(self):
        """Kernel has shape (n_bins, n_bins) and dtype float64."""
        graph = _line_graph(3)
        kernel = compute_diffusion_kernels(
            graph, volumes=np.ones(3), sigma=1.0, mode="transition"
        )
        assert kernel.shape == (3, 3)
        assert kernel.dtype == np.float64

    def test_kernel_symmetry_uniform_grid(self):
        """On a uniform grid (uniform M) the transition kernel is symmetric."""
        graph = _grid_graph_with_A(3)
        kernel = compute_diffusion_kernels(
            graph, volumes=np.ones(9), sigma=1.0, mode="transition"
        )
        assert np.allclose(kernel, kernel.T, atol=1e-10)

    def test_transition_column_stochastic(self):
        """transition mode: each column sums to 1."""
        graph = _line_graph(4)
        kernel = compute_diffusion_kernels(
            graph, volumes=np.ones(4), sigma=0.5, mode="transition"
        )
        np.testing.assert_allclose(kernel.sum(axis=0), 1.0, atol=1e-10)

    def test_density_weighted_columns_integrate_to_one(self):
        """density mode: sum_i M_i K[i, j] = 1 with non-uniform volumes."""
        graph = nx.Graph()
        graph.add_node(0, pos=(0.0,))
        graph.add_node(1, pos=(1.0,))
        graph.add_node(2, pos=(3.0,))
        graph.add_edge(0, 1, distance=1.0, A=1.0)
        graph.add_edge(1, 2, distance=2.0, A=1.0)
        volumes = np.array([1.0, 1.5, 2.0])

        kernel = compute_diffusion_kernels(
            graph, volumes=volumes, sigma=1.0, mode="density"
        )
        weighted = volumes @ kernel
        np.testing.assert_allclose(weighted, 1.0, atol=1e-10)

    def test_average_row_stochastic(self):
        """average mode: each row sums to 1."""
        graph = _line_graph(4)
        kernel = compute_diffusion_kernels(
            graph, volumes=np.array([1.0, 2.0, 1.5, 1.0]), sigma=0.7, mode="average"
        )
        np.testing.assert_allclose(kernel.sum(axis=1), 1.0, atol=1e-10)

    def test_mass_conservation(self):
        """transition mode conserves total mass under kernel @ field."""
        graph = _line_graph(5)
        kernel = compute_diffusion_kernels(
            graph, volumes=np.ones(5), sigma=1.0, mode="transition"
        )
        field = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        smoothed = kernel @ field
        np.testing.assert_allclose(smoothed.sum(), field.sum(), atol=1e-10)

    def test_impulse_spreading(self):
        """An impulse spreads to neighbors, decaying with distance."""
        graph = _line_graph(5)
        kernel = compute_diffusion_kernels(
            graph, volumes=np.ones(5), sigma=1.0, mode="transition"
        )
        field = np.zeros(5)
        field[2] = 1.0
        smoothed = kernel @ field
        assert smoothed[2] > smoothed[1] > smoothed[0] > 0

    def test_disconnected_components_no_leak(self):
        """Smoothing does not leak mass between disconnected components."""
        graph = nx.Graph()
        graph.add_node(0, pos=(0.0,))
        graph.add_node(1, pos=(1.0,))
        graph.add_edge(0, 1, distance=1.0, A=1.0)
        graph.add_node(2, pos=(10.0,))
        graph.add_node(3, pos=(11.0,))
        graph.add_edge(2, 3, distance=1.0, A=1.0)

        kernel = compute_diffusion_kernels(
            graph, volumes=np.ones(4), sigma=1.0, mode="transition"
        )
        field = np.array([1.0, 0.0, 0.0, 0.0])
        smoothed = kernel @ field
        assert smoothed[0] + smoothed[1] == pytest.approx(1.0, abs=1e-12)
        assert smoothed[2] + smoothed[3] == pytest.approx(0.0, abs=1e-12)

    def test_larger_bandwidth_spreads_more(self):
        """A larger sigma keeps less mass at the source and spreads farther."""
        graph = _line_graph(5)
        field = np.zeros(5)
        field[2] = 1.0
        small = (
            compute_diffusion_kernels(
                graph, volumes=np.ones(5), sigma=0.3, mode="transition"
            )
            @ field
        )
        large = (
            compute_diffusion_kernels(
                graph, volumes=np.ones(5), sigma=2.0, mode="transition"
            )
            @ field
        )
        assert small[2] > large[2]
        assert large[0] > small[0]

    def test_does_not_mutate_input_graph(self):
        """The primitive reads edges but must not add attributes to the caller's graph."""
        graph = _grid_graph_with_A(3)
        before = {(u, v): dict(d) for u, v, d in graph.edges(data=True)}
        compute_diffusion_kernels(
            graph, volumes=np.ones(9), sigma=1.0, mode="transition"
        )
        after = {(u, v): dict(d) for u, v, d in graph.edges(data=True)}
        assert before == after, "input graph edges must be unchanged"


class TestComputeDiffusionKernelsValidation:
    """Parameter validation (C6) at the primitive level."""

    def test_missing_A_raises(self):
        graph = _line_graph(2)
        del graph.edges[0, 1]["A"]
        with pytest.raises(ValueError, match=r"missing 'A'"):
            compute_diffusion_kernels(
                graph, volumes=np.ones(2), sigma=1.0, mode="transition"
            )

    def test_missing_distance_raises(self):
        graph = _line_graph(2)
        del graph.edges[0, 1]["distance"]
        with pytest.raises(ValueError, match=r"missing 'A' and/or 'distance'"):
            compute_diffusion_kernels(
                graph, volumes=np.ones(2), sigma=1.0, mode="transition"
            )

    def test_bad_volumes_shape_raises(self):
        graph = _line_graph(2)
        with pytest.raises(ValueError, match=r"volumes must have shape"):
            compute_diffusion_kernels(
                graph, volumes=np.ones(3), sigma=1.0, mode="transition"
            )

    def test_nonpositive_volumes_raise(self):
        graph = _line_graph(2)
        with pytest.raises(ValueError, match=r"volumes must be finite"):
            compute_diffusion_kernels(
                graph, volumes=np.array([1.0, 0.0]), sigma=1.0, mode="transition"
            )

    def test_nonpositive_sigma_raises(self):
        graph = _line_graph(2)
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError, match=r"sigma must be finite and > 0"):
                compute_diffusion_kernels(
                    graph, volumes=np.ones(2), sigma=bad, mode="transition"
                )

    def test_invalid_mode_raises(self):
        graph = _line_graph(2)
        with pytest.raises(ValueError, match=r"Invalid mode"):
            compute_diffusion_kernels(
                graph, volumes=np.ones(2), sigma=1.0, mode="invalid"
            )


class TestKernelEdgeCases:
    """Edge cases in kernel computation."""

    def test_single_node_graph(self):
        graph = nx.Graph()
        graph.add_node(0, pos=(0.0,))
        kernel = compute_diffusion_kernels(
            graph, volumes=np.ones(1), sigma=1.0, mode="transition"
        )
        assert kernel.shape == (1, 1)
        assert np.isclose(kernel[0, 0], 1.0)

    def test_two_node_graph(self):
        graph = _line_graph(2)
        kernel = compute_diffusion_kernels(
            graph, volumes=np.ones(2), sigma=1.0, mode="transition"
        )
        assert kernel.shape == (2, 2)
        np.testing.assert_allclose(kernel.sum(axis=0), 1.0)
        assert np.allclose(kernel, kernel.T, atol=1e-10)

    def test_very_small_bandwidth_localizes(self):
        graph = _line_graph(5)
        kernel = compute_diffusion_kernels(
            graph, volumes=np.ones(5), sigma=0.01, mode="transition"
        )
        assert np.diag(kernel).sum() > 0.95

    def test_very_large_bandwidth_spreads(self):
        graph = _line_graph(5)
        kernel = compute_diffusion_kernels(
            graph, volumes=np.ones(5), sigma=100.0, mode="transition"
        )
        for col in range(5):
            assert np.std(kernel[:, col]) < 0.3


class TestEnvironmentComputeKernel:
    """Tests for Environment.compute_kernel() wrapper method."""

    def test_compute_kernel_basic(self):
        data = np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
        )
        env = Environment.from_samples(data, bin_size=1.0)
        kernel = env.compute_kernel(bandwidth=1.0, mode="transition")
        assert kernel.shape == (env.n_bins, env.n_bins)
        assert kernel.dtype == np.float64

    def test_compute_kernel_density_uses_layout_volumes(self):
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 10, (100, 2))
        env = Environment.from_samples(data, bin_size=1.0)
        kernel = env.compute_kernel(bandwidth=1.0, mode="density")
        assert kernel.shape == (env.n_bins, env.n_bins)

    def test_compute_kernel_cache_behavior(self):
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 10, (50, 2))
        env = Environment.from_samples(data, bin_size=1.0)
        kernel1 = env.compute_kernel(bandwidth=1.0, mode="transition", cache=True)
        kernel2 = env.compute_kernel(bandwidth=1.0, mode="transition", cache=True)
        assert kernel1 is kernel2
        kernel3 = env.compute_kernel(bandwidth=2.0, mode="transition", cache=True)
        assert kernel3 is not kernel1
        kernel4 = env.compute_kernel(bandwidth=1.0, mode="density", cache=True)
        assert kernel4 is not kernel1

    def test_compute_kernel_cache_disabled(self):
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 10, (50, 2))
        env = Environment.from_samples(data, bin_size=1.0)
        kernel1 = env.compute_kernel(bandwidth=1.0, mode="transition", cache=False)
        kernel2 = env.compute_kernel(bandwidth=1.0, mode="transition", cache=False)
        assert kernel1 is not kernel2
        assert np.allclose(kernel1, kernel2)

    def test_compute_kernel_rejects_invalid_mode(self):
        rng = np.random.default_rng(0)
        env = Environment.from_samples(rng.uniform(0, 10, (50, 2)), bin_size=1.0)
        with pytest.raises(ValueError, match=r"mode must be one of"):
            env.compute_kernel(bandwidth=1.0, mode="bogus")

    def test_compute_kernel_high_bin_warns_and_returns(self, monkeypatch):
        from neurospatial.ops import smoothing

        rng = np.random.default_rng(0)
        data = rng.uniform(0, 10, (50, 2))
        env = Environment.from_samples(data, bin_size=1.0)
        assert env.n_bins > 1
        monkeypatch.setattr(smoothing, "_LARGE_KERNEL_THRESHOLD", 1)
        with pytest.warns(UserWarning, match="GB"):
            kernel = env.compute_kernel(bandwidth=1.0, mode="transition", cache=False)
        assert kernel.shape == (env.n_bins, env.n_bins)

    def test_compute_kernel_requires_fitted(self):
        from neurospatial.layout.engines.regular_grid import RegularGridLayout

        layout = RegularGridLayout()
        env = Environment(name="test", layout=layout)
        assert not env._is_fitted
        with pytest.raises(RuntimeError, match="fully initialized"):
            env.compute_kernel(bandwidth=1.0)


class TestKernelHighBinWarning:
    """The high-bin memory WARNING (there is no hard limit)."""

    def test_high_bin_warns_and_returns(self, monkeypatch):
        from neurospatial.ops import smoothing

        monkeypatch.setattr(smoothing, "_LARGE_KERNEL_THRESHOLD", 5)
        graph = _line_graph(6)
        with pytest.warns(UserWarning) as record:
            kernel = smoothing.compute_diffusion_kernels(
                graph, volumes=np.ones(6), sigma=1.0, mode="transition"
            )
        message = str(record[0].message)
        assert "6" in message
        assert "GB" in message
        assert "binned" in message
        assert kernel.shape == (6, 6)

    def test_does_not_raise(self, monkeypatch):
        from neurospatial.ops import smoothing

        monkeypatch.setattr(smoothing, "_LARGE_KERNEL_THRESHOLD", 2)
        graph = _line_graph(8)
        with pytest.warns(UserWarning, match="GB"):
            kernel = smoothing.compute_diffusion_kernels(
                graph, volumes=np.ones(8), sigma=0.5, mode="transition"
            )
        assert kernel.shape == (8, 8)

    def test_under_threshold_no_warn(self, monkeypatch):
        import warnings as _warnings

        from neurospatial.ops import smoothing

        monkeypatch.setattr(smoothing, "_LARGE_KERNEL_THRESHOLD", 5)
        graph = _line_graph(5)  # 5 is NOT > 5
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            kernel = smoothing.compute_diffusion_kernels(
                graph, volumes=np.ones(5), sigma=1.0, mode="transition"
            )
            kernel_warnings = [
                warning for warning in w if "diffusion kernel" in str(warning.message)
            ]
            assert len(kernel_warnings) == 0
        assert kernel.shape == (5, 5)

    def test_docstring_mentions_memory_and_performance(self):
        from neurospatial.ops.smoothing import compute_diffusion_kernels

        doc = compute_diffusion_kernels.__doc__.lower()
        assert "memory" in doc
        assert "gb" in doc
        assert "performance" in doc


class TestApplyKernel:
    """Test apply_kernel function for forward and adjoint operations."""

    @pytest.fixture
    def simple_kernel(self):
        return np.array(
            [[0.5, 0.2, 0.1], [0.3, 0.6, 0.2], [0.2, 0.2, 0.7]], dtype=np.float64
        )

    @pytest.fixture
    def simple_field(self):
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)

    @pytest.fixture
    def bin_sizes_3(self):
        return np.array([1.0, 2.0, 1.0], dtype=np.float64)

    def test_forward_mode_basic(self, simple_kernel, simple_field):
        from neurospatial.ops.smoothing import apply_kernel

        result = apply_kernel(simple_field, simple_kernel, mode="forward")
        np.testing.assert_allclose(result, simple_kernel @ simple_field, rtol=1e-10)

    def test_adjoint_mode_no_bin_sizes(self, simple_kernel, simple_field):
        from neurospatial.ops.smoothing import apply_kernel

        result = apply_kernel(simple_field, simple_kernel, mode="adjoint")
        np.testing.assert_allclose(result, simple_kernel.T @ simple_field, rtol=1e-10)

    def test_adjoint_mode_with_bin_sizes(
        self, simple_kernel, simple_field, bin_sizes_3
    ):
        from neurospatial.ops.smoothing import apply_kernel

        result = apply_kernel(
            simple_field, simple_kernel, mode="adjoint", bin_sizes=bin_sizes_3
        )
        m_field = bin_sizes_3 * simple_field
        expected = (simple_kernel.T @ m_field) / bin_sizes_3
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_invalid_mode_raises(self, simple_kernel, simple_field):
        from neurospatial.ops.smoothing import apply_kernel

        with pytest.raises(ValueError, match="mode must be"):
            apply_kernel(simple_field, simple_kernel, mode="invalid")

    def test_non_square_kernel_raises(self, simple_field):
        from neurospatial.ops.smoothing import apply_kernel

        bad_kernel = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="Kernel must be square"):
            apply_kernel(simple_field, bad_kernel, mode="forward")

    def test_field_size_mismatch_raises(self, simple_kernel):
        from neurospatial.ops.smoothing import apply_kernel

        bad_field = np.array([1.0, 2.0], dtype=np.float64)
        with pytest.raises(ValueError, match=r"Field size.*does not match"):
            apply_kernel(bad_field, simple_kernel, mode="forward")

    def test_bin_sizes_mismatch_raises(self, simple_kernel, simple_field):
        from neurospatial.ops.smoothing import apply_kernel

        bad_bin_sizes = np.array([1.0, 2.0], dtype=np.float64)
        with pytest.raises(ValueError, match=r"bin_sizes size.*does not match"):
            apply_kernel(
                simple_field, simple_kernel, mode="adjoint", bin_sizes=bad_bin_sizes
            )

    def test_non_positive_bin_sizes_raises(self, simple_kernel, simple_field):
        from neurospatial.ops.smoothing import apply_kernel

        bad_bin_sizes = np.array([1.0, 0.0, -1.0], dtype=np.float64)
        with pytest.raises(ValueError, match="bin_sizes must have strictly positive"):
            apply_kernel(
                simple_field, simple_kernel, mode="adjoint", bin_sizes=bad_bin_sizes
            )

    def test_forward_adjoint_duality_no_bin_sizes(self, simple_kernel):
        from neurospatial.ops.smoothing import apply_kernel

        u = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        v = np.array([0.5, 1.5, 0.8], dtype=np.float64)
        ku = apply_kernel(u, simple_kernel, mode="forward")
        ktv = apply_kernel(v, simple_kernel, mode="adjoint")
        np.testing.assert_allclose(np.dot(ku, v), np.dot(u, ktv), rtol=1e-10)

    def test_forward_adjoint_duality_with_bin_sizes(self, simple_kernel, bin_sizes_3):
        from neurospatial.ops.smoothing import apply_kernel

        u = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        v = np.array([0.5, 1.5, 0.8], dtype=np.float64)
        ku = apply_kernel(u, simple_kernel, mode="forward")
        kstar_v = apply_kernel(v, simple_kernel, mode="adjoint", bin_sizes=bin_sizes_3)
        np.testing.assert_allclose(
            np.dot(ku * bin_sizes_3, v), np.dot(u * bin_sizes_3, kstar_v), rtol=1e-10
        )

    def test_bin_sizes_allowed_in_forward_mode(
        self, simple_kernel, simple_field, bin_sizes_3
    ):
        from neurospatial.ops.smoothing import apply_kernel

        result = apply_kernel(
            simple_field, simple_kernel, mode="forward", bin_sizes=bin_sizes_3
        )
        np.testing.assert_allclose(result, simple_kernel @ simple_field, rtol=1e-10)
