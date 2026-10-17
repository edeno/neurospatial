"""Tests for neurospatial.encoding.grid module (TDD RED phase).

These tests verify that all grid cell functions are importable from the new
encoding.grid location after the module reorganization.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment


# ==============================================================================
# Test imports from encoding.grid
# ==============================================================================
class TestEncodingGridImports:
    """Test all imports from neurospatial.encoding.grid."""

    # --- Dataclasses ---

    def test_import_grid_properties_dataclass(self) -> None:
        """GridProperties should be importable from encoding.grid."""
        from neurospatial.encoding.grid import GridProperties

        assert GridProperties is not None

    # --- Functions from metrics/grid_cells.py ---

    def test_import_grid_score(self) -> None:
        """grid_score should be importable from encoding.grid."""
        from neurospatial.encoding.grid import grid_score

        assert callable(grid_score)

    def test_import_spatial_autocorrelation(self) -> None:
        """spatial_autocorrelation should be importable from encoding.grid."""
        from neurospatial.encoding.grid import (
            spatial_autocorrelation,
        )

        assert callable(spatial_autocorrelation)

    def test_import_grid_scale(self) -> None:
        """grid_scale should be importable from encoding.grid."""
        from neurospatial.encoding.grid import grid_scale

        assert callable(grid_scale)

    def test_import_grid_properties_function(self) -> None:
        """grid_properties should be importable from encoding.grid."""
        from neurospatial.encoding.grid import grid_properties

        assert callable(grid_properties)

    def test_import_periodicity_score(self) -> None:
        """periodicity_score should be importable from encoding.grid."""
        from neurospatial.encoding.grid import periodicity_score

        assert callable(periodicity_score)


class TestEncodingGridFromEncodingInit:
    """Test all imports from neurospatial.encoding (via __init__.py)."""

    def test_import_grid_score_from_encoding(self) -> None:
        """grid_score should be importable from encoding __init__."""
        from neurospatial.encoding import grid_score

        assert callable(grid_score)

    def test_import_grid_properties_dataclass_from_encoding(self) -> None:
        """GridProperties should be importable from encoding __init__."""
        from neurospatial.encoding import GridProperties

        assert GridProperties is not None

    def test_import_spatial_autocorrelation_from_encoding(self) -> None:
        """spatial_autocorrelation should be importable from encoding __init__."""
        from neurospatial.encoding import spatial_autocorrelation

        assert callable(spatial_autocorrelation)

    def test_import_grid_properties_from_encoding(self) -> None:
        """grid_properties should be importable from encoding __init__."""
        from neurospatial.encoding import grid_properties

        assert callable(grid_properties)


# ==============================================================================
# Test basic functionality
# ==============================================================================
def _create_hexagonal_autocorr(size: int = 100, radius: float = 20.0) -> np.ndarray:
    """Create synthetic hexagonal autocorrelogram for testing."""
    autocorr = np.zeros((size, size))
    center = size // 2

    # Coordinate grids
    y_grid, x_grid = np.ogrid[:size, :size]

    # Central peak
    dist_from_center = np.sqrt((y_grid - center) ** 2 + (x_grid - center) ** 2)
    autocorr = np.exp(-(dist_from_center**2) / (2 * 5**2))

    # Add 6 peaks at 60 degrees intervals (hexagonal pattern)
    for angle_deg in [0, 60, 120, 180, 240, 300]:
        angle_rad = np.radians(angle_deg)
        peak_y = center + int(radius * np.sin(angle_rad))
        peak_x = center + int(radius * np.cos(angle_rad))
        peak_dist = np.sqrt((y_grid - peak_y) ** 2 + (x_grid - peak_x) ** 2)
        autocorr += 0.8 * np.exp(-(peak_dist**2) / (2 * 5**2))

    return autocorr / autocorr.max()


class TestEncodingGridFunctionality:
    """Test basic functionality of encoding.grid functions."""

    @pytest.fixture
    def env_and_data(self):
        """Create test environment and trajectory data."""
        # Create regular 2D grid environment using deterministic grid
        x = np.linspace(-20, 20, 21)
        xx, yy = np.meshgrid(x, x)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=4.0)

        # Random firing pattern
        rng = np.random.default_rng(42)
        firing_rate = rng.random(env.n_bins) * 5.0

        return env, firing_rate

    def test_spatial_autocorrelation_fft_runs(self, env_and_data) -> None:
        """spatial_autocorrelation (FFT) should run and return correct shape."""
        from neurospatial.encoding.grid import (
            spatial_autocorrelation,
        )

        env, firing_rate = env_and_data
        result = spatial_autocorrelation(env, firing_rate)

        assert result.ndim == 2
        assert result.shape == env.grid_shape

    def test_spatial_autocorrelation_graph_runs(self, env_and_data) -> None:
        """spatial_autocorrelation (graph) should run and return tuple."""
        from neurospatial.encoding.grid import (
            spatial_autocorrelation_radial,
        )

        env, firing_rate = env_and_data
        result = spatial_autocorrelation_radial(env, firing_rate, n_distance_bins=30)

        assert isinstance(result, tuple)
        assert len(result) == 2
        distances, correlations = result
        assert distances.shape == correlations.shape
        assert len(distances) == 30

    def test_grid_score_runs(self) -> None:
        """grid_score should run and return float."""
        from neurospatial.encoding.grid import grid_score

        autocorr = _create_hexagonal_autocorr()
        score = grid_score(autocorr)

        assert isinstance(score, float)
        assert -2.0 <= score <= 2.0

    def test_grid_scale_runs(self) -> None:
        """grid_scale should run and return float."""
        from neurospatial.encoding.grid import grid_scale

        autocorr = _create_hexagonal_autocorr()
        scale = grid_scale(autocorr, bin_size=2.0)

        assert isinstance(scale, float)
        assert scale > 0

    def test_grid_orientation_via_grid_properties_runs(self) -> None:
        """Orientation is now exposed via grid_properties."""
        from neurospatial.encoding.grid import grid_properties

        autocorr = _create_hexagonal_autocorr()
        props = grid_properties(autocorr, bin_size=1.0)

        assert isinstance(props.orientation, float)
        assert isinstance(props.orientation_std, float)
        assert 0.0 <= props.orientation < 60.0

    def test_periodicity_score_runs(self) -> None:
        """periodicity_score should run and return float."""
        from neurospatial.encoding.grid import periodicity_score

        # Create perfect sinusoidal correlation profile
        distances = np.linspace(0, 100, 200)
        period = 20.0
        correlations = 0.5 * np.sin(2 * np.pi * distances / period) + 0.5

        score = periodicity_score(distances, correlations, min_peaks=2)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_grid_properties_runs(self) -> None:
        """grid_properties should run and return GridProperties dataclass."""
        from neurospatial.encoding.grid import GridProperties, grid_properties

        autocorr = _create_hexagonal_autocorr()
        props = grid_properties(autocorr, bin_size=2.0)

        assert isinstance(props, GridProperties)
        assert hasattr(props, "score")
        assert hasattr(props, "scale")
        assert hasattr(props, "orientation")
        assert hasattr(props, "orientation_std")
        assert hasattr(props, "peak_coords")
        assert hasattr(props, "n_peaks")

    def test_hexagonal_pattern_high_grid_score(self) -> None:
        """Hexagonal pattern should produce high grid score."""
        from neurospatial.encoding.grid import grid_score

        autocorr = _create_hexagonal_autocorr()
        score = grid_score(autocorr)

        # Hexagonal pattern should have score > 0.3
        assert score > 0.3


class TestFFTAutocorrelationNeedsRectangularCells:
    """The FFT autocorrelogram is defined only on a rectangular lattice."""

    @pytest.fixture
    def hexagonal_env_and_rate(self):
        x = np.linspace(-20, 20, 41)
        xx, yy = np.meshgrid(x, x)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=4.0, layout="Hexagonal")
        rate = np.random.default_rng(1).random(env.n_bins) * 5.0
        return env, rate

    def test_hexagonal_layout_is_refused_with_the_radial_alternative(
        self, hexagonal_env_and_rate
    ) -> None:
        from neurospatial.encoding.grid import (
            spatial_autocorrelation,
            spatial_autocorrelation_radial,
        )

        env, rate = hexagonal_env_and_rate
        # The hexagonal lattice has a 2D shape, which the FFT path used to
        # accept although its offset rows are not a rectangular lag grid.
        assert len(env.grid_shape) == 2

        with pytest.raises(NotImplementedError, match="Hexagonal") as error:
            spatial_autocorrelation(env, rate)

        assert "spatial_autocorrelation_radial" in str(error.value)
        distances, correlations = spatial_autocorrelation_radial(env, rate)
        assert distances.shape == correlations.shape

    def test_summary_table_reports_no_grid_score_on_a_hexagonal_layout(
        self, hexagonal_env_and_rate
    ) -> None:
        from neurospatial import compute_spatial_rate

        env, _ = hexagonal_env_and_rate
        stimulus = np.random.default_rng(3)
        times = np.arange(6000) / 30.0
        positions = stimulus.uniform(-20, 20, size=(len(times), 2))
        spike_times = np.sort(np.random.default_rng(4).uniform(0, times[-1], 400))

        table = compute_spatial_rate(env, spike_times, times, positions).summary_table()

        assert np.isnan(table["grid_score"]).all()

    def test_sparse_grid_is_refused_with_the_same_error(self) -> None:
        from neurospatial.encoding.grid import spatial_autocorrelation

        mask = np.zeros((10, 10), dtype=bool)
        mask[:3, :] = True
        env = Environment.from_grid_mask(mask, grid_edges=(np.arange(11.0),) * 2)
        rate = np.random.default_rng(2).random(env.n_bins)

        with pytest.raises(NotImplementedError, match="only 30 of its 100"):
            spatial_autocorrelation(env, rate)


class TestFFTAutocorrelationNeedsOneCellWidth:
    """FFT lags are cell counts, so they are distances only for one cell width."""

    @staticmethod
    def _grid(x_edges, y_edges):
        """Fully active rectangular cells on the given edges, from explicit bins."""
        from neurospatial.binning import (
            Binning,
            MissingFaces,
            RectilinearGrid,
            RectilinearLocator,
            SpatialBins,
        )

        shape = (len(x_edges) - 1, len(y_edges) - 1)
        x_mid, y_mid = (
            (x_edges[:-1] + x_edges[1:]) / 2,
            (y_edges[:-1] + y_edges[1:]) / 2,
        )
        centers = np.stack(np.meshgrid(x_mid, y_mid, indexing="ij"), -1).reshape(-1, 2)
        areas = np.outer(np.diff(x_edges), np.diff(y_edges)).ravel()
        index = np.arange(centers.shape[0]).reshape(shape)
        pairs = np.vstack(
            [
                np.column_stack((index[:-1].ravel(), index[1:].ravel())),
                np.column_stack((index[:, :-1].ravel(), index[:, 1:].ravel())),
            ]
        )
        distances = np.linalg.norm(centers[pairs[:, 1]] - centers[pairs[:, 0]], axis=1)
        mask = np.ones(shape, dtype=bool)
        edges = (x_edges, y_edges)
        return Environment.from_bins(
            Binning(
                "RegularGrid",
                SpatialBins.from_arrays(centers, areas, 2, pairs, distances),
                RectilinearLocator(edges, shape, mask, None),
                MissingFaces("No finite-volume faces supplied."),
                rectilinear=RectilinearGrid(edges, shape, mask),
            )
        )

    @pytest.mark.parametrize(
        "x_edges",
        [
            np.concatenate([np.arange(0.0, 20.0, 2.0), np.arange(20.0, 41.0, 4.0)]),
            np.arange(0.0, 61.0, 3.0),
        ],
        ids=["uneven", "non-square"],
    )
    def test_cells_of_different_widths_are_refused(self, x_edges) -> None:
        from neurospatial.encoding.grid import spatial_autocorrelation

        env = self._grid(x_edges, np.arange(0.0, 41.0, 2.0))
        rate = np.random.default_rng(5).random(env.n_bins)

        with pytest.raises(NotImplementedError, match="not all one width") as error:
            spatial_autocorrelation(env, rate)

        assert "spatial_autocorrelation_radial" in str(error.value)

    def test_rounding_in_far_from_origin_edges_is_one_width(self) -> None:
        from neurospatial.encoding.grid import spatial_autocorrelation

        # Edges computed in floating point far from the origin: the widths
        # differ from 0.7 cm only by rounding, about 3e-12 of a cell.
        edges = 1.0e4 + 0.1 + 0.7 * np.arange(21)
        assert len(np.unique(np.diff(edges))) > 1
        env = self._grid(edges, 1.0e4 + 0.1 + 0.7 * np.arange(21))
        rate = np.random.default_rng(6).random(env.n_bins)

        autocorr = spatial_autocorrelation(env, rate)

        assert autocorr.shape == env.grid_shape
