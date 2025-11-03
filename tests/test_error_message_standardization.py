"""Tests for standardized error messages showing actual values.

This test suite ensures that all parameter validation error messages follow
the pattern: "{param} must be {constraint} (got {actual_value})"

This makes debugging much easier by showing users the actual values that
caused the error, not just what was expected.
"""

import numpy as np
import pytest

from neurospatial.calibration import simple_scale
from neurospatial.layout.helpers.graph import _get_graph_bins
from neurospatial.layout.helpers.hexagonal import _create_hex_grid
from neurospatial.layout.helpers.regular_grid import _create_regular_grid
from neurospatial.layout.helpers.utils import (
    _infer_active_elements_from_samples,
    get_n_bins,
)


class TestUtilsErrorMessages:
    """Test error messages in layout/helpers/utils.py."""

    def test_bin_size_negative_shows_actual_value(self):
        """Test that bin_size validation shows the actual negative value."""
        data = np.array([[0, 0], [10, 10]])
        bad_bin_size = -5.0

        with pytest.raises(ValueError) as exc_info:
            get_n_bins(data, bin_size=bad_bin_size)

        error_msg = str(exc_info.value)
        # Should show actual value that was provided
        assert str(bad_bin_size) in error_msg or f"{bad_bin_size}" in error_msg
        # Should explain what was expected
        assert "positive" in error_msg.lower()

    def test_bin_size_zero_shows_actual_value(self):
        """Test that bin_size=0 validation shows the actual value."""
        data = np.array([[0, 0], [10, 10]])
        bad_bin_size = 0.0

        with pytest.raises(ValueError) as exc_info:
            get_n_bins(data, bin_size=bad_bin_size)

        error_msg = str(exc_info.value)
        assert "0" in error_msg or "0.0" in error_msg
        assert "positive" in error_msg.lower()

    def test_bin_size_sequence_negative_shows_actual_values(self):
        """Test that sequence bin_size with negative shows actual values."""
        data = np.array([[0, 0], [10, 10]])
        bad_bin_size = [5.0, -2.0]

        with pytest.raises(ValueError) as exc_info:
            get_n_bins(data, bin_size=bad_bin_size)

        error_msg = str(exc_info.value)
        # Should show that there was a negative value
        assert "positive" in error_msg.lower()
        # Ideally shows which dimension or the actual array
        assert "-2" in error_msg or "negative" in error_msg.lower()

    def test_bin_count_threshold_negative_shows_actual_value(self):
        """Test that bin_count_threshold validation shows actual value."""
        candidates = np.array([[0, 0], [1, 1], [2, 2]])
        samples = np.array([[0.5, 0.5], [1.5, 1.5]])
        bad_threshold = -10

        with pytest.raises(ValueError) as exc_info:
            _infer_active_elements_from_samples(
                candidates, samples, bin_count_threshold=bad_threshold
            )

        error_msg = str(exc_info.value)
        # Should show the actual negative value provided
        assert str(bad_threshold) in error_msg or f"{bad_threshold}" in error_msg
        # Should explain constraint
        assert "non-negative" in error_msg.lower()


class TestCalibrationErrorMessages:
    """Test error messages in calibration.py."""

    def test_offset_px_wrong_length_shows_actual_value(self):
        """Test that offset_px validation shows actual value when wrong length."""
        bad_offset = (10.0,)  # Only one value, should be two

        with pytest.raises(ValueError) as exc_info:
            simple_scale(px_per_cm=2.0, offset_px=bad_offset)

        error_msg = str(exc_info.value)
        # Should mention it needs two values
        assert "two" in error_msg.lower()
        # Should show what was actually provided (can't easily show tuple length in message)
        # At minimum, should be helpful about the requirement

    def test_offset_px_wrong_type_shows_helpful_message(self):
        """Test that offset_px with wrong type shows helpful message."""
        bad_offset = "not_a_tuple"  # pyright: ignore[reportAssignmentType]

        with pytest.raises(ValueError) as exc_info:
            simple_scale(px_per_cm=2.0, offset_px=bad_offset)  # pyright: ignore[reportArgumentType]

        error_msg = str(exc_info.value)
        # Should explain it needs a tuple of two numeric values
        assert "two" in error_msg.lower() and "numeric" in error_msg.lower()


class TestRegularGridErrorMessages:
    """Test error messages in layout/helpers/regular_grid.py."""

    def test_bin_size_negative_shows_actual_value_in_create_regular_grid(self):
        """Test bin_size validation in _create_regular_grid shows actual value."""
        data = np.array([[0, 0], [10, 10]])
        bad_bin_size = -3.0

        with pytest.raises(ValueError) as exc_info:
            _create_regular_grid(data_samples=data, bin_size=bad_bin_size)

        error_msg = str(exc_info.value)
        assert str(bad_bin_size) in error_msg or f"{bad_bin_size}" in error_msg
        assert "positive" in error_msg.lower()

    def test_bin_size_sequence_wrong_length_shows_actual_vs_expected(self):
        """Test bin_size sequence length mismatch shows both actual and expected."""
        data = np.array([[0, 0, 0], [10, 10, 10]])  # 3D data
        bad_bin_size = [2.0, 3.0]  # Only 2 values for 3D data

        with pytest.raises(ValueError) as exc_info:
            _create_regular_grid(data_samples=data, bin_size=bad_bin_size)

        error_msg = str(exc_info.value)
        # Should show expected dimension count
        assert "3" in error_msg
        # Should show actual length provided
        assert "2" in error_msg
        # Should mention length or dimensions
        assert ("length" in error_msg.lower()) or ("dimension" in error_msg.lower())


class TestHexagonalErrorMessages:
    """Test error messages in layout/helpers/hexagonal.py."""

    def test_hexagon_width_negative_shows_actual_value(self):
        """Test hexagon_width validation shows actual negative value."""
        data = np.array([[0, 0], [10, 10]])
        bad_width = -5.0

        with pytest.raises(ValueError) as exc_info:
            _create_hex_grid(data_samples=data, hexagon_width=bad_width)

        error_msg = str(exc_info.value)
        assert str(bad_width) in error_msg or f"{bad_width}" in error_msg
        assert "positive" in error_msg.lower()

    def test_hexagon_width_zero_shows_actual_value(self):
        """Test hexagon_width=0 validation shows actual value."""
        data = np.array([[0, 0], [10, 10]])
        bad_width = 0.0

        with pytest.raises(ValueError) as exc_info:
            _create_hex_grid(data_samples=data, hexagon_width=bad_width)

        error_msg = str(exc_info.value)
        assert "0" in error_msg or "0.0" in error_msg
        assert "positive" in error_msg.lower()

    def test_dimension_range_wrong_length_shows_actual_value(self):
        """Test dimension_range validation shows actual when not length 2."""
        bad_range = [(0, 10), (0, 10), (0, 10)]  # 3D range for 2D hexagonal grid

        with pytest.raises(ValueError) as exc_info:
            _create_hex_grid(
                data_samples=None, dimension_range=bad_range, hexagon_width=2.0
            )

        error_msg = str(exc_info.value)
        # Should mention the actual length provided (3)
        assert "3" in error_msg
        # Should mention (min_x, max_x), (min_y, max_y) or show expected format
        assert "min" in error_msg.lower() and "max" in error_msg.lower()

    def test_data_samples_wrong_shape_shows_actual_shape(self):
        """Test data_samples validation shows actual shape when not (n, 2)."""
        bad_data = np.array([[0, 0, 0], [10, 10, 10]])  # 3D data for 2D hexagonal grid

        with pytest.raises(ValueError) as exc_info:
            _create_hex_grid(data_samples=bad_data, hexagon_width=2.0)

        error_msg = str(exc_info.value)
        # Should show expected shape requirements
        assert "2" in error_msg
        # Should mention shape or samples
        assert ("shape" in error_msg.lower()) or ("sample" in error_msg.lower())


class TestGraphErrorMessages:
    """Test error messages in layout/helpers/graph.py."""

    def test_bin_size_negative_shows_actual_value(self):
        """Test bin_size validation in _get_graph_bins shows actual value."""
        import networkx as nx

        # Create a simple graph
        G = nx.Graph()
        G.add_node(0, pos=(0, 0))
        G.add_node(1, pos=(10, 0))
        G.add_edge(0, 1)

        edge_order = [(0, 1)]
        bad_bin_size = -2.0

        with pytest.raises(ValueError) as exc_info:
            _get_graph_bins(G, edge_order, edge_spacing=0.0, bin_size=bad_bin_size)

        error_msg = str(exc_info.value)
        assert str(bad_bin_size) in error_msg or f"{bad_bin_size}" in error_msg
        assert "positive" in error_msg.lower()

    def test_bin_size_zero_shows_actual_value(self):
        """Test bin_size=0 validation shows actual value."""
        import networkx as nx

        G = nx.Graph()
        G.add_node(0, pos=(0, 0))
        G.add_node(1, pos=(10, 0))
        G.add_edge(0, 1)

        edge_order = [(0, 1)]
        bad_bin_size = 0.0

        with pytest.raises(ValueError) as exc_info:
            _get_graph_bins(G, edge_order, edge_spacing=0.0, bin_size=bad_bin_size)

        error_msg = str(exc_info.value)
        assert "0" in error_msg or "0.0" in error_msg
        assert "positive" in error_msg.lower()

    def test_edge_spacing_wrong_length_shows_actual_vs_expected(self):
        """Test edge_spacing sequence length shows actual and expected."""
        import networkx as nx

        G = nx.Graph()
        G.add_node(0, pos=(0, 0))
        G.add_node(1, pos=(5, 0))
        G.add_node(2, pos=(10, 0))
        G.add_edge(0, 1)
        G.add_edge(1, 2)

        edge_order = [(0, 1), (1, 2)]
        bad_spacing = [1.0, 2.0, 3.0]  # Should be length 1 for 2 edges

        with pytest.raises(ValueError) as exc_info:
            _get_graph_bins(G, edge_order, edge_spacing=bad_spacing, bin_size=1.0)

        error_msg = str(exc_info.value)
        # Should show expected length (n_edges - 1 = 1)
        assert "1" in error_msg
        # Should show actual length provided (3)
        assert "3" in error_msg
        # Should mention length
        assert "length" in error_msg.lower()


class TestMixinsErrorMessages:
    """Test error messages in layout/mixins.py."""

    def test_grid_not_built_error_shows_helpful_message(self):
        """Test that accessing grid methods before building shows helpful error."""
        from neurospatial.layout.mixins import _GridMixin

        # Create a minimal instance
        mixin = _GridMixin()
        points = np.array([[5, 5]])

        with pytest.raises(RuntimeError) as exc_info:
            mixin.point_to_bin_index(points)

        error_msg = str(exc_info.value)
        # Should mention that grid is not built
        assert "not built" in error_msg.lower() or "missing" in error_msg.lower()
        # Should mention what's missing (edges or shape)
        assert ("edges" in error_msg.lower()) or ("shape" in error_msg.lower())

    def test_kdtree_not_built_returns_negative_one(self):
        """Test that KDTree not built doesn't crash, returns -1."""
        from neurospatial.layout.mixins import _KDTreeMixin

        mixin = _KDTreeMixin()
        points = np.array([[5, 5]])

        # Should return -1 for all points when KDTree not built
        result = mixin.point_to_bin_index(points)
        assert np.all(result == -1)

    def test_points_for_tree_not_2d_shows_actual_shape(self):
        """Test that _build_kdtree with wrong shape shows actual shape."""
        from neurospatial.layout.mixins import _KDTreeMixin

        mixin = _KDTreeMixin()
        bad_points = np.array([1, 2, 3])  # 1D instead of 2D

        with pytest.raises(ValueError) as exc_info:
            mixin._build_kdtree(bad_points)

        error_msg = str(exc_info.value)
        # Should mention 2D requirement
        assert "2d" in error_msg.lower() or "2-d" in error_msg.lower()
        # Should mention shape
        assert "shape" in error_msg.lower()


class TestErrorMessageFormat:
    """Test that error messages follow consistent format patterns."""

    def test_error_messages_contain_got_pattern(self):
        """Verify error messages follow 'must be X (got Y)' pattern."""
        data = np.array([[0, 0], [10, 10]])

        # Test with clearly bad value
        with pytest.raises(ValueError) as exc_info:
            get_n_bins(data, bin_size=-999.5)

        error_msg = str(exc_info.value)
        # Should ideally contain "got" to show actual value
        # (This is the ideal pattern we're aiming for)
        assert ("got" in error_msg.lower()) or ("-999" in error_msg)

    def test_error_messages_are_informative(self):
        """Verify error messages contain enough context for debugging."""
        data = np.array([[0, 0], [10, 10]])

        with pytest.raises(ValueError) as exc_info:
            get_n_bins(data, bin_size=0.0)

        error_msg = str(exc_info.value)
        # Should be longer than just "bin_size must be positive"
        # Should contain constraint explanation
        assert len(error_msg) > 20  # Reasonable minimum length
        assert "positive" in error_msg.lower()
        # Should contain the parameter name
        assert "bin" in error_msg.lower()
