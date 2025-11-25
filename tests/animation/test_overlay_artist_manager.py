"""Tests for OverlayArtistManager lifecycle and artist reuse.

This test module verifies the OverlayArtistManager:
1. Is created once per figure
2. Uses set_data/set_offsets for efficient artist updates (not recreate-on-each-frame)
3. Properly reinitializes in fallback paths
4. Properly clears artists when reset

These tests ensure stable overlay rendering for the widget backend.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation._parallel import OverlayArtistManager
from neurospatial.animation.overlays import (
    BodypartData,
    HeadDirectionData,
    OverlayData,
    PositionData,
)
from neurospatial.animation.skeleton import Skeleton

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_env() -> Environment:
    """Create a simple 2D environment for testing."""
    positions = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    env = Environment.from_samples(positions, bin_size=5.0)
    env.units = "cm"
    return env


@pytest.fixture
def mock_ax() -> MagicMock:
    """Create a mock matplotlib Axes object."""
    ax = MagicMock()

    # Mock scatter to return a PathCollection mock
    scatter_mock = MagicMock()
    scatter_mock.set_offsets = MagicMock()
    scatter_mock.set_facecolor = MagicMock()
    scatter_mock.set_visible = MagicMock()
    scatter_mock.remove = MagicMock()
    ax.scatter.return_value = scatter_mock

    # Mock add_collection - capture and set remove method
    def mock_add_collection(collection):
        # Set up the remove method properly for LineCollection
        collection._remove_method = lambda x: None
        return collection

    ax.add_collection = MagicMock(side_effect=mock_add_collection)

    # Mock add_patch
    ax.add_patch = MagicMock()

    # Mock quiver to return a Quiver mock
    quiver_mock = MagicMock()
    quiver_mock.remove = MagicMock()
    ax.quiver.return_value = quiver_mock

    return ax


@pytest.fixture
def position_overlay_data() -> OverlayData:
    """Create test position overlay data with trail."""
    rng = np.random.default_rng(42)
    n_frames = 10
    positions = rng.random((n_frames, 2)) * 10
    return OverlayData(
        positions=[PositionData(data=positions, color="red", size=10.0, trail_length=3)]
    )


@pytest.fixture
def bodypart_overlay_data() -> OverlayData:
    """Create test bodypart overlay data with skeleton."""
    rng = np.random.default_rng(42)
    n_frames = 10
    nose = rng.random((n_frames, 2)) * 10
    tail = rng.random((n_frames, 2)) * 10
    skeleton = Skeleton(
        name="test",
        nodes=("nose", "tail"),
        edges=(("nose", "tail"),),
        edge_color="white",
        edge_width=2.0,
    )
    return OverlayData(
        bodypart_sets=[
            BodypartData(
                bodyparts={"nose": nose, "tail": tail},
                skeleton=skeleton,
                colors={"nose": "red", "tail": "blue"},
            )
        ]
    )


@pytest.fixture
def head_direction_overlay_data() -> OverlayData:
    """Create test head direction overlay data."""
    rng = np.random.default_rng(42)
    n_frames = 10
    angles = rng.random(n_frames) * 2 * np.pi
    return OverlayData(
        head_directions=[HeadDirectionData(data=angles, color="yellow", length=5.0)]
    )


@pytest.fixture
def all_overlays_data() -> OverlayData:
    """Create overlay data with all types."""
    rng = np.random.default_rng(42)
    n_frames = 10
    positions = rng.random((n_frames, 2)) * 10
    nose = rng.random((n_frames, 2)) * 10
    tail = rng.random((n_frames, 2)) * 10
    angles = rng.random(n_frames) * 2 * np.pi

    skeleton = Skeleton(
        name="test",
        nodes=("nose", "tail"),
        edges=(("nose", "tail"),),
        edge_color="white",
        edge_width=2.0,
    )
    return OverlayData(
        positions=[
            PositionData(data=positions, color="red", size=10.0, trail_length=3)
        ],
        bodypart_sets=[
            BodypartData(
                bodyparts={"nose": nose, "tail": tail},
                skeleton=skeleton,
                colors={"nose": "red", "tail": "blue"},
            )
        ],
        head_directions=[HeadDirectionData(data=angles, color="yellow", length=5.0)],
    )


# ============================================================================
# Initialization Tests
# ============================================================================


class TestOverlayArtistManagerInitialization:
    """Tests for OverlayArtistManager initialization behavior."""

    def test_manager_starts_uninitialized(
        self,
        mock_ax: MagicMock,
        simple_env: Environment,
        position_overlay_data: OverlayData,
    ):
        """Test manager starts with _initialized=False."""
        manager = OverlayArtistManager(
            ax=mock_ax,
            env=simple_env,
            overlay_data=position_overlay_data,
            show_regions=False,
            region_alpha=0.3,
        )

        assert manager._initialized is False

    def test_initialize_sets_initialized_flag(
        self,
        mock_ax: MagicMock,
        simple_env: Environment,
        position_overlay_data: OverlayData,
    ):
        """Test initialize() sets _initialized=True."""
        manager = OverlayArtistManager(
            ax=mock_ax,
            env=simple_env,
            overlay_data=position_overlay_data,
            show_regions=False,
            region_alpha=0.3,
        )

        manager.initialize(frame_idx=0)

        assert manager._initialized is True

    def test_initialize_called_once_via_update_frame(
        self,
        mock_ax: MagicMock,
        simple_env: Environment,
        position_overlay_data: OverlayData,
    ):
        """Test update_frame calls initialize if not already initialized."""
        manager = OverlayArtistManager(
            ax=mock_ax,
            env=simple_env,
            overlay_data=position_overlay_data,
            show_regions=False,
            region_alpha=0.3,
        )

        # Should auto-initialize via update_frame
        manager.update_frame(frame_idx=0)

        assert manager._initialized is True

    def test_initialize_idempotent(
        self,
        mock_ax: MagicMock,
        simple_env: Environment,
        position_overlay_data: OverlayData,
    ):
        """Test calling initialize() multiple times is idempotent."""
        manager = OverlayArtistManager(
            ax=mock_ax,
            env=simple_env,
            overlay_data=position_overlay_data,
            show_regions=False,
            region_alpha=0.3,
        )

        manager.initialize(frame_idx=0)
        initial_scatter_call_count = mock_ax.scatter.call_count

        # Call initialize again - should be no-op
        manager.initialize(frame_idx=1)

        # scatter should not be called again
        assert mock_ax.scatter.call_count == initial_scatter_call_count

    def test_initialize_creates_position_artists(
        self,
        mock_ax: MagicMock,
        simple_env: Environment,
        position_overlay_data: OverlayData,
    ):
        """Test initialize creates position marker and trail artists."""
        manager = OverlayArtistManager(
            ax=mock_ax,
            env=simple_env,
            overlay_data=position_overlay_data,
            show_regions=False,
            region_alpha=0.3,
        )

        manager.initialize(frame_idx=0)

        # Should have position marker
        assert len(manager._position_markers) == 1
        # Should have trail LineCollection (since trail_length=3)
        assert len(manager._position_trails) == 1
        assert manager._position_trails[0] is not None

    def test_initialize_creates_bodypart_artists(
        self,
        mock_ax: MagicMock,
        simple_env: Environment,
        bodypart_overlay_data: OverlayData,
    ):
        """Test initialize creates bodypart scatter and skeleton artists."""
        manager = OverlayArtistManager(
            ax=mock_ax,
            env=simple_env,
            overlay_data=bodypart_overlay_data,
            show_regions=False,
            region_alpha=0.3,
        )

        manager.initialize(frame_idx=0)

        # Should have bodypart points
        assert len(manager._bodypart_points) == 1
        # Should have skeleton LineCollection
        assert len(manager._bodypart_skeletons) == 1
        assert manager._bodypart_skeletons[0] is not None

    def test_skeleton_initialization_order_of_operations(
        self,
        mock_ax: MagicMock,
        simple_env: Environment,
        bodypart_overlay_data: OverlayData,
    ):
        """Test skeleton is appended before update is called (regression test for IndexError).

        This test verifies the fix for the bug where _update_bodypart_skeleton was
        called with an index before the skeleton was appended to the list.
        """
        # This test would have raised IndexError before the fix
        manager = OverlayArtistManager(
            ax=mock_ax,
            env=simple_env,
            overlay_data=bodypart_overlay_data,
            show_regions=False,
            region_alpha=0.3,
        )

        # Should not raise IndexError during initialization
        manager.initialize(frame_idx=0)

        # Verify skeleton was created and list length matches bodypart_sets
        assert len(manager._bodypart_skeletons) == len(
            bodypart_overlay_data.bodypart_sets
        )

    def test_initialize_creates_head_direction_artist(
        self,
        mock_ax: MagicMock,
        simple_env: Environment,
        head_direction_overlay_data: OverlayData,
    ):
        """Test initialize creates head direction quiver artist."""
        manager = OverlayArtistManager(
            ax=mock_ax,
            env=simple_env,
            overlay_data=head_direction_overlay_data,
            show_regions=False,
            region_alpha=0.3,
        )

        manager.initialize(frame_idx=0)

        # Should have head direction quiver
        assert len(manager._head_direction_quivers) == 1
        # quiver should be called
        assert mock_ax.quiver.called

    def test_initialize_with_none_overlay_data(
        self, mock_ax: MagicMock, simple_env: Environment
    ):
        """Test initialize with no overlay data sets _initialized=True."""
        manager = OverlayArtistManager(
            ax=mock_ax,
            env=simple_env,
            overlay_data=None,
            show_regions=False,
            region_alpha=0.3,
        )

        manager.initialize(frame_idx=0)

        assert manager._initialized is True
        assert len(manager._position_markers) == 0


# ============================================================================
# Artist Update Tests (set_data/set_offsets reuse)
# ============================================================================


class TestOverlayArtistManagerUpdate:
    """Tests for efficient artist update behavior."""

    def test_update_frame_uses_set_offsets_for_position_marker(
        self,
        mock_ax: MagicMock,
        simple_env: Environment,
        position_overlay_data: OverlayData,
    ):
        """Test update_frame uses set_offsets() instead of recreating scatter."""
        manager = OverlayArtistManager(
            ax=mock_ax,
            env=simple_env,
            overlay_data=position_overlay_data,
            show_regions=False,
            region_alpha=0.3,
        )

        manager.initialize(frame_idx=0)
        initial_scatter_count = mock_ax.scatter.call_count

        # Update to new frame
        manager.update_frame(frame_idx=5)

        # scatter should NOT be called again (artist reused via set_offsets)
        assert mock_ax.scatter.call_count == initial_scatter_count

        # set_offsets SHOULD be called
        marker = manager._position_markers[0]
        assert marker.set_offsets.called

    def test_update_frame_uses_set_segments_for_trail(
        self,
        mock_ax: MagicMock,
        simple_env: Environment,
        position_overlay_data: OverlayData,
    ):
        """Test update_frame uses set_segments() for trail LineCollection."""
        # Create a mock LineCollection with set_segments
        mock_line_collection = MagicMock()
        mock_line_collection.set_segments = MagicMock()
        mock_line_collection.set_colors = MagicMock()
        mock_line_collection.set_visible = MagicMock()

        with patch(
            "neurospatial.animation._parallel.LineCollection",
            return_value=mock_line_collection,
        ):
            manager = OverlayArtistManager(
                ax=mock_ax,
                env=simple_env,
                overlay_data=position_overlay_data,
                show_regions=False,
                region_alpha=0.3,
            )

            manager.initialize(frame_idx=0)

            # Update to new frame
            manager.update_frame(frame_idx=5)

            # set_segments should be called for trail updates
            assert mock_line_collection.set_segments.called

    def test_update_frame_uses_set_offsets_for_bodypart_points(
        self,
        mock_ax: MagicMock,
        simple_env: Environment,
        bodypart_overlay_data: OverlayData,
    ):
        """Test update_frame uses set_offsets() for bodypart scatter."""
        manager = OverlayArtistManager(
            ax=mock_ax,
            env=simple_env,
            overlay_data=bodypart_overlay_data,
            show_regions=False,
            region_alpha=0.3,
        )

        manager.initialize(frame_idx=0)
        initial_scatter_count = mock_ax.scatter.call_count

        # Update to new frame
        manager.update_frame(frame_idx=5)

        # scatter should NOT be called again
        assert mock_ax.scatter.call_count == initial_scatter_count

        # set_offsets SHOULD be called
        points = manager._bodypart_points[0]
        assert points.set_offsets.called

    def test_update_frame_uses_set_segments_for_skeleton(
        self,
        mock_ax: MagicMock,
        simple_env: Environment,
        bodypart_overlay_data: OverlayData,
    ):
        """Test update_frame uses set_segments() for skeleton LineCollection."""
        mock_line_collection = MagicMock()
        mock_line_collection.set_segments = MagicMock()
        mock_line_collection.set_visible = MagicMock()

        with patch(
            "neurospatial.animation._parallel.LineCollection",
            return_value=mock_line_collection,
        ):
            manager = OverlayArtistManager(
                ax=mock_ax,
                env=simple_env,
                overlay_data=bodypart_overlay_data,
                show_regions=False,
                region_alpha=0.3,
            )

            manager.initialize(frame_idx=0)
            mock_line_collection.set_segments.reset_mock()

            # Update to new frame
            manager.update_frame(frame_idx=5)

            # set_segments should be called for skeleton updates
            assert mock_line_collection.set_segments.called

    def test_update_frame_recreates_quiver_for_head_direction(
        self,
        mock_ax: MagicMock,
        simple_env: Environment,
        head_direction_overlay_data: OverlayData,
    ):
        """Test update_frame recreates quiver for head direction (matplotlib limitation)."""
        manager = OverlayArtistManager(
            ax=mock_ax,
            env=simple_env,
            overlay_data=head_direction_overlay_data,
            show_regions=False,
            region_alpha=0.3,
        )

        manager.initialize(frame_idx=0)
        quiver_calls_after_init = mock_ax.quiver.call_count
        old_quiver = manager._head_direction_quivers[0]

        # Update to new frame
        manager.update_frame(frame_idx=5)

        # quiver should be called again (recreated each frame - matplotlib limitation)
        assert mock_ax.quiver.call_count > quiver_calls_after_init
        # old quiver should have been removed
        assert old_quiver.remove.called

    def test_multiple_update_frames_dont_create_new_artists(
        self,
        mock_ax: MagicMock,
        simple_env: Environment,
        position_overlay_data: OverlayData,
    ):
        """Test calling update_frame many times doesn't create new artists."""
        manager = OverlayArtistManager(
            ax=mock_ax,
            env=simple_env,
            overlay_data=position_overlay_data,
            show_regions=False,
            region_alpha=0.3,
        )

        manager.initialize(frame_idx=0)
        initial_scatter_count = mock_ax.scatter.call_count

        # Update through many frames
        for frame_idx in range(1, 10):
            manager.update_frame(frame_idx)

        # scatter should still not be called again
        assert mock_ax.scatter.call_count == initial_scatter_count


# ============================================================================
# Clear Method Tests
# ============================================================================


class TestOverlayArtistManagerClear:
    """Tests for clear() method behavior."""

    def test_clear_removes_position_artists(
        self,
        mock_ax: MagicMock,
        simple_env: Environment,
        position_overlay_data: OverlayData,
    ):
        """Test clear() removes position marker and calls remove()."""
        manager = OverlayArtistManager(
            ax=mock_ax,
            env=simple_env,
            overlay_data=position_overlay_data,
            show_regions=False,
            region_alpha=0.3,
        )

        manager.initialize(frame_idx=0)
        marker = manager._position_markers[0]

        manager.clear()

        # remove() should be called on marker
        assert marker.remove.called
        # Lists should be cleared
        assert len(manager._position_markers) == 0
        assert len(manager._position_trails) == 0

    def test_clear_removes_bodypart_artists(
        self,
        mock_ax: MagicMock,
        simple_env: Environment,
        bodypart_overlay_data: OverlayData,
    ):
        """Test clear() removes bodypart scatter and skeleton."""
        manager = OverlayArtistManager(
            ax=mock_ax,
            env=simple_env,
            overlay_data=bodypart_overlay_data,
            show_regions=False,
            region_alpha=0.3,
        )

        manager.initialize(frame_idx=0)
        points = manager._bodypart_points[0]

        manager.clear()

        # remove() should be called on points
        assert points.remove.called
        # Lists should be cleared
        assert len(manager._bodypart_points) == 0
        assert len(manager._bodypart_skeletons) == 0

    def test_clear_removes_head_direction_quiver(
        self,
        mock_ax: MagicMock,
        simple_env: Environment,
        head_direction_overlay_data: OverlayData,
    ):
        """Test clear() removes head direction quiver."""
        manager = OverlayArtistManager(
            ax=mock_ax,
            env=simple_env,
            overlay_data=head_direction_overlay_data,
            show_regions=False,
            region_alpha=0.3,
        )

        manager.initialize(frame_idx=0)
        quiver = manager._head_direction_quivers[0]

        manager.clear()

        # remove() should be called on quiver
        assert quiver.remove.called
        # List should be cleared
        assert len(manager._head_direction_quivers) == 0

    def test_clear_resets_initialized_flag(
        self,
        mock_ax: MagicMock,
        simple_env: Environment,
        position_overlay_data: OverlayData,
    ):
        """Test clear() resets _initialized to False."""
        manager = OverlayArtistManager(
            ax=mock_ax,
            env=simple_env,
            overlay_data=position_overlay_data,
            show_regions=False,
            region_alpha=0.3,
        )

        manager.initialize(frame_idx=0)
        assert manager._initialized is True

        manager.clear()

        assert manager._initialized is False

    def test_clear_allows_reinitialize(
        self,
        mock_ax: MagicMock,
        simple_env: Environment,
        position_overlay_data: OverlayData,
    ):
        """Test after clear(), initialize() can be called again."""
        manager = OverlayArtistManager(
            ax=mock_ax,
            env=simple_env,
            overlay_data=position_overlay_data,
            show_regions=False,
            region_alpha=0.3,
        )

        manager.initialize(frame_idx=0)
        manager.clear()

        # Should be able to initialize again
        manager.initialize(frame_idx=5)

        assert manager._initialized is True
        assert len(manager._position_markers) == 1


# ============================================================================
# Edge Cases
# ============================================================================


class TestOverlayArtistManagerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_position_overlay_with_nan_data(
        self, mock_ax: MagicMock, simple_env: Environment
    ):
        """Test manager handles NaN position data gracefully."""
        rng = np.random.default_rng(42)
        n_frames = 10
        positions = rng.random((n_frames, 2)) * 10
        positions[5] = [np.nan, np.nan]  # Frame 5 has NaN

        overlay_data = OverlayData(
            positions=[
                PositionData(data=positions, color="red", size=10.0, trail_length=3)
            ]
        )

        manager = OverlayArtistManager(
            ax=mock_ax,
            env=simple_env,
            overlay_data=overlay_data,
            show_regions=False,
            region_alpha=0.3,
        )

        # Should not raise
        manager.initialize(frame_idx=0)
        manager.update_frame(frame_idx=5)  # NaN frame

        # Marker should be hidden for NaN
        marker = manager._position_markers[0]
        marker.set_visible.assert_called_with(False)

    def test_bodypart_overlay_with_partial_nan_data(
        self, mock_ax: MagicMock, simple_env: Environment
    ):
        """Test manager handles partial NaN bodypart data gracefully."""
        rng = np.random.default_rng(42)
        n_frames = 10
        nose = rng.random((n_frames, 2)) * 10
        tail = rng.random((n_frames, 2)) * 10
        nose[5] = [np.nan, np.nan]  # Nose is NaN at frame 5

        skeleton = Skeleton(
            name="test",
            nodes=("nose", "tail"),
            edges=(("nose", "tail"),),
            edge_color="white",
            edge_width=2.0,
        )

        overlay_data = OverlayData(
            bodypart_sets=[
                BodypartData(
                    bodyparts={"nose": nose, "tail": tail},
                    skeleton=skeleton,
                    colors={"nose": "red", "tail": "blue"},
                )
            ]
        )

        manager = OverlayArtistManager(
            ax=mock_ax,
            env=simple_env,
            overlay_data=overlay_data,
            show_regions=False,
            region_alpha=0.3,
        )

        # Should not raise
        manager.initialize(frame_idx=0)
        manager.update_frame(frame_idx=5)  # Partial NaN frame

        # Should still have artists
        assert len(manager._bodypart_points) == 1

    def test_head_direction_with_nan_angle(
        self, mock_ax: MagicMock, simple_env: Environment
    ):
        """Test manager handles NaN head direction angle gracefully."""
        rng = np.random.default_rng(42)
        n_frames = 10
        angles = rng.random(n_frames) * 2 * np.pi
        angles[5] = np.nan  # NaN angle at frame 5

        overlay_data = OverlayData(
            head_directions=[HeadDirectionData(data=angles, color="yellow", length=5.0)]
        )

        manager = OverlayArtistManager(
            ax=mock_ax,
            env=simple_env,
            overlay_data=overlay_data,
            show_regions=False,
            region_alpha=0.3,
        )

        # Should not raise
        manager.initialize(frame_idx=0)

        # Update to NaN frame - should not raise
        manager.update_frame(frame_idx=5)

        # Quiver should be None for NaN angle
        assert manager._head_direction_quivers[0] is None

    def test_empty_overlay_lists(self, mock_ax: MagicMock, simple_env: Environment):
        """Test manager handles empty overlay lists."""
        overlay_data = OverlayData(
            positions=[],
            bodypart_sets=[],
            head_directions=[],
        )

        manager = OverlayArtistManager(
            ax=mock_ax,
            env=simple_env,
            overlay_data=overlay_data,
            show_regions=False,
            region_alpha=0.3,
        )

        # Should not raise
        manager.initialize(frame_idx=0)
        manager.update_frame(frame_idx=5)

        assert manager._initialized is True
        assert len(manager._position_markers) == 0

    def test_multiple_position_overlays(
        self, mock_ax: MagicMock, simple_env: Environment
    ):
        """Test manager handles multiple position overlays (multi-animal)."""
        rng = np.random.default_rng(42)
        n_frames = 10
        positions1 = rng.random((n_frames, 2)) * 10
        positions2 = rng.random((n_frames, 2)) * 10

        overlay_data = OverlayData(
            positions=[
                PositionData(data=positions1, color="red", size=10.0, trail_length=3),
                PositionData(data=positions2, color="blue", size=10.0, trail_length=3),
            ]
        )

        manager = OverlayArtistManager(
            ax=mock_ax,
            env=simple_env,
            overlay_data=overlay_data,
            show_regions=False,
            region_alpha=0.3,
        )

        manager.initialize(frame_idx=0)

        # Should have 2 position markers
        assert len(manager._position_markers) == 2
        assert len(manager._position_trails) == 2

    def test_position_without_trail(self, mock_ax: MagicMock, simple_env: Environment):
        """Test manager handles position overlay without trail."""
        rng = np.random.default_rng(42)
        n_frames = 10
        positions = rng.random((n_frames, 2)) * 10

        overlay_data = OverlayData(
            positions=[
                PositionData(data=positions, color="red", size=10.0, trail_length=None)
            ]
        )

        manager = OverlayArtistManager(
            ax=mock_ax,
            env=simple_env,
            overlay_data=overlay_data,
            show_regions=False,
            region_alpha=0.3,
        )

        manager.initialize(frame_idx=0)

        # Should have position marker but no trail
        assert len(manager._position_markers) == 1
        assert len(manager._position_trails) == 1
        assert manager._position_trails[0] is None  # No trail LineCollection

    def test_bodypart_overlay_without_skeleton(
        self, mock_ax: MagicMock, simple_env: Environment
    ):
        """Test manager handles bodypart overlay without skeleton."""
        rng = np.random.default_rng(42)
        n_frames = 10
        nose = rng.random((n_frames, 2)) * 10
        tail = rng.random((n_frames, 2)) * 10

        overlay_data = OverlayData(
            bodypart_sets=[
                BodypartData(
                    bodyparts={"nose": nose, "tail": tail},
                    skeleton=None,  # No skeleton
                    colors={"nose": "red", "tail": "blue"},
                )
            ]
        )

        manager = OverlayArtistManager(
            ax=mock_ax,
            env=simple_env,
            overlay_data=overlay_data,
            show_regions=False,
            region_alpha=0.3,
        )

        manager.initialize(frame_idx=0)

        # Should have bodypart points but no skeleton
        assert len(manager._bodypart_points) == 1
        assert len(manager._bodypart_skeletons) == 1
        assert manager._bodypart_skeletons[0] is None  # No skeleton LineCollection


# ============================================================================
# Integration with PersistentFigureRenderer
# ============================================================================


class TestOverlayArtistManagerIntegration:
    """Tests for integration with PersistentFigureRenderer."""

    def test_persistent_renderer_creates_manager_once(
        self, simple_env: Environment, position_overlay_data: OverlayData
    ):
        """Test PersistentFigureRenderer creates manager once per figure."""
        from neurospatial.animation.backends.widget_backend import (
            PersistentFigureRenderer,
        )

        rng = np.random.default_rng(42)
        fields = [rng.random(simple_env.n_bins) for _ in range(10)]

        renderer = PersistentFigureRenderer(
            env=simple_env,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=100,
        )

        try:
            # First render creates manager
            renderer.render(fields[0], frame_idx=0, overlay_data=position_overlay_data)
            first_manager = renderer._overlay_manager

            # Second render should reuse manager
            renderer.render(fields[1], frame_idx=1, overlay_data=position_overlay_data)
            second_manager = renderer._overlay_manager

            # Same manager instance
            assert first_manager is second_manager
        finally:
            renderer.close()

    def test_persistent_renderer_manager_updates_via_update_frame(
        self, simple_env: Environment, position_overlay_data: OverlayData
    ):
        """Test PersistentFigureRenderer uses manager.update_frame() for subsequent renders."""
        from neurospatial.animation.backends.widget_backend import (
            PersistentFigureRenderer,
        )

        rng = np.random.default_rng(42)
        fields = [rng.random(simple_env.n_bins) for _ in range(10)]

        renderer = PersistentFigureRenderer(
            env=simple_env,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=100,
        )

        try:
            # First render
            renderer.render(fields[0], frame_idx=0, overlay_data=position_overlay_data)

            # Spy on update_frame
            original_update = renderer._overlay_manager.update_frame
            update_called = [False]

            def spy_update(frame_idx):
                update_called[0] = True
                return original_update(frame_idx)

            renderer._overlay_manager.update_frame = spy_update

            # Second render should call update_frame
            renderer.render(fields[1], frame_idx=1, overlay_data=position_overlay_data)

            assert update_called[0] is True
        finally:
            renderer.close()

    def test_persistent_renderer_fallback_creates_new_manager(
        self, simple_env: Environment, position_overlay_data: OverlayData
    ):
        """Test fallback path in PersistentFigureRenderer creates new manager after ax.clear()."""
        from neurospatial.animation.backends.widget_backend import (
            PersistentFigureRenderer,
        )

        rng = np.random.default_rng(42)
        fields = [rng.random(simple_env.n_bins) for _ in range(10)]

        renderer = PersistentFigureRenderer(
            env=simple_env,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=100,
        )

        try:
            # First render
            renderer.render(fields[0], frame_idx=0, overlay_data=position_overlay_data)
            first_manager = renderer._overlay_manager
            first_manager_id = id(first_manager)

            # Force fallback by clearing mesh
            renderer._mesh = None

            # Next render triggers fallback (_do_full_rerender)
            renderer.render(fields[1], frame_idx=1, overlay_data=position_overlay_data)
            second_manager = renderer._overlay_manager

            # Should be a new manager instance (axes were cleared)
            assert id(second_manager) != first_manager_id
        finally:
            renderer.close()
