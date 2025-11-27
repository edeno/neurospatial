"""Tests for annotation state management (no napari required)."""

from neurospatial.annotation._state import AnnotationModeState, make_unique_name


class TestMakeUniqueName:
    """Tests for the make_unique_name utility function."""

    def test_unique_name_when_not_exists(self):
        """Return base_name when it doesn't exist."""
        result = make_unique_name("arena", [])
        assert result == "arena"

    def test_unique_name_with_existing(self):
        """Return arena_2 when arena exists."""
        result = make_unique_name("arena", ["arena"])
        assert result == "arena_2"

    def test_unique_name_with_multiple_existing(self):
        """Return arena_3 when arena and arena_2 exist."""
        result = make_unique_name("arena", ["arena", "arena_2"])
        assert result == "arena_3"

    def test_unique_name_preserves_gaps(self):
        """Return arena_2 when arena exists but arena_2 doesn't."""
        result = make_unique_name("arena", ["arena", "arena_3"])
        assert result == "arena_2"

    def test_unique_name_with_different_names(self):
        """Return base_name when only different names exist."""
        result = make_unique_name("region_1", ["arena", "hole_1"])
        assert result == "region_1"


class TestAnnotationModeState:
    """Tests for AnnotationModeState dataclass."""

    def test_initial_state(self):
        """Create state with initial role."""
        state = AnnotationModeState(region_type="environment")

        assert state.region_type == "environment"
        assert state.environment_count == 0
        assert state.hole_count == 0
        assert state.region_count == 0

    def test_cycle_region_type_environment_to_hole(self):
        """Cycle from environment to hole."""
        state = AnnotationModeState(region_type="environment")

        state.cycle_region_type()

        assert state.region_type == "hole"

    def test_cycle_region_type_hole_to_region(self):
        """Cycle from hole to region."""
        state = AnnotationModeState(region_type="hole")

        state.cycle_region_type()

        assert state.region_type == "region"

    def test_cycle_region_type_region_to_environment(self):
        """Cycle from region back to environment."""
        state = AnnotationModeState(region_type="region")

        state.cycle_region_type()

        assert state.region_type == "environment"

    def test_full_cycle(self):
        """Complete cycle returns to original role."""
        state = AnnotationModeState(region_type="environment")

        state.cycle_region_type()  # hole
        state.cycle_region_type()  # region
        state.cycle_region_type()  # environment

        assert state.region_type == "environment"

    def test_default_name_for_environment(self):
        """Default name for environment is 'arena'."""
        state = AnnotationModeState(region_type="environment")

        assert state.default_name() == "arena"

    def test_default_name_for_hole(self):
        """Default name for hole is empty (auto-named on creation)."""
        state = AnnotationModeState(region_type="hole")

        assert state.default_name() == ""

    def test_default_name_for_region(self):
        """Default name for region is empty (user provides name)."""
        state = AnnotationModeState(region_type="region")

        assert state.default_name() == ""

    def test_generate_auto_name_environment(self):
        """Auto-generate name for environment."""
        state = AnnotationModeState(region_type="environment")

        name = state.generate_auto_name([])

        assert name == "arena"

    def test_generate_auto_name_hole(self):
        """Auto-generate incremented name for hole."""
        state = AnnotationModeState(region_type="hole", hole_count=0)

        name = state.generate_auto_name([])

        assert name == "hole_1"

    def test_generate_auto_name_hole_increments(self):
        """Auto-generate incremented name based on count."""
        state = AnnotationModeState(region_type="hole", hole_count=2)

        name = state.generate_auto_name([])

        assert name == "hole_3"

    def test_generate_auto_name_region(self):
        """Auto-generate incremented name for region."""
        state = AnnotationModeState(region_type="region", region_count=0)

        name = state.generate_auto_name([])

        assert name == "region_1"

    def test_generate_auto_name_avoids_duplicates(self):
        """Auto-generated name avoids existing names."""
        state = AnnotationModeState(region_type="region", region_count=0)

        name = state.generate_auto_name(["region_1"])

        assert name == "region_1_2"

    def test_record_shape_added_environment(self):
        """Recording environment shape increments environment_count."""
        state = AnnotationModeState(region_type="environment")

        state.record_shape_added("environment")

        assert state.environment_count == 1
        assert state.hole_count == 0
        assert state.region_count == 0

    def test_record_shape_added_hole(self):
        """Recording hole shape increments hole_count."""
        state = AnnotationModeState(region_type="hole")

        state.record_shape_added("hole")

        assert state.hole_count == 1

    def test_record_shape_added_region(self):
        """Recording region shape increments region_count."""
        state = AnnotationModeState(region_type="region")

        state.record_shape_added("region")

        assert state.region_count == 1

    def test_sync_counts_from_region_types(self):
        """Sync counts from list of roles."""
        state = AnnotationModeState(region_type="environment")

        state.sync_counts_from_region_types(["environment", "hole", "hole", "region"])

        assert state.environment_count == 1
        assert state.hole_count == 2
        assert state.region_count == 1

    def test_sync_counts_from_empty_roles(self):
        """Sync from empty list resets counts."""
        state = AnnotationModeState(
            region_type="environment",
            environment_count=5,
            hole_count=3,
            region_count=2,
        )

        state.sync_counts_from_region_types([])

        assert state.environment_count == 0
        assert state.hole_count == 0
        assert state.region_count == 0

    def test_status_text(self):
        """Generate status text from counts."""
        state = AnnotationModeState(
            region_type="environment",
            environment_count=1,
            hole_count=2,
            region_count=3,
        )

        text = state.status_text()

        assert text == "Annotations: 1 environment, 2 holes, 3 regions"

    def test_status_text_zeros(self):
        """Status text with all zeros."""
        state = AnnotationModeState(region_type="environment")

        text = state.status_text()

        assert text == "Annotations: 0 environment, 0 holes, 0 regions"

    def test_has_environment_false(self):
        """has_environment False when no environment drawn."""
        state = AnnotationModeState(region_type="environment")

        assert state.has_environment is False

    def test_has_environment_true(self):
        """has_environment True after environment drawn."""
        state = AnnotationModeState(region_type="environment", environment_count=1)

        assert state.has_environment is True


class TestAnnotationModeStateIntegration:
    """Integration tests simulating annotation workflows."""

    def test_typical_workflow(self):
        """Simulate typical annotation workflow."""
        state = AnnotationModeState(region_type="environment")

        # Draw environment
        assert state.region_type == "environment"
        state.record_shape_added("environment")
        assert state.has_environment

        # Switch to hole mode
        state.cycle_region_type()
        assert state.region_type == "hole"

        # Draw two holes
        state.record_shape_added("hole")
        state.record_shape_added("hole")
        assert state.hole_count == 2

        # Switch to region mode
        state.cycle_region_type()
        assert state.region_type == "region"

        # Draw regions
        state.record_shape_added("region")
        state.record_shape_added("region")
        state.record_shape_added("region")

        # Verify final state
        assert state.environment_count == 1
        assert state.hole_count == 2
        assert state.region_count == 3
        assert state.status_text() == "Annotations: 1 environment, 2 holes, 3 regions"

    def test_region_only_workflow(self):
        """Simulate region-only annotation (no environment)."""
        state = AnnotationModeState(region_type="region")

        # Draw multiple regions
        existing = []
        for _ in range(3):
            name = state.generate_auto_name(existing)
            existing.append(name)
            state.record_shape_added("region")

        assert state.environment_count == 0
        assert state.region_count == 3
        assert existing == ["region_1", "region_2", "region_3"]
