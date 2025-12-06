"""Tests for decoding subpackage re-exports from stats module.

Per PLAN.md, the decoding subpackage should re-export key shuffle and surrogate
functions from stats/ for discoverability in decoding workflows. Users should be
able to import these from either the canonical location (stats/) or the convenient
location (decoding/).

Canonical Location             | Re-export Location
-------------------------------|-------------------------------
stats.shuffle                  | decoding
stats.surrogates               | decoding
"""


class TestDecodingReexportsFromStatsImports:
    """Test that decoding re-exports from stats are importable."""

    def test_shuffle_time_bins_importable_from_decoding(self):
        """Test shuffle_time_bins is re-exported from decoding."""
        from neurospatial.decoding import shuffle_time_bins

        assert callable(shuffle_time_bins)

    def test_shuffle_cell_identity_importable_from_decoding(self):
        """Test shuffle_cell_identity is re-exported from decoding."""
        from neurospatial.decoding import shuffle_cell_identity

        assert callable(shuffle_cell_identity)

    def test_compute_shuffle_pvalue_importable_from_decoding(self):
        """Test compute_shuffle_pvalue is re-exported from decoding."""
        from neurospatial.decoding import compute_shuffle_pvalue

        assert callable(compute_shuffle_pvalue)

    def test_shuffle_test_result_importable_from_decoding(self):
        """Test ShuffleTestResult is re-exported from decoding."""
        from neurospatial.decoding import ShuffleTestResult

        assert isinstance(ShuffleTestResult, type)

    def test_generate_poisson_surrogates_importable_from_decoding(self):
        """Test generate_poisson_surrogates is re-exported from decoding."""
        from neurospatial.decoding import generate_poisson_surrogates

        assert callable(generate_poisson_surrogates)


class TestDecodingReexportsAreIdentical:
    """Test that re-exported symbols are identical to canonical location."""

    def test_shuffle_time_bins_is_same_function(self):
        """Test shuffle_time_bins in decoding is same as in stats."""
        from neurospatial.decoding import shuffle_time_bins as decoding_func
        from neurospatial.stats.shuffle import shuffle_time_bins as stats_func

        assert decoding_func is stats_func

    def test_shuffle_cell_identity_is_same_function(self):
        """Test shuffle_cell_identity in decoding is same as in stats."""
        from neurospatial.decoding import shuffle_cell_identity as decoding_func
        from neurospatial.stats.shuffle import shuffle_cell_identity as stats_func

        assert decoding_func is stats_func

    def test_compute_shuffle_pvalue_is_same_function(self):
        """Test compute_shuffle_pvalue in decoding is same as in stats."""
        from neurospatial.decoding import compute_shuffle_pvalue as decoding_func
        from neurospatial.stats.shuffle import compute_shuffle_pvalue as stats_func

        assert decoding_func is stats_func

    def test_shuffle_test_result_is_same_class(self):
        """Test ShuffleTestResult in decoding is same as in stats."""
        from neurospatial.decoding import ShuffleTestResult as DecodingShuffleResult
        from neurospatial.stats.shuffle import ShuffleTestResult as StatsShuffleResult

        assert DecodingShuffleResult is StatsShuffleResult

    def test_generate_poisson_surrogates_is_same_function(self):
        """Test generate_poisson_surrogates in decoding is same as in stats."""
        from neurospatial.decoding import generate_poisson_surrogates as decoding_func
        from neurospatial.stats.surrogates import (
            generate_poisson_surrogates as stats_func,
        )

        assert decoding_func is stats_func


class TestDecodingAllIncludesReexports:
    """Test that decoding.__all__ includes the re-exported symbols."""

    def test_shuffle_time_bins_in_all(self):
        """Test shuffle_time_bins is in decoding.__all__."""
        import neurospatial.decoding

        assert "shuffle_time_bins" in neurospatial.decoding.__all__

    def test_shuffle_cell_identity_in_all(self):
        """Test shuffle_cell_identity is in decoding.__all__."""
        import neurospatial.decoding

        assert "shuffle_cell_identity" in neurospatial.decoding.__all__

    def test_compute_shuffle_pvalue_in_all(self):
        """Test compute_shuffle_pvalue is in decoding.__all__."""
        import neurospatial.decoding

        assert "compute_shuffle_pvalue" in neurospatial.decoding.__all__

    def test_shuffle_test_result_in_all(self):
        """Test ShuffleTestResult is in decoding.__all__."""
        import neurospatial.decoding

        assert "ShuffleTestResult" in neurospatial.decoding.__all__

    def test_generate_poisson_surrogates_in_all(self):
        """Test generate_poisson_surrogates is in decoding.__all__."""
        import neurospatial.decoding

        assert "generate_poisson_surrogates" in neurospatial.decoding.__all__


class TestDecodingReexportsModuleStructure:
    """Test module structure for re-exports."""

    def test_reexports_are_documented_in_docstring(self):
        """Test that decoding module docstring mentions re-exports."""
        import neurospatial.decoding

        docstring = neurospatial.decoding.__doc__ or ""
        # Should mention shuffle in context of decoding workflows
        assert "shuffle" in docstring.lower() or "Shuffle" in docstring

    def test_all_reexports_list(self):
        """Test that all expected re-exports are present."""
        import neurospatial.decoding

        expected_reexports = [
            "shuffle_time_bins",
            "shuffle_cell_identity",
            "compute_shuffle_pvalue",
            "ShuffleTestResult",
            "generate_poisson_surrogates",
        ]

        for name in expected_reexports:
            assert hasattr(neurospatial.decoding, name), f"Missing re-export: {name}"
            assert name in neurospatial.decoding.__all__, f"Not in __all__: {name}"
