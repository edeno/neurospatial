"""Tests for decoding subpackage imports.

These tests verify that the decoding subpackage structure is correctly set up
and all expected public APIs are importable.
"""


class TestPackageImports:
    """Test that package structure is correct."""

    def test_decoding_subpackage_imports(self):
        """Test that decoding subpackage can be imported."""
        import neurospatial.decoding

        # Verify it's a module
        assert hasattr(neurospatial.decoding, "__all__")

    def test_decode_position_importable(self):
        """Test that decode_position can be imported from subpackage."""
        from neurospatial.decoding import decode_position

        assert callable(decode_position)

    def test_decoding_result_importable(self):
        """Test that DecodingResult can be imported from subpackage."""
        from neurospatial.decoding import DecodingResult

        # DecodingResult is a class
        assert isinstance(DecodingResult, type)

    def test_top_level_imports(self):
        """Test that main exports are available from neurospatial."""
        from neurospatial.decoding import DecodingResult, decode_position

        assert callable(decode_position)
        assert isinstance(DecodingResult, type)


class TestNormalizeToPosteriorImport:
    """Test that normalize_to_posterior can be imported."""

    def test_normalize_to_posterior_importable(self):
        """Test that normalize_to_posterior can be imported from subpackage."""
        from neurospatial.decoding import normalize_to_posterior

        assert callable(normalize_to_posterior)


class TestPublicAPICompleteness:
    """Test that all public API exports are available.

    These tests verify that the decoding subpackage and main neurospatial
    package expose all documented public APIs per PLAN.md.
    """

    def test_top_level_decoding_exports(self):
        """Test that key decoding functions are available from main package.

        Per PLAN.md, these should be exported at the top level:
        - DecodingResult
        - decode_position
        - decoding_error
        - median_decoding_error
        """
        from neurospatial.decoding import (
            DecodingResult,
            decode_position,
            decoding_error,
            median_decoding_error,
        )

        assert isinstance(DecodingResult, type)
        assert callable(decode_position)
        assert callable(decoding_error)
        assert callable(median_decoding_error)

    def test_decoding_subpackage_result_containers(self):
        """Test that result container classes are exported from decoding subpackage."""
        from neurospatial.decoding import (
            DecodingResult,
            IsotonicFitResult,
            LinearFitResult,
            RadonDetectionResult,
        )

        assert isinstance(DecodingResult, type)
        assert isinstance(IsotonicFitResult, type)
        assert isinstance(LinearFitResult, type)
        assert isinstance(RadonDetectionResult, type)
        # NOTE: ShuffleTestResult moved to neurospatial.stats.shuffle

    def test_decoding_subpackage_likelihood_functions(self):
        """Test that likelihood functions are exported from decoding subpackage."""
        from neurospatial.decoding import (
            log_poisson_likelihood,
            poisson_likelihood,
        )

        assert callable(log_poisson_likelihood)
        assert callable(poisson_likelihood)

    def test_decoding_subpackage_posterior_functions(self):
        """Test that posterior functions are exported from decoding subpackage."""
        from neurospatial.decoding import (
            decode_position,
            normalize_to_posterior,
        )

        assert callable(decode_position)
        assert callable(normalize_to_posterior)

    def test_decoding_subpackage_estimate_functions(self):
        """Test that estimate functions are exported from decoding subpackage."""
        from neurospatial.decoding import (
            credible_region,
            entropy,
            map_estimate,
            map_position,
            mean_position,
        )

        assert callable(credible_region)
        assert callable(entropy)
        assert callable(map_estimate)
        assert callable(map_position)
        assert callable(mean_position)

    def test_decoding_subpackage_trajectory_functions(self):
        """Test that trajectory functions are exported from decoding subpackage."""
        from neurospatial.decoding import (
            detect_trajectory_radon,
            fit_isotonic_trajectory,
            fit_linear_trajectory,
        )

        assert callable(detect_trajectory_radon)
        assert callable(fit_isotonic_trajectory)
        assert callable(fit_linear_trajectory)

    def test_decoding_subpackage_metrics_functions(self):
        """Test that metrics functions are exported from decoding subpackage."""
        from neurospatial.decoding import (
            confusion_matrix,
            decoding_correlation,
            decoding_error,
            median_decoding_error,
        )

        assert callable(confusion_matrix)
        assert callable(decoding_correlation)
        assert callable(decoding_error)
        assert callable(median_decoding_error)

    # NOTE: Shuffle functions have been moved to neurospatial.stats.shuffle
    # See tests/stats/test_stats_shuffle.py for shuffle function tests

    def test_decoding_all_list_completeness(self):
        """Test that __all__ contains all expected exports."""
        import neurospatial.decoding

        expected_exports = [
            # Result containers
            "AssemblyDetectionResult",
            "AssemblyPattern",
            "DecodingResult",
            "ExplainedVarianceResult",
            "IsotonicFitResult",
            "LinearFitResult",
            "RadonDetectionResult",
            "ShuffleTestResult",
            # Cell assembly detection
            "assembly_activation",
            "detect_assemblies",
            "explained_variance_reactivation",
            "marchenko_pastur_threshold",
            "pairwise_correlations",
            "reactivation_strength",
            # Main entry point
            "decode_position",
            # Likelihood
            "log_poisson_likelihood",
            "poisson_likelihood",
            # Posterior
            "normalize_to_posterior",
            # Estimates
            "credible_region",
            "entropy",
            "map_estimate",
            "map_position",
            "mean_position",
            # Trajectory
            "detect_trajectory_radon",
            "fit_isotonic_trajectory",
            "fit_linear_trajectory",
            # Metrics
            "confusion_matrix",
            "decoding_correlation",
            "decoding_error",
            "median_decoding_error",
            # Re-exported from stats (for discoverability in decoding workflows)
            "compute_shuffle_pvalue",
            "generate_poisson_surrogates",
            "shuffle_cell_identity",
            "shuffle_time_bins",
        ]

        actual_all = set(neurospatial.decoding.__all__)
        expected_set = set(expected_exports)

        # Check all expected are present
        missing = expected_set - actual_all
        assert not missing, f"Missing from __all__: {missing}"

    def test_main_package_sparse_exports(self):
        """Test that main neurospatial.__all__ has only sparse exports.

        Per Milestone 9, the top-level neurospatial package now only exports
        5 core classes. Decoding exports must be imported from the submodule.
        """
        import neurospatial

        # Main __all__ should only have 5 core classes
        expected_in_main = {
            "Environment",
            "EnvironmentNotFittedError",
            "Region",
            "Regions",
            "CompositeEnvironment",
        }

        actual_all = set(neurospatial.__all__)
        assert actual_all == expected_in_main, (
            f"Expected only core classes in main __all__, got: {actual_all}"
        )

        # Decoding exports should be in decoding submodule, not main
        expected_in_decoding = [
            "DecodingResult",
            "decode_position",
            "decoding_error",
            "median_decoding_error",
        ]

        for name in expected_in_decoding:
            assert name not in actual_all, (
                f"'{name}' should not be in neurospatial.__all__, "
                f"use neurospatial.decoding instead"
            )
