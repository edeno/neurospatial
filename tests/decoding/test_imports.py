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
        from neurospatial import DecodingResult, decode_position

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
        from neurospatial import (
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
            ShuffleTestResult,
        )

        assert isinstance(DecodingResult, type)
        assert isinstance(IsotonicFitResult, type)
        assert isinstance(LinearFitResult, type)
        assert isinstance(RadonDetectionResult, type)
        assert isinstance(ShuffleTestResult, type)

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

    def test_decoding_subpackage_shuffle_temporal(self):
        """Test that temporal shuffle functions are exported from decoding subpackage."""
        from neurospatial.decoding import (
            shuffle_time_bins,
            shuffle_time_bins_coherent,
        )

        assert callable(shuffle_time_bins)
        assert callable(shuffle_time_bins_coherent)

    def test_decoding_subpackage_shuffle_cell_identity(self):
        """Test that cell identity shuffle functions are exported from decoding subpackage."""
        from neurospatial.decoding import (
            shuffle_cell_identity,
            shuffle_place_fields_circular,
            shuffle_place_fields_circular_2d,
        )

        assert callable(shuffle_cell_identity)
        assert callable(shuffle_place_fields_circular)
        assert callable(shuffle_place_fields_circular_2d)

    def test_decoding_subpackage_shuffle_posterior(self):
        """Test that posterior shuffle functions are exported from decoding subpackage."""
        from neurospatial.decoding import (
            shuffle_posterior_circular,
            shuffle_posterior_weighted_circular,
        )

        assert callable(shuffle_posterior_circular)
        assert callable(shuffle_posterior_weighted_circular)

    def test_decoding_subpackage_shuffle_surrogates(self):
        """Test that surrogate generation functions are exported from decoding subpackage."""
        from neurospatial.decoding import (
            generate_inhomogeneous_poisson_surrogates,
            generate_poisson_surrogates,
        )

        assert callable(generate_inhomogeneous_poisson_surrogates)
        assert callable(generate_poisson_surrogates)

    def test_decoding_subpackage_shuffle_significance(self):
        """Test that significance testing functions are exported from decoding subpackage."""
        from neurospatial.decoding import (
            ShuffleTestResult,
            compute_shuffle_pvalue,
            compute_shuffle_zscore,
        )

        assert callable(compute_shuffle_pvalue)
        assert callable(compute_shuffle_zscore)
        assert isinstance(ShuffleTestResult, type)

    def test_decoding_all_list_completeness(self):
        """Test that __all__ contains all expected exports."""
        import neurospatial.decoding

        expected_exports = [
            # Result containers
            "DecodingResult",
            "IsotonicFitResult",
            "LinearFitResult",
            "RadonDetectionResult",
            "ShuffleTestResult",
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
            # Shuffles - Temporal
            "shuffle_time_bins",
            "shuffle_time_bins_coherent",
            # Shuffles - Cell Identity
            "shuffle_cell_identity",
            "shuffle_place_fields_circular",
            "shuffle_place_fields_circular_2d",
            # Shuffles - Posterior
            "shuffle_posterior_circular",
            "shuffle_posterior_weighted_circular",
            # Shuffles - Surrogates
            "generate_inhomogeneous_poisson_surrogates",
            "generate_poisson_surrogates",
            # Shuffles - Significance
            "compute_shuffle_pvalue",
            "compute_shuffle_zscore",
        ]

        actual_all = set(neurospatial.decoding.__all__)
        expected_set = set(expected_exports)

        # Check all expected are present
        missing = expected_set - actual_all
        assert not missing, f"Missing from __all__: {missing}"

    def test_main_package_all_includes_decoding_exports(self):
        """Test that main neurospatial.__all__ includes decoding exports."""
        import neurospatial

        # These should be in the main __all__
        expected_in_main = [
            "DecodingResult",
            "decode_position",
            "decoding_error",
            "median_decoding_error",
        ]

        actual_all = set(neurospatial.__all__)
        for name in expected_in_main:
            assert name in actual_all, f"'{name}' not in neurospatial.__all__"
