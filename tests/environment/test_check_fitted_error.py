"""Tests for enhanced @check_fitted error messages with usage examples.

This test suite ensures that the @check_fitted decorator provides helpful
error messages with concrete examples of correct usage when users try to
access methods on an uninitialized Environment.
"""

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.environment.decorators import EnvironmentNotFittedError


class TestCheckFittedErrorMessage:
    """Test that @check_fitted errors include helpful examples."""

    def test_error_mentions_factory_methods(self):
        """Test that error message mentions factory methods."""
        # Create an uninitialized Environment
        env = Environment.__new__(Environment)
        points = np.array([[5, 5]])

        with pytest.raises(RuntimeError) as exc_info:
            env.bin_at(points)

        error_msg = str(exc_info.value)
        # Should mention factory method(s)
        assert "from_samples" in error_msg or "factory method" in error_msg.lower()

    def test_error_shows_example_of_correct_usage(self):
        """Test that error message shows an example of correct usage."""
        env = Environment.__new__(Environment)
        points = np.array([[5, 5]])

        with pytest.raises(RuntimeError) as exc_info:
            env.bin_at(points)

        error_msg = str(exc_info.value)
        # Should show an example with Environment.from_
        assert "Environment.from_" in error_msg
        # Should show code-like example
        assert "=" in error_msg or "env" in error_msg.lower()

    def test_error_includes_method_name(self):
        """Test that error message includes the method that was called."""
        env = Environment.__new__(Environment)
        points = np.array([[5, 5]])

        with pytest.raises(RuntimeError) as exc_info:
            env.bin_at(points)

        error_msg = str(exc_info.value)
        # Should mention the method being called
        assert "bin_at" in error_msg

    def test_error_mentions_initialization_requirement(self):
        """Test that error message mentions initialization requirement."""
        env = Environment.__new__(Environment)
        points = np.array([[5, 5]])

        with pytest.raises(RuntimeError) as exc_info:
            env.bin_at(points)

        error_msg = str(exc_info.value)
        # Should explain what's wrong
        assert (
            "initialized" in error_msg.lower()
            or "created" in error_msg.lower()
            or "fitted" in error_msg.lower()
        )

    def test_error_works_for_different_methods(self):
        """Test that error works for various @check_fitted methods."""
        env = Environment.__new__(Environment)

        # Test with different methods
        methods_to_test = [
            lambda: env.n_bins,
            lambda: env.bin_at(np.array([[5, 5]])),
            lambda: env.contains(np.array([[5, 5]])),
            lambda: env.neighbors(0),
        ]

        for method_call in methods_to_test:
            with pytest.raises(RuntimeError) as exc_info:
                method_call()

            error_msg = str(exc_info.value)
            # All should mention factory methods or initialization
            assert (
                "from_" in error_msg
                or "factory" in error_msg.lower()
                or "initialized" in error_msg.lower()
            )

    def test_properly_initialized_environment_does_not_raise(self):
        """Test that properly initialized Environment does not raise error."""
        # Create a proper Environment using factory method
        data = np.array([[0, 0], [10, 10], [5, 5]])
        env = Environment.from_samples(data, bin_size=2.0)

        # These should all work without error
        assert env.n_bins > 0
        bin_indices = env.bin_at(np.array([[5, 5]]))
        assert bin_indices is not None
        contains = env.contains(np.array([[5, 5]]))
        assert contains is not None

    def test_error_message_is_helpful_and_actionable(self):
        """Test that error message provides actionable guidance."""
        env = Environment.__new__(Environment)

        with pytest.raises(RuntimeError) as exc_info:
            _ = env.n_bins

        error_msg = str(exc_info.value)
        # Should be sufficiently detailed (not just "not initialized")
        assert len(error_msg) > 50
        # Should mention how to fix
        assert (
            "Environment.from_" in error_msg
            or "Use" in error_msg
            or "create" in error_msg.lower()
        )


class TestCheckFittedFormatConsistency:
    """Test that @check_fitted errors are consistent across methods."""

    def test_all_check_fitted_errors_have_consistent_format(self):
        """Verify all @check_fitted decorated methods produce consistent errors."""
        env = Environment.__new__(Environment)

        # Sample of methods with @check_fitted
        methods_and_calls = [
            ("n_bins", lambda: env.n_bins),
            ("bin_at", lambda: env.bin_at(np.array([[0, 0]]))),
            ("contains", lambda: env.contains(np.array([[0, 0]]))),
            ("neighbors", lambda: env.neighbors(0)),
            ("bin_sizes", lambda: env.bin_sizes()),
        ]

        error_messages = []
        for method_name, method_call in methods_and_calls:
            with pytest.raises(RuntimeError) as exc_info:
                method_call()
            error_messages.append((method_name, str(exc_info.value)))

        # All should contain the method name they're called from
        for method_name, error_msg in error_messages:
            assert method_name in error_msg.lower() or "Environment." in error_msg

        # All should mention initialization or factory methods
        for method_name, error_msg in error_messages:
            has_init_mention = any(
                keyword in error_msg.lower()
                for keyword in ["initialized", "factory", "from_", "created"]
            )
            assert has_init_mention, (
                f"Error for {method_name} lacks initialization guidance"
            )


class TestEnvironmentNotFittedErrorType:
    """Test custom EnvironmentNotFittedError exception class."""

    def test_raises_specific_exception_type(self):
        """Test that @check_fitted raises EnvironmentNotFittedError."""
        env = Environment.__new__(Environment)
        points = np.array([[5, 5]])

        with pytest.raises(EnvironmentNotFittedError) as exc_info:
            env.bin_at(points)

        # Verify it's the custom exception
        assert isinstance(exc_info.value, EnvironmentNotFittedError)
        # Verify it's still a RuntimeError (backward compatibility)
        assert isinstance(exc_info.value, RuntimeError)

    def test_exception_has_useful_attributes(self):
        """Test that EnvironmentNotFittedError has class_name and method_name."""
        env = Environment.__new__(Environment)
        points = np.array([[5, 5]])

        with pytest.raises(EnvironmentNotFittedError) as exc_info:
            env.bin_at(points)

        exception = exc_info.value
        assert hasattr(exception, "class_name")
        assert hasattr(exception, "method_name")
        assert hasattr(exception, "error_code")
        assert exception.class_name == "Environment"
        assert exception.method_name == "bin_at"
        assert exception.error_code == "E1004"

    def test_can_catch_as_runtime_error(self):
        """Test backward compatibility - can catch as RuntimeError."""
        env = Environment.__new__(Environment)

        # Should be catchable as RuntimeError
        with pytest.raises(RuntimeError):
            _ = env.n_bins

    def test_can_catch_specifically(self):
        """Test that users can catch EnvironmentNotFittedError specifically."""
        env = Environment.__new__(Environment)

        # Should be catchable specifically
        try:
            _ = env.n_bins
        except EnvironmentNotFittedError as e:
            # This is the expected path
            assert "Environment" in str(e)
            assert "from_samples" in str(e)
        except Exception:
            pytest.fail("Should have raised EnvironmentNotFittedError")

    def test_exception_message_format(self):
        """Test that exception message has expected format."""
        exception = EnvironmentNotFittedError("TestClass", "test_method")

        message = str(exception)
        assert "[E1004]" in message
        assert "TestClass.test_method()" in message
        assert "from_samples" in message
        assert "factory method" in message

    def test_custom_error_code(self):
        """Test that custom error code can be specified."""
        exception = EnvironmentNotFittedError("TestClass", "method", error_code="E9999")

        message = str(exception)
        assert "[E9999]" in message
        assert exception.error_code == "E9999"

    def test_all_check_fitted_methods_raise_custom_exception(self):
        """Test that all @check_fitted methods raise EnvironmentNotFittedError."""
        env = Environment.__new__(Environment)

        # Test various methods
        test_cases = [
            lambda: env.n_bins,
            lambda: env.bin_at(np.array([[0, 0]])),
            lambda: env.contains(np.array([[0, 0]])),
            lambda: env.neighbors(0),
            lambda: env.bin_sizes(),
        ]

        for test_func in test_cases:
            with pytest.raises(EnvironmentNotFittedError):
                test_func()
