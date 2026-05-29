"""Tests for the EnvironmentNotFittedError exception raised by @check_fitted.

Coverage:
- The exception is raised on @check_fitted methods when the env was built
  with ``Environment.__new__`` and never fitted.
- The exception carries ``class_name`` / ``method_name`` / ``error_code``
  attributes and formats them into a stable message shape.
- The free-function construction form (``is_function=True``, M3.1) emits a
  message without a class qualifier and preserves qualified names.
"""

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.environment.decorators import EnvironmentNotFittedError


class TestCheckFittedRaises:
    """The decorator should raise EnvironmentNotFittedError on unfitted envs."""

    def test_raises_specific_exception_type(self):
        """@check_fitted raises EnvironmentNotFittedError, a RuntimeError subclass."""
        env = Environment.__new__(Environment)
        points = np.array([[5, 5]])

        with pytest.raises(EnvironmentNotFittedError) as exc_info:
            env.bin_at(points)

        assert isinstance(exc_info.value, EnvironmentNotFittedError)

    def test_exception_has_useful_attributes(self):
        """EnvironmentNotFittedError exposes class_name, method_name, error_code."""
        env = Environment.__new__(Environment)
        points = np.array([[5, 5]])

        with pytest.raises(EnvironmentNotFittedError) as exc_info:
            env.bin_at(points)

        exc = exc_info.value
        assert exc.class_name == "Environment"
        assert exc.method_name == "bin_at"
        assert exc.error_code == "E1004"

    def test_can_catch_specifically(self):
        """Users can catch EnvironmentNotFittedError on its own."""
        env = Environment.__new__(Environment)
        with pytest.raises(EnvironmentNotFittedError) as exc_info:
            _ = env.n_bins
        message = str(exc_info.value)
        assert "Environment" in message
        assert "from_samples" in message

    def test_all_check_fitted_methods_raise_custom_exception(self):
        """A representative spread of @check_fitted methods all raise the exception."""
        env = Environment.__new__(Environment)
        cases = [
            lambda: env.n_bins,
            lambda: env.bin_at(np.array([[0, 0]])),
            lambda: env.contains(np.array([[0, 0]])),
            lambda: env.neighbors(0),
            lambda: env.bin_sizes(),
        ]
        for call in cases:
            with pytest.raises(EnvironmentNotFittedError):
                call()

    def test_public_methods_name_themselves_in_fitted_error(self):
        """Methods guarded only indirectly used to name the wrong method.

        ``distance_between``, ``distance_to``, ``reachable_from``, ``subset``,
        ``apply_transform``, and ``region_membership`` previously lacked their
        own ``@check_fitted`` and relied on an internal ``n_bins`` / ``bin_at``
        call to fail. The resulting ``EnvironmentNotFittedError`` then named
        the internal helper (e.g. ``bin_at``) instead of the public method the
        user actually called. Each should now name itself.
        """
        from neurospatial.ops.transforms import translate

        env = Environment.__new__(Environment)

        cases = {
            "distance_between": lambda: env.distance_between(
                np.array([0.0, 0.0]), np.array([1.0, 1.0])
            ),
            "distance_to": lambda: env.distance_to([0], metric="geodesic"),
            "reachable_from": lambda: env.reachable_from(0, radius=None),
            "subset": lambda: env.subset(bins=np.array([True])),
            "apply_transform": lambda: env.apply_transform(translate(1.0, 1.0)),
            "region_membership": lambda: env.region_membership(),
        }

        for method_name, call in cases.items():
            with pytest.raises(EnvironmentNotFittedError) as exc_info:
                call()
            assert exc_info.value.method_name == method_name, (
                f"{method_name}() raised an error naming "
                f"{exc_info.value.method_name!r} instead of itself"
            )

    def test_properly_initialized_environment_does_not_raise(self):
        """Happy path: a factory-built env runs the same methods without raising."""
        data = np.array([[0, 0], [10, 10], [5, 5]])
        env = Environment.from_samples(data, bin_size=2.0)
        assert env.n_bins > 0
        # bin_at / contains have well-defined return shapes; just exercise them.
        env.bin_at(np.array([[5, 5]]))
        env.contains(np.array([[5, 5]]))


class TestExceptionMessageFormat:
    """The exception formats its arguments into a stable message shape."""

    def test_exception_message_format(self):
        """Bound-method form: ``[E1004] Class.method() ... from_samples ... factory method``."""
        message = str(EnvironmentNotFittedError("TestClass", "test_method"))
        assert "[E1004]" in message
        assert "TestClass.test_method()" in message
        assert "from_samples" in message
        assert "factory method" in message

    def test_custom_error_code(self):
        """A custom error_code is reflected in both the attribute and the message."""
        exc = EnvironmentNotFittedError("TestClass", "method", error_code="E9999")
        assert exc.error_code == "E9999"
        assert "[E9999]" in str(exc)


class TestEnvironmentNotFittedErrorFunctionForm:
    """M3.1 free-function construction form."""

    def test_function_form_message_omits_class_qualifier(self):
        exc = EnvironmentNotFittedError("path_progress", is_function=True)
        message = str(exc)
        assert "[E1004]" in message
        assert "path_progress()" in message
        # The bound-method form qualifies with "Class.method()"; the free
        # function form must not.
        assert "." not in message.split("path_progress()")[0].split("] ")[1]
        assert "factory method" in message

    def test_function_form_attributes(self):
        exc = EnvironmentNotFittedError("path_progress", is_function=True)
        assert exc.is_function is True
        assert exc.class_name is None
        assert exc.method_name == "path_progress"
        assert exc.error_code == "E1004"

    def test_function_form_supports_qualified_names(self):
        exc = EnvironmentNotFittedError(
            "neurospatial.behavior.navigation.path_progress",
            is_function=True,
        )
        assert "neurospatial.behavior.navigation.path_progress()" in str(exc)
        assert exc.method_name == "neurospatial.behavior.navigation.path_progress"

    def test_function_form_custom_error_code(self):
        exc = EnvironmentNotFittedError("fn", is_function=True, error_code="E9999")
        assert "[E9999]" in str(exc)

    def test_bound_method_form_still_requires_method_name(self):
        with pytest.raises(TypeError, match="method_name"):
            EnvironmentNotFittedError("Environment")  # missing method_name

    def test_bound_method_form_unchanged(self):
        """Existing two-arg construction still produces the qualified message."""
        exc = EnvironmentNotFittedError("Environment", "bin_at")
        assert exc.is_function is False
        assert exc.class_name == "Environment"
        assert exc.method_name == "bin_at"
        assert "Environment.bin_at()" in str(exc)
