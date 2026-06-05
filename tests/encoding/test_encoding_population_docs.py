"""Doc-snippet and lint tests for population encoding docstrings.

These tests assert that the public API docstrings and error messages teach the
vectorized batch path (compute_spatial_rates) rather than the per-neuron loop
(compute_spatial_rate).
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.encoding.population import population_coverage


class TestPopulationCoverageDocstring:
    """Ensure population_coverage docstring teaches the batch path."""

    def test_no_per_neuron_loop_pattern(self) -> None:
        """Docstring must not contain the old per-neuron for-loop pattern."""
        doc = population_coverage.__doc__
        assert doc is not None, "population_coverage must have a docstring"
        assert "for spikes in spike_times" not in doc, (
            "Docstring must not teach the per-neuron 'for spikes in spike_times' loop"
        )

    def test_batch_function_present_in_docstring(self) -> None:
        """Docstring must reference compute_spatial_rates (plural)."""
        doc = population_coverage.__doc__
        assert doc is not None
        assert "compute_spatial_rates(" in doc, (
            "Docstring must reference compute_spatial_rates (plural)"
        )

    def test_singular_loop_absent_from_docstring(self) -> None:
        """Docstring must not show a per-neuron compute_spatial_rate( call."""
        doc = population_coverage.__doc__
        assert doc is not None
        # Check for the old loop pattern: compute_spatial_rate followed by a
        # list-comprehension iterating over neurons
        old_loop = re.search(
            r"compute_spatial_rate\(.*for\s+\w+\s+in\s+spike_times", doc, re.DOTALL
        )
        assert old_loop is None, (
            "Docstring must not contain a per-neuron compute_spatial_rate( "
            "list-comprehension loop"
        )

    def test_docstring_example_uses_env_first_arg_order(self) -> None:
        """Docstring example must use env-first argument order (canonical v0.6 convention)."""
        doc = population_coverage.__doc__
        assert doc is not None
        assert "population_coverage(env, firing_rates)" in doc, (
            "Docstring example must call population_coverage(env, firing_rates) "
            "with env as the first argument"
        )
        assert "population_coverage(firing_rates, env)" not in doc, (
            "Docstring example must not use the old reversed order "
            "population_coverage(firing_rates, env)"
        )


class TestPopulationCoverageErrorMessage:
    """Ensure the shape-mismatch ValueError message teaches the batch path."""

    def _make_env(self) -> tuple[Environment, int]:
        """Create a small fitted environment and return (env, n_bins)."""
        positions = np.random.default_rng(0).random((200, 2)) * 100
        env = Environment.from_samples(positions, bin_size=10.0)
        return env, env.n_bins

    def test_error_message_contains_batch_function(self) -> None:
        """ValueError for wrong shape must recommend compute_spatial_rates."""
        env, n_bins = self._make_env()
        # Build a firing_rates array with wrong number of bins (n_bins + 1)
        wrong_rates = np.zeros((3, n_bins + 1))
        with pytest.raises(ValueError, match=r"compute_spatial_rates\("):
            population_coverage(env, wrong_rates)

    def test_error_message_no_old_per_neuron_recipe(self) -> None:
        """ValueError must not contain the old np.stack([rate1, rate2, ...]) recipe."""
        env, n_bins = self._make_env()
        wrong_rates = np.zeros((3, n_bins + 1))
        with pytest.raises(ValueError) as exc_info:
            population_coverage(env, wrong_rates)
        msg = str(exc_info.value)
        assert "firing_rates = np.stack([rate1, rate2" not in msg, (
            "Error message must not contain the old np.stack per-neuron recipe"
        )

    def test_error_message_no_singular_with_stack(self) -> None:
        """ValueError must not show the old singular .firing_rate + np.stack pattern."""
        env, n_bins = self._make_env()
        wrong_rates = np.zeros((3, n_bins + 1))
        with pytest.raises(ValueError) as exc_info:
            population_coverage(env, wrong_rates)
        msg = str(exc_info.value)
        # The old recipe ended with: .firing_rate\n  firing_rates = np.stack(...)
        assert ".firing_rate\n  firing_rates = np.stack" not in msg


class TestComputeSpatialRateDocstring:
    """Ensure the singular compute_spatial_rate docstring points to the batch path."""

    def test_batch_pointer_present(self) -> None:
        """compute_spatial_rate docstring must mention compute_spatial_rates."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        doc = compute_spatial_rate.__doc__
        assert doc is not None, "compute_spatial_rate must have a docstring"
        assert "compute_spatial_rates" in doc, (
            "compute_spatial_rate docstring must contain a pointer to "
            "compute_spatial_rates (plural)"
        )
