"""The shared uniformity check keeps its error rate on a grid and refuses misuse."""

import re

import numpy as np
import pytest
from calibration_helpers import assert_pvalues_uniform, assert_rejections_at_rate


def _on_grid(uniform, n_shuffles):
    """Round values in (0, 1] up to the Monte Carlo grid; ``None`` leaves them."""
    if n_shuffles is None:
        return uniform
    return np.ceil(uniform * (n_shuffles + 1)) / (n_shuffles + 1)


def _balanced(n_shuffles, n_cases):
    """``n_cases`` p-values spread evenly over the grid of ``n_shuffles``."""
    return (np.arange(n_cases) % (n_shuffles + 1) + 1) / (n_shuffles + 1)


def _fails(p_values):
    """Whether the uniformity check fails ``p_values``."""
    try:
        assert_pvalues_uniform(p_values)
    except AssertionError:
        return True
    return False


@pytest.mark.parametrize("n_shuffles", [39, 199, None])
def test_exact_p_values_fail_at_most_at_the_level(n_shuffles):
    """Exact p-values fail 0.1% of the time at most, on a grid or continuous.

    500 samples of 1000 p-values. A Kolmogorov-Smirnov test against the
    continuous uniform fails 4.6% of the 39-shuffle samples, about 23 of 500.
    """
    rng = np.random.default_rng(0)
    failed = [
        _fails(_on_grid(1 - rng.uniform(size=1000), n_shuffles)) for _ in range(500)
    ]
    assert_rejections_at_rate(failed, 0.001, conservative=True)


@pytest.mark.parametrize("n_shuffles", [39, 199, None])
@pytest.mark.parametrize("power", [0.5, 2.0])
def test_miscalibrated_p_values_fail(n_shuffles, power):
    """p-values that run large (power 0.5) or small (power 2) are not uniform."""
    uniform = 1 - np.random.default_rng(1).uniform(size=1000)
    with pytest.raises(AssertionError, match="are not uniform"):
        assert_pvalues_uniform(_on_grid(uniform**power, n_shuffles))


def test_p_values_never_below_a_half_fail():
    """A gap in the values is seen: only a pile at one value is refused."""
    p_values = np.random.default_rng(2).uniform(0.5, 1.0, size=1000)
    with pytest.raises(AssertionError, match="are not uniform"):
        assert_pvalues_uniform(p_values)


@pytest.mark.parametrize(
    "p_values",
    [np.ones(1000), np.repeat([0.5, 1.0], 500), _balanced(19, 2000)],
    ids=["always 1", "two values", "19 shuffles for 2000 cases"],
)
def test_p_values_too_coarse_for_the_sample_are_refused(p_values):
    """Each sample is exact on its own grid, but one value hides too much.

    The check rejects 2000 cases at a distance of 0.043, less than the 0.05
    of the sample at each p-value from 19 shuffles.
    """
    with pytest.raises(
        ValueError, match="too coarse to check for uniformity"
    ) as caught:
        assert_pvalues_uniform(p_values)
    # The suggested number of shuffles is enough for a sample of this size.
    n_shuffles = int(re.search(r"at least (\d+) shuffles", str(caught.value))[1])
    assert_pvalues_uniform(_balanced(n_shuffles, p_values.size))


def test_the_same_grid_is_accepted_for_a_smaller_sample():
    """1000 cases are rejected at 0.062, more than the 0.05 at each value."""
    assert_pvalues_uniform(_balanced(19, 1000))
