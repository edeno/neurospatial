"""Shared fixtures for encoding tests."""

from __future__ import annotations

from collections.abc import Callable, Generator
from typing import Any

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def restore_numpy_random_state() -> Generator[None, None, None]:
    """Make legacy global np.random use deterministic and order-independent."""
    previous_state: tuple[Any, ...] = np.random.get_state()
    np.random.seed(0)
    yield
    np.random.set_state(previous_state)


@pytest.fixture(autouse=True)
def restore_jax_x64_config() -> Generator[None, None, None]:
    """Keep tests that enable JAX x64 from leaking global config state."""
    # Resolve ``is_jax_available`` dynamically. ``restore_backend_availability_cache``
    # below calls ``importlib.reload(backend_module)`` on teardown, which
    # rebinds ``is_jax_available`` to a fresh function inside the module —
    # a module-level ``from … import is_jax_available`` here would silently
    # point at the *pre-reload* function (with its own stale LRU cache) on
    # every test after the first.
    import neurospatial.encoding._backend as backend_module

    if not backend_module.is_jax_available():
        yield
        return

    import jax

    previous_value = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous_value)


@pytest.fixture(autouse=True)
def restore_backend_availability_cache() -> Generator[None, None, None]:
    """Isolate the ``_backend`` module's LRU cache across tests.

    Tests that monkeypatch ``sys.platform`` or call ``cache_clear()``
    leave the module holding stale availability state once
    ``sys.platform`` is restored. Without a teardown, subsequent tests
    run against that perturbed module — order-dependent under xdist.

    Autouse so any test in the encoding suite that mutates platform or
    reloads ``_backend`` is cleaned up, even if it forgets to request
    the fixture. Clears pre-yield and reloads the module on teardown,
    giving every test a fresh ``is_jax_available()`` lookup.
    """
    import importlib

    import neurospatial.encoding._backend as backend_module

    backend_module.is_jax_available.cache_clear()
    yield
    importlib.reload(backend_module)


# ---------------------------------------------------------------------------
# Directional / head-direction fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def uniform_heading_trajectory() -> tuple[np.ndarray, np.ndarray]:
    """Trajectory whose heading sweeps all directions uniformly.

    Returns
    -------
    times : ndarray, shape (1800,)
        Timestamps over a 60 s recording.
    headings : ndarray, shape (1800,)
        Head direction in radians sweeping [0, 2*pi) multiple times so all
        bins are visited roughly uniformly.
    """
    times = np.linspace(0, 60, 1800)
    # Sweep through all directions several times to ensure uniform coverage.
    headings = (np.linspace(0, 2 * np.pi * 30, 1800) % (2 * np.pi)).astype(np.float64)
    return times, headings


@pytest.fixture
def von_mises_hd_spikes() -> Callable[..., np.ndarray]:
    """Factory drawing Poisson spikes from a von Mises head-direction tuning.

    Returns
    -------
    callable
        ``draw(times, headings, preferred_direction, concentration, *,
        peak_rate=30.0, seed=0)`` returning a sorted spike-time array. The
        instantaneous rate follows a von Mises tuning curve centered on
        ``preferred_direction``; spikes are drawn per frame as Poisson counts
        and jittered within the frame interval.
    """

    def draw(
        times: np.ndarray,
        headings: np.ndarray,
        preferred_direction: float,
        concentration: float,
        *,
        peak_rate: float = 30.0,
        seed: int = 0,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        dt = np.diff(times)
        frame_headings = headings[:-1]
        # Von Mises tuning normalized to peak 1 at preferred_direction.
        tuning = np.exp(
            concentration * (np.cos(frame_headings - preferred_direction) - 1.0)
        )
        rate = peak_rate * tuning
        expected = rate * dt
        counts = rng.poisson(expected)
        spike_times = []
        for i, c in enumerate(counts):
            if c > 0:
                spike_times.append(times[i] + rng.uniform(0, dt[i], size=c))
        if spike_times:
            return np.sort(np.concatenate(spike_times))
        return np.array([], dtype=np.float64)

    return draw


@pytest.fixture(params=["nan", "inf"])
def nan_block_heading_trajectory(
    request: pytest.FixtureRequest,
) -> dict[str, np.ndarray | slice]:
    """Uniform trajectory with a contiguous block of non-finite headings.

    Parametrized over a NaN block and an Inf block. Provides both the
    corrupted arrays and a ``_clean`` version with the bad frames removed so
    "bin 0 not inflated" tests can compare against ground truth.

    Returns
    -------
    dict
        Keys: ``times``, ``headings`` (with the block set non-finite),
        ``times_clean``, ``headings_clean`` (block frames removed),
        ``block`` (the slice that was corrupted), and ``bad_value``.
    """
    times = np.linspace(0, 60, 1800)
    headings = (np.linspace(0, 2 * np.pi * 30, 1800) % (2 * np.pi)).astype(np.float64)

    bad_value = np.nan if request.param == "nan" else np.inf
    block = slice(500, 600)

    headings_bad = headings.copy()
    headings_bad[block] = bad_value

    keep = np.ones(len(times), dtype=bool)
    keep[block] = False
    times_clean = times[keep]
    headings_clean = headings[keep]

    return {
        "times": times,
        "headings": headings_bad,
        "times_clean": times_clean,
        "headings_clean": headings_clean,
        "block": block,
        "bad_value": np.asarray(bad_value),
    }


# ---------------------------------------------------------------------------
# Phase-precession fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def precessing_spikes() -> dict[str, np.ndarray | float]:
    """Synthetic phase-precession data with a planted negative slope.

    Returns
    -------
    dict
        Keys: ``positions``, ``phases`` (negative-slope precession with
        von-Mises jitter), ``phase_shuffled`` (the same phases permuted, a
        null control), and ``true_slope``.
    """
    rng = np.random.default_rng(0)
    n_spikes = 200
    positions = np.sort(rng.uniform(0, 50, n_spikes))
    true_slope = -0.1  # rad per position unit
    offset = 2 * np.pi
    # Von Mises jitter around the linear phase-position relationship.
    jitter = rng.vonmises(0.0, 8.0, n_spikes)
    phases = (offset + true_slope * positions + jitter) % (2 * np.pi)
    phase_shuffled = rng.permutation(phases)
    return {
        "positions": positions,
        "phases": phases,
        "phase_shuffled": phase_shuffled,
        "true_slope": true_slope,
    }
