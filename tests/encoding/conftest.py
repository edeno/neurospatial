"""Shared fixtures for encoding tests."""

from __future__ import annotations

from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from neurospatial import Environment


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
# MRF eigenbasis-resolver fixtures
#
# Deterministic finite-volume geometries for the live-component eigenbasis
# resolver. Occupancy vectors are built directly in active-bin order by the
# tests, so no neural data is needed -- the resolver is pure geometry.
# ---------------------------------------------------------------------------


@pytest.fixture
def open_field_env() -> Environment:
    """2D open field: single W-component, uniform bin volumes."""
    from neurospatial import Environment

    edges = np.linspace(0.0, 16.0, 9)
    return Environment.from_grid_mask(
        active_mask=np.ones((8, 8), dtype=bool),
        grid_edges=(edges, edges),
        connect_diagonal_neighbors=True,
    )


@pytest.fixture
def sparse_regime_env() -> Environment:
    """Larger single-component grid so a modest ``rank`` stays truncated.

    400 bins with a rank well below ``dense_fraction * n`` (0.5 * 400 = 200)
    keeps the resolver in the sparse (persisted) eigenbasis regime, so
    eigensolve-reuse across calls is observable.
    """
    from neurospatial import Environment

    edges = np.linspace(0.0, 40.0, 21)
    return Environment.from_grid_mask(
        active_mask=np.ones((20, 20), dtype=bool),
        grid_edges=(edges, edges),
        connect_diagonal_neighbors=True,
    )


@pytest.fixture
def two_component_env() -> Environment:
    """Masked 2D grid split into two disconnected W-components by a wall."""
    from neurospatial import Environment

    mask = np.ones((6, 6), dtype=bool)
    mask[:, 3] = False  # vertical wall -> left (18 bins) / right (12 bins)
    return Environment.from_grid_mask(
        active_mask=mask,
        grid_edges=(np.linspace(0.0, 12.0, 7), np.linspace(0.0, 12.0, 7)),
        connect_diagonal_neighbors=False,
    )


@pytest.fixture
def two_path_env() -> Environment:
    """Two disjoint 3-node linear tracks (2 W-components) via ``from_graph``."""
    import networkx as nx

    from neurospatial import Environment

    graph = nx.Graph()
    for node, pos in [
        (0, (0.0, 0.0)),
        (1, (6.0, 0.0)),
        (2, (12.0, 0.0)),
        (3, (100.0, 0.0)),
        (4, (106.0, 0.0)),
        (5, (112.0, 0.0)),
    ]:
        graph.add_node(node, pos=pos)
    graph.add_edge(0, 1, distance=6.0)
    graph.add_edge(1, 2, distance=6.0)
    graph.add_edge(3, 4, distance=6.0)
    graph.add_edge(4, 5, distance=6.0)
    return Environment.from_graph(
        graph,
        edge_order=[(0, 1), (1, 2), (3, 4), (4, 5)],
        edge_spacing=50.0,
        bin_size=6.0,
    )


@pytest.fixture
def four_component_env() -> Environment:
    """1D grid broken into four disconnected segments by three gaps.

    The first segment (2 bins) is small enough that a live-only ``r_eff`` can
    fall below ``n_components == 4`` -- exercising the first-solve ``G`` floor.
    """
    from neurospatial import Environment

    # Four segments (sizes 2, 5, 5, 5) separated by single-bin gaps.
    mask = np.concatenate(
        [
            np.ones(2, dtype=bool),
            np.zeros(1, dtype=bool),
            np.ones(5, dtype=bool),
            np.zeros(1, dtype=bool),
            np.ones(5, dtype=bool),
            np.zeros(1, dtype=bool),
            np.ones(5, dtype=bool),
        ]
    )
    return Environment.from_grid_mask(
        active_mask=mask, grid_edges=(np.arange(mask.size + 1, dtype=float),)
    )


@pytest.fixture
def asymmetric_two_component_env() -> Environment:
    """A thin strip (weak chain) + a compact block (tight), two W-components.

    The strip's Fiedler eigenvalue is much smaller than the block's, so the
    globally smallest positive modes are spread across both components -- a
    faithful test of positivity-based fill selection.
    """
    from neurospatial import Environment

    mask = np.zeros((8, 10), dtype=bool)
    mask[:, 0] = True  # thin 8x1 strip (weak chain)
    mask[0:4, 5:9] = True  # compact 4x4 block (tight)
    return Environment.from_grid_mask(
        active_mask=mask,
        grid_edges=(np.linspace(0.0, 8.0, 9), np.linspace(0.0, 10.0, 11)),
        connect_diagonal_neighbors=False,
    )


@pytest.fixture
def dead_dominant_env() -> Environment:
    """A large dead strip + a small live block (2 W-components).

    The dead strip's many low modes crowd the bottom of the global spectrum
    below the live block's Fiedler, so the resolver must grow ``G`` past those
    dead modes before enough live fill modes appear.
    """
    from neurospatial import Environment

    mask = np.zeros((6, 30), dtype=bool)
    mask[0, 0:24] = True  # long 1x24 dead strip (dense low spectrum)
    mask[0:3, 27:30] = True  # small 3x3 live block
    return Environment.from_grid_mask(
        active_mask=mask,
        grid_edges=(np.linspace(0.0, 6.0, 7), np.linspace(0.0, 30.0, 31)),
        connect_diagonal_neighbors=False,
    )


@pytest.fixture
def polar_env() -> Any:
    """Egocentric polar env: strongly non-uniform bin volumes, one component."""
    from neurospatial.environment.polar import EgocentricPolarEnvironment

    return EgocentricPolarEnvironment.create(
        distance_range=(0.0, 20.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=5.0,
        angle_bin_size=np.pi / 4,
    )


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


# ---------------------------------------------------------------------------
# Object-vector cell session fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def ovc_session() -> tuple[
    Environment,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Synthetic session for object-vector-cell classification.

    The animal orbits a single object at a roughly fixed distance while its
    heading tracks the tangent of the orbit, so the object sits at a
    consistent egocentric bearing. Spikes are concentrated whenever the
    object is at that preferred egocentric (distance, direction), producing a
    non-degenerate egocentric rate map -- so both the free
    :func:`is_object_vector_cell` and
    :meth:`EgocentricRateResult.is_object_vector_cell` return a meaningful,
    equal boolean across thresholds.

    Returns
    -------
    tuple
        ``(env, spike_times, times, positions, headings, object_positions)``.
    """
    from neurospatial import Environment

    rng = np.random.default_rng(0)

    n_samples = 4000
    times = np.linspace(0.0, 200.0, n_samples)
    object_positions = np.array([[50.0, 50.0]])

    # Orbit the object at radius ~20 cm, several laps.
    orbit_radius = 20.0
    theta = np.linspace(0.0, 2 * np.pi * 8, n_samples)
    positions = np.column_stack(
        [
            50.0 + orbit_radius * np.cos(theta),
            50.0 + orbit_radius * np.sin(theta),
        ]
    )
    # Heading tracks the orbit tangent so the object stays at a fixed
    # egocentric bearing (roughly 90 deg to one side throughout).
    headings = (theta + np.pi / 2) % (2 * np.pi)
    headings = np.arctan2(np.sin(headings), np.cos(headings))

    env = Environment.from_samples(positions, bin_size=2.0)

    # Concentrate spikes over short bursts at regular phase intervals so the
    # egocentric polar rate map has a clear, informative peak.
    spike_mask = np.sin(theta * 3) > 0.7
    candidate_times = times[spike_mask]
    spike_times = np.sort(
        rng.choice(candidate_times, size=min(300, candidate_times.size), replace=True)
    )

    return env, spike_times, times, positions, headings, object_positions
