"""One analysis cache per geometry: shared by relabelled copies, never pickled."""

from __future__ import annotations

import pickle
import threading
import time

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment, Regions
from neurospatial.environment._cache import (
    DIFFERENTIAL_OPERATOR,
    KDTREE,
    AnalysisCache,
    CacheKey,
)
from neurospatial.environment._fields import diffusion_eigenbasis
from neurospatial.ops.transforms import translate

N_THREADS = 4


def _refuse_build():
    raise AssertionError("expected a cached value")


@pytest.fixture
def env() -> Environment:
    """A 30 x 30 grid of unit cells."""
    return Environment.from_grid_mask(
        np.ones((30, 30), dtype=bool),
        grid_edges=(np.arange(31.0),) * 2,
        name="arena",
        units="cm",
    )


def _populate(env: Environment) -> dict[str, np.ndarray]:
    """Run analyses that fill the cache and return their results."""
    field = np.linspace(0.0, 1.0, env.n_bins)
    return {
        # A wide bandwidth keeps a truncated eigenbasis in the cache.
        "smoothed": env.smooth(field, bandwidth=4.0, mode="average"),
        "nearest": env.nearest_bin(env.bin_centers + 0.1),
        "kernel": env.compute_kernel(bandwidth=1.0, mode="density"),
        "gaussian": np.array(env.compute_gaussian_kernel(bandwidth=1.0)),
        "boundary": env.boundary_bins,
        "differential": env.get_differential_operator().toarray(),
    }


@pytest.mark.parametrize(
    "relabel",
    [
        lambda env: env.with_metadata(name="renamed"),
        lambda env: env.with_metadata(units="m", frame="room"),
        lambda env: env.with_regions(Regions()),
    ],
    ids=["name", "units-frame", "regions"],
)
def test_metadata_copies_reuse_cached_analyses(env, relabel):
    expected = _populate(env)
    eigenbasis = diffusion_eigenbasis(env._cache).basis
    kdtree = env._cache.get(KDTREE, _refuse_build)

    copy = relabel(env)

    assert copy._cache is env._cache
    assert diffusion_eigenbasis(copy._cache).basis is eigenbasis
    assert copy._cache.get(KDTREE, _refuse_build) is kdtree
    for name, value in _populate(copy).items():
        np.testing.assert_array_equal(value, expected[name], err_msg=name)
    assert diffusion_eigenbasis(copy._cache).basis is eigenbasis


def test_analysis_on_a_copy_is_reused_by_its_source(env):
    renamed = env.with_metadata(name="renamed")
    operator = renamed.get_differential_operator()

    assert DIFFERENTIAL_OPERATOR in env._cache
    np.testing.assert_array_equal(
        env.get_differential_operator().toarray(), operator.toarray()
    )


def test_clearing_a_copy_clears_the_shared_cache(env):
    _populate(env)
    renamed = env.with_metadata(name="renamed")

    renamed.clear_cache()

    assert len(env._cache) == 0


@pytest.mark.parametrize(
    "derive",
    [
        lambda env: env.subset(mask=np.ones(env.n_bins, dtype=bool)),
        lambda env: env.rebin(factor=2),
        lambda env: env.clip(box(0.0, 0.0, 30.0, 30.0)),
        lambda env: env.apply_transform(translate(5.0, 0.0)),
        lambda env: Environment.combine(
            {"a": env, "b": env.apply_transform(translate(50.0, 0.0))}
        ),
        lambda env: pickle.loads(pickle.dumps(env)),
        lambda env: Environment.from_dict(env.to_dict()),
    ],
    ids=["subset", "rebin", "clip", "apply_transform", "combine", "pickle", "dict"],
)
def test_derived_and_restored_environments_start_with_their_own_cache(env, derive):
    _populate(env)
    populated = len(env._cache)

    derived = derive(env)

    assert derived._cache is not env._cache
    assert len(derived._cache) == 0
    assert len(env._cache) == populated
    derived.nearest_bin(derived.bin_centers)
    derived.compute_gaussian_kernel(bandwidth=1.0)
    assert len(env._cache) == populated


def test_file_read_starts_with_its_own_cache(env, tmp_path):
    _populate(env)
    env.to_file(tmp_path / "arena")

    loaded = Environment.from_file(tmp_path / "arena")

    assert len(loaded._cache) == 0


def test_pickle_round_trip_has_empty_cache_and_equal_results(env):
    expected = _populate(env)

    restored = pickle.loads(pickle.dumps(env))

    assert len(restored._cache) == 0
    for name, value in _populate(restored).items():
        np.testing.assert_array_equal(value, expected[name], err_msg=name)


def _run_threads(calls):
    """Run each callable in its own daemon thread, all released at once.

    A deadlock fails the test after the join timeout instead of hanging the
    run: daemon threads do not keep the interpreter alive.
    """
    barrier = threading.Barrier(len(calls))
    results: list[object] = [None] * len(calls)
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        barrier.wait()
        try:
            results[index] = calls[index]()
        except BaseException as error:
            errors.append(error)

    threads = [
        threading.Thread(target=worker, args=(i,), daemon=True)
        for i in range(len(calls))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not any(thread.is_alive() for thread in threads), "threads deadlocked"
    assert not errors, errors
    return results


def _run_concurrently(call):
    """Call ``call`` from N_THREADS threads released at the same moment."""
    return _run_threads([call] * N_THREADS)


def _counted(function, calls):
    """Wrap a builder to record each call."""

    def wrapper(*args, **kwargs):
        calls.append(threading.get_ident())
        return function(*args, **kwargs)

    return wrapper


def _slow_counted(function, calls):
    """Wrap a builder so concurrent callers overlap while it runs."""

    def wrapper(*args, **kwargs):
        calls.append(threading.get_ident())
        time.sleep(0.05)
        return function(*args, **kwargs)

    return wrapper


def test_concurrent_first_kdtree_lookup_builds_one_tree(env, monkeypatch):
    from neurospatial.environment import _lookup

    calls: list[int] = []
    monkeypatch.setattr(_lookup, "cKDTree", _slow_counted(_lookup.cKDTree, calls))
    points = env.bin_centers + 0.1

    results = _run_concurrently(lambda: env.nearest_bin(points))

    assert len(calls) == 1
    for result in results:
        np.testing.assert_array_equal(result, np.arange(env.n_bins))


def test_concurrent_first_diffusion_geometry_builds_once(env, monkeypatch):
    from neurospatial.environment import _fields

    calls: list[int] = []
    monkeypatch.setattr(
        _fields,
        "_build_diffusion_geometry",
        _slow_counted(_fields._build_diffusion_geometry, calls),
    )

    results = _run_concurrently(env.get_diffusion_geometry)

    assert len(calls) == 1
    for result in results:
        assert np.shares_memory(result.W.data, results[0].W.data)


def test_concurrent_kernels_build_nested_geometry_once(env, monkeypatch):
    """A kernel needs the diffusion geometry; neither is built twice or deadlocks."""
    from neurospatial.environment import _fields

    geometry_calls: list[int] = []
    kernel_calls: list[int] = []
    monkeypatch.setattr(
        _fields,
        "_build_diffusion_geometry",
        _slow_counted(_fields._build_diffusion_geometry, geometry_calls),
    )
    monkeypatch.setattr(
        _fields,
        "heat_kernel_from_W",
        _slow_counted(_fields.heat_kernel_from_W, kernel_calls),
    )

    results = _run_concurrently(
        lambda: env.compute_kernel(bandwidth=1.0, mode="density")
    )

    assert len(geometry_calls) == 1
    assert len(kernel_calls) == 1
    for result in results:
        np.testing.assert_array_equal(result, results[0])


def test_cache_gives_concurrent_callers_one_value():
    cache = AnalysisCache()
    key: CacheKey[object] = CacheKey("value")
    calls: list[int] = []

    def build() -> object:
        calls.append(threading.get_ident())
        time.sleep(0.05)
        return object()

    results = _run_concurrently(lambda: cache.get(key, build))

    assert len(calls) == 1
    assert all(result is results[0] for result in results)


def test_cache_updates_run_one_at_a_time_on_the_latest_value():
    """Concurrent updates of one key each see the value the previous one stored."""
    cache = AnalysisCache()
    key: CacheKey[int] = CacheKey("count")

    def step(count: int) -> tuple[int, int]:
        time.sleep(0.01)  # overlap the callers while one update runs
        return count + 1, count

    results = _run_concurrently(lambda: cache.update(key, lambda: 0, step))

    assert sorted(results) == list(range(N_THREADS))
    assert cache.get(key, _refuse_build) == N_THREADS


def test_cache_pickles_and_copies_empty():
    import copy

    cache = AnalysisCache()
    key: CacheKey[list[int]] = CacheKey("value")
    cache.get(key, lambda: [1, 2, 3])

    for restored in (pickle.loads(pickle.dumps(cache)), copy.deepcopy(cache)):
        assert isinstance(restored, AnalysisCache)
        assert len(restored) == 0
        assert restored.get(key, lambda: [4]) == [4]
    assert cache.get(key, _refuse_build) == [1, 2, 3]


def test_concurrent_smoothing_solves_the_eigenbasis_once(env, monkeypatch):
    """Threads smoothing at one bandwidth share one eigensolve and one answer."""
    from neurospatial.ops import diffusion

    calls: list[int] = []
    monkeypatch.setattr(
        diffusion,
        "_adaptive_symmetric_basis",
        _slow_counted(diffusion._adaptive_symmetric_basis, calls),
    )
    field = np.linspace(0.0, 1.0, env.n_bins)
    expected = pickle.loads(pickle.dumps(env)).smooth(
        field, bandwidth=4.0, mode="average"
    )
    calls.clear()

    results = _run_concurrently(
        lambda: env.smooth(field, bandwidth=4.0, mode="average")
    )

    assert len(calls) == 1
    assert diffusion_eigenbasis(env._cache).basis is not None
    for result in results:
        np.testing.assert_array_equal(result, expected)


def test_concurrent_mrf_bases_solve_the_eigenbasis_once(env, monkeypatch):
    """Threads requesting one MRF basis share one global eigensolve."""
    from neurospatial.ops import diffusion

    calls: list[int] = []
    monkeypatch.setattr(
        diffusion,
        "_symmetric_eigenbasis",
        _slow_counted(diffusion._symmetric_eigenbasis, calls),
    )
    occupancy = np.ones(env.n_bins)

    results = _run_concurrently(lambda: env.get_mrf_basis(occupancy, rank=20))

    assert len(calls) == 1
    for result in results:
        np.testing.assert_array_equal(result.B, results[0].B)


def test_threads_on_an_environment_and_its_copies_share_one_computation(
    env, monkeypatch
):
    """Relabelled copies share locks as well as values: one build in total."""
    from neurospatial.environment import _fields, _lookup
    from neurospatial.ops import diffusion

    field = np.linspace(0.0, 1.0, env.n_bins)
    points = env.bin_centers + 0.1
    reference = pickle.loads(pickle.dumps(env))
    expected_smoothed = reference.smooth(field, bandwidth=4.0, mode="average")
    expected_nearest = reference.nearest_bin(points)

    geometry_calls: list[int] = []
    eigensolve_calls: list[int] = []
    tree_calls: list[int] = []
    monkeypatch.setattr(
        _fields,
        "_build_diffusion_geometry",
        _counted(_fields._build_diffusion_geometry, geometry_calls),
    )
    monkeypatch.setattr(
        diffusion,
        "_adaptive_symmetric_basis",
        _counted(diffusion._adaptive_symmetric_basis, eigensolve_calls),
    )
    monkeypatch.setattr(_lookup, "cKDTree", _counted(_lookup.cKDTree, tree_calls))
    renamed = env.with_metadata(name="renamed")
    relabelled = env.with_regions(Regions())

    results = _run_threads(
        [
            lambda: env.smooth(field, bandwidth=4.0, mode="average"),
            lambda: renamed.smooth(field, bandwidth=4.0, mode="average"),
            lambda: relabelled.smooth(field, bandwidth=4.0, mode="average"),
            lambda: renamed.nearest_bin(points),
            lambda: relabelled.nearest_bin(points),
            lambda: env.nearest_bin(points),
        ]
    )

    assert (len(geometry_calls), len(eigensolve_calls), len(tree_calls)) == (1, 1, 1)
    for smoothed in results[:3]:
        np.testing.assert_array_equal(smoothed, expected_smoothed)
    for nearest in results[3:]:
        np.testing.assert_array_equal(nearest, expected_nearest)


def test_clear_cache_during_smoothing_neither_blocks_nor_changes_the_result(
    env, monkeypatch
):
    """Clearing while a smoothing call holds the eigenbasis returns at once.

    Every wait shares one short deadline, so a deadlock fails the test in
    seconds instead of after a chain of long timeouts.
    """
    from neurospatial.environment import _fields

    field = np.linspace(0.0, 1.0, env.n_bins)
    expected = pickle.loads(pickle.dumps(env)).smooth(
        field, bandwidth=4.0, mode="average"
    )
    deadline = time.monotonic() + 20.0

    def remaining() -> float:
        return max(0.0, deadline - time.monotonic())

    applying = threading.Event()
    cleared = threading.Event()
    original = _fields._diffusion_apply

    def apply_after_clear(*args, **kwargs):
        applying.set()
        cleared.wait(timeout=remaining())
        return original(*args, **kwargs)

    monkeypatch.setattr(_fields, "_diffusion_apply", apply_after_clear)
    smoothed: list[np.ndarray] = []
    smoothing = threading.Thread(
        target=lambda: smoothed.append(
            env.smooth(field, bandwidth=4.0, mode="average")
        ),
        daemon=True,
    )
    smoothing.start()
    assert applying.wait(timeout=remaining()), "smoothing never started"

    clearing = threading.Thread(
        target=lambda: (env.clear_cache(), cleared.set()), daemon=True
    )
    clearing.start()
    clearing.join(timeout=remaining())
    assert not clearing.is_alive(), "clear_cache() blocked while smoothing ran"
    smoothing.join(timeout=remaining())
    assert not smoothing.is_alive(), "smoothing deadlocked"
    np.testing.assert_array_equal(smoothed[0], expected)
    monkeypatch.setattr(_fields, "_diffusion_apply", original)
    np.testing.assert_array_equal(
        env.smooth(field, bandwidth=4.0, mode="average"), expected
    )


def _track() -> Environment:
    return Environment.linear_track(
        endpoints=[(0.0, 0.0), (20.0, 0.0)], bin_size=1.0, track_width=1.0, units="cm"
    )


def _mesh() -> Environment:
    from neurospatial.layout import create_layout

    return Environment.from_layout(
        create_layout(
            "TriangularMesh", boundary_polygon=box(0, 0, 8, 8), point_spacing=2.0
        )
    )


def _clipped() -> Environment:
    from shapely.geometry import Point

    square = Environment.from_grid_mask(
        np.ones((10, 10), dtype=bool), grid_edges=(np.arange(11.0),) * 2
    )
    return square.clip(Point(5.0, 5.0).buffer(4.0))


_LOOKUP_LAYOUTS = pytest.mark.parametrize(
    "build", [_track, _mesh, _clipped], ids=["kd-tree", "triangles", "polygons"]
)


@_LOOKUP_LAYOUTS
def test_lookup_structures_live_in_the_analysis_cache(build):
    env = build()
    points = env.bin_centers
    locator = env._binning.locator
    pickled = len(pickle.dumps(locator))
    assert len(env._cache) == 0

    np.testing.assert_array_equal(env.bin_at(points), np.arange(env.n_bins))

    assert len(env._cache) > 0
    # The locator keeps no search structure of its own.
    assert len(pickle.dumps(locator)) == pickled
    renamed = env.with_metadata(name="renamed")
    np.testing.assert_array_equal(renamed.bin_at(points), np.arange(env.n_bins))
    assert len(renamed._cache) == len(env._cache)
    env.clear_cache()
    assert len(renamed._cache) == 0
    np.testing.assert_array_equal(renamed.bin_at(points), np.arange(env.n_bins))


@_LOOKUP_LAYOUTS
def test_concurrent_first_lookups_build_one_search_structure(build, monkeypatch):
    from neurospatial.geometry import _locators, _polygon_cells

    env = build()
    calls: list[int] = []
    for module, name in (
        (_locators, "cKDTree"),
        (_locators, "_triangle_index"),
        (_polygon_cells, "STRtree"),
    ):
        monkeypatch.setattr(module, name, _slow_counted(getattr(module, name), calls))

    results = _run_concurrently(lambda: env.bin_at(env.bin_centers))

    assert len(calls) == 1
    for result in results:
        np.testing.assert_array_equal(result, np.arange(env.n_bins))


@pytest.mark.parametrize("first", ["bin_at", "nearest_bin"])
def test_track_lookups_share_one_kdtree(first, monkeypatch):
    from neurospatial.environment import _lookup
    from neurospatial.geometry import _locators

    env = _track()
    calls: list[int] = []
    for module in (_lookup, _locators):
        monkeypatch.setattr(module, "cKDTree", _counted(module.cKDTree, calls))
    points = env.bin_centers + 0.25
    second = "nearest_bin" if first == "bin_at" else "bin_at"

    for method in (first, second, first):
        np.testing.assert_array_equal(
            getattr(env, method)(points), np.arange(env.n_bins)
        )

    assert len(calls) == 1


def test_track_lookups_resolve_equidistant_points_alike():
    """bin_at and nearest_bin search one tree, so ties go to the same bin."""
    import networkx as nx

    corners = {0: (0.0, 0.0), 1: (40.0, 0.0), 2: (40.0, 40.0), 3: (0.0, 40.0)}
    graph = nx.Graph()
    for node, position in corners.items():
        graph.add_node(node, pos=position)
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]:
        length = float(np.hypot(*np.subtract(corners[u], corners[v])))
        graph.add_edge(u, v, distance=length)
    env = Environment.from_graph(
        graph, edge_order=list(graph.edges), bin_size=1.0, track_width=8.0
    )
    # Points on a half-unit lattice are often equidistant from several centers.
    points = np.round(np.random.default_rng(0).uniform(0, 40, (5000, 2)) * 2) / 2

    nearest = env.nearest_bin(points, tie_break="closest_center", max_distance=np.inf)

    bins = env.bin_at(points)
    on_track = bins >= 0
    assert on_track.mean() > 0.5
    np.testing.assert_array_equal(bins[on_track], nearest[on_track])


def test_analyses_reuse_the_environment_kdtree(env, monkeypatch):
    """Probability mapping and basis selection search env.bin_center_tree()."""
    import scipy.spatial

    from neurospatial.environment import _lookup
    from neurospatial.ops import (
        heat_kernel_wavelet_basis,
        map_probabilities,
        select_basis_centers,
    )

    calls: list[int] = []
    counted = _counted(scipy.spatial.cKDTree, calls)
    monkeypatch.setattr(scipy.spatial, "cKDTree", counted)
    monkeypatch.setattr(_lookup, "cKDTree", counted)
    probs = np.full(env.n_bins, 1.0 / env.n_bins)

    for mode in ("nearest", "inverse-distance-weighted"):
        mapped = map_probabilities(env, env, probs, mode=mode, n_neighbors=3)
        assert np.isclose(mapped.sum(), 1.0)
    select_basis_centers(env, n_centers=4, method="kmeans", rng=0)
    heat_kernel_wavelet_basis(env, centers=np.array([0, 5]))
    env.nearest_bin(env.bin_centers)

    assert len(calls) == 1
