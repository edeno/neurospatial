"""Provider boundary, numerical contracts, and geometry-cache ownership."""

import warnings
from datetime import datetime, timezone
from unittest.mock import patch

import networkx as nx
import numpy as np
import pytest
from environment_helpers import (
    built_from,
    face_graph,
    publish_binning,
    publish_environment,
)
from scipy import linalg
from shapely.geometry import Polygon

from neurospatial import Environment, _constants
from neurospatial.environment._fields import (
    diffusion_eigenbasis,
    face_geometry,
    mrf_eigenbasis,
)
from neurospatial.geometry._binning import Binning, KnownFaces, SpatialBins
from neurospatial.geometry._diffusion import (
    DiffusionGeometry,
    FaceGeometry,
)
from neurospatial.geometry._locators import NearestBinLocator
from neurospatial.io import from_dict, from_file, to_dict, to_file
from neurospatial.layout import create_layout
from neurospatial.layout.engines.masked_grid import MaskedGridLayout
from neurospatial.ops.basis import heat_kernel_wavelet_basis
from neurospatial.ops.calculus import divergence, gradient
from neurospatial.ops.smoothing import compute_diffusion_kernels
from neurospatial.ops.transforms import Affine2D


def _face_environment(scale=1.0):
    """Publish owned data for unequal cells, independently of a layout callback."""
    layout = MaskedGridLayout()
    layout.build(active_mask=np.ones(3, dtype=bool), grid_edges=(np.arange(4.0),))
    graph = layout.connectivity.copy()
    volumes = np.array([1.0, 2.0, 4.0])
    for (u, v), area in zip(graph.edges(), (2.0, 3.0), strict=True):
        graph.edges[u, v]["A"] = area * scale
    faces = FaceGeometry.from_graph(graph, volumes)
    bins = SpatialBins.from_arrays(
        layout.bin_centers, volumes, 1, faces.edges, np.ones(2)
    )
    binning = Binning("Faces", bins, NearestBinLocator(bins.centers), KnownFaces(faces))
    owned = publish_binning(
        binning,
        dimension_ranges=layout.dimension_ranges,
        layout_type="Faces",
        build_params={},
    )
    return publish_environment(layout=owned, layout_type_used="Faces")


@pytest.fixture
def face_env():
    return _face_environment()


def test_all_diffusion_consumers_share_provider_snapshot(face_env, monkeypatch):
    """Owned unequal-cell data drives every solver without solver registration."""
    env = face_env
    builds = []
    build = KnownFaces.build

    def counting_build(provider):
        builds.append(provider)
        return build(provider)

    monkeypatch.setattr(KnownFaces, "build", counting_build)
    mass = np.diag([1.0, 2.0, 4.0])
    stiffness = np.array([[2, -2, 0], [-2, 5, -3], [0, -3, 3]])
    generator = np.linalg.solve(mass, stiffness)
    heat = linalg.expm(-0.5 * generator)
    field = np.array([1.0, 3.0, 0.0])

    for mode, expected in (
        ("average", heat),
        ("transition", heat.T),
        ("density", heat @ np.linalg.inv(mass)),
    ):
        np.testing.assert_allclose(
            env.compute_kernel(bandwidth=1.0, mode=mode), expected
        )
        np.testing.assert_allclose(
            env.diffuse(field, bandwidth=1.0, mode=mode), expected @ field, atol=1e-12
        )

    wavelets = heat_kernel_wavelet_basis(
        env, centers=np.array([0]), scales=[0.5], normalize="none"
    )
    np.testing.assert_allclose(wavelets[0], heat[:, 0], atol=1e-12)

    basis = env.get_mrf_basis(np.ones(3), rank=3)
    np.testing.assert_allclose(basis.B.T @ mass @ basis.B, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(
        basis.B.T @ stiffness @ basis.B, np.diag(basis.d), atol=1e-12
    )
    # The same faces drive the differential operators: div(grad f) = -M^-1 L f.
    np.testing.assert_allclose(
        divergence(env, gradient(env, field)), -generator @ field, atol=1e-12
    )
    assert len(builds) == 1  # one face build serves every consumer


def test_clear_cache_rebuilds_geometry_and_both_eigenbases(face_env):
    env = face_env
    before = env.compute_kernel(bandwidth=1.0, mode="average")
    env.diffuse(np.ones(3), bandwidth=1.0)
    env.get_mrf_basis(np.ones(3), rank=3)
    old_geometry = env._diffusion_geometry
    old_diffusion_cache = diffusion_eigenbasis(env._cache)
    old_mrf_cache = mrf_eigenbasis(env._cache, env.n_bins)

    env.clear_cache()
    np.testing.assert_allclose(
        env.compute_kernel(bandwidth=1.0, mode="average"), before
    )
    replacement = _face_environment(scale=2.0)
    after = replacement.compute_kernel(bandwidth=1.0, mode="average")
    np.testing.assert_allclose(after, before @ before, atol=1e-12)
    np.testing.assert_allclose(
        replacement.diffuse(np.arange(3.0), bandwidth=1.0, mode="average"),
        after @ np.arange(3.0),
    )
    assert env._diffusion_geometry is not old_geometry
    assert diffusion_eigenbasis(env._cache) is not old_diffusion_cache
    assert mrf_eigenbasis(env._cache, env.n_bins) is not old_mrf_cache
    np.testing.assert_allclose(
        env.compute_kernel(bandwidth=1.0, mode="average"), before
    )


def test_layout_subclass_publishes_through_its_binning():
    """An unregistered engine subclass uses the same binning() contract and faces."""

    class RenamedGrid(MaskedGridLayout):
        pass

    params = {
        "active_mask": np.ones((2, 3), dtype=bool),
        "grid_edges": (np.arange(3.0), np.arange(4.0)),
    }
    subclass = RenamedGrid()
    subclass.build(**params)
    original = Environment.from_layout(create_layout("MaskedGrid", **params))
    renamed = Environment.from_layout(subclass)
    assert renamed.layout_type == original.layout_type == "MaskedGrid"
    np.testing.assert_array_equal(
        renamed.compute_kernel(bandwidth=1.0), original.compute_kernel(bandwidth=1.0)
    )


def test_geometry_owns_its_inputs_and_ignores_zero_area_faces():
    graph = nx.Graph()
    graph.add_nodes_from([2, 0, 1])
    graph.add_edge(0, 1, A=2.0, distance=4.0)
    graph.add_edge(1, 2, A=0.0, distance=1.0)
    before = list(graph.edges(data=True))
    volumes = np.array([1.0, 3.0, 5.0])
    geometry = DiffusionGeometry.from_graph(graph, volumes)
    assert list(graph.edges(data=True)) == before
    graph.edges[0, 1]["A"] = 9.0
    volumes[:] = 9.0
    np.testing.assert_array_equal(
        geometry.W.toarray(), [[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]]
    )
    np.testing.assert_array_equal(geometry.volumes, [1, 3, 5])
    assert geometry.n_components == 2
    assert geometry.labels[0] == geometry.labels[1] != geometry.labels[2]
    for buffer in (
        geometry.W.data,
        geometry.W.indices,
        geometry.W.indptr,
        geometry.volumes,
        geometry.labels,
    ):
        assert not buffer.flags.writeable


@pytest.mark.parametrize("volumes", [[1], [1, 0], [1, -1], [1, np.nan], [1, np.inf]])
def test_invalid_cell_measures_fail_at_provider_boundary(volumes):
    with pytest.raises(ValueError, match="volumes must"):
        DiffusionGeometry.from_graph(nx.path_graph(2), volumes)


def test_anisotropic_3d_grid_face_weights_and_cell_measures():
    env = Environment.from_grid_mask(
        np.ones((2, 2, 2), dtype=bool),
        grid_edges=(np.arange(3.0), np.arange(3.0) * 2, np.arange(3.0) * 4),
        connect_diagonal_neighbors=True,
    )
    geometry = env._diffusion_geometry
    np.testing.assert_array_equal(geometry.volumes, np.full(8, 8.0))
    for u, v in env.connectivity.edges:
        offset = np.abs(env.bin_centers[u] - env.bin_centers[v])
        expected = 8.0 / np.sum(offset) ** 2 if np.count_nonzero(offset) == 1 else 0.0
        assert geometry.W[u, v] == expected
        assert "A" not in env.connectivity.edges[u, v]


# ---------------------------------------------------------------------------
# Reconstructed environments keep the saved graph on their layout.
# ---------------------------------------------------------------------------
def _round_trip(env, how, tmp_path):
    if how == "dict":
        return from_dict(to_dict(env))
    if how == "file":
        to_file(env, tmp_path / "env")
        return from_file(tmp_path / "env")
    pynwb = pytest.importorskip("pynwb")
    from neurospatial.io.nwb import read_environment, write_environment

    nwbfile = pynwb.NWBFile(
        session_description="round trip",
        identifier="round-trip",
        session_start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    write_environment(nwbfile, env)
    return read_environment(nwbfile)


def _four_connected_grid():
    positions = np.random.default_rng(0).uniform(0, 30, (4000, 2))
    return Environment.from_samples(
        positions, bin_size=3.0, connect_diagonal_neighbors=False
    )


def _polar(circular_angle=True):
    return Environment.from_polar_egocentric(
        distance_range=(0.0, 30.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=5.0,
        angle_bin_size=np.pi / 6,
        circular_angle=circular_angle,
    )


def _open_polar():
    return _polar(circular_angle=False)


@pytest.mark.parametrize("how", ["dict", "file", "nwb"])
@pytest.mark.parametrize(
    "make_env",
    [_four_connected_grid, _polar, _open_polar],
    ids=["grid4", "polar", "open-polar"],
)
def test_round_trips_and_their_copies_use_the_saved_graph(make_env, how, tmp_path):
    """Read-back environments and their copies keep 4-connectivity and polar seams.

    The rebuilt layout would otherwise supply an 8-connected grid graph (NWB) or
    a Cartesian, seamless polar graph, which derived environments and the geometry read.
    """
    original = make_env()
    field = np.sin(np.arange(original.n_bins) * 0.37)
    expected_divergence = divergence(original, gradient(original, field))
    restored = _round_trip(original, how, tmp_path)
    for env in (restored, restored.with_metadata()):
        assert list(env.connectivity.edges(data=True)) == list(
            env.connectivity.edges(data=True)
        )
        np.testing.assert_array_equal(env.bin_centers, env.bin_centers)
        for actual, expected in zip(env.grid_edges, env.grid_edges, strict=True):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(env.active_mask, env.active_mask)
        assert list(env.connectivity.edges()) == list(original.connectivity.edges())
        np.testing.assert_allclose(gradient(env, field), gradient(original, field))
        np.testing.assert_allclose(
            divergence(env, gradient(env, field)), expected_divergence
        )
        np.testing.assert_allclose(
            env.compute_kernel(bandwidth=3.0),
            original.compute_kernel(bandwidth=3.0),
            atol=1e-12,
        )


def test_missing_optional_descriptions_keep_authoritative_binning_geometry():
    """Descriptive grid arrays can be omitted when encoded components retain them."""
    grid = _four_connected_grid()
    data = to_dict(grid)
    for name in tuple(data["arrays"]):
        if name.startswith("grid_edges_"):
            data["arrays"].pop(name)
            data["array_specs"].pop(name)
    data.pop("grid_shape")
    restored = from_dict(data)
    np.testing.assert_array_equal(
        restored.bin_at(grid.bin_centers), np.arange(grid.n_bins)
    )

    hexagonal = Environment.from_layout(
        create_layout(
            "Hexagonal", hexagon_width=4.0, dimension_ranges=[(0, 20), (0, 20)]
        )
    )
    restored = from_dict(to_dict(hexagonal))
    assert tuple(restored.grid_edges) == ()
    assert tuple(restored.with_metadata().grid_edges) == ()
    np.testing.assert_array_equal(
        restored.bin_at(hexagonal.bin_centers), np.arange(hexagonal.n_bins)
    )


@pytest.mark.parametrize("how", ["dict", "file", "nwb"])
def test_reading_constructs_the_environment_once(how, tmp_path, monkeypatch):
    """Readers decode geometry and set the environment up exactly once."""
    import neurospatial.environment
    from neurospatial.environment import _publication

    original = _four_connected_grid()
    setup = _publication.publish
    calls = []

    def counting_setup(cls, binning, **kwargs):
        calls.append(cls)
        return setup(cls, binning, **kwargs)

    monkeypatch.setattr(_publication, "publish", counting_setup)
    monkeypatch.setattr(neurospatial.environment, "publish", counting_setup)

    with patch.object(
        _publication, "validate_components", wraps=_publication.validate_components
    ) as component_check:
        restored = _round_trip(original, how, tmp_path)
    assert component_check.call_count == 1
    assert len(calls) == 1
    assert list(restored.connectivity.edges()) == list(original.connectivity.edges())


def test_saved_bin_centers_must_match_the_authoritative_binning():
    data = to_dict(_four_connected_grid())
    data["arrays"]["bin_centers"][0][0] += 1.0
    with pytest.raises(
        ValueError, match=r"bin_centers.*authoritative binning centers"
    ) as exc:
        from_dict(data)
    assert "\nFix:" in str(exc.value)


@pytest.mark.parametrize(
    ("origin", "float32_resolves_bins"), [(0.0, True), (1e6, True), (1e7, False)]
)
def test_center_check_rejects_a_bin_shift_at_any_origin(origin, float32_resolves_bins):
    """Cell geometry catches shifted centers regardless of coordinate origin."""
    # Unit-width bins with half-integer centers, e.g. origin + 0.5, + 1.5, ...
    edges = origin + np.arange(11.0)
    env = Environment.from_grid_mask(
        np.ones((10, 10), dtype=bool), grid_edges=(edges,) * 2
    )
    stored = np.asarray(to_dict(env)["arrays"]["binning/bins/centers"])

    shifted = to_dict(env)
    centers = (stored + np.array([1.0, 0.0])).tolist()
    shifted["arrays"]["binning/bins/centers"] = centers
    shifted["arrays"]["bin_centers"] = centers
    for node in shifted["graph"]["nodes"]:
        node["pos"] = centers[node["id"]]
    with pytest.raises(ValueError, match=r"centers.*does not match"):
        from_dict(shifted)

    # float32 has unit spacing near 1e7: half-integer centers round by half a bin.
    float32 = to_dict(env)
    float32["arrays"]["binning/bins/centers"] = (
        stored.astype(np.float32).astype(float).tolist()
    )
    if float32_resolves_bins:
        assert from_dict(float32).n_bins == env.n_bins
    else:
        with pytest.raises(ValueError, match="centers"):
            from_dict(float32)


@pytest.mark.parametrize(
    ("active_bins", "neighbor_pair"),
    [((0, 1, 10, 20), (0, 1)), ((0, 9, 10, 20), (1, 2))],
)
def test_center_check_rejects_neighbor_substitution_in_sparse_mask(
    active_bins: tuple[int, ...], neighbor_pair: tuple[int, int]
) -> None:
    """Inactive stretches must not hide the separation of nearby active bins."""
    mask = np.zeros(21, dtype=bool)
    mask[list(active_bins)] = True
    env = Environment.from_grid_mask(mask, grid_edges=(1e6 + 0.1 * np.arange(22.0),))
    data = to_dict(env)
    restored = from_dict(data)
    np.testing.assert_array_equal(restored.bin_centers, env.bin_centers)

    rounded = to_dict(env)
    rounded["arrays"]["binning/bins/centers"] = (
        env.bin_centers.astype(np.float32).astype(float).tolist()
    )
    with pytest.raises(ValueError, match="centers"):
        from_dict(rounded)

    source, neighbor = neighbor_pair
    centers = data["arrays"]["binning/bins/centers"]
    centers[source] = centers[neighbor].copy()
    with pytest.raises(ValueError, match="centers"):
        from_dict(data)


def test_non_finite_saved_centers_are_rejected():
    data = to_dict(_four_connected_grid())
    data["arrays"]["binning/bins/centers"][0][0] = float("nan")
    with pytest.raises(ValueError, match=r"centers.*finite") as exc:
        from_dict(data)
    assert "\nFix:" in str(exc.value)


def test_saved_graph_is_rebuilt_from_the_binning():
    """The stored graph is for other tools; reading rebuilds it from the bins."""
    original = _four_connected_grid()
    data = to_dict(original)
    for link in data["graph"]["links"]:
        link.pop("edge_id")
        link["distance"] = -1.0
    restored = from_dict(data)
    assert list(restored.connectivity.edges(data=True)) == list(
        original.connectivity.edges(data=True)
    )


def test_native_faces_remain_independent_of_later_builder_graph_changes():
    layout = MaskedGridLayout()
    layout.build(
        active_mask=np.ones((3, 3), dtype=bool),
        grid_edges=(np.arange(4.0),) * 2,
        connect_diagonal_neighbors=False,
    )
    binning = layout.binning()
    expected = binning.faces.build()
    layout.connectivity.clear()
    actual = binning.faces.build()
    for first, second in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(first, second)


# ---------------------------------------------------------------------------
# Polar geometry belongs to the polar layout.
# ---------------------------------------------------------------------------
def test_polar_layout_supplies_annular_sector_geometry():
    """Asking the layout directly gives sector areas, not Cartesian dr*dtheta."""
    env = built_from(_polar)
    assert env._binning.rectilinear is not None
    distance_edges, angle_edges = env.grid_edges
    sectors = (0.5 * np.diff(distance_edges**2)[:, None] * np.diff(angle_edges)).ravel()
    np.testing.assert_allclose(env.bin_sizes, sectors)
    np.testing.assert_allclose(env.bin_sizes, sectors)
    graph, volumes = face_graph(env._binning)
    np.testing.assert_allclose(volumes, sectors)
    geometry = DiffusionGeometry.from_graph(graph, volumes)
    assert (geometry.W != env._diffusion_geometry.W).nnz == 0


def test_common_class_from_layout_builds_cartesian_geometry():
    env = Environment.from_layout(
        create_layout(
            "RegularGrid",
            bin_size=1.0,
            dimension_ranges=[(0.0, 2.0)] * 2,
            infer_active_bins=False,
        )
    )
    assert env.coordinate_kind == "cartesian"
    np.testing.assert_array_equal(env.bin_sizes, np.ones(env.n_bins))


@pytest.mark.parametrize("circular", [False, True])
@pytest.mark.parametrize("diagonal", [False, True])
def test_two_polar_sectors_aggregate_both_circular_interfaces(circular, diagonal):
    """Half-annuli have one shared face on an open axis and two on a circle."""
    env = built_from(
        lambda: Environment.from_polar_egocentric(
            distance_range=(1.0, 4.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=1.0,
            angle_bin_size=np.pi,
            circular_angle=circular,
            connect_diagonal_neighbors=diagonal,
        )
    )
    graph, _ = face_graph(env._binning)
    for ring in range(3):
        assert graph[2 * ring][2 * ring + 1]["A"] == pytest.approx(2 if circular else 1)
    for u, v, data in graph.edges(data=True):
        if (
            env.bin_centers[u, 0] != env.bin_centers[v, 0]
            and env.bin_centers[u, 1] != env.bin_centers[v, 1]
        ):
            assert data["A"] == 0


@pytest.mark.parametrize("circular", [False, True])
def test_two_polar_sectors_diffuse_at_the_physical_angular_rate(circular):
    """Diffusion agrees with the analytical solution for the two-cell system."""
    env = Environment.from_polar_egocentric(
        distance_range=(1.0, 2.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=1.0,
        angle_bin_size=np.pi,
        circular_angle=circular,
        connect_diagonal_neighbors=False,
    )
    radius, radial_width, angular_width = 1.5, 1.0, np.pi
    face_length = radial_width * (2 if circular else 1)
    conductance = face_length / (radius * angular_width)
    area = radius * radial_width * angular_width
    bandwidth = 2.0
    contrast = np.exp(-2 * conductance / area * bandwidth**2 / 2)
    expected = 0.5 * np.array([1 + contrast, 1 - contrast])
    np.testing.assert_allclose(
        env.diffuse(np.array([1.0, 0.0]), bandwidth=bandwidth, mode="average"), expected
    )


@pytest.mark.parametrize("circular", [False, True])
@pytest.mark.parametrize("diagonal", [False, True])
def test_polar_layout_builds_physical_geometry(circular, diagonal):
    """Edges, lengths, faces and areas match closed-form annular-sector geometry."""
    from neurospatial.layout.engines._polar import PolarGridLayout

    n_r, n_t, dr, dt = 3, 4, 2.0, np.pi / 2
    r_edges = 1.0 + dr * np.arange(n_r + 1)
    t_edges = np.linspace(-np.pi, np.pi, n_t + 1)
    layout = PolarGridLayout()
    layout.build(
        active_mask=np.ones((n_r, n_t), dtype=bool),
        grid_edges=(r_edges, t_edges),
        circular_angle=circular,
        connect_diagonal_neighbors=diagonal,
    )
    assert layout._build_params_used["circular_angle"] is circular

    radial, angular = (n_r - 1) * n_t, n_r * (n_t - 1)
    diagonals = 2 * (n_r - 1) * (n_t - 1) if diagonal else 0
    seams = (n_r + (2 * (n_r - 1) if diagonal else 0)) if circular else 0
    graph, volumes = face_graph(layout.binning())
    assert graph.number_of_edges() == radial + angular + diagonals + seams

    r, theta = layout.bin_centers.T
    for u, v, data in graph.edges(data=True):
        steps_r = round(abs(r[u] - r[v]) / dr)
        steps_t = round(abs(theta[u] - theta[v]) / dt)
        steps_t = min(steps_t, n_t - steps_t)  # a seam edge wraps to one step
        r_mid = 0.5 * (r[u] + r[v])
        face = {(1, 0): r_mid * dt, (0, 1): dr, (1, 1): 0.0}[(steps_r, steps_t)]
        assert data["distance"] == pytest.approx(
            np.hypot(steps_r * dr, steps_t * r_mid * dt)
        )
        assert data["A"] == pytest.approx(face)
    sectors = 0.5 * np.diff(r_edges**2)[:, None] * np.diff(t_edges)[None, :]
    np.testing.assert_allclose(volumes, sectors.ravel())


def test_masked_polar_layout_preserves_active_bin_indices():
    """Seams join active bins only, using their renumbered graph indices."""
    from neurospatial.layout.engines._polar import PolarGridLayout

    mask = np.array([[True, False, True, True], [False, True, True, True]])
    layout = PolarGridLayout()
    layout.build(
        active_mask=mask,
        grid_edges=(np.arange(3.0) * 2, np.linspace(-np.pi, np.pi, 5)),
        circular_angle=True,
        connect_diagonal_neighbors=True,
    )
    assert set(layout.connectivity.nodes()) == set(range(mask.sum()))
    by_flat = {
        data["source_grid_flat_index"]: node
        for node, data in layout.connectivity.nodes(data=True)
    }
    assert layout.connectivity.has_edge(by_flat[0], by_flat[3])
    assert layout.connectivity.has_edge(by_flat[0], by_flat[7])
    assert len(layout.connectivity.nodes()) == mask.sum()
    graph, volumes = face_graph(layout.binning())
    assert DiffusionGeometry.from_graph(graph, volumes).W.shape == (6, 6)


def test_polar_rebuild_refreshes_active_bin_mapping():
    """Reusing a layout with a new mask must rebuild seam and lookup indices."""
    from neurospatial.layout.engines._polar import PolarGridLayout

    edges = (np.arange(3.0) * 2, np.linspace(-np.pi, np.pi, 5))
    layout = PolarGridLayout()
    layout.build(active_mask=np.ones((2, 4), dtype=bool), grid_edges=edges)
    mask = np.array([[True, False, True, True], [False, True, True, True]])
    layout.build(active_mask=mask, grid_edges=edges)

    expected = PolarGridLayout()
    expected.build(active_mask=mask, grid_edges=edges)
    assert list(layout.connectivity.edges()) == list(expected.connectivity.edges())
    np.testing.assert_array_equal(
        layout.binning().locator.locate(layout.bin_centers), np.arange(mask.sum())
    )
    graph, volumes = face_graph(layout.binning())
    assert DiffusionGeometry.from_graph(graph, volumes).W.shape == (6, 6)


@pytest.mark.parametrize("circular", [False, True])
def test_legacy_polar_parameters_restore_saved_connectivity(circular):
    """Old dictionaries without the polar build option keep their saved graph."""
    original = _polar(circular_angle=circular)
    data = to_dict(original)
    data["layout_parameters"].pop("circular_angle")
    restored = from_dict(data)
    assert restored._angular_is_circular is circular
    assert list(restored.connectivity.edges()) == list(original.connectivity.edges())
    np.testing.assert_allclose(
        restored.compute_kernel(bandwidth=3.0), original.compute_kernel(bandwidth=3.0)
    )


# ---------------------------------------------------------------------------
# Capability protocols and actionable errors.
# ---------------------------------------------------------------------------
def test_capability_protocol_reports_what_layouts_can_supply(reloaded_graph_env):
    for env in (_four_connected_grid(), _polar(), reloaded_graph_env):
        assert env._binning.faces.available
        assert len(face_geometry(env._cache, env._binning).volumes) == env.n_bins


@pytest.mark.parametrize(
    ("call", "operation"),
    [
        (lambda env: env.smooth(np.ones(env.n_bins), bandwidth=3.0), "smoothing"),
        (lambda env: env.compute_kernel(bandwidth=3.0), "smoothing"),
        (lambda env: env.diffuse(np.ones(env.n_bins), bandwidth=3.0), "smoothing"),
        (lambda env: gradient(env, np.ones(env.n_bins)), "gradient/divergence"),
        (
            lambda env: divergence(env, np.ones(env.connectivity.number_of_edges())),
            "gradient/divergence",
        ),
        (
            lambda env: heat_kernel_wavelet_basis(env, centers=np.array([0])),
            "heat_kernel_wavelet_basis",
        ),
    ],
)
def test_unsupported_transform_errors_name_the_operation_and_recovery(call, operation):
    env = _four_connected_grid().apply_transform(
        Affine2D(np.array([[1.0, 0.25, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    )
    with pytest.raises(NotImplementedError) as exc:
        call(env)
    message = str(exc.value)
    assert message.startswith(f"{operation} needs ")
    assert "shear" in message.lower() or "affine" in message.lower()
    assert "right angle" in message
    assert "\nWhy:" in message
    assert "\nFix:" in message
    assert "original environment" in message
    assert "rebuild" in message
    assert "implement" not in message


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiGraph])
def test_compute_diffusion_kernels_rejects_graphs_that_sum_edges(graph_type):
    """Reciprocal or parallel edges would double a conductance; say how to fix."""
    graph = nx.path_graph(4, create_using=graph_type)
    nx.set_edge_attributes(graph, 1.0, "A")
    nx.set_edge_attributes(graph, 1.0, "distance")
    with pytest.raises(ValueError, match=r"undirected simple nx\.Graph.*nx\.Graph\("):
        compute_diffusion_kernels(graph, volumes=np.ones(4), sigma=1.0, mode="average")
    kernel = compute_diffusion_kernels(
        nx.Graph(graph), volumes=np.ones(4), sigma=1.0, mode="average"
    )
    np.testing.assert_allclose(kernel.sum(axis=1), 1.0)


def test_dense_kernel_warning_points_at_the_callers_line(monkeypatch, face_env):
    monkeypatch.setattr(_constants, "LARGE_KERNEL_THRESHOLD", 1)
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 1.0, "A")
    nx.set_edge_attributes(graph, 1.0, "distance")
    with pytest.warns(UserWarning, match="dense diffusion kernel") as record:
        compute_diffusion_kernels(graph, volumes=np.ones(3), sigma=1.0, mode="average")
        face_env.compute_kernel(bandwidth=1.0, cache=False)
    dense = [w for w in record if "dense diffusion kernel" in str(w.message)]
    assert [w.filename for w in dense] == [__file__, __file__]


def test_mesh_skew_warning_is_raised_once_per_geometry_at_the_callers_line():
    shear = 6.0
    corners = [(x + shear * y, y) for x, y in [(-30, -8), (30, -8), (30, 8), (-30, 8)]]
    env = Environment.from_layout(
        create_layout(
            "TriangularMesh", boundary_polygon=Polygon(corners), point_spacing=4.0
        )
    )
    with pytest.warns(UserWarning, match="non-K-orthogonal") as record:
        env.compute_kernel(bandwidth=3.0)
    skew = [w for w in record if "non-K-orthogonal" in str(w.message)]
    assert [w.filename for w in skew] == [__file__]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        env.compute_kernel(bandwidth=5.0, cache=False)  # the cached geometry is reused
    env.clear_cache()
    with pytest.warns(UserWarning, match="non-K-orthogonal"):
        env.smooth(np.ones(env.n_bins), bandwidth=3.0)


# ---------------------------------------------------------------------------
# Invalid layout geometry fails at the environment boundary.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("volumes", "scale", "match"),
    [
        (np.ones(4), 1.0, r"volumes must have shape \(3,\)"),
        (np.array([1.0, -2.0, 4.0]), 1.0, "strictly positive"),
        (np.array([1.0, 2.0, 4.0]), -1.0, "invalid face measure"),
    ],
)
def test_invalid_layout_faces_fail_before_publication(volumes, scale, match):
    """Invalid provider data must be repaired before an owned snapshot is made."""
    layout = MaskedGridLayout()
    layout.build(active_mask=np.ones(3, dtype=bool), grid_edges=(np.arange(4.0),))
    graph = layout.connectivity.copy()
    for (u, v), area in zip(graph.edges(), (2.0, 3.0), strict=True):
        graph.edges[u, v]["A"] = area * scale
    with pytest.raises(ValueError, match=match):
        FaceGeometry.from_graph(graph, volumes)
