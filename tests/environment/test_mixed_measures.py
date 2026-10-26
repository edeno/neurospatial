"""A track combined with an open field keeps each bin's own measure."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from shapely.geometry import LineString, box

from neurospatial import Environment
from neurospatial.arena import OpenArena
from neurospatial.binning import SpatialBins
from neurospatial.encoding import compute_spatial_rate
from neurospatial.ops.transforms import AffineND, scale_2d, translate


def _track(**options):
    # An approach corridor ending 3 cm from the field's west edge.
    return Environment.linear_track(
        endpoints=[[-3.0, 5.0], [-23.0, 5.0]],
        bin_size=2.0,
        track_width=2.0,
        units="cm",
        **options,
    )


def _field():
    # A 10 x 10 cm floor of 2 cm bins.
    edges = np.arange(0.0, 11.0, 2.0)
    return Environment.from_grid_mask(
        np.ones((5, 5), dtype=bool), grid_edges=(edges, edges), units="cm"
    )


@pytest.fixture(scope="module")
def corridor_and_field():
    return Environment.combine({"track": _track(), "field": _field()})


def _component(env, key):
    return np.array([source == key for source, _ in env.bin_sources])


def test_each_bin_keeps_its_components_measure(corridor_and_field):
    env = corridor_and_field
    bins = env.binning.bins
    track, field = _component(env, "track"), _component(env, "field")
    np.testing.assert_array_equal(bins.measure_dims[track], 1)
    np.testing.assert_array_equal(bins.measure_dims[field], 2)
    assert bins.measure_dim is None
    np.testing.assert_allclose(bins.sizes[track], 2.0)  # cm of track
    np.testing.assert_allclose(bins.sizes[field], 4.0)  # cm^2 of floor
    summary = env.info()
    assert "Length of 10 bins" in summary
    assert "Area of 25 bins" in summary


def test_bin_sizes_refuses_to_mix_lengths_and_areas(corridor_and_field):
    env = corridor_and_field
    with pytest.raises(
        ValueError,
        match=r"10 by length \(cm\), 25 by area \(cm\^2\)(.|\n)*env\.subset",
    ):
        env.bin_sizes  # noqa: B018
    field = env.subset(mask=_component(env, "field"))
    np.testing.assert_allclose(field.bin_sizes, 4.0)


def test_diffusion_stays_within_a_component_without_a_bridge_conductance(
    corridor_and_field,
):
    env = corridor_and_field
    track = _component(env, "track")
    mass = np.where(np.arange(env.n_bins) == np.flatnonzero(track)[0], 1.0, 0.0)
    spread = env.diffuse(mass, bandwidth=6.0, mode="transition")
    # The navigation bridge carries no diffusion, so no mass reaches the field.
    np.testing.assert_allclose(spread[~track], 0.0, atol=1e-12)
    assert spread[track].sum() == pytest.approx(1.0)


def test_rate_maps_and_occupancy_cover_both_components(corridor_and_field):
    env = corridor_and_field
    rng = np.random.default_rng(1)
    times = np.arange(0.0, 60.0, 1 / 30)
    on_track = times < 20
    positions = np.column_stack(
        [
            np.where(on_track, -23.0 + times, rng.uniform(0, 10, times.size)),
            np.where(on_track, 5.0, rng.uniform(0, 10, times.size)),
        ]
    )
    spike_times = np.sort(np.random.default_rng(2).uniform(0, 60, 300))
    occupancy = env.occupancy(times, positions)
    assert occupancy[_component(env, "track")].sum() > 0
    assert occupancy[_component(env, "field")].sum() > 0
    rates = compute_spatial_rate(
        env, spike_times, times, positions, method="binned"
    ).firing_rate
    assert np.isfinite(rates[occupancy > 0]).all()


def test_transforms_scale_each_measure_by_its_own_dimension(corridor_and_field):
    env = corridor_and_field
    bins = env.binning.bins
    moved = env.apply_transform(translate(5.0, -2.0)).binning.bins
    np.testing.assert_array_equal(moved.sizes, bins.sizes)
    np.testing.assert_array_equal(moved.measure_dims, bins.measure_dims)
    scaled = env.apply_transform(scale_2d(2.0), units="mm").binning.bins
    ratio = scaled.sizes / bins.sizes
    np.testing.assert_allclose(ratio[bins.measure_dims == 1], 2.0)
    np.testing.assert_allclose(ratio[bins.measure_dims == 2], 4.0)


def test_mixed_measures_survive_storage(corridor_and_field, tmp_path):
    env = corridor_and_field
    corridor_and_field.to_file(tmp_path / "corridor")
    for restored in (
        Environment.from_dict(env.to_dict()),
        Environment.from_file(tmp_path / "corridor"),
    ):
        np.testing.assert_array_equal(
            restored.binning.bins.measure_dims, env.binning.bins.measure_dims
        )


def test_a_conductance_cannot_join_a_track_bin_to_a_field_bin():
    # Diffusion between a density per length and a density per area changes
    # with the units, whatever the conductance, so it is refused.
    track, field = _track(), _field()
    row = Environment.combine({"track": track, "field": field})
    row = row.get_bridge_attributes().iloc[0]
    key = (
        (row["source_component"], int(row["source_local_bin"])),
        (row["target_component"], int(row["target_local_bin"])),
    )
    with pytest.raises(
        ValueError,
        match=r"conducting bridge joins bin \d+, measured in dimension [12], to bin \d+",
    ):
        Environment.combine(
            {"track": track, "field": field}, bridge_conductances={key: 1.0}
        )
    navigation_only = Environment.combine(
        {"track": track, "field": field}, bridge_conductances={key: 0.0}
    )
    assert navigation_only.get_bridge_attributes()["conductance"].tolist() == [0.0]


@pytest.mark.parametrize("derive", ["none", "subset", "transform"])
def test_published_measure_dimensions_must_match_the_components(
    corridor_and_field, derive
):
    env = corridor_and_field
    if derive == "subset":
        env = env.subset(mask=np.arange(env.n_bins) != 5)
    elif derive == "transform":
        env = env.apply_transform(scale_2d(2.0), units="mm")
    binning = replace(env.binning, kind="corridor_and_field")
    dims = binning.bins.measure_dims.copy()
    track = np.flatnonzero(dims == 1)[0]
    field = np.flatnonzero(dims == 2)[0]
    dims[[track, field]] = dims[[field, track]]
    swapped = replace(binning, bins=replace(binning.bins, measure_dims=dims))
    with pytest.raises(ValueError, match=r"bins\.measure_dims does not match"):
        Environment.from_bins(swapped, arena=env.arena)
    Environment.from_bins(binning, arena=env.arena)  # the unswapped bins publish


def test_default_bandwidth_uses_each_bins_own_length():
    # One bin per component and no bridge: no edge lengths to take a median of.
    track = _track().with_metadata(units=None)
    field = _field().with_metadata(units=None)
    lone = Environment.combine(
        {
            "track": track.subset(mask=np.arange(track.n_bins) == 0),
            "field": field.subset(mask=np.arange(field.n_bins) == 0),
        },
        auto_bridge=False,
    )
    lengths = lone.coordinate_system.fallback_lengths(lone)
    np.testing.assert_allclose(lengths, [2.0, 2.0])  # 2 cm of track, a 2 cm square
    times = np.arange(0.0, 10.0, 0.1)
    positions = np.tile([1.0, 1.0], (times.size, 1))
    result = compute_spatial_rate(
        lone, np.array([1.0]), times, positions, method="diffusion_kde"
    )
    assert result.bandwidth == pytest.approx(5.0)  # 2.5 bin lengths


def test_combined_tracks_report_the_extent_of_both():
    first = _track()
    second = Environment.linear_track(
        endpoints=[[-3.0, 15.0], [-23.0, 15.0]], bin_size=2.0, track_width=2.0
    ).with_metadata(units="cm")
    env = Environment.combine({"first": first, "second": second})
    np.testing.assert_allclose(env.dimension_ranges, [(-23.0, -3.0), (5.0, 15.0)])


@pytest.mark.parametrize(
    ("measure_dim", "match"),
    [
        (np.array([1, 2]), r"measure_dims must be one integer per bin, shape \(3,\)"),
        (np.array([1, 2, 3]), r"measure_dims\[2\] must be between 1 and 2"),
        (np.array([1.0, 2.0, 2.0]), "measure_dims must be one integer per bin"),
        (True, "measure_dim must be an integer between 1 and 2"),
    ],
)
def test_spatial_bins_validate_each_bins_measure_dimension(measure_dim, match):
    with pytest.raises(ValueError, match=match):
        SpatialBins.from_arrays(np.zeros((3, 2)), np.ones(3), measure_dim)


def test_blocked_edges_name_walls_of_a_component_next_to_a_track():
    edges = np.arange(0.0, 11.0, 2.0)
    walled = Environment.from_grid_mask(
        np.ones((5, 5), dtype=bool),
        grid_edges=(edges, edges),
        units="cm",
        arena=OpenArena(
            box(0, 0, 10, 10), walls={"divider": LineString([(4, 0), (4, 10)])}
        ),
    )
    env = Environment.combine({"track": _track(), "field": walled})
    table = env.get_blocked_edges()
    assert len(table) == len(walled.get_blocked_edges())
    assert set(table["wall"]) == {"divider"}
    field_bins = _component(env, "field")
    assert field_bins[table["source"]].all() and field_bins[table["target"]].all()
    subset = env.subset(mask=np.ones(env.n_bins, dtype=bool))
    assert set(subset.get_blocked_edges()["wall"]) == {"divider"}
    rotated = env.apply_transform(translate(5.0, 0.0)).apply_transform(
        AffineND(np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    )
    assert list(rotated.get_blocked_edges()["wall"]) == list(table["wall"])


@pytest.mark.parametrize(("area", "accepted"), [(1.0, False), (0.0, True)])
def test_a_conducting_face_cannot_join_bins_of_different_measure(area, accepted):
    from neurospatial.binning import (
        BinConnections,
        Binning,
        FaceGeometry,
        KnownFaces,
        NearestBinLocator,
    )

    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    binning = Binning(
        "mixed",
        SpatialBins.from_arrays(centers, [1.0, 1.0], [1, 2]),
        BinConnections.from_arrays([[0, 1]], [1.0]),
        NearestBinLocator(centers, max_distance=0.4),
        KnownFaces(
            FaceGeometry(
                np.array([[0, 1]]),
                np.array([area]),
                np.array([1.0]),
                np.array([1.0, 1.0]),
            )
        ),
    )
    if accepted:
        Environment.from_bins(binning)  # a navigation-only link
    else:
        with pytest.raises(
            ValueError,
            match=r"conducting face joins bin 0, measured in dimension 1, to bin 1, measured in dimension 2",
        ):
            Environment.from_bins(binning)
