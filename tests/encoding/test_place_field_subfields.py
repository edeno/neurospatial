"""Two fields that merge at the threshold are separated (``subfields=``).

With ``"core"`` the subfield rule replaces a field by its core when a second
field shares the region. The next candidate peak can then sit on the core's
flank, with a region that runs into the second field. That candidate is a
flank and is discarded, but the second field must still be found from its own
peak (``detect_place_fields`` Notes, steps 4 and 5). With ``"watershed"`` the
same fields are divided at the valley between them (Notes, step 4).
"""

from __future__ import annotations

import heapq

import networkx as nx
import numpy as np
import pytest
from scipy import ndimage
from skimage.segmentation import watershed

from neurospatial import Environment
from neurospatial.encoding import compute_spatial_rates, detect_place_fields
from neurospatial.encoding.population import population_coverage


@pytest.fixture(scope="module")
def box():
    """A 100 x 100 cm open field of 2.5 cm bins."""
    edges = np.arange(0.0, 100.1, 2.5)
    return Environment.from_grid_mask(
        np.ones((40, 40), dtype=bool), grid_edges=(edges, edges), units="cm"
    )


def _two_fields(env, second_peak, separation):
    centers = env.bin_centers
    first = np.array([35.0, 50.0])
    second = first + np.array([separation, 0.0])
    width = 8.0
    rate = 10.0 * np.exp(-np.sum((centers - first) ** 2, axis=1) / (2 * width**2))
    rate += second_peak * np.exp(
        -np.sum((centers - second) ** 2, axis=1) / (2 * width**2)
    )
    return rate, first, second


# The second peak (3 Hz) is below the first field's flank just outside its
# core (0.36 x 10 = 3.6 Hz), so the next candidate after the split is on that
# flank, and its region runs into the second field.
@pytest.mark.parametrize(("second_peak", "separation"), [(3.3, 24.0), (3.0, 26.0)])
def test_neighbouring_field_is_found_after_a_subfield_split(
    box, second_peak, separation
):
    rate, first, second = _two_fields(box, second_peak, separation)
    # The saddle lies above the 0.2 floor, so the two fields form one region
    # at the first peak's threshold, and the subfield rule must split it.
    saddle = rate[box.bin_at(((first + second) / 2)[None])[0]] / rate.max()
    assert saddle > 0.2

    result = detect_place_fields(box, rate, bandwidth=0.0)

    peaks = [box.bin_centers[f[np.argmax(rate[f])]] for f in result.fields]
    assert len(result.fields) == 2
    np.testing.assert_allclose(peaks[0], first, atol=2.5)
    # The sum of two Gaussians pulls the second maximum toward the first
    # field by about one bin.
    np.testing.assert_allclose(peaks[1], second, atol=5.0)
    # The two fields share no bin.
    assert not set(result.fields[0].tolist()) & set(result.fields[1].tolist())


def _core_fields_from_the_image(env, rate, threshold=0.2):
    """The two ``"core"`` fields of a strong and a weak field, from the image.

    Regions are labeled with ``scipy.ndimage`` on the 40 x 40 grid image (8
    neighbors, as the box's bins). The strong field is the core around the
    map's maximum. The weak field is the region around the other local
    maximum at ``threshold`` times that peak, among the bins outside the
    core that are not above that peak (a higher bin there is a flank of the
    strong field).
    """
    cells = np.round((env.bin_centers - 1.25) / 2.5).astype(int)
    image = np.zeros((40, 40))
    image[cells[:, 1], cells[:, 0]] = rate

    def region(mask, seed):
        labels = ndimage.label(mask, structure=np.ones((3, 3)))[0]
        return labels == labels[seed]

    # Bins with no higher neighbor. The fields are mirror images about
    # y = 50, so each peak is two equal bins, one above the other.
    summits = np.argwhere(image == ndimage.maximum_filter(image, size=3))
    strong = tuple(summits[np.argmax(image[tuple(summits.T)])])
    beyond = summits[summits[:, 1] > strong[1] + 4]  # more than 10 cm east
    weak = tuple(beyond[np.argmax(image[tuple(beyond.T)])])
    floor = threshold * image[strong]
    core_level = floor + threshold * (image[strong] - floor)
    whole = region(image >= floor, strong)
    core = region(image >= core_level, strong)
    assert core.sum() < whole.sum() / 2  # the subfield rule splits the region
    free = ~core & (image <= image[weak])
    weak_floor = threshold * image[weak]
    weak_whole = region(free & (image >= weak_floor), weak)
    weak_core = region(
        free & (image >= weak_floor + threshold * (image[weak] - weak_floor)), weak
    )
    assert weak_core.sum() >= weak_whole.sum() / 2  # the weak field is not split
    return [
        np.flatnonzero(mask[cells[:, 1], cells[:, 0]]) for mask in (core, weak_whole)
    ]


@pytest.mark.parametrize(
    ("second_peak", "separation", "n_bins", "n_nearer_strong", "centroid_x"),
    [(3.3, 24.0, 184, 80, 47.03), (3.0, 26.0, 192, 78, 48.68)],
)
def test_core_gives_the_weaker_field_the_stronger_fields_skirt(
    box, second_peak, separation, n_bins, n_nearer_strong, centroid_x
):
    """With ``"core"`` the weak field wraps around the strong one.

    The strong field keeps only its core. Its skirt stays unassigned and
    joins the weak field's region, which is larger than the weak field and
    has its centroid pulled toward the strong field (``subfields`` in
    ``detect_place_fields``); ``"watershed"`` divides the two at the valley.
    """
    rate, first, second = _two_fields(box, second_peak, separation)
    centers = box.bin_centers

    def nearer_strong(bins):
        to_first = np.linalg.norm(centers[bins] - first, axis=1)
        return int(np.sum(to_first < np.linalg.norm(centers[bins] - second, axis=1)))

    result = detect_place_fields(box, rate, bandwidth=0.0, subfields="core")

    strong, weak = result.fields
    expected_strong, expected_weak = _core_fields_from_the_image(box, rate)
    np.testing.assert_array_equal(strong, expected_strong)
    np.testing.assert_array_equal(weak, expected_weak)
    assert nearer_strong(strong) >= len(strong) - 2  # the core, around its peak
    # The weak field: the documented size, with the skirt on every side of
    # the strong peak, the far side included.
    assert len(weak) == n_bins
    assert nearer_strong(weak) == n_nearer_strong
    skirt = centers[weak] - first
    assert skirt[:, 0].min() < -12.5  # west of the strong peak, away from the weak
    assert skirt[:, 1].min() < -12.5
    assert skirt[:, 1].max() > 12.5
    table = result.fields_table()
    assert table["centroid_x"][1] == pytest.approx(centroid_x, abs=0.01)
    assert second[0] - table["centroid_x"][1] > 11.5  # pulled toward the strong field

    divided = detect_place_fields(box, rate, bandwidth=0.0, subfields="watershed")

    assert nearer_strong(divided.fields[1]) == 0
    assert len(divided.fields[1]) < len(weak) / 2
    np.testing.assert_allclose(
        _centroid(box, rate, divided.fields[1]), second, atol=1.5
    )


@pytest.mark.parametrize("subfields", [True, False, "Core", "basin"])
def test_unknown_subfield_rule_is_refused(box, subfields):
    rate, _, _ = _two_fields(box, 3.3, 24.0)
    with pytest.raises(ValueError, match="subfields must be one of"):
        detect_place_fields(box, rate, bandwidth=0.0, subfields=subfields)


# ---------------------------------------------------------------------------
# subfields="watershed"
# ---------------------------------------------------------------------------


def _gaussian(env, center, peak, width=8.0):
    squared = np.sum((env.bin_centers - np.asarray(center)) ** 2, axis=1)
    return peak * np.exp(-squared / (2 * width**2))


def _centroid(env, rate, bins):
    return (env.bin_centers[bins] * rate[bins, None]).sum(axis=0) / rate[bins].sum()


def _joined_to(env, seed, mask):
    """Sorted bins of ``mask`` joined to ``seed`` through ``env.connectivity``.

    A networkx component of the induced subgraph: the expected fields below
    do not use the detector's own flood fill.
    """
    inside = env.connectivity.subgraph(np.flatnonzero(mask).tolist())
    return np.array(sorted(nx.node_connected_component(inside, seed)))


def _planted_peak(env, rate, center, radius=5.0):
    """The highest bin within ``radius`` of a planted field center.

    The sum of two fields moves the weaker maximum toward the stronger field
    by about a bin, so the summit is searched near the center.
    """
    near = np.linalg.norm(env.bin_centers - np.asarray(center), axis=1) < radius
    return int(np.argmax(np.where(near, rate, -np.inf)))


def _dominance_fields(env, components, threshold=0.2):
    """Bins where each component is the larger, at ``threshold`` of its peak."""
    rate = components[0] + components[1]
    fields = []
    for own, other in (components, components[::-1]):
        mine = own > other
        peak = int(np.argmax(np.where(mine, rate, -np.inf)))
        fields.append(_joined_to(env, peak, mine & (rate >= threshold * rate[peak])))
    return fields


def _flooded_fields(env, rate, peaks, threshold=0.2):
    """Watershed fields of ``peaks`` by flooding ``env.connectivity`` directly.

    ``detect_place_fields`` Notes, step 4: from all peaks at once, the
    highest flooded bin (at equal rates the lowest bin index) gives its basin
    to its unflooded finite neighbors. Each field is the part of its peak's
    basin joined to the peak with a rate from ``threshold`` times the peak up
    to the peak.
    """
    basin = np.full(env.n_bins, -1)
    basin[list(peaks)] = np.arange(len(peaks))
    flooded = [(-rate[peak], peak) for peak in peaks]
    heapq.heapify(flooded)
    while flooded:
        _, highest = heapq.heappop(flooded)
        for neighbor in env.connectivity.neighbors(highest):
            if basin[neighbor] < 0 and np.isfinite(rate[neighbor]):
                basin[neighbor] = basin[highest]
                heapq.heappush(flooded, (-rate[neighbor], neighbor))
    return [
        _joined_to(
            env,
            peak,
            (basin == k) & (rate >= threshold * rate[peak]) & (rate <= rate[peak]),
        )
        for k, peak in enumerate(peaks)
    ]


@pytest.fixture(scope="module")
def track():
    """A 200 cm linear track of 2 cm bins."""
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(200.0, 0.0))
    graph.add_edge(0, 1)
    return Environment.from_graph(graph, [(0, 1)], bin_size=2.0, units="cm")


@pytest.fixture(scope="module")
def hexagons():
    """A 100 x 100 cm open field of 4 cm hexagonal bins."""
    positions = np.random.default_rng(5).uniform(0, 100, (20000, 2))
    return Environment.from_samples(
        positions, bin_size=4.0, layout="hexagonal", units="cm"
    )


def _skimage_watershed_fields(env, rate, peaks, threshold=0.2):
    """The same fields from scikit-image's marker watershed on the grid image.

    The box's bins are a 40 x 40 grid joined to all 8 neighbors, so
    ``connectivity=2`` floods the same neighbor pairs.
    """
    cells = np.round((env.bin_centers - 1.25) / 2.5).astype(int)
    image = np.zeros((40, 40))
    image[cells[:, 1], cells[:, 0]] = rate
    markers = np.zeros((40, 40), dtype=int)
    for label, peak in enumerate(peaks, start=1):
        markers[cells[peak, 1], cells[peak, 0]] = label
    basins = watershed(-image, markers, connectivity=2)
    fields = []
    for label, peak in enumerate(peaks, start=1):
        mask = (basins == label) & (image >= threshold * rate[peak])
        components = ndimage.label(mask, structure=np.ones((3, 3)))[0]
        mask = components == components[cells[peak, 1], cells[peak, 0]]
        fields.append(np.flatnonzero(mask[cells[:, 1], cells[:, 0]]))
    return fields


def test_watershed_divides_touching_fields_at_the_valley(box):
    """A weak field next to a strong one keeps its own place and size.

    10 Hz at (35, 50) and 3.3 Hz at (59, 50), 8 cm wide: with ``"core"`` the
    second field grows around the first (184 bins, centroid x = 47.0 cm).
    Alone, each covers 112 and 104 bins at 20% of its peak.
    """
    first_center, second_center = np.array([35.0, 50.0]), np.array([59.0, 50.0])
    components = (
        _gaussian(box, first_center, 10.0),
        _gaussian(box, second_center, 3.3),
    )
    rate = components[0] + components[1]
    core = detect_place_fields(box, rate, bandwidth=0.0, subfields="core")
    result = detect_place_fields(box, rate, bandwidth=0.0, subfields="watershed")

    assert len(result) == len(core) == 2
    peaks = [int(field[np.argmax(rate[field])]) for field in result]
    assert peaks == [int(field[np.argmax(rate[field])]) for field in core]
    # An independent marker watershed (scikit-image) gives the same bins, and
    # so does flooding the bin graph, the reference for the other layouts.
    assert peaks == [
        _planted_peak(box, rate, center) for center in (first_center, second_center)
    ]
    for field, expected, flooded in zip(
        result,
        _skimage_watershed_fields(box, rate, peaks),
        _flooded_fields(box, rate, peaks),
        strict=True,
    ):
        np.testing.assert_array_equal(field, expected)
        np.testing.assert_array_equal(flooded, expected)

    # Each centroid is within one bin of its field's center (36.1 and
    # 60.4 cm measured); "core" puts the second one 12 cm off.
    for field, center in zip(result, (first_center, second_center), strict=True):
        np.testing.assert_allclose(_centroid(box, rate, field), center, atol=2.5)
    # Sizes are near the bins each Gaussian dominates (116 and 94); the
    # valley of the summed map lies about 3.5 cm nearer the weaker field than
    # the line where the two are equal, so the measured 122 and 82 differ by
    # 5% and 13%.
    for field, dominated in zip(
        result, _dominance_fields(box, components), strict=True
    ):
        assert abs(len(field) - len(dominated)) <= 0.2 * len(dominated)
    assert not set(result[0].tolist()) & set(result[1].tolist())


def test_watershed_on_a_track_ends_each_field_at_the_valley(track):
    """On a line the basins meet at the lowest rate between the two peaks.

    10 Hz at 60 cm and 3.3 Hz at 88 cm merge at the first field's threshold.
    A 2.5 Hz field at 150 cm is above that threshold too, but nothing joins
    it to them, so it is a field of its own.
    """
    x = track.bin_centers[:, 0]
    centers = (60.0, 88.0, 150.0)
    rate = sum(
        peak * np.exp(-((x - center) ** 2) / 128)
        for center, peak in zip(centers, (10.0, 3.3, 2.5), strict=True)
    )
    core = detect_place_fields(track, rate, bandwidth=0.0, subfields="core")
    result = detect_place_fields(track, rate, bandwidth=0.0, subfields="watershed")
    assert len(core) == len(result) == 3

    # By hand, along the bins in order of position. The valley is the lowest
    # bin between the two planted centers; the flood reaches it from its
    # higher neighbor, so it goes to that side. The far field's side starts
    # at the lowest bin before it.
    order = np.argsort(x)
    along, rate_along = x[order], rate[order]

    def lowest_between(low, high):
        between = np.flatnonzero((along > low) & (along < high))
        return int(between[np.argmin(rate_along[between])])

    valley = lowest_between(centers[0], centers[1])
    to_first = rate_along[valley - 1] > rate_along[valley + 1]
    first_end = valley + 1 if to_first else valley
    gap = lowest_between(centers[1], centers[2])
    assert rate_along[gap] < 0.2 * 2.5  # below every field's threshold
    for (start, stop), field in zip(
        ((0, first_end), (first_end, gap), (gap, len(along))), result, strict=True
    ):
        summit = start + int(np.argmax(rate_along[start:stop]))
        floor = 0.2 * rate_along[summit]
        low = high = summit
        while low > start and rate_along[low - 1] >= floor:
            low -= 1
        while high + 1 < stop and rate_along[high + 1] >= floor:
            high += 1
        np.testing.assert_array_equal(field, np.sort(order[low : high + 1]))
    # The same intervals, read off the map: 2 cm bins centered on odd cm.
    assert [(x[f].min(), x[f].max()) for f in result] == [
        (47.0, 77.0),
        (79.0, 101.0),
        (137.0, 163.0),
    ]
    # "core" lets the second field run through the first's edge (centroid
    # 85.6 cm for a field at 88); the watershed keeps it on its peak (88.6).
    second_field = result[1]
    assert abs(_centroid(track, rate, second_field)[0] - 88.0) < 1.0


def test_watershed_on_hexagonal_bins(hexagons):
    """A non-grid layout: the weak field stays on its own peak.

    A third field far from the two that touch is above the strong field's
    threshold, but no bin joins it to them, so it stays a field of its own.
    """
    centers = (np.array([30.0, 30.0]), np.array([54.0, 30.0]), np.array([80.0, 80.0]))
    components = tuple(
        _gaussian(hexagons, center, peak)
        for center, peak in zip(centers, (10.0, 3.3, 2.5), strict=True)
    )
    rate = sum(components)
    core = detect_place_fields(hexagons, rate, bandwidth=0.0, subfields="core")
    result = detect_place_fields(hexagons, rate, bandwidth=0.0, subfields="watershed")
    assert len(core) == len(result) == 3
    # The fields of the planted summits, flooded on the bin graph.
    peaks = [_planted_peak(hexagons, rate, center) for center in centers]
    for field, expected in zip(
        result, _flooded_fields(hexagons, rate, peaks), strict=True
    ):
        np.testing.assert_array_equal(field, expected)
    # Measured centroids 31.5 and 56.3 cm; "core" puts the second at 42.5 cm.
    for field, center in zip(result, centers, strict=True):
        np.testing.assert_allclose(_centroid(hexagons, rate, field), center, atol=4.0)
    # Measured 53 and 33 bins against 48 and 41 dominated bins: the valley
    # lies nearer the weaker field, as on the grid.
    for field, dominated in zip(
        result.fields[:2], _dominance_fields(hexagons, components[:2]), strict=True
    ):
        assert abs(len(field) - len(dominated)) <= 0.25 * len(dominated)


@pytest.mark.parametrize(
    "centers_peaks",
    [
        [((50.0, 50.0), 6.0)],
        [((25.0, 25.0), 9.0), ((75.0, 75.0), 4.0)],
        [((30.0, 50.0), 8.0), ((70.0, 50.0), 8.0)],
        [((2.0, 50.0), 7.0)],
    ],
)
def test_watershed_leaves_separate_fields_as_they_are(box, centers_peaks):
    """Fields that do not meet are the same with every rule."""
    rate = sum(_gaussian(box, center, peak) for center, peak in centers_peaks)
    expected = detect_place_fields(box, rate, bandwidth=0.0, subfields=None)
    result = detect_place_fields(box, rate, bandwidth=0.0, subfields="watershed")
    assert len(result) == len(expected) == len(centers_peaks)
    for field, other in zip(result, expected, strict=True):
        np.testing.assert_array_equal(field, other)


def _noise_maps(env, bandwidth):
    """Seeded untuned maps: gamma noise around 2 Hz, smoothed when bandwidth > 0."""
    maps = []
    for seed in range(4):
        noise = np.random.default_rng(seed).gamma(2.0, 1.0, env.n_bins)
        if bandwidth > 0:
            noise = np.asarray(env.diffuse(noise, bandwidth=bandwidth, mode="density"))
            noise *= 2.0 / noise.mean()
        maps.append(noise)
    return maps


def _assert_watershed_fields_are_valid(env, rate, fields, core, threshold=0.2):
    """Disjoint, connected, decreasing peaks, each within its own threshold,
    and each the whole part of its peak's basin within that threshold."""
    assert len(fields) <= len(core)
    used = np.concatenate([np.asarray(f) for f in fields] + [np.array([], int)])
    assert len(used) == len(set(used.tolist()))
    peaks = [float(np.max(rate[f])) for f in fields]
    assert peaks == sorted(peaks, reverse=True)
    for field, peak in zip(fields, peaks, strict=True):
        assert np.min(rate[field]) >= threshold * peak
        assert nx.is_connected(env.connectivity.subgraph(field.tolist()))
    # No bin is missing: flooding the bin graph from the same peaks gives
    # the same fields.
    peak_bins = [int(field[np.argmax(rate[field])]) for field in fields]
    for field, flooded in zip(
        fields, _flooded_fields(env, rate, peak_bins, threshold), strict=True
    ):
        np.testing.assert_array_equal(field, flooded)


@pytest.mark.parametrize("bandwidth", [0.0, 5.0])
@pytest.mark.parametrize("layout", ["box", "track", "hexagons"])
def test_watershed_on_noise_reports_no_more_fields_than_core(
    request, layout, bandwidth
):
    """Noise maxima do not fragment fields: never more fields than "core".

    Watershed fields come from the "core" peaks, so the bound is 0. In a
    separate run of 16 seeded noise maps on each of five layouts, it reported
    0-3 fewer fields per white-noise map (a basin too small to be a field)
    and the same count on maps smoothed with a 5 cm bandwidth.
    """
    env = request.getfixturevalue(layout)
    for rate in _noise_maps(env, bandwidth):
        core = detect_place_fields(env, rate, bandwidth=bandwidth, subfields="core")
        result = detect_place_fields(
            env, rate, bandwidth=bandwidth, subfields="watershed"
        )
        _assert_watershed_fields_are_valid(env, rate, result.fields, core.fields)


def test_watershed_on_untuned_spikes(ou_2min, ou_2min_env, noise_trains):
    """Untuned 0.5 Hz Poisson units, 2 min: no unit gains a field."""
    times, positions, _ = ou_2min
    rates = compute_spatial_rates(ou_2min_env, noise_trains(120.0), times, positions)
    for unit in rates:
        core = detect_place_fields(ou_2min_env, unit, subfields="core")
        result = detect_place_fields(ou_2min_env, unit, subfields="watershed")
        _assert_watershed_fields_are_valid(
            ou_2min_env, unit.firing_rate, result.fields, core.fields
        )


def test_population_coverage_passes_the_watershed_rule(box):
    rate = _gaussian(box, (35.0, 50.0), 10.0) + _gaussian(box, (59.0, 50.0), 3.3)
    maps = np.stack([rate, rate[::-1]])
    coverage = population_coverage(box, maps, bandwidth=0.0, subfields="watershed")
    for fields, unit_map in zip(coverage.place_fields, maps, strict=True):
        single = detect_place_fields(
            box, unit_map, bandwidth=0.0, subfields="watershed"
        )
        assert len(fields) == len(single) == 2
        for field, other in zip(fields, single, strict=True):
            np.testing.assert_array_equal(field, other)


def test_watershed_drops_a_peak_whose_basin_is_too_small(track):
    """A 1.2 Hz bin on the edge of a 10 Hz field: a field with "core" only.

    "core" gives the bin a two-bin field through the large field's edge. Its
    basin holds only the bin itself, which is no larger than the smallest
    field (one bin at bandwidth 0), so the watershed drops its peak and
    floods again.
    """
    x = track.bin_centers[:, 0]
    rate = 10.0 * np.exp(-((x - 61.0) ** 2) / 128)
    spike = int(np.argmin(np.abs(x - 81.0)))
    rate[spike] = 1.2
    core = detect_place_fields(track, rate, bandwidth=0.0, subfields="core")
    result = detect_place_fields(track, rate, bandwidth=0.0, subfields="watershed")
    assert len(core) == 2
    assert spike in core.fields[1]
    assert len(result) == 1
    np.testing.assert_array_equal(result.fields[0], core.fields[0])
