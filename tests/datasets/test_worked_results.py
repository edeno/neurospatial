"""Each dataset's worked result: a finding of its paper, reproduced on the loaded data.

Every test computes its result with public neurospatial functions on the
recording's arrays, as a reader would, and checks it against a criterion
fixed before the data were analyzed: a tolerance around the paper's number
(level A) or a property the paper states (level B). The quote, the
parameters and our number are each dataset's ``WORKED_RESULT``; each test
also checks that the recorded number is the one it computed, so the record
cannot drift from the code.

Sargolini et al. (2006) and Valero et al. (2022) have no test: their
``WORKED_RESULT`` says why a result is not possible yet.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import i0, i1

from neurospatial.datasets import (
    comrie_2026,
    forli_2025,
    grosmark_2016,
    load,
    peyrache_2015,
    sargolini_2006,
    valero_2022,
)

RECORDED_DIGITS = 3
# The files' devices carry the empty 'manufacturer' field pynwb now deprecates.
NWB_DEVICE_DEPRECATION = pytest.mark.filterwarnings(
    "ignore:The 'manufacturer' field is deprecated:DeprecationWarning"
)


def _direction_intervals(times, x, sign):
    """[start, stop) runs of sample intervals whose 1-D displacement has ``sign``.

    Parameters
    ----------
    times : ndarray, shape (n_time,)
    x : ndarray, shape (n_time,)
        Position along the track.
    sign : {1, -1}

    Returns
    -------
    ndarray, shape (n_runs, 2)
    """
    moving = np.sign(np.diff(x)) == sign
    edges = np.diff(np.concatenate([[0], moving.astype(np.int8), [0]]))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return np.column_stack([times[starts], times[stops]])


def _recorded(record, value):
    """The recorded number equals the computed one at the recorded precision."""
    assert record["ours"] == round(float(value), RECORDED_DIGITS), (
        f"WORKED_RESULT['ours'] is {record['ours']}, the test computed {value:.6f}"
    )


@pytest.mark.realdata
@NWB_DEVICE_DEPRECATION
def test_peyrache_adn_head_direction_share():
    """Level A: the share of ADn units classified as HD cells, against 71%.

    The paper's classification (Methods, "Head direction cells"): tuning
    curves in 1° bins smoothed with a 6° s.d. Gaussian; an HD cell has a
    von Mises concentration above 1, a peak rate above 1 Hz and a Rayleigh
    p below 0.001. The maximum-likelihood von Mises fitted to a tuning
    curve, read as a distribution of angles, has A1(kappa) equal to the
    curve's mean vector length, so kappa > 1 is mean vector length >
    A1(1) = I1(1) / I0(1) = 0.446. The Rayleigh test is on the headings at
    the unit's spikes.

    Tolerance, fixed before the run: 71% is an average over sessions with
    no stated spread, so we allow the binomial sampling spread of one
    session of 31 units, two standard deviations,
    2 * sqrt(0.71 * 0.29 / 31) = 0.163.
    """
    from neurospatial.encoding import compute_directional_rates
    from neurospatial.stats import rayleigh_test

    rec = load("peyrache_2015")
    wake = rec.params["analysis_epochs"]
    adn = np.flatnonzero(rec.units["brain_region"].to_numpy() == "adn")
    spikes = [rec.spike_times[i] for i in adn]

    tuning = compute_directional_rates(
        spikes,
        rec.times,
        rec.headings,
        bin_size=np.deg2rad(1.0),
        bandwidth=np.deg2rad(6.0),
        epochs=wake,
    )
    kappa_one = i1(1.0) / i0(1.0)
    concentrated = tuning.mean_vector_lengths() > kappa_one
    above_1hz = tuning.peak_firing_rate() > 1.0

    tracked = np.isfinite(rec.headings)
    t_tracked, h_tracked = rec.times[tracked], rec.headings[tracked]
    rayleigh_p = np.empty(len(spikes))
    for k, train in enumerate(spikes):
        in_wake = train[(train >= wake[0, 0]) & (train < wake[0, 1])]
        nearest = np.clip(np.searchsorted(t_tracked, in_wake), 0, t_tracked.size - 1)
        rayleigh_p[k] = rayleigh_test(h_tracked[nearest])[1]
    hd_cell = concentrated & above_1hz & (rayleigh_p < 0.001)

    share = hd_cell.mean()
    assert abs(share - 0.71) <= 0.163, (
        f"{hd_cell.sum()} of {adn.size} ADn units ({share:.0%}) are HD cells; "
        "the paper's average is 71%"
    )
    _recorded(peyrache_2015.WORKED_RESULT, share)


@pytest.mark.realdata
@NWB_DEVICE_DEPRECATION
# A pyramidal unit whose maze spikes all fall in tracking gaps longer than
# max_gap has no spike in a direction's analyzed time. The shuffle warns about
# it (as if the clocks differed), and its p-value is NaN, so it is no place
# cell.
@pytest.mark.filterwarnings(
    "ignore:shuffle_spike_times_circular. none of the:UserWarning"
)
def test_grosmark_place_fields_tile_the_track():
    """Level B: directional place fields tile the stem and crowd its ends.

    The paper's place-field method (Supplementary Materials, "Place Field
    Analysis"): 2 cm bins, spike counts and occupancy each smoothed with a
    5 cm Gaussian, running at 5 cm/s or more, each running direction
    separately. A unit is a place cell in a direction when its spatial
    information beats circular spike shifts (p < 0.01, 200 shifts, fixed
    seed) and its peak rate is at least 1 Hz; the paper's own null
    resamples spikes by occupancy instead.

    Criteria, fixed before the run:

    - tiling: in each direction, each of the stem's eight 20 cm segments
      holds the rate-map peak of at least one place cell;
    - "enriched near the start and goal ends of the track" (Fig. S2): more
      than 25% of the peaks, both directions pooled, lie within 20 cm of an
      end, the 25% of the stem that 2 x 20 cm is.
    """
    from neurospatial import compute_spatial_rates
    from neurospatial.encoding import place_cell_significance

    rec = load("grosmark_2016")
    pyramidal = np.flatnonzero(rec.units["label"].to_numpy() == "pE")
    spikes = [rec.spike_times[i] for i in pyramidal]
    x = rec.positions[:, 0]
    settings = {
        "method": "binned",
        "bandwidth": 5.0,
        "min_speed": rec.params["min_speed"],
    }
    start, stop = rec.env.dimension_ranges[0]
    segment_edges = np.linspace(start, stop, 9)

    peaks = []
    for sign in (1, -1):
        epochs = _direction_intervals(rec.times, x, sign)
        rates = compute_spatial_rates(
            rec.env, spikes, rec.times, rec.positions, epochs=epochs, **settings
        )
        shifts = place_cell_significance(
            rec.env,
            spikes,
            rec.times,
            rec.positions,
            epochs=epochs,
            n_shuffles=200,
            rng=np.random.default_rng(2016),
            **settings,
        )
        p_values = shifts.p_value
        place = (p_values < 0.01) & (rates.peak_firing_rate() >= 1.0)
        direction_peaks = rates.peak_locations()[place, 0]
        counts, _ = np.histogram(direction_peaks, bins=segment_edges)
        assert np.all(counts >= 1), (
            f"direction {sign:+d}: place-field peaks per 20 cm segment {counts}"
        )
        peaks.append(direction_peaks)

    peaks_all = np.concatenate(peaks)
    near_end = (peaks_all < start + 20.0) | (peaks_all > stop - 20.0)
    share = near_end.mean()
    assert share > 0.25, (
        f"{near_end.sum()} of {peaks_all.size} peaks ({share:.0%}) within 20 cm "
        "of an end; uniform would be 25%"
    )
    _recorded(grosmark_2016.WORKED_RESULT, share)


@pytest.mark.realdata
def test_comrie_decoding_mostly_local_while_running():
    """Level B: while running, the decoded position is usually the rat's own segment.

        The preprint decodes during running above 10 cm/s and calls a time
        non-local when "the most likely decoded position ... was in a track
        segment distinct to the animal's actual position" (Methods, "Non-local
        representation of Stay and Switch paths"); its Introduction: place
        cells' activity "typically signals the actual location of the animal".
    Its decoder is clusterless and state-space, in
        2 ms bins, fitted and decoded on the same data; ours is a Bayesian
        decoder on the file's sorted units in 0.25 s bins, fitted on alternating
        60 s blocks and decoded on the others.

        Criterion, fixed before the run: in held-out time bins while running
        above 10 cm/s, the MAP bin lies on the rat's own track segment (graph
        edge) in more than half of them; nine segments make chance about 11%.
    """
    from neurospatial.behavior import compute_speed
    from neurospatial.decoding import BayesianDecoder

    rec = load("comrie_2026")
    params = rec.params
    decoder = BayesianDecoder(rec.env, dt=params["dt"]).fit(
        list(rec.spike_times),
        rec.times,
        rec.positions,
        min_speed=params["min_speed"],
        epochs=params["train_epochs"],
    )
    result = decoder.predict(
        list(rec.spike_times), rec.times, epochs=params["test_epochs"]
    )

    nearest = np.clip(np.searchsorted(rec.times, result.times), 0, rec.times.size - 1)
    speed = compute_speed(rec.times, rec.positions)[nearest]
    actual_bin = rec.env.bin_at(rec.positions[nearest])
    decoded_bin = result.map_bin
    usable = (speed > params["min_speed"]) & (actual_bin >= 0) & (decoded_bin >= 0)

    segment = rec.env.get_bin_attributes()["source_edge_id"].to_numpy()
    local = segment[decoded_bin[usable]] == segment[actual_bin[usable]]
    share = local.mean()
    assert share > 0.5, (
        f"{local.sum()} of {usable.sum()} running bins ({share:.0%}) decode to "
        "the rat's own segment"
    )
    _recorded(comrie_2026.WORKED_RESULT, share)


@pytest.fixture(scope="module")
def forli_tuned():
    """Which flight-active units of the bat are stable and spatially tuned in flight.

    The paper (Results, Fig. 1): "a large portion of single units that were
    active during flight exhibited robust spatial selectivity (place cells:
    990 out of 1,386 flight-active neurons (71%), n = 6 bats". Its place
    cells (Methods, "Identification of spatially tuned cells") have more
    than one spike per flight, more than 3 Hz at the field peak, a
    stability (Spearman correlation of fields from the first and second
    half of the flights) of at least 0.4, and one prominent field, on 1-D
    fields of each clustered flight path in 0.15 m bins.

    Ours: 3-D rate maps over the room's 0.2 m voxels during the flights of
    the neural recording, smoothed with a 0.2 m bandwidth (our choice, one
    voxel); flight-active is more than one spike per flight on average;
    tuned is a peak above 3 Hz and a first-half against second-half
    Spearman correlation of at least 0.4 over the voxels visited in both.
    No one-field condition: it needs the 1-D path fields.

    Criterion, fixed before the run: more than half of the flight-active
    units are tuned.
    """
    from neurospatial import compute_spatial_rates
    from neurospatial.encoding import compare_spatial_rates

    rec = load("forli_2025")
    flights = np.asarray(rec.params["analysis_epochs"])
    n_flights = flights.shape[0]
    in_flight = np.array(
        [
            sum(np.count_nonzero((t >= a) & (t < b)) for a, b in flights)
            for t in rec.spike_times
        ]
    )
    active = np.flatnonzero(in_flight > n_flights)
    spikes = [rec.spike_times[i] for i in active]
    ids = rec.units.index.to_numpy()[active]

    def rates(epochs):
        return compute_spatial_rates(
            rec.env,
            spikes,
            rec.times,
            rec.positions,
            bandwidth=0.2,
            epochs=epochs,
            unit_ids=ids,
        )

    half = n_flights // 2
    whole = rates(flights)
    stability = compare_spatial_rates(
        rates(flights[:half]), rates(flights[half:]), method="spearman"
    )
    correlation = stability.summary_table()["correlation"].loc[ids].to_numpy()
    return (whole.peak_firing_rate() > 3.0) & (correlation >= 0.4)


@pytest.mark.realdata
def test_forli_record_matches(forli_tuned):
    """The recorded share is the computed one."""
    _recorded(forli_2025.WORKED_RESULT, forli_tuned.mean())


@pytest.mark.realdata
@pytest.mark.xfail(
    strict=True,
    reason=(
        "does not reproduce: 31 of 204 flight-active units (15%) are tuned in "
        "3-D, the paper reports 71% of 1-D path fields; see WORKED_RESULT"
    ),
)
def test_forli_flight_place_cells_are_most_flight_active_units(forli_tuned):
    """Level B: most flight-active units are stable, spatially tuned in flight.

    Criterion, fixed before the run: more than half of the flight-active
    units are tuned (``forli_tuned`` has the paper's definition and ours).
    """
    share = forli_tuned.mean()
    assert share > 0.5, (
        f"{forli_tuned.sum()} of {forli_tuned.size} flight-active units "
        f"({share:.0%}) are tuned; the paper's share is 71%"
    )


RECORD_KEYS = {
    "status",
    "level",
    "finding",
    "quote",
    "source",
    "paper",
    "ours",
    "criterion",
    "parameters",
    "unmatched",
    "why",
}
ALL = [
    comrie_2026,
    forli_2025,
    grosmark_2016,
    peyrache_2015,
    sargolini_2006,
    valero_2022,
]


@pytest.mark.parametrize("module", ALL, ids=lambda module: module.__name__)
def test_record_is_complete(module):
    """Each record has the same fields; a result has its quote, a non-result says why."""
    record = module.WORKED_RESULT
    assert set(record) == RECORD_KEYS
    if record["status"] == "not possible yet":
        assert record["level"] is None
        assert record["ours"] is None
        assert record["why"]
    else:
        assert record["status"] in {"reproduced", "not reproduced"}
        assert record["level"] in {"A", "B"}
        assert isinstance(record["ours"], float)
        assert record["why"] is None
        for field in ("finding", "quote", "source", "criterion", "parameters"):
            assert record[field], field
