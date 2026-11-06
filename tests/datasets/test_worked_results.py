"""Each dataset's worked result: a finding of its paper, reproduced on the loaded data.

Every test computes its result with public neurospatial functions on the
recording's arrays, as a reader would, and checks it against a criterion
fixed before the data were analyzed: a tolerance around the paper's number
(level A), or a property the paper states, tested against a null computed
on the same pipeline (level B). The quote, the parameters and our number
are each dataset's ``WORKED_RESULT``; each test also checks that the
recorded number is the one it computed, so the record cannot drift from the
code.

Sargolini et al. (2006), Valero et al. (2022) and Forli et al. (2025) have
no test: their ``WORKED_RESULT`` says why a result is not possible yet.
Bhattarai et al. (2020), Carey et al. (2019), Igata et al. (2021) and Huszar et
al. (2022) are tested in their own files (``test_bhattarai_2020.py`` and so on),
which pin
their recorded numbers; this file checks only that their records are complete.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import i0, i1

from neurospatial.datasets import (
    bhattarai_2020,
    carey_2019,
    comrie_2026,
    forli_2025,
    grosmark_2016,
    huszar_2022,
    igata_2021,
    load,
    peyrache_2015,
    sargolini_2006,
    valero_2022,
)

RECORDED_DIGITS = 3
#: Draws of each level B null, and the seed they are drawn with.
N_NULL = 20
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


def _tracked_runs(times, max_gap):
    """[start, stop] of each run of samples with no gap longer than ``max_gap``.

    Parameters
    ----------
    times : ndarray, shape (n_time,)
    max_gap : float
        Seconds.

    Returns
    -------
    ndarray, shape (n_runs, 2)
    """
    breaks = np.flatnonzero(np.diff(times) > max_gap)
    starts = np.concatenate([[0], breaks + 1])
    stops = np.concatenate([breaks, [times.size - 1]])
    keep = stops > starts
    return np.column_stack([times[starts[keep]], times[stops[keep]]])


def _recorded(record, value):
    """The recorded number equals the computed one at the recorded precision."""
    assert record["ours"] == round(float(value), RECORDED_DIGITS), (
        f"the record is stale: update WORKED_RESULT['ours'] from {record['ours']} "
        f"to the computed {value:.{RECORDED_DIGITS}f} (the criterion itself passed)"
    )


@pytest.mark.realdata
@NWB_DEVICE_DEPRECATION
def test_peyrache_adn_head_direction_share():
    """Level A: the share of ADn units classified as HD cells, against 71%.

    The paper's classification (Methods, "Head direction cells"): tuning
    curves in 1° bins smoothed with a 6° s.d. Gaussian; an HD cell has a
    von Mises concentration above 1, a peak rate above 1 Hz and a Rayleigh
    p below 0.001; "Times when one of the LEDs was not detected were
    excluded from the analysis". The maximum-likelihood von Mises fitted to
    a tuning curve, read as a distribution of angles, has A1(kappa) equal
    to the curve's mean vector length, so kappa > 1 is mean vector length >
    A1(1) = I1(1) / I0(1) = 0.446. The Rayleigh test is on the headings at
    the unit's spikes, each the nearest tracked sample's.

    An LED dropout is a missing stretch of samples: every sample interval
    of the file is 0.025 to 0.026 s except 64 of 0.1 to 8.6 s. A sample
    interval longer than 1.5 sampling periods (0.0384 s) is a dropout: it
    is left out of the tuning curves (``max_gap``), and spikes inside it are
    left out of the Rayleigh test.

    Tolerance, fixed before the first run: 71% is an average over sessions
    with no stated spread, so we allow the binomial sampling spread of one
    session of 31 units, two standard deviations,
    2 * sqrt(0.71 * 0.29 / 31) = 0.163.
    """
    from neurospatial.encoding import compute_directional_rates
    from neurospatial.stats import rayleigh_test

    rec = load("peyrache_2015")
    wake = rec.params["analysis_epochs"]
    adn = np.flatnonzero(rec.units["brain_region"].to_numpy() == "adn")
    spikes = [rec.spike_times[i] for i in adn]
    max_gap = 1.5 * np.median(np.diff(rec.times))

    tuning = compute_directional_rates(
        spikes,
        rec.times,
        rec.headings,
        bin_size=np.deg2rad(1.0),
        bandwidth=np.deg2rad(6.0),
        max_gap=max_gap,
        epochs=wake,
    )
    kappa_one = i1(1.0) / i0(1.0)
    concentrated = tuning.mean_vector_lengths() > kappa_one
    above_1hz = tuning.peak_firing_rate() > 1.0

    times = rec.times
    rayleigh_p = np.empty(len(spikes))
    for k, train in enumerate(spikes):
        train = train[(train >= wake[0, 0]) & (train < wake[0, 1])]
        after = np.searchsorted(times, train)
        inside = (after > 0) & (after < times.size)
        train, after = train[inside], after[inside]
        before = after - 1
        tracked = times[after] - times[before] <= max_gap
        train, after, before = train[tracked], after[tracked], before[tracked]
        nearest = np.where(times[after] - train < train - times[before], after, before)
        rayleigh_p[k] = rayleigh_test(rec.headings[nearest])[1]
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
# max_gap has none to shift; the shuffle warns about it, and it has no map.
@pytest.mark.filterwarnings(
    "ignore:shuffle_spike_times_circular. none of the:UserWarning"
)
def test_grosmark_place_fields_enriched_near_the_ends():
    """Level B: directional rate-map peaks crowd the stem's ends beyond chance.

    "the place fields ... are similarly enriched near the start and goal
    ends of the track" (Fig. S2 legend). The paper's place-field method
    (Supplementary Materials, "Place Field Analysis"): 2 cm bins, spike
    counts and occupancy each smoothed with a 5 cm Gaussian, running at
    5 cm/s or more, each running direction separately.

    The statistic: for each direction, the rate maps of the pyramidal units
    that peak at 1 Hz or more; the share of their peaks, both directions
    pooled, within 20 cm of an end of the 160 cm stem. Occupancy, speed and
    smoothing at the ends could crowd peaks there without any place tuning,
    so the null is the same statistic on the same pipeline with each unit's
    spikes shifted circularly on the tracked time (runs without a gap over
    1 s; ``stats.shuffle_spike_times_circular``, shifts of at least 20 s),
    ``N_NULL`` = 20 draws, seed 2016.

    Criterion, fixed before the run: the real share exceeds the null's 95th
    percentile (``numpy.quantile``, linear interpolation).
    """
    from neurospatial import compute_spatial_rates
    from neurospatial.stats import shuffle_spike_times_circular

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
    directions = [_direction_intervals(rec.times, x, sign) for sign in (1, -1)]

    def near_end_share(trains):
        peaks = []
        for epochs in directions:
            rates = compute_spatial_rates(
                rec.env, trains, rec.times, rec.positions, epochs=epochs, **settings
            )
            peaks.append(rates.peak_locations()[rates.peak_firing_rate() >= 1.0, 0])
        pooled = np.concatenate(peaks)
        return np.mean((pooled < start + 20.0) | (pooled > stop - 20.0))

    share = near_end_share(spikes)

    windows = _tracked_runs(rec.times, max_gap=1.0)
    rng = np.random.default_rng(2016)
    shifted = [
        list(shuffle_spike_times_circular(train, windows, n_shuffles=N_NULL, rng=rng))
        for train in spikes
    ]
    null = np.array(
        [
            near_end_share([draws[k] if draws else np.empty(0) for draws in shifted])
            for k in range(N_NULL)
        ]
    )
    bar = np.quantile(null, 0.95)
    assert share > bar, (
        f"near-end share {share:.3f} does not exceed the shifted-spike null's "
        f"95th percentile {bar:.3f} (null {np.round(null, 3)})"
    )
    _recorded(grosmark_2016.WORKED_RESULT, share)


@pytest.mark.realdata
def test_comrie_decoding_mostly_local_while_running():
    """Level B sanity property: running time decodes to the rat's own segment.

    Not a finding of the preprint: its Introduction says place cells'
    activity "typically signals the actual location of the animal" (citing
    earlier work), and its Methods define non-local times as those when
    "the most likely decoded position ... was in a track segment distinct
    to the animal's actual position" ("Non-local representation of Stay and
    Switch paths"). This checks that the data and decoder show that
    baseline. The preprint's decoder is clusterless and state-space, in
    2 ms bins, fitted and decoded on the same data; ours is a Bayesian
    decoder on the file's sorted units in 0.25 s bins, fitted on
    alternating 60 s blocks and decoded on the others.

    The statistic: among held-out decoding bins while running above
    10 cm/s, the share whose MAP bin lies on the rat's own track segment
    (graph edge). Two baselines, on the same bins: a constant decoder that
    always names the segment the rat runs on most, and a null decoding the
    same held-out bins with the same decoder after shifting every unit's
    spikes by one common circular offset of at least 20 s on the session's
    span, ``N_NULL`` = 20 draws, seed 2026.

    Criterion, fixed before the run: the real share exceeds both the
    largest-segment share and the null's 95th percentile.
    """
    from neurospatial.behavior import compute_speed
    from neurospatial.decoding import BayesianDecoder

    rec = load("comrie_2026")
    params = rec.params
    spikes = list(rec.spike_times)
    decoder = BayesianDecoder(rec.env, dt=params["dt"]).fit(
        spikes,
        rec.times,
        rec.positions,
        min_speed=params["min_speed"],
        epochs=params["train_epochs"],
    )
    segment = rec.env.get_bin_attributes()["source_edge_id"].to_numpy()
    speed = compute_speed(rec.times, rec.positions)

    def decoded_and_actual(trains):
        result = decoder.predict(trains, rec.times, epochs=params["test_epochs"])
        nearest = np.clip(
            np.searchsorted(rec.times, result.times), 0, rec.times.size - 1
        )
        actual_bin = rec.env.bin_at(rec.positions[nearest])
        usable = (
            (speed[nearest] > params["min_speed"])
            & (actual_bin >= 0)
            & (result.map_bin >= 0)
        )
        return segment[result.map_bin[usable]], segment[actual_bin[usable]]

    decoded, actual = decoded_and_actual(spikes)
    share = np.mean(decoded == actual)
    _, counts = np.unique(actual, return_counts=True)
    largest = counts.max() / counts.sum()

    t0, span = rec.times[0], rec.times[-1] - rec.times[0]
    rng = np.random.default_rng(2026)
    null = np.empty(N_NULL)
    for k in range(N_NULL):
        offset = rng.uniform(20.0, span - 20.0)
        trains = [np.sort(t0 + (train - t0 + offset) % span) for train in spikes]
        null_decoded, null_actual = decoded_and_actual(trains)
        null[k] = np.mean(null_decoded == null_actual)
    bar = np.quantile(null, 0.95)
    assert share > largest, (
        f"local share {share:.3f} does not exceed the largest-segment share "
        f"{largest:.3f}"
    )
    assert share > bar, (
        f"local share {share:.3f} does not exceed the shifted-spike null's 95th "
        f"percentile {bar:.3f} (null {np.round(null, 3)})"
    )
    _recorded(comrie_2026.WORKED_RESULT, share)


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
    "attempted",
    "why",
}
ATTEMPT_KEYS = {"value", "criterion", "parameters", "differs"}
ALL = [
    bhattarai_2020,
    carey_2019,
    comrie_2026,
    forli_2025,
    grosmark_2016,
    huszar_2022,
    igata_2021,
    peyrache_2015,
    sargolini_2006,
    valero_2022,
]


@pytest.mark.parametrize("module", ALL, ids=lambda module: module.__name__)
def test_record_is_complete(module):
    """Each record has the same fields; a result has its quote, a non-result says why."""
    record = module.WORKED_RESULT
    assert set(record) - {"planted_recall"} == RECORD_KEYS
    if record["status"] == "not possible yet":
        assert record["level"] is None
        assert record["ours"] is None
        assert record["why"]
        attempted = record["attempted"]
        if attempted is not None:
            assert set(attempted) == ATTEMPT_KEYS
            assert isinstance(attempted["value"], float)
            assert attempted["differs"]
    else:
        assert record["status"] in {"reproduced", "sanity property"}
        assert record["level"] in {"A", "B"}
        assert isinstance(record["ours"], float)
        assert record["attempted"] is None
        assert record["why"] is None
        for field in ("finding", "quote", "source", "criterion", "parameters"):
            assert record[field], field


@pytest.mark.parametrize(
    "record",
    [
        bhattarai_2020.WORKED_RESULT,
        forli_2025.OWN_3D_ANALYSIS,
        huszar_2022.WORKED_RESULT,
        igata_2021.WORKED_RESULT,
    ],
    ids=["bhattarai_2020", "forli_2025", "huszar_2022", "igata_2021"],
)
def test_replay_records_state_what_planted_replay_limits(record):
    """A replay count is read beside its recall, with the control's own limits."""
    planted = record["planted_recall"]
    reading = planted["reading"]
    assert f"{100 * planted['judged'] / planted['planted']:.1f}%" in reading
    assert "limit of what the number can say" in reading
    assert "the model the decoder assumes" in reading
    assert "compression 15" in reading
    assert "evidence" in reading


def test_huszar_record_says_its_criterion_is_easy_to_pass():
    """Every null draw is 0, so one judged event would meet the criterion."""
    assert (
        "would pass on 1 of 20"
        in huszar_2022.WORKED_RESULT["planted_recall"]["reading"]
    )
