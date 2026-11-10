"""False-positive rates, p-value calibration and power of the cell-type verdicts.

Untuned cells are homogeneous Poisson processes at 0.5-3 Hz (rates drawn per
cell) recorded for 10 minutes while a simulated animal explores a 1 m box
(``simulate_trajectory_ou``: 10 cm/s mean speed, sampled at 30 Hz, 5 cm
bins); headings are the direction of movement. A circular-shift p-value
``(k + 1) / (n + 1)`` of an exchangeable null rejects at the exact rate
``monte_carlo_rejection_rate``, and the rejection count is checked from both
ends. Each trajectory and each set of spike trains has its own seed.

A rejection rate measured on a few hundred cells is uncertain by about a
percentage point (one standard error of 600 cells at 4.5% is 0.85 points),
so each test states how many cells it uses and what that number can tell
apart.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from calibration_helpers import (
    assert_pvalues_uniform,
    assert_rejections_at_rate,
    monte_carlo_rejection_rate,
)

from neurospatial import encoding

pytestmark = pytest.mark.slow

ALPHA = 0.05
N_SHUFFLES = 199
DURATION = 600.0
OBJECT = np.array([[50.0, 50.0]])


@cache
def _box_session(seed: int):
    """Ten minutes of OU exploration in a 1 m box: env, times, positions, headings."""
    from shapely.geometry import box

    from neurospatial import Environment
    from neurospatial.behavior import heading_from_velocity
    from neurospatial.simulation import simulate_trajectory_ou

    env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=5.0, units="cm")
    positions, times = simulate_trajectory_ou(
        env,
        duration=DURATION,
        speed_units="cm",
        dt=1 / 30,
        speed_mean=10.0,
        speed_std=5.2,
        rng=seed,
    )
    return env, times, positions, heading_from_velocity(times, positions)


def _untuned_trains(times, n_units: int, seed: int) -> list[np.ndarray]:
    """Constant-rate Poisson trains, one rate per unit drawn from 0.5-3 Hz."""
    from neurospatial.simulation import generate_poisson_spikes

    rng = np.random.default_rng(seed)
    return [
        generate_poisson_spikes(
            np.full(times.size, rate), times, rng=int(rng.integers(2**32))
        )
        for rate in rng.uniform(0.5, 3.0, n_units)
    ]


def _significance(family: str, session, trains, **kwargs):
    """Per-unit p-values of one family's population shuffle test."""
    env, times, positions, headings = session
    calls = {
        "place": lambda: encoding.place_cell_significance(
            env, trains, times, positions, **kwargs
        ),
        "head_direction": lambda: encoding.head_direction_cell_significance(
            trains, times, headings, **kwargs
        ),
        "spatial_view": lambda: encoding.spatial_view_cell_significance(
            env,
            trains,
            times,
            positions,
            headings,
            boundary_source="analysis",
            **kwargs,
        ),
        "object_vector": lambda: encoding.object_vector_cell_significance(
            env, trains, times, positions, OBJECT, **kwargs
        ),
        "egocentric_object_vector": (
            lambda: encoding.egocentric_object_vector_cell_significance(
                env, trains, times, positions, headings, OBJECT, **kwargs
            )
        ),
    }
    result = calls[family]()
    tests = result.values() if isinstance(result, dict) else result
    return np.array([test.p_value for test in tests])


def _verdict(family: str, session, train, **kwargs) -> bool:
    """One unit's verdict from the family's public ``is_*`` predicate."""
    env, times, positions, headings = session
    calls = {
        "place": lambda: encoding.is_place_cell(env, train, times, positions, **kwargs),
        "head_direction": lambda: encoding.is_head_direction_cell(
            train, times, headings, **kwargs
        ),
        "spatial_view": lambda: encoding.is_spatial_view_cell(
            env,
            train,
            times,
            positions,
            headings,
            boundary_source="analysis",
            **kwargs,
        ),
        "object_vector": lambda: encoding.is_object_vector_cell(
            env, train, times, positions, OBJECT, **kwargs
        ),
        "egocentric_object_vector": lambda: encoding.is_egocentric_object_vector_cell(
            env, train, times, positions, headings, OBJECT, **kwargs
        ),
    }
    return calls[family]()


FAMILIES = [
    "place",
    "head_direction",
    "spatial_view",
    "object_vector",
    "egocentric_object_vector",
]


@pytest.mark.parametrize(
    ("family", "n_sessions"),
    [
        # 1400 place cells tell a 7.25% rate (seen once on 400 cells sharing
        # one trajectory) from 4.5% with about 90% power at 99% coverage.
        ("place", 7),
        ("head_direction", 3),
        ("spatial_view", 3),
        ("object_vector", 3),
        ("egocentric_object_vector", 3),
    ],
)
def test_untuned_p_values_reject_at_the_monte_carlo_rate(family, n_sessions):
    """200 untuned cells per trajectory, a new trajectory for each 200.

    With 199 shuffles ``p < 0.05`` rejects 9 of 200 equally likely ranks
    (4.5%); the count must lie in the 99% binomial interval from both ends
    and the p-values must be uniform.

    The head direction shuffle leaves out shifts shorter than ``min_shift``
    (20 s by default), so its null is not exactly exchangeable. Its rate on
    90000 cells was 4.66% (Notes of ``is_head_direction_cell``), which 600
    cells cannot tell from 4.5%: their interval, 15 to 41, would miss a rate
    of 5.2% with probability 0.97. The exact case, ``min_shift=0``, is
    tested in ``test_head_direction_p_values_are_exact_without_min_shift``.
    """
    p_values = np.concatenate(
        [
            _significance(
                family,
                _box_session(seed),
                _untuned_trains(_box_session(seed)[1], 200, seed=1000 + seed),
                n_shuffles=N_SHUFFLES,
                rng=seed,
            )
            for seed in range(1, n_sessions + 1)
        ]
    )
    assert p_values.size == 200 * n_sessions
    assert_pvalues_uniform(p_values)
    assert_rejections_at_rate(
        p_values < ALPHA, monte_carlo_rejection_rate(N_SHUFFLES, ALPHA)
    )


def test_head_direction_p_values_are_exact_without_min_shift():
    """4000 untuned cells (20 trajectories of 200) with ``min_shift=0``.

    With no minimum shift, every circular shift of a constant-rate train is
    as likely as the train itself, so the shuffle test is exact: 4.5% of
    cells have ``p < 0.05`` with 199 shifts. The count is checked from both
    ends of the 99% binomial interval, 147 to 215, and the p-values must be
    uniform over their 200 possible values.

    4000 cells resolve a rate of 5.7% or more, or 3.4% or less, with at
    least 80% power (6% with 0.95). They do not resolve a small excess: a
    true rate of 5.2% falls outside the interval with probability 0.29.
    That takes tens of thousands of cells; the rates of 90000 cells are in
    the Notes of ``is_head_direction_cell``.
    """
    from scipy import stats

    p_values = []
    for seed in range(1, 21):
        session = _box_session(seed)
        p_values.append(
            _significance(
                "head_direction",
                session,
                _untuned_trains(session[1], 200, seed=1000 + seed),
                n_shuffles=N_SHUFFLES,
                min_shift=0.0,
                rng=seed,
            )
        )
    p_values = np.concatenate(p_values)
    assert p_values.size == 4000
    assert_rejections_at_rate(
        p_values < ALPHA, monte_carlo_rejection_rate(N_SHUFFLES, ALPHA)
    )
    # The p-values take 200 values, (k + 1) / 200, so uniformity is tested
    # on the count of each value, not against a continuous uniform.
    ranks = np.rint(p_values * (N_SHUFFLES + 1)).astype(int)
    np.testing.assert_allclose(ranks / (N_SHUFFLES + 1), p_values)
    per_value = np.bincount(ranks, minlength=N_SHUFFLES + 2)[1:]
    assert stats.chisquare(per_value).pvalue >= 0.001


@pytest.mark.parametrize(
    "family", [family for family in FAMILIES if family != "head_direction"]
)
def test_untuned_verdicts_are_at_most_the_shuffle_rate(family):
    """The public predicate of an information-tuned cell type on untuned cells.

    These verdicts need a significant shuffle test and bias-corrected
    information at ``min_info``. No untuned cell of 200 (two trajectories)
    reaches both, so the verdicts are conservative: they are held to the
    high end of the shuffle's binomial interval only, and to zero. The head
    direction verdict has no information cutoff and is tested separately.
    """
    rejected = []
    for seed in (1, 2):
        session = _box_session(seed)
        trains = _untuned_trains(session[1], 100, seed=2000 + seed)
        rejected.extend(
            _verdict(family, session, train, n_shuffles=N_SHUFFLES, rng=1000 * seed + k)
            for k, train in enumerate(trains)
        )
    assert_rejections_at_rate(
        rejected, monte_carlo_rejection_rate(N_SHUFFLES, ALPHA), conservative=True
    )
    assert sum(rejected) == 0


def test_untuned_head_direction_verdict_rejects_at_the_shuffle_rate():
    """``is_head_direction_cell`` on 2000 untuned cells (ten trajectories).

    The head direction verdict is the shuffle test alone, so it calls
    untuned cells head direction cells at the shuffle's rate, not below it:
    4.5% with 199 shifts (4.66% of 90000 cells at the default
    ``min_shift``, a difference 2000 cells cannot see), checked from both
    ends of the 99% binomial interval, 67 to 115 of 2000. A true rate of 7%
    falls outside it with probability 0.99, 6.5% with 0.91 and 3% with
    0.81; 400 cells, with an interval of 8 to 29, would pass at 7% six
    times in ten.

    The predicate's default is 1000 shifts, where an exact test rejects
    5.0%. That costs five times as much per cell; its measured rate, 331 of
    6600 cells (5.0%), is in the Notes of ``is_head_direction_cell``.
    """
    rejected = []
    for seed in range(1, 11):
        session = _box_session(seed)
        trains = _untuned_trains(session[1], 200, seed=2000 + seed)
        rejected.extend(
            _verdict(
                "head_direction",
                session,
                train,
                n_shuffles=N_SHUFFLES,
                rng=1000 * seed + k,
            )
            for k, train in enumerate(trains)
        )
    assert len(rejected) == 2000
    assert_rejections_at_rate(rejected, monte_carlo_rejection_rate(N_SHUFFLES, ALPHA))


def test_head_direction_threshold_screen_rejects_no_untuned_cell():
    """MVL >= 0.4 and the tuning test: 600 untuned cells, none passes."""
    flagged = []
    for seed in (1, 2, 3):
        _, times, _, headings = _box_session(seed)
        rates = encoding.compute_directional_rates(
            _untuned_trains(times, 200, seed=1000 + seed), times, headings
        )
        flagged.append(np.asarray(rates.classify()))
    assert_rejections_at_rate(np.concatenate(flagged), ALPHA, conservative=True)
    assert not np.concatenate(flagged).any()


def _wall_point(rng) -> np.ndarray:
    """A point on the box wall, away from the corners: what a ray-cast gaze sees."""
    along = rng.uniform(20.0, 80.0)
    return [
        np.array([along, 0.0]),
        np.array([along, 100.0]),
        np.array([0.0, along]),
        np.array([100.0, along]),
    ][rng.integers(4)]


def _tuned_model(family, env, strength, rng):
    """A tuned cell of ``family`` with peak rate (Hz) ``strength``."""
    from neurospatial.simulation import (
        HeadDirectionCellModel,
        ObjectVectorCellModel,
        PlaceCellModel,
        SpatialViewCellModel,
    )

    if family == "place":
        return PlaceCellModel(
            env,
            center=rng.uniform(20.0, 80.0, 2),
            width=10.0,
            max_rate=strength,
            baseline_rate=0.1,
        )
    if family == "head_direction":
        return HeadDirectionCellModel(
            preferred_direction=rng.uniform(-np.pi, np.pi),
            concentration=1.0,
            max_rate=strength,
            baseline_rate=1.0,
        )
    if family == "spatial_view":
        return SpatialViewCellModel(
            env,
            preferred_view_location=_wall_point(rng),
            max_rate=strength,
            baseline_rate=0.1,
            boundary_source="analysis",
        )
    return ObjectVectorCellModel(
        env,
        object_positions=OBJECT,
        preferred_distance=20.0,
        distance_tolerance=5.0,
        preferred_direction=rng.uniform(-np.pi, np.pi),
        direction_frame="egocentric"
        if family == "egocentric_object_vector"
        else "allocentric",
        max_rate=strength,
        baseline_rate=0.1,
    )


# First line of the warning for a significant unit whose information bias is
# unknown (``neurospatial.encoding._metrics._warn_unknown_bias``), by the cell
# type it names. In this file only weak object-vector cells are refused this
# way: their maps have more independent bins than the cells have spikes.
UNKNOWN_BIAS_WARNING = (
    "1 of 1 units have significant information but fewer spikes than their "
    "rate maps have independent bins, so their information bias is unknown "
    "and they are not called {kind} cells."
)
UNKNOWN_BIAS_KIND = {
    "object_vector": "object-vector",
    "egocentric_object_vector": "egocentric object-vector",
}


@pytest.mark.parametrize(
    ("family", "strengths"),
    [
        ("place", (1.0, 2.0, 5.0)),
        ("head_direction", (1.25, 1.6, 2.5)),
        ("spatial_view", (0.5, 1.0, 2.0)),
        ("object_vector", (2.5, 5.0, 10.0)),
        ("egocentric_object_vector", (2.5, 5.0, 10.0)),
    ],
)
def test_detection_rises_with_tuning_strength(family, strengths):
    """20 tuned cells at each of three peak rates, 10 minutes.

    Detection does not fall as the peak rate rises, rises from the weakest
    to the strongest level, and the strongest cells are detected.

    A significant cell with fewer spikes than its map has independent bins
    is not detected, and the verdict says so in a warning. That happens to
    the object-vector families only: at their weakest level (about 100
    spikes) most cells are refused this way (19 or 20 of 20 on four sets of
    cells), which is why none is detected there. No other warning is
    raised, and none at all for the other families.
    """
    import warnings

    from neurospatial.simulation import generate_poisson_spikes

    session = _box_session(1)
    env, times, positions, headings = session
    detected, refused = [], []
    for strength in strengths:
        rng = np.random.default_rng(3)
        hits, unknown_bias = 0, 0
        for k in range(20):
            model = _tuned_model(family, env, strength, rng)
            rate = (
                model.firing_rate(times, positions)
                if family == "place"
                else model.firing_rate(times, positions, headings=headings)
            )
            train = generate_poisson_spikes(rate, times, rng=int(rng.integers(2**32)))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                verdict = _verdict(family, session, train, n_shuffles=N_SHUFFLES, rng=k)
            hits += verdict
            if caught:
                (warning,) = caught
                assert issubclass(warning.category, UserWarning)
                assert str(warning.message).splitlines()[0] == (
                    UNKNOWN_BIAS_WARNING.format(kind=UNKNOWN_BIAS_KIND[family])
                )
                assert not verdict
                unknown_bias += 1
        detected.append(hits)
        refused.append(unknown_bias)
    assert detected == sorted(detected), detected
    assert detected[0] < detected[-1], detected
    assert detected[-1] >= 18, detected
    if family in UNKNOWN_BIAS_KIND:
        assert refused[0] > 10, refused
    else:
        assert refused == [0, 0, 0]


@pytest.mark.parametrize(
    ("gaze", "passes"),
    [
        ({"gaze_model": "ray_cast"}, False),
        ({"gaze_model": "fixed_distance", "view_distance": 10.0}, True),
    ],
)
def test_place_cells_as_spatial_view_cells(gaze, passes):
    """50 place cells (8 Hz peak, 10 cm field) given to ``is_spatial_view_cell``.

    With a fixed 10 cm gaze distance the viewed point stays within 10 cm of
    the animal, so every place cell passes: the verdict alone cannot tell
    place from view tuning (documented in ``is_spatial_view_cell``).

    With a ray-cast gaze the viewed wall point moves with heading, and no
    place cell passes (0 of 200 on four pairs of trajectories). The shuffle
    is not what refuses them: the view-information test alone is
    significant for far more place cells than for untuned ones (13 to 23 of
    50, against the 99% binomial interval's end of 7 at the 4.5% of an
    exact test). Their bias-corrected view information stays under the 0.5
    bits/spike cutoff (0.46 at most).
    """
    from calibration_helpers import null_rejection_interval

    from neurospatial.simulation import PlaceCellModel, generate_poisson_spikes

    verdicts, p_values = [], []
    for seed in (1, 2):
        session = _box_session(seed)
        env, times, positions, _ = session
        rng = np.random.default_rng(40 + seed)
        trains = [
            generate_poisson_spikes(
                PlaceCellModel(
                    env,
                    center=rng.uniform(15.0, 85.0, 2),
                    width=10.0,
                    max_rate=8.0,
                    baseline_rate=0.1,
                ).firing_rate(times, positions),
                times,
                rng=int(rng.integers(2**32)),
            )
            for _ in range(25)
        ]
        verdicts.extend(
            _verdict(
                "spatial_view",
                session,
                train,
                n_shuffles=N_SHUFFLES,
                rng=1000 * seed + k,
                **gaze,
            )
            for k, train in enumerate(trains)
        )
        p_values.append(
            _significance(
                "spatial_view", session, trains, n_shuffles=N_SHUFFLES, rng=seed, **gaze
            )
        )
    verdicts, p_values = np.array(verdicts), np.concatenate(p_values)
    assert verdicts.size == 50
    if passes:
        assert verdicts.all()
        return
    assert not verdicts.any()
    _, untuned_most = null_rejection_interval(
        p_values.size, monte_carlo_rejection_rate(N_SHUFFLES, ALPHA)
    )
    assert np.sum(p_values < ALPHA) > untuned_most


def _circle_session(seed: int):
    """Ten minutes of OU exploration in a 1 m diameter circular arena."""
    from shapely.geometry import Point

    from neurospatial import Environment
    from neurospatial.simulation import simulate_trajectory_ou

    env = Environment.from_polygon(
        Point(50.0, 50.0).buffer(50.0, 64), bin_size=5.0, units="cm"
    )
    positions, times = simulate_trajectory_ou(
        env,
        duration=DURATION,
        speed_units="cm",
        dt=1 / 30,
        speed_mean=10.0,
        speed_std=5.2,
        rng=seed,
    )
    return env, times, positions


def _arena_session(arena: str, seed: int):
    if arena == "square":
        return _box_session(seed)[:3]
    return _circle_session(seed)


# Labels of untuned cells in 400 sessions of 100 cells in each arena
# (trajectory seeds 100-499, spike seeds 5000 + trajectory seed), the
# measurement behind the Notes of ``SpatialRatesResult.label_cell_types``.
# Grid labels: 729 (square) and 716 (circle) of 40000, binomial from cell to
# cell (variance 1.06 and 0.99 of binomial between sessions).
UNTUNED_GRID_RATE = 0.018
# Border labels: 14.8% (square) and 11.5% (circle), but a session's rate runs
# from 0 to 63% (standard deviation 13.8 points) in the square and from 0 to
# 46% (8.0 points) in the circle, 15 and 6 times the binomial variance of 100
# cells. The rate of 20 sessions is therefore held to the central 99% of the
# mean of 20 sessions drawn with replacement from the 400 measured ones (a
# normal interval, rate +- 2.58 sd / sqrt(20), is 0.068-0.227 and
# 0.068-0.161, a little short on the skewed side).
UNTUNED_BORDER_RATE_OF_20_SESSIONS = {
    "square": (0.076, 0.233),
    "circle": (0.0715, 0.164),
}
# Sessions whose median untuned border score lay within 0.015 of the
# session's ``untuned_border_score``: 390 (square) and 395 (circle) of 400.
MEDIAN_AT_BASELINE_RATE = 785 / 800


@pytest.mark.parametrize("arena", ["square", "circle"])
def test_untuned_grid_and_border_label_rates(arena):
    """What ``label_cell_types`` documents for untuned cells, on 20 sessions.

    2000 untuned cells (20 trajectories of 100) at the default cutoffs. The
    labels are threshold screens with no nominal rate; the rates are the
    measured ones in the Notes of ``SpatialRatesResult.label_cell_types``.

    No untuned cell is labeled place. Grid labels fall independently from
    cell to cell, so their count is checked against the 99% binomial
    interval at the measured 1.8%.

    Border labels do not. A session's untuned cells score near the
    session's ``untuned_border_score``, which changes with the bins the
    animal visited along the walls, so:

    - the sessions do not share one per-cell rate (a chi-square test of
      equal rates across the 20 sessions rejects at 0.001);
    - the session's rate rises with its baseline (Spearman, one-sided
      p < 0.01);
    - the median untuned score of a session is within 0.015 of the
      baseline in 98% of sessions: the count of such sessions is at or
      above the low end of its 99% binomial interval, 17 of 20;
    - the pooled rate of 20 sessions is only loosely determined. It must
      lie in the central 99% of that rate over the measured sessions,
      7.6-23.3% in the square and 7.2-16.4% in the circle. The two ranges
      overlap: 20 sessions do not tell the arenas apart.
    """
    from calibration_helpers import null_rejection_interval
    from scipy import stats

    from neurospatial.encoding.border import untuned_border_score

    labels, baselines, medians = [], [], []
    for seed in range(100, 120):
        env, times, positions = _arena_session(arena, seed)
        rates = encoding.compute_spatial_rates(
            env, _untuned_trains(times, 100, seed=5000 + seed), times, positions
        )
        scores = np.asarray(rates.border_scores())
        baselines.append(
            untuned_border_score(env, support=np.asarray(rates.spatial_support))
        )
        medians.append(np.nanmedian(scores))
        labels.append(rates.label_cell_types(border_scores=scores))
    labels = np.array(labels)  # (n_sessions, n_cells)
    baselines, medians = np.array(baselines), np.array(medians)
    n_sessions, n_cells = labels.shape
    assert not np.isin(labels, ["place", "undetermined"]).any()
    assert_rejections_at_rate(
        np.char.find(labels, "grid").ravel() >= 0, UNTUNED_GRID_RATE
    )

    border = np.char.find(labels, "border") >= 0
    session_rate = border.mean(axis=1)
    pooled = session_rate.mean()
    low, high = UNTUNED_BORDER_RATE_OF_20_SESSIONS[arena]
    assert low <= pooled <= high, pooled
    # Equal per-cell rates would make this chi-square on n_sessions - 1
    # degrees of freedom (about 19); measured, it is 63 to 434.
    equal_rates = np.sum(
        (border.sum(axis=1) - n_cells * pooled) ** 2
        / (n_cells * pooled * (1.0 - pooled))
    )
    assert equal_rates > stats.chi2.ppf(0.999, n_sessions - 1), equal_rates
    rising = stats.spearmanr(session_rate, baselines, alternative="greater")
    assert rising.pvalue < 0.01, rising
    at_baseline = int(np.sum(np.abs(medians - baselines) < 0.015))
    fewest, _ = null_rejection_interval(n_sessions, MEDIAN_AT_BASELINE_RATE)
    assert at_baseline >= fewest, (at_baseline, fewest)
