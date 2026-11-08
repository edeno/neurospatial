"""False-positive rates, p-value calibration and power of the cell-type verdicts.

Untuned cells are homogeneous Poisson processes at 0.5-3 Hz (rates drawn per
cell) recorded for 10 minutes while a simulated animal explores a 1 m box
(``simulate_trajectory_ou``: 10 cm/s mean speed, sampled at 30 Hz, 5 cm
bins); headings are the direction of movement. A circular-shift p-value
``(k + 1) / (n + 1)`` of an exchangeable null rejects at the exact rate
``monte_carlo_rejection_rate``, and the rejection count is checked from both
ends. Each trajectory and each set of spike trains has its own seed.
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


@pytest.mark.parametrize("family", FAMILIES)
def test_untuned_verdicts_are_at_most_the_shuffle_rate(family):
    """The public predicate on untuned cells.

    The head direction verdict is the shuffle test alone, so it rejects at
    the Monte Carlo rate of its default 1000 shuffles: 600 cells (three
    trajectories), checked from both ends. The other verdicts also need the
    bias-corrected information to reach ``min_info``, which none of 200
    untuned cells (two trajectories) does: they are conservative, held to
    the high end only.
    """
    shuffle_only = family == "head_direction"
    rejected = []
    for seed in (1, 2, 3) if shuffle_only else (1, 2):
        session = _box_session(seed)
        n_units = 200 if shuffle_only else 100
        trains = _untuned_trains(session[1], n_units, seed=2000 + seed)
        for k, train in enumerate(trains):
            kwargs = {"rng": 1000 * seed + k}
            if not shuffle_only:
                kwargs["n_shuffles"] = N_SHUFFLES
            rejected.append(_verdict(family, session, train, **kwargs))
    if shuffle_only:
        assert_rejections_at_rate(rejected, monte_carlo_rejection_rate(1000, ALPHA))
    else:
        assert_rejections_at_rate(
            rejected,
            monte_carlo_rejection_rate(N_SHUFFLES, ALPHA),
            conservative=True,
        )
        assert sum(rejected) == 0


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
@pytest.mark.filterwarnings("ignore:.*information bias is unknown:UserWarning")
def test_detection_rises_with_tuning_strength(family, strengths):
    """20 tuned cells at each of three peak rates, 10 minutes.

    Detection does not fall as the peak rate rises, rises from the weakest
    to the strongest level, and the strongest cells are detected. A weak
    cell with fewer spikes than its map has independent bins is not
    detected, with a warning.
    """
    from neurospatial.simulation import generate_poisson_spikes

    session = _box_session(1)
    env, times, positions, headings = session
    detected = []
    for strength in strengths:
        rng = np.random.default_rng(3)
        hits = 0
        for k in range(20):
            model = _tuned_model(family, env, strength, rng)
            rate = (
                model.firing_rate(times, positions)
                if family == "place"
                else model.firing_rate(times, positions, headings=headings)
            )
            train = generate_poisson_spikes(rate, times, rng=int(rng.integers(2**32)))
            hits += _verdict(family, session, train, n_shuffles=N_SHUFFLES, rng=k)
        detected.append(hits)
    assert detected == sorted(detected), detected
    assert detected[0] < detected[-1], detected
    assert detected[-1] >= 18, detected


@pytest.mark.parametrize(
    ("gaze", "expected_rate"),
    [
        ({"gaze_model": "ray_cast"}, None),
        ({"gaze_model": "fixed_distance", "view_distance": 10.0}, 1.0),
    ],
)
def test_place_cells_as_spatial_view_cells(gaze, expected_rate):
    """A place cell (8 Hz peak, 10 cm field) tested as a spatial view cell.

    With a ray-cast gaze the viewed wall point moves with heading, so a
    place cell carries little view information and passes no more often
    than an untuned cell. With a fixed 10 cm gaze distance the viewed point
    stays within 10 cm of the animal, so every place cell passes: the
    verdict alone cannot tell place from view tuning (documented in
    ``is_spatial_view_cell``).
    """
    from neurospatial.simulation import PlaceCellModel, generate_poisson_spikes

    passed = []
    for seed in (1, 2):
        session = _box_session(seed)
        env, times, positions, headings = session
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
        p_values = np.array(
            [
                test.p_value
                for test in encoding.spatial_view_cell_significance(
                    env,
                    trains,
                    times,
                    positions,
                    headings,
                    n_shuffles=N_SHUFFLES,
                    rng=seed,
                    boundary_source="analysis",
                    **gaze,
                ).values()
            ]
        )
        rates = encoding.compute_view_rates(
            env, trains, times, positions, headings, boundary_source="analysis", **gaze
        )
        information = np.asarray(rates.spatial_information()) - np.asarray(
            rates.spatial_information_bias()
        )
        # The verdict of is_spatial_view_cell: significant and min_info=0.5.
        passed.append((p_values < ALPHA) & (information >= 0.5))
    passed = np.concatenate(passed)
    if expected_rate is None:
        assert_rejections_at_rate(
            passed, monte_carlo_rejection_rate(N_SHUFFLES, ALPHA), conservative=True
        )
    else:
        assert passed.mean() >= 0.9


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


@pytest.mark.parametrize(
    ("arena", "border_rate", "grid_rate"),
    [("square", 0.20, 0.0165), ("circle", 0.12, 0.018)],
)
def test_untuned_grid_and_border_label_rates(arena, border_rate, grid_rate):
    """The label rates ``label_cell_types`` documents for untuned cells.

    2000 untuned cells (20 trajectories of 100) at the default cutoffs. The
    labels are threshold screens with no nominal rate; the rates are the
    ones in the Notes of ``SpatialRatesResult.label_cell_types``. Grid
    labels fall independently from cell to cell, so their count is checked
    against the 99% binomial interval. Border labels do not: an untuned
    cell's border score sits at the session's ``untuned_border_score``, so
    whole sessions pass or fail together and only the pooled rate is
    pinned (to within 0.03). No untuned cell is labeled place.
    """
    from neurospatial.encoding.border import untuned_border_score

    labels, offsets = [], []
    for seed in range(100, 120):
        env, times, positions = _arena_session(arena, seed)
        rates = encoding.compute_spatial_rates(
            env, _untuned_trains(times, 100, seed=5000 + seed), times, positions
        )
        scores = np.asarray(rates.border_scores())
        baseline = untuned_border_score(env, support=np.asarray(rates.spatial_support))
        offsets.append(np.nanmedian(scores) - baseline)
        labels.append(rates.label_cell_types(border_scores=scores))
    labels = np.concatenate(labels)
    assert not np.isin(labels, ["place", "undetermined"]).any()
    assert np.max(np.abs(offsets)) < 0.015
    assert_rejections_at_rate(np.char.find(labels, "grid") >= 0, grid_rate)
    border = np.mean(np.char.find(labels, "border") >= 0)
    assert border == pytest.approx(border_rate, abs=0.03)
