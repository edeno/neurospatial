"""Rank-order verdicts: the t-distribution, alpha, templates and pooled SSIs."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy import stats

from neurospatial import Environment
from neurospatial.decoding import DecodingResult
from neurospatial.replay import (
    RankOrderMethod,
    compare_replay_methods,
    detect_rank_order_replay,
)
from neurospatial.replay.methods import (
    CHENANI_2019,
    DIBA_BUZSAKI_2007,
    FOSTER_WILSON_2006,
)

DT = 0.02


@pytest.fixture(scope="module")
def place_code():
    """40 cells with fields in order along a 200 cm track."""
    env = Environment.linear_track(
        endpoints=[[0.0, 0.0], [200.0, 0.0]], bin_size=4.0, units="cm"
    )
    x = env.bin_centers[:, 0]
    rates = 0.5 + 25.0 * np.exp(
        -0.5 * ((x - np.linspace(0.0, 200.0, 40)[:, None]) / 10.0) ** 2
    )
    return env, rates


def _firing(order, n_cells=40, start=1.0, step=0.005):
    """One spike per cell, the k-th cell of ``order`` at start + k * step."""
    trains = [np.array([]) for _ in range(n_cells)]
    for k, cell in enumerate(order):
        trains[cell] = np.array([start + step * k])
    return trains


class TestFosterWilson:
    def test_p_value_is_the_t_distribution_of_rho_with_n_minus_2_df(self, place_code):
        """14 cells fire in a partly shuffled field order: rho = 0.69."""
        env, rates = place_code
        order = [11, 10, 14, 12, 22, 13, 17, 15, 23, 16, 19, 21, 18, 20]
        result = detect_rank_order_replay(
            env, _firing(order), (1.0, 1.2), rates, rank_order_method=FOSTER_WILSON_2006
        )
        reference = stats.spearmanr(np.arange(14), order)
        assert result.score == pytest.approx(reference.statistic)
        assert 0.6 < result.score < 0.8
        t = result.score * np.sqrt(12 / (1 - result.score**2))
        assert result.p_value == pytest.approx(2 * stats.t.sf(t, 12), rel=1e-9)
        # scipy's spearmanr p-value uses the same t-distribution.
        assert result.p_value == pytest.approx(reference.pvalue, rel=1e-9)
        assert result.is_replay

    def test_a_perfect_sequence_has_p_zero(self, place_code):
        env, rates = place_code
        result = detect_rank_order_replay(
            env,
            _firing(range(10, 24)),
            (1.0, 1.2),
            rates,
            rank_order_method=FOSTER_WILSON_2006,
        )
        assert result.score == pytest.approx(1.0)
        assert result.p_value == 0.0
        assert result.is_replay


def test_a_p_value_equal_to_alpha_is_not_replay(place_code):
    """12 cells in field order against 19 permutations: p = 1 / 20 = alpha."""
    env, rates = place_code
    method = RankOrderMethod(
        name="cell_order_19",
        reference="test",
        test="cell_order",
        alternative="greater",
        n_shuffles=19,
        alpha=0.05,
    )
    spikes = _firing(range(10, 22))
    result = detect_rank_order_replay(
        env, spikes, (1.0, 1.2), rates, rank_order_method=method, rng=0
    )
    assert result.p_value == 1 / 20
    assert not result.is_replay
    above = dataclasses.replace(method, name="alpha_above", alpha=0.051)
    assert detect_rank_order_replay(
        env, spikes, (1.0, 1.2), rates, rank_order_method=above, rng=0
    ).is_replay


def test_diba_prefers_the_template_with_more_cells(place_code):
    """Diba & Buzsaki (2007): more participating cells first, then |rho|.

    20 cells fire; on template "all" (40 template cells) their order gives
    rho = 0.86. Template "few" keeps 15 of them as template cells (the
    others peak below 5 Hz), which fire in perfect order: rho = 1 on 15
    cells. Both are replay; "all" has more cells.
    """
    env, rates = place_code
    few = rates.copy()
    few[[c for c in range(40) if not 10 <= c < 25]] *= 0.1
    order = [10, 11, 12, 13, 14, 15, 16, 17, 25, 18, 19, 26, 20, 27, 21, 28, 22, 29]
    order += [23, 24]
    templates = {"all": rates, "few": few}
    spikes = _firing(order)
    each = {
        name: detect_rank_order_replay(
            env,
            spikes,
            (1.0, 1.2),
            templates,
            rank_order_method=DIBA_BUZSAKI_2007,
            template=name,
            rng=0,
        )
        for name in templates
    }
    assert each["all"].is_replay and each["few"].is_replay
    assert each["all"].score == pytest.approx(stats.spearmanr(range(20), order)[0])
    assert each["few"].score == pytest.approx(1.0)
    chosen = detect_rank_order_replay(
        env,
        spikes,
        (1.0, 1.2),
        templates,
        rank_order_method=DIBA_BUZSAKI_2007,
        rng=0,
    )
    assert chosen.template == "all"


def _epoch(env, n_cells, orders, start=10.0):
    """One event per order, 1 s apart, and decoded events holding only times."""
    trains = [[] for _ in range(n_cells)]
    events = []
    for index, order in enumerate(orders):
        onset = start + index
        for k, cell in enumerate(order):
            trains[cell].append(onset + 0.01 * k)
        times = onset + DT / 2 + DT * np.arange(5)
        events.append(
            DecodingResult(
                posterior=np.full((5, env.n_bins), 1.0 / env.n_bins),
                env=env,
                times=times,
            )
        )
    return [np.sort(np.array(train)) for train in trains], events


class TestChenaniPooledShuffles:
    def test_the_null_is_the_largest_ssi_over_templates(self, place_code):
        """With a second template in reverse field order, every shuffle's
        largest SSI is |SSI| >= 0. An event with rho = 0 then has p = 1;
        from one template alone, about half the shuffles fall below 0.
        """
        env, rates = place_code
        cells = [5, 10, 15, 20, 25, 30, 35]
        order = [cells[k] for k in (0, 3, 5, 6, 4, 2, 1)]  # rho = 0
        assert stats.spearmanr(range(7), order)[0] == pytest.approx(0.0, abs=1e-12)
        trains, events = _epoch(env, 40, [order] * 3)
        table = compare_replay_methods(
            env,
            events,
            replay_methods=[CHENANI_2019],
            spike_times=trains,
            encoding_models={"out": rates, "in": rates[::-1]},
            dt=DT,
            rng=0,
        )
        np.testing.assert_allclose(table["score"], 0.0, atol=1e-12)
        np.testing.assert_array_equal(table["p_value"], 1.0)

    def test_cells_without_a_field_are_not_active(self, place_code):
        """3 cells with a field and 3 without fire: fewer than 5 active cells."""
        env, rates = place_code
        maps = rates[[5, 20, 35, 0, 1, 2]].copy()
        maps[3:] = 0.0
        trains, events = _epoch(env, 6, [[0, 3, 1, 4, 2, 5]] * 2)
        table = compare_replay_methods(
            env,
            events,
            replay_methods=[dataclasses.replace(CHENANI_2019, min_epoch_events=1)],
            spike_times=trains,
            encoding_models=maps,
            dt=DT,
            rng=0,
        )
        assert table["score"].isna().all()
        assert not table["is_replay"].any()
        assert (table["error"] != "").all()
