"""Tests for simulation validation module."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from neurospatial.simulation import PlaceCellModel, SimulationSession, simulate_session
from neurospatial.simulation.validation import (
    SimulationValidationResult,
    validate_simulation,
)


def test_plot_session_summary_labels(simple_2d_env):
    import matplotlib.pyplot as plt

    from neurospatial.simulation import plot_session_summary

    session = simulate_session(
        simple_2d_env, duration=30, n_cells=5, seed=0, show_progress=False
    )
    fig, axes = plot_session_summary(session, unit_ids=[1, 3])
    assert [ax.get_title() for ax in axes[2:4]] == ["Cell 1", "Cell 3"]
    assert all(not ax.axison for ax in axes[4:8])
    plt.close(fig)
    labels = np.array([7, 11, 19, 23, 31])
    relabeled = replace(
        session,
        unit_ids=labels,
        ground_truth={
            int(new): session.ground_truth[int(old)]
            for old, new in zip(session.unit_ids, labels, strict=True)
        },
    )
    fig, axes = plot_session_summary(relabeled, unit_ids=[31, 11])
    assert [ax.get_title() for ax in axes[2:4]] == ["Cell 31", "Cell 11"]
    assert [tick.get_text() for tick in axes[-1].get_yticklabels()] == [
        "7",
        "11",
        "19",
        "23",
        "31",
    ]
    plt.close(fig)


@pytest.fixture(scope="module")
def place_session(connected_arena_env):
    """Five 12 cm place fields over three minutes: every cell is recovered."""
    return simulate_session(
        connected_arena_env,
        duration=180.0,
        n_cells=5,
        cell_type="place",
        width=12.0,
        max_rate=20.0,
        seed=0,
        show_progress=False,
    )


@pytest.fixture(scope="module")
def place_report(place_session):
    return validate_simulation(place_session)


def test_select_by_label(place_session, place_report):
    """unit_ids select by label, in the requested order, after relabelling too."""
    session, full = place_session, place_report
    assert full.n_evaluated == 5
    labels = np.array([7, 11, 19, 23, 31], dtype=np.int64)
    relabeled = replace(
        session,
        unit_ids=labels,
        ground_truth={
            int(label): session.ground_truth[i] for i, label in enumerate(labels)
        },
    )
    reordered = validate_simulation(relabeled, unit_ids=[31, 7, 19])
    np.testing.assert_array_equal(reordered.unit_ids, [31, 7, 19])
    for key in ("center_errors", "correlations"):
        np.testing.assert_array_equal(
            getattr(reordered, key), getattr(full, key)[[4, 0, 2]]
        )
    with pytest.raises(ValueError, match=r"99.*Valid labels:.*7.*11.*\n.*\nFix:"):
        validate_simulation(relabeled, unit_ids=[99])
    with pytest.raises(TypeError):
        validate_simulation(env=session.env)


class TestValidateSimulation:
    """Tests for validate_simulation() on a session whose cells can be evaluated."""

    def test_returns_a_frozen_result_with_read_only_arrays(self, place_report):
        from dataclasses import FrozenInstanceError

        assert isinstance(place_report, SimulationValidationResult)
        assert place_report.n_selected == 5
        assert place_report.n_evaluated == 5
        for array in (
            place_report.center_errors,
            place_report.correlations,
            place_report.max_center_errors,
            place_report.spike_counts,
            place_report.unit_ids,
        ):
            assert array.shape == (5,)
            with pytest.raises(ValueError, match="read-only"):
                array[0] = 0
        with pytest.raises(FrozenInstanceError):
            place_report.min_correlation = 0.0  # type: ignore[misc]

    def test_center_error_is_the_distance_from_the_rate_peak_to_the_center(
        self, place_session, place_report
    ):
        from neurospatial.encoding import compute_spatial_rate

        env = place_session.env
        for row, model in enumerate(place_session.models):
            rate = compute_spatial_rate(
                env,
                place_session.spike_times[row],
                place_session.times,
                place_session.positions,
            ).firing_rate
            peak = env.bin_centers[np.nanargmax(rate)]
            assert place_report.center_errors[row] == pytest.approx(
                np.linalg.norm(peak - model.center)
            )
        # A recovered 12 cm field peaks within one field width of its center.
        assert np.all(place_report.center_errors <= 12.0)

    def test_recovered_fields_correlate_with_the_true_fields(self, place_report):
        assert np.all(place_report.correlations > 0.9)
        assert np.all(place_report.correlations <= 1.0)

    def test_status_follows_both_thresholds(self, place_session, place_report):
        np.testing.assert_allclose(place_report.max_center_errors, 1.5 * 12.0)
        assert place_report.unit_status == ("passed",) * 5
        assert place_report.passed

        strict = validate_simulation(
            place_session, max_center_error=0.0, min_correlation=0.5
        )
        expected = tuple(
            "passed" if error == 0.0 else "failed" for error in strict.center_errors
        )
        assert strict.unit_status == expected
        assert "failed" in expected
        assert not strict.passed
        np.testing.assert_array_equal(strict.max_center_errors, 0.0)
        impossible = validate_simulation(place_session, min_correlation=1.01)
        assert impossible.unit_status == ("failed",) * 5

    def test_summary_and_summary_table_follow_the_result_conventions(
        self, place_report
    ):
        summary = place_report.summary()
        assert summary["n_selected"] == summary["n_evaluated"] == 5
        assert summary["passed"] is True
        assert summary["mean_center_error"] == pytest.approx(
            np.mean(place_report.center_errors)
        )
        assert all(np.isscalar(value) for value in summary.values())

        table = place_report.summary_table()
        assert table.index.name == "unit_id"
        np.testing.assert_array_equal(table.index, place_report.unit_ids)
        assert list(table.columns) == [
            "center_error",
            "correlation",
            "max_center_error",
            "spike_count",
            "status",
            "skip_reason",
        ]
        assert table.attrs["method"] == "diffusion_kde"
        assert table.attrs["units"] == "cm"
        assert table.attrs["min_correlation"] == 0.9
        assert "SimulationValidationResult(n_selected=5" in repr(place_report)

    def test_report_text_states_the_counts_and_verdict(self, place_report):
        text = place_report.report()
        assert "Cells selected: 5 (of 5 total)" in text
        assert "Cells fully evaluated: 5" in text
        assert "Threshold: 1.5 x each cell's true field width" in text
        assert "Overall: PASSED" in text

    def test_plot_draws_each_evaluated_unit(self, place_report):
        import matplotlib.pyplot as plt

        ax = place_report.plot()
        try:
            np.testing.assert_allclose(
                ax.collections[0].get_offsets(),
                np.column_stack(
                    [place_report.center_errors, place_report.correlations]
                ),
            )
        finally:
            plt.close(ax.figure)

    def test_unit_ids_select_and_order_units(self, place_session, place_report):
        selected = validate_simulation(place_session, unit_ids=[4, 0, 2])
        np.testing.assert_array_equal(selected.unit_ids, [4, 0, 2])
        np.testing.assert_array_equal(
            selected.center_errors, place_report.center_errors[[4, 0, 2]]
        )
        assert selected.n_session_units == 5

    def test_a_silent_unit_is_not_evaluable(self, place_session):
        trains = list(place_session.spike_times)
        trains[1] = np.array([])
        session = replace(place_session, spike_times=trains)
        report = validate_simulation(session)
        assert report.unit_status[1] == "not_evaluable"
        assert report.skip_reasons[1] == "no_spikes"
        assert report.spike_counts[1] == 0
        assert report.n_evaluated == 4
        assert not report.passed

    def test_binned_method_evaluates_cells(self, place_session):
        report = validate_simulation(place_session, method="binned")
        assert report.n_evaluated == 5
        assert report.summary_table().attrs["method"] == "binned"

    def test_validate_simulation_reproducible(self, place_session, place_report):
        again = validate_simulation(place_session)
        assert np.all(np.isfinite(place_report.center_errors))
        assert np.all(np.isfinite(place_report.correlations))
        np.testing.assert_array_equal(again.center_errors, place_report.center_errors)
        np.testing.assert_array_equal(again.correlations, place_report.correlations)

    def test_validate_simulation_invalid_session_type(self):
        with pytest.raises(TypeError, match="SimulationSession"):
            validate_simulation("not a session")

    def test_session_without_ground_truth_is_refused(self, place_session):
        with pytest.raises(ValueError, match="ground_truth"):
            replace(place_session, ground_truth={})


def test_default_center_error_threshold(simple_2d_env):
    """The default limit is 1.5 true field widths, whatever the bins."""
    session = simulate_session(
        simple_2d_env,
        duration=10.0,
        n_cells=1,
        cell_type="place",
        seed=0,
        show_progress=False,
    )

    result = validate_simulation(session)
    np.testing.assert_allclose(result.max_center_errors, [1.5 * 11.1])
    assert "1.5 x each cell's true field width" in result.report()


def test_detected_center_ignores_unresolved_bins():
    """The detected peak is taken over finite bins only.

    The track extends far beyond the visited stretch. An unsmoothed binned
    rate keeps the unvisited bins NaN and its peak at the generating bin;
    a plain argmax would pick the first NaN bin as the peak.
    """
    from neurospatial import Environment

    env = Environment.from_samples(np.linspace(0, 140, 281)[:, None], bin_size=10.0)
    times = np.arange(0, 60, 0.01)
    positions = (20 + 20 * np.sin(times))[:, None]  # visits [0, 40] only
    near_bin_1 = np.abs(positions[:, 0] - env.bin_centers[1, 0]) < 2.0
    spike_times = times[near_bin_1][::5]

    session = SimulationSession(
        env=env,
        spike_times=[spike_times],
        unit_ids=np.array([0], dtype=np.int64),
        models=[
            PlaceCellModel(env, center=env.bin_centers[1], width=5.0, max_rate=10.0)
        ],
        metadata={},
        positions=positions,
        times=times,
        ground_truth={
            0: {"center": env.bin_centers[1], "width": 5.0, "max_rate": 10.0}
        },
    )

    from neurospatial.encoding import compute_spatial_rate

    field = compute_spatial_rate(
        env, spike_times, times, positions, method="binned", bandwidth=0.0
    ).firing_rate
    assert np.isnan(field).any()
    assert np.isnan(field[np.argmax(field)])
    result = validate_simulation(session, method="binned", bandwidth=0.0)
    assert result.center_errors[0] == 0.0


def test_true_field_is_the_simulated_field(hairpin_track_env):
    """Geodesic place cells are compared with their own geodesic fields.

    On a hairpin, a Euclidean field centered on one arm also covers the other
    arm 1 cm away, which the simulated geodesic cell never fires on.
    """
    from scipy.stats import pearsonr

    from neurospatial.encoding import compute_spatial_rate

    env = hairpin_track_env
    session = simulate_session(
        env,
        duration=120.0,
        n_cells=3,
        trajectory_method="laps",
        n_laps=12,
        metric="geodesic",
        width=15.0,
        max_rate=30.0,
        seed=0,
        show_progress=False,
    )
    report = validate_simulation(session)
    assert report.n_evaluated == 3
    for row, model in enumerate(session.models):
        detected = compute_spatial_rate(
            env, session.spike_times[row], session.times, session.positions
        ).firing_rate
        truth = model.firing_rate(np.zeros(env.n_bins), env.bin_centers)
        both = np.isfinite(detected) & np.isfinite(truth)
        expected = pearsonr(detected[both], truth[both])[0]
        assert report.correlations[row] == pytest.approx(expected, rel=1e-12)


def test_mixed_sessions_validate_their_place_cells(connected_arena_env):
    """Border and grid units are not evaluable; place units are evaluated."""
    session = simulate_session(
        connected_arena_env,
        duration=120.0,
        n_cells=5,
        cell_type="mixed",
        width=15.0,
        max_rate=30.0,
        seed=0,
        show_progress=False,
    )
    cell_types = [session.ground_truth[int(u)]["cell_type"] for u in session.unit_ids]
    with pytest.warns(UserWarning, match="not place fields"):
        report = validate_simulation(session)
    for cell_type, status, reason in zip(
        cell_types, report.unit_status, report.skip_reasons, strict=True
    ):
        if cell_type == "place":
            assert status in {"passed", "failed"} and reason is None
        else:
            assert (status, reason) == ("not_evaluable", "not_a_place_field")
    assert report.n_evaluated == cell_types.count("place") == 3
    assert not report.passed


def test_default_center_error_threshold_on_hairpin(hairpin_track_env):
    """On a track the default limit is also 1.5 true field widths."""
    times = np.arange(0, 20, 0.01)
    positions = np.column_stack([50 + 40 * np.sin(times), np.zeros_like(times)])
    center = np.array([50.0, 0.0])
    near_center = np.linalg.norm(positions - center, axis=1) < 3.0
    session = SimulationSession(
        env=hairpin_track_env,
        spike_times=[times[near_center][::5]],
        unit_ids=np.array([0], dtype=np.int64),
        models=[
            PlaceCellModel(hairpin_track_env, center=center, width=15.0, max_rate=10.0)
        ],
        metadata={},
        positions=positions,
        times=times,
        ground_truth={0: {"center": center, "width": 15.0, "max_rate": 10.0}},
    )

    result = validate_simulation(session)
    np.testing.assert_allclose(result.max_center_errors, [22.5])


class TestPlotSessionSummary:
    """Tests for plot_session_summary() function."""

    def test_plot_session_summary_returns_tuple(self, simple_2d_env):
        """plot_session_summary() should return (fig, axes) tuple."""
        simple_2d_env = simple_2d_env.with_metadata(units="cm")

        session = simulate_session(
            simple_2d_env,
            duration=30.0,
            n_cells=5,
            cell_type="place",
            seed=42,
            show_progress=False,
        )

        from neurospatial.simulation.validation import plot_session_summary

        result = plot_session_summary(session)

        # Should return tuple
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_plot_session_summary_returns_figure_and_axes(self, simple_2d_env):
        """plot_session_summary() should return matplotlib Figure and axes."""
        import matplotlib.figure
        import matplotlib.pyplot as plt

        simple_2d_env = simple_2d_env.with_metadata(units="cm")

        session = simulate_session(
            simple_2d_env,
            duration=30.0,
            n_cells=5,
            seed=42,
            show_progress=False,
        )

        from neurospatial.simulation.validation import plot_session_summary

        fig, axes = plot_session_summary(session)

        # Check types
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(axes, np.ndarray)

        plt.close(fig)

    def test_plot_session_summary_default_unit_ids(self, simple_2d_env):
        """plot_session_summary() should default to first 6 cells."""
        import matplotlib.pyplot as plt

        simple_2d_env = simple_2d_env.with_metadata(units="cm")

        session = simulate_session(
            simple_2d_env,
            duration=30.0,
            n_cells=10,
            seed=42,
            show_progress=False,
        )

        from neurospatial.simulation.validation import plot_session_summary

        fig, _ = plot_session_summary(session)

        # Should not crash with 10 cells
        assert fig is not None

        plt.close(fig)

    def test_plot_session_summary_custom_unit_ids(self, simple_2d_env):
        """plot_session_summary() should accept custom unit_ids."""
        import matplotlib.pyplot as plt

        simple_2d_env = simple_2d_env.with_metadata(units="cm")

        session = simulate_session(
            simple_2d_env,
            duration=30.0,
            n_cells=10,
            seed=42,
            show_progress=False,
        )

        from neurospatial.simulation.validation import plot_session_summary

        # Plot specific cells
        fig, _ = plot_session_summary(session, unit_ids=[0, 2, 5])

        assert fig is not None

        plt.close(fig)

    def test_plot_session_summary_custom_figsize(self, simple_2d_env):
        """plot_session_summary() should accept custom figsize."""
        import matplotlib.pyplot as plt

        simple_2d_env = simple_2d_env.with_metadata(units="cm")

        session = simulate_session(
            simple_2d_env,
            duration=30.0,
            n_cells=5,
            seed=42,
            show_progress=False,
        )

        from neurospatial.simulation.validation import plot_session_summary

        fig, _ = plot_session_summary(session, figsize=(12, 8))

        # Check figsize was applied
        assert fig.get_size_inches()[0] == 12
        assert fig.get_size_inches()[1] == 8

        plt.close(fig)

    def test_plot_session_summary_with_empty_spike_trains(self, simple_2d_env):
        """plot_session_summary() should handle cells with no spikes."""
        import matplotlib.pyplot as plt

        simple_2d_env = simple_2d_env.with_metadata(units="cm")

        session = simulate_session(
            simple_2d_env,
            duration=5.0,  # Very short, may have empty trains
            n_cells=3,
            seed=42,
            show_progress=False,
        )

        from neurospatial.simulation.validation import plot_session_summary

        # Should not crash even if some cells have no spikes
        fig, _ = plot_session_summary(session)

        assert fig is not None

        plt.close(fig)

    def test_plot_session_summary_invalid_unit_ids(self, simple_2d_env):
        """plot_session_summary() should raise error for invalid unit_ids."""
        simple_2d_env = simple_2d_env.with_metadata(units="cm")

        session = simulate_session(
            simple_2d_env,
            duration=30.0,
            n_cells=5,
            seed=42,
            show_progress=False,
        )

        from neurospatial.simulation.validation import plot_session_summary

        # Try to plot non-existent cells
        with pytest.raises(ValueError, match="unit_ids"):
            plot_session_summary(session, unit_ids=[0, 10, 20])

    def test_plot_session_summary_invalid_session_type(self):
        """plot_session_summary() should raise error for invalid session."""
        from neurospatial.simulation.validation import plot_session_summary

        with pytest.raises(TypeError):
            plot_session_summary("not a session")

    def test_plot_session_summary_has_trajectory_plot(self, simple_2d_env):
        """plot_session_summary() should include trajectory visualization."""
        import matplotlib.pyplot as plt

        simple_2d_env = simple_2d_env.with_metadata(units="cm")

        session = simulate_session(
            simple_2d_env,
            duration=30.0,
            n_cells=5,
            seed=42,
            show_progress=False,
        )

        from neurospatial.simulation.validation import plot_session_summary

        fig, axes = plot_session_summary(session)

        # At least one subplot should exist
        assert len(axes.flat) > 0

        plt.close(fig)

    def test_plot_session_summary_reproducible(self, simple_2d_env):
        """plot_session_summary() should produce consistent plots for same session."""
        import matplotlib.pyplot as plt

        simple_2d_env = simple_2d_env.with_metadata(units="cm")

        session = simulate_session(
            simple_2d_env,
            duration=30.0,
            n_cells=3,
            seed=42,
            show_progress=False,
        )

        from neurospatial.simulation.validation import plot_session_summary

        fig1, axes1 = plot_session_summary(session, unit_ids=[0, 1, 2])
        fig2, axes2 = plot_session_summary(session, unit_ids=[0, 1, 2])

        # Same session should produce same structure
        assert fig1.get_size_inches()[0] == fig2.get_size_inches()[0]
        assert fig1.get_size_inches()[1] == fig2.get_size_inches()[1]
        assert len(axes1.flat) == len(axes2.flat)

        plt.close(fig1)
        plt.close(fig2)

    def test_plot_session_summary_truncates_many_cells(self, simple_2d_env):
        """plot_session_summary() should warn and truncate when >6 cells requested."""
        import warnings

        import matplotlib.pyplot as plt

        simple_2d_env = simple_2d_env.with_metadata(units="cm")

        session = simulate_session(
            simple_2d_env,
            duration=30.0,
            n_cells=15,
            seed=42,
            show_progress=False,
        )

        from neurospatial.simulation.validation import plot_session_summary

        # Should emit UserWarning about truncation
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig, _ = plot_session_summary(session, unit_ids=list(range(10)))

            # Filter for just the truncation warning (other warnings may be emitted)
            truncation_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, UserWarning)
                and "Only first 6 will be plotted" in str(warning.message)
            ]
            assert len(truncation_warnings) == 1

        plt.close(fig)


def test_field_correlation_needs_only_center_and_width(simple_2d_env):
    """Pearson correlation ignores the true field's baseline and peak rate."""
    session = simulate_session(
        simple_2d_env,
        duration=30,
        n_cells=3,
        seed=0,
        show_progress=False,
        max_rate=20.0,
        baseline_rate=0.01,
        width=6.0,
    )
    shape_only = replace(
        session,
        ground_truth={
            label: {"center": truth["center"], "width": truth["width"]}
            for label, truth in session.ground_truth.items()
        },
    )
    full = validate_simulation(session)
    np.testing.assert_allclose(
        validate_simulation(shape_only).correlations,
        full.correlations,
        rtol=1e-12,
    )
    assert np.isfinite(full.correlations).any()


@pytest.mark.slow
def test_default_thresholds_separate_correct_from_misspecified_fields():
    """Correct ground truth passes; a center one width off, or a 3x width, fails."""
    from neurospatial.simulation import open_field_session

    session = open_field_session(duration=600.0, n_place_cells=4, seed=0)
    report = validate_simulation(session)
    assert report.unit_status == ("passed",) * 4
    widths = np.array([session.ground_truth[int(u)]["width"] for u in session.unit_ids])
    np.testing.assert_allclose(report.max_center_errors, 1.5 * widths)

    def with_truth(change):
        truth = {label: change(t) for label, t in session.ground_truth.items()}
        return validate_simulation(replace(session, ground_truth=truth))

    moved = with_truth(
        lambda t: {**t, "center": np.asarray(t["center"]) + np.array([t["width"], 0.0])}
    )
    assert moved.unit_status == ("failed",) * 4
    widened = with_truth(lambda t: {**t, "width": 3 * t["width"]})
    assert widened.unit_status == ("failed",) * 4
