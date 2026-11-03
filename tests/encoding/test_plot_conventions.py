"""Static plots show physical axes, keep unobserved bins NaN and respect the caller."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.decoding import DecodingResult
from neurospatial.encoding import (
    compute_boundary_vector_rates,
    plot_head_direction_tuning,
    plot_object_vector_tuning,
    plot_phase_precession,
    plot_population_coverage,
    population_coverage,
)
from neurospatial.encoding.directional import DirectionalRateResult
from neurospatial.events import peri_event_histogram, plot_peri_event_histogram
from neurospatial.ops import geodesic_rbf_basis, plot_basis_functions
from neurospatial.regions import Regions, plot_regions
from neurospatial.simulation import open_field_session, plot_session_summary
from neurospatial.stats import plot_circular_basis_tuning


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def square_env() -> Environment:
    """A fully active 2-D grid, so bin numbering is the grid's row-major order."""
    xx, yy = np.meshgrid(np.arange(0.0, 50.0, 2.5), np.arange(0.0, 50.0, 2.5))
    return Environment.from_samples(
        np.column_stack([xx.ravel(), yy.ravel()]), bin_size=10.0, units="cm"
    )


@pytest.fixture(scope="module")
def track_env() -> Environment:
    return Environment.from_samples(
        np.linspace(0.0, 100.0, 200)[:, None], bin_size=5.0, units="cm"
    )


def _gaussian_posterior(env: Environment, truth: np.ndarray, width: float = 8.0):
    """Posterior mass per bin around ``truth`` ``(n_time, n_dims)``."""
    squared = ((env.bin_centers[None] - truth[:, None]) ** 2).sum(-1)
    posterior = np.exp(-squared / (2 * width**2))
    return posterior / posterior.sum(axis=1, keepdims=True)


def _colorbar_labels(fig, ax) -> list[str]:
    return [other.get_ylabel() for other in fig.axes if other is not ax]


# --- population coverage -----------------------------------------------------


@pytest.mark.parametrize("dims", ["2d", "1d"])
def test_population_coverage_count_keeps_unobserved_nan(dims, square_env, track_env):
    env = square_env if dims == "2d" else track_env
    rates = np.zeros((3, env.n_bins))
    rates[0, : env.n_bins // 3] = 5.0
    rates[1, : env.n_bins // 4] = 6.0
    rates[:, -5:] = np.nan  # no map observed the last bins
    result = population_coverage(env, rates, bandwidth=0.0)
    assert (~result.is_observed).sum() == 5

    ax = plot_population_coverage(env, result, show_field_count=True)
    if dims == "2d":
        shown = np.ma.filled(ax.collections[0].get_array().T.ravel(), np.nan)
        # Fields per bin appear on a colorbar with integer ticks.
        colorbar = next(other for other in ax.figure.axes if other is not ax)
        assert colorbar.get_ylabel() == "Fields per bin"
        ticks = colorbar.get_yticks()
        np.testing.assert_array_equal(
            ticks, np.arange(int(result.field_count.max()) + 1)
        )
    else:
        shown = ax.lines[0].get_ydata()
        assert ax.get_ylabel() == "Fields per bin"
    np.testing.assert_array_equal(np.isnan(shown), ~result.is_observed)
    observed = result.is_observed
    np.testing.assert_array_equal(shown[observed], result.field_count[observed])


def test_population_coverage_binary_has_no_continuous_colorbar(square_env):
    rates = np.zeros((2, square_env.n_bins))
    rates[0, :10] = 5.0
    result = population_coverage(square_env, rates, bandwidth=0.0)
    ax = plot_population_coverage(square_env, result)
    assert len(ax.figure.axes) == 1


# --- decoding ----------------------------------------------------------------


def test_decoding_plot_1d_axis_in_cm(track_env):
    times = np.arange(40) * 0.1
    truth = (50 + 40 * np.sin(times))[:, None]
    result = DecodingResult(_gaussian_posterior(track_env, truth), track_env, times)
    ax = result.plot(show_map=True, true_times=times, true_positions=truth)

    centers = track_env.bin_centers[:, 0]
    low, high = ax.get_ylim()
    assert low <= centers.min()
    assert high >= centers.max()
    assert high - low < 1.2 * (centers.max() - centers.min())  # no bin-index axis
    assert "cm" in ax.get_ylabel()
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "MAP",
        "Actual position",
    ]
    # Lines are drawn at physical positions: the MAP of a delta posterior sits
    # on its bin center, the true position at its bin's center.
    map_line = next(line for line in ax.lines if line.get_label() == "MAP")
    np.testing.assert_allclose(map_line.get_ydata(), result.map_position[:, 0])
    # The colorbar is on by default.
    assert "Probability" in _colorbar_labels(ax.figure, ax)


def _gapped_track_envs():
    from neurospatial.simulation import mazes

    positions = np.r_[np.linspace(0, 40, 200), np.linspace(70, 100, 200)][:, None]
    return {
        "w_maze": mazes.make_w_maze().env_track,
        "gapped_1d_grid": Environment.from_samples(positions, bin_size=5.0, units="cm"),
    }


@pytest.mark.parametrize("name", ["w_maze", "gapped_1d_grid"])
def test_decoding_plot_places_data_at_bin_centers_across_gaps(name):
    from neurospatial.environment._queries import _track_extents

    env = _gapped_track_envs()[name]
    starts, ends = _track_extents(env)
    centers = (starts + ends) / 2
    n = env.n_bins
    times = np.arange(n) * 0.1
    result = DecodingResult(np.eye(n), env, times)
    truth = result.map_position  # the position of each decoded bin
    ax = result.plot(show_map=True, true_times=times, true_positions=truth)

    map_line = next(line for line in ax.lines if line.get_label() == "MAP")
    np.testing.assert_allclose(map_line.get_ydata(), centers[result.map_bin])
    actual = next(line for line in ax.lines if line.get_label() == "Actual position")
    np.testing.assert_allclose(actual.get_ydata(), centers[result.map_bin])
    # Gaps between segments are blank, not stretched over: the image's rows lie
    # at their bins' edges.
    mesh = ax.collections[0]
    y_edges = np.unique(mesh.get_coordinates()[..., 1])
    assert set(np.round(np.r_[starts, ends], 9)) <= set(np.round(y_edges, 9))
    blank = np.isnan(np.ma.filled(mesh.get_array().astype(float), np.nan)).all(axis=1)
    order = np.argsort(starts)
    n_gaps = int((starts[order][1:] > ends[order][:-1] + 1e-9).sum())
    assert n_gaps > 0
    assert blank.sum() == n_gaps


def test_decoding_plot_2d_panels_share_one_color_scale(square_env):
    times = np.arange(6) * 0.1
    truth = np.column_stack([10 + 20 * times, np.full(6, 40.0)])
    result = DecodingResult(
        _gaussian_posterior(square_env, truth, 3.0), square_env, times
    )
    ax_x, ax_y = result.plot()
    assert ax_x.images[0].get_clim() == ax_y.images[0].get_clim()
    assert ax_x.images[0].get_clim()[0] == 0.0
    top = max(ax_x.images[0].get_array().max(), ax_y.images[0].get_array().max())
    assert ax_x.images[0].get_clim()[1] == pytest.approx(top)


def test_decoding_marginal_on_a_rotated_grid_has_grid_many_columns():
    from neurospatial.ops.transforms import rotate_2d

    side = np.arange(0.0, 40.0, 2.0)
    lattice = np.stack(np.meshgrid(side, side), -1).reshape(-1, 2)
    env = Environment.from_samples(lattice, bin_size=2.0, units="cm").apply_transform(
        rotate_2d(np.deg2rad(30))
    )
    rng = np.random.default_rng(0)
    posterior = rng.random((5, env.n_bins))
    posterior /= posterior.sum(axis=1, keepdims=True)
    ax_x, ax_y = DecodingResult(posterior, env, np.arange(5) * 0.1).plot()
    for ax in (ax_x, ax_y):
        columns = ax.images[0].get_array().shape[0]
        assert 20 <= columns <= 2 * 20 * 1.5  # about the grid's width in bins
        np.testing.assert_allclose(ax.images[0].get_array().sum(axis=0), 1.0)


def test_decoding_plot_2d_marginals(square_env):
    env = square_env
    times = np.arange(12) * 0.1
    truth = np.column_stack([10 + 25 * times, 40 - 20 * times])
    posterior = _gaussian_posterior(env, truth, width=6.0)
    result = DecodingResult(posterior, env, times)

    ax_x, ax_y = result.plot(show_map=True, true_times=times, true_positions=truth)
    assert ax_x.get_ylabel() == "x (cm)"
    assert ax_y.get_ylabel() == "y (cm)"
    assert ax_y.get_xlabel() == "Time (s)"

    # The panels hold the posterior mass summed over the other coordinate.
    n_x = len(np.unique(env.bin_centers[:, 0]))
    n_y = len(np.unique(env.bin_centers[:, 1]))
    by_cell = np.zeros((len(times), n_x, n_y))
    x_index = np.searchsorted(np.unique(env.bin_centers[:, 0]), env.bin_centers[:, 0])
    y_index = np.searchsorted(np.unique(env.bin_centers[:, 1]), env.bin_centers[:, 1])
    by_cell[:, x_index, y_index] = posterior
    np.testing.assert_allclose(ax_x.images[0].get_array().T, by_cell.sum(axis=2))
    np.testing.assert_allclose(ax_y.images[0].get_array().T, by_cell.sum(axis=1))
    # Each panel's actual-position line is the matching coordinate.
    for ax, coordinate in ((ax_x, 0), (ax_y, 1)):
        line = next(line for line in ax.lines if line.get_label() == "Actual position")
        np.testing.assert_allclose(line.get_ydata(), truth[:, coordinate])


def test_decoding_plot_2d_accepts_axes_pair_and_refuses_one(square_env):
    posterior = _gaussian_posterior(square_env, np.array([[20.0, 20.0]] * 3))
    result = DecodingResult(posterior, square_env, np.arange(3) * 0.1)
    _, (top, bottom) = plt.subplots(2, 1)
    drawn = result.plot(ax=(top, bottom))
    assert drawn == (top, bottom)
    assert top.images
    assert bottom.images
    _, single = plt.subplots()
    with pytest.raises(ValueError, match="ax=\\(ax_x, ax_y\\)"):
        result.plot(ax=single)


def test_decoding_plot_colorbar_can_be_turned_off(track_env):
    result = DecodingResult(
        _gaussian_posterior(track_env, np.array([[20.0], [30.0]])), track_env
    )
    ax = result.plot(colorbar=False)
    assert len(ax.figure.axes) == 1


# --- directional place fields --------------------------------------------------


def _directional_fields(env: Environment):
    n = env.n_bins
    from neurospatial.encoding.spatial import DirectionalPlaceFields

    ramp = np.linspace(1.0, 5.0, n)
    return DirectionalPlaceFields(
        firing_rates={"out": ramp, "back": ramp[::-1].copy()},
        occupancy={"out": np.ones(n), "back": np.ones(n)},
        env=env,
    )


def test_directional_place_fields_plot_axes(track_env, square_env):
    fields = _directional_fields(track_env)
    ax = fields.plot()
    assert "Bin index" not in ax.get_xlabel()
    assert "cm" in ax.get_xlabel()
    np.testing.assert_allclose(
        ax.lines[0].get_xdata(), np.sort(track_env.bin_centers[:, 0])
    )
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["out", "back"]

    panels = _directional_fields(square_env).plot()
    assert len(panels) == 2
    for panel, label in zip(panels, ("out", "back"), strict=True):
        assert panel.get_title() == label
        assert "cm" in panel.get_xlabel()
        assert "Bin index" not in panel.get_xlabel()
        # One map per panel, on shared color limits.
        assert panel.collections[0].get_clim() == (0.0, 5.0)


def test_directional_place_fields_break_lines_across_track_gaps():
    positions = np.r_[np.linspace(0, 40, 100), np.linspace(70, 100, 100)][:, None]
    env = Environment.from_samples(positions, bin_size=5.0, units="cm")
    ax = _directional_fields(env).plot()
    x = ax.lines[0].get_xdata()
    assert np.isnan(x).sum() == 1
    split = int(np.flatnonzero(np.isnan(x))[0])
    assert x[split + 1] - x[split - 1] > 20  # the gap is not drawn across


def test_directional_place_fields_2d_needs_one_axes_per_direction(square_env):
    fields = _directional_fields(square_env)
    _, axes = plt.subplots(1, 3)
    with pytest.raises(ValueError, match="one axes per direction"):
        fields.plot(ax=axes)
    _, pair = plt.subplots(1, 2)
    assert fields.plot(ax=pair) == tuple(pair)


# --- polar axes ----------------------------------------------------------------


@pytest.fixture(scope="module")
def head_direction_result() -> DirectionalRateResult:
    centers = np.linspace(-np.pi, np.pi, 60, endpoint=False)
    return DirectionalRateResult(
        firing_rate=10.0 * np.exp(2.0 * (np.cos(centers - np.pi / 2) - 1)),
        occupancy=np.full(60, 0.5),
        bin_centers=centers,
        bin_size=np.pi / 30,
        bandwidth=None,
    )


@pytest.mark.parametrize(
    "draw",
    [
        lambda ax, r, **_: plot_head_direction_tuning(r["hd"], ax=ax),
        lambda ax, r, **_: r["hd"].plot(ax=ax),
        lambda ax, r, **_: plot_object_vector_tuning(r["allocentric"][0], ax=ax),
        lambda ax, r, **_: plot_circular_basis_tuning(0.5, 1.0, ax=ax),
    ],
    ids=["head_direction", "directional_result", "object_vector", "circular_basis"],
)
def test_polar_plot_on_cartesian_axes_raises(
    draw, head_direction_result, rate_family_results
):
    results = {"hd": head_direction_result, **rate_family_results}
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="projection"):
        draw(ax, results)


def test_polar_refusal_names_an_alternative_the_function_accepts(rate_family_results):
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match=r"result\.plot\(ax=ax\)") as refusal:
        plot_object_vector_tuning(rate_family_results["allocentric"][0], ax=ax)
    assert "polar=False" not in str(refusal.value)
    # The suggested call works.
    assert rate_family_results["allocentric"][0].plot(ax=ax) is ax


def test_polar_plots_accept_the_cartesian_alternative(head_direction_result):
    _, ax = plt.subplots()
    assert head_direction_result.plot(ax=ax, polar=False) is ax
    assert plot_head_direction_tuning(head_direction_result, ax=ax, polar=False) is ax
    assert plot_circular_basis_tuning(0.5, 1.0, ax=ax, projection="linear") is ax


def test_head_direction_polar_plots_label_the_radial_axis(head_direction_result):
    preferred = np.rad2deg(head_direction_result.preferred_direction) % 360.0
    for ax in (
        head_direction_result.plot(),
        plot_head_direction_tuning(head_direction_result),
    ):
        assert "Hz" in ax.get_ylabel()
        # The radial tick labels sit opposite the preferred direction, away
        # from the tuning curve and the metrics box.
        gap = abs((ax.get_rlabel_position() - preferred + 180.0) % 360.0 - 180.0)
        assert gap == pytest.approx(180.0, abs=1.0)


# --- titles ----------------------------------------------------------------------


def _psth_inputs():
    spikes = np.array([0.8, 1.0, 1.2, 2.8, 3.0, 3.2])
    return peri_event_histogram(
        spikes, np.array([1.0, 3.0]), window=(-0.5, 0.5), dt=0.1
    )


def _coverage_inputs(env: Environment):
    rates = np.zeros((2, env.n_bins))
    rates[0, :5] = 5.0
    return population_coverage(env, rates, bandwidth=0.0)


@pytest.mark.parametrize(
    "draw",
    [
        lambda ax, env: env.plot(ax=ax),
        lambda ax, env: env.plot_field(np.arange(env.n_bins, dtype=float), ax=ax),
        lambda ax, env: plot_peri_event_histogram(_psth_inputs(), ax=ax),
        lambda ax, env: plot_population_coverage(env, _coverage_inputs(env), ax=ax),
    ],
    ids=["env_plot", "plot_field", "psth", "coverage"],
)
@pytest.mark.parametrize("env_name", ["grid_2d", "track_1d"])
def test_plots_keep_user_title(draw, env_name, square_env, track_env):
    env = square_env if env_name == "grid_2d" else track_env
    _, ax = plt.subplots()
    ax.set_title("X")
    assert draw(ax, env).get_title() == "X"


def test_linearized_track_field_and_outline_keep_user_title():
    track = Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=5.0, units="cm"
    )
    for draw in (
        lambda ax: track.plot_field(np.arange(track.n_bins, dtype=float), ax=ax),
        lambda ax: track.plot(ax=ax),
    ):
        _, ax = plt.subplots()
        ax.set_title("X")
        assert draw(ax).get_title() == "X"


def test_directional_place_fields_keep_user_title(track_env, square_env):
    _, ax = plt.subplots()
    ax.set_title("X")
    assert _directional_fields(track_env).plot(ax=ax).get_title() == "X"
    _, pair = plt.subplots(1, 2)
    pair[0].set_title("X")
    assert _directional_fields(square_env).plot(ax=pair)[0].get_title() == "X"


def test_psth_explicit_title_still_wins():
    _, ax = plt.subplots()
    ax.set_title("X")
    plot_peri_event_histogram(_psth_inputs(), ax=ax, title="Reward")
    assert ax.get_title() == "Reward"


def test_env_plot_default_title_joins_only_what_exists(square_env):
    assert square_env.plot().get_title() == "RegularGrid"  # no leading "()"
    unnamed = Environment.from_samples(
        np.stack(np.meshgrid(np.arange(0.0, 20.0), np.arange(0.0, 20.0)), -1).reshape(
            -1, 2
        ),
        bin_size=5.0,
        name="",
    )
    assert unnamed.plot().get_title() == "RegularGrid"
    named = Environment.from_samples(
        unnamed.bin_centers, bin_size=5.0, name="arena"
    ).plot()
    assert named.get_title() == "arena (RegularGrid)"


def test_functions_title_axes_they_create(square_env):
    assert (
        "Bin Coverage"
        in plot_population_coverage(
            square_env, _coverage_inputs(square_env)
        ).get_title()
    )
    assert "PSTH" in plot_peri_event_histogram(_psth_inputs()).get_title()
    assert "RegularGrid" in square_env.plot().get_title()


# --- colorbars ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def boundary_vector_rates(rate_family_inputs):
    env, times, positions, headings, spikes, _ = rate_family_inputs
    return compute_boundary_vector_rates(
        env,
        spikes,
        times,
        positions,
        headings,
        direction_frame="egocentric",
        boundary_source="analysis",
    )


@pytest.mark.parametrize(
    "name", ["allocentric", "egocentric", "view", "spatial", "boundary_vector"]
)
@pytest.mark.parametrize("single", [True, False], ids=["unit", "population"])
def test_rate_colorbars_labelled(
    name, single, rate_family_results, boundary_vector_rates
):
    results = {**rate_family_results, "boundary_vector": boundary_vector_rates}
    result = results[name]
    fig, ax = plt.subplots()
    if single:
        result[0].plot(ax=ax)
    else:
        result.plot(0, ax=ax)
    labels = _colorbar_labels(fig, ax)
    assert any("Hz" in label for label in labels), labels
    assert "Firing rate (Hz)" in labels


def test_object_vector_tuning_colorbar_flag(rate_family_results):
    result = rate_family_results["allocentric"][0]
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    plot_object_vector_tuning(result, ax=ax)
    assert "Firing rate (Hz)" in _colorbar_labels(fig, ax)
    fig2, ax2 = plt.subplots(subplot_kw={"projection": "polar"})
    plot_object_vector_tuning(result, ax=ax2, colorbar=False)
    assert _colorbar_labels(fig2, ax2) == []


# --- axis labels ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("angle_unit", "label", "top_tick"),
    [("rad", "Phase (radians)", "4π"), ("deg", "Phase (degrees)", "720°")],
)
def test_phase_precession_y_label_follows_angle_unit(angle_unit, label, top_tick):
    positions = np.linspace(0, 50, 80)
    phases = (2 * np.pi - positions * 0.1) % (2 * np.pi)
    if angle_unit == "deg":
        phases = np.rad2deg(phases)
    ax = plot_phase_precession(positions, phases, angle_unit=angle_unit)
    assert ax.get_ylabel() == label
    assert ax.get_yticklabels()[-1].get_text() == top_tick
    # The plotted phases are in radians either way.
    assert ax.collections[0].get_offsets()[:, 1].max() <= 4 * np.pi


# --- plot_regions ----------------------------------------------------------------------


def test_plot_regions_equal_aspect_and_colors():
    from shapely.geometry import box

    regions = Regions()
    regions.add("left", polygon=box(0.0, 0.0, 20.0, 20.0))
    regions.add("middle", polygon=box(30.0, 0.0, 50.0, 20.0))
    regions.add("right", polygon=box(60.0, 0.0, 80.0, 20.0))
    regions.add("well", point=(40.0, 40.0))
    _, ax = plt.subplots()
    plot_regions(regions, ax=ax)

    assert ax.get_aspect() == 1.0
    faces = [tuple(patch.get_facecolor()) for patch in ax.patches]
    assert len(faces) == 3
    assert len(set(faces)) == 3
    legend = ax.get_legend()
    assert [text.get_text() for text in legend.get_texts()] == list(regions)
    # The legend sits inside the axes rather than to the right of them.
    ax.figure.canvas.draw()
    assert ax.bbox.x0 <= legend.get_window_extent().x0
    assert legend.get_window_extent().x1 <= ax.bbox.x1


def test_plot_regions_keeps_an_explicit_color():
    from shapely.geometry import box

    regions = Regions()
    regions.add("a", polygon=box(0.0, 0.0, 1.0, 1.0))
    regions.add("b", polygon=box(2.0, 0.0, 3.0, 1.0))
    _, ax = plt.subplots()
    plot_regions(regions, ax=ax, a={"facecolor": "tab:red"})
    assert ax.patches[0].get_facecolor()[:3] == pytest.approx(
        (0.839, 0.153, 0.157), abs=1e-3
    )
    assert ax.patches[1].get_facecolor()[:3] != ax.patches[0].get_facecolor()[:3]


# --- env.plot readability ------------------------------------------------------------


@pytest.mark.parametrize("env_name", ["grid_2d", "track_1d"])
def test_env_plot_shows_ticks_with_units(env_name, square_env, track_env):
    env = square_env if env_name == "grid_2d" else track_env
    ax = env.plot()
    assert "cm" in ax.get_xlabel()
    assert ax.xaxis.get_tick_params()["labelbottom"]
    assert len(ax.get_xticks()) > 1


def test_env_plot_markers_shrink_on_a_fine_grid():
    xx, yy = np.meshgrid(np.arange(0.0, 100.0, 1.0), np.arange(0.0, 100.0, 1.0))
    fine = Environment.from_samples(
        np.column_stack([xx.ravel(), yy.ravel()]), bin_size=2.0, units="cm"
    )
    coarse = Environment.from_samples(
        np.column_stack([xx.ravel(), yy.ravel()]), bin_size=20.0, units="cm"
    )

    def marker_area(env):
        ax = env.plot()
        return max(
            float(np.max(c.get_sizes()))
            for c in ax.collections
            if hasattr(c, "get_sizes") and len(c.get_sizes())
        )

    assert marker_area(fine) < marker_area(coarse)


# --- axes supplied by the caller ---------------------------------------------------------


def test_summary_plots_accept_axes():
    session = open_field_session(duration=20.0, n_place_cells=8, rng=0)
    fig, _ = plot_session_summary(session)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 8))
    mine = np.array([fig.add_subplot(3, 3, i + 1) for i in range(9)], dtype=object)
    returned_fig, returned = plot_session_summary(session, ax=mine)
    assert returned_fig is fig
    assert list(returned) == list(mine)
    assert mine[0].collections  # trajectory scattered into the first axes
    assert mine[2].get_title() == f"Cell {int(session.unit_ids[0])}"
    assert fig._suptitle is None  # the caller's figure keeps its own title

    with pytest.raises(ValueError, match="nine axes"):
        plot_session_summary(session, ax=mine[:4])


def test_session_summary_raster_follows_unit_ids():
    session = open_field_session(duration=20.0, n_place_cells=8, rng=0)
    chosen = [int(session.unit_ids[5]), int(session.unit_ids[1])]
    fig, axes = plot_session_summary(session, unit_ids=chosen)
    raster = axes[-1]
    assert [t.get_text() for t in raster.get_yticklabels()] == [str(c) for c in chosen]
    assert raster.get_ylim() == (-0.5, 1.5)
    drawn = sorted(
        float(row)
        for c in raster.collections
        for row in np.unique(c.get_offsets()[:, 1])
    )
    assert set(drawn) <= {0.0, 1.0}
    plt.close(fig)


def test_basis_functions_accept_axes(square_env):
    basis = geodesic_rbf_basis(square_env, n_centers=6, bandwidth=10.0, rng=0)
    fig, axes = plt.subplots(2, 2)
    returned = plot_basis_functions(square_env, basis, ax=axes, indices=[0, 1, 2])
    assert returned is fig
    assert [a.get_title() for a in axes.ravel()[:3]] == [
        "Basis 0",
        "Basis 1",
        "Basis 2",
    ]
    assert not axes.ravel()[3].get_visible()
    assert fig._suptitle is None
    with pytest.raises(ValueError, match="one axes per basis function"):
        plot_basis_functions(square_env, basis, ax=axes.ravel()[:2], indices=[0, 1, 2])
