"""Public function signatures follow the shared input conventions."""

import importlib
import inspect
import re

import pytest
from test_public_api_snapshot import NAMESPACES

from neurospatial._pynapple import SAMPLE_ARGUMENTS


def public_functions():
    """Iterate distinct public functions across the snapshot namespaces."""
    seen = set()
    for namespace in NAMESPACES:
        module = importlib.import_module(namespace)
        for name in module.__all__:
            function = getattr(module, name)
            if inspect.isfunction(function) and function not in seen:
                seen.add(function)
                yield function, inspect.signature(function).parameters


def _required_positional(parameters):
    return [
        p.name
        for p in parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        and p.default is p.empty
    ]


# ``compute_*`` takes raw recordings, so it has ``times`` or ``spike_times`` among
# its positional parameters. A function that derives a quantity from other
# arrays is a bare noun (``field_emd``, ``step_lengths``).
COMPUTE_WITHOUT_TIMES = {
    # Takes the position and theta phase of each spike, recorded together: the
    # raw per-spike recording, with no sample clock to pass.
    "compute_phase_precession",
}


def test_compute_prefix_rule():
    raw = {"times", "spike_times"}
    offenders = [
        function.__name__
        for function, parameters in public_functions()
        if function.__name__.startswith("compute_")
        and function.__name__ not in COMPUTE_WITHOUT_TIMES
        and not raw
        & {
            p.name
            for p in parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        }
    ]
    assert not offenders, (
        f"compute_* functions without times or spike_times positionally: {offenders}. "
        "Drop the prefix if they derive a quantity from arrays."
    )


def test_compute_prefix_allowlist_is_current():
    """Each allowlisted function exists and would fail the rule without its entry."""
    raw = {"times", "spike_times"}
    functions = {
        function.__name__: parameters for function, parameters in public_functions()
    }
    assert set(functions) >= COMPUTE_WITHOUT_TIMES
    for name in COMPUTE_WITHOUT_TIMES:
        positional = {
            p.name
            for p in functions[name].values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        }
        assert not raw & positional, f"{name} now takes times; drop its allowlist entry"


def test_times_before_positions():
    offenders = []
    for function, parameters in public_functions():
        required = _required_positional(parameters)
        if (
            "times" in required
            and "positions" in required
            and required.index("times") > required.index("positions")
        ):
            offenders.append(function.__name__)
    assert not offenders, sorted(offenders)


def test_no_spike_trains_parameter_name():
    offenders = [
        (function.__name__, name)
        for function, parameters in public_functions()
        for name in parameters
        if name in {"spike_trains", "trains", "trajectory"}
    ]
    assert not offenders, sorted(offenders)


def test_physical_smoothing_width_is_named_bandwidth():
    """A physical kernel width is ``bandwidth`` on every public function.

    ``neurospatial.ops`` primitives (``diffusion_kernels``,
    ``geodesic_rbf_basis``) are covered along with the analysis functions.
    """
    offenders = [
        f"{function.__module__}.{function.__name__}"
        for function, parameters in public_functions()
        if "sigma" in parameters
    ]
    assert not offenders, sorted(offenders)


def test_env_is_first():
    offenders = []
    for function, parameters in public_functions():
        positional = [
            p.name
            for p in parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        if "env" in positional and positional[0] != "env":
            first = positional[0]
            if not first.startswith("position_bins") and first not in {
                "trials",
                "nwbfile",
            }:
                offenders.append(function.__name__)
    assert not offenders, sorted(offenders)


# Defaulted arguments a public function may take by position: data that an
# analysis can do without or derive, never an option. Every other defaulted
# argument is an option and must be keyword-only, so that a value passed by
# position cannot be bound to the wrong option silently.
FREE_FUNCTION_OPTIONAL_DATA = {
    # Heading samples; needed only when distances are measured in the
    # animal's frame.
    ("boundary_distances", "headings"),
    ("compute_boundary_vector_rate", "headings"),
    ("compute_boundary_vector_rates", "headings"),
    # Reward locations; read from the tracked positions at the reward times
    # when absent.
    ("distance_to_reward", "reward_positions"),
    # Retired argument kept by position so the former call
    # mutual_information(rates, occupancy) raises with migration guidance.
    ("mutual_information", "occupancy"),
    # Per-time-bin weights of the correlation.
    ("decoding_correlation", "weights"),
    # Center bins of the basis functions; selected with n_centers= and
    # center_method= when absent.
    ("chebyshev_filter_basis", "centers"),
    ("geodesic_rbf_basis", "centers"),
    ("heat_kernel_wavelet_basis", "centers"),
    # The components of the transform being built.
    ("from_rotation_matrix", "translation"),
    ("scale_2d", "sx"),
    ("scale_2d", "sy"),
    ("scale_3d", "sx"),
    ("scale_3d", "sy"),
    ("scale_3d", "sz"),
    ("simple_scale", "offset_px"),
    ("translate", "tx"),
    ("translate", "ty"),
    ("translate_3d", "tx"),
    ("translate_3d", "ty"),
    ("translate_3d", "tz"),
    # The trials or other intervals that select samples and their targets.
    ("path_progress", "intervals"),
    ("distance_to_goal", "intervals"),
    ("approach_rate", "intervals"),
    ("angular_efficiency", "intervals"),
    ("compute_goal_directed_metrics", "intervals"),
    ("compute_path_efficiency", "intervals"),
    ("goal_bias", "intervals"),
    ("graph_turn_sequence", "intervals"),
    ("instantaneous_goal_alignment", "intervals"),
    # The control period of a reactivation comparison (rest before the task).
    ("compare_reactivation_methods", "control_epochs"),
    # Covariance of the fitted coefficients, for the Wald test.
    ("circular_basis_metrics", "cov_matrix"),
    # Observations and fits drawn over the tuning curve or scatter.
    ("plot_circular_basis_tuning", "angles"),
    ("plot_circular_basis_tuning", "rates"),
    ("plot_phase_precession", "result"),
    # Values sampled at the times being converted.
    ("to_pynapple", "values"),
    # Records written to the file.
    ("write_region_crossings", "crossings"),
    # The video or region the annotation works on.
    ("annotate_track_graph", "video_path"),
    ("validate_annotations", "boundary"),
    # The public dataset session loaded; None is the dataset's default.
    ("load", "session"),
}


# Maze builders take their dimensions positionally; ``dims=None`` selects the
# published dimensions. Every other argument is an option.
FREE_FUNCTION_OPTIONAL_DATA |= {
    (name, "dims")
    for name in (
        *("make_barnes_maze", "make_cheeseboard_maze", "make_crossword_maze"),
        *("make_hamlet_maze", "make_hampton_court_maze", "make_honeycomb_maze"),
        *("make_linear_track", "make_plus_maze", "make_radial_arm_maze"),
        *("make_rat_hexmaze", "make_repeated_t_maze", "make_repeated_y_maze"),
        *("make_small_hex_maze", "make_square_maze", "make_sungod_maze"),
        *("make_t_maze", "make_w_maze", "make_watermaze", "make_y_maze"),
    )
}


def _defaulted_positional(parameters):
    return [
        p.name
        for p in parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        and p.default is not p.empty
    ]


def test_free_function_options_are_keyword_only():
    """Public functions take data positionally and every option by keyword.

    A positional option with a default would let ``credible_region(env, p, 0.5)``
    or ``convert_to_cm(xy, frame, 0.1)`` bind a value to an option the caller
    did not mean. Only the optional data listed above may be passed by position.
    """
    offenders = [
        f"{function.__name__}({name}=)"
        for function, parameters in public_functions()
        for name in _defaulted_positional(parameters)
        if (function.__name__, name) not in FREE_FUNCTION_OPTIONAL_DATA
    ]
    assert not offenders, sorted(offenders)


def test_free_function_optional_data_list_is_current():
    """Each listed argument still exists and is still a defaulted positional."""
    present = {
        (function.__name__, name)
        for function, parameters in public_functions()
        for name in _defaulted_positional(parameters)
    }
    assert not FREE_FUNCTION_OPTIONAL_DATA - present, sorted(
        FREE_FUNCTION_OPTIONAL_DATA - present
    )


# Names of required arguments that are data wherever they appear, since a public
# name has one meaning everywhere. A required option such as dt, bin_size,
# duration or a threshold is keyword-only even without a default, so a new
# required positional name fails until it is classified here as data.
REQUIRED_DATA_NAMES = {
    # Environments, graphs and recordings, and samples of them.
    *("env", "env_a", "env_b", "source_env", "target_env"),
    *("graph", "nodes", "sources", "bin_from", "bin_to", "centers"),
    *("nwbfile", "session", "obj", "data", "df", "path"),
    *("video_path", "xml_path", "json_path", "mask", "project"),
    *("times", "positions", "position", "points"),
    *("positions_a", "positions_b"),
    *("headings", "heading", "nose", "tail", "speeds", "lfp", "signals"),
    *("spike_times", "spike_counts", "control_counts", "match_counts"),
    *("targets", "ego_points", "object_positions", "cue_positions"),
    *("goal", "goal_bins", "start", "subgoals", "regions"),
    *("angles", "angles_a", "angles_b", "angle", "linear_values", "phases"),
    # Events, intervals, trials and labels.
    *("event_times", "events", "reference_events", "start_events", "stop_events"),
    *("entry_time", "reward_times"),
    *("interval", "intervals", "epochs", "windows", "trials", "trial_labels"),
    *("template_epochs", "match_epochs"),
    *("laps", "visits", "labels", "direction_labels"),
    # The window of each event, by position in the canonical events order
    # f(spike_times, event_times, window, *, dt).
    "window",
    # Fields, rate maps, fitted models and the results computed from them.
    *("field", "fields", "edge_field", "kernel", "basis", "bin_edges"),
    *("firing_rate", "firing_rate_a", "firing_rate_b"),
    *("firing_rates", "firing_rates_a", "firing_rates_b"),
    *("rate_map_a", "rate_map_b", "rates_a", "rates_b", "all_place_fields"),
    *("joint_counts", "autocorr_2d", "rate_map"),
    *("field_bins", "field_bins_a", "field_bins_b"),
    *("encoding_models", "models", "pattern", "posterior", "log_likelihood"),
    *("patterns", "reference_patterns", "activation"),
    *("explained_variance", "reversed_explained_variance"),
    *("templates", "reference_templates"),
    *("source_probs", "decoded_positions", "actual_positions", "actual_bins"),
    *("template_correlations", "match_correlations", "rates"),
    *("template_bias", "event_bias"),
    *("observed", "null_scores", "beta_sin", "beta_cos", "cov_matrix"),
    *("vte_index", "z_head_sweep", "z_speed_inv", "head_sweeps"),
    *("result", "expected_bins", "table", "null_table"),
    # Calibration measurements and the components of a transform.
    *("landmarks_px", "landmarks_cm", "p1_px", "p2_px", "known_length_cm"),
    *("data_px", "data_cm", "src", "dst", "frame_size_px", "frame_height_px"),
    *("px_per_cm", "rotation_matrix"),
}

# Required arguments that are data in these functions only.
REQUIRED_DATA_ARGUMENTS = {
    # The layout looked up or built.
    ("create_layout", "kind"),
    ("find_layout_type", "name"),
    ("get_layout_parameters", "layout_type"),
    # Sizes of the data the function describes.
    ("field_density_map", "n_bins"),
    ("marchenko_pastur_threshold", "n_neurons"),
    ("marchenko_pastur_threshold", "n_time_bins"),
    # The session's mean posterior that each shuffled peak is drawn from.
    ("shuffle_posterior_weighted_circular", "peak_weights"),
    # The variable sampled with the trajectory that the rates are binned by.
    ("compute_tuning_curve", "values"),
    ("compute_tuning_curves", "values"),
    # The geometry the maze helpers build from; ``width`` is keyword-only.
    ("make_buffered_line", "end"),
    ("make_corridor_polygon", "end"),
    ("make_circular_arena", "center"),
    ("make_circular_arena", "radius"),
    ("union_polygons", "polygons"),
    # The public dataset loaded or fetched.
    ("load", "dataset"),
    ("fetch_all", "dataset"),
}


def test_free_function_required_options_are_keyword_only():
    """A required option, such as ``dt`` or ``bin_size``, is keyword-only too.

    ``decode_position(env, counts, rates, 0.025)`` reads as data; naming the
    bin width says what the number is. Required positional arguments must be
    data named above.
    """
    offenders = [
        f"{function.__name__}({name})"
        for function, parameters in public_functions()
        for name in _required_positional(parameters)
        if name not in REQUIRED_DATA_NAMES
        and (function.__name__, name) not in REQUIRED_DATA_ARGUMENTS
    ]
    assert not offenders, sorted(offenders)


def test_free_function_required_data_lists_are_current():
    """Each listed name is still a required positional of a public function."""
    present = {
        (function.__name__, name)
        for function, parameters in public_functions()
        for name in _required_positional(parameters)
    }
    names = {name for _, name in present}
    assert not REQUIRED_DATA_NAMES - names, sorted(REQUIRED_DATA_NAMES - names)
    assert not REQUIRED_DATA_ARGUMENTS - present, sorted(
        REQUIRED_DATA_ARGUMENTS - present
    )


@pytest.mark.parametrize(
    ("name", "data", "option", "value"),
    [
        ("read_environment", 1, "name", "arena"),
        ("write_environment", 2, "name", "arena"),
        ("read_position", 1, "processing_module", "behavior"),
        ("read_head_direction", 1, "processing_module", "behavior"),
        ("read_pose", 1, "pose_estimation_name", "dlc"),
        ("read_events", 1, "table_name", "laps"),
        ("read_intervals", 1, "interval_name", "trials"),
        ("write_events", 2, "name", "rewards"),
        ("write_laps", 2, "description", "laps"),
        ("write_region_crossings", 2, "description", "crossings"),
        ("dataframe_to_events_table", 1, "name", "rewards"),
    ],
)
def test_nwb_option_passed_by_position_names_its_keyword(name, data, option, value):
    """An NWB name or description passed by position gets the corrected call.

    These options were positional before; the error says which keyword the
    value belongs to. It is raised before the file is touched, so a placeholder
    stands in for the NWB file and data.
    """
    from neurospatial.io import nwb

    with pytest.raises(TypeError, match="positional") as caught:
        getattr(nwb, name)(*[object()] * data, value)
    assert f"pass {option}={value!r} by keyword" in str(caught.value)


def test_environment_method_options_are_keyword_only():
    """Environment methods take data positionally and every option by keyword.

    A positional option with a default would let ``env.smooth(rate, 5.0, mode="average")``
    or ``env.distance_between(a, b, None)`` change meaning silently. The
    optional arguments allowed by position are data.
    """
    from neurospatial import Environment

    optional_data = {
        ("rebin", "factor"),
        ("region_membership", "regions"),
        ("transitions", "bins"),
    }
    # Options with no default value: still options, so still keywords.
    required_options = {"bandwidth", "bin_size", "hops", "mode", "radius"}
    offenders = []
    for name in dir(Environment):
        if name.startswith("_"):
            continue
        member = inspect.getattr_static(Environment, name)
        function = getattr(member, "__func__", member)
        if isinstance(member, property) or not callable(function):
            continue
        for parameter in inspect.signature(function).parameters.values():
            if parameter.kind is not parameter.POSITIONAL_OR_KEYWORD:
                continue
            has_default = parameter.default is not parameter.empty
            if (
                has_default and (name, parameter.name) not in optional_data
            ) or parameter.name in required_options:
                offenders.append(f"{name}({parameter.name}=)")
    assert not offenders, sorted(offenders)


def test_environment_coordinate_arguments_are_named_positions():
    """Methods that map coordinates take them as ``positions``, like analyses."""
    from neurospatial import Environment

    for name in ("bin_at", "contains", "nearest_bin", "to_linear", "interpolate"):
        parameters = inspect.signature(getattr(Environment, name)).parameters
        assert "positions" in parameters, name
    off_contract = {
        "points_nd",
        "nd_position",
        "trajectory",
        "xy",
        "timestamps",
        "sigma",
        "use_geodesic",
    }
    offenders = []
    for name in dir(Environment):
        if name.startswith("_"):
            continue
        member = inspect.getattr_static(Environment, name)
        function = getattr(member, "__func__", member)
        if isinstance(member, property) or not callable(function):
            continue
        offenders += [
            (name, parameter)
            for parameter in inspect.signature(function).parameters
            if parameter in off_contract
        ]
    assert not offenders, offenders


def _public_callables():
    """Public functions and the public methods of public classes.

    Class and static methods are included through their underlying function.
    """
    seen = set()
    for namespace in NAMESPACES:
        module = importlib.import_module(namespace)
        for name in module.__all__:
            obj = getattr(module, name)
            members = [obj]
            if inspect.isclass(obj):
                members = []
                for attr in dir(obj):
                    static = inspect.getattr_static(obj, attr)
                    if isinstance(static, (classmethod, staticmethod)):
                        static = static.__func__
                    if not attr.startswith("_") and inspect.isfunction(static):
                        members.append(static)
            for member in members:
                if inspect.isfunction(member) and member not in seen:
                    seen.add(member)
                    yield member


def test_sample_arguments_read_pynapple_objects():
    """Every public callable taking samples reads pynapple objects correctly.

    ``np.asarray(tsd)`` gives a pynapple series' values, not its timestamps,
    and a ``TsdFrame``'s columns may be y, x. Every callable taking a
    timestamp or sample argument is wrapped by ``reads_pynapple_samples``.
    Those that also take ``epochs`` adopt the time support of pynapple
    positions or headings and say so in their ``epochs`` documentation.
    """
    undecorated, undocumented = [], []
    for function in _public_callables():
        parameters = inspect.signature(function).parameters
        if not set(SAMPLE_ARGUMENTS) & set(parameters):
            continue
        name = f"{function.__module__}.{function.__qualname__}"
        if not getattr(function, "__reads_pynapple_samples__", False):
            undecorated.append(name)
        if "epochs" not in parameters or not (
            {"positions", "headings"} & set(parameters)
        ):
            continue
        lines = (inspect.getdoc(function) or "").splitlines()
        starts = [i for i, line in enumerate(lines) if line.startswith("epochs : ")]
        if starts:
            entry = []
            for line in lines[starts[0] + 1 :]:
                if not line.startswith("    "):
                    break
                entry.append(line)
            if "time_support" not in " ".join(entry):
                undocumented.append(name)
    assert not undecorated, sorted(undecorated)
    assert not undocumented, sorted(undocumented)


def test_warnings_from_sample_functions_point_at_the_caller():
    """The pynapple-reading wrapper adds a frame; warnings must skip it."""
    import warnings

    import numpy as np

    from neurospatial.behavior import restrict

    times = np.arange(10.0)
    positions = np.zeros((10, 2))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Sliced samples from two windows lose the gap between them.
        restrict(times, positions, epochs=[(0.0, 3.0), (5.0, 9.0)])
    assert len(caught) == 1, [str(w.message) for w in caught]
    assert {w.filename for w in caught} == {__file__}


# --- One name per concept ----------------------------------------------------
# Each test below enforces one row of the argument vocabulary in
# .claude/rules/public-api.md.


def _public_objects():
    """``(object, label, parameters)`` of public functions, methods, constructors."""
    seen = set()
    for namespace in NAMESPACES:
        module = importlib.import_module(namespace)
        for name in module.__all__:
            obj = getattr(module, name)
            if not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            candidates = [obj]
            if inspect.isclass(obj):
                for attr in dir(obj):
                    static = inspect.getattr_static(obj, attr)
                    if isinstance(static, (classmethod, staticmethod)):
                        static = static.__func__
                    if not attr.startswith("_") and inspect.isfunction(static):
                        candidates.append(static)
            for candidate in candidates:
                if candidate in seen:
                    continue
                seen.add(candidate)
                try:
                    parameters = inspect.signature(candidate).parameters
                except (TypeError, ValueError):
                    continue
                label = (
                    f"{candidate.__module__}.{getattr(candidate, '__qualname__', name)}"
                )
                yield candidate, label, parameters


def _public_signatures():
    """``(label, parameters)`` for public functions, methods and constructors."""
    for _, label, parameters in _public_objects():
        yield label, parameters


def _documented_entry(function, parameter):
    """The Parameters-section text documenting ``parameter`` ('' if absent)."""
    lines = (inspect.getdoc(function) or "").splitlines()
    for start, line in enumerate(lines):
        if line.split(" :")[0].strip() == parameter and " :" in line:
            entry = []
            for following in lines[start + 1 :]:
                if following and not following.startswith("    "):
                    break
                entry.append(following)
            return " ".join(entry)
    return ""


def _public_callables_with_parameters():
    """``(object, parameters)`` for public functions, methods and constructors."""
    for obj, _, parameters in _public_objects():
        yield obj, parameters


def test_randomness_is_rng():
    """Randomness is ``rng``, never ``seed`` or ``<something>_seed``."""
    offenders = [
        f"{label}({name}=)"
        for label, parameters in _public_signatures()
        for name in parameters
        if name == "seed" or name.endswith("_seed")
    ]
    assert not offenders, sorted(offenders)


def test_rng_is_keyword_only_on_functions():
    offenders = [
        f"{function.__module__}.{function.__qualname__}"
        for function, parameters in _public_callables_with_parameters()
        if "rng" in parameters
        and parameters["rng"].kind is not inspect.Parameter.KEYWORD_ONLY
    ]
    assert not offenders, sorted(offenders)


# ``rng`` is read as ``np.random.default_rng(rng)``: ``rng=0`` and a fresh
# ``np.random.default_rng(0)`` give the same result, and a generator advances,
# so two calls with one generator differ. Each public function taking ``rng``
# runs here on a small input. The record of the seed (a ``rng``,
# ``rng_entropy`` or ``rng_seed`` entry) is left out of the comparison: a generator has no
# int to record.
_RECORD_KEYS = frozenset({"rng", "rng_entropy", "rng_seed"})


def _canonical(value, depth=0):
    """A comparable form of ``value``: arrays by bytes (NaN equals NaN)."""
    import dataclasses
    from collections.abc import Iterator, Mapping

    import numpy as np
    import pandas as pd

    from neurospatial import Environment

    assert depth < 40, f"nesting too deep at {type(value).__name__}"
    deeper = depth + 1
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return value
    if isinstance(value, np.generic):
        return _canonical(value.item(), deeper)
    if isinstance(value, float):
        return ("float", np.float64(value).tobytes())
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            items = tuple(_canonical(item, deeper) for item in value.ravel())
            return ("objects", value.shape, items)
        return ("array", value.shape, value.dtype.str, value.tobytes())
    if isinstance(value, Environment):
        return ("env", value.bin_centers.tobytes())
    if isinstance(value, pd.DataFrame):
        columns = tuple(
            (str(name), _canonical(value[name].to_numpy(dtype=object), deeper))
            for name in value.columns
        )
        return ("frame", _canonical(value.index.to_numpy(dtype=object)), columns)
    if isinstance(value, Mapping):
        items = (
            (repr(key), _canonical(item, deeper))
            for key, item in value.items()
            if key not in _RECORD_KEYS
        )
        return ("mapping", tuple(sorted(items)))
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(item, deeper) for item in value)
    if dataclasses.is_dataclass(value):
        fields = (
            (field.name, _canonical(getattr(value, field.name), deeper))
            for field in dataclasses.fields(value)
            if field.name not in _RECORD_KEYS
        )
        return (type(value).__name__, tuple(fields))
    assert not isinstance(value, Iterator), "materialize lazy draws with list()"
    if callable(value):
        return ("callable", getattr(value, "__qualname__", repr(value)))
    if hasattr(value, "__dict__"):
        return (type(value).__name__, _canonical(vars(value), deeper))
    return (type(value).__name__, repr(value))


def _rng_inputs():
    """Small shared inputs for the ``rng`` calls, built once."""
    from types import SimpleNamespace

    import numpy as np
    from shapely.geometry import box

    from neurospatial import Environment
    from neurospatial.decoding import DecodingResult

    if hasattr(_rng_inputs, "cached"):
        return _rng_inputs.cached
    draw = np.random.default_rng(7)
    times = np.arange(0, 60, 0.1)
    track = Environment.linear_track(
        endpoints=[[0.0, 0.0], [120.0, 0.0]], bin_size=10.0, units="cm"
    )
    x = track.bin_centers[:, 0]
    rates = 20.0 * np.exp(-0.5 * ((x - np.linspace(10, 110, 8)[:, None]) / 8.0) ** 2)
    sweep = np.eye(track.n_bins)[np.arange(10)]
    counts = draw.poisson(1.0, (200, 6)).astype(float)
    counts[:, [1, 3, 4]] += 3 * draw.poisson(0.3, (200, 1))
    inputs = SimpleNamespace(
        arena=Environment.from_polygon(box(0, 0, 100, 100), bin_size=10.0, units="cm"),
        times=times,
        positions=np.c_[50 + 20 * np.sin(times / 3), 50 + 20 * np.cos(times / 4)],
        headings=np.sin(times / 5),
        objects=np.array([[50.0, 50.0]]),
        trains=[np.sort(draw.uniform(0, 59.9, n)) for n in (55, 65, 75)],
        track=track,
        rates=rates,
        sweep=sweep,
        sweep_times=np.arange(10) * 0.02,
        events=[
            DecodingResult(
                posterior=np.eye(track.n_bins)[start + np.arange(6)],
                env=track,
                times=np.arange(6) * 0.02,
            )
            for start in (0, 2, 4, 6)
        ],
        ordered=[np.array([1.0 + 0.01 * i]) for i in range(8)],
        counts=counts,
        session_trains=[np.sort(draw.uniform(0, 300, 300)) for _ in range(6)],
        bias=np.sign(np.subtract.outer(np.arange(8), np.arange(8))).T * 1.0,
        rest_trains=[np.sort(draw.uniform(0, 3600, 2000)) for _ in range(6)],
    )
    _rng_inputs.cached = inputs
    return inputs


def _rng_calls():
    """``{name: call}``: each call runs one public function with ``rng``."""
    import dataclasses

    import numpy as np

    from neurospatial import animation, encoding, ops, reactivation, replay, stats
    from neurospatial import simulation as sim
    from neurospatial.replay import methods

    i = _rng_inputs()
    shuffle = {"n_shuffles": 9}
    place = (i.arena, i.trains[0], i.times, i.positions)
    view = (*place, i.headings)
    ovc = (*place, i.objects)
    ego = (*view, i.objects)
    hd = (i.trains[0], i.times, i.headings)
    counts = i.counts.astype(np.int64)
    windows = np.array([[0.0, 60.0]])
    rest_epochs = {
        "template_epochs": [[1200.0, 2400.0]],
        "match_epochs": [[2400.0, 3600.0]],
        "control_epochs": [[0.0, 1200.0]],
    }
    lansink = dataclasses.replace(reactivation.LANSINK_2009, n_shuffles=20)
    nadasdy = dataclasses.replace(
        methods.NADASDY_1999, surrogates=("across_trains",), n_surrogates=5
    )
    # Phases that precess half a cycle across the field, inside the slope bounds.
    phases = (np.pi - np.pi * np.linspace(0, 1, 40)) % (2 * np.pi)
    davidson = dataclasses.replace(
        methods.DAVIDSON_2009, nulls=("column_cycle",), n_shuffles=9
    )
    place_models = [
        sim.PlaceCellModel(i.arena, center=center, width=10.0)
        for center in ([30.0, 30.0], [70.0, 70.0])
    ]
    return {
        "neurospatial.animation.estimate_colormap_range_from_subset": lambda rng: (
            animation.estimate_colormap_range_from_subset(
                i.counts, n_samples=20, rng=rng
            )
        ),
        "neurospatial.encoding.compute_phase_precession": lambda rng: (
            encoding.compute_phase_precession(
                np.linspace(0, 1, 40), phases, **shuffle, rng=rng
            )
        ),
        "neurospatial.encoding.has_phase_precession": lambda rng: (
            encoding.has_phase_precession(
                np.linspace(0, 1, 40), phases, **shuffle, rng=rng
            )
        ),
        "neurospatial.encoding.place_cell_significance": lambda rng: (
            encoding.place_cell_significance(
                i.arena,
                i.trains,
                i.times,
                i.positions,
                method="binned",
                **shuffle,
                rng=rng,
            )
        ),
        "neurospatial.encoding.is_place_cell": lambda rng: encoding.is_place_cell(
            *place, method="binned", **shuffle, rng=rng
        ),
        "neurospatial.encoding.spatial_cell_type_significance": lambda rng: (
            encoding.spatial_cell_type_significance(
                *place, method="binned", **shuffle, rng=rng
            )
        ),
        "neurospatial.encoding.head_direction_cell_significance": lambda rng: (
            encoding.head_direction_cell_significance(*hd, **shuffle, rng=rng)
        ),
        "neurospatial.encoding.is_head_direction_cell": lambda rng: (
            encoding.is_head_direction_cell(*hd, **shuffle, rng=rng)
        ),
        "neurospatial.encoding.spatial_view_cell_significance": lambda rng: (
            encoding.spatial_view_cell_significance(
                *view, method="binned", **shuffle, rng=rng
            )
        ),
        "neurospatial.encoding.is_spatial_view_cell": lambda rng: (
            encoding.is_spatial_view_cell(*view, method="binned", **shuffle, rng=rng)
        ),
        "neurospatial.encoding.object_vector_cell_significance": lambda rng: (
            encoding.object_vector_cell_significance(*ovc, **shuffle, rng=rng)
        ),
        "neurospatial.encoding.is_object_vector_cell": lambda rng: (
            encoding.is_object_vector_cell(*ovc, **shuffle, rng=rng)
        ),
        "neurospatial.encoding.egocentric_object_vector_cell_significance": (
            lambda rng: encoding.egocentric_object_vector_cell_significance(
                *ego, **shuffle, rng=rng
            )
        ),
        "neurospatial.encoding.is_egocentric_object_vector_cell": lambda rng: (
            encoding.is_egocentric_object_vector_cell(*ego, **shuffle, rng=rng)
        ),
        "neurospatial.ops.select_basis_centers": lambda rng: ops.select_basis_centers(
            i.arena, n_centers=5, rng=rng
        ),
        "neurospatial.ops.geodesic_rbf_basis": lambda rng: ops.geodesic_rbf_basis(
            i.arena, n_centers=5, rng=rng
        ),
        "neurospatial.ops.chebyshev_filter_basis": lambda rng: (
            ops.chebyshev_filter_basis(i.arena, n_centers=5, max_degree=2, rng=rng)
        ),
        "neurospatial.ops.heat_kernel_wavelet_basis": lambda rng: (
            ops.heat_kernel_wavelet_basis(i.arena, n_centers=5, scales=(1.0,), rng=rng)
        ),
        "neurospatial.ops.spatial_basis": lambda rng: ops.spatial_basis(
            i.arena, n_features=10, rng=rng
        ),
        "neurospatial.reactivation.compare_reactivation_methods": lambda rng: (
            reactivation.compare_reactivation_methods(
                i.rest_trains,
                **rest_epochs,
                reactivation_methods=[lansink],
                unit_groups={"region": np.repeat([0, 1], 3)},
                rng=rng,
            )
        ),
        "neurospatial.reactivation.detect_assemblies": lambda rng: (
            reactivation.detect_assemblies(
                i.counts,
                component_threshold="circular_shift",
                percentile=95.0,
                n_shuffles=9,
                rng=rng,
            )
        ),
        "neurospatial.reactivation.equalize_firing_rates": lambda rng: (
            reactivation.equalize_firing_rates(i.counts[:100], i.counts[100:], rng=rng)
        ),
        "neurospatial.reactivation.null_reactivation_data": lambda rng: (
            reactivation.null_reactivation_data(
                i.session_trains, [[200.0, 300.0]], kind="cell_identity", rng=rng
            )
        ),
        "neurospatial.reactivation.simulated_reactivation_session": lambda rng: (
            reactivation.simulated_reactivation_session(
                n_units=6, n_assemblies=1, assembly_size=3, duration=60.0, rng=rng
            )
        ),
        "neurospatial.replay.best_sequence_match": lambda rng: (
            replay.best_sequence_match(
                i.track,
                i.ordered,
                (1.0, 1.1),
                i.rates,
                exact_max_letters=2,
                n_samples=50,
                rng=rng,
            )
        ),
        "neurospatial.replay.bias_similarity_significance": lambda rng: (
            replay.bias_similarity_significance(
                i.ordered, (1.0, 1.1), i.bias, **shuffle, rng=rng
            )
        ),
        "neurospatial.replay.compare_replay_methods": lambda rng: (
            replay.compare_replay_methods(
                i.track,
                i.events,
                replay_methods=[davidson],
                rng=rng,
            )
        ),
        "neurospatial.replay.dataset_values": lambda rng: replay.dataset_values(
            i.track,
            i.events,
            replay_method=methods.SILVA_2015,
            null="column_cycle",
            n_datasets=3,
            rng=rng,
        ),
        "neurospatial.replay.detect_bias_sequence_replay": lambda rng: (
            replay.detect_bias_sequence_replay(
                i.ordered,
                (1.0, 1.1),
                i.bias,
                bias_sequence_method=dataclasses.replace(
                    methods.LIU_2023_BIAS, n_shuffles=9
                ),
                rng=rng,
            )
        ),
        "neurospatial.replay.detect_pairwise_order_replay": lambda rng: (
            replay.detect_pairwise_order_replay(
                i.track,
                i.ordered,
                (1.0, 1.1),
                i.rates,
                pairwise_order_method=dataclasses.replace(
                    methods.GUPTA_2010, n_shuffles=9
                ),
                rng=rng,
            )
        ),
        "neurospatial.replay.detect_rank_order_replay": lambda rng: (
            replay.detect_rank_order_replay(
                i.track,
                i.ordered,
                (1.0, 1.1),
                i.rates,
                rank_order_method=dataclasses.replace(
                    methods.DIBA_BUZSAKI_2007, n_shuffles=9
                ),
                rng=rng,
            )
        ),
        "neurospatial.replay.detect_replay": lambda rng: replay.detect_replay(
            i.track,
            i.sweep,
            i.sweep_times,
            replay_method=davidson,
            rng=rng,
        ),
        "neurospatial.replay.detect_template_match_replay": lambda rng: (
            replay.detect_template_match_replay(
                i.track,
                i.ordered,
                (1.0, 1.1),
                i.rates,
                template_match_method=dataclasses.replace(
                    methods.JI_WILSON_2007, n_shuffles=9
                ),
                rng=rng,
            )
        ),
        "neurospatial.replay.detect_trajectory_radon": lambda rng: (
            replay.detect_trajectory_radon(
                i.track, i.sweep, i.sweep_times, n_lines=50, rng=rng
            )
        ),
        "neurospatial.replay.fit_linear_trajectory": lambda rng: (
            replay.fit_linear_trajectory(
                i.track, i.sweep, i.sweep_times, n_samples_per_time_bin=20, rng=rng
            )
        ),
        "neurospatial.replay.null_events": lambda rng: replay.null_events(
            i.ordered, np.array([[1.0, 1.1]]), kind="spike_time_uniform", rng=rng
        ),
        "neurospatial.replay.pairwise_order_significance": lambda rng: (
            replay.pairwise_order_significance(
                i.track, i.ordered, (1.0, 1.1), i.rates, **shuffle, rng=rng
            )
        ),
        "neurospatial.replay.rank_order_significance": lambda rng: (
            replay.rank_order_significance(
                i.track, i.ordered, (1.0, 1.1), i.rates, **shuffle, rng=rng
            )
        ),
        "neurospatial.replay.recurring_sequences_significance": lambda rng: (
            replay.recurring_sequences_significance(
                i.session_trains[:3],
                recurring_sequence_method=nadasdy,
                reference_unit=0,
                rng=rng,
            )
        ),
        "neurospatial.replay.replay_significance": lambda rng: (
            replay.replay_significance(
                i.track,
                i.sweep,
                i.sweep_times,
                nulls=("column_cycle",),
                **shuffle,
                rng=rng,
            )
        ),
        "neurospatial.replay.shuffle_cell_identity": lambda rng: list(
            replay.shuffle_cell_identity(counts, i.counts.T, n_shuffles=3, rng=rng)
        ),
        "neurospatial.replay.shuffle_events": lambda rng: replay.shuffle_events(
            i.track, i.events, null="time_bin_permutation", rng=rng
        ),
        "neurospatial.replay.shuffle_place_fields_circular": lambda rng: list(
            replay.shuffle_place_fields_circular(
                i.track, i.rates, n_shuffles=3, rng=rng
            )
        ),
        "neurospatial.replay.shuffle_place_fields_circular_2d": lambda rng: list(
            replay.shuffle_place_fields_circular_2d(
                i.arena,
                np.ones((2, i.arena.n_bins)).cumsum(1),
                n_shuffles=3,
                rng=rng,
            )
        ),
        "neurospatial.replay.shuffle_posterior_circular": lambda rng: list(
            replay.shuffle_posterior_circular(i.track, i.sweep, n_shuffles=3, rng=rng)
        ),
        "neurospatial.replay.shuffle_posterior_weighted_circular": lambda rng: list(
            replay.shuffle_posterior_weighted_circular(
                i.track,
                i.sweep,
                np.ones(i.track.n_bins),
                n_shuffles=3,
                rng=rng,
            )
        ),
        "neurospatial.replay.shuffle_spike_counts_circular": lambda rng: list(
            replay.shuffle_spike_counts_circular(counts, n_shuffles=3, rng=rng)
        ),
        "neurospatial.replay.shuffle_spike_counts_uniform": lambda rng: list(
            replay.shuffle_spike_counts_uniform(counts, n_shuffles=3, rng=rng)
        ),
        "neurospatial.replay.simulated_null_events": lambda rng: (
            replay.simulated_null_events(i.track, i.rates, n_events=2, rng=rng)
        ),
        "neurospatial.replay.surrogate_spike_trains": lambda rng: (
            replay.surrogate_spike_trains(
                i.session_trains[:2], kind="isi_shuffle", rng=rng
            )
        ),
        "neurospatial.replay.template_correlation_test": lambda rng: (
            replay.template_correlation_test(
                [np.arange(8.0)], [np.arange(8.0)[::-1]], **shuffle, rng=rng
            )
        ),
        "neurospatial.replay.template_match_dataset_test": lambda rng: (
            replay.template_match_dataset_test(
                i.track,
                [np.arange(3) * 2.0 + 0.05 * k for k in range(8)],
                np.column_stack([np.arange(3) * 2.0, np.arange(3) * 2.0 + 0.5]),
                i.rates,
                template_match_method=dataclasses.replace(
                    methods.JI_WILSON_2007, n_shuffles=9
                ),
                rng=rng,
            )
        ),
        "neurospatial.simulation.add_modulation": lambda rng: sim.add_modulation(
            i.trains[0], modulation_freq=8.0, rng=rng
        ),
        "neurospatial.simulation.generate_poisson_spikes": lambda rng: (
            sim.generate_poisson_spikes(np.full(i.times.size, 5.0), i.times, rng=rng)
        ),
        "neurospatial.simulation.generate_population_spikes": lambda rng: (
            sim.generate_population_spikes(
                place_models, i.times, i.positions, show_progress=False, rng=rng
            )
        ),
        "neurospatial.simulation.simulate_trajectory_ou": lambda rng: (
            sim.simulate_trajectory_ou(i.arena, duration=2.0, speed_units="cm", rng=rng)
        ),
        "neurospatial.simulation.simulate_trajectory_laps": lambda rng: (
            sim.simulate_trajectory_laps(
                i.track, n_laps=1, sampling_frequency=20.0, rng=rng
            )
        ),
        "neurospatial.simulation.simulate_session": lambda rng: sim.simulate_session(
            i.arena, duration=5.0, n_cells=2, show_progress=False, rng=rng
        ),
        "neurospatial.simulation.simulate_replay_event": lambda rng: (
            sim.simulate_replay_event(i.track, i.rates, speed=200.0, rng=rng)
        ),
        "neurospatial.simulation.open_field_session": lambda rng: (
            sim.open_field_session(duration=5.0, n_place_cells=2, rng=rng)
        ),
        "neurospatial.simulation.linear_track_session": lambda rng: (
            sim.linear_track_session(
                duration=10.0, track_length=50.0, n_place_cells=2, n_laps=1, rng=rng
            )
        ),
        "neurospatial.simulation.grid_cell_session": lambda rng: (
            sim.grid_cell_session(
                duration=5.0, arena_size=60.0, n_grid_cells=2, rng=rng
            )
        ),
        "neurospatial.simulation.border_cell_session": lambda rng: (
            sim.border_cell_session(
                duration=5.0, n_border_cells=2, n_place_cells=1, rng=rng
            )
        ),
        "neurospatial.simulation.tmaze_alternation_session": lambda rng: (
            sim.tmaze_alternation_session(
                duration=20.0, n_trials=2, n_place_cells=2, rng=rng
            )
        ),
        "neurospatial.stats.generate_inhomogeneous_poisson_surrogates": lambda rng: (
            list(
                stats.generate_inhomogeneous_poisson_surrogates(
                    counts, n_surrogates=3, rng=rng
                )
            )
        ),
        "neurospatial.stats.generate_jittered_spikes": lambda rng: list(
            stats.generate_jittered_spikes(
                i.trains[0], jitter_std=0.01, n_surrogates=3, rng=rng
            )
        ),
        "neurospatial.stats.generate_poisson_surrogates": lambda rng: list(
            stats.generate_poisson_surrogates(counts, n_surrogates=3, rng=rng)
        ),
        "neurospatial.stats.shuffle_spike_times_circular": lambda rng: list(
            stats.shuffle_spike_times_circular(
                i.trains[0], windows, n_shuffles=3, rng=rng
            )
        ),
        "neurospatial.stats.shuffle_spikes_isi": lambda rng: list(
            stats.shuffle_spikes_isi(i.trains[0], n_shuffles=3, rng=rng)
        ),
        "neurospatial.stats.shuffle_time_bins": lambda rng: list(
            stats.shuffle_time_bins(counts, n_shuffles=3, rng=rng)
        ),
        "neurospatial.stats.shuffle_time_bins_coherent": lambda rng: list(
            stats.shuffle_time_bins_coherent(counts, n_shuffles=3, rng=rng)
        ),
        "neurospatial.stats.shuffle_trials": lambda rng: list(
            stats.shuffle_trials(np.repeat(np.arange(4), 5), n_shuffles=3, rng=rng)
        ),
    }


def _public_rng_functions():
    """Public function paths whose snapshot signature takes ``rng=``."""
    from pathlib import Path

    snapshot = Path(__file__).parent / "data" / "public_api.txt"
    return sorted(
        line.split("(", 1)[0]
        for line in snapshot.read_text().splitlines()
        if re.search(r"[(,*] ?rng=", line)
    )


def test_rng_calls_cover_every_public_rng_function():
    assert sorted(_rng_calls()) == _public_rng_functions()


@pytest.mark.parametrize("name", _public_rng_functions())
def test_seed_and_its_fresh_generator_agree(name):
    """``rng=0`` and ``rng=np.random.default_rng(0)`` give the same result."""
    import numpy as np

    call = _rng_calls()[name]
    seeded, generated = call(0), call(np.random.default_rng(0))
    assert _canonical(seeded) == _canonical(generated), name


@pytest.mark.parametrize(
    "name",
    [
        "neurospatial.encoding.place_cell_significance",
        "neurospatial.replay.shuffle_events",
        "neurospatial.replay.replay_significance",
        "neurospatial.reactivation.compare_reactivation_methods",
        "neurospatial.reactivation.detect_assemblies",
        "neurospatial.simulation.simulate_session",
        "neurospatial.simulation.generate_population_spikes",
        "neurospatial.stats.shuffle_spike_times_circular",
    ],
)
def test_one_generator_advances_between_calls(name):
    """Two calls with one generator draw different values."""
    import numpy as np

    call = _rng_calls()[name]
    generator = np.random.default_rng(0)
    assert _canonical(call(generator)) != _canonical(call(generator)), name


# Functions whose streams derive from one entropy drawn from ``rng``
# (``neurospatial._random._root``): a generator's state, not its seed, counts.
_ENTROPY_FUNCTIONS = [
    "neurospatial.encoding.place_cell_significance",
    "neurospatial.replay.shuffle_events",
    "neurospatial.replay.detect_pairwise_order_replay",
    "neurospatial.replay.compare_replay_methods",
    "neurospatial.replay.dataset_values",
    "neurospatial.replay.recurring_sequences_significance",
    "neurospatial.reactivation.compare_reactivation_methods",
]


@pytest.mark.parametrize("name", _ENTROPY_FUNCTIONS)
def test_a_generator_restored_to_a_state_repeats_that_state(name):
    """A generator set to ``default_rng(0)``'s state gives ``rng=0``'s result."""
    import numpy as np

    call = _rng_calls()[name]
    restored = np.random.default_rng(123)
    restored.bit_generator.state = np.random.default_rng(0).bit_generator.state
    expected = _canonical(call(np.random.default_rng(0)))
    assert _canonical(call(restored)) == expected, name


@pytest.mark.parametrize("name", _ENTROPY_FUNCTIONS)
def test_an_advanced_generator_differs_from_its_seed(name):
    """``default_rng(0)`` after 1000 draws is not ``rng=0``."""
    import numpy as np

    call = _rng_calls()[name]
    advanced = np.random.default_rng(0)
    advanced.random(1000)
    assert _canonical(call(advanced)) != _canonical(call(0)), name


@pytest.mark.parametrize(
    "name",
    [
        "neurospatial.replay.compare_replay_methods",
        "neurospatial.replay.dataset_values",
        "neurospatial.reactivation.compare_reactivation_methods",
    ],
)
def test_rng_entropy_of_a_generator_determines_the_table(name, monkeypatch):
    """``rng_entropy`` is the generator's first draw and fixes every stream.

    Fed back through the entropy step every stream derives from, the
    recorded int gives the same table, whatever ``rng`` is then passed.
    """
    import numpy as np

    from neurospatial import _random

    call = _rng_calls()[name]
    table = call(np.random.default_rng(5))
    entropy = table.attrs["rng_entropy"]
    assert type(entropy) is int
    assert entropy == int(np.random.default_rng(5).integers(2**63))
    monkeypatch.setattr(_random, "_entropy", lambda rng: entropy)
    assert _canonical(call(None)) == _canonical(table), name


@pytest.mark.parametrize(
    "name",
    [
        "neurospatial.replay.compare_replay_methods",
        "neurospatial.replay.dataset_values",
        "neurospatial.reactivation.compare_reactivation_methods",
    ],
)
def test_rng_seed_repeats_a_run_without_a_seed(name):
    """``rng_seed`` repeats an ``rng=None`` run and an int run; None for a generator."""
    import numpy as np

    call = _rng_calls()[name]
    unseeded = call(None)
    seed = unseeded.attrs["rng_seed"]
    assert type(seed) is int
    assert _canonical(call(seed)) == _canonical(unseeded), name
    assert call(7).attrs["rng_seed"] == 7
    assert call(np.random.default_rng(7)).attrs["rng_seed"] is None


def test_one_generator_advances_a_shuffle_verdict():
    """Repeated shuffle verdicts with one generator draw new shuffles.

    A verdict is one bool, so two calls can agree by chance. This cell has
    no field, and its p-value from 9 shuffles is 0.1 or 0.2, either side
    of ``alpha=0.15``, depending on the draw (``min_info=0`` leaves the verdict
    to the shuffle test), so verdicts from one generator vary.
    """
    import numpy as np

    from neurospatial import encoding

    i = _rng_inputs()
    generator = np.random.default_rng(0)
    verdicts = {
        encoding.is_place_cell(
            i.arena,
            i.trains[0],
            i.times,
            i.positions,
            method="binned",
            min_info=0.0,
            alpha=0.15,
            n_shuffles=9,
            rng=generator,
        )
        for _ in range(8)
    }
    assert verdicts == {True, False}


def test_single_region_is_region():
    """One region is ``region=``; two roles in one call carry role prefixes."""
    offenders = [
        f"{label}({name}=)"
        for label, parameters in _public_signatures()
        for name in parameters
        if name in {"region_name", "goal_region"}
    ]
    assert not offenders, sorted(offenders)


def test_time_bin_width_is_dt():
    """A bin width in seconds is ``dt``; ``bin_size`` is spatial or angular."""
    in_seconds = re.compile(r"\bseconds?\b|\bin s\b|\s\(s\)", re.IGNORECASE)
    offenders = [
        f"{function.__module__}.{function.__qualname__}"
        for function, parameters in _public_callables_with_parameters()
        if "bin_size" in parameters
        and in_seconds.search(_documented_entry(function, "bin_size"))
    ]
    assert not offenders, sorted(offenders)


def test_pre_entry_window_is_window_duration():
    offenders = [
        f"{label}({name}=)"
        for label, parameters in _public_signatures()
        for name in parameters
        if name == "pre_window"
    ]
    assert not offenders, sorted(offenders)


def test_paired_inputs_use_a_b():
    """Two inputs of one role are suffixed ``_a`` / ``_b``."""
    pair = re.compile(r"^(?P<stem>.+?)(?:_1|_2|_i|_j|1|2)$")
    offenders = []
    for label, parameters in _public_signatures():
        names = set(parameters)
        for name in names:
            match = pair.match(name)
            if not match:
                continue
            stem = match["stem"]
            partner = {
                "_1": "_2",
                "_2": "_1",
                "_i": "_j",
                "_j": "_i",
                "1": "2",
                "2": "1",
            }[name[len(stem) :]]
            if stem + partner in names:
                offenders.append(f"{label}({name}=)")
    assert not offenders, sorted(offenders)


def test_unit_selection_is_unit_ids():
    offenders = [
        f"{label}({name}=)"
        for label, parameters in _public_signatures()
        for name in parameters
        # ``cells`` of ``ClippedLocator`` are polygon grid cells, not units.
        if name == "cells" and not label.endswith("ClippedLocator")
    ]
    assert not offenders, sorted(offenders)


def test_smoothing_widths_say_their_unit():
    """``smooth_window`` is seconds; a width counted in bins is ``smooth_bins``."""
    in_bins = re.compile(r"\bbins?\b", re.IGNORECASE)
    in_seconds = re.compile(r"\bseconds?\b|\s\(s\)", re.IGNORECASE)
    offenders = []
    for function, parameters in _public_callables_with_parameters():
        name = f"{function.__module__}.{function.__qualname__}"
        if "smoothing_window" in parameters:
            offenders.append(f"{name}(smoothing_window=)")
        if "smooth_window" in parameters:
            entry = _documented_entry(function, "smooth_window")
            if in_bins.search(entry) and not in_seconds.search(entry):
                offenders.append(f"{name}(smooth_window= counts bins)")
        if "smooth_bins" in parameters:
            entry = _documented_entry(function, "smooth_bins")
            if in_seconds.search(entry) and not in_bins.search(entry):
                offenders.append(f"{name}(smooth_bins= is in seconds)")
    assert not offenders, sorted(offenders)


# Functions that take ``alpha``: each is a significance level.
SIGNIFICANCE_ALPHA = {
    "DirectionalRateResult.is_head_direction_cell",
    "DirectionalRateResult.interpretation",
    "DirectionalRatesResult.classify",
    "is_head_direction_cell",
    "is_egocentric_object_vector_cell",
    "is_object_vector_cell",
    "PhasePrecessionResult.is_significant",
    "has_phase_precession",
    "CellTypeSignificanceResult.label_cell_types",
    "CellTypeSignificanceResult.summary_table",
    "PlaceCellSignificanceResult.classify",
    "PlaceCellSignificanceResult.summary_table",
    "is_place_cell",
    "is_spatial_view_cell",
    "apply_alpha",
    "is_modulated",
    "ShuffleTestResult.is_significant",
    # Method and recipe records: the level a p-value must be below.
    "BiasSequenceMethod",
    "PairwiseOrderMethod",
    "RankOrderMethod",
    "ReplayMethod",
    "RecurringSequenceMethod",
    "TemplateMatchMethod",
    "RecurringSequencesSignificanceResult",
    "DatasetTestRecipe",
    "ReactivationMethod",
}

# ``alpha`` that is not a significance level, each with its meaning.
OTHER_ALPHA = {
    # Matplotlib opacity of the drawn layer.
    "EpochOverlay": "matplotlib opacity",
    "TimeSeriesOverlay": "matplotlib opacity",
    "VideoOverlay": "matplotlib opacity",
    # The alpha-shape parameter of computational geometry.
    "BoundaryConfig": "alpha-shape parameter (computational-geometry term)",
}


def test_alpha_is_significance_level():
    """``alpha`` is a significance level only; other weights get their own name."""
    offenders = [
        f"{function.__module__}.{function.__qualname__}"
        for function, parameters in _public_callables_with_parameters()
        if "alpha" in parameters
        and function.__qualname__ not in SIGNIFICANCE_ALPHA | OTHER_ALPHA.keys()
    ]
    assert not offenders, sorted(offenders)


def test_ax_is_keyword_only():
    """The Matplotlib target is ``ax=``, keyword-only, and never ``axes=``."""
    offenders = []
    for label, parameters in _public_signatures():
        if "axes" in parameters:
            offenders.append(f"{label}(axes=)")
        if (
            "ax" in parameters
            and parameters["ax"].kind is not inspect.Parameter.KEYWORD_ONLY
        ):
            offenders.append(f"{label}(ax=)")
    assert not offenders, sorted(offenders)


def test_graph_argument_name():
    offenders = [
        f"{label}({name}=)"
        for label, parameters in _public_signatures()
        for name in parameters
        if name == "G"
    ]
    assert not offenders, sorted(offenders)


def test_sample_times_are_times():
    offenders = [
        f"{label}({name}=)"
        for label, parameters in _public_signatures()
        for name in parameters
        if name == "t"
    ]
    assert not offenders, sorted(offenders)


def test_field_resampling_takes_environments_first():
    """``resample_field`` orders its arguments like ``map_probabilities``."""
    from neurospatial.ops import map_probabilities, resample_field

    resample = list(inspect.signature(resample_field).parameters)
    assert resample[:3] == ["source_env", "target_env", "field"]
    assert list(inspect.signature(map_probabilities).parameters)[:2] == [
        "source_env",
        "target_env",
    ]


# --- Defaults ----------------------------------------------------------------
# A default that always resolves to one value is written in the signature, so
# ``help()`` and the API reference show it. ``None`` is kept only where the
# value depends on the data, the environment or another argument.


def test_no_constant_none_sentinels():
    """Thresholds, levels and shuffle counts never hide a constant behind None."""
    offenders = [
        f"{label}({name}=None)"
        for label, parameters in _public_signatures()
        for name, parameter in parameters.items()
        if name in {"min_info", "alpha", "n_shuffles", "min_shift", "min_mvl"}
        and parameter.default is None
    ]
    assert not offenders, sorted(offenders)


# Lower shuffle counts kept on purpose, each with its documented reason.
_N_SHUFFLES_EXCEPTIONS: dict[str, str] = {}


def test_n_shuffles_defaults_agree():
    """Every shuffle test defaults to 1000 shuffles unless documented otherwise."""
    defaults = {
        label: parameters["n_shuffles"].default
        for label, parameters in _public_signatures()
        if "n_shuffles" in parameters
        and parameters["n_shuffles"].default is not inspect.Parameter.empty
    }
    assert len(defaults) >= 20
    offenders = {
        label: value
        for label, value in defaults.items()
        if value != 1000 and label not in _N_SHUFFLES_EXCEPTIONS
    }
    assert not offenders, offenders


# These take positions or speeds but no environment, so they cannot know the
# length unit; a numeric default (5 cm/s, the cited stop threshold) would be
# wrong for positions in meters or pixels. Their docs say why it is required.
_UNITLESS_MIN_SPEED = {
    "neurospatial.behavior.decisions.compute_pre_decision_metrics",
    "neurospatial.behavior.decisions.pre_decision_heading_stats",
    "neurospatial.behavior.navigation.heading_direction_labels",
    "neurospatial.behavior.navigation.headings_to_direction_labels",
    "neurospatial.behavior.segmentation.segment_by_velocity",
    "neurospatial.behavior.vte.compute_vte_trial",
    "neurospatial.behavior.vte.head_sweep_from_positions",
}


def test_min_speed_has_default():
    """``min_speed`` is never required where the unit of speed is known."""
    required = {
        label
        for label, parameters in _public_signatures()
        if "min_speed" in parameters
        and parameters["min_speed"].default is inspect.Parameter.empty
    }
    assert required == _UNITLESS_MIN_SPEED
    for label in _UNITLESS_MIN_SPEED:
        module, name = label.rsplit(".", 1)
        function = getattr(importlib.import_module(module), name)
        assert "Required" in _documented_entry(function, "min_speed"), label
