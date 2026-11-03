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

    ``neurospatial.ops`` primitives (``compute_diffusion_kernels``,
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
    ("compute_boundary_distances", "headings"),
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
