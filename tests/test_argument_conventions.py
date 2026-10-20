"""Public function signatures follow the shared input conventions."""

import importlib
import inspect

import pytest
from test_public_api_snapshot import NAMESPACES


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


def test_times_before_positions():
    offenders = []
    for function, parameters in public_functions():
        required = [
            p.name
            for p in parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            and p.default is p.empty
        ]
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
    # The trajectory, which the caller can replace by speed= and heading=.
    ("heading_direction_labels", "times"),
    ("heading_direction_labels", "positions"),
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
    # Covariance of the fitted coefficients, for the Wald test.
    ("circular_basis_metrics", "cov_matrix"),
    # Observations and fits drawn over the tuning curve or scatter.
    ("plot_circular_basis_tuning", "angles"),
    ("plot_circular_basis_tuning", "rates"),
    ("plot_phase_precession", "result"),
    # Values sampled at the times being converted.
    ("to_pynapple", "values"),
    # Records written to the file.
    ("write_laps", "lap_types"),
    ("write_trials", "trials"),
    # The video or region the annotation works on.
    ("annotate_track_graph", "video_path"),
    ("validate_annotations", "boundary"),
    # The Axes to draw on, as in result.plot(ax) and Environment.plot(ax).
    ("plot_circular_basis_tuning", "ax"),
    ("plot_head_direction_tuning", "ax"),
    ("plot_object_vector_tuning", "ax"),
    ("plot_phase_precession", "ax"),
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
        ("write_place_field", 3, "name", "cell_1"),
        ("write_occupancy", 3, "name", "occupancy_1"),
        ("write_laps", 3, "description", "laps"),
        ("write_region_crossings", 4, "description", "crossings"),
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
    optional arguments allowed by position are data or the Axes to draw on.
    """
    from neurospatial import Environment

    optional_data = {
        ("plot", "ax"),
        ("plot_1d", "ax"),
        ("plot_field", "ax"),
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
