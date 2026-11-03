"""Contract tests for the split terminal verbs.

The v0.6 api-contract requires that the analysis-ending verbs mean ONE thing on
every result class:

- ``to_dataframe()`` -> dense tidy: one row per ``(unit, bin)`` (single-unit:
  one row per ``bin``), ALWAYS carrying a ``unit_id`` column.
- ``summary_table()`` -> one row per unit, ``unit_id``-indexed, scalar columns.

These tests assert the contract holds, and that the column vocabulary is
consistent across the four batch encoding result classes.
"""

from __future__ import annotations

import numpy as np
import pytest
from result_instances import RESULT_FACTORIES, public_result_classes

from neurospatial import Environment
from neurospatial.encoding.directional import DirectionalRatesResult
from neurospatial.encoding.egocentric import compute_egocentric_rates
from neurospatial.encoding.spatial import compute_spatial_rate, compute_spatial_rates
from neurospatial.encoding.tuning import compute_tuning_curves
from neurospatial.encoding.view import compute_view_rates


@pytest.fixture
def spatial_batch():
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 50, (500, 2))
    env = Environment.from_samples(positions, bin_size=5.0)
    times = np.linspace(0, 50, 500)
    spike_times = [np.sort(rng.uniform(0, 50, n)) for n in (30, 40, 20)]
    return compute_spatial_rates(env, spike_times, times, positions, bandwidth=10.0)


@pytest.fixture
def directional_batch():
    rng = np.random.default_rng(0)
    n = 60
    bin_centers = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return DirectionalRatesResult(
        firing_rates=rng.random((3, n)) * 10,
        occupancy=np.ones(n) * 0.5,
        bin_centers=bin_centers,
        bin_size=np.pi / 30,
        bandwidth=None,
    )


@pytest.fixture
def view_batch():
    rng = np.random.default_rng(0)
    env = Environment.from_samples(rng.random((200, 2)) * 50, bin_size=5.0, units="cm")
    times = np.linspace(0, 10, 200)
    trajectory = rng.random((200, 2)) * 50
    headings = rng.uniform(-np.pi, np.pi, 200)
    spike_times = [np.sort(rng.uniform(0, 10, 15)) for _ in range(3)]
    return compute_view_rates(
        env,
        spike_times,
        times,
        trajectory,
        headings,
        gaze_model="fixed_distance",
        view_distance=10.0,
        boundary_source="analysis",
    )


@pytest.fixture
def egocentric_batch():
    rng = np.random.default_rng(0)
    times = np.linspace(0, 100, 1000)
    positions = rng.uniform(10, 90, (1000, 2))
    headings = rng.uniform(-np.pi, np.pi, 1000)
    object_positions = np.array([[50.0, 50.0]])
    spike_times = [np.sort(rng.uniform(0, 100, k)) for k in (100, 150, 50)]
    return compute_egocentric_rates(
        None,
        spike_times,
        times,
        positions,
        headings,
        object_positions,
        distance_range=(0.0, 50.0),
        distance_bin_size=5.0,
    )


@pytest.fixture
def boundary_batch():
    from shapely.geometry import box

    from neurospatial.encoding import compute_boundary_vector_rates

    rng = np.random.default_rng(0)
    env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=5.0, units="cm")
    times = np.linspace(0, 100, 1000)
    positions = rng.uniform(10, 90, (1000, 2))
    headings = rng.uniform(-np.pi, np.pi, 1000)
    spike_times = [np.sort(rng.uniform(0, 100, k)) for k in (100, 150, 50)]
    return compute_boundary_vector_rates(
        env, spike_times, times, positions, headings, direction_frame="egocentric"
    )


@pytest.fixture
def tuning_batch():
    rng = np.random.default_rng(0)
    times = np.linspace(0, 100, 1000)
    progress = (times % 10) / 10
    spike_times = [np.sort(rng.uniform(0, 100, k)) for k in (100, 150, 50)]
    return compute_tuning_curves(
        spike_times, times, progress, bin_edges=np.linspace(0, 1, 11)
    )


BATCH_FIXTURES = [
    "spatial_batch",
    "directional_batch",
    "view_batch",
    "egocentric_batch",
    "boundary_batch",
    "tuning_batch",
]


@pytest.mark.parametrize("fixture_name", BATCH_FIXTURES)
def test_batch_to_dataframe_is_dense_and_carries_unit_id(fixture_name, request):
    """to_dataframe() is one row per (unit, bin), carrying unit_id."""
    result = request.getfixturevalue(fixture_name)
    df = result.to_dataframe()

    n_units = len(result)
    n_bins = np.asarray(result.firing_rates).shape[1]
    assert len(df) == n_units * n_bins

    # Always carries unit_id and bin columns plus firing_rate/occupancy.
    assert {"unit_id", "bin", "firing_rate", "occupancy"} <= set(df.columns)
    # The unit_id column carries the real per-unit labels.
    assert list(df["unit_id"].unique()) == list(result.unit_ids)
    # Each unit contributes exactly n_bins rows.
    assert (df.groupby("unit_id").size() == n_bins).all()


@pytest.mark.parametrize("fixture_name", BATCH_FIXTURES)
def test_batch_summary_table_is_one_row_per_unit(fixture_name, request):
    """summary_table() is one row per unit, unit_id-indexed."""
    result = request.getfixturevalue(fixture_name)
    summary = result.summary_table()

    assert len(summary) == len(result)
    assert summary.index.name == "unit_id"
    assert list(summary.index) == list(result.unit_ids)
    # peak_rate is the shared scalar metric on every summary table.
    assert "peak_firing_rate" in summary.columns


@pytest.mark.parametrize("fixture_name", BATCH_FIXTURES)
def test_batch_summary_table_custom_unit_ids(fixture_name, request):
    """summary_table(unit_ids=...) re-labels the index."""
    result = request.getfixturevalue(fixture_name)
    custom = [f"u{i}" for i in range(len(result))]
    summary = result.summary_table(unit_ids=custom)
    assert list(summary.index) == custom


def test_to_dataframe_bin_center_vocabulary_is_consistent(
    spatial_batch,
    directional_batch,
    view_batch,
    egocentric_batch,
    boundary_batch,
    tuning_batch,
):
    """Bin-center columns use the shared vocabulary per coordinate space."""
    # Cartesian results -> bin_center_x / bin_center_y.
    for result in (spatial_batch, view_batch):
        cols = set(result.to_dataframe().columns)
        assert {"bin_center_x", "bin_center_y"} <= cols

    # Directional result (no env, angular) -> bin_center_angle.
    assert "bin_center_angle" in set(directional_batch.to_dataframe().columns)

    # Tuning curves (no env, values of a variable) -> bin_center.
    assert "bin_center" in set(tuning_batch.to_dataframe().columns)

    # Egocentric polar results -> bin_center_distance / bin_center_angle.
    for result in (egocentric_batch, boundary_batch):
        cols = set(result.to_dataframe().columns)
        assert {"bin_center_distance", "bin_center_angle"} <= cols


def test_single_unit_to_dataframe_carries_unit_id():
    """Single-unit to_dataframe() is one row per bin, carrying unit_id."""
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 50, (500, 2))
    env = Environment.from_samples(positions, bin_size=5.0)
    times = np.linspace(0, 50, 500)
    spike_times = np.sort(rng.uniform(0, 50, 30))
    result = compute_spatial_rate(env, spike_times, times, positions, bandwidth=10.0)

    df = result.to_dataframe()
    assert len(df) == env.n_bins
    assert {"unit_id", "bin", "firing_rate"} <= set(df.columns)
    # Standalone single-unit computation has no identity -> the unit_id column
    # is a single absent value (pd.NA), not a fabricated 0. Every row shares
    # that same absent identity.
    assert df["unit_id"].isna().all()
    assert df["unit_id"].nunique(dropna=False) == 1


def test_single_unit_to_dataframe_uses_unit_id_when_set(spatial_batch):
    """Indexing a batch carries the unit's real label into the dense frame."""
    single = spatial_batch[1]
    df = single.to_dataframe()
    assert (df["unit_id"] == spatial_batch.unit_ids[1]).all()


# ---------------------------------------------------------------------------
# Every public result class
# ---------------------------------------------------------------------------

_VERBS = (
    "summary",
    "to_dataframe",
    "summary_table",
    "to_xarray",
    "plot",
    "samples_table",
    "aligned_table",
    "trials_table",
    "events_table",
    "time_course_table",
    "category_counts_table",
    "words_table",
    "groups_table",
)
_TABLE_VERBS = tuple(verb for verb in _VERBS if verb.endswith(("table", "dataframe")))
RATE_VERBS = ("summary", "to_dataframe", "summary_table", "to_xarray", "plot")
PSTH_VERBS = ("summary", "to_dataframe", "summary_table", "plot")

# The terminal verbs of each public result class. A verb a class does not
# define is absent from its row, so an intentional exception (a domain table
# such as ``samples_table()`` instead of ``to_dataframe()``) is visible here.
EXPECTED_VERBS: dict[str, frozenset[str]] = {
    name: frozenset(verbs)
    for name, verbs in {
        "AnnotationResult": (),
        "AssemblyDetectionResult": ("summary",),
        "AssemblyEventResponseResult": ("summary",),
        "AssemblyMemberCofiringResult": ("summary",),
        "AssemblyReactivationResult": ("summary",),
        "AssemblySimilarityResult": ("summary",),
        "BiasSimilarityResult": ("summary",),
        "BoundaryVectorRateResult": RATE_VERBS,
        "BoundaryVectorRatesResult": RATE_VERBS,
        "CellTypeSignificanceResult": ("summary", "summary_table"),
        "CircularBasisResult": (),
        "CircularSequenceResult": ("summary",),
        "ContinuousEventResult": (
            "summary",
            "plot",
            "aligned_table",
            "events_table",
            "time_course_table",
            "category_counts_table",
        ),
        "CumulativeDistanceResult": ("summary",),
        "DatasetTestResult": ("summary", "plot"),
        "DecisionAnalysisResult": ("summary",),
        "DecodingResult": ("summary", "to_dataframe", "to_xarray", "plot"),
        "DirectionalRateResult": RATE_VERBS,
        "DirectionalRatesResult": RATE_VERBS,
        "DistributionTestResult": ("summary",),
        "ExplainedVarianceContributionResult": ("summary",),
        "ExplainedVarianceResult": ("summary",),
        "IsotonicFitResult": ("summary",),
        "JointDecodingResult": ("summary",),
        "LinearFitResult": ("summary",),
        "MSDResult": ("summary",),
        "MatchingIndexResult": ("summary",),
        "ObjectVectorRateResult": RATE_VERBS,
        "ObjectVectorRatesResult": RATE_VERBS,
        "PairwiseBiasResult": ("summary",),
        "PairwiseOrderResult": ("summary",),
        "PairwiseOrderSignificanceResult": ("summary",),
        "PathEfficiencyResult": ("summary", "summary_table"),
        "PeriEventDistanceResult": PSTH_VERBS,
        "PeriEventResult": PSTH_VERBS,
        "PhasePrecessionResult": ("summary",),
        "PlaceCellSignificanceResult": ("summary", "summary_table", "plot"),
        "PlaceFieldsResult": ("summary", "to_dataframe", "summary_table", "plot"),
        "PopulationCoverageResult": ("summary",),
        "PopulationPeriEventResult": PSTH_VERBS,
        "RadonDetectionResult": ("summary",),
        "RankOrderResult": ("summary",),
        "RankOrderSignificanceResult": ("summary",),
        "RateComparisonResult": RATE_VERBS,
        "RecurringSequencesResult": ("summary",),
        "RecurringSequencesSignificanceResult": ("summary",),
        "RegressionLineScoreResult": ("summary",),
        "ReplayDetectionResult": ("summary",),
        "ReplaySegmentsResult": ("summary",),
        "ReplaySignificanceResult": ("summary",),
        "ResampledPathResult": ("summary", "samples_table"),
        "SequenceMatchResult": ("summary",),
        "ShuffleTestResult": ("summary", "plot"),
        "SimulationValidationResult": ("summary", "summary_table", "plot"),
        "SpatialFeatureResult": ("summary", "samples_table"),
        "SpatialRateResult": RATE_VERBS,
        "SpatialRatesResult": RATE_VERBS,
        "SpikeWordsResult": ("summary", "words_table"),
        "SubgoalEfficiencyResult": ("summary",),
        "TemplateMatchDatasetResult": ("summary", "groups_table"),
        "TrackGraphResult": (),
        "TuningCurveResult": RATE_VERBS,
        "TuningCurvesResult": RATE_VERBS,
        "VTESessionResult": ("summary", "trials_table"),
        "VTETrialResult": ("summary",),
        "ViewRateResult": RATE_VERBS,
        "ViewRatesResult": RATE_VERBS,
        "ViewshedResult": (),
        "WeightedCorrelationResult": ("summary",),
    }.items()
}

_PUBLIC_RESULTS = public_result_classes()
# Population results whose units are 2-D maps (or own nulls): plot() needs a unit.
_PLOT_NEEDS_UNIT = {
    "SpatialRatesResult",
    "ViewRatesResult",
    "ObjectVectorRatesResult",
    "BoundaryVectorRatesResult",
    "PlaceCellSignificanceResult",
}
_SCALARS = (bool, int, float, str, type(None), np.generic)


@pytest.fixture(scope="module")
def result_instances():
    """One instance per public result class, built once for the module."""
    cache: dict[str, object] = {}

    def build(name: str) -> object:
        if name not in cache:
            cache[name] = RESULT_FACTORIES[name]()
        return cache[name]

    return build


def test_every_public_result_class_has_an_expected_verb_row():
    assert set(EXPECTED_VERBS) == set(_PUBLIC_RESULTS)


@pytest.mark.parametrize("name", sorted(_PUBLIC_RESULTS))
def test_result_class_matches_its_expected_verb_row(name, result_instances):
    """The class defines exactly the verbs of its row, with their contracts."""
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    cls, result = _PUBLIC_RESULTS[name], result_instances(name)
    present = {verb for verb in _VERBS if callable(getattr(cls, verb, None))}
    if "summary" in present:
        try:
            summary = result.summary()
        except NotImplementedError:  # ResultMixin's unspecialized summary()
            present.discard("summary")
        else:
            assert isinstance(summary, dict)
            assert all(isinstance(value, _SCALARS) for value in summary.values())
    assert present == EXPECTED_VERBS[name]

    for verb in sorted(present & set(_TABLE_VERBS)):
        table = getattr(result, verb)()
        assert "units" in table.attrs, f"{name}.{verb}() has no attrs['units']"
    per_unit = hasattr(result, "unit_id") or hasattr(result, "unit_ids")
    if per_unit and "to_dataframe" in present:
        assert "unit_id" in result.to_dataframe().columns
    if "plot" in present:
        if name in _PLOT_NEEDS_UNIT:
            with pytest.raises(ValueError, match="requires a unit index"):
                result.plot()
            ax = result.plot(0)
        else:
            ax = result.plot()
        # A 2-D decode draws its x and y marginals on a pair of axes.
        axes = ax if name == "DecodingResult" and isinstance(ax, tuple) else (ax,)
        assert all(isinstance(each, Axes) for each in axes)
        plt.close("all")


def _resolve(result: object, path: str) -> object:
    """Follow a summary key from ``result``: ``a.b``, ``a[key]`` and ``a()``.

    A bracket key that is all digits indexes an array; any other indexes a
    mapping.
    """
    import re

    value = result
    for attribute, call, key in re.findall(r"\.?(\w+)(\(\))?|\[([^\]]+)\]", path):
        if attribute:
            value = getattr(value, attribute)
            if call:
                value = value()
        else:
            value = value[int(key)] if key.isdigit() else value[key]
    return value


def _same(read: object, shown: object) -> bool:
    """Whether a value read back equals the summary's (NaN equals NaN)."""
    if isinstance(shown, float) and np.isnan(shown):
        return isinstance(read, float | np.floating) and bool(np.isnan(read))
    if isinstance(read, np.ndarray):
        # An array reads back only as the list the summary shows of it, not
        # as a scalar.
        return isinstance(shown, tuple) and read.tolist() == list(shown)
    return bool(read == shown)


@pytest.mark.parametrize("name", sorted(_PUBLIC_RESULTS))
def test_repr_names_are_accessible(name, result_instances):
    """Every ``summary()`` key reads back from the result as the repr shows it.

    The repr shows ``name=`` for an attribute and ``name()=`` for a method
    that needs no argument; a key may be a path, ``a.b``, ``a[key]`` or
    ``a()[0]``. Following the key from the result gives the summary's value.
    """
    import inspect

    from neurospatial._results import ResultMixin

    result = result_instances(name)
    if not isinstance(result, ResultMixin):
        pytest.skip(f"{name} has no summary-based repr")
    # The keys the repr is built from: summary(), less any a class keeps out
    # of its repr (counts that depend on a method's options).
    summary = result._summary_for_repr()
    assert isinstance(summary, dict), f"{name} repr has no names"
    assert summary, f"{name} repr has no names"
    text = repr(result)
    for key, shown in summary.items():
        member = getattr(type(result), key, None)
        if callable(member):
            required = [
                p
                for p in list(inspect.signature(member).parameters.values())[1:]
                if p.default is p.empty
                and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            ]
            assert not required, f"{name}.{key}() needs {required}"
            assert f"{key}()=" in text, f"{name} repr does not show {key}()"
            read = getattr(result, key)()
        else:
            assert f"{key}=" in text, f"{name} repr does not show {key}"
            read = _resolve(result, key)
            assert not callable(read), f"{name}: {key} is a method"
        if isinstance(read, list):
            read = tuple(read)
        if isinstance(shown, list):
            shown = tuple(shown)
        assert _same(read, shown), f"{name}: {key} reads {read!r}, summary {shown!r}"
