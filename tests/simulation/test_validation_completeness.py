"""Recovery reports must keep missing metrics and unit identity visible."""

from dataclasses import replace
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.simulation import PlaceCellModel, SimulationSession
from neurospatial.simulation.validation import validate_simulation


def _session(n_bins=4):
    env = Environment.from_grid_mask(
        np.ones(n_bins, dtype=bool), grid_edges=(np.arange(n_bins + 1.0),)
    )
    env = env.with_metadata(units="cm")
    times = np.arange(n_bins * 50) * 0.1
    positions = np.repeat(env.bin_centers, 50, axis=0)
    spikes = np.concatenate(
        [
            times[i * 50 : (i + 1) * 50 : step]
            for i, step in enumerate((5, 10, 25, 50)[:n_bins])
        ]
    )
    model = PlaceCellModel(env, center=env.bin_centers[0], width=1.0, max_rate=10.0)
    return SimulationSession(
        env=env,
        spike_times=[spikes],
        unit_ids=np.array([11], dtype=np.int64),
        times=times,
        positions=positions,
        models=[model],
        ground_truth={11: model.ground_truth},
        metadata={},
    )


def test_missing_correlation_cannot_pass_and_can_be_plotted():
    """A two-bin map has a peak but insufficient bins for field correlation."""
    with pytest.warns(UserWarning, match="No cells could be fully validated"):
        result = validate_simulation(
            _session(2), method="binned", bandwidth=0.0, show_plots=True
        )
    try:
        assert result["center_errors"][0] == 0.0
        assert np.isnan(result["correlations"][0])
        assert result["passed"] is False
        assert result["unit_status"] == ("not_evaluable",)
        assert result["skip_reasons"] == ("insufficient_finite_bins",)
        assert result["n_evaluated"] == 0
        assert result["plots"].axes[2].collections[0].get_offsets().shape == (0, 2)
    finally:
        plt.close(result["plots"])


def test_partial_report_preserves_requested_labels_and_spike_counts():
    """A passing unit cannot hide a silent or unsupported selected unit."""
    session = _session()
    train = session.spike_times[0]
    session = replace(
        session,
        unit_ids=np.array([7, 11, 19]),
        spike_times=[np.array([]), train, train],
        models=session.models * 3,
        ground_truth={
            7: session.ground_truth[11],
            11: session.ground_truth[11],
            19: {"center": session.env.bin_centers[0]},
        },
    )
    with pytest.warns(UserWarning, match="19: skipping field-correlation"):
        result = validate_simulation(
            session,
            unit_ids=[19, 7, 11],
            method="binned",
            bandwidth=0.0,
            show_plots=True,
        )
    try:
        np.testing.assert_array_equal(result["unit_ids"], [19, 7, 11])
        np.testing.assert_array_equal(
            result["spike_counts"], [len(train), 0, len(train)]
        )
        assert result["unit_status"] == ("not_evaluable", "not_evaluable", "passed")
        assert result["skip_reasons"] == ("unsupported_ground_truth", "no_spikes", None)
        assert result["n_selected"] == 3
        assert result["n_evaluated"] == 1
        assert result["passed"] is False
        assert "Cells selected: 3 (of 3 total)" in result["summary"]
        assert "Cells fully evaluated: 1" in result["summary"]
        np.testing.assert_allclose(
            result["plots"].axes[2].collections[0].get_offsets(),
            [[result["center_errors"][2], result["correlations"][2]]],
        )
    finally:
        plt.close(result["plots"])


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        (np.ones(4), "constant_rate_map"),
        (np.full(4, np.nan), "no_finite_rate_bins"),
    ],
)
def test_degenerate_rate_maps_are_not_evaluable(monkeypatch, field, reason):
    # Supply exact degenerate estimates to isolate report handling from the
    # estimator's floating-point exposure calculations.
    monkeypatch.setattr(
        "neurospatial.encoding.compute_spatial_rate",
        lambda *args, **kwargs: SimpleNamespace(firing_rate=field),
    )
    with pytest.warns(UserWarning, match="No cells could be fully validated"):
        result = validate_simulation(_session())
    assert result["passed"] is False
    assert result["unit_status"] == ("not_evaluable",)
    assert result["skip_reasons"] == (reason,)


def test_complete_report_can_pass_or_fail_recovery_thresholds():
    session = _session()
    passed = validate_simulation(session, method="binned", bandwidth=0.0)
    assert passed["passed"] is True
    assert passed["unit_status"] == ("passed",)
    assert passed["n_evaluated"] == passed["n_selected"] == 1
    assert passed["skip_reasons"] == (None,)

    failed = validate_simulation(
        session, method="binned", bandwidth=0.0, min_correlation=0.999
    )
    assert failed["passed"] is False
    assert failed["unit_status"] == ("failed",)
    assert failed["skip_reasons"] == (None,)
    assert failed["n_evaluated"] == 1


def test_empty_selection_cannot_pass():
    with pytest.warns(UserWarning, match="0 selected"):
        result = validate_simulation(_session(), unit_ids=[])
    assert result["passed"] is False
    assert result["n_selected"] == result["n_evaluated"] == 0
    assert result["unit_status"] == result["skip_reasons"] == ()
