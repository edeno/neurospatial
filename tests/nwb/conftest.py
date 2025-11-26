"""
Fixtures for NWB integration tests.

These fixtures provide sample NWB files with various data types for testing
the neurospatial NWB integration module.

Usage
-----
Tests requiring NWB dependencies should use pytest.importorskip:

>>> pynwb = pytest.importorskip("pynwb")

This ensures tests are skipped gracefully when NWB is not installed.
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import numpy as np
import pytest

# Mark all tests in this directory as integration tests
pytestmark = pytest.mark.integration


def _get_pynwb():
    """Import pynwb or skip test if not available."""
    return pytest.importorskip("pynwb")


def _get_ndx_pose():
    """Import ndx_pose or skip test if not available."""
    return pytest.importorskip("ndx_pose")


def _get_ndx_events():
    """Import ndx_events or skip test if not available."""
    return pytest.importorskip("ndx_events")


def create_empty_nwb():
    """
    Create a minimal empty NWB file for testing.

    Returns
    -------
    NWBFile
        A minimal NWB file with required metadata.

    Examples
    --------
    >>> nwbfile = create_empty_nwb()
    >>> assert nwbfile.identifier is not None
    """
    _get_pynwb()  # Ensure pynwb is available
    from pynwb import NWBFile

    nwbfile = NWBFile(
        session_description="Test session for neurospatial NWB integration",
        identifier=str(uuid4()),
        session_start_time=datetime.now().astimezone(),
        experimenter=["Test User"],
        lab="Test Lab",
        institution="Test Institution",
    )
    return nwbfile


@pytest.fixture
def empty_nwb():
    """
    Fixture providing an empty NWB file.

    Returns
    -------
    NWBFile
        A minimal NWB file with required metadata.
    """
    return create_empty_nwb()


@pytest.fixture
def sample_nwb_with_position():
    """
    Create NWB file with Position data for testing.

    Creates a Position container with a SpatialSeries containing
    simulated 2D position data (1000 samples at 30 Hz).

    Returns
    -------
    NWBFile
        NWB file with Position data in processing/behavior/.
    """
    _get_pynwb()  # Ensure pynwb is available
    from pynwb.behavior import Position, SpatialSeries

    nwbfile = create_empty_nwb()

    # Generate simulated position data (random walk)
    n_samples = 1000
    dt = 1.0 / 30.0  # 30 Hz
    timestamps = np.arange(n_samples) * dt

    # Random walk in 2D bounded to [0, 100] cm
    rng = np.random.default_rng(42)
    velocity = rng.normal(0, 5, (n_samples, 2))
    positions = np.cumsum(velocity, axis=0) + 50  # Start at center
    positions = np.clip(positions, 0, 100)  # Bound to arena

    # Create Position container
    position = Position(name="Position")
    spatial_series = SpatialSeries(
        name="position",
        description="Animal position in arena",
        data=positions,
        timestamps=timestamps,
        reference_frame="Corner of arena (0, 0)",
        unit="cm",
    )
    position.add_spatial_series(spatial_series)

    # Add to processing module
    behavior_module = nwbfile.create_processing_module(
        name="behavior", description="Behavioral data including position"
    )
    behavior_module.add(position)

    return nwbfile


@pytest.fixture
def sample_nwb_with_position_multiple_series():
    """
    Create NWB file with Position containing multiple SpatialSeries.

    This tests the auto-discovery behavior when multiple series exist.

    Returns
    -------
    NWBFile
        NWB file with Position containing 'head' and 'body' SpatialSeries.
    """
    _get_pynwb()  # Ensure pynwb is available
    from pynwb.behavior import Position, SpatialSeries

    nwbfile = create_empty_nwb()

    n_samples = 500
    timestamps = np.arange(n_samples) / 30.0
    rng = np.random.default_rng(42)

    position = Position(name="Position")

    # Add 'body' position (alphabetically second)
    body_pos = rng.uniform(10, 90, (n_samples, 2))
    position.add_spatial_series(
        SpatialSeries(
            name="body",
            description="Body centroid position",
            data=body_pos,
            timestamps=timestamps,
            reference_frame="Arena corner",
            unit="cm",
        )
    )

    # Add 'head' position (alphabetically first)
    head_pos = body_pos + rng.normal(0, 2, (n_samples, 2))
    position.add_spatial_series(
        SpatialSeries(
            name="head",
            description="Head position",
            data=head_pos,
            timestamps=timestamps,
            reference_frame="Arena corner",
            unit="cm",
        )
    )

    behavior_module = nwbfile.create_processing_module(
        name="behavior", description="Behavioral data"
    )
    behavior_module.add(position)

    return nwbfile


@pytest.fixture
def sample_nwb_with_head_direction():
    """
    Create NWB file with CompassDirection data for testing.

    Returns
    -------
    NWBFile
        NWB file with CompassDirection in processing/behavior/.
    """
    _get_pynwb()  # Ensure pynwb is available
    from pynwb.behavior import CompassDirection, SpatialSeries

    nwbfile = create_empty_nwb()

    n_samples = 1000
    timestamps = np.arange(n_samples) / 30.0

    # Simulated head direction (slowly rotating)
    rng = np.random.default_rng(42)
    base_angle = np.linspace(0, 4 * np.pi, n_samples)
    noise = rng.normal(0, 0.1, n_samples)
    angles = (base_angle + noise) % (2 * np.pi)

    compass_direction = CompassDirection(name="CompassDirection")
    spatial_series = SpatialSeries(
        name="head_direction",
        description="Head direction angle",
        data=angles,
        timestamps=timestamps,
        reference_frame="0 = East, increasing counterclockwise",
        unit="radians",
    )
    compass_direction.add_spatial_series(spatial_series)

    behavior_module = nwbfile.create_processing_module(
        name="behavior", description="Behavioral data"
    )
    behavior_module.add(compass_direction)

    return nwbfile


@pytest.fixture
def sample_nwb_with_pose():
    """
    Create NWB file with PoseEstimation data for testing.

    Requires ndx-pose to be installed. Creates a PoseEstimation container
    with skeleton (nose, body, tail) and corresponding pose series.

    Returns
    -------
    NWBFile
        NWB file with PoseEstimation in processing/behavior/.
    """
    _get_ndx_pose()  # Ensure ndx-pose is available
    from ndx_pose import PoseEstimation, PoseEstimationSeries, Skeleton

    nwbfile = create_empty_nwb()

    n_samples = 500
    timestamps = np.arange(n_samples) / 30.0
    rng = np.random.default_rng(42)

    # Create skeleton
    skeleton = Skeleton(
        name="mouse_skeleton",
        nodes=["nose", "body", "tail"],
        edges=np.array([[0, 1], [1, 2]], dtype=np.uint8),  # nose-body, body-tail
    )

    # Generate pose data (body at center, nose/tail relative)
    body_pos = rng.uniform(20, 80, (n_samples, 2))
    direction = rng.uniform(0, 2 * np.pi, n_samples)

    # Nose is 5 cm in front of body
    nose_pos = body_pos + 5 * np.column_stack([np.cos(direction), np.sin(direction)])
    # Tail is 8 cm behind body
    tail_pos = body_pos - 8 * np.column_stack([np.cos(direction), np.sin(direction)])

    confidence = rng.uniform(0.8, 1.0, n_samples)

    # Create pose estimation series for each bodypart
    pose_series = []
    for name, data in [("nose", nose_pos), ("body", body_pos), ("tail", tail_pos)]:
        series = PoseEstimationSeries(
            name=name,
            description=f"{name} position",
            data=data,
            confidence=confidence,
            timestamps=timestamps,
            reference_frame="Arena corner",
            unit="cm",
        )
        pose_series.append(series)

    # Create PoseEstimation container
    pose_estimation = PoseEstimation(
        name="PoseEstimation",
        pose_estimation_series=pose_series,
        description="Pose estimation from DeepLabCut",
        skeleton=skeleton,
        source_software="DeepLabCut",
    )

    behavior_module = nwbfile.create_processing_module(
        name="behavior", description="Behavioral data including pose"
    )
    behavior_module.add(pose_estimation)

    # Add skeleton to file
    if "Skeletons" not in nwbfile.processing:
        nwbfile.create_processing_module(
            name="Skeletons", description="Skeleton definitions"
        )
    nwbfile.processing["Skeletons"].add(skeleton)

    return nwbfile


@pytest.fixture
def sample_nwb_with_events():
    """
    Create NWB file with EventsTable for testing.

    Requires ndx-events to be installed. Creates an EventsTable
    representing lap detection events.

    Returns
    -------
    NWBFile
        NWB file with EventsTable in processing/behavior/.
    """
    _get_ndx_events()  # Ensure ndx-events is available
    from ndx_events import EventsTable

    nwbfile = create_empty_nwb()

    # Create lap events
    events = EventsTable(
        name="laps",
        description="Detected lap events during linear track running",
    )

    # Add columns
    events.add_column(
        name="direction", description="Lap direction (0=outbound, 1=inbound)"
    )
    events.add_column(name="duration", description="Lap duration in seconds")

    # Add sample lap events
    rng = np.random.default_rng(42)
    n_laps = 20
    lap_times = np.sort(rng.uniform(0, 600, n_laps))  # 10 minutes of data
    directions = rng.choice([0, 1], size=n_laps)
    durations = rng.uniform(3, 8, n_laps)

    for i in range(n_laps):
        events.add_row(
            timestamp=lap_times[i],
            direction=directions[i],
            duration=durations[i],
        )

    behavior_module = nwbfile.create_processing_module(
        name="behavior", description="Behavioral data"
    )
    behavior_module.add(events)

    return nwbfile


@pytest.fixture
def sample_environment():
    """
    Create a sample Environment for testing NWB round-trip.

    Returns
    -------
    Environment
        A 2D grid environment with sample regions.
    """
    from neurospatial import Environment

    rng = np.random.default_rng(42)
    positions = rng.uniform(0, 100, (1000, 2))

    env = Environment.from_samples(positions, bin_size=5.0)
    env.units = "cm"
    env.frame = "test_session"

    # Add sample regions
    env.regions.add("start", point=(10.0, 10.0))
    env.regions.add("goal", point=(90.0, 90.0))

    return env
