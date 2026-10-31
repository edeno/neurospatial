"""Pose readers and adapters preserve acquisition clocks and coordinates."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ndx_pose")
pytest.importorskip("pynwb")

from neurospatial import Environment
from neurospatial.io.nwb import bodypart_overlay_from_nwb, read_pose


def _add_pose(
    nwbfile,
    *,
    clocks=None,
    units=("cm", "cm"),
    frames=("arena", "arena"),
    conversion=1.0,
    n_samples=4,
):
    from ndx_pose import PoseEstimation, PoseEstimationSeries, Skeleton

    if clocks is None:
        clocks = ({"timestamps": np.arange(n_samples) / 10},) * 2
    skeleton = Skeleton(
        name="mouse", nodes=["nose", "tail"], edges=np.array([[0, 1]], dtype=np.uint8)
    )
    series = [
        PoseEstimationSeries(
            name=name,
            data=np.arange(n_samples * 2, dtype=float).reshape(n_samples, 2),
            confidence=np.ones(n_samples),
            unit=unit,
            reference_frame=frame,
            conversion=conversion,
            **clock,
        )
        for name, clock, unit, frame in zip(
            ("nose", "tail"), clocks, units, frames, strict=True
        )
    ]
    pose = PoseEstimation(
        name="PoseEstimation",
        pose_estimation_series=series,
        skeleton=skeleton,
        source_software="test",
    )
    nwbfile.create_processing_module(name="behavior", description="Behavior").add(pose)
    nwbfile.create_processing_module(name="Skeletons", description="Skeletons").add(
        skeleton
    )
    return pose


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("clock_kind", ["explicit", "implicit", "mixed", "rate"])
def test_rejects_same_length_different_clocks(empty_nwb, tmp_path, lazy, clock_kind):
    """A shared sample count does not imply a shared acquisition clock."""
    from pynwb import NWBHDF5IO

    explicit = {"timestamps": np.arange(4) / 10}
    implicit = {"rate": 10.0, "starting_time": 0.0}
    if clock_kind == "explicit":
        clocks = (explicit, {"timestamps": np.arange(4) / 10 + 0.05})
    elif clock_kind == "implicit":
        clocks = (implicit, {"rate": 10.0, "starting_time": 0.05})
    elif clock_kind == "mixed":
        clocks = (explicit, {"rate": 10.0, "starting_time": 0.05})
    else:
        clocks = (implicit, {"rate": 9.0, "starting_time": 0.0})
    _add_pose(empty_nwb, clocks=clocks)
    path = tmp_path / "pose.nwb"
    with NWBHDF5IO(str(path), "w") as io:
        io.write(empty_nwb)
    with NWBHDF5IO(str(path), "r") as io:
        nwbfile = io.read()
        with pytest.raises(ValueError, match=r"tail.*timestamps do not match.*nose"):
            read_pose(nwbfile, lazy=lazy)
        with pytest.raises(ValueError, match="timestamps do not match"):
            bodypart_overlay_from_nwb(nwbfile)


@pytest.mark.parametrize("lazy", [False, True])
def test_equivalent_explicit_and_implicit_clocks_keep_handles(
    empty_nwb, tmp_path, lazy
):
    """Roundoff in mathematically equal clocks is harmless; lazy data stays lazy."""
    import h5py
    from pynwb import NWBHDF5IO

    expected = np.arange(4) / 10
    pose = _add_pose(
        empty_nwb,
        clocks=(
            {"timestamps": np.arange(4) * 0.1},
            {"rate": 10.0, "starting_time": 0.0},
        ),
    )
    path = tmp_path / "pose.nwb"
    with NWBHDF5IO(str(path), "w") as io:
        io.write(empty_nwb)
    with NWBHDF5IO(str(path), "r") as io:
        nwbfile = io.read()
        result = read_pose(nwbfile, lazy=lazy)
        parts, times, skeleton = result.bodyparts, result.times, result.skeleton
        assert set(skeleton.nodes) == {"nose", "tail"}
        np.testing.assert_allclose(times[:], expected)
        if lazy:
            stored = nwbfile.processing["behavior"][pose.name]
            assert isinstance(times, h5py.Dataset)
            assert times is stored.pose_estimation_series["nose"].timestamps
            for name, handle in parts.items():
                assert isinstance(handle, h5py.Dataset)
                assert handle is stored.pose_estimation_series[name].data


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("metadata", ["units", "reference_frame"])
def test_conflicting_bodypart_metadata_is_rejected(empty_nwb, lazy, metadata):
    """One overlay cannot safely interpret incompatible coordinate declarations."""
    options = (
        {"units": ("cm", "m")}
        if metadata == "units"
        else {"frames": ("arena", "camera")}
    )
    _add_pose(empty_nwb, **options)
    with pytest.raises(ValueError, match=f"tail.*{metadata}.*does not match"):
        read_pose(empty_nwb, lazy=lazy)
    with pytest.raises(ValueError, match=f"tail.*{metadata}.*does not match"):
        bodypart_overlay_from_nwb(empty_nwb)


def _environment(*, units="cm", frame="arena"):
    return Environment.from_grid_mask(
        np.ones((8, 8), dtype=bool),
        (np.arange(9.0), np.arange(9.0)),
        units=units,
        frame=frame,
    )


def test_adapter_preserves_metadata_aliases_and_nwb_scaling(empty_nwb):
    """NWB conversion applies once, while unit aliases only normalize spelling."""
    _add_pose(empty_nwb, units=("centimetres", "cm"), conversion=0.1)
    overlay = bodypart_overlay_from_nwb(empty_nwb, units="centimeters", frame="arena")
    assert overlay.units == "cm"
    assert overlay.frame == "arena"
    data = overlay.convert_to_data(np.arange(4) / 10, 4, _environment())
    expected = np.arange(8).reshape(4, 2) * 0.1
    np.testing.assert_allclose(data.bodyparts["nose"], expected)
    np.testing.assert_allclose(data.bodyparts["tail"], expected)


@pytest.mark.parametrize("env_metadata", [{"units": "m"}, {"frame": "camera"}])
def test_adapter_metadata_checked_against_environment(empty_nwb, env_metadata):
    _add_pose(empty_nwb)
    overlay = bodypart_overlay_from_nwb(empty_nwb)
    with pytest.raises(
        ValueError, match=r"BodypartOverlay.*does not match environment"
    ):
        overlay.convert_to_data(np.arange(4) / 10, 4, _environment(**env_metadata))


@pytest.mark.parametrize(
    "overrides",
    [
        {"units": "m"},
        {"units": None},
        {"frame": "camera"},
        {"frame": None},
    ],
)
def test_keyword_metadata_cannot_relabel_known_coordinates(empty_nwb, overrides):
    _add_pose(empty_nwb)
    with pytest.raises(ValueError, match="conflicts with NWB pose"):
        bodypart_overlay_from_nwb(empty_nwb, **overrides)


def test_reference_frame_description_is_preserved_verbatim(empty_nwb):
    description = "Image corner (0, 0); x right, y down"
    _add_pose(empty_nwb, frames=(description, description))
    overlay = bodypart_overlay_from_nwb(empty_nwb)
    assert overlay.frame == description
    overlay.convert_to_data(np.arange(4) / 10, 4, _environment(frame=description))
    # An undeclared environment does not infer meaning from the description.
    overlay.convert_to_data(np.arange(4) / 10, 4, _environment(frame=None))


def test_keyword_can_supply_missing_frame(empty_nwb):
    _add_pose(empty_nwb, frames=("", ""))
    overlay = bodypart_overlay_from_nwb(empty_nwb, frame="arena")
    assert overlay.frame == "arena"
    overlay.convert_to_data(np.arange(4) / 10, 4, _environment())


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("frames", [("arena", ""), ("", "arena")])
def test_missing_frame_cannot_inherit_another_bodyparts_declaration(
    empty_nwb, lazy, frames
):
    _add_pose(empty_nwb, frames=frames)
    with pytest.raises(ValueError, match=r"reference_frame.*does not match"):
        read_pose(empty_nwb, lazy=lazy)
    with pytest.raises(ValueError, match=r"reference_frame.*does not match"):
        bodypart_overlay_from_nwb(empty_nwb, frame="arena")


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize(
    "times",
    [
        np.array([0.0, 0.1, 0.1, 0.3]),
        np.array([0.0, 0.2, 0.1, 0.3]),
        np.array([0.0, np.nan, 0.2, 0.3]),
        np.array([0.0, 0.1, 0.2, np.inf]),
    ],
)
def test_invalid_first_clock_is_rejected(empty_nwb, lazy, times):
    _add_pose(empty_nwb, clocks=({"timestamps": times},) * 2)
    with pytest.raises(ValueError, match=r"nose.*finite and strictly increasing"):
        read_pose(empty_nwb, lazy=lazy)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("units", [("cm", ""), ("", "cm")])
def test_missing_units_cannot_inherit_another_bodyparts_declaration(
    empty_nwb, lazy, units
):
    _add_pose(empty_nwb, units=units)
    with pytest.raises(ValueError, match=r"units.*does not match"):
        read_pose(empty_nwb, lazy=lazy)


@pytest.mark.parametrize("lazy", [False, True])
def test_clock_validation_crosses_chunk_boundaries(empty_nwb, lazy):
    n_samples = 65538
    times = np.arange(n_samples) / 10
    times[65536] = times[65535]
    _add_pose(empty_nwb, clocks=({"timestamps": times},) * 2, n_samples=n_samples)
    with pytest.raises(ValueError, match=r"nose.*finite and strictly increasing"):
        read_pose(empty_nwb, lazy=lazy)


def test_custom_units_are_spelled_as_position_reads_them(empty_nwb):
    """Pose and position readers normalize unit declarations identically."""
    from neurospatial.io.nwb._adapters import position_units_from_series

    pose = _add_pose(empty_nwb, units=(" Arena ", "Arena"))
    overlay = bodypart_overlay_from_nwb(empty_nwb)
    assert overlay.units == "Arena"
    assert overlay.units == position_units_from_series(
        pose.pose_estimation_series["nose"]
    )


def _add_scored_pose(nwbfile):
    """Nose with a NaN (untracked) frame and a 0.01-confidence frame."""
    from ndx_pose import PoseEstimation, PoseEstimationSeries, Skeleton

    skeleton = Skeleton(
        name="mouse", nodes=["nose", "tail"], edges=np.array([[0, 1]], dtype=np.uint8)
    )
    stored = {
        "nose": np.array([[10.0, 20.0], [np.nan, np.nan], [30.0, 40.0]]),
        "tail": np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]]),
    }
    scores = {"nose": np.array([0.99, 0.0, 0.01]), "tail": np.array([0.9, 0.8, 0.7])}
    series = [
        PoseEstimationSeries(
            name=name,
            data=stored[name],
            confidence=scores[name],
            timestamps=np.arange(3) / 10,
            unit="pixels",
            reference_frame="image corner",
            conversion=0.5,
            offset=1.0,
        )
        for name in ("nose", "tail")
    ]
    pose = PoseEstimation(
        name="PoseEstimation",
        pose_estimation_series=series,
        skeleton=skeleton,
        source_software="test",
    )
    nwbfile.create_processing_module(name="behavior", description="Behavior").add(pose)
    nwbfile.create_processing_module(name="Skeletons", description="Skeletons").add(
        skeleton
    )
    return stored, scores


def test_eager_pose_survives_file_close_with_nan_and_confidence(empty_nwb, tmp_path):
    """Values, NaN keypoints, confidence and declarations outlive the file."""
    from pynwb import NWBHDF5IO

    from neurospatial.io.nwb import NWBPose

    stored, scores = _add_scored_pose(empty_nwb)
    path = tmp_path / "scored_pose.nwb"
    with NWBHDF5IO(str(path), "w") as io:
        io.write(empty_nwb)
    with NWBHDF5IO(str(path), "r") as io:
        pose = read_pose(io.read())
    assert isinstance(pose, NWBPose)
    assert (pose.units, pose.frame) == ("px", "image corner")
    np.testing.assert_allclose(pose.times, np.arange(3) / 10)
    for name in ("nose", "tail"):
        np.testing.assert_array_equal(pose.bodyparts[name], stored[name] * 0.5 + 1.0)
        np.testing.assert_array_equal(pose.confidence[name], scores[name])
    assert np.isnan(pose.bodyparts["nose"][1]).all()
    # A 0.01-confidence keypoint is returned unmasked; its score says so.
    np.testing.assert_array_equal(pose.bodyparts["nose"][2], [16.0, 21.0])
    assert pose.confidence["nose"][2] == 0.01
    assert set(pose.skeleton.nodes) == {"nose", "tail"}
    with pytest.raises(TypeError):
        pose.bodyparts["nose"] = np.zeros((3, 2))
    with pytest.raises(ValueError, match="read-only"):
        pose.bodyparts["nose"][0, 0] = 0.0


def test_lazy_pose_confidence_is_a_file_handle(empty_nwb, tmp_path):
    import h5py
    from pynwb import NWBHDF5IO

    _add_pose(empty_nwb)
    path = tmp_path / "pose.nwb"
    with NWBHDF5IO(str(path), "w") as io:
        io.write(empty_nwb)
    with NWBHDF5IO(str(path), "r") as io:
        pose = read_pose(io.read(), lazy=True)
        assert isinstance(pose.confidence["nose"], h5py.Dataset)
        np.testing.assert_array_equal(pose.confidence["nose"][:], np.ones(4))


def test_pose_holder_checks_rows_and_pickles():
    import pickle

    from neurospatial.io.nwb import NWBPose

    times = np.arange(3.0)
    with pytest.raises(ValueError, match=r"Length mismatch: bodyparts\['nose'\]=2"):
        NWBPose({"nose": np.zeros((2, 2))}, times, None)
    with pytest.raises(
        ValueError, match=r"confidence names unknown bodyparts \['ear'\]"
    ):
        NWBPose({"nose": np.zeros((3, 2))}, times, None, confidence={"ear": times})
    pose = NWBPose(
        {"nose": np.zeros((3, 2))}, times, None, units="cm", confidence={"nose": times}
    )
    restored = pickle.loads(pickle.dumps(pose))
    np.testing.assert_array_equal(restored.confidence["nose"], times)
    assert restored.units == "cm"


def test_pose_without_skeleton_reads(empty_nwb):
    from ndx_pose import PoseEstimation, PoseEstimationSeries

    series = PoseEstimationSeries(
        name="nose",
        data=np.zeros((3, 2)),
        confidence=np.ones(3),
        timestamps=np.arange(3) / 10,
        unit="cm",
        reference_frame="arena",
    )
    empty_nwb.create_processing_module(name="behavior", description="b").add(
        PoseEstimation(name="pose", pose_estimation_series=[series])
    )
    pose = read_pose(empty_nwb)
    assert pose.skeleton is None
    assert list(pose.bodyparts) == ["nose"]


def test_overlay_refuses_skeleton_nodes_that_are_not_series_names(empty_nwb):
    """Series 'PoseEstimationSeriesNose' is not node 'nose': fail at build time."""
    from ndx_pose import PoseEstimation, PoseEstimationSeries, Skeleton

    from neurospatial.animation.overlays import BodypartOverlay

    skeleton = Skeleton(
        name="mouse", nodes=["nose", "tail"], edges=np.array([[0, 1]], dtype=np.uint8)
    )
    series = [
        PoseEstimationSeries(
            name=f"PoseEstimationSeries{node.title()}",
            data=np.full((4, 2), float(i)),
            confidence=np.ones(4),
            timestamps=np.arange(4) / 10,
            unit="cm",
            reference_frame="arena",
        )
        for i, node in enumerate(("nose", "tail"))
    ]
    empty_nwb.create_processing_module(name="behavior", description="b").add(
        PoseEstimation(name="pose", pose_estimation_series=series, skeleton=skeleton)
    )
    empty_nwb.create_processing_module(name="Skeletons", description="s").add(skeleton)
    with pytest.raises(ValueError) as caught:
        bodypart_overlay_from_nwb(empty_nwb)
    message = str(caught.value)
    assert (
        "Skeleton 'mouse' connects nodes ['nose', 'tail'] that are not "
        "PoseEstimationSeries names ['PoseEstimationSeriesNose', "
        "'PoseEstimationSeriesTail']" in message
    )
    # The documented fix: map nodes to series explicitly.
    pose = read_pose(empty_nwb)
    overlay = BodypartOverlay(
        data={
            "nose": pose.bodyparts["PoseEstimationSeriesNose"],
            "tail": pose.bodyparts["PoseEstimationSeriesTail"],
        },
        times=pose.times,
        skeleton=pose.skeleton,
        units=pose.units,
        frame=pose.frame,
    )
    data = overlay.convert_to_data(np.arange(4) / 10, 4, _environment())
    np.testing.assert_array_equal(data.bodyparts["tail"], np.ones((4, 2)))
