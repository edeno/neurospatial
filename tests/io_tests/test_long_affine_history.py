"""Valid affine histories are not limited by the composite-domain depth cap."""

from datetime import datetime, timezone

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.arena import OpenArena, decode_arena
from neurospatial.ops.transforms import translate


@pytest.mark.parametrize("storage", ["dict", "file", "nwb"])
def test_thirty_three_affine_frames_restore_the_original_queries(tmp_path, storage):
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool),
        grid_edges=(np.arange(3.0),) * 2,
        arena=OpenArena(box(0, 0, 2, 2)),
    )
    for _ in range(33):
        env = env.apply_transform(translate(0.125, -0.25))
    points = np.vstack((env.bin_centers, [[1000.0, 1000.0]]))
    expected = env.bin_at(points)
    if storage == "dict":
        restored = Environment.from_dict(env.to_dict())
    elif storage == "file":
        env.to_file(tmp_path / "translated")
        restored = Environment.from_file(tmp_path / "translated")
    else:
        pynwb = pytest.importorskip("pynwb")
        from neurospatial.io.nwb import read_environment, write_environment

        nwbfile = pynwb.NWBFile(
            "affine history",
            "affine-history",
            datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        write_environment(nwbfile, env)
        with pynwb.NWBHDF5IO(str(tmp_path / "translated.nwb"), "w") as io:
            io.write(nwbfile)
        with pynwb.NWBHDF5IO(str(tmp_path / "translated.nwb"), "r") as io:
            restored = read_environment(io.read())
    np.testing.assert_array_equal(restored.bin_at(points), expected)
    np.testing.assert_array_equal(
        restored.arena.contains(points), env.arena.contains(points)
    )
    np.testing.assert_array_equal(restored.bin_sizes, env.bin_sizes)


def test_cyclic_arena_metadata_is_rejected_before_recursive_decoding():
    meta = {"codec": "_TransformedArena", "codec_version": 1}
    meta["base"] = meta
    with pytest.raises(ValueError, match="Cyclic arena metadata"):
        decode_arena(meta, {})
