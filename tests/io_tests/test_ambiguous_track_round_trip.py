"""Embedded tracks can have distinct path bins at identical coordinates."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from neurospatial import Environment


@pytest.mark.parametrize("storage", ["dict", "file", "nwb"])
def test_track_with_repeated_embedded_centers_round_trips(storage, tmp_path):
    env = Environment.linear_track(
        node_positions=[(0, 0), (10, 0), (0, 0)], bin_size=1.0
    )
    if storage == "dict":
        restored = Environment.from_dict(env.to_dict())
    elif storage == "file":
        env.to_file(tmp_path / "track")
        restored = Environment.from_file(tmp_path / "track")
    else:
        pynwb = pytest.importorskip("pynwb")
        from neurospatial.io.nwb import read_environment, write_environment

        nwbfile = pynwb.NWBFile("Repeated track", "track", datetime.now(timezone.utc))
        write_environment(nwbfile, env)
        restored = read_environment(nwbfile)

    np.testing.assert_array_equal(restored.bin_centers, env.bin_centers)
    np.testing.assert_array_equal(restored.bin_sizes, env.bin_sizes)
    points = np.array([[0.5, 0.0], [3.2, 0.1], [9.5, 0.0]])
    np.testing.assert_array_equal(restored.bin_at(points), env.bin_at(points))
    np.testing.assert_array_equal(restored.to_linear(points), env.to_linear(points))
