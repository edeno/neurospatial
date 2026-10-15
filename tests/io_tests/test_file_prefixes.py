"""Environment file pairs preserve dots that identify different recordings."""

from pathlib import Path

import numpy as np
import pytest

from neurospatial import Environment


def _environment(name: str) -> Environment:
    return Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool),
        grid_edges=(np.arange(3.0), np.arange(3.0)),
        name=name,
    )


@pytest.mark.parametrize("extension", ["", ".json", ".npz"])
@pytest.mark.parametrize("as_string", [False, True])
def test_dotted_prefix_round_trips_with_optional_format_extension(
    tmp_path, extension, as_string
):
    env = _environment("day1")
    path = tmp_path / f"rat1.day1{extension}"
    argument = str(path) if as_string else path
    env.to_file(argument)
    assert (tmp_path / "rat1.day1.json").is_file()
    assert (tmp_path / "rat1.day1.npz").is_file()
    restored = Environment.from_file(argument)
    assert restored.name == env.name
    np.testing.assert_array_equal(restored.bin_centers, env.bin_centers)


def test_loading_dotted_prefix_uses_the_entire_recording_name(tmp_path):
    env = _environment("day1")
    env.to_file(tmp_path / "temporary")
    for extension in ("json", "npz"):
        (tmp_path / f"temporary.{extension}").rename(
            tmp_path / f"rat1.day1.{extension}"
        )
    restored = Environment.from_file(tmp_path / "rat1.day1")
    assert restored.name == "day1"


def test_separate_dotted_recordings_do_not_collide_or_overwrite_each_other(tmp_path):
    first, second = _environment("day1"), _environment("day2")
    first.to_file(tmp_path / "rat1.day1")
    second.to_file(tmp_path / "rat1.day2")
    _environment("updated day1").to_file(tmp_path / "rat1.day1", overwrite=True)
    assert Environment.from_file(tmp_path / "rat1.day1").name == "updated day1"
    assert Environment.from_file(tmp_path / "rat1.day2").name == "day2"
    assert {path.name for path in tmp_path.iterdir()} == {
        "rat1.day1.json",
        "rat1.day1.npz",
        "rat1.day2.json",
        "rat1.day2.npz",
    }


def test_overwrite_guard_applies_to_the_same_dotted_prefix(tmp_path):
    env = _environment("day1")
    env.to_file(tmp_path / "rat1.day1")
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(FileExistsError):
        env.to_file(Path(str(tmp_path / "rat1.day1") + ".npz"))
    for path, contents in before.items():
        assert path.read_bytes() == contents
