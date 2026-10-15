"""Coarsening changes bins while retaining continuous region annotations."""

from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment, Regions


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("factor", [1, 2, (2, 1)])
@pytest.mark.parametrize("transport", ["live", "dict", "file"])
def test_factor_rebin_preserves_owned_region_coordinates_and_metadata(
    tmp_path: Path, masked: bool, factor: int | tuple[int, ...], transport: str
) -> None:
    regions = Regions()
    regions.add("goal", point=[4.5, 4.5], metadata={"task": {"labels": ["reward"]}})
    regions.add("zone", polygon=box(3.25, 3.25, 4.75, 4.75), metadata={"color": "blue"})
    expected = {name: region.to_dict() for name, region in regions.items()}
    positions = np.array(
        [(x + 0.5, y + 0.5) for x in range(5) for y in range(5) if (x, y) != (2, 2)]
    )
    env = Environment.from_layout(
        "RegularGrid",
        {
            "bin_size": 1.0,
            "dimension_ranges": [(0, 5), (0, 5)],
            "positions": positions,
            "infer_active_bins": masked,
            "dilate": False,
            "fill_holes": False,
            "close_gaps": False,
        },
        regions=regions,
    )
    if factor == 1:
        coarse = env.rebin(factor)
    else:
        with pytest.warns(UserWarning, match="truncated"):
            coarse = env.rebin(factor)
    if transport == "dict":
        coarse = Environment.from_dict(coarse.to_dict())
    elif transport == "file":
        path = tmp_path / "coarse"
        coarse.to_file(path)
        coarse = Environment.from_file(path)

    assert tuple(coarse.regions) == ("goal", "zone")
    assert {
        name: region.to_dict() for name, region in coarse.regions.items()
    } == expected
    assert coarse.regions is not env.regions
    assert not np.shares_memory(coarse.regions["goal"].data, env.regions["goal"].data)
    assert (coarse.units, coarse.frame) == (env.units, env.frame)
    # The goal remains an annotation even when truncation removes its bins.
    np.testing.assert_array_equal(coarse.regions["goal"].data, [4.5, 4.5])
    regions.update_region("goal", point=[0.5, 0.5])
    assert {
        name: region.to_dict() for name, region in coarse.regions.items()
    } == expected
    with pytest.raises(TypeError):
        coarse.regions["goal"].metadata["task"]["labels"] = ["changed"]
