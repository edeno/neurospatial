"""Capture pre-Stage-D wall and rebin behavior on deterministic geometry."""

import json

import numpy as np
from shapely.geometry import LineString, box

from neurospatial import Environment
from neurospatial.arena import OpenArena
from neurospatial.encoding.border import border_score
from neurospatial.ops.visibility import compute_viewed_location


def main() -> None:
    """Report lookup support, physical walls and observed consumer outputs."""
    arena = OpenArena(box(0, 0, 10, 10), walls=(LineString([(5, 0), (5, 10)]),))
    full = Environment.from_grid_mask(
        np.ones((10, 10), dtype=bool),
        grid_edges=(np.arange(11.0),) * 2,
        arena=arena,
        units="cm",
        frame="lab",
    )
    crop = full.subset(bins=full.bin_centers[:, 0] < 4)
    observations = {}
    for name, env in (("full", full), ("crop", crop)):
        field = (env.bin_centers[:, 0] == 3.5).astype(float)
        observations[name] = {
            "ray_east": compute_viewed_location(
                np.array([[2.5, 4.5]]),
                np.array([0.0]),
                env=env,
                method="ray_cast",
                max_distance=30,
            ).tolist(),
            "border_euclidean": border_score(env, field, metric="euclidean"),
            "boundary_bins": env.boundary_bins.tolist(),
        }
    try:
        full.rebin(bin_size=2.0)
    except TypeError as error:
        observations["new_resolution"] = str(error)
    print(json.dumps(observations, indent=2))


if __name__ == "__main__":
    main()
