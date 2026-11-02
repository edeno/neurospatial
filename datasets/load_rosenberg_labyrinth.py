"""Rosenberg, Zhang, Perona & Meister (2021) mice exploring a binary-tree labyrinth."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from _record import Range, Recording

DATASET = {
    "key": "rosenberg_labyrinth",
    "title": "Rosenberg et al. 2021, mouse in a 6-level binary-tree labyrinth (behavior only)",
    "source": "https://github.com/markusmeister/Rosenberg-2021-Repository",
    "license": "MIT",
    "download_size": "6.3 MB",
}

# Pre-registered from the article and the repository's maze code, before the
# first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_dims": 2,
    "sampling_rate_hz": Range(29.9, 30.1),
    "maze_has_127_nodes": lambda rec: rec.geometry["graph"].number_of_nodes() == 127,
    "maze_has_63_junctions": lambda rec: sum(
        1 for _, degree in rec.geometry["graph"].degree() if degree == 3
    )
    == 63,
}


def load(cache: Path | None = None) -> Recording:
    raise NotImplementedError
