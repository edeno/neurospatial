"""Type definitions for track graph annotation module."""

from __future__ import annotations

from typing import Literal

# Interaction modes for track graph builder
# - "add_node": Click to add nodes (waypoints on the track)
# - "add_edge": Click two nodes to create edges (two-click pattern)
# - "delete": Click node/edge to delete
TrackGraphMode = Literal["add_node", "add_edge", "delete"]
