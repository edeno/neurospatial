"""Pure state objects for annotation mode tracking.

This module contains napari-independent state management, making annotation
logic testable without a GUI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from neurospatial.annotation._helpers import REGION_TYPE_CATEGORIES
from neurospatial.annotation._types import RegionType


@dataclass
class AnnotationModeState:
    """Pure state object tracking annotation mode and shape counts.

    Encapsulates:
    - Current annotation region type (environment/hole/region/wall/landmark)
    - Per-type shape counts for auto-naming
    - Default name generation logic

    This class is deliberately napari-independent so state transitions
    and naming rules can be tested without a GUI.

    Parameters
    ----------
    region_type : RegionType
        Current annotation mode.

    Attributes
    ----------
    environment_count : int
        Number of environment shapes created.
    hole_count : int
        Number of hole shapes created.
    region_count : int
        Number of region shapes created.
    wall_count : int
        Number of wall shapes created.
    landmark_count : int
        Number of landmark shapes created.

    Examples
    --------
    >>> state = AnnotationModeState(region_type="environment")
    >>> state.default_name()
    'arena'
    >>> state.cycle_region_type()
    >>> state.region_type
    'hole'
    >>> state.default_name()
    ''

    """

    region_type: RegionType
    environment_count: int = field(default=0)
    hole_count: int = field(default=0)
    region_count: int = field(default=0)
    wall_count: int = field(default=0)
    landmark_count: int = field(default=0)

    def cycle_region_type(self) -> None:
        """Cycle to the next annotation region type.

        Transition order: environment → hole → region → wall → landmark →
        environment
        """
        order = REGION_TYPE_CATEGORIES
        self.region_type = order[(order.index(self.region_type) + 1) % len(order)]

    def default_name(self) -> str:
        """Get the default name for the current region type.

        Returns
        -------
        str
            Default name based on current region type:
            - "arena" for environment
            - "" for every other type (auto-named on creation)

        """
        if self.region_type == "environment":
            return "arena"
        # Holes and regions use empty string - auto-named on shape creation
        return ""

    def generate_auto_name(self, existing_names: list[str]) -> str:
        """Generate an auto-incremented name for the current region type.

        Used when user draws a shape without providing a name.

        Parameters
        ----------
        existing_names : list of str
            Names already in use (for uniqueness check).

        Returns
        -------
        str
            Auto-generated unique name like "hole_1", "wall_2", etc.

        """
        if self.region_type == "environment":
            base = "arena"
        else:
            base = f"{self.region_type}_{self._count(self.region_type) + 1}"

        return make_unique_name(base, existing_names)

    def _count(self, region_type: RegionType) -> int:
        return int(getattr(self, f"{region_type}_count"))

    def record_shape_added(self, region_type: RegionType) -> None:
        """Update counts after a shape is added.

        Parameters
        ----------
        region_type : RegionType
            The region type of the shape that was added.

        """
        name = f"{region_type}_count"
        setattr(self, name, getattr(self, name) + 1)

    def sync_counts_from_region_types(self, region_types: list[RegionType]) -> None:
        """Synchronize counts from an existing list of region types.

        Useful when initializing state from existing annotation data.

        Parameters
        ----------
        region_types : list of RegionType
            List of region types for existing shapes.

        """
        for kind in REGION_TYPE_CATEGORIES:
            setattr(self, f"{kind}_count", sum(1 for r in region_types if r == kind))

    def status_text(self) -> str:
        """Generate annotation status text for display.

        Returns
        -------
        str
            Human-readable status like
            "Annotations: 1 environment, 0 holes, 2 regions, 1 walls, 0 landmarks"

        """
        return (
            f"Annotations: {self.environment_count} environment, "
            f"{self.hole_count} holes, {self.region_count} regions, "
            f"{self.wall_count} walls, {self.landmark_count} landmarks"
        )

    @property
    def has_environment(self) -> bool:
        """Whether an environment boundary has been drawn."""
        return self.environment_count > 0


def make_unique_name(base_name: str, existing_names: list[str]) -> str:
    """Generate a unique name by appending a suffix if needed.

    Parameters
    ----------
    base_name : str
        The desired name.
    existing_names : list of str
        Names that are already in use.

    Returns
    -------
    str
        A unique name (base_name if available, or base_name_N if not).

    Examples
    --------
    >>> make_unique_name("arena", [])
    'arena'
    >>> make_unique_name("arena", ["arena"])
    'arena_2'
    >>> make_unique_name("arena", ["arena", "arena_2"])
    'arena_3'

    """
    if base_name not in existing_names:
        return base_name

    # Find next available suffix
    counter = 2
    while f"{base_name}_{counter}" in existing_names:
        counter += 1
    return f"{base_name}_{counter}"
