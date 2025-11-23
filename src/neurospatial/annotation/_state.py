"""Pure state objects for annotation mode tracking.

This module contains napari-independent state management, making annotation
logic testable without a GUI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from neurospatial.annotation._types import Role


@dataclass
class AnnotationModeState:
    """
    Pure state object tracking annotation mode and shape counts.

    Encapsulates:
    - Current annotation role (environment/hole/region)
    - Per-role shape counts for auto-naming
    - Default name generation logic

    This class is deliberately napari-independent so state transitions
    and naming rules can be tested without a GUI.

    Parameters
    ----------
    role : Role
        Current annotation mode.

    Attributes
    ----------
    environment_count : int
        Number of environment shapes created.
    hole_count : int
        Number of hole shapes created.
    region_count : int
        Number of region shapes created.

    Examples
    --------
    >>> state = AnnotationModeState(role="environment")
    >>> state.default_name()
    'arena'
    >>> state.cycle_role()
    >>> state.role
    'hole'
    >>> state.default_name()
    ''
    """

    role: Role
    environment_count: int = field(default=0)
    hole_count: int = field(default=0)
    region_count: int = field(default=0)

    def cycle_role(self) -> None:
        """
        Cycle to the next annotation role.

        Transition order: environment → hole → region → environment
        """
        if self.role == "environment":
            self.role = "hole"
        elif self.role == "hole":
            self.role = "region"
        else:
            self.role = "environment"

    def default_name(self) -> str:
        """
        Get the default name for the current role.

        Returns
        -------
        str
            Default name based on current role:
            - "arena" for environment
            - "" for hole (auto-named on creation)
            - "" for region (user should provide name)
        """
        if self.role == "environment":
            return "arena"
        # Holes and regions use empty string - auto-named on shape creation
        return ""

    def generate_auto_name(self, existing_names: list[str]) -> str:
        """
        Generate an auto-incremented name for the current role.

        Used when user draws a shape without providing a name.

        Parameters
        ----------
        existing_names : list of str
            Names already in use (for uniqueness check).

        Returns
        -------
        str
            Auto-generated unique name like "hole_1", "region_2", etc.
        """
        if self.role == "environment":
            base = "arena"
        elif self.role == "hole":
            base = f"hole_{self.hole_count + 1}"
        else:
            base = f"region_{self.region_count + 1}"

        return make_unique_name(base, existing_names)

    def record_shape_added(self, role: Role) -> None:
        """
        Update counts after a shape is added.

        Parameters
        ----------
        role : Role
            The role of the shape that was added.
        """
        if role == "environment":
            self.environment_count += 1
        elif role == "hole":
            self.hole_count += 1
        else:
            self.region_count += 1

    def sync_counts_from_roles(self, roles: list[Role]) -> None:
        """
        Synchronize counts from an existing list of roles.

        Useful when initializing state from existing annotation data.

        Parameters
        ----------
        roles : list of Role
            List of roles for existing shapes.
        """
        self.environment_count = sum(1 for r in roles if r == "environment")
        self.hole_count = sum(1 for r in roles if r == "hole")
        self.region_count = sum(1 for r in roles if r == "region")

    def status_text(self) -> str:
        """
        Generate annotation status text for display.

        Returns
        -------
        str
            Human-readable status like "Annotations: 1 environment, 0 holes, 2 regions"
        """
        return (
            f"Annotations: {self.environment_count} environment, "
            f"{self.hole_count} holes, {self.region_count} regions"
        )

    @property
    def has_environment(self) -> bool:
        """Whether an environment boundary has been drawn."""
        return self.environment_count > 0


def make_unique_name(base_name: str, existing_names: list[str]) -> str:
    """
    Generate a unique name by appending a suffix if needed.

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
