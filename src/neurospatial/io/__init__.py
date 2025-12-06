"""
Input/Output utilities.

This module provides tools for saving and loading environments,
regions, and neural data in various formats.

Submodules
----------
files : File I/O (to_file, from_file, to_dict, from_dict)
nwb : NeurodataWithoutBorders integration
"""

from neurospatial.io.files import from_dict, from_file, to_dict, to_file

__all__ = [
    "from_dict",
    "from_file",
    "to_dict",
    "to_file",
]
