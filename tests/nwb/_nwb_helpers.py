"""Shared helpers for NWB disk round-trip tests."""

from __future__ import annotations

from typing import Any


def create_roundtrip_nwb() -> Any:
    """Create a minimal NWBFile for on-disk round-trip tests."""
    from datetime import datetime
    from uuid import uuid4

    from pynwb import NWBFile

    return NWBFile(
        session_description="Disk round-trip test session",
        identifier=str(uuid4()),
        session_start_time=datetime.now().astimezone(),
    )
