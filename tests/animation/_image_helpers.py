"""Shared image helpers for animation backend tests."""

from __future__ import annotations

import io

import numpy as np
import pytest


def decode_png(image_bytes: bytes) -> np.ndarray:
    """Decode PNG/JPEG bytes to an (H, W, 3) uint8 RGB array."""
    pil = pytest.importorskip("PIL.Image")
    return np.array(pil.open(io.BytesIO(image_bytes)).convert("RGB"))
