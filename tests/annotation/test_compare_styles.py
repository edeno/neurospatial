"""Scientific invariants of the reproducible annotation comparison recipe."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest

from neurospatial.annotation import (
    RegionGrowProvider,
    SegmentationPrompt,
    SegmentationProposal,
)

_EXAMPLE = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "examples/compare_annotation_styles.py")
)


def test_mask_metrics_distinguish_connectivity_and_exterior_background():
    metrics = _EXAMPLE["mask_metrics"](np.ones((4, 4), dtype=bool))
    assert metrics["area_px"] == 16
    assert metrics["holes_background_4"] == 0
    assert metrics["holes_background_8"] == 0

    mask = np.ones((3, 3), dtype=bool)
    mask[0, 0] = mask[1, 1] = False
    metrics = _EXAMPLE["mask_metrics"](mask)
    assert metrics["holes_background_4"] == 1
    assert metrics["holes_background_8"] == 0

    metrics = _EXAMPLE["mask_metrics"](np.eye(3, dtype=bool))
    assert metrics["components_foreground_4"] == 3
    assert metrics["components_foreground_8"] == 1


def test_factorial_conversion_reports_fragment_and_hole_loss_without_mutating_raw():
    pytest.importorskip("cv2")
    mask = np.zeros((60, 70), dtype=bool)
    mask[5:40, 5:40] = True
    mask[15:25, 15:25] = False
    mask[45:55, 50:60] = True
    proposal = SegmentationProposal(mask, metadata={"source": "test"})
    prompt = SegmentationPrompt(np.array([[10, 10]]), np.array([1]))
    records, outlines = _EXAMPLE["compare_polygons"](proposal, prompt)

    assert records["preserve"]["error"] is None
    assert records["cvat"]["error"] is None
    assert records["preserve"]["cleanup"]["removed_components"] == 1
    assert records["preserve"]["polygon"]["holes"] == 1
    assert records["preserve"]["polygon"]["components"] == 1
    assert records["cvat"]["polygon"]["holes"] == 0
    assert records["cvat"]["polygon"]["components"] == 1
    assert outlines["cvat"].proposal.mask.sum() == mask.sum()
    assert outlines["preserve"].proposal.mask.sum() == mask.sum() - 100
    np.testing.assert_array_equal(proposal.mask, mask)
    assert proposal.metadata == {"source": "test"}


def test_degenerate_cvat_panel_is_reported_and_other_panel_survives():
    pytest.importorskip("cv2")
    mask = np.zeros((6, 6), dtype=bool)
    mask[3, 3] = True
    prompt = SegmentationPrompt(np.array([[3, 3]]), np.array([1]))
    records, outlines = _EXAMPLE["compare_polygons"](SegmentationProposal(mask), prompt)
    assert records["preserve"]["error"] is None
    assert records["preserve"]["polygon"]["area_px2"] > 0
    assert records["preserve"]["conversion_mask"]["area_px"] == 1
    assert "degenerate" in records["cvat"]["error"]
    assert set(outlines) == {"preserve"}


def test_cpu_comparison_reuses_one_prediction():
    class CountingRegionGrow(RegionGrowProvider):
        calls = 0

        def predict(self, image, prompt):
            self.calls += 1
            return super().predict(image, prompt)

    provider = CountingRegionGrow(tolerance=0.1)
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    prompt = SegmentationPrompt(np.array([[2, 2]]), np.array([1]))
    raw, timings = _EXAMPLE["_predict"](image, prompt, provider)
    assert provider.calls == 1
    assert raw["refinement"] is raw["single_mask"]
    assert timings["single_mask"]["reused_from"] == "refinement"
