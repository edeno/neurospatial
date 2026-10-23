"""Scientific support, pixel coordinates, and provider adapter contracts."""

from __future__ import annotations

import builtins
import importlib
import inspect
import pickle
import runpy
import sys
from copy import deepcopy
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import Point
from shapely.ops import unary_union

from neurospatial.annotation.segmentation import (
    RegionGrowProvider,
    SegmentAnythingProvider,
    SegmentationPrompt,
    SegmentationProposal,
    mask_to_polygons,
)


def _prompt(points=((1, 1),), labels=(1,), box=None):
    return SegmentationPrompt(
        np.asarray(points).reshape(-1, 2), np.asarray(labels), box=box
    )


def test_pixel_cells_preserve_holes_components_bounds_and_area():
    mask = np.zeros((9, 13), dtype=bool)
    mask[1:7, 2:8] = True
    mask[2:6, 3:7] = False
    mask[8, 12] = True
    polygons = mask_to_polygons(mask)
    assert len(polygons) == 2
    assert len(polygons[0].interiors) == 1
    assert polygons[0].bounds == (1.5, 0.5, 7.5, 6.5)
    assert polygons[1].bounds == (11.5, 7.5, 12.5, 8.5)
    assert sum(p.area for p in polygons) == mask.sum()
    union = unary_union(polygons)
    for y, x in np.ndindex(mask.shape):
        assert union.covers(Point(x, y)) == mask[y, x]
    assert [p.wkb for p in polygons] == [p.wkb for p in mask_to_polygons(mask)]


def test_pixel_geometry_diagonal_components_and_single_pixel():
    polygons = mask_to_polygons(np.eye(3, dtype=bool))
    assert len(polygons) == 3
    assert all(p.is_valid and p.area == 1 for p in polygons)
    assert polygons[0].bounds == (-0.5, -0.5, 0.5, 0.5)
    assert mask_to_polygons(np.zeros((3, 4), dtype=bool)) == []


def test_simplification_preserves_every_component_and_hole():
    mask = np.zeros((30, 40), dtype=bool)
    mask[1:25, 1:25] = True
    mask[8:18, 8:18] = False
    mask[28, 38] = True
    simplified = mask_to_polygons(mask, simplify_tolerance=5)
    assert len(simplified) == 2
    assert sum(len(p.interiors) for p in simplified) == 1
    assert all(p.is_valid and not p.is_empty for p in simplified)


def test_simplified_components_never_overlap_and_match_the_outline_preview():
    from neurospatial.annotation._polygon_outline import (
        polygon_outline,
        simplify_polygon_outline,
    )

    # Simplifying these components one at a time makes neighbors overlap.
    mask = np.array(
        [
            [0, 1, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 0],
        ],
        dtype=bool,
    )
    simplified = mask_to_polygons(mask, simplify_tolerance=1.0)
    assert len(simplified) == len(mask_to_polygons(mask))
    assert all(
        a.intersection(b).area == 0
        for i, a in enumerate(simplified)
        for b in simplified[i + 1 :]
    )
    preview = simplify_polygon_outline(polygon_outline(SegmentationProposal(mask)), 1.0)
    assert sorted(p.wkt for p in simplified) == sorted(p.wkt for p in preview.polygons)


@pytest.mark.parametrize("tolerance", [-1, np.nan, np.inf])
def test_mask_polygon_tolerance_validation(tolerance):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        mask_to_polygons(np.ones((2, 3), dtype=bool), simplify_tolerance=tolerance)


@pytest.mark.parametrize(
    "mask", [np.ones((2, 3)), np.ones(3, dtype=bool), np.zeros((0, 2), dtype=bool)]
)
def test_mask_polygon_input_validation(mask):
    with pytest.raises(ValueError, match="2D boolean"):
        mask_to_polygons(mask)


def test_prompt_ownership_and_frozen_arrays():
    points = np.array([[7.0, 2.0], [3.0, 1.0]])
    labels = np.array([1, 0])
    bounds = np.array([1.0, 0.0, 8.0, 5.0])
    prompt = SegmentationPrompt(points, labels, box=bounds)
    points[:] = 0
    labels[:] = 0
    bounds[:] = 0
    np.testing.assert_array_equal(prompt.points, [[7, 2], [3, 1]])
    np.testing.assert_array_equal(prompt.labels, [1, 0])
    np.testing.assert_array_equal(prompt.box, [1, 0, 8, 5])
    for restored in [prompt, deepcopy(prompt), pickle.loads(pickle.dumps(prompt))]:
        assert not restored.points.flags.writeable
        assert not restored.labels.flags.writeable
        assert not restored.box.flags.writeable
    with pytest.raises(FrozenInstanceError):
        prompt.points = points


@pytest.mark.parametrize(
    ("points", "labels", "bounds"),
    [
        ([[1, 2, 3]], [1], None),
        ([[np.nan, 2]], [1], None),
        ([[1, 2]], [1, 0], None),
        ([[1, 2]], [0.5], None),
        ([[1, 2]], [2], None),
        ([[1, 2]], [1], [0, 2, 3]),
        ([[1, 2]], [1], [0, 2, 3, 2]),
        ([[1, 2]], [1], [0, 0, np.inf, 2]),
    ],
)
def test_prompt_rejects_malformed_data(points, labels, bounds):
    with pytest.raises(ValueError, match="Prompt"):
        SegmentationPrompt(np.asarray(points), np.asarray(labels), box=bounds)


def test_proposal_owns_mask_and_nested_json_metadata():
    mask = np.array([[False, True]])
    metadata = {"settings": {"seeds": [[7, 2]], "tolerance": 0.2}}
    proposal = SegmentationProposal(mask, score=0.9, metadata=metadata)
    mask[:] = False
    metadata["settings"]["seeds"][0][0] = 99
    exported = proposal.metadata
    exported["settings"]["seeds"][0][0] = 100
    assert proposal.metadata["settings"]["seeds"] == [[7, 2]]
    np.testing.assert_array_equal(proposal.mask, [[False, True]])
    for restored in [
        proposal,
        deepcopy(proposal),
        pickle.loads(pickle.dumps(proposal)),
    ]:
        assert not restored.mask.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        proposal.mask[0, 0] = True
    with pytest.raises(FrozenInstanceError):
        proposal.score = 1


@pytest.mark.parametrize(
    ("mask", "score", "metadata"),
    [
        (np.array([[1]]), None, None),
        (np.array([[False]]), None, None),
        (np.array([[True]]), np.nan, None),
        (np.array([[True]]), np.inf, None),
        (np.array([[True]]), None, {"bad": np.nan}),
        (np.array([[True]]), None, {"bad": np.array([1])}),
        (np.array([[True]]), None, {1: "not a string"}),
    ],
)
def test_proposal_rejects_invalid_outputs(mask, score, metadata):
    with pytest.raises(ValueError, match=r"mask|foreground|score|metadata"):
        SegmentationProposal(mask, score=score, metadata=metadata)


@pytest.mark.parametrize(
    "metadata",
    [
        {"settings": {1: "numeric key", "1": "string key"}},
        {"settings": [{False: "boolean key"}]},
        {"settings": ({"nested": [{2.5: "numeric key"}]},)},
    ],
)
def test_proposal_rejects_nested_keys_that_json_would_coerce(metadata):
    with pytest.raises(ValueError, match="JSON-compatible provenance"):
        SegmentationProposal(np.array([[True]]), metadata=metadata)


@pytest.mark.parametrize(
    ("constructor", "option"),
    [
        (SegmentationPrompt, "box"),
        (SegmentationProposal, "score"),
        (SegmentationProposal, "metadata"),
    ],
)
def test_segmentation_options_are_keyword_only(constructor, option):
    assert (
        inspect.signature(constructor).parameters[option].kind
        is inspect.Parameter.KEYWORD_ONLY
    )


def test_region_grow_xy_seed_and_disconnected_color_support():
    image = np.zeros((5, 12, 3), dtype=np.uint8)
    image[1:4, 7:10] = [240, 120, 30]
    image[1:3, 1:3] = [240, 120, 30]
    proposal = RegionGrowProvider(tolerance=0).predict(image, _prompt(((8, 2),)))
    expected = np.zeros((5, 12), dtype=bool)
    expected[1:4, 7:10] = True
    np.testing.assert_array_equal(proposal.mask, expected)
    assert proposal.metadata["provider"] == "region_grow"
    both = RegionGrowProvider(tolerance=0).predict(
        image, _prompt(((8, 2), (1, 1)), (1, 1))
    )
    assert both.mask.sum() == 13


def test_region_grow_negative_colors_remove_leaking_foreground():
    image = np.full((5, 9, 3), [100, 100, 100], dtype=np.uint8)
    image[:, 5:] = [110, 110, 110]
    foreground = _prompt(((2, 2),))
    provider = RegionGrowProvider(tolerance=0.1)
    assert provider.predict(image, foreground).mask.all()
    constrained = provider.predict(image, _prompt(((2, 2), (7, 2)), (1, 0)))
    assert constrained.mask[:, :5].all()
    assert not constrained.mask[:, 5:].any()


def test_region_grow_tolerance_is_normalized_rgb_distance():
    image = np.array([[[0, 0, 0], [25, 25, 25], [30, 30, 30]]], dtype=np.uint8)
    proposal = RegionGrowProvider(tolerance=0.1).predict(image, _prompt(((0, 0),)))
    np.testing.assert_array_equal(proposal.mask, [[True, True, False]])


def test_region_grow_box_clips_pixel_centers():
    image = np.zeros((6, 11, 3), dtype=np.uint8)
    proposal = RegionGrowProvider().predict(image, _prompt(((7, 2),), box=[5, 1, 8, 3]))
    expected = np.zeros((6, 11), dtype=bool)
    expected[1:4, 5:9] = True
    np.testing.assert_array_equal(proposal.mask, expected)


@pytest.mark.parametrize("tolerance", [-0.1, 1.1, np.nan, np.inf])
def test_region_grow_tolerance_validation(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        RegionGrowProvider(tolerance=tolerance)


@pytest.mark.parametrize(
    "image",
    [np.zeros((3, 4)), np.zeros((3, 4, 3)), np.zeros((3, 4, 4), dtype=np.uint8)],
)
def test_provider_image_validation(image):
    with pytest.raises(ValueError, match="RGB uint8"):
        RegionGrowProvider().predict(image, _prompt())


@pytest.mark.parametrize("points", [((-1, 2),), ((12, 2),), ((2, 5),)])
def test_region_grow_rejects_points_outside_image(points):
    with pytest.raises(ValueError, match=r"outside.*pixel"):
        RegionGrowProvider().predict(
            np.zeros((5, 12, 3), dtype=np.uint8), _prompt(points)
        )


def test_region_grow_missing_or_contradictory_foreground_fails_actionably():
    image = np.zeros((5, 9, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="foreground click"):
        RegionGrowProvider().predict(image, _prompt(labels=(0,)))
    with pytest.raises(ValueError, match="no foreground"):
        RegionGrowProvider().predict(image, _prompt(((2, 2), (3, 3)), (1, 0)))
    with pytest.raises(ValueError, match="no foreground"):
        RegionGrowProvider().predict(image, _prompt(((1, 1),), box=[3, 3, 4, 4]))
    with pytest.raises(ValueError, match="box lies outside"):
        RegionGrowProvider().predict(image, _prompt(box=[20, 20, 30, 30]))


class _Predictor:
    def __init__(self):
        self.images = []
        self.prompts = []
        self.image = None
        self.output = None
        self.fail_set = False
        self.fail_predict = False

    def set_image(self, image):
        self.image = image.copy()
        self.images.append(image.copy())
        if self.fail_set:
            raise RuntimeError("embedding failure")

    def predict(self, **kwargs):
        self.prompts.append(kwargs)
        if self.fail_predict:
            raise RuntimeError("decoder failure")
        if self.output is not None:
            return self.output
        # Depend on the embedded image to exercise meaningful cache state.
        mask = self.image[:, :, 0] > 0
        return np.array([~mask, mask]), np.array([0.2, 0.9]), np.ones((2, 2, 2))


def test_sam_prediction_selects_best_and_passes_xy_prompt():
    predictor = _Predictor()
    image = np.zeros((5, 12, 3), dtype=np.uint8)
    image[1:3, 7:10] = 255
    prompt = _prompt(((8, 2), (3, 1)), (1, 0), [5, 0, 11, 4])
    proposal = SegmentAnythingProvider(predictor).predict(image, prompt)
    np.testing.assert_array_equal(proposal.mask, image[:, :, 0] > 0)
    assert proposal.score == 0.9
    assert proposal.metadata == {
        "provider": "segment_anything",
        "selected_mask": 1,
        "refinement_steps": 0,
    }
    passed = predictor.prompts[0]
    np.testing.assert_array_equal(passed["point_coords"], [[8, 2], [3, 1]])
    np.testing.assert_array_equal(passed["point_labels"], [1, 0])
    np.testing.assert_array_equal(passed["box"], [5, 0, 11, 4])
    assert not np.shares_memory(passed["point_coords"], prompt.points)


@pytest.mark.parametrize(
    "points,labels,box,multimask",
    [
        # Meta's guidance: only a lone click is ambiguous enough for candidates.
        (((7, 2),), (1,), None, True),
        (((7, 2),), (1,), [5, 0, 11, 4], False),
        (((7, 2), (1, 1)), (1, 0), None, False),
        ((), (), [5, 0, 11, 4], False),
    ],
)
def test_sam_requests_candidate_masks_only_for_a_single_click(
    points, labels, box, multimask
):
    predictor = _Predictor()
    image = np.ones((5, 12, 3), dtype=np.uint8)
    SegmentAnythingProvider(predictor).predict(image, _prompt(points, labels, box))
    assert predictor.prompts[-1]["multimask_output"] is multimask


def test_sam_embedding_cache_content_and_shape_invalidation():
    predictor = _Predictor()
    provider = SegmentAnythingProvider(predictor)
    image = np.zeros((5, 12, 3), dtype=np.uint8)
    image[1, 7] = 255
    prompt = _prompt(((7, 1),))
    first = provider.predict(image, prompt)
    second = provider.predict(image.copy(), _prompt(((8, 1),)))
    assert len(predictor.images) == 1
    np.testing.assert_array_equal(first.mask, second.mask)
    image[3, 9] = 255
    third = provider.predict(image, prompt)
    assert len(predictor.images) == 2
    assert third.mask[3, 9] and not first.mask[3, 9]
    resized = np.full((4, 10, 3), 255, dtype=np.uint8)
    assert provider.predict(resized, prompt).mask.shape == (4, 10)
    assert len(predictor.images) == 3


def test_sam_refinement_uses_selected_raw_logits_and_keeps_direct_calls_independent():
    predictor = _Predictor()
    image = np.ones((5, 12, 3), dtype=np.uint8)
    logits = np.array([[[-4.5, 0.25], [2.5, -1.5]], [[1.25, -3.5], [0.5, 4.5]]])
    predictor.output = (np.ones((2, 5, 12)), np.array([0.2, 0.9]), logits)
    provider = SegmentAnythingProvider(predictor)
    initial = provider._predict_with_refinement(image, _prompt(((7, 2),)))
    assert initial.steps == 0
    assert predictor.prompts[-1]["multimask_output"] is True
    assert "mask_input" not in predictor.prompts[-1]
    np.testing.assert_array_equal(initial.logits, logits[1:2])
    assert not initial.logits.flags.writeable
    assert not initial.image.flags.writeable
    # Mutating a predictor-owned output must not change saved refinement logits.
    logits[:] = 17
    expected = np.array([[[1.25, -3.5], [0.5, 4.5]]], dtype=np.float32)
    second = provider._predict_with_refinement(
        image, _prompt(((7, 2), (1, 1)), (1, 0)), initial
    )
    np.testing.assert_array_equal(predictor.prompts[-1]["mask_input"], expected)
    assert not np.shares_memory(predictor.prompts[-1]["mask_input"], initial.logits)
    assert predictor.prompts[-1]["multimask_output"] is False
    assert second.proposal.metadata["refinement_steps"] == 1
    # Inside clicks are grouped first by the UI; adding one shifts outside clicks.
    prompt = _prompt(((7, 2), (9, 3), (1, 1)), (1, 1, 0))
    third = provider._predict_with_refinement(image, prompt, second)
    np.testing.assert_array_equal(predictor.prompts[-1]["mask_input"], second.logits)
    assert third.steps == 2
    assert len(predictor.images) == 1
    # Identical updates are idempotent, including their refinement count.
    assert provider._predict_with_refinement(image.copy(), prompt, third) is third
    assert len(predictor.prompts) == 3
    direct = provider.predict(image, prompt)
    assert direct.metadata["refinement_steps"] == 0
    assert "mask_input" not in predictor.prompts[-1]


@pytest.mark.parametrize(
    "change", ["move", "remove", "label", "box", "image", "provider"]
)
def test_sam_refinement_restarts_when_its_context_changes(change):
    predictor = _Predictor()
    provider = SegmentAnythingProvider(predictor)
    image = np.ones((5, 12, 3), dtype=np.uint8)
    points, labels, bounds = [(7, 2), (1, 1)], [1, 0], [0, 0, 11, 4]
    previous = provider._predict_with_refinement(image, _prompt(points, labels, bounds))
    if change == "move":
        points[0] = (8, 2)
    elif change == "remove":
        points, labels = points[:1], labels[:1]
    elif change == "label":
        labels[1] = 1
    elif change == "box":
        bounds = [1, 0, 11, 4]
    elif change == "image":
        image[0, 0] = 255
    else:
        predictor = _Predictor()
        provider = SegmentAnythingProvider(predictor)
    # Even adding another point must not carry stale context into the new request.
    points.append((9, 3))
    labels.append(1)
    result = provider._predict_with_refinement(
        image, _prompt(points, labels, bounds), previous
    )
    assert "mask_input" not in predictor.prompts[-1]
    assert result.steps == 0
    assert previous.image[0, 0, 0] == 1


def test_sam_failed_refinement_does_not_mutate_previous_state():
    predictor = _Predictor()
    provider = SegmentAnythingProvider(predictor)
    image = np.ones((5, 12, 3), dtype=np.uint8)
    previous = provider._predict_with_refinement(image, _prompt(((7, 2),)))
    logits = previous.logits.copy()
    predictor.fail_predict = True
    with pytest.raises(RuntimeError, match="SAM segmentation failed"):
        provider._predict_with_refinement(
            image, _prompt(((7, 2), (8, 2)), (1, 1)), previous
        )
    assert previous.steps == 0
    np.testing.assert_array_equal(previous.logits, logits)


@pytest.mark.parametrize("stage", ["set", "predict"])
def test_sam_failure_invalidates_cached_embedding(stage):
    predictor = _Predictor()
    provider = SegmentAnythingProvider(predictor)
    image = np.full((3, 4, 3), 255, dtype=np.uint8)
    provider.predict(image, _prompt())
    if stage == "set":
        image[0, 0] = 0
        predictor.fail_set = True
    else:
        predictor.fail_predict = True
    with pytest.raises(RuntimeError, match="SAM segmentation failed"):
        provider.predict(image, _prompt())
    predictor.fail_set = predictor.fail_predict = False
    provider.predict(image, _prompt())
    assert len(predictor.images) == (3 if stage == "set" else 2)


@pytest.mark.parametrize(
    "output",
    [
        (np.ones((1, 3, 4)), np.ones(1)),
        (np.ones((1, 4, 3)), np.ones(1), np.ones((1, 2, 2))),
        (np.ones((1, 3, 4)), np.ones(2), np.ones((1, 2, 2))),
        (np.ones((1, 3, 4)), np.array([np.nan]), np.ones((1, 2, 2))),
        (np.full((1, 3, 4), np.nan), np.ones(1), np.ones((1, 2, 2))),
        (np.full((1, 3, 4), 0.5), np.ones(1), np.ones((1, 2, 2))),
        (np.ones((1, 3, 4)), np.ones(1), np.full((1, 2, 2), np.inf)),
        (np.ones((1, 3, 4)), np.ones(1), np.ones((2, 2))),
        (np.ones((1, 3, 4)), np.ones(1), np.ones((2, 2, 2))),
        (np.zeros((1, 3, 4)), np.ones(1), np.ones((1, 2, 2))),
    ],
)
def test_sam_rejects_malformed_or_empty_outputs(output):
    predictor = _Predictor()
    predictor.output = output
    with pytest.raises(ValueError, match=r"SAM predictor|no foreground"):
        SegmentAnythingProvider(predictor).predict(
            np.zeros((3, 4, 3), dtype=np.uint8), _prompt()
        )


def test_sam_supports_box_only_and_checks_predictor_contract():
    predictor = _Predictor()
    provider = SegmentAnythingProvider(predictor)
    provider.predict(np.ones((3, 4, 3), dtype=np.uint8), _prompt((), (), [0, 0, 3, 2]))
    assert predictor.prompts[0]["point_coords"] is None
    assert predictor.prompts[0]["point_labels"] is None
    with pytest.raises(ValueError, match="clicks or a bounding box"):
        provider.predict(np.zeros((3, 4, 3), dtype=np.uint8), _prompt((), ()))
    with pytest.raises(TypeError, match="set_image"):
        SegmentAnythingProvider(object())


def test_module_import_does_not_import_optional_model_or_viewer_libraries(monkeypatch):
    original = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "sam2", "napari", "qtpy"}:
            raise AssertionError(f"Optional import {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    runpy.run_path(sys.modules["neurospatial.annotation.segmentation"].__file__)


def test_from_sam2_checks_local_checkpoint_before_model_import(tmp_path, monkeypatch):
    imports = []
    monkeypatch.setattr(importlib, "import_module", lambda name: imports.append(name))
    with pytest.raises(FileNotFoundError, match="not downloaded automatically"):
        SegmentAnythingProvider.from_sam2("config.yaml", tmp_path / "absent.pt")
    assert not imports


def test_from_sam2_missing_dependency_is_actionable(tmp_path, monkeypatch):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"fake weights")

    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(
        ImportError, match=r"PyTorch and Meta SAM 2.*RegionGrowProvider"
    ):
        SegmentAnythingProvider.from_sam2("config.yaml", checkpoint)


def test_from_sam2_loads_local_model_on_requested_device(tmp_path, monkeypatch):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"fake weights")
    calls = []
    predictor = _Predictor()
    model = object()

    def build(config, weights, *, device):
        calls.append((config, weights, device))
        return model

    def make_predictor(received):
        assert received is model
        return predictor

    modules = {
        "torch": _fake_torch(cuda=False, mps=False),
        "sam2.build_sam": SimpleNamespace(build_sam2=build),
        "sam2.sam2_image_predictor": SimpleNamespace(SAM2ImagePredictor=make_predictor),
    }
    monkeypatch.setattr(importlib, "import_module", modules.__getitem__)
    provider = SegmentAnythingProvider.from_sam2(
        "configs/sam2.1/model.yaml", checkpoint, device="cpu"
    )
    assert calls == [("configs/sam2.1/model.yaml", str(checkpoint), "cpu")]
    assert provider.predict(np.ones((3, 4, 3), dtype=np.uint8), _prompt()).mask.all()


def _fake_torch(*, cuda, mps):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


@pytest.mark.parametrize(
    "cuda,mps,expected",
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_from_sam2_selects_the_fastest_available_device(
    tmp_path, monkeypatch, cuda, mps, expected
):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"fake weights")
    devices = []

    def build(config, weights, *, device):
        devices.append(device)
        return object()

    modules = {
        "torch": _fake_torch(cuda=cuda, mps=mps),
        "sam2.build_sam": SimpleNamespace(build_sam2=build),
        "sam2.sam2_image_predictor": SimpleNamespace(
            SAM2ImagePredictor=lambda model: _Predictor()
        ),
    }
    monkeypatch.setattr(importlib, "import_module", modules.__getitem__)
    SegmentAnythingProvider.from_sam2("configs/sam2.1/model.yaml", checkpoint)
    assert devices == [expected]


def test_from_sam2_incompatible_weights_error_has_context(tmp_path, monkeypatch):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"fake weights")

    def fail(*args, **kwargs):
        raise ValueError("wrong state dict")

    modules = {
        "torch": _fake_torch(cuda=False, mps=False),
        "sam2.build_sam": SimpleNamespace(build_sam2=fail),
        "sam2.sam2_image_predictor": SimpleNamespace(SAM2ImagePredictor=_Predictor),
    }
    monkeypatch.setattr(importlib, "import_module", modules.__getitem__)
    with pytest.raises(
        RuntimeError,
        match=r"Could not load.*config.yaml.*weights.pt.*cpu.*wrong state dict",
    ):
        SegmentAnythingProvider.from_sam2("config.yaml", checkpoint)
