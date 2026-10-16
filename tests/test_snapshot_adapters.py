"""Installed adapters terminate in owned, validated built-in snapshots."""

from __future__ import annotations

import copy
import gc
import json
import weakref
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pytest
from environment_helpers import publish_environment
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.arena import OpenArena
from neurospatial.extensions import (
    SnapshotAdapter,
    _decode_snapshot,
    _snapshot_adapters,
    _snapshot_provider,
)
from neurospatial.geometry._arena import decode_arena
from neurospatial.geometry._binning import (
    Binning,
    MissingFaces,
    SpatialBins,
    TrackLinearization,
)
from neurospatial.geometry._codecs import decode, encode
from neurospatial.geometry._locators import (
    NearestBinLocator,
    SubsetLocator,
    TransformedLocator,
)
from neurospatial.geometry._polygon import ClippedLocator
from neurospatial.ops.transforms import translate


class _ExternalLocator:
    def __init__(self):
        self.centers = np.array([[0.0, 0.0], [2.0, 0.0]])


@dataclass
class _EntryPoint:
    name: str
    adapter: Any
    value: str = "external_package:adapter"

    def load(self):
        return self.adapter


def _adapter(**overrides):
    values = {
        "name": "laboratory.lookup",
        "version": 1,
        "role": "locator",
        "provider_type": _ExternalLocator,
        "snapshot": lambda provider: NearestBinLocator(provider.centers),
        "decode": lambda meta, arrays, prefix: NearestBinLocator(
            arrays[f"{prefix}/centers"]
        ),
    }
    values.update(overrides)
    return SnapshotAdapter(**values)


@pytest.fixture(autouse=True)
def _registry(monkeypatch):
    from neurospatial import extensions

    entries = []
    original_entry_points = extensions.metadata.entry_points

    def installed(*, group):
        if group == "neurospatial.snapshot_adapters.v1":
            return entries
        return original_entry_points(group=group)

    monkeypatch.setattr(extensions.metadata, "entry_points", installed)
    _snapshot_adapters.cache_clear()
    yield entries
    _snapshot_adapters.cache_clear()


def test_locator_snapshot_owns_data_and_saves_as_a_builtin(_registry):
    _registry.append(_EntryPoint("lookup", _adapter()))
    provider = _ExternalLocator()
    snapshot = _snapshot_provider(provider, "locator")
    assert type(snapshot) is NearestBinLocator
    provider.centers[:] = 999
    np.testing.assert_array_equal(snapshot.centers, [[0, 0], [2, 0]])
    assert not snapshot.centers.flags.writeable
    saved, arrays = encode(snapshot, "lookup")
    assert saved["codec"] == "NearestBinLocator"
    _registry.clear()
    _snapshot_adapters.cache_clear()
    restored = decode(saved, arrays, "lookup")
    np.testing.assert_array_equal(
        restored.locate(np.array([[0.0, 0.0], [2.0, 0.0]])), [0, 1]
    )


def test_saved_custom_codec_uses_installed_decoder_without_mutating_payload(_registry):
    def adapter_decode(meta, arrays, prefix):
        meta["edited"] = True
        result = NearestBinLocator(arrays[f"{prefix}/centers"])
        arrays[f"{prefix}/centers"][:] = 100
        return result

    _registry.append(_EntryPoint("lookup", _adapter(decode=adapter_decode)))
    meta = {"codec": "laboratory.lookup", "codec_version": 1}
    arrays = {"old/centers": np.array([[0.0, 0.0], [2.0, 0.0]])}
    before = copy.deepcopy((meta, arrays))
    restored = decode(meta, arrays, "old")
    assert type(restored) is NearestBinLocator
    assert meta == before[0]
    np.testing.assert_array_equal(arrays["old/centers"], before[1]["old/centers"])
    np.testing.assert_array_equal(restored.centers, [[0.0, 0.0], [2.0, 0.0]])


def test_custom_payload_version_two_is_routed_before_builtin_version_checks(_registry):
    _registry.append(_EntryPoint("lookup", _adapter(version=2)))
    result = decode(
        {"codec": "laboratory.lookup", "codec_version": 2},
        {"old/centers": np.array([[0.0, 0.0], [2.0, 0.0]])},
        "old",
    )
    assert type(result) is NearestBinLocator


def test_published_snapshot_does_not_retain_provider_or_returned_array_headers(
    _registry,
):
    value = NearestBinLocator(np.array([[0.0, 0.0], [2.0, 0.0]]))
    _registry.append(_EntryPoint("lookup", _adapter(snapshot=lambda provider: value)))
    provider = _ExternalLocator()
    reference = weakref.ref(provider)
    snapshot = _snapshot_provider(provider, "locator")
    del provider
    assert reference() is None
    assert not np.shares_memory(snapshot.centers, value.centers)
    value.centers.shape = (1, 4)
    assert snapshot.centers.shape == (2, 2)


def test_custom_nested_locator_decodes_and_resaves_as_canonical_builtin(_registry):
    from neurospatial import Environment

    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
    )
    locator = env._binning.locator
    _registry.append(_EntryPoint("lookup", _adapter(decode=lambda m, a, p: locator)))
    meta, arrays = encode(env._binning, "old")
    meta["locator"] = {"codec": "laboratory.lookup", "codec_version": 1}
    restored = decode(meta, arrays, "old")
    np.testing.assert_array_equal(
        restored.locator.locate(env.bin_centers), np.arange(env.n_bins)
    )
    canonical, buffers = encode(restored, "new")
    assert canonical["locator"]["codec"] == "RectilinearLocator"
    _registry.clear()
    _snapshot_adapters.cache_clear()
    assert type(decode(canonical, buffers, "new")) is Binning


def test_wrong_custom_role_cannot_fill_a_builtin_locator_field(_registry):
    from neurospatial import Environment
    from neurospatial.arena import UnknownArena

    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
    )
    _registry.append(
        _EntryPoint(
            "lookup", _adapter(role="arena", decode=lambda m, a, p: UnknownArena(2))
        )
    )
    meta, arrays = encode(env._binning, "old")
    meta["locator"] = {"codec": "laboratory.lookup", "codec_version": 1}
    with pytest.raises(ValueError, match="point-lookup"):
        decode(meta, arrays, "old")


def test_builtin_decode_does_not_discover_or_import_extensions(monkeypatch):
    from neurospatial import extensions

    def fail(**kwargs):
        pytest.fail("built-in records must not load installed adapters")

    monkeypatch.setattr(extensions.metadata, "entry_points", fail)
    meta, arrays = encode(NearestBinLocator(np.array([[0.0, 0.0]])), "builtin")
    assert type(decode(meta, arrays, "builtin")) is NearestBinLocator


@pytest.mark.parametrize("version", [None, True, 1.0, 0, 2, "1"])
def test_unknown_or_malformed_version_refuses_before_calling_decoder(
    _registry, version
):
    def fail(*args):
        pytest.fail("unsupported payload version must not call the decoder")

    _registry.append(_EntryPoint("lookup", _adapter(decode=fail)))
    with pytest.raises(ValueError, match="version"):
        _decode_snapshot(
            {"codec": "laboratory.lookup", "codec_version": version}, {}, "old"
        )


def test_missing_installed_adapter_explains_package_requirement():
    with pytest.raises(ValueError, match=r"install.*package"):
        decode({"codec": "laboratory.lookup", "codec_version": 1}, {}, "old")
    with pytest.raises(TypeError, match="snapshot adapter"):
        _snapshot_provider(_ExternalLocator(), "locator")


@pytest.mark.parametrize("duplicate", ["name", "provider_role"])
def test_duplicate_entries_are_rejected_deterministically(_registry, duplicate):
    first = _adapter()
    second = (
        _adapter(name="laboratory.other")
        if duplicate == "provider_role"
        else _adapter(provider_type=dict)
    )
    for entries in (
        [_EntryPoint("z", first), _EntryPoint("a", second)],
        [_EntryPoint("a", second), _EntryPoint("z", first)],
    ):
        _registry[:] = entries
        _snapshot_adapters.cache_clear()
        with pytest.raises(ValueError, match="duplicate") as caught:
            _snapshot_adapters()
        if entries[0].name == "z":
            original = str(caught.value)
        else:
            assert str(caught.value) == original


@pytest.mark.parametrize("role", ["locator", "track", "arena", "binning"])
def test_each_supported_role_normalizes_a_builtin_record(_registry, role):
    from neurospatial import Environment
    from neurospatial.arena import MaskArena

    if role == "locator":
        value = NearestBinLocator(np.array([[0.0, 0.0], [2.0, 0.0]]))
    elif role == "arena":
        value = MaskArena((np.arange(3.0),), np.array([True, False]))
    else:
        env = Environment.linear_track(endpoints=[(0, 0), (4, 0)], bin_size=1)
        value = env._binning.linearization if role == "track" else env._binning
        assert type(value) is (TrackLinearization if role == "track" else Binning)
    _registry.append(
        _EntryPoint("adapter", _adapter(role=role, snapshot=lambda p: value))
    )
    snapshot = _snapshot_provider(_ExternalLocator(), role)
    assert type(snapshot) is type(value)
    assert snapshot is not value


@pytest.mark.parametrize("result", [object(), _ExternalLocator(), None])
def test_adapter_cannot_preserve_a_custom_runtime_object(_registry, result):
    _registry.append(_EntryPoint("lookup", _adapter(snapshot=lambda provider: result)))
    with pytest.raises(ValueError, match=r"built-in.*locator"):
        _snapshot_provider(_ExternalLocator(), "locator")


def test_wrong_declared_role_is_not_accepted(_registry):
    _registry.append(_EntryPoint("lookup", _adapter(role="arena")))
    with pytest.raises(ValueError, match=r"built-in.*arena"):
        _snapshot_provider(_ExternalLocator(), "arena")


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", ""),
        ("version", True),
        ("version", 0),
        ("role", "faces"),
        ("provider_type", object()),
        ("snapshot", None),
        ("decode", None),
    ],
)
def test_adapter_declaration_is_validated(field, value):
    with pytest.raises((TypeError, ValueError)):
        _adapter(**{field: value})


def test_entrypoint_requires_an_adapter_object_and_cannot_shadow_builtins(_registry):
    _registry.append(_EntryPoint("wrong", lambda: _adapter()))
    with pytest.raises(TypeError, match="SnapshotAdapter"):
        _snapshot_adapters()
    _registry[:] = [_EntryPoint("shadow", _adapter(name="Binning"))]
    with pytest.raises(ValueError, match="built-in"):
        _snapshot_adapters()


class _ExternalBinning:
    def __init__(self, data):
        self.data = data


class _ExternalLayout(_ExternalBinning):
    is_linearized_track = False
    _layout_type_tag = "external"

    def __init__(self, data):
        super().__init__(data)
        self._build_params_used = {"callback": lambda: self}

    def __deepcopy__(self, memo):
        pytest.fail("an external provider must be adapted before copying")

    def build(self, **kwargs):
        pytest.fail("publication must not execute an external layout builder")


class _ExternalWrappedLocator(_ExternalBinning):
    def locate(self, points):
        pytest.fail("publication must retain built-in lookup data")


class _ExternalTrack(_ExternalBinning):
    def to_linear(self, positions):
        pytest.fail("publication must retain built-in track data")

    def linear_to_nd(self, positions):
        pytest.fail("publication must retain built-in track data")


class _ExternalArena:
    n_dims = 2
    coordinate_kind = "cartesian"
    domain_known = True
    physical_footprint_known = True
    walls = ()

    def contains(self, positions):
        pytest.fail("an external arena must be adapted before center selection")

    def polygonal_domain(self):
        pytest.fail("publication must retain built-in arena data")

    def transformed(self, matrix):
        pytest.fail("publication must retain built-in arena data")


def _data_adapter(provider_type, role, *, name=None, decode=None):
    return _adapter(
        name=name or f"laboratory.{role}",
        role=role,
        provider_type=provider_type,
        snapshot=lambda provider: provider.data,
        decode=decode or (lambda meta, arrays, prefix: pytest.fail("unused decoder")),
    )


def _explicit_binning(env):
    meta, arrays = encode(env._binning, "native")
    return decode(meta, arrays, "native")


@pytest.mark.parametrize("ingress", ["constructor", "from_bins"])
def test_public_binning_adapter_owns_data_and_drops_provider_provenance(
    _registry, ingress
):
    original = Environment.linear_track(endpoints=[(0, 0), (4, 0)], bin_size=1)
    data = _explicit_binning(original)
    provider_type = _ExternalLayout if ingress == "constructor" else _ExternalBinning
    _registry.append(_EntryPoint("binning", _data_adapter(provider_type, "binning")))
    provider = provider_type(data)
    reference = weakref.ref(provider)
    if ingress == "constructor":
        env = publish_environment("adapted", provider, "external")
        assert env.layout_parameters == {}
    else:
        env = Environment.from_bins(provider, name="adapted", units="cm", frame="lab")
        assert (env.units, env.frame) == ("cm", "lab")
    assert env.is_linearized_track == env.is_linearized_track
    assert env.is_linearized_track
    np.testing.assert_allclose(
        env.to_linear(np.array([[0.25, 0], [3.5, 0]])), [0.25, 3.5]
    )
    np.testing.assert_allclose(env.linear_to_nd([0.25, 3.5]), [[0.25, 0], [3.5, 0]])
    data.bins.centers.shape = (1, data.bins.centers.size)
    assert env.bin_centers.shape == (4, 2)
    del provider
    gc.collect()
    assert reference() is None
    np.testing.assert_array_equal(env.bin_at(env.bin_centers), np.arange(4))


def _externalize_locator(locator):
    if type(locator) in (SubsetLocator, TransformedLocator, ClippedLocator):
        base, provider = _externalize_locator(locator.base)
        return replace(locator, base=base), provider
    provider = _ExternalWrappedLocator(locator)
    return provider, provider


def test_public_locator_ingress_detaches_caller_buffers_and_provider(_registry):
    provider = _ExternalLocator()
    source = Binning(
        "sample_sites",
        SpatialBins.from_arrays(provider.centers, [1.0, 1.0], 2, [[0, 1]], [2.0]),
        provider,
        MissingFaces("No physical faces are available."),
    )
    _registry.append(
        _EntryPoint(
            "lookup",
            _adapter(
                snapshot=lambda value: NearestBinLocator(
                    value.centers, max_distance=0.1
                )
            ),
        )
    )
    env = Environment.from_bins(source)
    provider.centers[:] = 999
    np.testing.assert_array_equal(env.bin_at([[0, 0], [2, 0], [10, 0]]), [0, 1, -1])
    np.testing.assert_array_equal(env.bin_centers, [[0, 0], [2, 0]])
    reference = weakref.ref(provider)
    del source, provider
    gc.collect()
    assert reference() is None


def test_public_adapter_result_still_requires_cross_component_consistency(_registry):
    native = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
    )
    data = _explicit_binning(native)
    inconsistent = replace(
        data,
        bins=SpatialBins.from_arrays(
            data.bins.centers + 0.1,
            data.bins.sizes,
            data.bins.measure_dim,
            data.bins.pairs,
            data.bins.pair_distances,
        ),
    )
    _registry.append(_EntryPoint("binning", _data_adapter(_ExternalBinning, "binning")))
    with pytest.raises(ValueError, match="centers"):
        Environment.from_bins(_ExternalBinning(inconsistent))


@pytest.mark.parametrize("operation", ["clip", "subset", "transform", "nested"])
def test_public_ingress_adapts_nested_locators_without_changing_maps(
    _registry, operation
):
    env = Environment.from_grid_mask(
        np.ones((3, 2), dtype=bool), (np.arange(4.0), np.arange(3.0))
    ).clip(box(0.2, 0.2, 2.8, 1.8))
    if operation in ("subset", "nested"):
        env = env.subset(bins=np.array([True, False, True, True, False, True]))
    if operation in ("transform", "nested"):
        env = env.apply_transform(translate(10, -3))
    original = _explicit_binning(env)
    locator, provider = _externalize_locator(original.locator)
    source = replace(original, locator=locator)
    _registry.append(
        _EntryPoint("lookup", _data_adapter(_ExternalWrappedLocator, "locator"))
    )
    queries = np.vstack((env.bin_centers, [[999.0, 999.0]]))
    expected = env.bin_at(queries)
    restored = Environment.from_bins(source, arena=env.arena)
    np.testing.assert_array_equal(restored.bin_at(queries), expected)
    np.testing.assert_array_equal(restored.bin_sizes, env.bin_sizes)
    np.testing.assert_array_equal(
        restored.clip_mapping.parent_bin_indices, env.clip_mapping.parent_bin_indices
    )
    assert restored.clip_mapping.n_parent_bins == env.clip_mapping.n_parent_bins
    provider.data.edges[0].shape = (1, -1)
    np.testing.assert_array_equal(restored.bin_at(queries), expected)
    reference = weakref.ref(provider)
    del source, locator, provider
    gc.collect()
    assert reference() is None


def test_public_track_adapter_drops_delegated_conversions(_registry):
    original = Environment.linear_track(endpoints=[(0, 0), (4, 0)], bin_size=1)
    data = _explicit_binning(original)
    provider = _ExternalTrack(data.linearization)
    source = replace(data, linearization=provider)
    _registry.append(_EntryPoint("track", _data_adapter(_ExternalTrack, "track")))
    env = Environment.from_bins(source)
    assert env.is_linearized_track
    np.testing.assert_allclose(
        env.to_linear(np.array([[0.25, 0], [3.5, 0]])), [0.25, 3.5]
    )
    np.testing.assert_allclose(env.linear_to_nd([0.25, 3.5]), [[0.25, 0], [3.5, 0]])
    provider.data.grid_edges[0].shape = (1, -1)
    np.testing.assert_allclose(env.to_linear(np.array([[0.25, 0]])), [0.25])
    reference = weakref.ref(provider)
    del source, provider
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("ingress", ["layout_factory", "from_bins", "mask_factory"])
def test_public_arena_adapter_normalizes_before_lookup_and_selection(
    _registry, ingress
):
    _registry.append(
        _EntryPoint(
            "arena",
            _adapter(
                name="laboratory.domain",
                role="arena",
                provider_type=_ExternalArena,
                snapshot=lambda provider: OpenArena(box(0, 0, 1, 2)),
            ),
        )
    )
    provider = _ExternalArena()
    reference = weakref.ref(provider)
    shape = (2, 2) if ingress == "mask_factory" else (1, 2)
    native = Environment.from_grid_mask(
        np.ones(shape, dtype=bool), (np.arange(shape[0] + 1.0), np.arange(3.0))
    )
    if ingress == "layout_factory":
        env = Environment.from_layout(
            "MaskedGrid",
            {"active_mask": native.active_mask, "grid_edges": native.grid_edges},
            name="adapted",
            arena=provider,
        )
    elif ingress == "from_bins":
        env = Environment.from_bins(_explicit_binning(native), arena=provider)
    else:
        env = Environment.from_grid_mask(
            np.ones(shape, dtype=bool),
            (np.arange(3.0), np.arange(3.0)),
            arena=provider,
        )
    assert type(env.arena) is OpenArena
    assert env.n_bins == 2
    np.testing.assert_array_equal(env.bin_at([[0.5, 0.5], [1.5, 0.5]]), [0, -1])
    del provider
    gc.collect()
    assert reference() is None


def test_custom_arena_decoder_receives_local_array_namespace(_registry):
    def read_domain(meta, arrays, prefix):
        assert prefix == ""
        assert "bounds" in arrays and "arena/bounds" not in arrays
        result = OpenArena(box(*arrays["bounds"]))
        arrays["bounds"][:] = 999
        return result

    _registry.append(
        _EntryPoint(
            "arena",
            _adapter(
                name="laboratory.domain", version=2, role="arena", decode=read_domain
            ),
        )
    )
    bounds = np.array([0.0, 0.0, 1.0, 2.0])
    arena = decode_arena(
        {"codec": "laboratory.domain", "codec_version": 2}, {"bounds": bounds}
    )
    assert type(arena) is OpenArena
    np.testing.assert_array_equal(bounds, [0, 0, 1, 2])
    np.testing.assert_array_equal(arena.contains([[0.5, 1], [1.5, 1]]), [True, False])


@pytest.mark.parametrize("custom_role", ["locator", "arena", "both"])
def test_custom_file_and_dict_load_resave_without_adapter_package(
    _registry, tmp_path, custom_role
):
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
    )
    locator = env._binning.locator
    _registry.extend(
        [
            _EntryPoint("lookup", _adapter(decode=lambda m, a, p: locator)),
            _EntryPoint(
                "arena",
                _adapter(
                    name="laboratory.domain",
                    version=2,
                    role="arena",
                    decode=lambda meta, arrays, prefix: OpenArena(
                        box(*arrays["bounds"])
                    ),
                ),
            ),
        ]
    )
    original = env.to_dict()
    if custom_role in ("locator", "both"):
        original["binning"]["locator"] = {
            "codec": "laboratory.lookup",
            "codec_version": 1,
        }
    if custom_role in ("arena", "both"):
        original["arena"] = {"codec": "laboratory.domain", "codec_version": 2}
        original["arrays"] = {
            key: value
            for key, value in original["arrays"].items()
            if not key.startswith("arena/")
        }
        original["arrays"]["arena/bounds"] = [0.0, 0.0, 2.0, 2.0]
        original["array_specs"] = {
            key: spec
            for key, spec in original["array_specs"].items()
            if not key.startswith("arena/")
        }
        original["array_specs"]["arena/bounds"] = {"shape": [4], "dtype": "<f8"}
    saved_arrays = original.pop("arrays")
    custom_path = tmp_path / "custom.json"
    custom_path.write_text(json.dumps(original))
    np.savez_compressed(
        custom_path.with_suffix(".npz"),
        **{
            key: np.asarray(value, dtype=original["array_specs"][key]["dtype"])
            for key, value in saved_arrays.items()
        },
    )
    dictionary = {**original, "arrays": saved_arrays}
    restored_values = [
        Environment.from_dict(dictionary),
        Environment.from_file(custom_path),
    ]
    canonical_dicts = [value.to_dict() for value in restored_values]
    canonical_paths = [tmp_path / f"canonical_{index}.json" for index in range(2)]
    for restored, path in zip(restored_values, canonical_paths, strict=True):
        restored.to_file(path)
    for canonical in canonical_dicts:
        assert canonical["binning"]["locator"]["codec"] == "RectilinearLocator"
        assert canonical["arena"]["codec"] == (
            "MaskArena" if custom_role == "locator" else "OpenArena"
        )
        assert "laboratory." not in json.dumps(canonical)
    _registry.clear()
    _snapshot_adapters.cache_clear()
    with pytest.raises(ValueError, match=r"install.*package"):
        Environment.from_dict(dictionary)
    with pytest.raises(ValueError, match=r"install.*package"):
        Environment.from_file(custom_path)
    for canonical in canonical_dicts:
        restored = Environment.from_dict(canonical)
        np.testing.assert_array_equal(restored.bin_at(env.bin_centers), np.arange(4))
    for path in canonical_paths:
        restored = Environment.from_file(path)
        np.testing.assert_array_equal(restored.bin_at(env.bin_centers), np.arange(4))


def test_public_contract_kit_accepts_an_adapter_published_environment(_registry):
    from neurospatial.testing import assert_environment_contract

    centers = np.array([[0.0, 0.0], [2.0, 0.0]])
    data = Binning(
        "sample_sites",
        SpatialBins.from_arrays(centers, [1.0, 1.0], 2, [[0, 1]], [2.0]),
        NearestBinLocator(centers, max_distance=0.1),
        MissingFaces("No physical faces are available."),
    )
    _registry.append(_EntryPoint("binning", _data_adapter(_ExternalBinning, "binning")))
    env = Environment.from_bins(_ExternalBinning(data))
    assert_environment_contract(env, [[0, 0], [2, 0], [10, 0]], [0, 1, -1])
