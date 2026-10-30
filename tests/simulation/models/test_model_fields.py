"""A model's fields are its constructor arguments; ``ground_truth`` holds the
values it fires with.

``dataclasses.replace`` and pickling must then build exactly the model that
constructing with the same arguments builds: a default derived from another
parameter, a unit or a seed is derived again, never carried over.
"""

import dataclasses
import pickle

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.simulation.models import (
    BorderCellModel,
    BoundaryVectorCellModel,
    GridCellModel,
    HeadDirectionCellModel,
    ObjectVectorCellModel,
    PlaceCellModel,
    SpatialViewCellModel,
)


@pytest.fixture(scope="module")
def box_cm():
    """A walled 100 x 100 cm box."""
    return Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0, units="cm")


@pytest.fixture(scope="module")
def box_m():
    """The same box in metres."""
    return Environment.from_polygon(box(0, 0, 1, 1), bin_size=0.02, units="m")


@pytest.fixture(scope="module")
def samples():
    rng = np.random.default_rng(5)
    n = 30
    return (
        np.arange(n) / 30.0,
        rng.uniform(0.05, 0.95, (n, 2)),
        rng.uniform(-np.pi, np.pi, n),
    )


def _defaulted_arguments(env_cm):
    """Per cell type, the model class and arguments that leave every derivable
    parameter to its default."""
    return {
        "place": (PlaceCellModel, {"env": env_cm, "seed": 1}),
        "border": (BorderCellModel, {"env": env_cm}),
        "boundary_vector": (BoundaryVectorCellModel, {"env": env_cm, "seed": 1}),
        "grid": (GridCellModel, {"env": env_cm}),
        "head_direction": (HeadDirectionCellModel, {"seed": 1}),
        "object_vector": (
            ObjectVectorCellModel,
            {
                "env": env_cm,
                "object_positions": [[30.0, 30.0]],
                "preferred_direction": 0.3,
            },
        ),
        "spatial_view": (
            SpatialViewCellModel,
            {"env": env_cm, "preferred_view_location": [100.0, 50.0]},
        ),
    }


def _defaulted_model(env_cm, cell_type):
    cls, arguments = _defaulted_arguments(env_cm)[cell_type]
    return cls(**arguments)


# Per cell type: changes that replace() must apply as a fresh constructor
# would. "env" is replaced by the metre box, with coordinates rescaled.
_CHANGES = {
    "place": [{"seed": 2}, {"max_rate": 5.0}, {"env": None}],
    "border": [{"preferred_distance": 3.0}, {"env": None}],
    "boundary_vector": [
        {"preferred_distance": 40.0},
        {"seed": 2},
        {"direction_tolerance": 1.0},
        {"env": None},
    ],
    "grid": [{"grid_orientation": 0.3}, {"env": None}],
    "head_direction": [{"seed": 2}, {"concentration": 2.0}],
    "object_vector": [
        {"preferred_distance": 5.0},
        {"env": None, "object_positions": [[0.3, 0.3]]},
    ],
    "spatial_view": [
        {"max_rate": 5.0},
        {"env": None, "preferred_view_location": [1.0, 0.5]},
    ],
}

# Fields of each type whose default is derived from another parameter, the
# environment's units or a seed.
_DEFAULTED_FIELDS = {
    "place": ["center", "width"],
    "border": ["distance_tolerance"],
    "boundary_vector": [
        "preferred_distance",
        "preferred_direction",
        "distance_tolerance",
    ],
    "grid": ["grid_spacing", "phase_offset"],
    "head_direction": ["preferred_direction"],
    "object_vector": ["preferred_distance", "distance_tolerance"],
    "spatial_view": ["view_field_width"],
}

_CASES = [
    (cell_type, change) for cell_type, changes in _CHANGES.items() for change in changes
]


def _arguments(model):
    """The model's constructor arguments: its init fields."""
    return {
        field.name: getattr(model, field.name)
        for field in dataclasses.fields(model)
        if field.init
    }


def _rates(model, samples, env):
    times, unit_positions, headings = samples
    scale = 100.0 if env is not None and env.units == "cm" else 1.0
    return model.firing_rate(times, unit_positions * scale, headings=headings)


@pytest.mark.parametrize(
    ("cell_type", "change"), _CASES, ids=[f"{c}-{sorted(d)}" for c, d in _CASES]
)
def test_replace_builds_the_model_the_constructor_builds(
    box_cm, box_m, samples, cell_type, change
):
    cls, arguments = _defaulted_arguments(box_cm)[cell_type]
    change = {k: (box_m if k == "env" else v) for k, v in change.items()}
    replaced = dataclasses.replace(cls(**arguments), **change)
    fresh = cls(**{**arguments, **change})

    np.testing.assert_equal(replaced.ground_truth, fresh.ground_truth)
    env = getattr(fresh, "env", None)
    np.testing.assert_array_equal(
        _rates(replaced, samples, env), _rates(fresh, samples, env)
    )


@pytest.mark.parametrize("cell_type", sorted(_DEFAULTED_FIELDS))
def test_pickle_and_fields_rebuild_the_model(box_cm, samples, cell_type):
    model = _defaulted_model(box_cm, cell_type)
    for copy in (pickle.loads(pickle.dumps(model)), type(model)(**_arguments(model))):
        np.testing.assert_equal(copy.ground_truth, model.ground_truth)
        np.testing.assert_array_equal(
            _rates(copy, samples, box_cm), _rates(model, samples, box_cm)
        )


@pytest.mark.parametrize("cell_type", sorted(_DEFAULTED_FIELDS))
def test_defaulted_fields_stay_none_and_ground_truth_resolves_them(box_cm, cell_type):
    model = _defaulted_model(box_cm, cell_type)
    for name in _DEFAULTED_FIELDS[cell_type]:
        assert getattr(model, name) is None, name
        resolved = np.asarray(model.ground_truth[name])
        assert np.all(np.isfinite(resolved)), name


def test_boundary_vector_tolerance_follows_a_replaced_distance(box_cm):
    """The default tolerance is d / 12 + 8 cm of the current distance."""
    model = BoundaryVectorCellModel(box_cm, preferred_direction=0.0)
    replaced = dataclasses.replace(model, preferred_distance=40.0)
    assert replaced.ground_truth["distance_tolerance"] == pytest.approx(40 / 12 + 8)


def test_head_direction_redraws_with_a_replaced_seed():
    model = HeadDirectionCellModel(seed=1)
    replaced = dataclasses.replace(model, seed=2)
    assert (
        replaced.ground_truth["preferred_direction"]
        == HeadDirectionCellModel(seed=2).ground_truth["preferred_direction"]
    )
    assert (
        dataclasses.replace(model, max_rate=5.0).ground_truth["preferred_direction"]
        == model.ground_truth["preferred_direction"]
    )


def test_place_width_follows_a_replaced_environment(box_cm, box_m):
    """The default 11.1 cm width is 0.111 in a metre environment."""
    model = PlaceCellModel(box_cm, center=[50.0, 50.0])
    replaced = dataclasses.replace(model, env=box_m, center=[0.5, 0.5])
    assert replaced.ground_truth["width"] == pytest.approx(0.111)


# Per cell type: the parameter drawn from the seed and a change unrelated to it.
_DRAWN = {
    "place": ("center", {"max_rate": 30.0}),
    "boundary_vector": ("preferred_direction", {"max_rate": 30.0}),
    "head_direction": ("preferred_direction", {"max_rate": 30.0}),
}


def _unseeded_arguments(box_cm, cell_type):
    cls, arguments = _defaulted_arguments(box_cm)[cell_type]
    return cls, {name: value for name, value in arguments.items() if name != "seed"}


@pytest.mark.parametrize("cell_type", sorted(_DRAWN))
def test_an_unseeded_draw_survives_replace_and_pickle(box_cm, cell_type):
    """With seed=None the model keeps the seed it drew with, as an int."""
    drawn, unrelated = _DRAWN[cell_type]
    cls, arguments = _unseeded_arguments(box_cm, cell_type)
    model = cls(**arguments)
    assert isinstance(model.seed, int)
    value = model.ground_truth[drawn]
    for copy in (
        dataclasses.replace(model, **unrelated),
        pickle.loads(pickle.dumps(model)),
        cls(**_arguments(model)),
    ):
        np.testing.assert_array_equal(copy.ground_truth[drawn], value)


@pytest.mark.parametrize("cell_type", sorted(_DRAWN))
def test_an_unseeded_model_redraws_with_a_replaced_seed(box_cm, cell_type):
    drawn, _ = _DRAWN[cell_type]
    cls, arguments = _unseeded_arguments(box_cm, cell_type)
    model = cls(**arguments)
    for seed in (1, 2):
        replaced = dataclasses.replace(model, seed=seed)
        assert replaced.seed == seed
        np.testing.assert_array_equal(
            replaced.ground_truth[drawn],
            cls(**arguments, seed=seed).ground_truth[drawn],
        )


def test_a_seed_is_kept_as_passed_and_drawn_only_for_a_draw(box_cm):
    assert PlaceCellModel(box_cm, seed=7).seed == 7
    assert PlaceCellModel(box_cm, center=[50.0, 50.0]).seed is None
