"""Native setup must serialize lazy reads before allocating GUI resources."""

from types import SimpleNamespace

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation import Animation, AnimationField
from neurospatial.animation.backends import fastplotlib_backend as backend


def test_native_field_statistics_sample_inside_shared_lock(monkeypatch):
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )
    times = np.array([0.0, 0.1, 0.2])
    reading = False
    reads = []

    class SourceLock:
        """Fails if a read is not serialized, or the lock is taken twice."""

        def __enter__(self):
            nonlocal reading
            assert not reading
            reading = True

        def __exit__(self, *exc):
            nonlocal reading
            reading = False

    class Source:
        shape = (3, env.n_bins)

        def __len__(self):
            return len(times)

        def __getitem__(self, index):
            assert reading, "Sampling a lazy row must hold the shared source lock"
            reads.append(index)
            return np.full(env.n_bins, index + 1.0)

    class BeforeAllocationError(Exception):
        pass

    def allocate_widget(**kwargs):
        assert not reading, "Native allocation must run outside the source lock"
        raise BeforeAllocationError

    monkeypatch.setattr(
        backend,
        "import_fastplotlib",
        lambda: SimpleNamespace(NDWidget=allocate_widget),
    )
    monkeypatch.setattr(backend, "require_animation_support", lambda _: None)
    scene = Animation(env, {"field": AnimationField(Source(), times)})
    scene._source_lock = SourceLock()
    with pytest.raises(BeforeAllocationError):
        scene.show(backend="fastplotlib")
    # The color scale's bounded sample is read once, before allocation.
    assert reads == [0, 1, 2]
