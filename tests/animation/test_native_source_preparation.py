"""Native setup must serialize lazy reads before allocating GUI resources."""

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation import AnimationField
from neurospatial.animation.backends import fastplotlib_backend as backend


def test_native_field_statistics_sample_inside_shared_lock(monkeypatch):
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0),) * 2
    )
    times = np.array([0.0, 0.1, 0.2])
    reading = False
    reads = []

    @contextmanager
    def source_lock():
        nonlocal reading
        assert not reading
        reading = True
        try:
            yield
        finally:
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
    field = AnimationField(Source(), times)
    with pytest.raises(BeforeAllocationError):
        backend.FastplotlibAnimation(
            env,
            {"field": np.ones((len(times), env.n_bins))},
            frame_times=times,
            _field_streams=(field,),
            _source_lock=source_lock(),
        )
    assert reads == [0, 1, 2]
