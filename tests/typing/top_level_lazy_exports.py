"""Static contract for concrete lazy root exports.

Run with ``uv run mypy tests/typing/top_level_lazy_exports.py``. The assertions
fail when a root export resolves through untyped ``__getattr__`` instead of
its concrete definition, while the guarded block leaves runtime imports lazy.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import assert_type

    import neurospatial
    from neurospatial import (
        compute_spatial_rate,
        compute_spatial_rates,
        decode_position,
        load_session,
        peri_event_histogram,
    )
    from neurospatial.decoding._result import DecodingResult as ConcreteDecodingResult
    from neurospatial.encoding.spatial import (
        SpatialRateResult as ConcreteSpatialRateResult,
    )
    from neurospatial.encoding.spatial import (
        SpatialRatesResult as ConcreteSpatialRatesResult,
    )
    from neurospatial.events._core import PeriEventResult as ConcretePeriEventResult
    from neurospatial.recording import Session as ConcreteSession

    assert_type(neurospatial.SpatialRateResult, type[ConcreteSpatialRateResult])
    assert_type(neurospatial.SpatialRatesResult, type[ConcreteSpatialRatesResult])
    assert_type(neurospatial.DecodingResult, type[ConcreteDecodingResult])
    assert_type(neurospatial.PeriEventResult, type[ConcretePeriEventResult])
    assert_type(neurospatial.Session, type[ConcreteSession])

    times = np.arange(10, dtype=np.float64)
    positions = np.zeros((10, 2), dtype=np.float64)
    spikes = np.array([1.0, 2.0], dtype=np.float64)
    env = neurospatial.Environment.from_samples(positions, bin_size=1.0)
    counts = np.zeros((1, 2), dtype=np.int64)
    maps = np.ones((2, env.n_bins), dtype=np.float64)
    events = np.array([1.0, 2.0], dtype=np.float64)

    assert_type(
        compute_spatial_rate(env, spikes, times, positions), ConcreteSpatialRateResult
    )
    assert_type(
        compute_spatial_rates(env, [spikes, spikes], times, positions),
        ConcreteSpatialRatesResult,
    )
    assert_type(decode_position(env, counts, maps, 0.025), ConcreteDecodingResult)
    assert_type(
        peri_event_histogram(spikes, events, (-0.5, 0.5)), ConcretePeriEventResult
    )
    assert_type(load_session("session.nwb"), ConcreteSession)
