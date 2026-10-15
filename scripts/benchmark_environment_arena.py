"""Measure Stage A construction and lookup costs on fixed seeded inputs.

Run this script with the same isolated interpreter and ``PYTHONPATH`` pointed
at each checkout's ``src``. It prints median wall times for five runs and
separate Python-tracked peak allocations; it makes no timing assertions.
The three lookup batches each contain one million identical query points.
"""

import json
import statistics
import time
import tracemalloc

import numpy as np
from shapely.geometry import Point

from neurospatial import Environment

rng = np.random.default_rng(20261014)
positions = rng.uniform(0, 40, (20000, 2))
queries = rng.uniform(-5, 45, (1000000, 2))
factories = {
    "samples": lambda: Environment.from_samples(positions, bin_size=1.0),
    "mask": lambda: Environment.from_grid_mask(
        np.ones((40, 40), bool), grid_edges=(np.arange(41.0), np.arange(41.0))
    ),
    "polygon": lambda: Environment.from_polygon(Point(20, 20).buffer(20), bin_size=1.0),
}
results = {}
for name, factory in factories.items():
    factory()
    construction = []
    for _ in range(5):
        start = time.perf_counter()
        env = factory()
        construction.append(time.perf_counter() - start)
    env.bin_at(queries[:1000])
    lookup = []
    for _ in range(5):
        start = time.perf_counter()
        assigned = env.bin_at(queries)
        lookup.append(time.perf_counter() - start)
    tracemalloc.start()
    env = factory()
    _, construction_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tracemalloc.start()
    env.bin_at(queries)
    _, lookup_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results[name] = {
        "n_bins": env.n_bins,
        "construction_median_s": statistics.median(construction),
        "lookup_median_s": statistics.median(lookup),
        "construction_peak_bytes": construction_peak,
        "lookup_peak_bytes": lookup_peak,
        "n_assigned": int(np.count_nonzero(assigned >= 0)),
    }
print(json.dumps(results, indent=2))
