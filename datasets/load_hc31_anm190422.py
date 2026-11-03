"""Rich, Liaw & Lee rat CA1 units on a 48 m linear track, repeat exposure (CRCNS hc-31, animal ANM00190422).

Not implemented yet: ``DATASET`` and ``EXPECTATIONS`` are registered before
the first load.
"""

from __future__ import annotations

from typing import Any

from _record import Range

DATASET = {
    "key": "hc31_anm190422",
    "title": "Rich, Liaw & Lee (2014), rat ANM00190422, CA1 units on a 48 m track, second run (CRCNS hc-31)",
    "source": "https://doi.org/10.6080/K0BV7DT2",
    "license": "CRCNS terms of use (account required; cite Rich et al. 2014 or the data set)",
    "download_size": "166.8 MB",
}

# Pre-registered from the hc-31 description and resSave_readme.txt (48 m
# track; posMazeLin is the position along it; run2 is the repeat exposure;
# interneurons and poor units have ID > 2) and the field listing of
# resSave(5) (63 units; posMaze 'run2' 'linear'), before the first run of this
# loader. The sampling rate is a guess (Neuralynx video at 25-30 Hz).
EXPECTATIONS: dict[str, Any] = {
    "n_dims": 1,
    "n_units": Range(20, 63),
    "sampling_rate_hz": Range(24.0, 31.0),
    "position_min": Range(-50.0, 100.0),
    "position_max": Range(4000.0, 5000.0),
}
