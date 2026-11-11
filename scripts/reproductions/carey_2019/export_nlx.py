"""Export a session's .nvt and .nev raw records to .mat for Octave Nlx2Mat shims."""

import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

from neurospatial.datasets import _neuralynx as nlx

src, out = Path(sys.argv[1]), Path(sys.argv[2])
vt = np.fromfile(
    next(src.glob("*.nvt")), dtype=nlx._NVT_RECORD, offset=nlx.HEADER_BYTES
)
n = vt.size
savemat(
    out / "VT_export.mat",
    {
        "Timestamps": vt["timestamp"].astype(np.float64)[None, :],
        "X": vt["x"].astype(np.float64)[None, :],
        "Y": vt["y"].astype(np.float64)[None, :],
        "Angles": vt["angle"].astype(np.float64)[None, :],
        "Targets": np.zeros((1, n)),
        "Points": np.zeros((1, n)),
    },
)
ev = np.fromfile(
    next(src.glob("*Events.nev")), dtype=nlx._NEV_RECORD, offset=nlx.HEADER_BYTES
)
texts = np.array([t.decode("latin-1").strip() for t in ev["text"]], dtype=object)
savemat(
    out / "EV_export.mat",
    {
        "EVTimeStamps": ev["timestamp"].astype(np.float64)[None, :],
        "EventStrings": texts[:, None],
    },
)
print(n, "frames;", ev.size, "events;", sorted(set(texts))[:12])
