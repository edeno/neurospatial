"""Export the Octave run's sequences as the reference tables pinned by the realdata tests.

usage: python export_reference.py <work-dir> <out-dir>

Writes ``authors_octave_sequences.csv`` (session, side, start_s, end_s: the
eligible sequences of getEligibleSequences.m, full float64 precision) and
``authors_octave_sessions.json`` (per session and side: the decoding units,
the raw sequence count of DecSeqDetectZ.m, the eligible count).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def main(work: Path, out: Path) -> None:
    rows = []
    sessions = {}
    for result in sorted(work.glob("sessions/*/files/octave_result.mat")):
        session = result.parent.parent.name
        o = loadmat(result, squeeze_me=True)
        eligible = np.atleast_2d(o["eligible"]).reshape(-1, 3)
        info = {}
        for code, side in ((1, "left"), (2, "right")):
            mine = eligible[eligible[:, 2] == code, :2]
            mine = mine[np.argsort(mine[:, 0])]
            rows += [(session, side, repr(float(a)), repr(float(b))) for a, b in mine]
            raw = np.atleast_2d(o[f"raw_{side}"]).reshape(-1, 2)
            info[side] = {
                "units": sorted(s.strip() for s in np.atleast_1d(o[f"units_{side}"])),
                "raw_sequences": len(raw),
                "eligible_sequences": len(mine),
            }
        sessions[session] = info
    out.mkdir(parents=True, exist_ok=True)
    with (out / "authors_octave_sequences.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["session", "side", "start_s", "end_s"])
        w.writerows(rows)
    (out / "authors_octave_sessions.json").write_text(
        json.dumps(sessions, indent=1) + "\n"
    )
    print(f"{len(rows)} sequences over {len(sessions)} sessions -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
