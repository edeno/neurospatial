"""hc-2 takes its units from hc-3's cell table and refuses a table that no longer matches."""

from __future__ import annotations

import numpy as np
import pytest

_WIDEBAND = 20000.0
_N_CLUSTERS = (10, 9, 9, 9)  # 37 clusters from 2 up on the four shanks


@pytest.fixture
def session(tmp_path) -> dict:
    """A synthetic hc-2 session: a walk around a 200-pixel square, 37 clusters.

    The two LEDs sit 3 pixels ahead of and behind the head along the
    direction of movement, so the loader's LED-order check passes.
    """
    rng = np.random.default_rng(0)
    n = 20_000  # frames at 39.06 Hz, about 512 s
    angle = np.cumsum(rng.normal(0.0, 0.15, n))
    step = 1.5 * np.column_stack([np.cos(angle), np.sin(angle)])
    head = np.empty((n, 2))
    head[0] = 110.0
    for k in range(1, n):  # reflect off the square's walls
        head[k] = head[k - 1] + step[k]
        for d in range(2):
            if not 10.0 <= head[k, d] <= 210.0:
                step[k:, d] *= -1
                head[k, d] = 2 * np.clip(head[k, d], 10.0, 210.0) - head[k, d]
    direction = np.gradient(head, axis=0)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    frames = np.hstack([head + 3.0 * direction, head - 3.0 * direction])
    files = {"whl": tmp_path / "s.whl", "xml": tmp_path / "s.xml"}
    np.savetxt(files["whl"], frames, fmt="%.3f")
    files["xml"].write_text(
        "<parameters><acquisitionSystem><samplingRate>20000</samplingRate>"
        "</acquisitionSystem></parameters>"
    )
    n_samples = int(n * 512)
    for shank, n_clusters in enumerate(_N_CLUSTERS, start=1):
        clusters = np.repeat(np.arange(n_clusters) + 2, 50)
        samples = np.sort(rng.integers(0, n_samples, clusters.size))
        files[f"res.{shank}"] = tmp_path / f"s.res.{shank}"
        files[f"clu.{shank}"] = tmp_path / f"s.clu.{shank}"
        np.savetxt(files[f"res.{shank}"], samples, fmt="%d")
        np.savetxt(files[f"clu.{shank}"], np.r_[n_clusters + 2, clusters], fmt="%d")
    # hc-3's table lists hc-2's shank s as electrode s + 4.
    rows = [
        {"ele": str(shank + 4), "clu": str(cluster + 2), "region": "CA1"}
        for shank, n_clusters in enumerate(_N_CLUSTERS, start=1)
        for cluster in range(n_clusters)
    ]
    return {"files": files, "rows": rows}


def _load(monkeypatch, session, rows):
    import load_hc2_ec013_527 as hc2

    monkeypatch.setattr(hc2, "fetch_session", lambda *a, **k: session["files"])
    monkeypatch.setattr(hc2, "cell_table", lambda *a, **k: rows)
    return hc2.load()


def test_hc2_refuses_a_cell_table_matching_fewer_than_36_of_37(monkeypatch, session):
    rows = session["rows"][2:]  # 35 of the 37 clusters are in the table
    with pytest.raises(ValueError, match="35 of the session's 37 clusters"):
        _load(monkeypatch, session, rows)


def test_hc2_keeps_the_units_of_a_table_matching_36_of_37(monkeypatch, session):
    rows = session["rows"][1:]  # shank 1's cluster 2 is not in the table
    rec = _load(monkeypatch, session, rows)
    assert len(rec.spike_times) == 36
    assert "1.2" not in rec.unit_ids
    assert set(rec.unit_ids) == {f"{int(r['ele']) - 4}.{r['clu']}" for r in rows}
