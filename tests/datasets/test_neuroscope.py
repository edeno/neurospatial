"""Readers for NeuroSuite files, on tiny files written in the format."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial.datasets._neuroscope import (
    read_acquisition,
    read_eeg,
    read_res_clu,
    read_whl,
)

XML = """<?xml version = '1.0'?>
<parameters creator="ndManager-1.1.1" version="1.0" >
 <acquisitionSystem>
  <nBits>16</nBits>
  <nChannels>{n_channels}</nChannels>
  <samplingRate>32552</samplingRate>
  <voltageRange>20</voltageRange>
  <amplification>1000</amplification>
  <offset>0</offset>
 </acquisitionSystem>
 <fieldPotentials>
  <lfpSamplingRate>1252</lfpSamplingRate>
 </fieldPotentials>
 <anatomicalDescription>
  <channelGroups>
   <group><channel skip="0" >0</channel><channel skip="0" >1</channel></group>
   <group><channel skip="0" >2</channel><channel skip="0" >3</channel></group>
  </channelGroups>
 </anatomicalDescription>
</parameters>
"""


def write_session(folder, base, *, n_channels=4, eeg=None, shanks=None):
    """A NeuroSuite session: ``.xml``, ``.res.N``/``.clu.N`` and ``.eeg``.

    ``shanks`` maps each shank to (samples, clusters, n_clusters).
    """
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{base}.xml").write_text(XML.format(n_channels=n_channels))
    for shank, (samples, clusters, n_clusters) in (shanks or {}).items():
        (folder / f"{base}.res.{shank}").write_text("".join(f"{s}\n" for s in samples))
        (folder / f"{base}.clu.{shank}").write_text(
            f"{n_clusters}\n" + "".join(f"{c}\n" for c in clusters)
        )
    if eeg is not None:
        np.asarray(eeg, dtype="<i2").tofile(folder / f"{base}.eeg")


def test_acquisition_states_the_rates_scale_and_groups(tmp_path):
    write_session(tmp_path, "s")
    acquisition = read_acquisition(tmp_path / "s.xml")
    assert acquisition.wideband_hz == 32552.0
    assert acquisition.lfp_hz == 1252.0
    assert acquisition.n_channels == 4
    assert acquisition.channel_groups == ((0, 1), (2, 3))
    # 20 V over 16 bits at a gain of 1000: 0.305 µV per step.
    assert acquisition.microvolts_per_bit == pytest.approx(20 / 65536 / 1000 * 1e6)


def test_acquisition_without_a_rate_is_refused(tmp_path):
    (tmp_path / "s.xml").write_text("<parameters><acquisitionSystem/></parameters>")
    with pytest.raises(ValueError, match="samplingRate"):
        read_acquisition(tmp_path / "s.xml")


def test_res_clu_pairs_each_spike_with_its_cluster(tmp_path):
    write_session(tmp_path, "s", shanks={1: ([5, 9, 30], [2, 0, 3], 4)})
    samples, clusters = read_res_clu(tmp_path / "s.res.1", tmp_path / "s.clu.1")
    assert samples.tolist() == [5, 9, 30]
    assert clusters.tolist() == [2, 0, 3]


@pytest.mark.parametrize(
    ("clusters", "n_clusters", "message"),
    [([2, 0], 4, "has 2 spikes"), ([2, 0, 5], 4, "declares 4 clusters")],
)
def test_res_clu_that_disagree_are_refused(tmp_path, clusters, n_clusters, message):
    write_session(tmp_path, "s", shanks={1: ([5, 9, 30], clusters, n_clusters)})
    with pytest.raises(ValueError, match=message):
        read_res_clu(tmp_path / "s.res.1", tmp_path / "s.clu.1")


def test_eeg_reads_chosen_channels_of_interleaved_frames(tmp_path):
    frames = np.arange(12, dtype="<i2").reshape(3, 4)  # 3 samples x 4 channels
    write_session(tmp_path, "s", eeg=frames)
    picked = read_eeg(tmp_path / "s.eeg", 4, (3, 0))
    assert picked.tolist() == [[3, 0], [7, 4], [11, 8]]
    with pytest.raises(ValueError, match="whole number"):
        read_eeg(tmp_path / "s.eeg", 5, (0,))
    with pytest.raises(ValueError, match="outside 0 to 3"):
        read_eeg(tmp_path / "s.eeg", 4, (4,))


def test_whl_keeps_the_frames_as_stored(tmp_path):
    (tmp_path / "s.whl").write_text("1 2 3 4\n-1 -1 5 6\n")
    assert read_whl(tmp_path / "s.whl").tolist() == [[1, 2, 3, 4], [-1, -1, 5, 6]]
    (tmp_path / "t.whl").write_text("1 2 3\n")
    with pytest.raises(ValueError, match="pairs"):
        read_whl(tmp_path / "t.whl")
