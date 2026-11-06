"""Reading a MATLAB file leaves the reader's pandas settings alone."""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import numpy as np
import pytest
import scipy.io as sio

PROBE = """
import sys

import pandas as pd

pd.options.future.infer_string = {setting}
from neurospatial.datasets._matio import load_mat

assert load_mat(sys.argv[1])["x"].ravel().tolist() == [1.0, 2.0]
assert pd.options.future.infer_string is {setting}
"""


@pytest.mark.skipif(
    importlib.util.find_spec("matio") is None, reason="needs the datasets extra"
)
@pytest.mark.parametrize("setting", [False, True])
def test_load_mat_keeps_pandas_string_inference(tmp_path, setting):
    """``import matio`` turns the option on; each setting survives a first read.

    A subprocess, since the import acts once and only on a fresh interpreter.
    """
    path = tmp_path / "small.mat"
    sio.savemat(path, {"x": np.array([1.0, 2.0])})
    result = subprocess.run(
        [sys.executable, "-c", PROBE.format(setting=setting), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
