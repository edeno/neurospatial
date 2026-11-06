"""``mat-io`` without its side effect on pandas."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_mat(path: Path | str) -> Any:
    """``matio.load_from_mat``, leaving pandas' options as they were.

    Importing ``matio`` sets ``pd.options.future.infer_string = True`` for the
    whole process, which would change the dtype of every string column the
    caller builds after loading a dataset.

    Parameters
    ----------
    path : pathlib.Path or str
        The MATLAB file.

    Returns
    -------
    dict
        The file's variables.
    """
    import pandas as pd

    infer_string = pd.options.future.infer_string
    import matio

    pd.options.future.infer_string = infer_string
    return matio.load_from_mat(str(path))
