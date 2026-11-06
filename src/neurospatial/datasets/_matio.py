"""``mat-io`` without its side effect on pandas."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def keeping_pandas_options() -> Iterator[None]:
    """Leave pandas' options as they were after importing a reader.

    Importing ``matio`` sets ``pd.options.future.infer_string = True`` for the
    whole process, which would change the dtype of every string column the
    caller builds after loading a dataset. The first import of ``matio`` must
    run inside this block, wherever it happens.
    """
    import pandas as pd

    current = pd.options.future.infer_string
    with pd.option_context("future.infer_string", current):
        yield


def load_mat(path: Path | str) -> Any:
    """``matio.load_from_mat``, leaving pandas' options as they were.

    Parameters
    ----------
    path : pathlib.Path or str
        The MATLAB file.

    Returns
    -------
    dict
        The file's variables.
    """
    with keeping_pandas_options():
        import matio

    return matio.load_from_mat(str(path))
