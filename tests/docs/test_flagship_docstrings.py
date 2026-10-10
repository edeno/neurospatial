"""Flagship docstring examples execute as written in isolated directories."""

import doctest
import inspect
import re
import sys

import matplotlib.pyplot as plt
import pytest

from ._flagship import FLAGSHIP, NWB_FLAGSHIP, resolve

_ALLOWED_SKIP = re.compile(r"backend\s*=\s*[\"'](napari|video)[\"']")


def _run_examples(dotted: str) -> None:
    obj = resolve(dotted)
    module = sys.modules[obj.__name__ if inspect.ismodule(obj) else obj.__module__]
    tests = doctest.DocTestFinder(recurse=False).find(
        obj, dotted, globs=dict(vars(module))
    )
    examples = [example for test in tests for example in test.examples]
    assert examples, f"{dotted} has no examples to run"
    skipped = [
        example.source
        for example in examples
        if example.options.get(doctest.SKIP)
        and not _ALLOWED_SKIP.search(example.source)
    ]
    assert not skipped, (
        f"{dotted}: {len(skipped)} disallowed +SKIP examples:\n{skipped[0]}"
    )
    runner = doctest.DocTestRunner(
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    )
    report = []
    try:
        for test in tests:
            runner.run(test, out=report.append)
    finally:
        plt.close("all")
    assert runner.failures == 0, "".join(report)


@pytest.mark.parametrize("dotted", FLAGSHIP)
def test_flagship_docstring_examples_run(dotted, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _run_examples(dotted)


@pytest.mark.nwb
@pytest.mark.parametrize("dotted", NWB_FLAGSHIP)
def test_nwb_flagship_docstring_examples_run(dotted, monkeypatch, tmp_path):
    pytest.importorskip("pynwb")
    monkeypatch.chdir(tmp_path)
    _run_examples(dotted)
