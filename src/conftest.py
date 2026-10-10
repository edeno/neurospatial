"""Run module doctests without leaving output files in the checkout."""

import pytest


@pytest.fixture(autouse=True)
def _doctest_in_tmp_path(request: pytest.FixtureRequest, tmp_path, monkeypatch) -> None:
    """Give each module doctest a temporary working directory."""
    if isinstance(request.node, pytest.DoctestItem):
        monkeypatch.chdir(tmp_path)
