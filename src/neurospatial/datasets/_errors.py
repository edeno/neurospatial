"""The error a dataset session raises when its automatic checks refuse it."""

from __future__ import annotations

from collections.abc import Mapping

from neurospatial._exceptions import NeurospatialError, _format_error

__all__ = ["REFUSING_CHECKS", "DatasetCheckError", "DownloadWarning"]

#: The checks whose failure refuses a load unless it is accepted with
#: ``accept_failed_checks``.
REFUSING_CHECKS = ("clock", "units")


class DownloadWarning(UserWarning):
    """Issued before a session's file is downloaded: its size, URL and cache.

    A session can be several GB, so the download is announced where it is
    shown by default. Silence it once a download is expected, for example::

        import warnings
        from neurospatial.datasets import DownloadWarning

        warnings.simplefilter("ignore", DownloadWarning)
    """


class DatasetCheckError(ValueError, NeurospatialError):
    """Raised when a loaded session fails a check that refuses the load.

    Two checks refuse, because a failure means the arrays themselves would
    be wrong: ``"clock"`` (spikes, events or epochs disagree with the
    tracking clock) and ``"units"`` (the position span or sampling rate
    disagrees with the dataset's stated apparatus and rate). The message
    lists each failed check with its numbers and what would fix it, and
    names the keyword that loads the session anyway, for example
    ``accept_failed_checks=("clock",)``. An accepted failure stays with the
    recording in ``rec.failed_checks`` and heads its "Watch out" list.

    Inherits from :class:`ValueError`, so ``except ValueError`` also catches
    it.

    Parameters
    ----------
    dataset, session : str
        The session that failed.
    failures : Mapping[str, str]
        Each failed check's name and its finding: what failed, with numbers,
        and what would fix it.

    Attributes
    ----------
    dataset, session : str
        The session that failed.
    failures : dict of str to str
        Each failed check's name and finding, in the order checked.
    checks : tuple of str
        The failed checks' names, the value to pass as
        ``accept_failed_checks`` to load anyway.

    Examples
    --------
    >>> from neurospatial.datasets import DatasetCheckError
    >>> error = DatasetCheckError(
    ...     "demo_2026", "s1", {"units": "largest position span 1000 cm ..."}
    ... )
    >>> error.checks
    ('units',)
    """

    def __init__(self, dataset: str, session: str, failures: Mapping[str, str]):
        self.dataset = dataset
        self.session = session
        self.failures = dict(failures)
        self.checks = tuple(self.failures)
        findings = "\n".join(
            f"- {name}: {finding}" for name, finding in self.failures.items()
        )
        named = " and ".join(f"{name!r}" for name in self.checks)
        super().__init__(
            _format_error(
                f"Session {session!r} of dataset {dataset!r} failed the {named} "
                f"check{'s' if len(self.checks) > 1 else ''}:\n{findings}",
                why="these checks refuse because the arrays would be wrong: "
                "spikes, events or epochs off the tracking clock, or positions "
                "in the wrong unit or scale, give plausible but wrong maps.",
                fix="correct the loader or its stated facts as each finding "
                "says; to load the session anyway, with the failure kept in "
                f"rec.failed_checks and printed first under 'Watch out', pass "
                f"accept_failed_checks={self.checks!r}.",
            )
        )
