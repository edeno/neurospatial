"""Name the keyword to use when an option is passed by position."""

from __future__ import annotations

import functools
import importlib
import inspect
from collections.abc import Callable, Mapping
from typing import Any, ParamSpec, TypeVar

from neurospatial._exceptions import _format_error

_P = ParamSpec("_P")
_R = TypeVar("_R")

_POSITIONAL = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
)


def _shown(value: Any) -> str:
    """Show a short value as written, and anything longer by its type."""
    text = repr(value)
    return text if len(text) <= 40 and "\n" not in text else f"<{type(value).__name__}>"


def keyword_hint(
    *order: str,
    redirect: str | None = None,
    fix: str | None = None,
    missing: Mapping[str, tuple[str, str]] | None = None,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Explain a call that passes options by position or to the wrong function.

    Python refuses an extra positional argument before the function runs, with
    a message that does not say which keyword was meant. The wrapped function
    raises a ``TypeError`` instead. When options have a natural positional
    ``order``, it maps each extra positional value onto those keywords and
    shows the corrected call; otherwise it lists the keyword-only options
    without guessing which one was meant. Every other call, including
    other argument errors, runs the function unchanged. The signature,
    docstring and ``__wrapped__`` are the function's own. On a method, the
    corrected call is shown on ``env``, the name the documentation gives an
    environment.

    Parameters
    ----------
    *order : str
        Keyword-only parameters that extra positional values stand for, in
        the order a caller would write them. Leave it empty when no order is
        natural.
    redirect : str, optional
        For a function that loads a saved environment and builds nothing,
        such as ``Environment.from_nwb``: the dotted path of the function
        that builds one. A call whose unexpected keywords are all parameters
        of that builder raises a ``TypeError`` that says the function only
        loads and shows the corrected call to the builder.
    fix : str, optional
        Fix to show for extra positional values instead of mapping them onto
        keywords, for values that have no keyword, such as a layout kind and
        its parameters passed to ``Environment.from_layout``.
    missing : mapping of str to (str, str), optional
        Required keyword-only options whose omission needs more than Python's
        "missing argument" message, each mapped to its ``Why:`` sentence and
        its fix: an option with no defensible default, where the caller has
        to choose.

    Returns
    -------
    callable
        A decorator that keeps the function's signature.
    """

    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        signature = inspect.signature(function)
        parameters = signature.parameters
        positional = [name for name, p in parameters.items() if p.kind in _POSITIONAL]
        if any(p.kind is p.VAR_POSITIONAL for p in parameters.values()):
            msg = "keyword_hint needs a function without *args."
            raise TypeError(msg)
        options = [name for name, p in parameters.items() if p.kind is p.KEYWORD_ONLY]
        unknown = [name for name in order if name not in options]
        if unknown:
            msg = f"keyword_hint order names unknown options {unknown}."
            raise TypeError(msg)
        required = dict(missing or {})
        optional = [
            name
            for name in required
            if name not in options
            or parameters[name].default is not inspect.Parameter.empty
        ]
        if optional:
            msg = (
                f"keyword_hint missing names options that are not required {optional}."
            )
            raise TypeError(msg)
        accepts_any_keyword = any(p.kind is p.VAR_KEYWORD for p in parameters.values())
        title = function.__qualname__
        # The caller neither passes nor sees the class or instance a
        # classmethod or method binds.
        bound = positional[:1] in (["cls"], ["self"])
        shown = positional[1:] if bound else positional
        caller = f"env.{function.__name__}" if positional[:1] == ["self"] else title

        def call_with(arguments: dict[str, Any]) -> str:
            given = [f"{name}={_shown(value)}" for name, value in arguments.items()]
            return f"{caller}({', '.join([*shown, *given])})"

        def positional_error(
            args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> TypeError:
            extra = args[len(positional) :]
            free = [name for name in order if name not in kwargs]
            mapped = dict(zip(free, extra, strict=False))
            count = len(shown)
            given_count = len(args) - len(positional) + count
            taken = (
                "no positional arguments"
                if count == 0
                else f"{count} positional argument{'' if count == 1 else 's'} "
                f"({', '.join(shown)})"
            )
            if fix is not None:
                advice = fix
            elif len(mapped) < len(extra):
                advice = "options are keyword-only; pass them by name: " + ", ".join(
                    f"{name}=" for name in options
                )
            else:
                listed = ", ".join(
                    f"{name}={_shown(value)}" for name, value in mapped.items()
                )
                advice = (
                    f"pass {listed} by keyword, as in {call_with({**mapped, **kwargs})}"
                )
            return TypeError(
                _format_error(
                    f"{title}() takes {taken} but {given_count} "
                    f"{'was' if given_count == 1 else 'were'} given.",
                    why="Why: options are keyword-only, so a value passed by "
                    "position cannot be matched to an option.",
                    fix=advice,
                )
            )

        def redirect_error(unexpected: dict[str, Any]) -> TypeError | None:
            if redirect is None:
                return None
            module_name, _, target_name = redirect.rpartition(".")
            target = getattr(importlib.import_module(module_name), target_name)
            target_parameters = inspect.signature(target).parameters
            if not all(name in target_parameters for name in unexpected):
                return None
            listed = ", ".join(unexpected)
            first = next(iter(target_parameters))
            given = ", ".join(
                f"{name}={_shown(value)}" for name, value in unexpected.items()
            )
            return TypeError(
                _format_error(
                    f"{title}() does not take {listed}.",
                    why=(
                        f"Why: {title} loads a saved environment and builds "
                        f"nothing; {listed} "
                        f"{'is an option' if len(unexpected) == 1 else 'are options'} "
                        f"of {redirect}, which builds one."
                    ),
                    fix=f"call {redirect}({first}, {given})",
                )
            )

        @functools.wraps(function)
        def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            if len(args) > len(positional):
                raise positional_error(args, kwargs)
            for name, (reason, advice) in required.items():
                if name not in kwargs:
                    raise TypeError(
                        _format_error(
                            f"{title}() needs {name}=.",
                            why=f"Why: {reason}",
                            fix=advice,
                        )
                    )
            if not accepts_any_keyword and redirect is not None:
                unexpected = {
                    name: value
                    for name, value in kwargs.items()
                    if name not in parameters
                }
                error = redirect_error(unexpected) if unexpected else None
                if error is not None:
                    raise error
            return function(*args, **kwargs)

        return guarded

    return decorate
