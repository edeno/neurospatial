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

Translate = Callable[[dict[str, Any]], tuple[str, dict[str, Any] | None]]
"""Map earlier arguments onto today's keywords.

It takes the values given for the ``former`` arguments, by name, and returns
a sentence on which keyword to use and the corrected keywords, or None when
no single correction fits the values.
"""


def _shown(value: Any) -> str:
    """Show a short value as written, and anything longer by its type."""
    text = repr(value)
    return text if len(text) <= 40 and "\n" not in text else f"<{type(value).__name__}>"


def keyword_hint(
    *order: str,
    redirect: str | None = None,
    fix: str | None = None,
    former: tuple[str, ...] = (),
    translate: Translate | None = None,
    why: str | None = None,
    missing: Mapping[str, tuple[str, str]] | None = None,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Explain a call that passes options by position or to the wrong function.

    Python refuses an extra positional argument before the function runs, with
    a message that does not say which keyword was meant. The wrapped function
    raises a ``TypeError`` instead. When an earlier version took options in a
    known positional ``order``, it maps each extra positional value onto those
    keywords and shows the corrected call; otherwise it lists the keyword-only
    options without guessing which one was meant. Every other call, including
    other argument errors, runs the function unchanged. The signature,
    docstring and ``__wrapped__`` are the function's own. On a method, the
    corrected call is shown on ``env``, the name the documentation gives an
    environment.

    Parameters
    ----------
    *order : str
        Keyword-only parameters that extra positional values stand for, in
        the order an earlier version took them positionally. Leave it empty
        for a function that never took options by position.
    redirect : str, optional
        Dotted path of the function that takes options this one does not,
        such as the builder for a function that only loads. A call whose
        unexpected keywords are all parameters of that function raises a
        ``TypeError`` naming it with the corrected call.
    fix : str, optional
        Fix to show for extra positional values instead of mapping them onto
        keywords, for a function whose earlier positional arguments have no
        keyword today.
    former : tuple of str, optional
        Arguments an earlier version took first by position, in that order,
        that ``translate`` maps onto today's keywords. Extra positional values
        stand for them, and so does a keyword call using a name that no
        longer exists. A name that is still a parameter is passed on as it
        is.
    translate : callable, optional
        Required with ``former``; see :data:`Translate`.
    why : str, optional
        ``Why:`` line for a call that ``former`` explains.
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
    if bool(former) != (translate is not None):
        raise TypeError("keyword_hint needs former and translate together.")

    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        signature = inspect.signature(function)
        parameters = signature.parameters
        positional = [name for name, p in parameters.items() if p.kind in _POSITIONAL]
        if any(p.kind is p.VAR_POSITIONAL for p in parameters.values()):
            raise TypeError("keyword_hint needs a function without *args.")
        options = [name for name, p in parameters.items() if p.kind is p.KEYWORD_ONLY]
        unknown = [name for name in order if name not in options]
        if unknown:
            raise TypeError(f"keyword_hint order names unknown options {unknown}.")
        required = dict(missing or {})
        optional = [
            name
            for name in required
            if name not in options
            or parameters[name].default is not inspect.Parameter.empty
        ]
        if optional:
            raise TypeError(
                f"keyword_hint missing names options that are not required {optional}."
            )
        removed = frozenset(former) - parameters.keys()
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

        def formerly(values: dict[str, Any], kwargs: dict[str, Any]) -> str:
            assert translate is not None
            kept = {
                name: values.pop(name) for name in list(values) if name in parameters
            }
            text, corrected = translate(values)
            if corrected is None:
                return text
            return f"{text}, as in {call_with({**corrected, **kept, **kwargs})}"

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
            reason = (
                "Why: options are keyword-only, so a value passed by "
                "position cannot be matched to an option."
            )
            if fix is not None:
                advice = fix
            elif former and len(extra) <= len(former):
                advice = formerly(dict(zip(former, extra, strict=False)), kwargs)
                reason = why or reason
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
                    why=reason,
                    fix=advice,
                )
            )

        def former_error(kwargs: dict[str, Any]) -> TypeError:
            values = {name: kwargs[name] for name in former if name in kwargs}
            rest = {name: value for name, value in kwargs.items() if name not in values}
            listed = ", ".join(name for name in values if name in removed)
            return TypeError(
                _format_error(
                    f"{title}() does not take {listed}.",
                    why=why or f"Why: {listed} is the name an earlier version used.",
                    fix=formerly(values, rest),
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
            if removed and not removed.isdisjoint(kwargs):
                raise former_error(kwargs)
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
