"""Public functions and classes document their inputs and usage."""

import inspect
import re
import sys

import pytest

import neurospatial

from ._flagship import FLAGSHIP, resolve

_EXEMPT: frozenset[str] = frozenset()


def _sections(doc: str) -> dict[str, str]:
    parts = re.split(r"(?m)^([A-Z][A-Za-z ]*)\n-{3,}\n", inspect.cleandoc(doc))
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def _documented_parameters(
    sections: dict[str, str], *, include_attributes: bool = False
) -> set[str]:
    names = ("Parameters", "Other Parameters")
    if include_attributes:
        names += ("Attributes",)
    text = "\n".join(sections.get(name, "") for name in names)
    return {
        name.strip().lstrip("*")
        for row in re.findall(r"(?m)^([^\s][^\n]*)", text)
        for name in row.split(":", 1)[0].split(",")
    }


def _public_objects() -> list[str]:
    paths = set(FLAGSHIP) | {f"neurospatial.{name}" for name in neurospatial.__all__}
    return sorted(
        path
        for path in paths
        if callable(resolve(path))
        and not (
            inspect.isclass(resolve(path)) and issubclass(resolve(path), BaseException)
        )
    )


@pytest.mark.parametrize("dotted", _public_objects())
def test_public_docstring_sections_and_parameters(dotted):
    if dotted in _EXEMPT:
        return
    obj = resolve(dotted)
    sections = _sections(inspect.getdoc(obj) or "")
    parameters = {
        name
        for name in inspect.signature(obj).parameters
        if name not in {"self", "cls"}
    }
    documented = _documented_parameters(
        sections, include_attributes=inspect.isclass(obj)
    )
    if inspect.isclass(obj):
        documented |= _documented_parameters(
            _sections(inspect.getdoc(obj.__init__) or ""), include_attributes=True
        )
    else:
        if parameters:
            assert "Parameters" in sections or "Other Parameters" in sections, (
                f"{dotted}: missing Parameters"
            )
        if inspect.signature(obj).return_annotation not in (None, type(None), "None"):
            assert "Returns" in sections or "Yields" in sections, (
                f"{dotted}: missing Returns"
            )
    assert "Examples" in sections, f"{dotted}: missing Examples"
    assert parameters <= documented, (
        f"{dotted}: undocumented parameters {sorted(parameters - documented)}"
    )


@pytest.mark.parametrize("dotted", sorted(_EXEMPT))
def test_exempt_public_names_still_resolve(dotted):
    assert callable(resolve(dotted)), f"remove obsolete exemption {dotted}"


def test_function_parameters_cannot_be_documented_only_as_attributes(monkeypatch):
    def function(value) -> None:
        """Validate an input.

        Parameters
        ----------
        other : int
            A different name from the required input.

        Attributes
        ----------
        value : int
            An attribute cannot document a function argument.

        Examples
        --------
        >>> function(1)
        """

    monkeypatch.setattr(sys.modules[__name__], "resolve", lambda _: function)
    with pytest.raises(AssertionError, match=r"undocumented parameters.*value"):
        test_public_docstring_sections_and_parameters("example.function")


def test_class_constructor_parameters_can_be_documented_as_attributes(monkeypatch):
    class Example:
        """Store a value.

        Attributes
        ----------
        value : int
            The value passed to the constructor.

        Examples
        --------
        >>> example = Example(1)
        """

        def __init__(self, value):
            self.value = value

    monkeypatch.setattr(sys.modules[__name__], "resolve", lambda _: Example)
    test_public_docstring_sections_and_parameters("example.Example")
