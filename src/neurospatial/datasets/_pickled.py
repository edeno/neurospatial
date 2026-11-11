"""Read an ``.npz`` whose object arrays are pickled, without letting the pickles run code.

``np.load`` reads an object array only with ``allow_pickle=True``, which lets
the file run any code it names. Some archives store spike trains that way, as
a 0-d object array holding a dict of unit index to spike times (Gardner et
al. 2022). :func:`read_npz` lists each pickle's imports from its opcodes
first (:func:`pickle_imports`), refuses any outside a short allow-list, and
then unpickles with :class:`SpikeUnpickler`, which resolves only those names.

A whole file can be a pickle too: Maboudi et al. (2018) deposit sessions as
``dill`` pickles of ``nelpy`` objects. :func:`read_dill_pickle` reads those
the same way, with :class:`DillUnpickler`: each listed class becomes an
:class:`InertObject` that holds the pickled state and runs none of the
class's code, and ``dill``'s module and attribute helpers resolve only to
NumPy's array and scalar reconstructors.
"""

from __future__ import annotations

import io
import pickle
import pickletools
import zipfile
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np

__all__ = [
    "DillUnpickler",
    "InertObject",
    "InertStandIn",
    "SpikeUnpickler",
    "inert_fields",
    "pickle_imports",
    "read_dill_pickle",
    "read_npz",
]


def _reconstruct() -> Any:
    """NumPy's ``_reconstruct``, from ``numpy._core`` (2.x) or ``numpy.core`` (1.x)."""
    from importlib import import_module

    try:
        multiarray = import_module("numpy._core.multiarray")
    except ImportError:  # NumPy 1.x
        multiarray = import_module("numpy.core.multiarray")
    return multiarray._reconstruct


# NumPy pickles every array as ``_reconstruct(numpy.ndarray, (0,), b"b")``
# followed by a BUILD that sets shape, dtype and data. The stream may name
# ``numpy.ndarray`` only as that first argument: resolving it to the real class
# would let a REDUCE call ``ndarray(shape, "O", buffer=...)`` and read raw
# bytes as object pointers.
_NDARRAY = object()


def _safe_reconstruct(cls: Any, shape: Any, dtype_code: Any) -> np.ndarray:
    """``_reconstruct`` of an empty placeholder array, the only call NumPy pickles make."""
    if cls is not _NDARRAY or shape != (0,) or dtype_code != b"b":
        msg = f"refusing _reconstruct({cls!r}, {shape!r}, {dtype_code!r})"
        raise pickle.UnpicklingError(msg)
    array: np.ndarray = _reconstruct()(np.ndarray, (0,), b"b")
    return array


# Built-in dtype objects, whose pickled state cannot be changed: a crafted
# stream cannot build a dtype whose flags read raw bytes as object pointers.
_DTYPES = {"O8": np.dtype("O8"), "f8": np.dtype("f8")}


def _dtype(spec: Any, align: bool = False, _copy: bool = False) -> np.dtype:
    """One of NumPy's built-in ``O8`` and ``f8`` dtypes; any other is refused."""
    if align or not isinstance(spec, str) or spec not in _DTYPES:
        msg = f"refusing dtype {spec!r} in a spike file (only O8 and f8 are expected)"
        raise pickle.UnpicklingError(msg)
    return _DTYPES[spec]


class SpikeUnpickler(pickle.Unpickler):
    """Unpickler that resolves only the globals a pickled dict of float arrays needs.

    Those are NumPy's ``_reconstruct`` (only as NumPy itself calls it),
    ``ndarray`` (only as its argument) and ``dtype`` (only the built-in
    ``O8`` and ``f8``). Any other global raises
    :class:`pickle.UnpicklingError` before it is called.
    """

    ALLOWED: ClassVar[dict[tuple[str, str], Any]] = {
        ("numpy.core.multiarray", "_reconstruct"): _safe_reconstruct,
        ("numpy._core.multiarray", "_reconstruct"): _safe_reconstruct,
        ("numpy", "ndarray"): _NDARRAY,
        ("numpy", "dtype"): _dtype,
    }

    def find_class(self, module: str, name: str) -> Any:
        """The allowed object for ``module.name``; refuse any other."""
        try:
            return self.ALLOWED[module, name]
        except KeyError:
            msg = f"refusing to load {module}.{name} from a spike file"
            raise pickle.UnpicklingError(msg) from None


def pickle_imports(blob: bytes) -> set[str]:
    """Every global a pickle would import, read from its opcodes without running it.

    Module and name strings are followed through the memo, as protocols 2-5
    write repeated strings once. ``"<unnamed>"`` marks a ``STACK_GLOBAL``
    whose names this reading cannot resolve; treat it as unexpected.

    Parameters
    ----------
    blob : bytes
        A pickle stream.

    Returns
    -------
    set of str
        ``"module.name"`` of each global the stream names.
    """
    found: set[str] = set()
    memo: dict[int, str | None] = {}
    recent: list[str] = []
    top: str | None = None
    for op, arg, _ in pickletools.genops(blob):
        if op.name in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8"}:
            top = str(arg)
            recent = [*recent[-1:], top]
        elif op.name in {"BINPUT", "LONG_BINPUT", "PUT"}:
            memo[cast("int", arg)] = top
        elif op.name == "MEMOIZE":
            memo[len(memo)] = top
        elif op.name in {"BINGET", "LONG_BINGET", "GET"}:
            top = memo.get(cast("int", arg))
            recent = [*recent[-1:], top] if top is not None else []
        elif op.name == "GLOBAL":
            found.add(str(arg).replace(" ", "."))
            recent, top = [], None
        elif op.name == "STACK_GLOBAL":
            found.add(".".join(recent) if len(recent) == 2 else "<unnamed>")
            recent, top = [], None
        else:
            recent, top = [], None
    return found


def read_npz(path: Path) -> dict[str, Any]:
    """Read every array of an ``.npz``; object arrays only through :class:`SpikeUnpickler`.

    Plain arrays are read from their ``.npy`` headers and bytes. An object
    array's pickle is refused if it imports any global outside
    ``SpikeUnpickler.ALLOWED`` (listed first, by :func:`pickle_imports`,
    without running it), and is otherwise unpickled by
    :class:`SpikeUnpickler`.

    Parameters
    ----------
    path : Path
        The ``.npz`` file.

    Returns
    -------
    dict
        Each member's name, without ``.npy``, to its array.

    Raises
    ------
    pickle.UnpicklingError
        If an object array's pickle names a global outside the allow-list.
    """
    from numpy.lib import format as npy

    arrays: dict[str, Any] = {}
    allowed = {".".join(key) for key in SpikeUnpickler.ALLOWED}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            with archive.open(member) as stream:
                version = npy.read_magic(stream)  # type: ignore[no-untyped-call]
                read_header = (
                    npy.read_array_header_1_0
                    if version == (1, 0)
                    else npy.read_array_header_2_0
                )
                shape, fortran, dtype = read_header(stream)  # type: ignore[no-untyped-call]
                name = member.removesuffix(".npy")
                if dtype.hasobject:
                    blob = stream.read()
                    if unexpected := pickle_imports(blob) - allowed:
                        msg = f"refusing {member}: its pickle imports {sorted(unexpected)}"
                        raise pickle.UnpicklingError(msg)
                    arrays[name] = SpikeUnpickler(io.BytesIO(blob)).load()
                else:
                    count = int(np.prod(shape))
                    data = np.frombuffer(stream.read(count * dtype.itemsize), dtype)
                    arrays[name] = data.reshape(shape, order="F" if fortran else "C")
    return arrays


#: The built-in dtypes a ``dill`` pickle of ``nelpy`` objects holds arrays in;
#: only the numeric ones can be the dtype of a pickled scalar.
_DILL_DTYPES = {
    spec: np.dtype(spec) for spec in ("O8", "f8", "i8", "u1", "u2", "u4", "u8")
}
_DILL_SCALAR_DTYPES = tuple(
    dtype for dtype in _DILL_DTYPES.values() if not dtype.hasobject
)
#: The modules ``dill``'s ``_import_module`` may name.
_DILL_MODULES = ("numpy.core.multiarray", "numpy._core.multiarray")


def _dill_dtype(spec: Any, align: bool = False, _copy: bool = False) -> np.dtype:
    """A built-in numeric or object dtype; any other is refused."""
    if align or not isinstance(spec, str) or spec not in _DILL_DTYPES:
        msg = f"refusing dtype {spec!r} (only {sorted(_DILL_DTYPES)} are expected)"
        raise pickle.UnpicklingError(msg)
    return _DILL_DTYPES[spec]


def _safe_scalar(dtype: Any, data: Any) -> np.generic:
    """``numpy.core.multiarray.scalar`` of a numeric dtype and its raw bytes only."""
    if (
        not any(dtype is allowed for allowed in _DILL_SCALAR_DTYPES)
        or type(data) is not bytes
        or len(data) != dtype.itemsize
    ):
        msg = f"refusing scalar({dtype!r}, {data!r:.40})"
        raise pickle.UnpicklingError(msg)
    value: np.generic = np.frombuffer(data, dtype=dtype)[0]
    return value


class _ModuleName:
    """What ``dill._import_module`` returns here: a module's name, not the module."""

    def __init__(self, name: str) -> None:
        self.name = name


def _import_module(name: Any) -> _ModuleName:
    """``dill._import_module`` for NumPy's ``multiarray`` only; nothing is imported."""
    if name not in _DILL_MODULES:
        msg = f"refusing dill._import_module({name!r})"
        raise pickle.UnpicklingError(msg)
    return _ModuleName(name)


def _get_attr(module: Any, name: Any) -> Any:
    """``dill._get_attr`` for NumPy's ``_reconstruct`` and ``scalar`` only."""
    if isinstance(module, _ModuleName):
        if name == "_reconstruct":
            return _safe_reconstruct
        if name == "scalar":
            return _safe_scalar
    where = module.name if isinstance(module, _ModuleName) else type(module).__name__
    msg = f"refusing dill._get_attr({where}, {name!r})"
    raise pickle.UnpicklingError(msg)


class InertObject:
    """An object of a class named in a pickle, rebuilt without any of that class's code.

    The pickle's constructor arguments go to ``args`` and the state its
    ``BUILD`` sets goes to ``state``; nothing else happens. Each pickled class
    becomes its own subclass, whose ``qualified_name`` is the class's
    ``module.name``.
    """

    qualified_name: ClassVar[str] = ""

    def __new__(cls, *args: Any) -> InertObject:
        """An empty instance holding ``args``."""
        instance = super().__new__(cls)
        instance.args = args
        instance.state = None
        return instance

    def __init__(self, *args: Any) -> None:
        self.args = args

    def __setstate__(self, state: Any) -> None:
        """Keep the pickled state as it is."""
        self.state = state

    def __repr__(self) -> str:
        return f"<inert {self.qualified_name}>"


class InertStandIn(InertObject):
    """A function, type, container or module namespace named in a pickle, inert.

    ``dill`` pickles a function as a call to ``_create_function`` with a code
    object it builds by calling ``_load_type("CodeType")``, and refers to the
    function's module through ``<module>.__dict__``; a ``deque`` is rebuilt by
    appending to it. A stand-in accepts each of these steps and records what
    it was given (``args``, ``items``), so the objects around it can be read,
    and runs nothing: calling one returns another stand-in.
    """

    def __call__(self, *args: Any) -> InertStandIn:
        """Another stand-in holding ``args``; nothing is called."""
        return type(self)(*args)

    def append(self, item: Any) -> None:
        """Keep ``item`` in ``items``."""
        self.__dict__.setdefault("items", []).append(item)

    def extend(self, items: Any) -> None:
        """Keep ``items`` in ``items``."""
        self.__dict__.setdefault("items", []).extend(items)


def inert_fields(obj: Any) -> dict[str, Any]:
    """The attributes an :class:`InertObject`'s pickled state set, as one dict.

    A class with ``__slots__`` pickles its state as ``(dict, slots)``; the two
    are merged.

    Raises
    ------
    TypeError
        If ``obj`` is not an inert object, or its state is not a dict or such
        a pair.
    """
    if not isinstance(obj, InertObject):
        msg = f"expected an inert pickled object, got {type(obj).__name__}"
        raise TypeError(msg)
    state = obj.state
    if isinstance(state, tuple) and len(state) == 2:
        merged = dict(state[0] or {})
        merged.update(state[1] or {})
        return merged
    if isinstance(state, dict):
        return dict(state)
    msg = f"{obj.qualified_name}: pickled state is {type(state).__name__}, not a dict"
    raise TypeError(msg)


class DillUnpickler(pickle.Unpickler):
    """Unpickler for a ``dill`` pickle of NumPy arrays and plain objects, running none of it.

    Resolves NumPy's ``ndarray`` (only as ``_reconstruct``'s argument) and
    ``dtype`` (only built-in numeric and object dtypes); ``dill``'s
    ``_import_module`` (only NumPy's ``multiarray``, as a name: nothing is
    imported) and ``_get_attr`` (only its ``_reconstruct``, as NumPy itself
    calls it, and ``scalar`` of raw numeric bytes); and each class in
    ``classes`` as an :class:`InertObject`. Any other global, module,
    attribute or dtype raises :class:`pickle.UnpicklingError` before it is
    used.

    Parameters
    ----------
    file : binary file
        The pickle stream.
    classes : collection of str
        ``"module.name"`` of each class to rebuild inertly.
    stand_ins : collection of str, default ()
        ``"module.name"`` of each function, type, container or module
        namespace (``"<module>.__dict__"``) to resolve to an
        :class:`InertStandIn`: for pickles of objects that hold ``dill``-pickled
        functions or deques, whose parts cannot otherwise be read.
    """

    ALLOWED: ClassVar[dict[tuple[str, str], Any]] = {
        ("numpy", "ndarray"): _NDARRAY,
        ("numpy", "dtype"): _dill_dtype,
        ("dill.dill", "_import_module"): _import_module,
        ("dill.dill", "_get_attr"): _get_attr,
    }

    def __init__(self, file: Any, classes: Any, stand_ins: Any = ()) -> None:
        super().__init__(file, encoding="latin1")
        self._classes = frozenset(classes)
        self._stand_ins = frozenset(stand_ins)
        self._inert: dict[str, type[InertObject]] = {}

    def find_class(self, module: str, name: str) -> Any:
        """The allowed object for ``module.name``; refuse any other."""
        if (module, name) in self.ALLOWED:
            return self.ALLOWED[module, name]
        qualified = f"{module}.{name}"
        if qualified in self._stand_ins:
            base: type[InertObject] = InertStandIn
        elif qualified in self._classes:
            base = InertObject
        else:
            msg = f"refusing to load {qualified} from a pickled file"
            raise pickle.UnpicklingError(msg)
        if qualified not in self._inert:
            self._inert[qualified] = type(name, (base,), {"qualified_name": qualified})
        return self._inert[qualified]


def read_dill_pickle(path: Path, classes: Any, stand_ins: Any = ()) -> Any:
    """Read a (gzipped) ``dill`` pickle without running any code it names.

    Its imports are listed first from its opcodes (:func:`pickle_imports`)
    and refused if any lies outside :attr:`DillUnpickler.ALLOWED` and
    ``classes``; then :class:`DillUnpickler` reads it.

    Parameters
    ----------
    path : Path
        The file; gzip-compressed or not, by its first two bytes.
    classes : collection of str
        ``"module.name"`` of each class the file may hold, rebuilt as
        :class:`InertObject`.
    stand_ins : collection of str, default ()
        ``"module.name"`` of each function, type, container or module
        namespace the file may name, resolved to an :class:`InertStandIn`.

    Returns
    -------
    object
        The pickled object, with each listed class's instances inert.

    Raises
    ------
    pickle.UnpicklingError
        If the pickle names a global, a ``dill`` module or attribute, or a
        dtype outside the allow-list.
    """
    import gzip

    path = Path(path)
    with path.open("rb") as stream:
        compressed = stream.read(2) == b"\x1f\x8b"
    if compressed:
        with gzip.open(path, "rb") as stream:
            blob = stream.read()
    else:
        blob = path.read_bytes()
    allowed = (
        {".".join(key) for key in DillUnpickler.ALLOWED} | set(classes) | set(stand_ins)
    )
    if unexpected := pickle_imports(blob) - allowed:
        msg = f"refusing {path.name}: its pickle imports {sorted(unexpected)}"
        raise pickle.UnpicklingError(msg)
    return DillUnpickler(io.BytesIO(blob), classes, stand_ins).load()
