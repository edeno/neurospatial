"""Read an ``.npz`` whose object arrays are pickled, without letting the pickles run code.

``np.load`` reads an object array only with ``allow_pickle=True``, which lets
the file run any code it names. Some archives store spike trains that way, as
a 0-d object array holding a dict of unit index to spike times (Gardner et
al. 2022). :func:`read_npz` lists each pickle's imports from its opcodes
first (:func:`pickle_imports`), refuses any outside a short allow-list, and
then unpickles with :class:`SpikeUnpickler`, which resolves only those names.
"""

from __future__ import annotations

import io
import pickle
import pickletools
import zipfile
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np

__all__ = ["SpikeUnpickler", "pickle_imports", "read_npz"]


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
