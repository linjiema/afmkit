"""Igor Binary Wave (``.ibw``) reader and writer.

This module provides :class:`IgorIBWLoader`, :func:`load_ibw`,
:func:`load_ibw_batch`, and :func:`save_ibw` for round-tripping force-
extension data through the legacy Igor Pro binary wave format.

The module's read path uses the optional third-party ``igor`` package
(``pip install igor>=0.4``).  The package's :func:`igor.binarywave.load`
returns a dict that is converted into an afmkit :class:`ForceCurve`.

The module's write path is a small, stdlib-only (:mod:`struct`) v2
binary wave emitter.  We do not use the upstream library for writing
because the released ``igor==0.3`` on PyPI ships with
:func:`igor.binarywave.save` raising ``NotImplementedError`` and
several ``pre_pack`` paths unimplemented.  The v2 format is documented
in WaveMetrics technical note TN003 and is sufficient for the
2-column (extension, force) waves afmkit produces: we store
``(extension_nm, force_pN)`` pairs interleaved into a 1-D
``float64`` array and tag the file with an afmkit-specific
``"afmkit=2col"`` marker in the wave ``note`` so the reader can
reconstruct the column structure.  All other metadata is encoded
in the wave ``note`` as ``"key=value; key=value; …"``.

The on-disk layout written by :func:`save_ibw` is:

====  ========================  =================================
Off   Field                     Notes
====  ========================  =================================
0     ``version`` (int16)       Always 2.
2     BinHeader2 (14 B)         ``wfmSize``, ``noteSize``,
                                 ``pictSize=0``, 16-bit ``checksum``.
16    WaveHeader2 (110 B)       ``type=NT_FP64``, 4-byte pointers,
                                 ``dataUnits="pN"``, ``xUnits="nm"``,
                                 ``npnts = 2*N``, ``hsA=1``, ``hsB=0``.
126   ``wData`` (2*N · 8 B)     Interleaved ``(ext[i], force[i])``.
126 + data_size
      ``padding`` (16 B)        v2 reserved zeros.
142 + data_size
      ``note`` (note_size B)    UTF-8 ``"afmkit=2col; k=…; …"``.
====  ========================  =================================

All sizes use the standard (no-alignment) convention; the on-disk
byte count matches the value of ``wfmSize``.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Any

import numpy as np

from afmkit.core.curve import CurveBatch, ForceCurve

#: Public re-exports from this module.
__all__ = ["IgorIBWLoader", "load_ibw", "load_ibw_batch", "save_ibw"]


# ---------------------------------------------------------------------------
# Optional ``igor`` import
# ---------------------------------------------------------------------------
#
# ``igor.binarywave.load`` is the loader entry point we delegate to.  The
# released package on PyPI (``igor==0.3``) is incompatible with NumPy >= 2.0
# at *import* time: ``binarywave.py`` references ``_numpy.complex`` which
# was removed in NumPy 1.20 and finally errored out in 2.0.  Re-add the
# alias for the duration of the import so the package can load.  We only
# install the alias if the real attribute is missing, so future ``igor``
# releases that fix this upstream continue to work unmodified.

try:
    import numpy as _np

    if not hasattr(_np, "complex"):
        _np.complex = complex  # type: ignore[attr-defined]
    import igor.binarywave as _binarywave
except ImportError as _exc:  # pragma: no cover - exercised only without igor
    raise ImportError(
        "afmkit.io.igor_ibw requires the optional 'igor' package. "
        "Install it with `pip install 'afmkit[igor]'` or "
        "`pip install 'igor>=0.4'`."
    ) from _exc


# ---------------------------------------------------------------------------
# Constants for the v2 binary wave on-disk layout
# ---------------------------------------------------------------------------

#: Constant for ``NT_FP64`` (64-bit IEEE float) in the v2 ``type`` field.
_NT_FP64: int = 4

#: Size of BinHeader2 on disk, in bytes (standard / no-alignment layout).
_BIN_HEADER2_SIZE: int = 14

#: Size of WaveHeader2 on disk, in bytes (standard / no-alignment layout).
#: ``igor.binarywave`` reads the same fields at the same offsets when
#: presented with this layout (verified empirically).
_WAVE_HEADER2_SIZE: int = 110

#: Size of the v2 reserved padding between the data and the note.
_PADDING2_SIZE: int = 16

#: 16-bit checksum, computed as ``(-sum_of_header_shorts) & 0xFFFF`` so that
#: the total sum of all 16-bit header values is zero.
_CHECKSUM_MOD: int = 0x10000

#: Wave-name length for v2 (including the trailing NUL).
_BNAME2_LEN: int = 20

#: Length of a v2 ``dataUnits`` / ``xUnits`` field, including the NUL.
_UNITS2_LEN: int = 4

#: Pattern we use to detect our own 2-column round-tripped files.
_NOTE_MARKER: str = "afmkit=2col"

#: A more permissive wave-name regex used to extract the F/B direction
#: suffix.  The original Igor's ``Load_JPK_FX_Data`` looked at a number
#: of conventions; the most common one in our old experiment folders is
#: ``"*"`` + ``"_F"`` / ``"_B"`` (Forward / Backward) right before the
#: ``.ibw`` extension.  We match case-insensitively and accept either
#: an exact F/B suffix or a numeric (e.g. ``"_001_F"``) prefix.
_DIRECTION_SUFFIX_RE = re.compile(
    r"^(?P<stem>.+?)[._](?P<dir>[FfBb])$",
)

#: Pattern for the ``"k=0.12"`` substring inside a wave's ``note`` text.
_K_NOTE_RE = re.compile(r"(?:^|;\s*)k\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


# ---------------------------------------------------------------------------
# Low-level helpers — packing the v2 binary wave on disk
# ---------------------------------------------------------------------------

#: ``=`` (native byte order, standard size) ensures 4-byte pointer fields
#: (``I``) without inserting any padding bytes between fields, matching the
#: offsets that ``igor.binarywave`` reads back.  Using ``<`` (little-endian,
#: standard size) gives the same byte count and is portable across hosts.
_FMT_BIN2 = "<lllh"
_FMT_WAVE2 = (
    "<h"  # type (h, 2 B)
    "I"  # next (I, 4 B)
    "20c"  # bname (20 c's)
    "h"  # whVersion (h, 2 B)
    "h"  # srcFldr (h, 2 B)
    "I"  # fileName (I, 4 B)
    "4c"  # dataUnits (4 c's)
    "4c"  # xUnits (4 c's)
    "l"  # npnts (l, 4 B)
    "h"  # aModified (h, 2 B)
    "d"  # hsA (d, 8 B)
    "d"  # hsB (d, 8 B)
    "h"  # wModified (h, 2 B)
    "h"  # swModified (h, 2 B)
    "h"  # fsValid (h, 2 B)
    "d"  # topFullScale (d, 8 B)
    "d"  # botFullScale (d, 8 B)
    "c"  # useBits (c, 1 B)
    "c"  # kindBits (c, 1 B)
    "I"  # formula (I, 4 B)
    "l"  # depID (l, 4 B)
    "L"  # creationDate (L, 4 B)
    "2c"  # wUnused (2 c's)
    "L"  # modDate (L, 4 B)
    "I"  # waveNoteH (I, 4 B)
)


def _split_bytes(blob: bytes) -> list[bytes]:
    """Expand a ``bytes`` object into a list of one-byte :class:`bytes`
    suitable for use with the ``c`` struct format code.

    ``struct`` treats ``"4c"`` as four separate ``c`` fields, each of
    which must receive a single-byte :class:`bytes` value (not a multi-
    byte string).  Iterating over :class:`bytes` yields ``int``, so this
    helper wraps each byte in a one-element :class:`bytes`.
    """
    return [bytes([b]) for b in blob]


def _truncate_bytes(text: str, max_len: int) -> bytes:
    """Encode ``text`` as UTF-8, NUL-terminate, and pad / truncate to ``max_len``.

    The v2 binary wave headers carry fixed-width byte arrays
    (e.g. 20 bytes for ``bname``, 4 bytes for ``dataUnits``).  The
    array is always exactly ``max_len`` bytes long: short names are
    NUL-padded on the right and over-long names are truncated while
    leaving room for the trailing NUL so the field never appears
    "open".
    """
    raw = text.encode("utf-8", errors="replace")
    if len(raw) >= max_len:
        # Leave room for the trailing NUL.
        raw = raw[: max_len - 1]
    return raw + b"\x00" * (max_len - len(raw))


def _compute_checksum(bin_no_cksum: bytes, wave_header: bytes) -> int:
    """Return the v2 16-bit checksum that zeroes the header sum.

    Mirrors :func:`igor.util.checksum` from the upstream package.  The
    checksum is chosen so that the 16-bit sum of every 16-bit word in
    the (bin header + wave header) buffer is zero.
    """
    full = bin_no_cksum + wave_header
    n_shorts = len(full) // 2
    shorts = np.frombuffer(full[: n_shorts * 2], dtype="<h").astype(np.int64)
    return int((-int(shorts.sum())) & 0xFFFF)


# ---------------------------------------------------------------------------
# Low-level helpers — unpacking wave ``note`` strings
# ---------------------------------------------------------------------------


def _parse_k_from_note(note: bytes | str | None) -> float | None:
    """Extract ``k_cantilever`` from a wave note string.

    Searches for the first ``"k=…"`` token, optionally preceded by ``"; "``
    (the afmkit convention is to emit ``"k=…"`` between semicolons).  A
    leading ``"; "`` is allowed to mirror the writer's output.  Returns
    ``None`` if no ``k=`` token is found or the value is not finite.
    """
    if note is None:
        return None
    text = note.decode("utf-8", errors="replace") if isinstance(note, bytes) else note
    match = _K_NOTE_RE.search(text)
    if match is None:
        return None
    try:
        value = float(match.group(1))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _detect_2col_marker(note: bytes | str | None) -> bool:
    """Return ``True`` if the wave note carries our ``afmkit=2col`` marker.

    The marker is emitted by :func:`save_ibw` so that the reader can
    distinguish genuine 2-column files from 1-D waves that should be
    rejected with :class:`ValueError`.
    """
    if note is None:
        return False
    text = note.decode("utf-8", errors="replace") if isinstance(note, bytes) else note
    return _NOTE_MARKER in text


def _direction_from_name(name: str) -> str:
    """Infer sweep direction from a wave / file name.

    Returns ``"approach"`` for names ending in ``_F`` (Forward) or
    ``"retract"`` for names ending in ``_B`` (Backward), in either case.
    Falls back to ``"unknown"`` when no suffix matches — the spec is
    intentionally permissive so that other naming conventions can be
    added later without changing the loader contract.
    """
    base = name.rsplit(".", 1)[0]
    match = _DIRECTION_SUFFIX_RE.match(base)
    if match is None:
        return "unknown"
    suffix = match.group("dir").upper()
    if suffix == "F":
        return "approach"
    if suffix == "B":
        return "retract"
    return "unknown"


# ---------------------------------------------------------------------------
# IgorIBWLoader
# ---------------------------------------------------------------------------


class IgorIBWLoader:
    """Read a single :class:`~afmkit.core.curve.ForceCurve` from a ``.ibw`` file.

    The loader is stateless; the same instance can be reused across
    files.  Use :func:`load_ibw` for the typical one-shot case.

    Examples
    --------
    >>> from afmkit.io.igor_ibw import IgorIBWLoader
    >>> curve = IgorIBWLoader().load("curve001.ibw")  # doctest: +SKIP
    """

    #: Short identifier used in the loader registry / pluggy entry point.
    name: str = "igor_ibw"

    def can_load(self, path: Path) -> bool:
        """Return ``True`` if ``path`` looks like a readable ``.ibw``.

        The check is intentionally cheap: the file must exist, have a
        ``.ibw`` suffix (case-insensitive), and its first two bytes
        must be a supported wave version (``1``, ``2``, ``3``, or
        ``5``).  The cheap version sniff is good enough to disambiguate
        ``.ibw`` from arbitrary files in a folder; deeper
        inconsistencies surface as a clear :class:`ValueError` from
        :meth:`load`.
        """
        if not path.exists() or not path.is_file():
            return False
        if path.suffix.lower() != ".ibw":
            return False
        try:
            with path.open("rb") as fh:
                version_bytes = fh.read(2)
        except OSError:
            return False
        if len(version_bytes) < 2:
            return False
        # The first two bytes are the version.  Both little- and
        # big-endian encodings are tried, in line with the upstream
        # library which probes byte order from the version itself.
        candidates = struct.unpack("<hh", version_bytes + version_bytes)
        version = candidates[0] if candidates[0] in (1, 2, 3, 5) else candidates[1]
        return version in (1, 2, 3, 5)

    def load(self, path: Path | str) -> ForceCurve:
        """Read one ``.ibw`` file and return a :class:`ForceCurve`.

        Parameters
        ----------
        path
            Path to the file.  May be a :class:`str` or
            :class:`pathlib.Path`.

        Returns
        -------
        ForceCurve
            A curve whose ``extension`` (nm) and ``force`` (pN) arrays
            are reconstructed from the wave's 2-column data.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If the wave is 1-D (no second column to split into
            ``force``), if the file is not a recognised Igor version
            (1/2/3/5), or if the wave's data type is not 64-bit float.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Igor .ibw file not found: {path}")

        try:
            data = _binarywave.load(str(path))
        except Exception as exc:  # pragma: no cover - delegated to library
            raise ValueError(f"Failed to parse Igor .ibw file {path}: {exc}") from exc

        wave = data.get("wave", {})
        wave_header = wave.get("wave_header", {})
        wdata = np.asarray(wave.get("wData", np.empty(0)), dtype=np.float64)
        note = wave.get("note", b"")

        # Reject 1-D waves per the documented contract: a 1-D wave has
        # no second column to split into ``force``.  Files we wrote
        # ourselves carry the ``afmkit=2col`` marker, so they are
        # de-interleaved below instead of being rejected here.
        if not _detect_2col_marker(note):
            if wdata.ndim == 1:
                raise ValueError(
                    f"Igor .ibw file {path} is 1-D; afmkit requires a 2-column wave "
                    f"with (extension, force). Re-export from Igor as a 2-column "
                    f"wave or use `afmkit save_ibw` to round-trip afmkit curves."
                )
            if wdata.ndim != 2:
                raise ValueError(
                    f"Igor .ibw file {path} has an unsupported ndim={wdata.ndim}; "
                    f"only 1-D afmkit-tagged waves and 2-D waves are accepted."
                )
            if wdata.shape[1] != 2:
                raise ValueError(
                    f"Igor .ibw file {path} has {wdata.shape[1]} columns; "
                    f"afmkit requires exactly 2 columns (extension, force)."
                )
            ext = wdata[:, 0].astype(np.float64, copy=True)
            force = wdata[:, 1].astype(np.float64, copy=True)
        else:
            # afmkit-tagged 1-D wave: de-interleave (ext, force) pairs.
            if wdata.ndim != 1 or wdata.size % 2 != 0:
                raise ValueError(
                    f"Igor .ibw file {path} carries the afmkit=2col marker but the "
                    f"data has shape {wdata.shape}; expected an even-length 1-D wave."
                )
            ext = wdata[0::2].astype(np.float64, copy=True)
            force = wdata[1::2].astype(np.float64, copy=True)

        if wave_header.get("type", _NT_FP64) != _NT_FP64:
            raise ValueError(
                f"Igor .ibw file {path}: only NT_FP64 (64-bit float) waves are "
                f"supported; got type={wave_header.get('type')!r}."
            )

        # Pull metadata from the wave header + note string.
        k_note = _parse_k_from_note(note)
        direction = _direction_from_name(path.name)
        metadata: dict[str, Any] = {
            "source_file": path.name,
            "direction": direction,
            "ibw_header": _stringify_wave_header(wave_header),
        }
        if k_note is not None:
            metadata["k_cantilever"] = k_note

        return ForceCurve(ext, force, metadata=metadata)


def _stringify_wave_header(wave_header: dict[str, Any]) -> dict[str, Any]:
    """Convert a wave header dict to JSON-friendly Python natives.

    ``igor.binarywave`` returns a mix of :class:`numpy.ndarray` (for the
    ``c`` array fields like ``dataUnits``), :class:`bytes` (for ``note``),
    and :class:`int` / :class:`float` scalars.  Forcing them into plain
    Python types makes the dict round-trippable through metadata
    serialisers without further work downstream.
    """
    out: dict[str, Any] = {}
    for key, value in wave_header.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tobytes().decode("utf-8", errors="replace").rstrip("\x00")
        elif isinstance(value, bytes):
            out[key] = value.decode("utf-8", errors="replace").rstrip("\x00")
        elif isinstance(value, np.integer | np.floating):
            out[key] = value.item()
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def load_ibw(path: Path | str) -> ForceCurve:
    """Read a single ``.ibw`` file and return a :class:`ForceCurve`.

    Convenience wrapper for :class:`IgorIBWLoader`.
    """
    return IgorIBWLoader().load(path)


def load_ibw_batch(
    paths: list[Path | str],
    *,
    k_cantilever: float | None = None,
) -> CurveBatch:
    """Read multiple ``.ibw`` files and return a :class:`CurveBatch`.

    Parameters
    ----------
    paths
        Iterable of paths to ``.ibw`` files.
    k_cantilever
        If supplied, overrides any cantilever stiffness parsed from the
        wave ``note``.  Pass this when the per-file ``k=`` annotation is
        stale or missing.

    Notes
    -----
    Pairing follows the convention used by the original Igor procedure
    ``Load_JPK_FX_Data_20110514.ipf``: when the file list contains
    ``2 * N`` entries whose basenames end in ``_F`` / ``_B``, the
    loader groups them as ``(approach, retract)`` pairs.  Files without
    an ``_F`` / ``_B`` suffix are emitted with ``direction="unknown"``.

    The pairing is a best-effort heuristic, not a hard contract — the
    caller is free to override the per-curve ``direction`` metadata
    after the batch is returned.
    """
    paths = list(paths)
    if not paths:
        return CurveBatch([], name=None)

    # First pass: load every file unconditionally so the caller always
    # gets back the curves it asked for, even if the pairing is
    # ambiguous.
    curves: list[ForceCurve] = []
    for p in paths:
        curve = load_ibw(p)
        if k_cantilever is not None:
            curve = curve.with_metadata(k_cantilever=k_cantilever)
        curves.append(curve)

    # Second pass: pair up F / B waves by basename.  We sort the list by
    # the basename *minus* the F/B suffix so a directory like
    # ``curve001_F.ibw`` + ``curve001_B.ibw`` lines up in input order.
    fb_indices = [
        i for i, c in enumerate(curves) if c.metadata.get("direction") in ("approach", "retract")
    ]
    if len(fb_indices) == 2 * (len(fb_indices) // 2):
        for i in range(0, len(fb_indices) - 1, 2):
            a_idx = fb_indices[i]
            b_idx = fb_indices[i + 1]
            a_dir = curves[a_idx].metadata.get("direction")
            b_dir = curves[b_idx].metadata.get("direction")
            if a_dir == b_dir:
                # Two ``_F`` or two ``_B`` files; leave the second as-is.
                continue
            if a_dir == "retract":
                a_idx, b_idx = b_idx, a_idx
            curves[a_idx] = curves[a_idx].with_metadata(direction="approach")
            curves[b_idx] = curves[b_idx].with_metadata(direction="retract")

    return CurveBatch(
        curves, name=None, metadata={"k_cantilever": k_cantilever} if k_cantilever else {}
    )


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def _encode_note(
    *,
    k_cantilever: float | None,
    extra: dict[str, Any] | None,
) -> bytes:
    """Build the wave ``note`` string for an afmkit-written ``.ibw`` file.

    The note always carries the ``afmkit=2col`` marker so the reader
    knows to de-interleave.  When ``k_cantilever`` is provided, it is
    emitted as ``"k=<value>"``.  Additional ``extra`` keys are appended
    in sorted order as ``"key=value"`` pairs, allowing round-trip
    preservation of arbitrary metadata without changing the binary
    layout.
    """
    tokens: list[str] = [_NOTE_MARKER]
    if k_cantilever is not None and np.isfinite(k_cantilever):
        tokens.append(f"k={k_cantilever:g}")
    if extra:
        for key in sorted(extra):
            value = extra[key]
            if value is None:
                continue
            tokens.append(f"{key}={value}")
    return ("; ".join(tokens) + "\x00").encode("utf-8")


def save_ibw(curve: ForceCurve, path: Path | str) -> None:
    """Write a single :class:`ForceCurve` to a ``.ibw`` file.

    The output is a v2 Igor Binary Wave with the (extension, force)
    pairs interleaved into a 1-D ``float64`` array and tagged with
    ``"afmkit=2col"`` in the wave ``note`` so :func:`load_ibw` can
    reconstruct the column structure on the way back in.

    Parameters
    ----------
    curve
        The curve to serialise.  Both ``extension`` and ``force`` must
        have the same length; this is enforced by :class:`ForceCurve`
        itself.
    path
        Destination file.  Parent directories are created if missing.
        May be a :class:`str` or :class:`pathlib.Path`.

    Notes
    -----
    The on-disk ``note`` carries the cantilever stiffness under the
    ``k=`` key (if known), plus any extra scalar metadata the caller
    has attached to ``curve.metadata``.  A 16-bit checksum is computed
    over the bin header and wave header so the file passes Igor's
    internal validation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = np.asarray(curve.extension, dtype=np.float64)
    force = np.asarray(curve.force, dtype=np.float64)
    n = ext.size

    # 1-D interleaved payload.
    interleaved = np.empty(2 * n, dtype=np.float64)
    interleaved[0::2] = ext
    interleaved[1::2] = force

    # Wave name — use the source file's stem if present, fall back to a
    # stable placeholder.
    meta = dict(curve.metadata)
    source = meta.get("source_file")
    wave_name = Path(source).stem if isinstance(source, str) and source else "afmkit_wave"
    bname = _truncate_bytes(wave_name, _BNAME2_LEN)

    # k_cantilever, if known, goes into the note (in addition to the
    # metadata dict) so the reader can recover it without consulting
    # the metadata blob.
    k_value = meta.get("k_cantilever")
    if k_value is not None:
        try:
            k_value = float(k_value)
            if not np.isfinite(k_value):
                k_value = None
        except (TypeError, ValueError):
            k_value = None
    extra: dict[str, Any] = {}
    for key, value in meta.items():
        if key in {"source_file", "k_cantilever", "direction", "ibw_header"}:
            continue
        if value is None:
            continue
        if isinstance(value, str | int | float | bool):
            extra[key] = value
    note = _encode_note(k_cantilever=k_value, extra=extra)
    note_size = len(note)

    data_bytes = interleaved.tobytes()
    wfm_size = _WAVE_HEADER2_SIZE + len(data_bytes) + _PADDING2_SIZE

    # Build the headers.  We pack everything in two steps so the
    # checksum is computed over the (bin header with placeholder
    # checksum + wave header) buffer and then substituted into the
    # trailing 16-bit word.
    bin_no_cksum = struct.pack(
        _FMT_BIN2[:-1],  # all but the trailing 'h' (checksum)
        wfm_size,
        note_size,
        0,  # pictSize
    )
    wave_header = struct.pack(
        _FMT_WAVE2,
        _NT_FP64,  # type
        0,  # next
        *_split_bytes(bname),  # bname
        0,  # whVersion
        0,  # srcFldr
        0,  # fileName
        *_split_bytes(b"pN\x00\x00"),  # dataUnits
        *_split_bytes(b"nm\x00\x00"),  # xUnits
        2 * n,  # npnts (interleaved length)
        0,  # aModified
        1.0,  # hsA
        0.0,  # hsB
        0,  # wModified
        0,  # swModified
        1,  # fsValid
        float(interleaved.max()) if n else 0.0,  # topFullScale
        float(interleaved.min()) if n else 0.0,  # botFullScale
        b"\x00",  # useBits
        b"\x00",  # kindBits
        0,  # formula
        0,  # depID
        0,  # creationDate
        *_split_bytes(b"\x00\x00"),  # wUnused
        0,  # modDate
        0,  # waveNoteH
    )
    if len(wave_header) != _WAVE_HEADER2_SIZE:
        # Defensive: a struct-mismatch here would silently corrupt every
        # file we write, so fail loudly in dev rather than ship broken
        # bytes downstream.
        raise RuntimeError(
            f"WaveHeader2 packed to {len(wave_header)} bytes; expected {_WAVE_HEADER2_SIZE}"
        )

    checksum = _compute_checksum(bin_no_cksum, wave_header)
    bin_header = bin_no_cksum + struct.pack("<H", checksum)

    padding = b"\x00" * _PADDING2_SIZE
    file_bytes = (
        struct.pack("<h", 2)  # version
        + bin_header  # BinHeader2
        + wave_header  # WaveHeader2
        + data_bytes  # wData
        + padding  # v2 padding
        + note  # note
    )
    path.write_bytes(file_bytes)
