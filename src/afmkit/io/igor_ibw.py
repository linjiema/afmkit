"""Igor Binary Wave (``.ibw``) reader and writer.

This module provides :class:`IgorIBWLoader`, :func:`load_ibw`,
:func:`load_ibw_batch`, and :func:`save_ibw` for round-tripping force-
extension data through the legacy Igor Pro binary wave format.

The module's read path uses the optional third-party ``igor`` package
(``pip install 'afmkit[igor]'`` → ``pip install igor>=0.3``).  The
package's :func:`igor.binarywave.load` returns a dict that is
converted into an afmkit :class:`ForceCurve`.  The reader
handles wave versions 1, 2, 3, and 5.

The module's write path is a small, stdlib-only (:mod:`struct`)
emitter.  We do not use the upstream library for writing because
the released ``igor==0.3`` on PyPI ships with
:func:`igor.binarywave.save` raising ``NotImplementedError`` and
several ``pre_pack`` paths unimplemented.  afmkit supports writing
two wave versions:

  - **v2** (default) — the original Igor Pro TN003 format.  1-D
    float64 wave, 18-char wave name, 3-char units, 16-byte trailing
    padding.  Sufficient for the 2-column (extension, force) waves
    afmkit produces.  All versions of Igor Pro can read this.

  - **v5** — the modern Igor Pro format (added in v6.00).  1-D
    float64 wave, 31-char wave name, extended dim units, no
    trailing padding.  Required by Igor Pro 7+ when writing waves
    that reference text or extended dimension units; also the
    format newer versions of Igor prefer for new waves.  We emit
    the minimal valid v5 (no dataEUnits, no dimEUnits, no
    dimLabels, no sIndices) so the file is still self-contained
    and round-trips through the ``igor`` reader.

In both versions, ``(extension_nm, force_pN)`` pairs are
interleaved into a 1-D ``float64`` array and the file is tagged
with an afmkit-specific ``"afmkit=2col"`` marker in the wave
``note`` so the reader can reconstruct the column structure.  All
other metadata is encoded in the wave ``note`` as
``"key=value; key=value; …"``.

The on-disk layouts written by :func:`save_ibw` are:

**v2** (default):

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

**v5**:

====  ========================  =================================
Off   Field                     Notes
====  ========================  =================================
0     ``version`` (int16)       Always 5.
2     BinHeader5 (62 / 128 B)   10 fields, 16-bit ``checksum`` first.
                                 Native ``l`` (4 B on 32-bit, 8 B on
                                 64-bit). All ``*Size`` counts
                                 (dataEUnitsSize, dimEUnitsSize,
                                 dimLabelsSize, sIndicesSize) are
                                 0 for our minimal v5.
     WaveHeader5 (~314 / 436 B) ``type=NT_FP64``, ``P`` (4 B
                                 unsigned) pointer fields,
                                 ``bname`` 31 chars + NUL,
                                 ``dataUnits="pN"``,
                                 ``dimUnits[0]="nm"``,
                                 ``npnts = 2*N``, ``nDim[0]=2*N``,
                                 ``nDim[1..3]=0``,
                                 ``sfA[0]=1``, ``sfB[0]=0``.
     ``wData`` (2*N · 8 B)     Interleaved ``(ext[i], force[i])``.
     ``note`` (note_size B)    UTF-8 ``"afmkit=2col; k=…; …"``.
====  ========================  =================================

The on-disk size of the v5 headers is **platform-dependent**:
v5 packs its headers with native field sizes (no ``<`` byte-order
prefix), matching :mod:`igor.binarywave`. On a 64-bit host the
BinHeader5 is 128 B (2 ``h`` + 6 alignment padding + 15 ``l`` x 8)
and the WaveHeader5 is 436 B; on a 32-bit host they are 62 B and
~314 B respectively. The writer computes the size at import time
from the actual struct format.

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

#: Constant for ``NT_FP64`` (64-bit IEEE float) in the v2 / v5 ``type`` field.
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

# ---------------------------------------------------------------------------
# v5 format constants — Igor Pro 6.00+ binary wave (TN003 v5 layout)
# ---------------------------------------------------------------------------

#: Wave-name length for v5 (31 chars + trailing NUL = 32 bytes on disk).
_BNAME5_LEN: int = 32

#: Length of a v5 ``dataUnits`` / ``dimUnits[i]`` field, including the NUL.
_UNITS5_LEN: int = 4

#: Size of BinHeader5 on disk, in bytes.
#:
#: Unlike v2 (which uses the standard ``<`` byte-order prefix and is
#: always 110 B regardless of host word size), v5 packs its headers
#: with the **``=`` (native byte order, standard field sizes)**
#: convention — the same one :mod:`igor.binarywave` uses on read
#: (``Wave.byte_order = '='`` in :mod:`igor.binarywave.load`).
#: On a little-endian host (everything we test on), ``=`` is
#: equivalent to ``<``, so the on-disk layout is platform-stable
#: at 62 B for the BinHeader5.
_BIN_HEADER5_SIZE: int = struct.calcsize("=" + "H" + "l" * 7 + "4l" + "4l")

#: Size of WaveHeader5 on disk, in bytes.
#:
#: Same ``=`` convention as :data:`_BIN_HEADER5_SIZE`; computed
#: from :data:`_FMT_WAVE5` at module import time. 320 B on
#: little-endian hosts (all CI platforms), regardless of whether
#: the host is 32- or 64-bit. The ``P`` → ``I`` substitution in
#: :mod:`igor.binarywave` is what makes this stable: without it,
#: the ``P`` (pointer) fields would be 8 B on 64-bit hosts and
#: the on-disk size would be 436 B, which the upstream reader
#: does not accept.
_WAVE_HEADER5_SIZE: int = struct.calcsize(
    "=I"  # next (P → I, 4 B)
    "L"  # creationDate
    "L"  # modDate
    "l"  # npnts
    "h"  # type
    "h"  # dLock
    "6c"  # whpad1
    "h"  # whVersion
    "32c"  # bname
    "l"  # whpad2
    "I"  # dFolder (P → I)
    "4l"  # nDim
    "4d"  # sfA
    "4d"  # sfB
    "4c"  # dataUnits
    "16c"  # dimUnits (4 dims x 4 B)
    "h"  # fsValid
    "h"  # whpad3
    "d"  # topFullScale
    "d"  # botFullScale
    "I"  # dataEUnits (P → I)
    "4I"  # dimEUnits (4xP → 4xI)
    "4I"  # dimLabels (4xP → 4xI)
    "I"  # waveNoteH (P → I)
    "16l"  # whUnused
    "h"  # aModified
    "h"  # wModified
    "h"  # swModified
    "c"  # useBits
    "c"  # kindBits
    "I"  # formula (P → I)
    "l"  # depID
    "h"  # whpad4
    "h"  # srcFldr
    "I"  # fileName (P → I)
    "I"  # sIndices (P → I)
)

#: Number of dimensions in a v5 wave (always 4 in the WaveHeader5 struct,
#: even for a 1-D wave — only ``nDim[0]`` is nonzero for us).
_WAVE_HEADER5_MAXDIMS: int = 4

#: ``whVersion`` value the v5 header wants (per the spec comment in
#: ``igor.binarywave``: "Write 1. Ignore on read.").
_WAVE_HEADER5_WHVERSION: int = 1

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

#: Pattern for the generic ``"key=value"`` token in a wave's note text.
#: Matches the key (a Python identifier; the writer only emits
#: :class:`str` keys from the curve metadata) and the value
#: (greedy through to the next ``; `` separator).  The leading
#: ``^|;`` anchor ensures the match doesn't pick up a substring
#: of a longer value (e.g. ``notes="k=foo; bar"`` would otherwise
#: match ``k=foo``).  Whitespace around ``=`` is tolerated.
_NOTE_TOKEN_RE = re.compile(r"(?:^|;\s*)([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+?)\s*(?=;|\x00|$)")


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


# ---------------------------------------------------------------------------
# Low-level helpers — packing the v5 binary wave on disk
# ---------------------------------------------------------------------------
#
# v5 uses the ``=`` (native byte order, standard field sizes) byte
# order prefix — the same convention :mod:`igor.binarywave` uses
# on read. On a little-endian host (everything we test on) ``=`` is
# identical to ``<``, so the on-disk layout is platform-stable at
# 62 B for the BinHeader5 and 320 B for the WaveHeader5. The ``P``
# (pointer) fields in WaveHeader5 are emitted as ``I`` (4-byte
# unsigned int) — the same ``P`` → ``I`` substitution the upstream
# reader applies for ILP32 / LP64 compatibility.
_FMT_BIN5_TRAILING = (
    "l"  # wfmSize
    "l"  # formulaSize
    "l"  # noteSize
    "l"  # dataEUnitsSize
    "4l"  # dimEUnitsSize[0..3] (4 dims)
    "4l"  # dimLabelsSize[0..3] (4 dims)
    "l"  # sIndicesSize
    "l"  # optionsSize1 (reserved; 0)
    "l"  # optionsSize2 (reserved; 0)
)

#: Full BinHeader5 — checksum is computed separately and substituted
#: in as the first 2 bytes. On-disk size: 62 B with the ``=`` byte
#: order prefix. See :data:`_BIN_HEADER5_SIZE` for the resolved
#: value.
_FMT_BIN5 = "=H" + _FMT_BIN5_TRAILING  # '=H' + 15x'l' (no padding)

#: WaveHeader5 — field order matches the ``WaveHeader5`` struct in
#: :mod:`igor.binarywave` (verified against igor==0.3). ``=`` byte
#: order prefix; on-disk size: 320 B.
#:
#: ``P`` (pointer) fields are emitted as ``I`` (4-byte unsigned
#: int) — the same ``P`` → ``I`` substitution the upstream reader
#: applies for ILP32 / LP64 compatibility. This includes the
#: ``next`` field at the start, which the igor source declares as
#: ``P`` but the on-disk size must be 4 B, not 8 B. (If we used
#: ``Q`` / 8 B for ``next``, the reader's 4-byte interpretation
#: would be off by 4 B and the entire WaveHeader5 would parse
#: wrong.)
_FMT_WAVE5 = (
    "=I"  # next (P → I, 4 B)
    "L"  # creationDate (4)
    "L"  # modDate (4)
    "l"  # npnts (4)
    "h"  # type (2)
    "h"  # dLock (2)
    "6c"  # whpad1 (6)
    "h"  # whVersion (2)
    "32c"  # bname (32)
    "l"  # whpad2 (4)
    "I"  # dFolder (P → I, 4 B)
    "4l"  # nDim[0..3] (16)
    "4d"  # sfA[0..3] (32)
    "4d"  # sfB[0..3] (32)
    "4c"  # dataUnits (4)
    "16c"  # dimUnits[0..3] (16) — 4 dims x 4 bytes
    "h"  # fsValid (2)
    "h"  # whpad3 (2)
    "d"  # topFullScale (8)
    "d"  # botFullScale (8)
    "I"  # dataEUnits (P → I, 4 B)
    "4I"  # dimEUnits[0..3] (16; 4 pointers → 4xI)
    "4I"  # dimLabels[0..3] (16; 4 pointers → 4xI)
    "I"  # waveNoteH (P → I, 4 B)
    "16l"  # whUnused[0..15] (64)
    "h"  # aModified (2)
    "h"  # wModified (2)
    "h"  # swModified (2)
    "c"  # useBits (1)
    "c"  # kindBits (1)
    "I"  # formula (P → I, 4 B)
    "l"  # depID (4)
    "h"  # whpad4 (2)
    "h"  # srcFldr (2)
    "I"  # fileName (P → I, 4 B)
    "I"  # sIndices (P → I, 4 B)
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
    """Return the 16-bit checksum that zeroes the header sum.

    Mirrors :func:`igor.util.checksum` from the upstream package.  The
    checksum is chosen so that the 16-bit sum of every 16-bit word in
    the (bin header + wave header) buffer is zero.

    The same algorithm works for both v2 and v5 — the only difference
    is *where* in the bin header the resulting checksum is stored
    (trailing 2 bytes for v2; leading 2 bytes for v5). The caller
    handles the placement; this function just computes the value.
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


def _coerce_note_value(raw: str) -> Any:
    """Coerce a note-string value to the most-restrictive Python type.

    The note is text, so the on-disk representation is always a
    string. The writer's contract is to emit scalars (str, int,
    float, bool) for the ``extra`` metadata, so we try the
    narrowest types first and fall back to the raw string. Booleans
    are matched as the exact strings ``"True"`` and ``"False"``
    (matching the writer's :func:`repr`); anything else stays a
    string.

    Returns the coerced value, or the raw string if no other type
    fits.
    """
    text = raw.strip()
    if text == "True":
        return True
    if text == "False":
        return False
    # Integer (only when there's no decimal point or exponent —
    # otherwise we'd swallow floats that happen to round-trip).
    if re.fullmatch(r"[-+]?\d+", text):
        try:
            return int(text)
        except ValueError:
            pass
    # Float (including scientific notation).
    try:
        value = float(text)
    except ValueError:
        return raw
    return value


def _parse_note_metadata(note: bytes | str | None) -> dict[str, Any]:
    """Extract all ``key=value`` pairs from an afmkit-written wave note.

    The first token is always the ``afmkit=2col`` marker and is
    skipped (the reader detects it via :func:`_detect_2col_marker`).
    Subsequent tokens are returned as a ``{key: value}`` dict with
    values coerced to their natural Python type
    (see :func:`_coerce_note_value`).

    Returns an empty dict if the note is ``None`` or carries no
    ``key=value`` tokens.
    """
    if note is None:
        return {}
    text = note.decode("utf-8", errors="replace") if isinstance(note, bytes) else note
    out: dict[str, Any] = {}
    for match in _NOTE_TOKEN_RE.finditer(text):
        key, raw = match.group(1), match.group(2)
        if key == "afmkit":  # the ``afmkit=2col`` marker, not data
            continue
        out[key] = _coerce_note_value(raw)
    return out


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

        # Pull metadata from the wave header + note string.  The
        # note is the v0.3+ afmkit-side store for round-trippable
        # metadata; :func:`_parse_note_metadata` re-hydrates every
        # ``key=value`` token the writer emitted, with the special
        # ``afmkit=2col`` marker stripped (it carries structural
        # information, not data).
        note_meta = _parse_note_metadata(note)
        # The note's on-disk key is ``k`` (the short form that fits
        # in the v2 255-byte note), but the curve metadata's
        # canonical key is ``k_cantilever``.  Rename before merging
        # so the round-trip preserves the user-facing attribute.
        if "k" in note_meta:
            note_meta["k_cantilever"] = note_meta.pop("k")
        direction = _direction_from_name(path.name)
        metadata: dict[str, Any] = {
            "source_file": path.name,
            "direction": direction,
            "ibw_header": _stringify_wave_header(wave_header),
        }
        # Note-driven metadata wins over the default wave-header
        # fields: the writer embeds ``k_cantilever`` (and any extra
        # scalar metadata) in the note, and that copy is the
        # user-facing source of truth.  The ``source_file`` /
        # ``direction`` / ``ibw_header`` keys are still set from the
        # filesystem / wave header because they are not part of
        # the note contract.
        for key, value in note_meta.items():
            if key not in metadata:
                metadata[key] = value

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


def roundtrip_ibw(curve: ForceCurve, path: Path | str, *, version: int = 2) -> ForceCurve:
    """Write a :class:`ForceCurve` to ``.ibw`` and read it back.

    Convenience wrapper around :func:`save_ibw` + :func:`load_ibw`
    that demonstrates the v0.5+ round-trip contract: the
    ``k_cantilever`` and any other scalar metadata embedded in the
    wave ``note`` by :func:`save_ibw` are re-hydrated by
    :func:`load_ibw`, so the returned curve carries the same
    metadata as the input — no explicit ``k_cantilever`` argument
    needed on the read side.

    Parameters
    ----------
    curve
        The curve to serialise.
    path
        Destination file.  Parent directories are created if missing.
    version
        Passed through to :func:`save_ibw`.  ``2`` is the default
        (works in all versions of Igor Pro); ``5`` is the modern
        Igor Pro 6.00+ layout.  Both round-trip through
        :func:`load_ibw`.

    Returns
    -------
    ForceCurve
        The curve as read back from ``path``.

    Notes
    -----
    The (extension, force) arrays are compared with
    :func:`numpy.testing.assert_allclose` (default
    ``rtol=1e-7``); a round-trip that changes the data values
    raises :class:`AssertionError`.  The metadata comparison is
    loose: scalar metadata keys (the ones the writer embeds in
    the note) are checked for equality, while ``ibw_header`` and
    ``source_file`` may differ because the reader re-derives them
    from the wave header / path.  This is a behavioural test, not
    a byte-equal contract.

    See Also
    --------
    save_ibw, load_ibw
    """
    save_ibw(curve, path, version=version)
    loaded = load_ibw(path)
    np.testing.assert_allclose(loaded.extension, curve.extension)
    np.testing.assert_allclose(loaded.force, curve.force)
    # The note re-hydration contract: every ``key=value`` token
    # the writer emitted must come back through ``metadata``.
    # We check the keys we care about (``k_cantilever`` plus any
    # extra scalar metadata the caller attached).
    for key, value in curve.metadata.items():
        if key in ("source_file", "ibw_header", "direction"):
            # Reader re-derives these from the wave header /
            # path; we don't compare them bit-for-bit.
            continue
        if key in loaded.metadata:
            assert loaded.metadata[key] == value, (
                f"round-trip mismatch for {key!r}: "
                f"wrote {value!r}, read back {loaded.metadata[key]!r}"
            )
    return loaded


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


def save_ibw(curve: ForceCurve, path: Path | str, *, version: int = 2) -> None:
    """Write a single :class:`ForceCurve` to a ``.ibw`` file.

    The (extension, force) pairs are interleaved into a 1-D
    ``float64`` array and the file is tagged with ``"afmkit=2col"``
    in the wave ``note`` so :func:`load_ibw` can reconstruct the
    column structure on the way back in.

    Parameters
    ----------
    curve
        The curve to serialise.  Both ``extension`` and ``force`` must
        have the same length; this is enforced by :class:`ForceCurve`
        itself.
    path
        Destination file.  Parent directories are created if missing.
        May be a :class:`str` or :class:`pathlib.Path`.
    version
        Igor Binary Wave version.  ``2`` is the default (works in
        all versions of Igor Pro).  ``5`` is the modern format
        (Igor Pro 6.00+) and is required by Igor Pro 7+ for
        waves that reference text or extended dimension units.
        Round-trips through the same :func:`load_ibw` either way.

    Raises
    ------
    ValueError
        If ``version`` is not 2 or 5.

    Notes
    -----
    The on-disk ``note`` carries the cantilever stiffness under the
    ``k=`` key (if known), plus any extra scalar metadata the caller
    has attached to ``curve.metadata``.  A 16-bit checksum is computed
    over the bin header and wave header so the file passes Igor's
    internal validation.

    See :func:`_save_ibw_v2` and :func:`_save_ibw_v5` for the
    per-version binary layout.
    """
    if version not in (2, 5):
        raise ValueError(f"save_ibw: version must be 2 or 5, got {version!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if version == 2:
        _save_ibw_v2(curve, path)
    else:
        _save_ibw_v5(curve, path)


def _build_ibw_payload(curve: ForceCurve) -> tuple[bytes, bytes, dict[str, Any]]:
    """Build the (interleaved_payload, note_bytes, metadata) tuple.

    Both :func:`_save_ibw_v2` and :func:`_save_ibw_v5` start from the
    same logical structure: a 1-D float64 array with
    (extension, force) pairs interleaved, a UTF-8 note string with
    the afmkit metadata, and a wave name derived from the source
    file. Splitting the construction out of the per-version writer
    keeps the per-version code focused on byte layout, not on
    afmkit-level semantics.

    Returns
    -------
    tuple
        ``(interleaved_bytes, note_bytes, ctx)`` where ``ctx`` is a
        dict with the keys ``wave_name``, ``bname_v2``,
        ``bname_v5``, ``n``, ``top``, ``bot`` consumed by the
        per-version writers.
    """
    ext = np.asarray(curve.extension, dtype=np.float64)
    force = np.asarray(curve.force, dtype=np.float64)
    n = ext.size

    # 1-D interleaved payload.
    interleaved = np.empty(2 * n, dtype=np.float64)
    interleaved[0::2] = ext
    interleaved[1::2] = force

    # Wave name — use the source file's stem if present, fall back to
    # a stable placeholder. The v2 bname is 20 bytes (incl. NUL);
    # the v5 bname is 32 bytes (incl. NUL). The same logical name
    # gets padded / truncated to whichever width the version
    # supports, so a v2 file and a v5 file for the same curve
    # round-trip through the same wave name in Igor Pro.
    meta = dict(curve.metadata)
    source = meta.get("source_file")
    wave_name = Path(source).stem if isinstance(source, str) and source else "afmkit_wave"

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

    ctx: dict[str, Any] = {
        "wave_name": wave_name,
        "bname_v2": _truncate_bytes(wave_name, _BNAME2_LEN),
        "bname_v5": _truncate_bytes(wave_name, _BNAME5_LEN),
        "n": n,
        "top": float(interleaved.max()) if n else 0.0,
        "bot": float(interleaved.min()) if n else 0.0,
    }
    return interleaved.tobytes(), note, ctx


def _save_ibw_v2(curve: ForceCurve, path: Path) -> None:
    """Write a v2 Igor Binary Wave (default; works in all Igor Pro versions).

    The on-disk layout is::

        0     version (int16)         = 2
        2     BinHeader2 (14 B)
        16    WaveHeader2 (110 B)
        126   wData (2*N · 8 B)
        126 + data_size
              padding (16 B)
        142 + data_size
              note (note_size B)

    See module docstring for the field-by-field table.
    """
    data_bytes, note, ctx = _build_ibw_payload(curve)
    n = ctx["n"]
    note_size = len(note)
    bname = ctx["bname_v2"]

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
        ctx["top"],  # topFullScale
        ctx["bot"],  # botFullScale
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


def _save_ibw_v5(curve: ForceCurve, path: Path) -> None:
    """Write a v5 Igor Binary Wave (Igor Pro 6.00+).

    The on-disk layout is::

        0     version (int16)         = 5
        2     BinHeader5 (62 B)
        64    WaveHeader5 (380 B)
        444   wData (2*N · 8 B)
        444 + data_size
              note (note_size B)

    We emit the minimal valid v5: no dataEUnits, no dimEUnits, no
    dimLabels, no sIndices. The corresponding ``*Size`` counts in
    BinHeader5 are zero, and the matching pointer fields in
    WaveHeader5 are zero (the ``igor`` reader treats those as
    "not present"). The ``whVersion`` field is 1, per the v5 spec
    comment in :mod:`igor.binarywave`.

    The 1-D float64 wave is dimensioned with ``nDim[0] = 2*N`` and
    ``nDim[1..3] = 0``. ``sfA[0] = 1.0``, ``sfB[0] = 0.0`` so the X
    value for point ``p`` is ``p`` itself, matching the v2
    ``hsA=1.0, hsB=0.0`` convention.

    See module docstring for the field-by-field table.
    """
    data_bytes, note, ctx = _build_ibw_payload(curve)
    n = ctx["n"]
    note_size = len(note)
    bname = ctx["bname_v5"]

    # BinHeader5 fields, in order (after the leading 16-bit checksum).
    # ``*Size`` counts are all zero for our minimal v5 (no extended
    # units, no dim labels, no string indices).
    wfm_size = _WAVE_HEADER5_SIZE + len(data_bytes)

    # WaveHeader5 fields. The nDim / sfA / sfB / dimUnits / dimEUnits /
    # dimLabels arrays are 4-wide each; only the [0] entry is meaningful
    # for a 1-D wave.
    n_dim = (2 * n, 0, 0, 0)
    sf_a = (1.0, 0.0, 0.0, 0.0)
    sf_b = (0.0, 0.0, 0.0, 0.0)
    dim_units = _split_bytes(b"nm\x00\x00") + _split_bytes(b"\x00\x00\x00\x00") * 3
    dim_eunits = (0, 0, 0, 0)  # 4 pointers; 0 = "no extended units"
    dim_labels = (0, 0, 0, 0)  # 4 pointers; 0 = "no labels"
    wh_unused = (0,) * 16

    # BinHeader5 packs the 16-bit ``checksum`` field first, then
    # 9 native ``l`` fields. Native alignment inserts 6 bytes of
    # padding between the ``H`` and the first ``l`` (on 64-bit
    # hosts), making the on-disk total 128 B. We pack the bin
    # header in one ``struct.pack`` call so the padding is added
    # by the struct module; the checksum is filled in with a
    # placeholder of 0 and substituted after the wave header is
    # packed.
    dim_eunits_sizes = (0, 0, 0, 0)  # dimEUnitsSize[0..3]
    dim_labels_sizes = (0, 0, 0, 0)  # dimLabelsSize[0..3]
    bin_header = struct.pack(
        _FMT_BIN5,
        0,  # checksum (placeholder, substituted below)
        wfm_size,
        0,  # formulaSize
        note_size,
        0,  # dataEUnitsSize
        *dim_eunits_sizes,
        *dim_labels_sizes,
        0,  # sIndicesSize
        0,  # optionsSize1
        0,  # optionsSize2
    )
    if len(bin_header) != _BIN_HEADER5_SIZE:
        raise RuntimeError(
            f"BinHeader5 packed to {len(bin_header)} bytes; expected {_BIN_HEADER5_SIZE}"
        )

    wave_header = struct.pack(
        _FMT_WAVE5,
        0,  # next (pointer; 0 in standalone files)
        0,  # creationDate
        0,  # modDate
        2 * n,  # npnts
        _NT_FP64,  # type
        0,  # dLock
        *_split_bytes(b"\x00" * 6),  # whpad1
        _WAVE_HEADER5_WHVERSION,  # whVersion
        *_split_bytes(bname),  # bname
        0,  # whpad2
        0,  # dFolder (pointer; 0)
        *n_dim,
        *sf_a,
        *sf_b,
        *_split_bytes(b"pN\x00\x00"),  # dataUnits
        *dim_units,
        1,  # fsValid
        0,  # whpad3
        ctx["top"],  # topFullScale
        ctx["bot"],  # botFullScale
        0,  # dataEUnits (pointer; 0)
        *dim_eunits,
        *dim_labels,
        0,  # waveNoteH (pointer; 0)
        *wh_unused,
        0,  # aModified
        0,  # wModified
        0,  # swModified
        b"\x00",  # useBits
        b"\x00",  # kindBits
        0,  # formula (pointer; 0)
        0,  # depID
        0,  # whpad4
        0,  # srcFldr
        0,  # fileName (pointer; 0)
        0,  # sIndices (pointer; 0)
    )
    if len(wave_header) != _WAVE_HEADER5_SIZE:
        # Defensive: a struct-mismatch here would silently corrupt every
        # file we write, so fail loudly in dev rather than ship broken
        # bytes downstream.
        raise RuntimeError(
            f"WaveHeader5 packed to {len(wave_header)} bytes; expected {_WAVE_HEADER5_SIZE}"
        )

    # ``_compute_checksum`` sums the 16-bit values in the (bin
    # header with placeholder checksum + wave header) buffer and
    # returns the value that makes the total ≡ 0 (mod 2^16). The
    # leading 2 bytes of the bin header are the checksum slot; the
    # padding bytes (positions 2-7 on 64-bit) are part of the sum.
    checksum = _compute_checksum(bin_header, wave_header)
    bin_header = struct.pack("<H", checksum) + bin_header[2:]

    file_bytes = (
        struct.pack("<h", 5)  # version
        + bin_header  # BinHeader5
        + wave_header  # WaveHeader5
        + data_bytes  # wData
        + note  # note (no v5 padding)
    )
    path.write_bytes(file_bytes)
