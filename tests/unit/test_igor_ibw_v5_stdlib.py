"""Unit tests for the v0.6+ stdlib-only v5 ``.ibw`` reader.

The whole point of the v0.6 #1 work is that v5 reads should be
possible **without** the optional ``igor`` package installed.  These
tests therefore:

  - do **not** use ``pytest.importorskip("igor")`` at the module
    level (the import of :mod:`afmkit.io.igor_ibw` itself must
    succeed on a minimal install);
  - do **not** call :func:`igor.binarywave.load` directly (the v5
    path must work via the stdlib reader alone);
  - verify the round-trip is *byte-exact* — every header field the
    writer emits is read back with the same value, not just the
    (ext, force) arrays.

Tests for the v1 / v2 / v3 read path (which still uses the optional
``igor`` package) and the v0.5 note re-hydration contract live in
:mod:`tests.unit.test_igor_ibw` and are gated on the package being
installable.
"""

from __future__ import annotations

import importlib
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

# -- Imports that must succeed without the ``igor`` package installed -------
#
# :mod:`afmkit.io.igor_ibw` is the public surface; the v0.6+ design is
# that *importing* it must not require the optional ``igor`` package.
# The v1/v2/v3 read paths still use it (lazily), but the import-time
# side effect is gone.
from afmkit.core.curve import ForceCurve
from afmkit.io.igor_ibw import (
    _BIN_HEADER5_SIZE,
    _FMT_BIN5,
    _FMT_WAVE5,
    _WAVE_HEADER5_SIZE,
    _get_binarywave,
    _load_ibw_v5_stdlib,
    load_ibw,
    roundtrip_ibw,
    save_ibw,
)

# -- Fixtures -------------------------------------------------------------


@pytest.fixture
def synthetic_curve() -> ForceCurve:
    """A noise-free WLC curve with metadata suitable for round-trip."""
    x = np.linspace(0.5, 199.5, 200)
    p = 0.4
    lc = 200.0
    force = (4.1 / p) * (0.25 * (1.0 - x / lc) ** -2 - 0.25 + x / lc)
    return ForceCurve(
        extension=x,
        force=force,
        metadata={
            "k_cantilever": 0.12,
            "temperature": 298.0,
            "source_file": "synthetic_curve.ibw",
            "direction": "retract",
        },
    )


# -- Module-level contract: the import is stdlib-only --------------------


class TestModuleImport:
    """The v0.6+ contract: ``import afmkit.io.igor_ibw`` must succeed
    on a minimal install that doesn't have the ``igor`` package.

    We verify the contract by patching :func:`builtins.__import__`
    so that any attempt to import ``igor`` (or any submodule of it)
    raises :class:`ImportError`.  A fresh re-import of
    :mod:`afmkit.io.igor_ibw` is then performed: if it succeeds
    (because :mod:`afmkit.io.igor_ibw` doesn't touch ``igor`` at
    import time), the v5 read path is truly stdlib-only.

    We can't simply drop ``igor`` from :data:`sys.modules` — that
    would make Python try to re-import the real package, which would
    succeed on a CI environment where ``igor`` happens to be
    installed (i.e. this exact test environment).  Patching
    :func:`builtins.__import__` is the only reliable way to make
    ``import igor`` raise without actually uninstalling the package.
    """

    def test_module_imports_without_igor_package(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Re-import :mod:`afmkit.io.igor_ibw` with ``igor`` blocked at
        the import-system level.

        A fresh import is required because the module may have been
        imported earlier in the test session (with ``igor`` present).
        We force a re-import by clearing the cached
        :mod:`afmkit.io.igor_ibw` and :mod:`afmkit.io` entries from
        :data:`sys.modules`, then we install a wrapper around
        :func:`builtins.__import__` that raises :class:`ImportError`
        for any ``igor`` import.  The test passes if the re-import
        succeeds (the re-import doesn't touch ``igor``) and
        :func:`_get_binarywave` raises a clear, actionable
        :class:`ImportError` when actually called.
        """
        import builtins

        # Remove cached afmkit modules so the re-import is real.
        for mod_name in list(sys.modules):
            if mod_name == "afmkit.io.igor_ibw" or mod_name == "afmkit.io":
                monkeypatch.delitem(sys.modules, mod_name)

        # Wrap builtins.__import__ so any ``import igor`` (or
        # ``import igor.binarywave`` etc.) raises ImportError.
        original_import = builtins.__import__

        def _blocked_import(
            name: str,
            globals: object | None = None,  # - match builtins signature
            locals: object | None = None,  # - match builtins signature
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name == "igor" or name.startswith("igor."):
                raise ImportError(f"igor is blocked for this test ({name!r})")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _blocked_import)

        # Re-import afmkit.io.igor_ibw — must not raise.
        mod = importlib.import_module("afmkit.io.igor_ibw")
        assert mod is not None

        # The lazy helper exists and is callable; calling it should
        # fail with a clear ImportError because the block is in place.
        with pytest.raises(ImportError, match="afmkit\\[igor\\]"):
            _get_binarywave()


# -- Direct stdlib reader — round-trip and field-by-field checks ----------


class TestStdlibV5ReaderRoundTrip:
    """save_ibw(version=5) → _load_ibw_v5_stdlib → verify every field.

    The stdlib reader is the v0.6+ path; the existing
    :func:`load_ibw` routes v5 to it and v1/v2/v3 to the
    ``igor.binarywave``-backed path.  These tests exercise the
    stdlib reader directly so the assertions are about the
    reader's behaviour, not the dispatch logic.
    """

    def test_v5_save_then_stdlib_read_extension_force(
        self, tmp_path: Path, synthetic_curve: ForceCurve
    ) -> None:
        p = tmp_path / "rt.ibw"
        save_ibw(synthetic_curve, p, version=5)

        data = _load_ibw_v5_stdlib(p)
        wdata = data["wave"]["wData"]

        np.testing.assert_array_equal(wdata[0::2], synthetic_curve.extension)
        np.testing.assert_array_equal(wdata[1::2], synthetic_curve.force)

    def test_v5_save_then_stdlib_read_wave_header(
        self, tmp_path: Path, synthetic_curve: ForceCurve
    ) -> None:
        """The wave_header fields the stdlib reader returns match the
        afmkit writer's emissions, field by field."""
        p = tmp_path / "wh.ibw"
        save_ibw(synthetic_curve, p, version=5)

        data = _load_ibw_v5_stdlib(p)
        wh = data["wave"]["wave_header"]

        # Scalar fields.
        assert wh["version"] == 5
        assert wh["type"] == 4  # NT_FP64
        assert wh["npnts"] == 2 * synthetic_curve.n_points
        assert wh["whVersion"] == 1
        assert wh["fsValid"] == 1

        # bname is the source-file stem, NUL-padded to 32 B.
        bname = bytes(wh["bname"]).rstrip(b"\x00")
        assert bname == b"synthetic_curve"

        # dataUnits is "pN", NUL-padded to 4 B.
        data_units = bytes(wh["dataUnits"]).rstrip(b"\x00")
        assert data_units == b"pN"

        # dimUnits[0] is "nm", NUL-padded; the other 3 dims are empty.
        dim_units_0 = bytes(wh["dimUnits"][0]).rstrip(b"\x00")
        assert dim_units_0 == b"nm"
        for i in (1, 2, 3):
            assert bytes(wh["dimUnits"][i]).rstrip(b"\x00") == b""

        # sfA[0] = 1.0, sfB[0] = 0.0 so X for point p is p itself.
        assert float(wh["sfA"][0]) == pytest.approx(1.0)
        assert float(wh["sfB"][0]) == pytest.approx(0.0)
        # The unused sfA / sfB entries are zero.
        for v in wh["sfA"][1:]:
            assert float(v) == 0.0
        for v in wh["sfB"][1:]:
            assert float(v) == 0.0

        # nDim[0] is the interleaved length, the rest are 0.
        assert list(wh["nDim"]) == [2 * synthetic_curve.n_points, 0, 0, 0]

    def test_v5_save_then_stdlib_read_note(
        self, tmp_path: Path, synthetic_curve: ForceCurve
    ) -> None:
        """The note string is the same bytes the writer emitted, byte-
        for-byte (the afmkit writer uses ``"afmkit=2col; k=0.12;
        temperature=298.0\x00"`` for this fixture)."""
        p = tmp_path / "n.ibw"
        save_ibw(synthetic_curve, p, version=5)

        data = _load_ibw_v5_stdlib(p)
        note = data["wave"]["note"]

        assert b"afmkit=2col" in note
        assert b"k=0.12" in note
        assert b"temperature=298" in note
        # The writer always NUL-terminates the note.
        assert note.endswith(b"\x00")

    def test_v5_full_load_ibw_round_trip(self, tmp_path: Path, synthetic_curve: ForceCurve) -> None:
        """The public :func:`load_ibw` path works for v5 without the
        ``igor`` package installed.  The v0.6+ contract is that v5
        users don't need ``afmkit[igor]`` for reads.
        """
        p = tmp_path / "full.ibw"
        save_ibw(synthetic_curve, p, version=5)

        loaded = load_ibw(p)

        np.testing.assert_allclose(loaded.extension, synthetic_curve.extension)
        np.testing.assert_allclose(loaded.force, synthetic_curve.force)
        assert loaded.metadata.get("k_cantilever") == pytest.approx(0.12)
        assert loaded.metadata.get("temperature") == 298.0

    def test_v5_roundtrip_helper(self, tmp_path: Path, synthetic_curve: ForceCurve) -> None:
        """``roundtrip_ibw(curve, path, version=5)`` works end-to-end
        on a minimal install."""
        p = tmp_path / "rt_helper.ibw"
        loaded = roundtrip_ibw(synthetic_curve, p, version=5)

        np.testing.assert_allclose(loaded.extension, synthetic_curve.extension)
        np.testing.assert_allclose(loaded.force, synthetic_curve.force)
        assert loaded.metadata.get("k_cantilever") == pytest.approx(0.12)

    def test_v5_can_load_recognises_stdlib_file(
        self, tmp_path: Path, synthetic_curve: ForceCurve
    ) -> None:
        """The cheap version sniff (used by the loader registry)
        still recognises v5 files written by the afmkit writer.
        """
        from afmkit.io.igor_ibw import IgorIBWLoader

        p = tmp_path / "sniff.ibw"
        save_ibw(synthetic_curve, p, version=5)
        assert IgorIBWLoader().can_load(p) is True


# -- Byte-exact verification against the raw file bytes -------------------


class TestByteExactV5RoundTrip:
    """The v0.6+ promise: the stdlib v5 reader is a **mirror** of
    the v5 writer, so the bytes the writer emits are exactly the
    bytes the reader parses (modulo well-known non-determinism like
    file modification timestamps, which the writer pins to 0).

    We read the file twice — once as raw bytes and once through the
    stdlib reader — and assert that the reader's parsed fields
    match the raw-bytes interpretation.
    """

    def test_writer_and_stdlib_reader_agree_on_every_field(
        self, tmp_path: Path, synthetic_curve: ForceCurve
    ) -> None:
        p = tmp_path / "be.ibw"
        save_ibw(synthetic_curve, p, version=5)

        raw = p.read_bytes()
        # The on-disk layout:
        #   [0:2]                version (LE int16)
        #   [2:2+62]             BinHeader5
        #   [64:64+320]          WaveHeader5
        #   [384:384+8*npnts]    wData (LE float64)
        #   trailing             note
        assert len(raw) >= 2 + _BIN_HEADER5_SIZE + _WAVE_HEADER5_SIZE

        # -- BinHeader5 ----------------------------------------------------
        bin_bytes = raw[2 : 2 + _BIN_HEADER5_SIZE]
        bin_fields = struct.unpack(_FMT_BIN5, bin_bytes)
        wfm_size = bin_fields[1]
        note_size = bin_fields[3]

        # -- WaveHeader5 ---------------------------------------------------
        wave_bytes = raw[2 + _BIN_HEADER5_SIZE : 2 + _BIN_HEADER5_SIZE + _WAVE_HEADER5_SIZE]
        wave_fields = struct.unpack(_FMT_WAVE5, wave_bytes)

        # -- Reader (the thing under test) ---------------------------------
        data = _load_ibw_v5_stdlib(p)
        wh = data["wave"]["wave_header"]

        # BinHeader5 fields the reader exposes match the raw bytes.
        assert wh["wfmSize"] == wfm_size
        assert wh["formulaSize"] == bin_fields[2]

        # WaveHeader5 fields the reader exposes match the raw bytes.
        assert wh["type"] == wave_fields[4]
        assert wh["npnts"] == wave_fields[3]
        assert wh["whVersion"] == wave_fields[12]
        assert bytes(wh["bname"]) == b"".join(wave_fields[13:45])
        assert bytes(wh["dataUnits"]) == b"".join(wave_fields[59:63])
        for i in range(4):
            assert bytes(wh["dimUnits"][i]) == b"".join(wave_fields[63 + i * 4 : 63 + i * 4 + 4])
        assert list(wh["sfA"]) == list(wave_fields[51:55])
        assert list(wh["sfB"]) == list(wave_fields[55:59])
        assert list(wh["nDim"]) == list(wave_fields[47:51])
        assert wh["fsValid"] == wave_fields[79]
        assert wh["topFullScale"] == pytest.approx(wave_fields[81])
        assert wh["botFullScale"] == pytest.approx(wave_fields[82])

        # wData: the reader's parsed wData is the raw bytes reinterpreted
        # as little-endian float64, de-interleaved.
        data_offset = 2 + _BIN_HEADER5_SIZE + _WAVE_HEADER5_SIZE
        data_size = 8 * wh["npnts"]
        raw_wdata = np.frombuffer(raw[data_offset : data_offset + data_size], dtype="<f8")
        np.testing.assert_array_equal(data["wave"]["wData"], raw_wdata)

        # note: the reader's note is the trailing ``note_size`` bytes
        # verbatim.
        note_offset = data_offset + data_size
        assert data["wave"]["note"] == raw[note_offset : note_offset + note_size]
        # The on-disk file size matches what BinHeader5 says it should
        # be (version + bin + wave + data + note).
        assert len(raw) == 2 + _BIN_HEADER5_SIZE + _WAVE_HEADER5_SIZE + data_size + note_size

    def test_reader_does_not_require_2col_marker_for_v5(
        self, tmp_path: Path, synthetic_curve: ForceCurve
    ) -> None:
        """The stdlib reader returns the raw interleaved 1-D wData
        regardless of whether the note carries the ``afmkit=2col``
        marker.  The 2-col de-interleaving happens in
        :meth:`IgorIBWLoader.load`, not here — these tests are
        about the stdlib reader, not the loader's post-processing.
        """
        p = tmp_path / "raw.ibw"
        save_ibw(synthetic_curve, p, version=5)

        data = _load_ibw_v5_stdlib(p)
        wdata = data["wave"]["wData"]
        # 1-D, length 2*N.
        assert wdata.ndim == 1
        assert wdata.shape == (2 * synthetic_curve.n_points,)


# -- Error paths ----------------------------------------------------------


class TestStdlibV5ReaderErrors:
    """Clear, actionable errors for malformed v5 files."""

    def test_truncated_version_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "short.ibw"
        p.write_bytes(b"\x05")  # 1 byte, not 2
        with pytest.raises(ValueError, match="truncated header"):
            _load_ibw_v5_stdlib(p)

    def test_wrong_version_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "v2.ibw"
        # v2 file: write the version but no headers.
        p.write_bytes(struct.pack("<h", 2) + b"\x00" * 100)
        with pytest.raises(ValueError, match="only handles v5"):
            _load_ibw_v5_stdlib(p)

    def test_truncated_bin_header_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "short_bin.ibw"
        # version is v5, but BinHeader5 is incomplete.
        p.write_bytes(struct.pack("<h", 5) + b"\x00" * 30)
        with pytest.raises(ValueError, match="truncated BinHeader5"):
            _load_ibw_v5_stdlib(p)

    def test_truncated_wave_header_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "short_wave.ibw"
        # version + BinHeader5, then truncated WaveHeader5.
        p.write_bytes(
            struct.pack("<h", 5)
            + b"\x00" * _BIN_HEADER5_SIZE
            + b"\x00" * 100  # too short for the full WaveHeader5
        )
        with pytest.raises(ValueError, match="truncated WaveHeader5"):
            _load_ibw_v5_stdlib(p)

    def test_truncated_wdata_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "short_data.ibw"
        # Full BinHeader5 + WaveHeader5 with npnts=10, but only
        # 5 float64s of data follow.
        # BinHeader5 has 16 fields total: 1 H (checksum) + 15 l.
        # After the first 5 values (checksum, wfmSize, formulaSize,
        # noteSize, dataEUnitsSize), 11 l's remain
        # (4 dimEUnitsSize + 4 dimLabelsSize + sIndicesSize + 2 options).
        bin_bytes = struct.pack(
            _FMT_BIN5,
            0,  # checksum
            _WAVE_HEADER5_SIZE,  # wfmSize
            0,  # formulaSize
            0,  # noteSize
            0,  # dataEUnitsSize
            *((0,) * 11),  # remaining 11 l's
        )
        wave_bytes = struct.pack(
            _FMT_WAVE5,
            0,  # next
            0,  # creationDate
            0,  # modDate
            10,  # npnts (= 5 ext + 5 force pairs)
            4,  # type (NT_FP64)
            0,  # dLock
            *(b"\x00" for _ in range(6)),  # whpad1
            1,  # whVersion
            *(b"\x00" for _ in range(32)),  # bname
            0,  # whpad2
            0,  # dFolder
            10,
            0,
            0,
            0,  # nDim
            1.0,
            0.0,
            0.0,
            0.0,  # sfA
            0.0,
            0.0,
            0.0,
            0.0,  # sfB
            b"p",
            b"N",
            b"\x00",
            b"\x00",  # dataUnits
            b"n",
            b"m",
            b"\x00",
            b"\x00",  # dimUnits[0]
            *(b"\x00" for _ in range(12)),  # dimUnits[1..3]
            1,  # fsValid
            0,  # whpad3
            0.0,  # topFullScale
            0.0,  # botFullScale
            0,  # dataEUnits
            0,
            0,
            0,
            0,  # dimEUnits
            0,
            0,
            0,
            0,  # dimLabels
            0,  # waveNoteH
            *(0 for _ in range(16)),  # whUnused
            0,  # aModified
            0,  # wModified
            0,  # swModified
            b"\x00",  # useBits
            b"\x00",  # kindBits
            0,  # formula
            0,  # depID
            0,  # whpad4
            0,  # srcFldr
            0,  # fileName
            0,  # sIndices
        )
        # only 5 float64s after the headers (need 10)
        p.write_bytes(struct.pack("<h", 5) + bin_bytes + wave_bytes + b"\x00" * (8 * 5))
        with pytest.raises(ValueError, match="truncated wData"):
            _load_ibw_v5_stdlib(p)

    def test_non_float64_type_raises(self, tmp_path: Path) -> None:
        """A v5 file with ``type != NT_FP64`` is rejected with a
        clear error rather than silently misinterpreting the data.
        """
        p = tmp_path / "wrong_type.ibw"
        # Build BinHeader5 + WaveHeader5 with type=2 (NT_I32) and a
        # minimal wData payload. We do this directly via struct.pack
        # so we don't have to wait for an afmkit writer to learn how
        # to emit non-FP64 waves.
        # BinHeader5 has 16 fields total: 1 H (checksum) + 15 l.
        # After the first 5 values, 11 l's remain.
        bin_bytes = struct.pack(
            _FMT_BIN5,
            0,  # checksum
            _WAVE_HEADER5_SIZE,  # wfmSize
            0,  # formulaSize
            0,  # noteSize
            0,  # dataEUnitsSize
            *((0,) * 11),  # remaining 11 l's
        )
        wave_bytes = struct.pack(
            _FMT_WAVE5,
            0,  # next
            0,
            0,  # creationDate, modDate
            2,  # npnts = 1 pair
            2,  # type = NT_I32 (NOT NT_FP64)
            0,  # dLock
            *(b"\x00" for _ in range(6)),
            1,  # whVersion
            *(b"\x00" for _ in range(32)),
            0,
            0,  # whpad2, dFolder
            2,
            0,
            0,
            0,  # nDim
            1.0,
            0.0,
            0.0,
            0.0,  # sfA
            0.0,
            0.0,
            0.0,
            0.0,  # sfB
            b"p",
            b"N",
            b"\x00",
            b"\x00",
            b"n",
            b"m",
            b"\x00",
            b"\x00",
            *(b"\x00" for _ in range(12)),
            1,
            0,  # fsValid, whpad3
            0.0,
            0.0,  # topFullScale, botFullScale
            0,  # dataEUnits
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,  # waveNoteH
            *(0 for _ in range(16)),
            0,
            0,
            0,  # a/w/swModified
            b"\x00",
            b"\x00",  # use/kindBits
            0,
            0,
            0,
            0,
            0,
            0,  # formula, depID, whpad4, srcFldr, fileName, sIndices
        )
        p.write_bytes(struct.pack("<h", 5) + bin_bytes + wave_bytes + b"\x00" * 16)
        with pytest.raises(ValueError, match="NT_FP64"):
            _load_ibw_v5_stdlib(p)


# -- Dispatch: load_ibw routes v5 to stdlib, v2 to binarywave -----------


class TestLoadIgbwDispatch:
    """The public :func:`load_ibw` dispatches based on the 2-byte
    version.  v5 → stdlib (no ``igor`` needed).  v2 (and v1, v3) →
    :func:`igor.binarywave.load` (needs the ``igor`` package).

    These tests cover the v5 side.  The v1/v2/v3 dispatch is covered
    by :mod:`tests.unit.test_igor_ibw` (gated on ``igor`` being
    installable).
    """

    def test_load_ibw_v5_does_not_call_binarywave(
        self, tmp_path: Path, synthetic_curve: ForceCurve, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Spy on :func:`_get_binarywave` to confirm v5 reads don't
        touch the ``igor`` package at all."""
        from afmkit.io import igor_ibw

        p = tmp_path / "spy.ibw"
        save_ibw(synthetic_curve, p, version=5)

        call_count = 0
        original = igor_ibw._get_binarywave

        def _spy() -> object:
            nonlocal call_count
            call_count += 1
            return original()

        monkeypatch.setattr(igor_ibw, "_get_binarywave", _spy)

        loaded = load_ibw(p)
        np.testing.assert_allclose(loaded.extension, synthetic_curve.extension)
        # Critical: the v5 path never called _get_binarywave.
        assert (
            call_count == 0
        ), f"v5 load_ibw unexpectedly called _get_binarywave {call_count} times"
