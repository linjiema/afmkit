"""Unit tests for :mod:`afmkit.io.igor_ibw`.

Round-trip and metadata extraction for the legacy Igor Binary Wave
read/write path. Every test is gated on the optional ``igor`` package
being importable — if it isn't, the whole module is skipped, which
mirrors the optional-dependency contract of the public surface
(``afmkit[igor]`` extra).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# The igor package itself is optional; without it, the whole module
# raises ImportError on import. Skip the entire test module in that
# case so `pytest tests/` stays green on a minimal install.
igor = pytest.importorskip("igor")

from afmkit.core.curve import ForceCurve  # noqa: E402
from afmkit.io.igor_ibw import (  # noqa: E402
    IgorIBWLoader,
    load_ibw,
    load_ibw_batch,
    save_ibw,
)

# -- Fixtures -------------------------------------------------------------


@pytest.fixture
def synthetic_curve() -> ForceCurve:
    """A noise-free WLC curve with metadata suitable for round-trip."""
    x = np.linspace(0.5, 199.5, 1000)
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


# -- can_load sniff -------------------------------------------------------


class TestCanLoad:
    """The cheap version sniff should accept real v2 waves and reject
    everything else."""

    def test_accepts_existing_ibw(self, tmp_path: Path, synthetic_curve: ForceCurve) -> None:
        p = tmp_path / "ok.ibw"
        save_ibw(synthetic_curve, p)
        assert IgorIBWLoader().can_load(p) is True

    def test_accepts_uppercase_suffix(self, tmp_path: Path, synthetic_curve: ForceCurve) -> None:
        p = tmp_path / "ok.IBW"
        save_ibw(synthetic_curve, p)
        assert IgorIBWLoader().can_load(p) is True

    def test_rejects_nonexistent(self, tmp_path: Path) -> None:
        assert IgorIBWLoader().can_load(tmp_path / "nope.ibw") is False

    def test_rejects_wrong_suffix(self, tmp_path: Path) -> None:
        p = tmp_path / "x.txt"
        p.write_text("not a wave", encoding="utf-8")
        assert IgorIBWLoader().can_load(p) is False

    def test_rejects_garbage_bytes(self, tmp_path: Path) -> None:
        p = tmp_path / "garbage.ibw"
        p.write_bytes(b"\x99\x99rest of the file is gibberish")
        assert IgorIBWLoader().can_load(p) is False


# -- Single-file round-trip -----------------------------------------------


class TestRoundTrip:
    """save_ibw → load_ibw should preserve (ext, force) and metadata."""

    def test_round_trip_preserves_extension_and_force(
        self, tmp_path: Path, synthetic_curve: ForceCurve
    ) -> None:
        p = tmp_path / "rt.ibw"
        save_ibw(synthetic_curve, p)

        loaded = load_ibw(p)

        np.testing.assert_allclose(loaded.extension, synthetic_curve.extension)
        np.testing.assert_allclose(loaded.force, synthetic_curve.force)
        assert loaded.n_points == synthetic_curve.n_points

    def test_round_trip_preserves_k_cantilever(
        self, tmp_path: Path, synthetic_curve: ForceCurve
    ) -> None:
        p = tmp_path / "k.ibw"
        save_ibw(synthetic_curve, p)

        loaded = load_ibw(p)

        assert loaded.metadata.get("k_cantilever") == pytest.approx(0.12)

    def test_round_trip_preserves_source_file(
        self, tmp_path: Path, synthetic_curve: ForceCurve
    ) -> None:
        """The loader's metadata contract is intentionally narrow:
        it preserves the keys afmkit's IBW writer emits (``k``,
        ``source_file``, ``direction``). The ``source_file`` key on
        a round-tripped curve is the *destination* path's filename
        (the wave's own ``bname`` field), not the original source
        path — IGOR's wave name is single-valued and 31 bytes max.

        Other keys in the original ForceCurve.metadata (e.g.
        ``temperature``) are intentionally not round-tripped — the
        v2 IBW note is a free-form text blob and afmkit does not yet
        promote arbitrary keys to it. Power users can reach the raw
        wave via ``igor.binarywave.load`` if they need full fidelity.
        """
        p = tmp_path / "sf.ibw"
        save_ibw(synthetic_curve, p)

        loaded = load_ibw(p)

        assert loaded.metadata.get("k_cantilever") == pytest.approx(0.12)
        assert loaded.metadata.get("source_file") == "sf.ibw"  # full filename
        assert "temperature" not in loaded.metadata  # not part of the contract

    def test_round_trip_marks_afmkit_origin(
        self, tmp_path: Path, synthetic_curve: ForceCurve
    ) -> None:
        """The 'afmkit=2col' marker in the wave note tells the reader
        the wave was written by afmkit (vs. some other tool that just
        happened to produce a v2 file)."""
        p = tmp_path / "m.ibw"
        save_ibw(synthetic_curve, p)

        # Read the raw wave via the underlying package to inspect the note.
        raw = igor.binarywave.load(str(p))
        note = raw["wave"]["note"] if isinstance(raw["wave"], dict) else raw["wave"].note
        assert (
            b"afmkit=2col" in note.encode("utf-8")
            if isinstance(note, str)
            else b"afmkit=2col" in note
        )

    def test_k_cantilever_from_curve_metadata_lands_in_note(
        self, tmp_path: Path, synthetic_curve: ForceCurve
    ) -> None:
        """The ``k=`` token in the note reflects the curve's own
        ``k_cantilever`` metadata. (The legacy v0.3 caller-arg
        override was removed; the only way to set ``k`` on the
        wave note is to put it in ``curve.metadata`` before the
        call.)"""
        # Override the curve's k_cantilever to a non-default value
        # and confirm the note picks it up.
        curve = ForceCurve(
            extension=synthetic_curve.extension,
            force=synthetic_curve.force,
            metadata={**synthetic_curve.metadata, "k_cantilever": 0.05},
        )
        p = tmp_path / "ko.ibw"
        save_ibw(curve, p)

        raw = igor.binarywave.load(str(p))
        note = raw["wave"]["note"]
        decoded = note.decode("utf-8", errors="replace") if isinstance(note, bytes) else str(note)
        assert "k=0.05" in decoded


# -- v5 round-trip ---------------------------------------------------------


class TestV5RoundTrip:
    """save_ibw(version=5) → load_ibw should preserve (ext, force) and
    metadata. The v5 format is the modern Igor Pro 6.00+ layout
    (WAVE_HEADER5 = 320 B, BIN_HEADER5 = 62 B, ``=`` byte order,
    ``P`` → ``I`` pointer substitution) and must round-trip through
    the same :mod:`igor.binarywave` reader the v2 path uses."""

    def test_v5_round_trip_preserves_extension_and_force(
        self, tmp_path: Path, synthetic_curve: ForceCurve
    ) -> None:
        p = tmp_path / "rt_v5.ibw"
        save_ibw(synthetic_curve, p, version=5)

        loaded = load_ibw(p)

        np.testing.assert_allclose(loaded.extension, synthetic_curve.extension)
        np.testing.assert_allclose(loaded.force, synthetic_curve.force)
        assert loaded.n_points == synthetic_curve.n_points

    def test_v5_round_trip_preserves_k_cantilever(
        self, tmp_path: Path, synthetic_curve: ForceCurve
    ) -> None:
        p = tmp_path / "k_v5.ibw"
        save_ibw(synthetic_curve, p, version=5)

        loaded = load_ibw(p)

        assert loaded.metadata.get("k_cantilever") == pytest.approx(0.12)

    def test_v5_round_trip_preserves_source_file(
        self, tmp_path: Path, synthetic_curve: ForceCurve
    ) -> None:
        """v5 bname is 31 chars + NUL (vs v2's 18 + NUL), so longer
        source filenames survive in the wave name. The loader's
        ``source_file`` metadata is the destination path's
        filename (same convention as v2)."""
        # Build a curve with a long source filename to exercise the
        # 31-char bname space.
        long_stem = "long_curve_name_abcdefghij"  # 26 chars
        curve = ForceCurve(
            extension=synthetic_curve.extension,
            force=synthetic_curve.force,
            metadata={**synthetic_curve.metadata, "source_file": long_stem + ".ibw"},
        )
        p = tmp_path / "sf_v5.ibw"
        save_ibw(curve, p, version=5)

        loaded = load_ibw(p)
        assert loaded.metadata.get("k_cantilever") == pytest.approx(0.12)
        # source_file in the loaded metadata is the *destination*
        # path's filename (same convention as v2). The wave's
        # bname (which is the source filename stem) is verified
        # via the raw igor read below.
        assert loaded.metadata.get("source_file") == "sf_v5.ibw"

        # The bname (31-char space) does hold the full long stem.
        raw = igor.binarywave.load(str(p))
        wh = raw["wave"]["wave_header"]
        bname = bytes(wh["bname"]).rstrip(b"\x00")
        assert bname == long_stem.encode()

    def test_v5_marker_in_note(self, tmp_path: Path, synthetic_curve: ForceCurve) -> None:
        """v5 files carry the same 'afmkit=2col' marker as v2 so
        :func:`load_ibw` can de-interleave the (ext, force)
        pairs."""
        p = tmp_path / "m_v5.ibw"
        save_ibw(synthetic_curve, p, version=5)

        raw = igor.binarywave.load(str(p))
        note = raw["wave"]["note"]
        decoded = note.decode("utf-8", errors="replace") if isinstance(note, bytes) else str(note)
        assert "afmkit=2col" in decoded

    def test_v5_wave_header_fields(self, tmp_path: Path, synthetic_curve: ForceCurve) -> None:
        """The v5 wave header carries the fields afmkit relies on
        for downstream analysis: ``type`` = NT_FP64, ``bname``
        matching the source file, ``dataUnits`` = "pN",
        ``dimUnits[0]`` = "nm", and ``sfA[0] = 1.0`` / ``sfB[0] = 0.0``
        for the 1-D (ext, force) index."""
        p = tmp_path / "wh_v5.ibw"
        save_ibw(synthetic_curve, p, version=5)

        raw = igor.binarywave.load(str(p))
        wh = raw["wave"]["wave_header"]

        assert int(wh["type"]) == 4  # NT_FP64
        assert int(wh["npnts"]) == 2 * synthetic_curve.n_points
        assert int(wh["fsValid"]) == 1
        # bname is a 32-byte array in v5; convert to bytes for inspection.
        bname = bytes(wh["bname"]).rstrip(b"\x00")
        # The writer uses ``Path(source).stem`` for the bname, so the
        # .ibw extension is dropped. The synthetic_curve fixture's
        # source_file is "synthetic_curve.ibw" → bname is
        # "synthetic_curve".
        assert bname == b"synthetic_curve"
        # dataUnits is a 4-byte array.
        data_units = bytes(wh["dataUnits"]).rstrip(b"\x00")
        assert data_units == b"pN"
        # dimUnits[0] (per-dim 4-byte array).
        dim_units_0 = bytes(wh["dimUnits"][0]).rstrip(b"\x00")
        assert dim_units_0 == b"nm"
        # sfA[0] = 1.0, sfB[0] = 0.0 so X for point p is p itself.
        assert float(wh["sfA"][0]) == pytest.approx(1.0)
        assert float(wh["sfB"][0]) == pytest.approx(0.0)

    def test_v5_v2_produce_equivalent_data(
        self, tmp_path: Path, synthetic_curve: ForceCurve
    ) -> None:
        """The two versions encode the same logical data; only the
        on-disk layout differs. After reading back, the v2 and v5
        files should yield curves with the same (ext, force) and
        metadata."""
        p_v2 = tmp_path / "v2.ibw"
        p_v5 = tmp_path / "v5.ibw"
        save_ibw(synthetic_curve, p_v2, version=2)
        save_ibw(synthetic_curve, p_v5, version=5)

        loaded_v2 = load_ibw(p_v2)
        loaded_v5 = load_ibw(p_v5)

        np.testing.assert_allclose(loaded_v2.extension, loaded_v5.extension)
        np.testing.assert_allclose(loaded_v2.force, loaded_v5.force)
        assert loaded_v2.metadata.get("k_cantilever") == loaded_v5.metadata.get("k_cantilever")

    def test_save_ibw_invalid_version_raises(
        self, tmp_path: Path, synthetic_curve: ForceCurve
    ) -> None:
        with pytest.raises(ValueError, match="version must be 2 or 5"):
            save_ibw(synthetic_curve, tmp_path / "x.ibw", version=3)
        with pytest.raises(ValueError, match="version must be 2 or 5"):
            save_ibw(synthetic_curve, tmp_path / "x.ibw", version=0)


# -- Batch load + direction pairing --------------------------------------


class TestBatchLoad:
    """load_ibw_batch groups _F / _B files into approach/retract pairs."""

    def test_pairing_f_and_b_suffix(self, tmp_path: Path, synthetic_curve: ForceCurve) -> None:
        f_path = tmp_path / "trace_F.ibw"
        b_path = tmp_path / "trace_B.ibw"
        # Build approach/retract with the same x but slightly different force
        save_ibw(synthetic_curve, f_path)
        save_ibw(synthetic_curve, b_path)

        batch = load_ibw_batch([f_path, b_path])

        assert batch.n_curves == 2
        # _F → approach, _B → retract
        directions = {c.metadata.get("direction") for c in batch}
        assert directions == {"approach", "retract"}

    def test_unknown_direction_for_unmarked_files(
        self, tmp_path: Path, synthetic_curve: ForceCurve
    ) -> None:
        p = tmp_path / "no_suffix.ibw"
        save_ibw(synthetic_curve, p)

        batch = load_ibw_batch([p])

        assert batch.n_curves == 1
        assert batch[0].metadata.get("direction") == "unknown"

    def test_batch_k_cantilever_applied_to_all(
        self, tmp_path: Path, synthetic_curve: ForceCurve
    ) -> None:
        paths = [tmp_path / f"c{i}.ibw" for i in range(3)]
        for p in paths:
            save_ibw(synthetic_curve, p)

        batch = load_ibw_batch(paths, k_cantilever=0.42)

        for curve in batch:
            assert curve.metadata.get("k_cantilever") == pytest.approx(0.42)


# -- Error paths ----------------------------------------------------------


class TestErrors:
    """The loader must raise clear, actionable errors for bad inputs."""

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises((FileNotFoundError, OSError)):
            load_ibw(tmp_path / "does_not_exist.ibw")

    def test_load_garbage_file_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.ibw"
        bad.write_bytes(b"this is not an ibw file at all")
        # the loader, the can_load check, or the underlying package may
        # all raise — they're all "garbage input" failure paths.
        with pytest.raises((ValueError, OSError, KeyError, IndexError, TypeError, RuntimeError)):
            load_ibw(bad)

    def test_save_to_readonly_path_raises(
        self, tmp_path: Path, synthetic_curve: ForceCurve
    ) -> None:
        # Make a directory at the target path so the writer can't create
        # the file. OSError/PermissionError on POSIX.
        target = tmp_path / "blocked"
        target.mkdir()
        with pytest.raises((OSError, PermissionError, IsADirectoryError)):
            save_ibw(synthetic_curve, target)


# -- Integration with CurveBatch -----------------------------------------


class TestCurveBatchIntegration:
    """The ibw loader should slot into the CurveBatch pipeline (HDF5
    round-trip, CSV export) the same way JPKTxtLoader does."""

    def test_ibw_then_hdf5_round_trip(self, tmp_path: Path, synthetic_curve: ForceCurve) -> None:
        pytest.importorskip("h5py")
        from afmkit.io.hdf5_store import load_hdf5, save_hdf5

        ibw = tmp_path / "rt.ibw"
        save_ibw(synthetic_curve, ibw)
        batch = load_ibw_batch([ibw])

        h5 = tmp_path / "rt.h5"
        save_hdf5(batch, h5)
        reloaded = load_hdf5(h5)

        assert reloaded.n_curves == 1
        np.testing.assert_allclose(reloaded[0].extension, synthetic_curve.extension)
        np.testing.assert_allclose(reloaded[0].force, synthetic_curve.force)
