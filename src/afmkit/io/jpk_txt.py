"""JPK 4-column ``.txt`` loader.

This module reads the legacy 4-column text format exported by JPK
Nanowizzard / ForceRobot AFM software. The conversion logic is
**bit-for-bit identical** to the original Igor procedure
``Load_JPK_FX_Data_20110514.ipf / FXImport()``:

.. code-block:: python

    force_pN = -deflection_N * 1e12
    extension_nm = piezo_m * 1e9 - force_pN / k_cantilever
    force_pN -= force_pN[:200].mean()  # mean of first 200 pts
    extension_nm -= extension_nm[-1]  # last point

File layout
-----------
The on-disk file is whitespace-separated, with four columns:

====  ===================  =============  =========================
Col   Channel              Native units   Meaning
====  ===================  =============  =========================
0     forward              m              Piezo position (approach)
1     forward              N              Vertical deflection
2     backward             m              Piezo position (retract)
3     backward             N              Vertical deflection
====  ===================  =============  =========================

The first non-empty line may be a text header (4 whitespace-separated
field names). Trailing blank lines are tolerated. The spring constant
``k_cantilever`` (pN/nm) is supplied by the caller, since it is not
stored in the file itself - JPK exports the pre-calibrated deflection
in newtons, so the cantilever stiffness is still needed to convert the
deflection back into a tip-sample separation in nanometres.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from afmkit.core.curve import CurveBatch, ForceCurve

__all__ = ["JPKTxtLoader", "load_jpk_txt"]


# Number of leading force points used to compute the force baseline
# (mean). Mirrors ``wavestats/Q/R=[0,199] Force_for_old`` in the
# original Igor code.
_BASELINE_NPOINTS = 200


class JPKTxtLoader:
    """Loader for the legacy JPK Nanowizzard 4-column ``.txt`` format.

    The loader is stateless; all per-file options are passed to
    :meth:`load` as keyword arguments.

    Examples
    --------
    >>> from pathlib import Path
    >>> from afmkit.io.jpk_txt import JPKTxtLoader
    >>> loader = JPKTxtLoader()
    >>> batch = loader.load(Path("curve001.txt"), k_cantilever=0.06)  # doctest: +SKIP
    >>> batch.n_curves  # doctest: +SKIP
    2
    """

    #: Short identifier used in the loader registry / pluggy entry point.
    name: str = "jpk_txt"

    # -- Public API --------------------------------------------------------

    def can_load(self, path: Path) -> bool:
        """Return ``True`` if ``path`` looks like a 4-column JPK ``.txt``.

        The check is intentionally cheap and conservative:

        1. The file exists and is a regular file.
        2. The suffix is ``.txt`` (case-insensitive).
        3. The first non-empty line has exactly 4 whitespace-separated
           fields. They may be numeric (no header) or text (a 4-column
           header row) — both shapes are accepted.

        This is sufficient to disambiguate JPK 4-column exports from
        arbitrary text files in a folder; column-count mismatches are
        reported with a clear error by :meth:`load`.
        """
        if not path.exists() or not path.is_file():
            return False
        if path.suffix.lower() != ".txt":
            return False
        try:
            first = _first_nonempty_line(path)
        except OSError:
            return False
        if not first:
            return False
        return len(first.split()) == 4

    def load(
        self,
        path: Path,
        *,
        k_cantilever: float,
        **_extras: Any,
    ) -> CurveBatch:
        """Load a JPK 4-column ``.txt`` file.

        Parameters
        ----------
        path
            Path to the ``.txt`` file.
        k_cantilever
            Cantilever spring constant in **pN/nm**. Must be finite and
            strictly positive.
        **_extras
            Absorbs any additional keyword arguments so this method
            remains structurally compatible with the :class:`Loader`
            protocol. Unrecognised keys are silently ignored.

        Returns
        -------
        CurveBatch
            A batch of two :class:`ForceCurve` instances:

            - ``batch[0]`` — approach sweep (``direction="approach"``).
            - ``batch[1]`` — retract sweep (``direction="retract"``).

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If ``k_cantilever`` is non-finite or non-positive, or if the
            file does not contain exactly 4 numeric columns.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JPK .txt file not found: {path}")

        if not np.isfinite(k_cantilever) or k_cantilever <= 0:
            raise ValueError(
                f"k_cantilever must be a finite positive number (pN/nm); got {k_cantilever!r}"
            )

        data = _read_4columns(path)
        piezo_f, defl_f, piezo_b, defl_b = (data[:, i] for i in range(4))

        fwd = _convert_one(piezo_f, defl_f, k_cantilever)
        bwd = _convert_one(piezo_b, defl_b, k_cantilever)

        source = str(path)
        per_curve_meta: dict[str, Any] = {
            "source_file": source,
            "k_cantilever": k_cantilever,
        }
        approach = ForceCurve(
            fwd["ext_nm"],
            fwd["force_pn"],
            metadata={**per_curve_meta, "direction": "approach"},
        )
        retract = ForceCurve(
            bwd["ext_nm"],
            bwd["force_pn"],
            metadata={**per_curve_meta, "direction": "retract"},
        )
        return CurveBatch(
            [approach, retract],
            metadata={"k_cantilever": k_cantilever, "source": source},
        )


# -- Module-level helper --------------------------------------------------


def load_jpk_txt(path: Path | str, *, k_cantilever: float) -> CurveBatch:
    """Convenience wrapper for :class:`JPKTxtLoader`.

    Equivalent to ``JPKTxtLoader().load(Path(path), k_cantilever=...)``.
    """
    return JPKTxtLoader().load(Path(path), k_cantilever=k_cantilever)


# -- Internals ------------------------------------------------------------


def _first_nonempty_line(path: Path) -> str:
    """Return the first non-empty line of ``path``, with leading/trailing
    whitespace stripped. Returns the empty string if the file is empty
    (or only contains blank lines)."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                return stripped
    return ""


def _looks_like_4_numbers(line: str) -> bool:
    """Return True if ``line`` parses as 4 whitespace-separated floats."""
    parts = line.split()
    if len(parts) != 4:
        return False
    for token in parts:
        try:
            float(token)
        except ValueError:
            return False
    return True


def _detect_header_rows(path: Path) -> int:
    """Return the number of leading rows to skip before numeric data.

    - 0 if the first non-empty line is 4 numeric fields (no header).
    - 1 if the first non-empty line is a 4-field text header.

    Anything else (wrong number of fields, empty file) is left for
    :func:`_read_4columns` to error out on with a clearer message.
    """
    first = _first_nonempty_line(path)
    if not first:
        return 0
    if _looks_like_4_numbers(first):
        return 0
    return 1


def _read_4columns(path: Path) -> np.ndarray:
    """Read the 4-column numeric block of ``path``.

    Raises :class:`ValueError` with a file-specific message if the
    file does not contain exactly 4 columns, or if ``numpy.loadtxt``
    fails for any other reason.
    """
    skip = _detect_header_rows(path)
    try:
        data = np.loadtxt(path, skiprows=skip)
    except (ValueError, UserWarning, OSError) as exc:
        # numpy emits UserWarning("loadtxt: input contained no data")
        # when the file (after skipping the header) is empty; projects
        # running with filterwarnings=["error"] turn that into an
        # exception, so we catch it explicitly and re-raise as a
        # ValueError for a consistent error type.
        raise ValueError(f"Failed to parse JPK .txt file {path}: {exc}") from exc

    if data.ndim == 1:
        # ``np.loadtxt`` returns a 1-D array for a single data row.
        # Promote to (1, 4) so downstream slicing works uniformly.
        if data.size == 4:
            return data.reshape(1, 4)
        raise ValueError(
            f"JPK .txt file {path} must contain exactly 4 columns; "
            f"got a 1-D data array of length {data.size}"
        )

    if data.shape[1] != 4:
        raise ValueError(
            f"JPK .txt file {path} must contain exactly 4 columns; got {data.shape[1]}"
        )
    return data


def _convert_one(
    piezo_m: np.ndarray,
    deflection_n: np.ndarray,
    k_cantilever: float,
) -> dict[str, np.ndarray]:
    """Apply the unit conversion + per-direction baseline correction.

    Returns a dict with two float64 arrays:

    - ``"force_pn"``  - deflection in pN, sign-flipped, baseline-subtracted.
    - ``"ext_nm"``    - piezo position in nm, cantilever-corrected,
                        baseline-subtracted.

    The formulas are a 1:1 translation of the original Igor block:
    ``force = -F * 1e12`` and ``ext = z*1e9 - F/k``.
    """
    # Local float64 copies via ``astype(copy=False)``; the
    # multiplications below produce fresh arrays, so the caller's
    # buffers are never mutated.
    force_pn = -deflection_n.astype(np.float64, copy=False) * 1e12
    ext_nm = piezo_m.astype(np.float64, copy=False) * 1e9 - force_pn / k_cantilever

    # Baseline correction: mean of the first ``_BASELINE_NPOINTS`` force
    # points; last extension point. Applied per-direction.
    force_pn = force_pn - force_pn[:_BASELINE_NPOINTS].mean()
    ext_nm = ext_nm - ext_nm[-1]

    return {"force_pn": force_pn, "ext_nm": ext_nm}
