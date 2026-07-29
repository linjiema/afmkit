"""Marko-Siggia worm-like chain (WLC) polymer model.

This module is the **numerical heart** of afmkit's fitting engine: the
:class:`WLCModel` class implements the classical Marko-Siggia
interpolation formula for the force-extension response of a single
semi-flexible polymer, and is wired 1:1 to the original Igor Pro
``LVFitWLC`` routine used in the lab's legacy pipelines.

Why bit-exact with the Igor code matters
----------------------------------------
The original Igor implementation is the gold standard for SMFS data
analysis in our lab. Every published dataset, every batch fit, every
historical control has been processed through it. If afmkit returns
slightly different numbers for the same input, every downstream
comparison (p-values, batch statistics, longitudinal studies) becomes
suspect.

So we deliberately hardcode the thermal energy as
``kB*T = 4.1 pN·nm`` (room temperature, T ≈ 298 K) — the same constant
used in the Igor code, which is itself a rounded version of the
NIST/CODATA value. Do **not** "correct" this to 4.047 or 4.11 without
running the full legacy-vs-new equivalence test suite.

Formula
-------
For extension ``x`` (nm), persistence length ``p`` (nm), and contour
length ``L`` (nm)::

    F(x; p, L) = (kB*T / p) * [0.25 * (1 - x/L)^(-2) - 0.25 + x/L]

with ``kB*T = 4.1 pN·nm`` and the result in **pN**. The formula has a
divergence at ``x = L`` (the polymer is fully extended) — we return
``+inf`` there instead of raising, so the fitter can handle the
singularity without special-casing.

Reference
---------
Marko, J. F. & Siggia, E. D. *Macromolecules* **28**, 8759 (1995).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

__all__ = ["WLCModel"]


@dataclass(frozen=True)
class WLCModel:
    """Marko-Siggia worm-like chain model.

    A parameter-free, immutable callable that maps an extension axis
    (nm) to a force axis (pN) given two physical parameters:

    - ``p`` — persistence length in nm (typical: 0.4 nm for unfolded
      protein, 0.5-1.0 nm for ssDNA/dsDNA, ~15 nm for dsDNA in some
      regimes; protein values are the lab's bread and butter).
    - ``L`` — contour length in nm (i.e. the maximum physically
      reachable extension).

    The class is decorated with :func:`dataclasses.dataclass` and
    ``frozen=True`` because the model is pure mathematics — there is
    nothing to configure per instance, and freezing the class prevents
    downstream code from accidentally mutating the metadata attributes
    (``param_names`` etc.) that the fitter relies on.

    Examples
    --------
    >>> import numpy as np
    >>> from afmkit.models.wlc import WLCModel
    >>> model = WLCModel()
    >>> x = np.linspace(1.0, 199.0, 50)  # avoid the x = L singularity
    >>> F = model(x, p=0.4, L=200.0)
    >>> F.shape
    (50,)
    >>> bool(F[0] < F[-1])  # WLC is monotonically increasing on (0, L)
    True
    """

    #: Canonical parameter order — used by the fitting engine to unpack
    #: fitter state into keyword arguments. **Do not reorder** without
    #: updating the fitter and the plugin docs.
    param_names: ClassVar[tuple[str, ...]] = ("p", "L")

    #: Physical bounds for the fitter, in the same order as
    #: :attr:`param_names`.
    #:
    #: - ``p`` ∈ (0.05, 5.0) nm — below 0.05 nm is unphysical for any
    #:   real biopolymer; above 5.0 nm is rigid-rod territory and the
    #:   WLC interpolation breaks down.
    #: - ``L`` ∈ (10.0, 1000.0) nm — covers the lab's typical protein
    #:   constructs and most nucleic-acid handles; longer chains should
    #:   use a custom plugin.
    param_bounds: ClassVar[tuple[tuple[float, float], ...]] = (
        (0.05, 5.0),
        (10.0, 1000.0),
    )

    #: Human-readable description per parameter — surfaced in the GUI
    #: and CLI ``--help``.
    param_hints: ClassVar[dict[str, str]] = {
        "p": "Persistence length (nm)",
        "L": "Contour length (nm)",
    }

    #: Thermal energy at room temperature, ``kB*T``, in **pN·nm**.
    #: Hardcoded to match the original Igor ``LVFitWLC`` implementation
    #: 1:1 — see the module docstring for the rationale.
    _KBT_PNNM: ClassVar[float] = 4.1

    def __call__(self, x: np.ndarray, *, p: float, L: float) -> np.ndarray:
        """Evaluate the WLC force at the given extension(s).

        Parameters
        ----------
        x
            Extension in **nm**. Any array-like convertible to a 1-D
            float array.
        p
            Persistence length in **nm**. Must be positive.
        L
            Contour length in **nm**. Must be positive and strictly
            greater than every entry of ``x`` for the result to be
            finite — values of ``x`` approaching or exceeding ``L``
            yield diverging / unphysical forces, which is the expected
            behaviour of the Marko-Siggia formula.

        Returns
        -------
        np.ndarray
            Force in **pN**, same shape as ``x``. The entry at ``x == L``
            (or close to it, within float round-off) is ``+inf``; no
            exception is raised.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        # Suppress the divide-by-zero warning that numpy emits when
        # (1 - x/L) == 0 (i.e. x == L). We deliberately return inf
        # there — this matches the Igor code and lets the fitter
        # down-weight the singularity region via loss masking.
        with np.errstate(divide="ignore", invalid="ignore"):
            reduced = x_arr / L
            return (self._KBT_PNNM / p) * (0.25 * (1.0 - reduced) ** -2 - 0.25 + reduced)

    def guess_params(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Suggest a starting point for non-linear fitting.

        Heuristic:

        - ``L`` is initialised to ``max(x) * 1.1`` so the fitter starts
          in a region where the model is finite (``x`` is always below
          the singularity ``L``). Clamped to
          :attr:`param_bounds` for ``L`` so short and very long curves
          never produce a guess that the fitter has to snap back.
        - ``p`` is set to 0.4 nm, a common default for unfolded
          proteins — appropriate for the lab's most common sample
          type. The fitter will move it freely.

        Parameters
        ----------
        x, y
            Observed data (extensions in nm, forces in pN). Only the
            shape and extrema of ``x`` are used; ``y`` is accepted for
            API symmetry with other models that may need it for slope
            estimation.

        Returns
        -------
        dict
            ``{"p": float, "L": float}``, both finite and inside
            :attr:`param_bounds`.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        l_lo, l_hi = self.param_bounds[1]
        # ``x.max() * 1.1`` keeps the fitter on the finite side of
        # the singularity; clamping to the search bounds keeps the
        # guess inside the legal region for unusually short or long
        # data.
        l0 = min(max(float(x_arr.max()) * 1.1, l_lo), l_hi)
        return {"p": 0.4, "L": l0}
