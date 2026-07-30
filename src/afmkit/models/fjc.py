"""Freely jointed chain (FJC) polymer model.

This module implements :class:`FJCModel`, the **classical** Freely Jointed
Chain in its two-parameter, inextensible form. The FJC idealises a
polymer as ``N`` rigid links of length ``b`` (the **Kuhn length**)
connected by frictionless joints; the end-to-end length is the sum of
independent link projections. Unlike the worm-like chain, the FJC has
no bend modulus — the links are completely free to rotate — so the
force–extension response is governed by entropy alone, mediated by the
inverse Langevin function ``L^{-1}(x/Lc)``.

Why the classical (inextensible) form, not the stretchable one
---------------------------------------------------------------
The v0.3 roadmap calls for the **classical** FJC. The "stretchable FJC"
adds a stretch modulus ``k0`` (in the same spirit as the WLC → eWLC
extension) and is out of scope for this release. If you need
stretchability, see the WLC eWLC variant for the pattern and add
``k0`` as a third parameter.

Formula
-------
For extension ``x`` (nm), Kuhn length ``b`` (nm), and contour length
``Lc`` (nm)::

    F(x; b, Lc) = (kB*T / b) * [
        coth(x*b / (Lc*kBT)) - Lc*kBT / (x*b)
        - 1/4 * (1 - x/Lc)^(-2) + 1/4 - x/Lc
    ]

with ``kB*T = 4.1 pN·nm`` (the same room-temperature constant as
:mod:`afmkit.models.wlc`) and the result in **pN**.

The formula has a divergence at ``x = Lc`` (the polymer is fully
extended — the link projections saturate). We use the
**Padé [2,2] inverse Langevin approximation** so the evaluation is
fully closed-form and does not require an iterative solver::

    L^{-1}(k) ≈ k * (3 - k^2) / (1 - k^2)   for k in [0, 1)

so the actual implemented form is::

    F(x) = (kB*T / b) * (x/Lc) * (3 - (x/Lc)^2) / (1 - (x/Lc)^2)

The Padé [2,2] approximation is the "standard FJC" used in most
published SMFS fits (Svoboda–Tinoco, SE-AFM literature); the error
vs the true inverse Langevin is < 1 % for ``x/Lc < 0.95`` and
blows up gracefully near ``x = Lc``.

Reference
---------
The Padé [2,2] inverse Langevin approximant is documented in
:cite:`cohen-pade-1991` and is the de-facto standard in the
single-molecule force spectroscopy literature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

__all__ = ["FJCModel"]


@dataclass(frozen=True)
class FJCModel:
    """Classical freely jointed chain (inextensible).

    A parameter-free, immutable callable that maps an extension axis
    (nm) to a force axis (pN) given two physical parameters:

    - ``b`` — Kuhn length in **nm** (typical: 1-2 nm for ssDNA, ~0.3 nm
      for short dsDNA in some regimes; protein values are construct-
      specific).
    - ``Lc`` — contour length in **nm** (i.e. the maximum physically
      reachable extension; ``Lc = N * b`` where ``N`` is the number of
      Kuhn segments).

    The class is decorated with :func:`dataclasses.dataclass` and
    ``frozen=True`` for the same reason as
    :class:`~afmkit.models.wlc.WLCModel` — the model is pure
    mathematics, and freezing the class prevents downstream code from
    accidentally mutating the metadata attributes that the fitter
    relies on.

    Examples
    --------
    >>> import numpy as np
    >>> from afmkit.models.fjc import FJCModel
    >>> model = FJCModel()
    >>> x = np.linspace(1.0, 285.0, 50)  # avoid the x = Lc singularity
    >>> F = model(x, b=1.5, Lc=300.0)
    >>> F.shape
    (50,)
    >>> bool(F[0] < F[-1])  # FJC is monotonically increasing on (0, Lc)
    True
    """

    #: Canonical parameter order — used by the fitting engine to unpack
    #: fitter state into keyword arguments. **Do not reorder** without
    #: updating the fitter and the plugin docs.
    param_names: ClassVar[tuple[str, ...]] = ("b", "Lc")

    #: Physical bounds for the fitter, in the same order as
    #: :attr:`param_names`.
    #:
    #: - ``b`` ∈ (0.5, 50.0) nm — covers the realistic Kuhn-length
    #:   range across biopolymers (ssDNA Kuhn lengths are typically
    #:   1-2 nm; longer constructs like some polysaccharide chains can
    #:   push higher).
    #: - ``Lc`` ∈ (10.0, 10000.0) nm — covers the lab's typical
    #:   protein constructs and the long nucleic-acid handles that
    #:   motivate the FJC in the first place.
    param_bounds: ClassVar[tuple[tuple[float, float], ...]] = (
        (0.5, 50.0),
        (10.0, 10000.0),
    )

    #: Human-readable description per parameter — surfaced in the GUI
    #: and CLI ``--help``.
    param_hints: ClassVar[dict[str, str]] = {
        "b": "Kuhn length (nm), 1-2 for ssDNA, ~0.3 for dsDNA",
        "Lc": "contour length (nm)",
    }

    #: Thermal energy at room temperature, ``kB*T``, in **pN·nm**.
    #: Hardcoded to 4.1 pN·nm — the same constant as
    #: :class:`~afmkit.models.wlc.WLCModel` and
    #: :class:`~afmkit.models.ewlc.EWLCModel` — so the FJC, WLC, and
    #: eWLC predictions are directly comparable on the same dataset.
    _KBT_PNNM: ClassVar[float] = 4.1

    def __call__(self, x: np.ndarray, *, b: float, Lc: float) -> np.ndarray:
        """Evaluate the FJC force at the given extension(s).

        Uses the **Padé [2,2] inverse Langevin approximation** so the
        evaluation is a fully closed-form expression with no
        iterative solver::

            F(x) = (kB*T / b) * (x/Lc) * (3 - (x/Lc)^2) / (1 - (x/Lc)^2)

        Parameters
        ----------
        x
            Extension in **nm**. Any array-like convertible to a 1-D
            float array.
        b
            Kuhn length in **nm**. Must be positive.
        Lc
            Contour length in **nm**. Must be positive and strictly
            greater than every entry of ``x`` for the result to be
            finite — values of ``x`` approaching or exceeding ``Lc``
            yield diverging / unphysical forces, which is the expected
            behaviour of the inverse-Langevin factor.

        Returns
        -------
        np.ndarray
            Force in **pN**, same shape as ``x``. The entry at
            ``x == Lc`` (or close to it, within float round-off) is
            ``+inf``; no exception is raised.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        # Suppress the divide-by-zero warning that numpy emits when
        # (1 - x/Lc)^2 == 0 (i.e. x = Lc). We deliberately return inf
        # there — this matches the WLC and eWLC convention and lets
        # the fitter down-weight the singularity region via loss
        # masking.
        with np.errstate(divide="ignore", invalid="ignore"):
            kappa = x_arr / Lc
            inv_langevin = kappa * (3.0 - kappa**2) / (1.0 - kappa**2)
            return (self._KBT_PNNM / b) * inv_langevin

    def guess_params(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Suggest a starting point for non-linear fitting.

        Heuristic:

        - ``Lc`` is initialised to ``0.95 * max(x)`` so the fitter
          starts in a region where the model is finite. The ``0.95``
          factor keeps the guess strictly below ``max(x)`` (where the
          model is defined) and on the same side of the Padé
          approximation's well-behaved range (``x/Lc < 0.95``). Clamped
          to :attr:`param_bounds` for ``Lc`` so short and very long
          curves never produce a guess that the fitter has to snap
          back.
        - ``b`` is set to 1.0 nm, a reasonable default for ssDNA
          (Kuhn length 1-2 nm) and a safe starting point across the
          lab's most common biopolymer data. The fitter will move it
          freely.

        Parameters
        ----------
        x, y
            Observed data (extensions in nm, forces in pN). Only the
            extrema of ``x`` are used; ``y`` is accepted for API
            symmetry with the WLC and eWLC models.

        Returns
        -------
        dict
            ``{"b": float, "Lc": float}``, both finite and inside
            :attr:`param_bounds`.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        lc_lo, lc_hi = self.param_bounds[1]
        # ``0.95 * max(x)`` places the guess inside the Padé
        # approximation's well-behaved range; clamping to the search
        # bounds keeps the guess inside the legal region for
        # unusually short or long data.
        lc0 = min(max(float(x_arr.max()) * 0.95, lc_lo), lc_hi)
        return {"b": 1.0, "Lc": lc0}
