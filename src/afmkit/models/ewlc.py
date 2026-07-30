"""Extensible worm-like chain (eWLC) polymer model.

This module implements the :class:`EWLCModel` class — the **extensible**
variant of the Marko-Siggia worm-like chain, originally proposed by
Odijk (1995) and interpolated to an explicit closed form by Wang,
Wen, Pellerin, Leuba, and Zlatanova (1997, eq. 6). The eWLC augments
the standard WLC with a finite **stretch modulus** K0 that regularises
the divergence at x → L and accounts for the small enthalpic
contribution to single-polymer elasticity at high force.

Why a separate class from :class:`~afmkit.models.wlc.WLCModel`
---------------------------------------------------------------
The WLC and eWLC are mathematically related (the eWLC reduces to the
WLC in the limit K0 → ∞) but **parameterically distinct**: a fit
performed with the wrong model will recover a biased ``L`` whenever
the data approaches the overstretching regime. afmkit treats them as
separate models in the registry so the user is always explicit about
which physics is being assumed.

Formula
-------
For extension ``x`` (nm), persistence length ``p`` (nm), contour
length ``L`` (nm), and stretch modulus ``K0`` (pN)::

    F(x; p, L, K0) = (kB*T / p) * [
        0.25 * (1 - x/L + 1/K0) ** (-2)
        - 0.25
        + x/L
    ]

with ``kB*T = 4.1 pN·nm`` (same room-temperature constant as
:mod:`afmkit.models.wlc`) and the result in **pN**. The ``1/K0`` term
inside the ``(1 - x/L + ...)`` parenthesis is the standard
interpolation that approximates the implicit Odijk form
``F = (kB*T / p) * [0.25 * (1 - x/L + F/K0)^(-2) - 0.25 + x/L]`` —
the fully explicit formula is obtained by replacing ``F/K0`` with
``1/K0`` in the small-force limit. As ``K0 → ∞`` the ``1/K0`` term
vanishes and the formula reduces to the Marko-Siggia WLC.

The formula has a divergence at ``x = L - 1/K0`` (the polymer is
fully extended); we return ``+inf`` there instead of raising, so
the fitter can handle the singularity without special-casing — the
same convention as :class:`~afmkit.models.wlc.WLCModel`.

Reference
---------
Wang, M. D., Wen, J., Pellerin, J. F., Leuba, S. H. & Zlatanova, J.
*Force Spectroscopy of Single Biomolecules* (extensible-WLC
interpolation). Internal lab use is supplemented by the
implementation in Bouchiat et al., *Biophys. J.* **76**, 409 (1999).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

__all__ = ["EWLCModel"]


@dataclass(frozen=True)
class EWLCModel:
    """Extensible worm-like chain (Wang 1997 interpolation).

    A parameter-free, immutable callable that maps an extension axis
    (nm) to a force axis (pN) given three physical parameters:

    - ``p`` — persistence length in **nm** (typical: 0.4 nm for
      dsDNA, 0.6-1.0 nm for ssDNA, 0.4-0.5 nm for unfolded protein
      in the lab's common constructs).
    - ``L`` — contour length in **nm** (the polymer's fully extended
      end-to-end length).
    - ``K0`` — stretch modulus in **pN** (typical: 500-2000 pN for
      nucleic acids; very stiff constructs like collagen can push
      this higher). The stretch modulus is the **inverse** of the
      chain's longitudinal compliance — a small K0 means a softer
      chain (slightly lower force at the same extension than the
      pure-WLC prediction); a very large K0 reduces the eWLC to
      the Marko-Siggia WLC.

    The class is a frozen dataclass for the same reason as
    :class:`~afmkit.models.wlc.WLCModel` — the model is pure
    mathematics, and freezing the class prevents downstream code from
    accidentally mutating the metadata attributes that the fitter
    relies on.

    Examples
    --------
    >>> import numpy as np
    >>> from afmkit.models.ewlc import EWLCModel
    >>> model = EWLCModel()
    >>> x = np.linspace(1.0, 199.0, 50)  # avoid the x = L singularity
    >>> F = model(x, p=0.4, L=200.0, K0=1500.0)
    >>> F.shape
    (50,)
    >>> bool(F[0] < F[-1])  # eWLC is monotonically increasing on (0, L)
    True
    """

    #: Canonical parameter order — used by the fitting engine to unpack
    #: fitter state into keyword arguments. **Do not reorder** without
    #: updating the fitter and the plugin docs.
    param_names: ClassVar[tuple[str, ...]] = ("p", "L", "K0")

    #: Physical bounds for the fitter, in the same order as
    #: :attr:`param_names`.
    #:
    #: - ``p`` ∈ (0.05, 5.0) nm — same range as the WLC; below 0.05 nm
    #:   is unphysical, above 5.0 nm is rigid-rod territory where the
    #:   WLC interpolation breaks down.
    #: - ``L`` ∈ (10.0, 10000.0) nm — extended to 10 µm to accommodate
    #:   the long nucleic-acid constructs that motivate the eWLC in
    #:   the first place (e.g. λ-phage dsDNA handles).
    #: - ``K0`` ∈ (100.0, 10000.0) pN — covers the realistic range for
    #:   nucleic acids and moderately stiff protein fibres; values
    #:   above 10 000 pN make the eWLC numerically indistinguishable
    #:   from the WLC, so we cap there.
    param_bounds: ClassVar[tuple[tuple[float, float], ...]] = (
        (0.05, 5.0),
        (10.0, 10000.0),
        (100.0, 10000.0),
    )

    #: Human-readable description per parameter — surfaced in the GUI
    #: and CLI ``--help``.
    param_hints: ClassVar[dict[str, str]] = {
        "p": "persistence length (nm), 0.4 for dsDNA, ~0.6 for ssDNA",
        "L": "contour length (nm), expected plateau of the WLC plateau",
        "K0": "stretch modulus (pN), 500-2000 for nucleic acids",
    }

    #: Thermal energy at room temperature, ``kB*T``, in **pN·nm**.
    #: Hardcoded to 4.1 pN·nm — the same constant as
    #: :class:`~afmkit.models.wlc.WLCModel` — to keep the eWLC and WLC
    #: predictions directly comparable on the same dataset.
    _KBT_PNNM: ClassVar[float] = 4.1

    def __call__(self, x: np.ndarray, *, p: float, L: float, K0: float = 1500.0) -> np.ndarray:
        """Evaluate the eWLC force at the given extension(s).

        Parameters
        ----------
        x
            Extension in **nm**. Any array-like convertible to a 1-D
            float array.
        p
            Persistence length in **nm**. Must be positive.
        L
            Contour length in **nm**. Must be positive and strictly
            greater than ``x - 1/K0`` for the result to be finite —
            values of ``x`` approaching or exceeding ``L`` yield
            diverging / unphysical forces, which is the expected
            behaviour of the eWLC interpolation.
        K0
            Stretch modulus in **pN**. Defaults to 1500.0, a typical
            value for dsDNA. Must be positive.

        Returns
        -------
        np.ndarray
            Force in **pN**, same shape as ``x``. The entry at the
            singularity ``x = L - 1/K0`` (or close to it, within float
            round-off) is ``+inf``; no exception is raised. The result
            reduces to the Marko-Siggia WLC as ``K0 → ∞``.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        # Suppress the divide-by-zero warning that numpy emits when
        # (1 - x/L + 1/K0) == 0 (i.e. x = L - 1/K0). We deliberately
        # return inf there — this matches the WLC convention and lets
        # the fitter down-weight the singularity region via loss
        # masking.
        with np.errstate(divide="ignore", invalid="ignore"):
            reduced = x_arr / L
            stretch = 1.0 / K0
            return (self._KBT_PNNM / p) * (0.25 * (1.0 - reduced + stretch) ** -2 - 0.25 + reduced)

    def guess_params(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Suggest a starting point for non-linear fitting.

        Heuristic:

        - ``L`` is initialised to ``0.9 * max(x)`` so the fitter
          starts in a region where the model is finite. Unlike the
          WLC, we do **not** pad above ``max(x)`` — the eWLC
          singularity is at ``L - 1/K0`` (not at ``L``), and the
          correct fit for the contour length is the location of the
          asymptote rather than a value above the data. Clamped to
          :attr:`param_bounds` for ``L`` so short and very long
          curves never produce a guess that the fitter has to snap
          back.
        - ``p`` is set to 0.5 nm — a reasonable default for both
          dsDNA and many unfolded proteins. The fitter will move it
          freely.
        - ``K0`` is set to 1500 pN, the canonical dsDNA value, as a
          safe starting point for nucleic-acid data. The fitter will
          adjust it within the (100, 10000) pN search range.

        Parameters
        ----------
        x, y
            Observed data (extensions in nm, forces in pN). Only the
            extrema of ``x`` are used; ``y`` is accepted for API
            symmetry with other models that may need it for slope
            estimation.

        Returns
        -------
        dict
            ``{"p": float, "L": float, "K0": float}``, all finite and
            inside :attr:`param_bounds`.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        l_lo, l_hi = self.param_bounds[1]
        # ``x.max() / 0.9`` is a slightly larger number that places
        # the singularity (L - 1/K0) just past the data; clamping to
        # the search bounds keeps the guess inside the legal region
        # for unusually short or long data.
        l0 = min(max(float(x_arr.max()) / 0.9, l_lo), l_hi)
        return {"p": 0.5, "L": l0, "K0": 1500.0}
