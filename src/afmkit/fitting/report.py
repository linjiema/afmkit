"""Fit outcome: :class:`FitResult` and serialisation helpers.

The :class:`FitResult` dataclass is the single object every fit engine in
afmkit returns. It bundles the best-fit parameters, their 1-sigma
uncertainties, the full covariance matrix, the residual statistics
(chi-square, reduced chi-square, AIC, BIC, R²), the residual array, the
``x`` axis used in the fit, the model evaluated on that axis, and a
free-form metadata bag (e.g. fit range, success flag, solver message).

A :class:`FitResult` is intentionally **immutable** in spirit: every
``to_dict()`` round-trip produces an equivalent value, so the class is
safe to persist (HDF5, JSON, parquet) and to share across processes
without worrying about accidental in-place mutation.

Why a dataclass, not a pydantic model?
---------------------------------------
The fitting engine produces the result internally; we don't validate
external input here. A plain ``@dataclass`` keeps the per-instance
overhead low and avoids the pydantic startup cost in tight inner loops
(e.g. batch fits over hundreds of curves). Serialisation helpers
(``to_dict`` / ``from_dict``) handle the cross-process boundary
explicitly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

if TYPE_CHECKING:
    import lmfit.model

__all__ = ["FitResult"]


@dataclass
class FitResult:
    """Outcome of a single curve fit.

    Attributes
    ----------
    model_name
        Human-readable name of the model (e.g. ``"wlc"``).
    params
        Best-fit parameter values, keyed by parameter name. The keys
        match the model's :attr:`~afmkit.models.base.PolymerModel.param_names`.
    stderr
        1-sigma uncertainties for ``params`` (same keys). Parameters
        whose uncertainty was not estimated by the solver (fixed
        parameters, or fits where the covariance could not be inverted)
        are reported as ``float("nan")`` so the dict shape always
        matches :attr:`params`.
    covariance
        Full ``(n_params, n_params)`` covariance matrix, or ``None`` if
        the solver could not estimate it.
    chi_square
        Residual sum of squares (after any weighting). For the
        Levenberg-Marquardt solver this equals ``sum((y_data -
        y_model)**2)`` when no weights are supplied, and ``sum(weights
        * (y_data - y_model)**2)`` otherwise.
    reduced_chi_square
        ``chi_square / (n_data - n_params)`` — the standard reduced
        chi-square used in goodness-of-fit testing.
    n_data
        Number of data points used in the fit (after any ``x_range``
        restriction and after dropping non-finite entries).
    n_params
        Number of **varying** parameters (i.e. parameters actually
        optimised). Fixed parameters do not count.
    aic
        Akaike Information Criterion, evaluated under the assumption of
        i.i.d. Gaussian residuals. Smaller is better.
    bic
        Bayesian Information Criterion, same units. Penalises model
        complexity more strongly than AIC when ``n_data`` is large.
    residual
        ``y_data - y_model`` for every data point, in the same order
        as :attr:`x_fit`.
    x_fit
        The ``x`` values used in the fit (after any ``x_range``
        restriction). Shape: ``(n_data,)``.
    y_fit
        ``model(x_fit, **params)``. Shape: ``(n_data,)``.
    metadata
        Free-form bag. The engine always populates ``"success"`` (bool)
        and ``"message"`` (str). Callers are free to add more keys
        (e.g. fit method, number of function evaluations, fit range
        before truncation).
    """

    model_name: str
    params: dict[str, float]
    stderr: dict[str, float]
    covariance: np.ndarray | None
    chi_square: float
    reduced_chi_square: float
    n_data: int
    n_params: int
    aic: float
    bic: float
    residual: np.ndarray
    x_fit: np.ndarray
    y_fit: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    #: Schema version for :meth:`to_dict` / :meth:`from_dict`. Bump when
    #: the on-disk format changes in a non-backwards-compatible way.
    SCHEMA_VERSION: ClassVar[int] = 1

    # -- Derived statistics ----------------------------------------------

    @property
    def r_squared(self) -> float:
        """Coefficient of determination, R² ∈ (-∞, 1].

        Defined as ``1 - SS_res / SS_tot``, where ``SS_res`` is the sum
        of squared residuals and ``SS_tot`` is the total sum of squares
        of ``y_data`` around its mean. A value of 1 means a perfect
        fit; negative values indicate the model is worse than a
        constant predictor and are a strong sign of model
        misspecification or a failed fit.

        Returns ``float("nan")`` when ``SS_tot`` is zero (all
        ``y_data`` identical) — R² is undefined in that degenerate
        case.
        """
        y = self.y_fit + self.residual  # y_data == y_fit + residual
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        ss_res = float(np.sum(self.residual**2))
        if ss_tot == 0.0:
            return float("nan")
        return 1.0 - ss_res / ss_tot

    # -- Human-readable rendering ----------------------------------------

    def summary(self) -> str:
        """Return a multi-line, human-readable description of the fit.

        The output is designed to be both copy-pasteable into a lab
        notebook and (roughly) greppable — every parameter line begins
        with the parameter name so ``grep '^p:'`` etc. work.

        Returns
        -------
        str
            A multi-line string. Always non-empty.
        """
        lines: list[str] = []
        title = f"FitResult[{self.model_name}]"
        lines.append(title)
        lines.append("-" * len(title))
        success = bool(self.metadata.get("success", True))
        status = "ok" if success else "FAILED"
        lines.append(f"  status            : {status}")
        if not success and self.metadata.get("message"):
            lines.append(f"  message           : {self.metadata['message']}")
        lines.append(f"  n_data            : {self.n_data}")
        lines.append(f"  n_params (vary)   : {self.n_params}")
        lines.append(f"  chi_square        : {self.chi_square:.6g}")
        lines.append(f"  reduced_chi_square: {self.reduced_chi_square:.6g}")
        lines.append(f"  AIC               : {self.aic:.6g}")
        lines.append(f"  BIC               : {self.bic:.6g}")
        lines.append(f"  R^2               : {self.r_squared:.6g}")
        lines.append("  parameters:")
        # Pad parameter names so columns line up across lines.
        name_w = max((len(n) for n in self.params), default=0)
        for name, value in self.params.items():
            err = self.stderr.get(name, float("nan"))
            err_str = f"{err:.4g}" if np.isfinite(err) else "  n/a"
            lines.append(f"    {name:<{name_w}} = {value:.6g}  +/- {err_str}")
        if self.metadata:
            # Surface the most common diagnostic keys, then the rest.
            extra = {k: v for k, v in self.metadata.items() if k not in {"success", "message"}}
            if extra:
                lines.append("  metadata:")
                for k, v in extra.items():
                    lines.append(f"    {k}: {v!r}")
        return "\n".join(lines)

    # -- Serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly ``dict`` representation.

        Numpy arrays are serialised as nested Python lists; numpy
        scalars become native Python floats / ints; ``None`` stays
        ``None``. The result is safe to feed to ``json.dumps`` or to
        write to a parquet/HDF5 cell.
        """
        return {
            "schema_version": self.SCHEMA_VERSION,
            "model_name": self.model_name,
            "params": dict(self.params),
            "stderr": dict(self.stderr),
            "covariance": _tolist(self.covariance),
            "chi_square": float(self.chi_square),
            "reduced_chi_square": float(self.reduced_chi_square),
            "n_data": int(self.n_data),
            "n_params": int(self.n_params),
            "aic": float(self.aic),
            "bic": float(self.bic),
            "residual": _tolist(self.residual),
            "x_fit": _tolist(self.x_fit),
            "y_fit": _tolist(self.y_fit),
            "metadata": _jsonable(self.metadata),
        }

    def to_json(self) -> str:
        """Convenience wrapper around :meth:`to_dict` + ``json.dumps``."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FitResult:
        """Reconstruct a :class:`FitResult` from :meth:`to_dict` output.

        The reverse of :meth:`to_dict`. Numeric values are coerced to
        ``float64`` / ``int`` so callers don't need to worry about
        JSON's silent downcast to ``int`` for whole numbers.

        Raises
        ------
        ValueError
            If ``d["schema_version"]`` is newer than what this version
            of afmkit knows how to read.
        """
        version = d.get("schema_version", 1)
        if version > cls.SCHEMA_VERSION:
            raise ValueError(
                f"FitResult schema version {version} is newer than supported "
                f"({cls.SCHEMA_VERSION}); upgrade afmkit."
            )
        return cls(
            model_name=str(d["model_name"]),
            params={k: float(v) for k, v in d["params"].items()},
            stderr={k: float(v) for k, v in d["stderr"].items()},
            covariance=_toarray(d.get("covariance")),
            chi_square=float(d["chi_square"]),
            reduced_chi_square=float(d["reduced_chi_square"]),
            n_data=int(d["n_data"]),
            n_params=int(d["n_params"]),
            aic=float(d["aic"]),
            bic=float(d["bic"]),
            residual=np.asarray(d["residual"], dtype=np.float64),
            x_fit=np.asarray(d["x_fit"], dtype=np.float64),
            y_fit=np.asarray(d["y_fit"], dtype=np.float64),
            metadata=dict(d.get("metadata", {})),
        )

    @classmethod
    def from_lmfit(
        cls,
        model_name: str,
        result: lmfit.model.ModelResult,
        x: np.ndarray,
        y: np.ndarray,
    ) -> FitResult:
        """Build a :class:`FitResult` from an ``lmfit`` ``ModelResult``.

        Parameters
        ----------
        model_name
            Label stored on the :class:`FitResult` (e.g. ``"wlc"``).
        result
            The :class:`lmfit.model.ModelResult` returned by
            ``Model.fit()``. All of its goodness-of-fit statistics
            (``chisqr``, ``redchi``, ``aic``, ``bic``, ``ndata``,
            ``nvarys``, ``rsquared``, ``best_values``, ``covar``) are
            copied across; missing stderr / covariance are reported as
            ``NaN`` / ``None`` rather than dropped, so the result
            always has a uniform shape.
        x, y
            The data that was passed to ``Model.fit``. ``x`` is stored
            as :attr:`x_fit`; ``y`` is kept implicitly via the
            ``residual = y - y_fit`` identity.
        """
        # Best values + 1-sigma uncertainties, in the order they appear
        # on the result. `result.params` is a dict-like; iterating it
        # preserves the lmfit insertion order (parameter declaration
        # order), which matches the model's `param_names`.
        best = result.best_values
        stderr: dict[str, float] = {}
        params: dict[str, float] = {}
        for name, value in best.items():
            params[name] = float(value)
            p = result.params[name]
            # Fixed parameters (vary=False) have p.stderr == 0.0 in
            # lmfit, not None. Treat both "no stderr" cases as NaN
            # so downstream code can use a single `math.isnan` check.
            if not p.vary or p.stderr is None:
                stderr[name] = float("nan")
            else:
                stderr[name] = float(p.stderr)
        # Fixed parameters (vary=False) don't appear in `best_values`.
        # Backfill them from the initial parameters so `params` always
        # has the full model parameter set.
        for name, p in result.params.items():
            if name not in params:
                params[name] = float(p.value)
                stderr[name] = float("nan")

        metadata: dict[str, Any] = {
            "success": bool(result.success),
            "message": str(result.message),
            "method": str(getattr(result, "method", "leastsq")),
            "nfev": int(getattr(result, "nfev", 0) or 0),
            "nfree": int(getattr(result, "nfree", result.ndata - result.nvarys)),
        }

        # `result.covar` may be None if the covariance could not be
        # estimated; downstream code should treat that as "no
        # uncertainties available" rather than a zero matrix.
        covar = result.covar
        covar_arr = np.asarray(covar, dtype=np.float64) if covar is not None else None

        # `result.best_fit` is the model evaluated at the independent
        # variables used in the fit — exactly what we want for
        # `y_fit`. Fall back to evaluating the model if for some
        # reason it isn't populated.
        y_fit = result.best_fit
        if y_fit is None:
            y_fit = np.full_like(x, np.nan, dtype=np.float64)
        y_fit_arr = np.asarray(y_fit, dtype=np.float64)

        # Residual: prefer lmfit's own array (it accounts for weights
        # and any internal scaling), fall back to a direct difference.
        residual = result.residual
        if residual is None:
            residual = np.asarray(y, dtype=np.float64) - y_fit_arr
        residual_arr = np.asarray(residual, dtype=np.float64)

        return cls(
            model_name=model_name,
            params=params,
            stderr=stderr,
            covariance=covar_arr,
            chi_square=float(result.chisqr),
            reduced_chi_square=float(result.redchi),
            n_data=int(result.ndata),
            n_params=int(result.nvarys),
            aic=float(result.aic),
            bic=float(result.bic),
            residual=residual_arr,
            x_fit=np.asarray(x, dtype=np.float64),
            y_fit=y_fit_arr,
            metadata=metadata,
        )

    # -- Misc ------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FitResult(model={self.model_name!r}, "
            f"n_data={self.n_data}, n_params={self.n_params}, "
            f"chi2={self.chi_square:.4g}, R^2={self.r_squared:.4g})"
        )


# -- Helpers --------------------------------------------------------------


def _tolist(arr: np.ndarray | None) -> list[Any] | None:
    """Convert a numpy array to a nested Python list, or pass ``None`` through.

    The conversion goes through ``tolist()`` rather than ``.tolist()`` on
    the bare array so the dtype is normalised — a 1-D array becomes a
    flat list of Python scalars, a 2-D array a list of lists, etc.
    """
    if arr is None:
        return None
    out: list[Any] = np.asarray(arr).tolist()
    return out


def _toarray(obj: Any) -> np.ndarray | None:
    """Inverse of :func:`_tolist`: nested list → ``np.ndarray`` (or None)."""
    if obj is None:
        return None
    return np.asarray(obj, dtype=np.float64)


def _jsonable(obj: Any) -> Any:
    """Recursively convert ``obj`` into a JSON-friendly Python structure.

    Handles the common cases — numpy arrays, numpy scalars, tuples —
    that survive in a typical fit metadata dict. Anything else is
    returned as-is, which means the caller is responsible for not
    stuffing arbitrary non-serialisable objects into ``metadata``.
    """
    if obj is None or isinstance(obj, str | bool | int | float):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):  # numpy scalar
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    return obj
