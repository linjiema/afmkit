"""Curve fitting engine: :class:`LmfitEngine` and the :func:`fit` helper.

This module is the bridge between afmkit's model layer
(:mod:`afmkit.models`) and the underlying :mod:`lmfit` solver. The
:class:`LmfitEngine` wraps an arbitrary :class:`PolymerModel` in an
:class:`lmfit.model.Model`, applies the model's bounds and starting
point, runs the Levenberg-Marquardt solver, and converts the
:class:`lmfit.model.ModelResult` back into our :class:`FitResult`
dataclass.

Failure mode
------------
A fit that does not converge (e.g. NaN in the data, completely wrong
starting point) **does not raise** — it returns a :class:`FitResult`
with ``metadata["success"] = False`` and ``metadata["message"]`` set
to the lmfit diagnostic. The caller can then decide whether to drop
the curve, retry with a different starting point, or surface a UI
warning. This is deliberate: in a 100-curve batch fit, raising would
abort the whole batch and lose the (good) fits that came before.

The exception is a *programming* error — passing a model with no
parameters, say — which still raises :class:`TypeError` /
:class:`ValueError` because nothing the caller does at runtime can
recover from it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import lmfit
import numpy as np

from afmkit.fitting.report import FitResult
from afmkit.models import get_model

if TYPE_CHECKING:
    from afmkit.core.curve import ForceCurve
    from afmkit.models.base import PolymerModel

__all__ = ["LmfitEngine", "fit"]


class LmfitEngine:
    """Wrapper around :mod:`lmfit` that fits a :class:`PolymerModel` to data.

    The engine is constructed once per model and re-used across many
    fits — the underlying :class:`lmfit.model.Model` is built eagerly
    in :meth:`__init__` and the per-fit work is just parameter setup
    and solver invocation.

    Parameters
    ----------
    model
        Any :class:`PolymerModel`. The engine reads ``param_names``,
        ``param_bounds``, and ``__call__`` from it. The model is
        stored as-is; no internal state is shared between the
        afmkit-side model and the wrapped ``lmfit.Model`` (the
        ``lmfit.Model`` keeps its own copy of the call signature).

    Attributes
    ----------
    model
        The :class:`PolymerModel` instance the engine was built for.
    lm_model
        The underlying :class:`lmfit.model.Model` — exposed for power
        users (e.g. plotting the model with lmfit's helpers). Most
        callers should not need it.
    """

    def __init__(self, model: PolymerModel) -> None:
        self.model: PolymerModel = model
        self.lm_model: lmfit.Model = _build_lmfit_model(model)

    # -- Public API -------------------------------------------------------

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        p0: dict[str, float] | None = None,
        x_range: tuple[float, float] | None = None,
        weights: np.ndarray | None = None,
    ) -> FitResult:
        """Fit the model to ``(x, y)`` and return a :class:`FitResult`.

        Parameters
        ----------
        x, y
            The data. 1-D ``np.ndarray`` of equal length. ``x`` must
            be finite (NaN in the independent variable is a
            programming error and raises :class:`ValueError`); ``y``
            is allowed to contain NaN / inf, in which case the fit
            is reported as failed (``metadata["success"] = False``)
            rather than raising — the caller is best placed to decide
            whether to drop the curve or retry after cleaning.
        p0
            Optional initial parameter values, keyed by
            :attr:`PolymerModel.param_names`. Missing keys fall back
            to the model's :meth:`~PolymerModel.guess_params` output;
            extra keys are ignored. Values are **clamped** to the
            model's :attr:`~PolymerModel.param_bounds` before being
            handed to the solver — the spec says "clamp
            parameters in ``[lo, hi]``", and lmfit's ``Parameter``
            only clamps on set, not on the initial dict we pass in.
        x_range
            Optional ``(x_min, x_max)`` to restrict the fit to data
            within the extension range. Implemented as a pre-fit
            filter on the data passed to the solver, so the reported
            ``n_data`` reflects the actual sample size used.
        weights
            Optional 1-D array of per-point weights, same length as
            ``x``. Forwarded to ``lmfit`` unchanged; the convention
            is that the solver minimises
            ``sum(weights * (y_data - y_model)**2)``.

        Returns
        -------
        FitResult
            Always returned. On success, ``metadata["success"]`` is
            ``True``; on a non-converging fit (or a NaN/Inf in
            ``y``) it is ``False`` and ``metadata["message"]``
            carries the diagnostic.
        """
        x_arr = _ensure_1d_finite("x", x)
        # `y` is allowed to contain NaN/Inf — a noisy data point is
        # an expected failure mode, not a bug. The downstream solver
        # call will surface it as a failed fit.
        y_arr = _ensure_1d("y", y)
        if y_arr.shape != x_arr.shape:
            raise ValueError(
                f"x and y must have the same length; got {x_arr.shape} vs {y_arr.shape}"
            )
        if weights is not None:
            w_arr = np.asarray(weights, dtype=np.float64)
            if w_arr.shape != x_arr.shape:
                raise ValueError(
                    f"weights must have the same length as x; got {w_arr.shape} vs {x_arr.shape}"
                )
        else:
            w_arr = None

        if x_range is not None:
            x_min, x_max = x_range
            if x_min > x_max:
                raise ValueError(f"x_range: x_min ({x_min}) must be <= x_max ({x_max})")
            mask = (x_arr >= x_min) & (x_arr <= x_max)
            if not np.any(mask):
                raise ValueError(f"x_range {x_range} selects no data points")
            x_arr = x_arr[mask]
            y_arr = y_arr[mask]
            if w_arr is not None:
                w_arr = w_arr[mask]

        # Compose the starting point: `p0` overrides the model's
        # `guess_params` output. Both are clamped to bounds.
        try:
            guess = self.model.guess_params(x_arr, y_arr)
        except Exception as exc:  # pragma: no cover - defensive
            # guess_params is a user extension point; a failure here
            # should not take down the whole fit. Fall back to a
            # midrange guess.
            guess = {
                name: 0.5 * (lo + hi)
                for name, (lo, hi) in zip(
                    self.model.param_names, self.model.param_bounds, strict=True
                )
            }
            guess_err: dict[str, Any] = {"guess_error": str(exc)}
        else:
            guess_err = {}

        if p0:
            for k, v in p0.items():
                guess[k] = v
        start = _clamp_to_bounds(guess, self.model.param_names, self.model.param_bounds)

        # Build the lmfit Parameters with the clamped starting point
        # and the model's bounds.
        params = self.lm_model.make_params()
        for name, (lo, hi) in zip(self.model.param_names, self.model.param_bounds, strict=True):
            params[name].set(value=start[name], min=lo, max=hi)

        # The actual solver call. We catch NaN errors here because
        # lmfit raises a bare ValueError with no return value when
        # the model function evaluates to NaN — we want to translate
        # that into a "failed fit" FitResult, not a crash.
        try:
            result = self.lm_model.fit(
                y_arr,
                params,
                x=x_arr,
                weights=w_arr,
                method="leastsq",
                nan_policy="raise",
            )
        except ValueError as exc:
            # NaN in the data or in the model output during evaluation.
            return _failed_result(
                model_name=self.model.__class__.__name__,
                x=x_arr,
                y=y_arr,
                params=params,
                message=f"ValueError: {exc}",
                extra={"fit_range": list(x_range)} if x_range is not None else {},
            )
        except Exception as exc:  # pragma: no cover - defensive
            # Any other solver-level failure — e.g. an unforseen
            # scipy exception. Still produce a FitResult so the caller
            # can decide.
            return _failed_result(
                model_name=self.model.__class__.__name__,
                x=x_arr,
                y=y_arr,
                params=params,
                message=f"{type(exc).__name__}: {exc}",
                extra={"fit_range": list(x_range)} if x_range is not None else {},
            )

        fit_result = FitResult.from_lmfit(
            model_name=self.model.__class__.__name__,
            result=result,
            x=x_arr,
            y=y_arr,
        )
        # Surface the user-facing start / range on the result.
        if x_range is not None:
            fit_result.metadata.setdefault("fit_range", list(x_range))
        fit_result.metadata.setdefault("initial_params", start)
        if guess_err:
            # guess_params raised; record but don't fail the fit.
            fit_result.metadata.setdefault("guess_warnings", guess_err)
        return fit_result

    def residual(self, params: dict[str, float], x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return ``y - model(x, **params)`` for the given parameter dict.

        Useful for downstream code that wants to use lmfit's
        differential-evolution or MCMC solvers on top of the same
        model — they need a residual function rather than a
        :class:`FitResult`.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        y_model = self.model(x_arr, **params)
        return y_arr - np.asarray(y_model, dtype=np.float64)


# -- Top-level convenience ------------------------------------------------


def fit(
    curve: ForceCurve,
    *,
    model: str = "wlc",
    x_range: tuple[float, float] | None = None,
    p0: dict[str, float] | None = None,
    weights: np.ndarray | None = None,
) -> FitResult:
    """Fit a polymer model to a :class:`ForceCurve` and return a :class:`FitResult`.

    Convenience wrapper around :class:`LmfitEngine` that resolves a
    model by name from the registry, instantiates the engine, and
    runs the fit. The same :class:`ForceCurve` can be re-fit with
    different models / ranges by calling ``fit`` repeatedly — the
    curve is treated as immutable.

    Parameters
    ----------
    curve
        The :class:`~afmkit.core.curve.ForceCurve` to fit.
    model
        Registry key of the polymer model (default ``"wlc"``).
    x_range
        Optional ``(x_min, x_max)`` in nm, restricting the fit to a
        sub-range of the curve. The data is sliced before fitting
        so the reported :attr:`FitResult.n_data` matches the actual
        sample size.
    p0
        Optional starting-point overrides.
    weights
        Optional per-point weights, same length as ``curve.extension``.

    Returns
    -------
    FitResult
        Always returned (never raises on a non-converging fit).
        ``metadata["success"]`` is ``True`` for a converged fit,
        ``False`` otherwise.
    """
    m = get_model(model)
    engine = LmfitEngine(m)
    return engine.fit(
        curve.extension,
        curve.force,
        p0=p0,
        x_range=x_range,
        weights=weights,
    )


# -- Helpers --------------------------------------------------------------


def _build_lmfit_model(model: PolymerModel) -> lmfit.Model:
    """Build a :class:`lmfit.model.Model` from a :class:`PolymerModel`.

    The bridging function takes ``x`` positionally and forwards the
    model's parameters as **explicit** keyword arguments — lmfit
    introspects the function signature with :func:`inspect.signature`
    to find the parameter names, and a ``**kwargs`` catch-all defeats
    that. We generate the function dynamically with ``exec`` so the
    afmkit-side model protocol (``__call__(x, **params)``) can stay
    keyword-agnostic while the lmfit side gets a concrete signature.
    """

    param_names = list(model.param_names)
    # Body of the generated function. Reads cleanly as a function:
    #     def _afmkit_lmfit_bridge(x, p, L):
    #         return _model(np.asarray(x, dtype=np.float64), p=p, L=L)
    kwarg_str = ", ".join(f"{n}={n}" for n in param_names)
    src = (
        f"def _afmkit_lmfit_bridge(x, {', '.join(param_names)}):\n"
        f"    return _model(np.asarray(x, dtype=np.float64), {kwarg_str})\n"
    )
    ns: dict[str, Any] = {"_model": model, "np": np}
    exec(compile(src, f"<lmfit-bridge for {model.__class__.__name__}>", "exec"), ns)
    bridge = ns["_afmkit_lmfit_bridge"]

    lm_model = lmfit.Model(
        bridge,
        independent_vars=["x"],
        param_names=param_names,
    )
    # Register bounds as hints so `make_params()` knows the legal
    # range for every parameter.
    for name, (lo, hi) in zip(model.param_names, model.param_bounds, strict=True):
        lm_model.set_param_hint(name, min=lo, max=hi)
    return lm_model


def _ensure_1d_finite(name: str, arr: Any) -> np.ndarray:
    """Validate that ``arr`` is a 1-D finite float array.

    Borrowed from the same routine in :mod:`afmkit.core.curve` — kept
    local here so the fitting layer has no upward dependency on the
    curve layer. The error messages are slightly shorter.
    """
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {out.shape}")
    if out.size == 0:
        raise ValueError(f"{name} must contain at least one point")
    if not np.all(np.isfinite(out)):
        n_bad = int(np.sum(~np.isfinite(out)))
        raise ValueError(f"{name} contains {n_bad} non-finite value(s)")
    return out


def _ensure_1d(name: str, arr: Any) -> np.ndarray:
    """Validate that ``arr`` is a 1-D array (NaN / Inf allowed).

    Companion to :func:`_ensure_1d_finite` for arrays where non-finite
    values are a legitimate input (the fit is then expected to be
    reported as failed, not raise). The dtype is always coerced to
    ``float64`` so downstream code can rely on it.
    """
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {out.shape}")
    if out.size == 0:
        raise ValueError(f"{name} must contain at least one point")
    return out


def _clamp_to_bounds(
    guess: dict[str, float],
    param_names: tuple[str, ...],
    param_bounds: tuple[tuple[float, float], ...],
) -> dict[str, float]:
    """Clamp every parameter in ``guess`` to its closed ``[lo, hi]`` range.

    Missing keys are filled in with the midpoint of the bound — the
    model is then handed a complete starting dict. lmfit's
    ``Parameter.set`` would also clamp, but doing it explicitly here
    keeps the contract obvious to readers and survives a future
    swap-out of the underlying solver.
    """
    out: dict[str, float] = {}
    bounds_by_name = dict(zip(param_names, param_bounds, strict=True))
    for name, (lo, hi) in bounds_by_name.items():
        v = guess.get(name, 0.5 * (lo + hi))
        # NaN guard: if guess_params returned NaN (very unusual), fall
        # back to the midpoint so the solver at least has a finite
        # value to start from.
        if not np.isfinite(v):
            v = 0.5 * (lo + hi)
        out[name] = float(min(max(v, lo), hi))
    return out


def _failed_result(
    *,
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    params: lmfit.Parameters,
    message: str,
    extra: dict[str, Any] | None = None,
) -> FitResult:
    """Build a :class:`FitResult` representing a failed fit.

    Used by the engine when the solver raises (e.g. NaN in the
    data) rather than returning a non-converging ``ModelResult``.
    The returned object has the same shape as a successful result —
    arrays are filled with zeros / NaN, statistics are NaN, and
    ``metadata["success"]`` is ``False``.
    """
    n = int(x.size)
    n_params = sum(1 for p in params.values() if p.vary)
    metadata: dict[str, Any] = {"success": False, "message": message}
    if extra:
        metadata.update(extra)
    # `params` values are the (un-fitted) starting point — surface
    # them so the caller can see where the fit got stuck.
    metadata["initial_params"] = {name: float(p.value) for name, p in params.items()}

    nan = float("nan")
    return FitResult(
        model_name=model_name,
        params={name: float(p.value) for name, p in params.items()},
        stderr={name: nan for name in params},
        covariance=None,
        chi_square=nan,
        reduced_chi_square=nan,
        n_data=n,
        n_params=n_params,
        aic=nan,
        bic=nan,
        residual=np.full(n, nan, dtype=np.float64),
        x_fit=np.asarray(x, dtype=np.float64),
        y_fit=np.full(n, nan, dtype=np.float64),
        metadata=metadata,
    )
