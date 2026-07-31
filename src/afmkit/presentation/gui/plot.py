"""Matplotlib-backed Textual widget that renders a force-extension curve.

This module ships :class:`ForceExtensionPlot`, a small
:class:`textual.widget.Widget` that draws a force-extension trace into a
Pillow image via matplotlib's Agg backend, then exposes the image as
the widget's :meth:`render` output.

The widget is **headless** in the sense that it does not depend on a
running terminal: rendering only touches matplotlib + Pillow, and the
underlying :func:`_render_to_pillow` helper is callable from a unit
test without instantiating a Textual ``App``. The Textual renderable
layer (the :meth:`render` method) is a thin wrapper that yields a
``rich.console.Group`` of the title plus a terminal image
of the cached bitmap.

Why matplotlib + Pillow rather than rich's built-in image protocol?
------------------------------------------------------------------
Textual's image story is intentionally minimal — there is no first-class
PIL-to-rich adapter in the standard library. We use a custom
:class:`_PillowImageRenderable` that paints the bitmap with the
``▀`` half-block character (two image rows per terminal row) so the
plot looks like a proper figure on a modern terminal (Kitty, iTerm2,
Windows Terminal) and degrades to a colourless block on legacy ones.
The matplotlib side is the workhorse: it is what actually draws the
curve, the peak markers, and the WLC fit overlay.

Native terminal image protocol (v0.6+)
--------------------------------------
When the optional ``[plot-native]`` extra (``textual-image``) is
installed and the running terminal supports a native image
protocol — Sixel, Terminal Graphics Protocol (Kitty, iTerm2,
wezterm, Konsole), Halfcell, or Unicode — the widget can render
the matplotlib output through that protocol instead of the
half-block fallback.  This produces a crisper, full-colour
plot at the native terminal resolution instead of the
``▀`` approximation.

The default is **auto-detect** (``prefer_native=None``): use the
native path when the package is installed, fall back to the
half-block path otherwise.  The half-block path is the v0.4
default contract and works on every terminal without any
optional dependencies, so the ``[plot-native]`` extra is a
strict upgrade — never a regression.

Optional dependency
-------------------
``matplotlib`` and ``Pillow`` are *optional* ``[plot]`` extras —
the module imports cleanly without them. :meth:`ForceExtensionPlot.render_curve`
raises a clear :class:`ImportError` with the install command if either
is missing at call time.  ``textual-image`` is an additional
*optional* ``[plot-native]`` extra for the native image renderable.
"""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING, Any, ClassVar

# matplotlib.use("Agg") MUST be set before pyplot is imported. Agg is
# the headless backend — it renders to an in-memory buffer, which is
# what we need for a TUI plot widget (no display, no X server).
import matplotlib as _matplotlib
import numpy as np

_matplotlib.use("Agg")

try:  # pragma: no cover - exercised only when matplotlib / Pillow are missing
    import matplotlib.pyplot as _plt
    from PIL import Image as _PILImage

    _MPL_IMPORT_ERROR: ImportError | None = None
except ImportError as _exc:  # pragma: no cover
    _MPL_IMPORT_ERROR = _exc
    # Stubs so the rest of the module can still parse. At runtime the
    # ``_MPL_IMPORT_ERROR is not None`` guard in :meth:`render_curve`
    # short-circuits before any of these names are referenced, so the
    # placeholder values never get used. ``Any`` makes mypy accept the
    # conditional redefinition.
    _PILImage: Any = None  # type: ignore[no-redef]
    _plt: Any = None  # type: ignore[no-redef]


# textual is an optional [gui] extra; mirror the pattern from
# afmkit.presentation.gui.app — guard the import so the module loads
# even when textual is not installed (e.g. on a CI build with only
# the [plot] extra). At call time the ``_TEXTUAL_IMPORT_ERROR is None``
# guard short-circuits before any textual symbol is referenced.
try:  # pragma: no cover - import-time guard
    from textual.widget import Widget as _TextualWidget
except ImportError as _textual_exc:  # pragma: no cover
    _TEXTUAL_IMPORT_ERROR: ImportError | None = _textual_exc
    # ``object`` is a stand-in base class so the class body below can
    # still parse. The ``_TEXTUAL_IMPORT_ERROR is not None`` guard in
    # ``__init__`` short-circuits before the placeholder is ever used.
    _TextualWidget: Any = object  # type: ignore[no-redef]
else:
    _TEXTUAL_IMPORT_ERROR = None


# ``textual-image`` is an optional ``[plot-native]`` extra; same
# pattern as the ``textual`` and ``matplotlib`` guards above.  The
# module-level ``Image`` symbol on the ``textual_image.renderable``
# package is the auto-dispatch: it resolves to ``SixelImage`` /
# ``TGPImage`` (Kitty + iTerm) / ``HalfcellImage`` / ``UnicodeImage``
# depending on what the running terminal supports, so importing the
# module is enough — we don't need a separate detection step.  When
# the package is missing the half-block :class:`_PillowImageRenderable`
# below is the v0.4 fallback (zero-dep).
try:  # pragma: no cover - import-time guard
    import textual_image.renderable as _native_renderable_module

    _NATIVE_IMPORT_ERROR: ImportError | None = None
except ImportError as _native_exc:  # pragma: no cover
    _native_renderable_module: Any = None  # type: ignore[no-redef]
    _NATIVE_IMPORT_ERROR = _native_exc


from afmkit.analysis.peak_detection import Peak
from afmkit.core.curve import ForceCurve
from afmkit.fitting.report import FitResult
from afmkit.models.wlc import WLCModel

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["ForceExtensionPlot"]


#: Approximate number of matplotlib figure pixels per character cell.
#: 10 px/char is the same value the original Igor Pro plotting layer
#: used for the v0.1 lab macro; it produces a ~800x200 PNG for an
#: 80x20 widget, which is the resolution the lab's published figures
#: use for the same dataset.
_PX_PER_CHAR: int = 10

#: DPI used to translate matplotlib figure inches to PIL pixels. Held
#: constant so that a 10 px/char ``figsize`` translates cleanly to a
#: ``width_chars * 10``-pixel-wide image.
_DPI: int = 100

#: Default matplotlib style for the data trace (black, thin). Kept
#: deliberately conservative — the lab's published figures use the
#: same colour so screenshots from the TUI match screenshots from
#: the static ``afmkit plot`` CLI.
_DATA_LINE_KW: dict[str, Any] = {"color": "black", "linewidth": 1.0, "label": "data"}

#: Matplotlib style for the WLC fit overlay. Dashed blue, thicker
#: than the data line, so it stands out without dominating the figure.
_FIT_LINE_KW: dict[str, Any] = {
    "color": "tab:blue",
    "linewidth": 1.5,
    "linestyle": "--",
    "label": "WLC fit",
}

#: Number of points used to draw the WLC fit overlay. 200 is enough
#: for a smooth dashed line without hitting the asymptotic ``x = L``
#: region too closely (matplotlib drops the ``inf`` automatically).
_FIT_OVERLAY_N: int = 200


# -- Terminal-side renderable ---------------------------------------------


class _PillowImageRenderable:
    """Minimal rich renderable that paints a PIL image as half-block characters.

    Each terminal row encodes two image rows: the **top** half comes
    from the upper pixel's foreground colour, the **bottom** half
    from the lower pixel's background colour, joined by the
    ``▀`` (UPPER HALF BLOCK) character. This is the same pattern
    ``chafa``, ``viu``, and most terminal image viewers use — it
    gives roughly square aspect ratio in modern terminals and
    degrades gracefully to a single colour block in legacy ones.

    The renderable does not need to be perfect: the *test* surface
    for the widget is :func:`ForceExtensionPlot._render_to_pillow`
    (which returns the raw PIL image); this class only exists so
    the :meth:`ForceExtensionPlot.render` method has something
    non-empty to yield when a real Textual App mounts the widget.
    The Lead (or downstream code) can swap in a richer adapter
    (``rich_pixels``, ``textual-image``, etc.) without touching the
    rest of this module.
    """

    def __init__(self, image: Any) -> None:
        """Store the image to be painted.

        Parameters
        ----------
        image
            A ``PIL.Image.Image`` instance. Kept loosely typed (rather
            than ``PIL.Image.Image``) so this class is constructable
            in environments where Pillow is not installed — though in
            practice the parent widget's :meth:`render` would never
            invoke this without an image, so that case does not arise.
        """
        self._image = image

    def __rich_console__(self, console: Any, options: Any) -> Iterable[Any]:
        """Yield rich segments that paint the image as half-block characters.

        Parameters
        ----------
        console
            The active :class:`rich.console.Console` (typed loosely
            so the rich import stays optional).
        options
            The :class:`rich.console.ConsoleOptions` for the current
            rendering context. Used for the maximum width / height
            so the rendered output fits the cell the parent App has
            given us.

        Yields
        ------
        rich.segment.Segment
            Segments whose ``text`` is a stream of ``▀`` characters
            with per-cell background / foreground styles encoding
            the two pixel colours.
        """
        # Local imports keep the module importable when rich is missing.
        from rich.segment import Segment
        from rich.style import Style

        img = self._image
        w, h = img.size

        # The available cell area — fall back to 80x24 (the standard
        # terminal default) when the option is missing. We render
        # two image rows per terminal row, so the *target* image
        # height in pixels is 2 x max_height.
        max_w = int(options.max_width or console.width or 80)
        max_h = int(options.height or console.height or 24) * 2

        # Pick the smaller of the per-axis scales so the image keeps
        # its aspect ratio. We also clamp to 1.0 to avoid upscaling
        # tiny test bitmaps into pixelated mush.
        scale = min(max_w / w, max_h / h, 1.0)
        new_w = max(1, int(round(w * scale)))
        new_h = max(2, int(round(h * scale)))
        if new_h % 2:  # half-block rendering needs an even row count
            new_h += 1

        resized = img.resize((new_w, new_h))
        rgb = resized.convert("RGB")
        pixels = rgb.load()

        for y in range(0, new_h, 2):
            line: list[Segment] = []
            for x in range(new_w):
                r_t, g_t, b_t = pixels[x, y]
                r_b, g_b, b_b = pixels[x, y + 1] if y + 1 < new_h else (r_t, g_t, b_t)
                fg = f"rgb({r_t},{g_t},{b_t})"
                bg = f"rgb({r_b},{g_b},{b_b})"
                line.append(Segment("▀", style=Style(color=bg, bgcolor=fg)))
            line.append(Segment("\n"))
            yield Segment("")  # start of line
            yield from line


def _make_native_image_renderable(pil_image: Any) -> Any:
    """Wrap a Pillow image in a native terminal-image renderable.

    Uses :class:`textual_image.renderable.Image`, which auto-dispatches
    at import time to the best available protocol for the running
    terminal — Sixel, Terminal Graphics Protocol (Kitty + iTerm +
    wezterm + Konsole), Halfcell, or Unicode block characters.  We
    don't have to choose the protocol ourselves: the textual-image
    package does the runtime query and resolves ``Image`` to the
    right concrete class.

    Returns
    -------
    textual_image.renderable.Image
        An instance of whichever concrete class the runtime
        environment picked.  Implements ``__rich_console__`` so it
        can be yielded directly from a Textual ``Widget.render()``
        method.

    Raises
    ------
    ImportError
        If the ``[plot-native]`` extra (the ``textual-image`` package)
        is not installed.  The :func:`_choose_renderable` helper
        guards against this by checking the module-level import flag
        before calling.
    """
    if _native_renderable_module is None:
        raise ImportError(
            "Native terminal image rendering requires the [plot-native] "
            "extra (`pip install 'afmkit[plot-native]'`); the "
            "`textual-image` package is not installed."
        ) from _NATIVE_IMPORT_ERROR
    return _native_renderable_module.Image(pil_image)


def _choose_renderable(
    pil_image: Any,
    prefer_native: bool | None,
) -> Any:
    """Pick the best rich renderable for a Pillow image.

    Parameters
    ----------
    pil_image
        The bitmap to wrap.  Any value with the right interface
        (i.e. a ``PIL.Image.Image``) is accepted; the function
        does not introspect it.
    prefer_native
        ``True`` to force the native terminal-image renderable
        (Sixel / TGP / Halfcell via ``textual-image``); ``False``
        to force the v0.4 half-block :class:`_PillowImageRenderable`;
        ``None`` to auto-detect — use the native renderable when
        the ``[plot-native]`` extra is installed, fall back to
        the half-block renderable otherwise.

    Returns
    -------
    rich renderable
        Either a :class:`_PillowImageRenderable` or a
        :class:`textual_image.renderable.Image` (the concrete
        class depends on the runtime terminal capability).

    Raises
    ------
    ImportError
        If ``prefer_native=True`` and the ``[plot-native]`` extra
        is not installed.  The error message points to the
        ``afmkit[plot-native]`` extra so the user can install the
        missing dependency.  For ``prefer_native=None`` / ``False``
        the import is never required, so the half-block path always
        works on a minimal install.
    """
    if prefer_native is False:
        return _PillowImageRenderable(pil_image)
    if prefer_native is True:
        return _make_native_image_renderable(pil_image)
    # prefer_native is None: auto-detect.
    if _native_renderable_module is not None:
        return _make_native_image_renderable(pil_image)
    return _PillowImageRenderable(pil_image)


# -- Widget ---------------------------------------------------------------


# Always inherit from ``_TextualWidget`` (the runtime textual symbol
# or the ``object`` stub) so the rest of the class body can subclass
# without a mypy [misc] error in the ``afmkit.presentation.gui.*``
# override. The ``_TEXTUAL_IMPORT_ERROR is not None`` guard in
# ``__init__`` raises a clean ImportError if textual is missing,
# matching the pattern from ``afmkit.presentation.gui.app``.
class ForceExtensionPlot(_TextualWidget):
    """A static matplotlib plot of a force-extension curve.

    Renders into a Pillow image via matplotlib's Agg backend, then
    exposes the image as a Textual renderable. The widget is
    headless: it does not depend on a live terminal and can be
    exercised in unit tests without spinning up a Textual App.

    Parameters
    ----------
    width
        Plot width in characters (default 80). The matplotlib figure
        is sized to roughly ``width x 10`` pixels — see
        :data:`_PX_PER_CHAR`.
    height
        Plot height in lines (default 20). Same scaling, so the
        resulting bitmap is ``height x 10`` pixels tall.
    title
        Optional title shown above the plot (rendered as a bold
        :class:`rich.text.Text` line in :meth:`render` and used as
        the matplotlib axes title in :meth:`render_curve`).
    name
        Forwarded to the underlying :class:`textual.widget.Widget`.
    id
        Forwarded to the underlying :class:`textual.widget.Widget`.
    classes
        Forwarded to the underlying :class:`textual.widget.Widget`.

    Notes
    -----
    The widget caches the most recent Pillow bitmap in
    :attr:`_image`; :meth:`render` is cheap on a no-op update because
    it only re-wraps the cached image. :meth:`clear` resets the
    cache to the empty placeholder state.
    """

    DEFAULT_CSS: ClassVar[str] = ""

    def __init__(
        self,
        width: int = 80,
        height: int = 20,
        title: str = "",
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        prefer_native: bool | None = None,
    ) -> None:
        """Initialise the widget with the requested character dimensions.

        The matplotlib / Pillow imports are deferred to
        :meth:`render_curve`, so constructing the widget does not
        require ``matplotlib`` or ``Pillow`` to be installed —
        the constructor only needs the optional ``textual`` package.
        The cached :attr:`_image` starts as ``None`` and the
        :meth:`render` placeholder shows the literal string
        ``"no data"`` until the first :meth:`render_curve` call.

        Parameters
        ----------
        width
            Plot width in characters (default 80).
        height
            Plot height in lines (default 20).
        title
            Optional title shown above the plot.
        name, id, classes
            Forwarded to the underlying :class:`textual.widget.Widget`.
        prefer_native
            ``True`` to force the native terminal-image renderable
            (Sixel / Kitty / iTerm graphics via the
            ``[plot-native]`` extra, ``textual-image`` package);
            ``False`` to force the v0.4 half-block
            :class:`_PillowImageRenderable`; ``None`` (default) to
            auto-detect — use the native renderable when the
            ``[plot-native]`` extra is installed, fall back to
            the half-block renderable otherwise.

            ``None`` is the right choice for almost all callers:
            the half-block path works on every terminal, and the
            native path is a strict upgrade when both the
            ``[plot-native]`` extra is installed and the terminal
            supports a native image protocol.
        """
        if _TEXTUAL_IMPORT_ERROR is not None:
            raise ImportError(
                "Textual is required for the ForceExtensionPlot widget. "
                "Install with `pip install 'afmkit[gui]'`."
            ) from _TEXTUAL_IMPORT_ERROR
        if prefer_native not in (None, True, False):
            raise TypeError(f"prefer_native must be None, True, or False; got {prefer_native!r}")
        super().__init__(name=name, id=id, classes=classes)
        if int(width) <= 0 or int(height) <= 0:
            raise ValueError(
                f"width and height must be positive; got width={width}, height={height}"
            )
        self._width_chars: int = int(width)
        self._height_chars: int = int(height)
        self._title: str = str(title)
        self._prefer_native: bool | None = prefer_native
        # The cached Pillow image; ``None`` means "no data — show
        # the empty placeholder". Always a fresh copy on assignment
        # so the caller can drop their reference to the bytes buffer.
        self._image: Any = None

    # -- Public API ------------------------------------------------------

    def render_curve(
        self,
        curve: ForceCurve,
        peaks: list[Peak] | None = None,
        fit: FitResult | None = None,
        x_range: tuple[float, float] | None = None,
    ) -> None:
        """Render a (curve, peaks, fit) triplet into the widget.

        If ``peaks`` is non-empty, mark each peak with a red
        vertical line and a dot at ``(peak.extension, peak.force)``.

        If ``fit`` is provided and :attr:`FitResult.model_name` is
        ``"wlc"``, overlay the fitted WLC curve from
        ``x_range[0]`` to ``x_range[1]`` (or the curve's natural
        range if ``x_range`` is ``None``). Other model names are
        silently ignored — the v0.3 widget only knows about WLC,
        and silently skipping is friendlier than raising in a TUI
        that is just trying to refresh its display.

        Uses matplotlib's Agg backend to draw into a Pillow image
        sized to the widget's character dimensions. The image is
        then exposed as the widget's renderable via rich's Image
        protocol and cached on :attr:`_image` for cheap re-renders.

        Parameters
        ----------
        curve
            The :class:`~afmkit.core.curve.ForceCurve` to plot.
        peaks
            Optional list of :class:`~afmkit.analysis.peak_detection.Peak`
            instances to mark on the trace. ``None`` and ``[]`` are
            equivalent — no markers drawn.
        fit
            Optional :class:`~afmkit.fitting.report.FitResult`. Only
            WLC fits are currently overlaid; other model names are
            silently ignored.
        x_range
            Optional ``(x_min, x_max)`` tuple in nm that restricts
            the plot's x-axis. When ``None`` (the default) the
            plot spans the full curve range.

        Raises
        ------
        ImportError
            If ``matplotlib`` or ``Pillow`` is not installed. The
            error message includes the exact ``pip install`` command
            needed to enable the widget.
        """
        if _MPL_IMPORT_ERROR is not None:
            raise ImportError(
                "matplotlib is required for the ForceExtensionPlot widget. "
                "Install with `pip install 'afmkit[plot]'`."
            ) from _MPL_IMPORT_ERROR
        # Delegate the actual draw to the public-ish helper so unit
        # tests can call it directly with explicit pixel dimensions
        # and skip the widget's own ``_width_chars`` / ``_height_chars``
        # defaults.
        self._image = self._render_to_pillow(
            curve=curve,
            peaks=peaks,
            fit=fit,
            x_range=x_range,
            width_px=self._width_chars * _PX_PER_CHAR,
            height_px=self._height_chars * _PX_PER_CHAR,
        )
        # Ask Textual to redraw on the next refresh tick. The call is
        # a no-op when the widget isn't mounted (e.g. in a unit test
        # that calls render_curve() and then introspects _image
        # directly), so we don't need to guard it.
        self.refresh()

    def clear(self) -> None:
        """Clear the plot.

        Subsequent :meth:`render` calls show an empty placeholder
        (the literal string ``"no data"``); the cached
        :attr:`_image` is reset to ``None`` so the test surface can
        detect the cleared state without a pixel comparison.
        """
        self._image = None
        self.refresh()

    # -- Internals -------------------------------------------------------

    def _render_to_pillow(
        self,
        curve: ForceCurve,
        peaks: list[Peak] | None,
        fit: FitResult | None,
        x_range: tuple[float, float] | None,
        width_px: int,
        height_px: int,
    ) -> Any:
        """Build a Pillow image of the curve + (peaks) + (fit overlay).

        This is the testable workhorse behind :meth:`render_curve`.
        It is intentionally a plain function (not bound to the
        widget instance) so the unit tests can call it with
        explicit pixel dimensions and skip the Textual refresh
        layer entirely. The widget dimension math
        (``width_chars * _PX_PER_CHAR``) lives in
        :meth:`render_curve` so a test that wants a 200x150 bitmap
        can ask for one without instantiating an 80x20 widget.

        Parameters
        ----------
        curve
            The :class:`~afmkit.core.curve.ForceCurve` to plot.
        peaks
            Optional list of :class:`~afmkit.analysis.peak_detection.Peak`.
            ``None`` and ``[]`` are equivalent.
        fit
            Optional :class:`~afmkit.fitting.report.FitResult`. Only
            WLC fits are overlaid.
        x_range
            Optional ``(x_min, x_max)`` in nm. When given, sets the
            matplotlib ``xlim`` to this range.
        width_px, height_px
            Target bitmap size in pixels. The matplotlib figure is
            sized at ``width_px / _DPI`` x ``height_px / _DPI``
            inches, and the resulting PNG is the requested pixel
            dimensions.

        Returns
        -------
        PIL.Image.Image
            A Pillow image of the plot. The image is a fresh
            ``copy()`` of the matplotlib buffer — callers can drop
            the underlying ``BytesIO`` reference without losing
            pixels.
        """
        ext = np.asarray(curve.extension, dtype=np.float64)
        force = np.asarray(curve.force, dtype=np.float64)
        xlo = float(ext.min())
        xhi = float(ext.max())
        if x_range is not None:
            xlo, xhi = float(x_range[0]), float(x_range[1])

        # ``figsize`` is in inches; combined with ``_DPI`` the
        # resulting raster is ``width_px`` x ``height_px`` pixels
        # — no bbox_inches="tight" magic that would silently
        # resize the canvas and break the test's dimension check.
        fig_w_in = max(width_px, 1) / _DPI
        fig_h_in = max(height_px, 1) / _DPI
        fig, ax = _plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=_DPI)
        try:
            # The data trace. matplotlib drops +/-inf from the
            # WLC singularity automatically, so we don't have to
            # filter the curve ourselves before plotting.
            ax.plot(ext, force, **_DATA_LINE_KW)

            if x_range is not None:
                ax.set_xlim(xlo, xhi)

            if peaks:
                for p in peaks:
                    # Vertical guide line in red, with a dot at the
                    # actual peak position. The line is thin
                    # (linewidth 0.6) and slightly transparent so
                    # a cluster of nearby peaks does not occlude
                    # the underlying data trace.
                    ax.axvline(p.extension, color="red", linewidth=0.6, alpha=0.7)
                    ax.plot([p.extension], [p.force], marker="o", color="red", markersize=4)

            if fit is not None and fit.model_name == "wlc":
                # 200 evenly-spaced points across the requested
                # range. ``np.errstate`` is also applied inside
                # WLCModel.__call__, but having it here too makes
                # the intent visible at the call site.
                x_overlay = np.linspace(xlo, xhi, _FIT_OVERLAY_N)
                with np.errstate(divide="ignore", invalid="ignore"):
                    y_overlay = WLCModel()(
                        x_overlay,
                        p=float(fit.params["p"]),
                        L=float(fit.params["L"]),
                    )
                finite = np.isfinite(y_overlay)
                if finite.any():
                    ax.plot(x_overlay[finite], y_overlay[finite], **_FIT_LINE_KW)

            ax.set_xlabel("extension (nm)")
            ax.set_ylabel("force (pN)")
            if self._title:
                ax.set_title(self._title, fontsize=10)
            ax.grid(True, alpha=0.3)
            # Render to an in-memory PNG so the Pillow image is a
            # plain ``Image.open``-able object (and a copy, not a
            # view onto a closed file handle).
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=_DPI)
            buf.seek(0)
            return _PILImage.open(buf).copy()
        finally:
            _plt.close(fig)

    def render(self) -> Any:
        """Return a rich renderable representing the current plot state.

        The output is a :class:`rich.console.Group` containing:

        1. The widget's :attr:`title` as a bold :class:`rich.text.Text`
           (if a title was given at construction time).
        2. Either a half-block :class:`_PillowImageRenderable` of
           the cached bitmap, a native terminal-image renderable
           (Sixel / Kitty / iTerm / Halfcell via
           :mod:`textual_image.renderable`), or the literal
           placeholder :class:`rich.text.Text` ``"no data"`` when
           :attr:`_image` is ``None`` (i.e. before the first
           :meth:`render_curve` or after :meth:`clear`).

        The choice between half-block and native is controlled by
        the ``prefer_native`` constructor argument (see
        :meth:`__init__`); default is auto-detect.

        Returns
        -------
        rich.console.Group
            A composable rich renderable. The Textual App mounts it
            via the standard ``Widget.render()`` contract; a
            headless test can call it directly to verify the
            placeholder behaviour.
        """
        from rich.console import Group
        from rich.text import Text

        parts: list[Any] = []
        if self._title:
            parts.append(Text(self._title, style="bold"))
        if self._image is None:
            parts.append(Text("no data", style="dim"))
        else:
            parts.append(_choose_renderable(self._image, self._prefer_native))
        return Group(*parts)
