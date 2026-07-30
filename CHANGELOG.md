# Changelog

All notable changes to **afmkit** are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
and the format is inspired by [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

The v0.3 retrospective process fix is in this section; it lands on
`develop` with the PR that introduced the workflow rules (no separate
v0.3.1 paperwork-pass release). New feature work will pile on top; the
v0.4 tag is the next cut.

### Added

- **Peak review state plumbed through `to_csv_fits` and `to_markdown`**.
  The first v0.4 task. `to_csv_fits(fits, path, *, reviewers=None)` and
  `to_markdown(batch, fits, path, *, reviewers=None)` now accept a
  `{curve_index: PeakReviewer}` mapping. When `reviewers` is provided,
  the CSV output switches to one row per (curve, peak) with the fit
  columns repeated and the per-peak columns appended (`curve_index`,
  `peak_index`, `extension_nm`, `force_pN`, `manual_force_pN`,
  `accepted`, `confidence`, `prominence_pN`, `width_points`,
  `height_drop_pN`, `note`). The Markdown output gets a new
  `## Peak review` section after `## Fit results`, with one row per
  peak. The legacy one-row-per-fit shape is preserved when `reviewers`
  is `None` (the default), so v0.1 → v0.3 callers are unaffected.
  Curves with a reviewer but zero peaks get a single empty row; curves
  without a reviewer entry still get a row so every fit is represented.

- **Real matplotlib plot in the TUI plot panel**. The v0.3 plot panel
  was a `Static` widget that only ever showed a textual summary
  ("plot: curve 3 + 5 reviewed peak(s) + WLC fit p=0.41 nm"); the
  matplotlib figure rendered by `ForceExtensionPlot.render_curve` was
  thrown away because the v0.3 code path didn't have a Textual
  Console to drop the renderable into. v0.4 replaces the `Static`
  with a `Container` that holds the actual `ForceExtensionPlot`
  widget, mounted once in `compose()`. `_render_plot` now calls
  `render_curve` on the mounted widget — the half-block image
  actually shows up in the TUI. When the `[plot]` extra is not
  installed, the container holds a `Static` fallback with the
  install hint; either way the `#plot-panel` id is preserved so
  the toggle / render paths don't have to branch.

- **`.ibw` v5 read + write**. The v0.3 writer was v2-only (the
  upstream `igor==0.3` package's `save()` raises `NotImplementedError`,
  so we wrote a stdlib-only v2 emitter). v0.4 adds a v5 write path
  — `save_ibw(curve, path, *, version=5)` — and the v0.3 reader
  already accepted v5 via the upstream `igor.binarywave.load`.
  v5 is the modern Igor Pro 6.00+ layout (`WAVE_HEADER5` = 320 B,
  `BIN_HEADER5` = 62 B, ``=`` byte order, ``P`` → ``I`` pointer
  substitution) and is required by Igor Pro 7+ for waves that
  reference text or extended dimension units. The v2 path stays
  the default for back-compat; the round-trip is identical
  between the two versions for the (ext, force) pairs afmkit
  emits. 7 new tests in `test_igor_ibw.py` cover the v5 path:
  round-trip preserves (ext, force) and k_cantilever, the
  31-char v5 bname holds long source filenames, the v5 wave
  header carries the right `type` / `dataUnits` / `dimUnits[0]`
  / `sfA` / `sfB` / `fsValid` / `topFullScale` / `botFullScale`
  fields, the v2 and v5 files produce equivalent curves, and
  `save_ibw(..., version=3)` (or any other non-{2,5}) raises
  `ValueError`.

### Process (v0.3 retrospective)

Two process regressions bit the v0.1 → v0.2 → v0.3 cycle: (1) every
change was pushed directly to `main` instead of through a branch
model, and (2) the README was never synced on release (it still
pinned to v0.1.0 after v0.3.0 shipped). The v0.3 retrospective
fixes both. v0.1 / v0.2 / v0.3 history is left as-is — rewriting
history would cause more pain than the broken process caused.
**Future work uses the new workflow.**

- **Git workflow** — `docs/git-workflow.md`. Branch model: `main`
  (releases only, tagged) / `develop` (daily integration) /
  `feature/*` / `fix/*` / `chore/*` (short-lived work branches).
  Conventional Commits with a top-level subpackage as scope, one
  commit per logical change, squash-merge by default. Hotfixes cut
  off `main`, non-squash merge, then back-merge to `develop`.
- **Release checklist** — `docs/release-checklist.md`. The
  pre-flight / paperwork / local-gates / CI-gates / tag-and-release
  / post-release flow. The "skipping the README sync" anti-pattern
  is called out as the most-burned-us mistake.
- **README sync to v0.3.0** — install pinned to `v0.3.0`; Features
  list reflects v0.3 scope (WLC + eWLC + FJC, peak detection,
  PeakReviewer, Igor .ibw round-trip, Textual TUI, plot widget,
  pluggy entry-point); "Try it in 30 seconds" uses the TUI +
  peak-review workflow; "Verified on" table reflects the 3 OS ×
  3 Python green matrix.

### Known limitations (v0.4 candidates)

Carried over from v0.3's Known limitations, in priority order:

- The `Docs` workflow's `Deploy to GitHub Pages` step needs the
  one-time GitHub UI action at repo Settings → Pages → Source =
  "GitHub Actions". Until that's done, the `Docs` job's deploy
  step fails (the build itself still succeeds and produces a
  `site/` artifact).
- `pre-commit` not yet enforced in CI.

## [0.3.0] — 2026-07-30

Four pieces, plus a TUI integration, plus a documentation site. The
big v0.3 story is that **the TUI is now actually usable for real
analysis** — you can open a directory of JPK files, fit WLC / eWLC
/ FJC, interactively review the auto-detected peaks (accept /
reject / override / re-fit), and see the curve + peaks + fit in a
matplotlib panel.

**Install**

```bash
pip install "afmkit @ git+https://github.com/linjiema/afmkit.git@v0.3.0"
```

Optional extras:

```bash
pip install "afmkit[igor] @ ..."      # Igor .ibw round-trip
pip install "afmkit[gui] @ ..."      # Textual TUI
pip install "afmkit[plot] @ ..."     # matplotlib panel inside the TUI
pip install "afmkit[parquet] @ ..."  # pyarrow Parquet export
```

### Added

- **Peak-review data model** — `afmkit.analysis.peak_review.PeakReviewer` and `ReviewedPeak`. The reviewer wraps a list of auto-detected `Peak`s and lets the caller accept / reject / override the force / re-fit a single peak in a local window / attach a free-form note. `to_dict()` exports the reviewer state to the same column-block shape as `to_csv_fits`, so the per-peak accept / reject / manual_force makes it into the output.

- **Matplotlib plot widget** — `afmkit.presentation.gui.plot.ForceExtensionPlot`. A Textual widget that renders a force-extension curve with optional peak markers and WLC fit overlay via the matplotlib Agg backend. Used inside the TUI (see below) and reusable by any future Textual surface.

- **FJC model + pluggy entry-point demo** — `afmkit.models.FJCModel`, the classical Freely Jointed Chain with the Padé [2,2] inverse Langevin approximation (b Kuhn length, Lc contour length). Registered both in `MODEL_REGISTRY["fjc"]` and in `pyproject.toml`'s `[project.entry-points."afmkit.models"]` block — the first time a third-party-style entry point has actually been wired in afmkit.

- **TUI integration** — `AFMkitApp` now has:
  - `P` (shift-p) — toggle the matplotlib plot panel
  - `p` — enter peak-review mode for the highlighted curve.
    Inside review mode: `j` / `k` navigate, `a` accept, `r` reject,
    `o` override force (asks via a small input), `R` re-fit the
    highlighted peak in a local window, `escape` exit
  - The fit result from `f` is now stashed in `_last_fit` and
    overlaid on the plot panel when `P` is on
  - The peak-review table and plot panel are CSS-toggled via a
    `hidden` class; the DOM stays put so toggling is instant

- **mkdocs + GitHub Pages deploy** — `mkdocs build --strict` is part of the `ci` workflow's optional `Docs` job; on every push to `main` the site is built and (once Pages is enabled in repo settings) deployed to `https://linjiema.github.io/afmkit/`. The `mkdocs.yml` nav now lists every doc with proper structure (Home, Quick start, Migration, Tutorials, API reference, Contributing, Team, Roadmap, Changelog).

- **`PeakReviewer.to_dict()` round-trip** — the per-peak accept / reject / manual_force state is a structured dict ready for the CSV / Markdown exporters; v0.4 will plumb it through `to_csv_fits` automatically.

### Infrastructure

- `pyproject.toml` `[dev]` extras now include `matplotlib`, `pillow`, `textual`, and `igor` (the latter pinned to `>=0.3`, the latest PyPI release — the `>=0.4` pin in the `[igor]` extra was aspirational; the 0.4 series is not on PyPI yet).
- New mypy overrides on `matplotlib.*` / `matplotlib.pyplot` / `PIL.*` so mypy --strict on the test matrix resolves the import-not-found complaint for the v0.3 plot widget.
- 380 + 12 doctest tests pass on every cell of the CI matrix (3 OS × 3 Python), up from 298 + 16 doctest in v0.2.0.

### Known limitations (v0.4+ roadmap)

- The peak-reviewer's CSV / Markdown export plumbing is not yet wired — `PeakReviewer.to_dict()` exists and the tests pass, but `to_csv_fits` / `to_markdown` still emit a row per `FitResult` only. Plumbing the per-peak state through the exporters is the first v0.4 task.
- The plot panel currently shows a textual summary ("plot: curve 3 + 5 reviewed peak(s) + WLC fit p=0.41 nm") rather than rendering the actual matplotlib image into the panel. A real matplotlib-in-Textual widget is a v0.4 story.
- `.ibw` v5 read / write is still v0.4.
- The `Docs` workflow's `Deploy to GitHub Pages` step needs the human action of going to repo Settings → Pages → Source = "GitHub Actions" (a one-time GitHub UI step that cannot be done from a workflow). Until that's done, the `Docs` job fails on deploy (the build itself still succeeds and produces a `site/` artifact).

## [0.2.0] — 2026-07-30

Four big pieces, each on its own feature branch and shipped together
once the full test matrix was green.

**Install**

```bash
pip install "afmkit @ git+https://github.com/linjiema/afmkit.git@v0.2.0"
```

To pull in the new optional extras:

```bash
pip install "afmkit[igor] @ ..."     # Igor .ibw round-trip
pip install "afmkit[gui]  @ ..."     # Textual TUI
pip install "afmkit[parquet] @ ..."  # pyarrow backend for to_parquet
```

### Added

- **Sawtooth peak detection** — `afmkit.analysis.find_sawtooth_peaks(curve, …)` returns a list of `Peak` dataclasses (extension, force, prominence, width, height_drop, confidence) for the retract sweep. Algorithm: centered moving-average smoothing → `scipy.signal.find_peaks` with prominence / width thresholds → per-peak confidence from the median prominence. Use this as the upstream of "how many unfolding steps did this molecule show, and at what forces".
- **Igor .ibw round-trip** — `afmkit.io.load_ibw`, `load_ibw_batch`, `save_ibw`, and the `IgorIBWLoader` class. The reader delegates to the optional `igor` PyPI package; the writer is a stdlib-only v2 binary emitter we wrote in-tree (the released `igor==0.3` ships a `save()` that raises `NotImplementedError`). The wave note carries an `afmkit=2col` marker so the reader can disambiguate. F/B files in the same folder are paired as approach/retract by basename.
- **Extensible WLC model** — `afmkit.models.EWLCModel` and the `"ewlc"` registry entry. The Wang 1997 / Odijk 1995 interpolation with the new `K0` (stretch modulus, pN) parameter; reduces to Marko-Siggia WLC as K0 → ∞ and is finite on the physical range [0, L] (the singularity lives at `x = L·(1 + 1/K0) > L` — strictly past the contour length, which is the practical advantage over WLC).
- **Textual TUI** — `afmkit gui` (or `python -m afmkit.presentation.gui.app`). Three-panel layout: directory input → data table → status line. Keybindings: `o` open directory, `f` fit selected curve with WLC, `e` export batch to CSV, `q` quit. No X server, no PySide6 — runs in any terminal over SSH.
- **Plugin system hardening** — `afmkit.models.register_model` and `MODEL_REGISTRY` are now the v0.2 first-class extension point. eWLC is registered as `"ewlc"` alongside the existing `"wlc"`. A pluggy entry-point (`afmkit-fjc`, etc.) for separately-installable plugin packages is documented but not yet exercised.

### Infrastructure

- Optional dependencies: `[igor]`, `[gui]`, `[parquet]`, `[plot]`, `[all]`.
- `afmkit.io.__init__` uses PEP 562 `__getattr__` to lazy-import the optional `.ibw` surface — `from afmkit.io import load_ibw` works but a minimal install (no `igor`) doesn't fail at package import.
- `pyproject.toml` pins `textual.*` and `afmkit.presentation.gui.*` with `disable_error_code = ["misc"]` to silence the mypy-version-dependent "Class cannot subclass value of type Any" false positive on the TUI base classes.
- 298 tests pass on every cell of the CI matrix (3 OS × 3 Python), up from 222 in v0.1.0. ~85 % line coverage on the core.

### Known limitations (v0.3 roadmap)

- Peak detection does not yet support manual review / re-fit of a single peak — that's a v0.3 interactive piece in the TUI.
- FJC model and `afmkit-fjc` standalone plugin package are v0.3.
- `.ibw` writer is v2 only (v5 in Igor supports 32-bit ints, dependency tracking, etc.). v2 covers everything afmkit produces; v5 support can come later if someone needs it.
- TUI is terminal-only — a real GUI (Qt / web) is still a future piece.

## [0.1.0] — 2026-07-30

First public release. Core library + CLI; no GUI yet.

**Install**

```bash
pip install "afmkit @ git+https://github.com/linjiema/afmkit.git@v0.1.0"
```

### Added

- **Core data model** — `ForceCurve` and `CurveBatch` (xarray-backed, immutable, full unit tests).
- **WLC model** — Marko-Siggia worm-like chain, 1:1 compatible with the original Igor `LVFitWLC` formula. Registered as the default `"wlc"` model in `MODEL_REGISTRY`.
- **JPK 4-column `.txt` loader** — `JPKTxtLoader` and `load_jpk_txt()`, bit-for-bit identical to the original Igor `FXImport()`. Handles optional text headers, k_cantilever in pN/nm, and returns a `CurveBatch` of two `ForceCurve`s (approach / retract).
- **HDF5 native store** — `HDF5Store`, `load_hdf5()`, `save_hdf5()`. gzip-compressed, JSON-encoded metadata, ragged batches supported.
- **Exporters** — `to_csv` (wide column-block), `to_csv_fits` (one-row-per-fit), `to_mat` (v5), `to_parquet` (pyarrow / fastparquet), `to_markdown` (human-readable report), and an `export()` dispatch helper that picks the format from `--format` or the file suffix.
- **Fitting engine** — `LmfitEngine` wraps `lmfit` and exposes a uniform `fit(curve, model=..., x_range=..., p0=...)` helper. Returns a `FitResult` with best-fit params, standard errors, χ², reduced χ², covariance, and the data the fit was performed on.
- **CLI** — `afmkit version / info / import / fit / export` subcommands (typer + rich), end-to-end wired.
- **Plugin system** — pluggy hookspecs (`afmkit.plugins`) for loaders, models, baselines, and fitters. Model registry is the v0.1 short-form; entry-point discovery lands in v0.2.
- **Dev toolchain** — `ruff`, `mypy` (non-strict in v0.1, strict by v0.2), `pytest`, `hypothesis`, `pre-commit`, GitHub Actions CI on macOS / Linux × Python 3.11 / 3.12 / 3.13, mkdocs-material docs.
- **Documentation** — quickstart, API reference, migration guide from Igor Pro, plugin authoring guide, team / agent workflow doc, end-to-end runnable example notebook (`examples/01_quickstart.ipynb`).

### Known limitations (v0.2 roadmap)

- Igor `.ibw` round-trip — read / write support was planned for v0.2 once the `igor` optional extra is unblocked on Python 3.13.
- eWLC and FJC models are plugin-only in v0.1; the registry is wired but no first-party implementations are shipped.
- Automated sawtooth peak detection is v0.2.
- GUI is v0.2.

[Unreleased]: https://github.com/linjiema/afmkit/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/linjiema/afmkit/releases/tag/v0.2.0
[0.1.0]: https://github.com/linjiema/afmkit/releases/tag/v0.1.0
