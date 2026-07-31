# Changelog

All notable changes to **afmkit** are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
and the format is inspired by [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

New feature work piles on top; the v0.6 tag is the next cut.

### Added

- **`.ibw` v5 stdlib-only reader** — `load_ibw` no longer needs
  the optional `igor` package for v5 files. The v5 path now goes
  through a small, stdlib + NumPy reader
  (`_load_ibw_v5_stdlib`) that mirrors the on-disk layout the v5
  writer emits (BinHeader5 = 62 B, WaveHeader5 = 320 B, both with
  the `=` / standard-size convention; wData as little-endian
  float64; note bytes verbatim). The `import igor.binarywave`
  call is now lazy (`:func:`_get_binarywave`), so
  `import afmkit.io.igor_ibw` works on a minimal install that
  doesn't have `igor`. v1 / v2 / v3 reads still delegate to
  `igor.binarywave.load` and require `afmkit[igor]`. The new
  `tests/unit/test_igor_ibw_v5_stdlib.py` covers the v5 path
  end-to-end and verifies the round-trip is **byte-exact**:
  every header field the writer emits is read back with the same
  value, including the on-disk `bname`, `dataUnits`, `dimUnits`,
  `sfA`, `sfB`, `nDim`, `fsValid`, `topFullScale`, `botFullScale`,
  and the `npnts`-length wData payload.

### Known limitations (v0.6+ roadmap)

- **Matplotlib TUI plot panel native image** — the v0.4
  plot panel renders the matplotlib figure as a half-block
  text image via `rich.Console`. A native image render via
  the Textual image protocol (Sixel / Kitty / iTerm
  graphics) would be sharper and faster on terminals that
  support it. The half-block path is the v0.4 default
  because it works everywhere.

## [0.5.0] — 2026-07-31

**The v0.4 symmetric-completion release.** v0.4 plumbed
`PeakReviewer` state through `to_csv_fits` and `to_markdown`;
v0.5 closes the symmetric pair (`to_mat` and `to_parquet`)
and adds the v0.5+ `.ibw` note re-hydration contract that
makes the round-trip truly loss-less without the caller
threading `k_cantilever` (or any other metadata key) through
manually. The headline is: every metadata key the v0.4
`save_ibw` writer embedded in the wave `note` now comes back
through the v0.5 `load_ibw` reader; the per-peak review
state is symmetric across all four exporter backends.

**Install**

```bash
pip install "afmkit @ git+https://github.com/linjiema/afmkit.git@v0.5.0"
```

Optional extras:

```bash
pip install "afmkit[igor] @ ..."      # Igor .ibw round-trip
pip install "afmkit[gui] @ ..."      # Textual TUI
pip install "afmkit[plot] @ ..."     # matplotlib panel inside the TUI
pip install "afmkit[parquet] @ ..."  # pyarrow Parquet export
```

### Added

- **`to_mat` and `to_parquet` accept `fits` and `reviewers`
  kwargs**. This is the v0.5 symmetric completion of the
  v0.4 #1 peak-review plumbing: `to_csv_fits` and
  `to_markdown` switched to one-row-per-(curve, peak) when
  `reviewers` is provided; `to_mat` and `to_parquet` now do
  the same. The three tables (`fits`, `peak_review`, and
  the original curves) join on `curve_index`, and the
  per-peak review state (accept / reject / override force /
  note) flows through to both backends.
  - **`to_mat`**: when `fits` is provided, a top-level
    `fits` struct is added to the `.mat` file (one entry
    per fit with the per-fit schema: model, params,
    std-errors, goodness-of-fit, `curve_index`). When
    `reviewers` is provided alongside `fits`, a top-level
    `peak_review` struct is added (one entry per (curve,
    peak) with the per-peak review columns joined to the
    fit columns). Both structs are emitted as 1-D
    structured ndarrays so the round-trip through
    `scipy.io.loadmat` is clean (squeeze the loaded
    `(1, N)` shape to recover the canonical 1-D layout).
    The legacy v0.4 shape (no fits, no reviewers) is
    preserved.
  - **`to_parquet`**: when `fits` is provided, a sibling
    `<path>.fits.parquet` file is written next to the
    main curves file. When `reviewers` is provided
    alongside `fits`, a sibling `<path>.peaks.parquet`
    file is written with the per-(curve, peak) review
    table. The three-file layout (curves / fits / peaks)
    matches the three natural shapes (one row per point /
    one row per fit / one row per (curve, peak)) and lets
    pandas / polars users load each one independently.
    The fastparquet fallback only supports the main
    curves file; the `fits` / `reviewers` paths require
    pyarrow and surface a clear `ImportError` when
    pyarrow is not installed.
  - The per-peak review column schema is the same as
    `to_csv_fits`: `curve_index`, `peak_index`,
    `extension_nm`, `force_pN`, `manual_force_pN`,
    `accepted`, `confidence`, `prominence_pN`,
    `width_points`, `height_drop_pN`, `note`. The
    `manual_force_pN` and `note` columns now use `NaN`
    (instead of empty string) as the "no override / no
    note" marker, which keeps pyarrow's column type
    inference happy and round-trips cleanly through
    pandas. The CSV / Markdown paths still produce the
    v0.4 output (empty string for `manual_force_pN`);
    the change is internal to the v0.5 row builder.
- **Refactored the per-peak row builder** out of
  `_write_per_peak_csv` into the shared
  `_build_peak_review_rows` helper. Same logic, but the
  new helper returns a list of plain dicts that the CSV
  / `.mat` / parquet writers can each post-process. Also
  added `_build_fit_rows` (one-row-per-fit table used by
  `to_mat` and `to_parquet`) and
  `_validate_reviewers_mapping` (the stray-index check
  previously inlined in `_write_per_peak_csv`).

- **`.ibw` note full re-hydration on read**. The v0.4
  `load_ibw` extracted only `k_cantilever` from the wave
  `note`; everything else the writer had embedded (the
  `temperature`, `experiment_id`, `n_averages`,
  `operator`, `notes`, etc. that the v0.4 writer passes
  through `_encode_note`'s `extra` kwargs) was silently
  dropped. v0.5 introduces a generic note parser
  (`_parse_note_metadata` + `_coerce_note_value` +
  `_NOTE_TOKEN_RE`) that re-hydrates every scalar
  `key=value` token with proper type coercion (int /
  float / bool / str), the `k=` short form is renamed to
  the canonical `k_cantilever` metadata key, and the
  legacy `k_cantilever`-only path is preserved as a
  backward-compatibility shim. The reader is now
  symmetric with the writer: every metadata key that
  goes in through `save_ibw(curve, path, *, version=...)`
  comes back through `load_ibw(path)`. A legacy file
  with a hand-written note that has no `k=` token still
  loads cleanly (just without the `k_cantilever` key,
  not a crash). 7 new tests in `tests/unit/test_igor_ibw.py`
  cover the round-trip contract for both v2 and v5.

- **`roundtrip_ibw(curve, path, *, version=2)`** — a thin
  convenience wrapper around `save_ibw` + `load_ibw` that
  demonstrates the round-trip contract in one call.
  Returns the loaded `ForceCurve` after asserting the
  `(extension, force)` arrays round-trip via
  `numpy.testing.assert_allclose` and that every scalar
  metadata key the writer emitted comes back through the
  loader. Useful for end-to-end smoke tests and for
  documenting the v0.5+ note-rehydration contract in
  downstream code.

- **Docs site refreshed to v0.5 state**. The
  `docs/index.md` install pin, `docs/quickstart.md` (full
  rewrite — old version said "Status: in progress. This
  page will populate as v0.1 features land."), the
  `docs/migration.md` status table (which claimed
  "eWLC and FJC plugin-only" and "PyQt6 GUI v0.2" until
  today), the `docs/api/index.md` module list (which
  listed non-existent modules like
  `afmkit.processing.smooth` and missed the actual
  `presentation.gui` / `analysis.peak_detection` /
  `analysis.peak_review` / `models.fjc` modules), and the
  `docs/team.md` phase ("Phase 1 — Core & IO"). All
  brought up to v0.5. `git-workflow.md` and
  `release-checklist.md` are now in the nav under a new
  "Development" section so the v0.3 retrospective doc
  set is discoverable.

### Infrastructure

- 436 unit tests + 12 doctest pass on every cell of the
  CI matrix (3 OS × 3 Python), up from 414 + 12 in
  v0.4.0. The 22-test delta is split: 15 from the
  `to_mat` / `to_parquet` reviewer plumbing, 7 from the
  `.ibw` note rehydration + `roundtrip_ibw` helper.
- `mkdocs build --strict` passes with no warnings. The
  v0.5 docs site publishes the v0.5-scope narrative
  (peak-review symmetry across all four exporter
  backends, full note re-hydration contract, TUI
  workflow, round-trip snippet) and the v0.3
  retrospective (`git-workflow.md` + `release-checklist.md`)
  in the nav.

### Known limitations (v0.6+ roadmap)

- **`.ibw` v5 stdlib-only reader**. The v0.4 reader
  still delegates to the upstream `igor.binarywave.load`
  for v5. A stdlib-only v5 reader (matching the v2 / v5
  writer pattern) would let us drop the `igor` runtime
  dep for the read path too, and would let us test the
  v5 round-trip byte-by-byte instead of via the
  upstream black box.
- **Matplotlib TUI plot panel native image**. The v0.4
  plot panel renders the matplotlib figure as a
  half-block text image via `rich.Console`. A native
  image render via the Textual image protocol (Sixel /
  Kitty / iTerm graphics) would be sharper and faster
  on terminals that support it. The half-block path
  is the v0.4 default because it works everywhere.

## [0.4.0] — 2026-07-30

**The v0.3 retrospective is fully landed, four v0.4 features ship on
top, and the docs site is live.** The headline is the process fix:
the workflow rules, the release checklist, and the pre-commit-in-CI
gate mean future releases can no longer land without a branch /
PR / paperwork pass / CI green — the regressions that bit
v0.1 → v0.2 → v0.3 are now structurally impossible. The four
v0.4 features (peak-review state in CSV/Markdown export, real
matplotlib rendering in the TUI plot panel, `.ibw` v5 read/write,
and the docs site going live) all ride on top of the new
infrastructure.

**Install**

```bash
pip install "afmkit @ git+https://github.com/linjiema/afmkit.git@v0.4.0"
```

Optional extras:

```bash
pip install "afmkit[igor] @ ..."      # Igor .ibw round-trip
pip install "afmkit[gui] @ ..."      # Textual TUI
pip install "afmkit[plot] @ ..."     # matplotlib panel inside the TUI
pip install "afmkit[parquet] @ ..."  # pyarrow Parquet export
```

### Added

- **Peak review state plumbed through `to_csv_fits` and
  `to_markdown`**. `to_csv_fits(fits, path, *, reviewers=None)` and
  `to_markdown(batch, fits, path, *, reviewers=None)` now accept a
  `{curve_index: PeakReviewer}` mapping. When `reviewers` is
  provided, the CSV output switches to one row per (curve, peak)
  with the fit columns repeated and the per-peak columns appended
  (`curve_index`, `peak_index`, `extension_nm`, `force_pN`,
  `manual_force_pN`, `accepted`, `confidence`, `prominence_pN`,
  `width_points`, `height_drop_pN`, `note`). The Markdown output
  gets a new `## Peak review` section after `## Fit results`, with
  one row per peak. The legacy one-row-per-fit shape is preserved
  when `reviewers` is `None` (the default), so v0.1 → v0.3 callers
  are unaffected. Curves with a reviewer but zero peaks get a
  single empty row; curves without a reviewer entry still get a
  row so every fit is represented.

- **Real matplotlib in the TUI plot panel**. The v0.3 plot panel
  was a `Static` widget that only ever showed a textual summary
  ("plot: curve 3 + 5 reviewed peak(s) + WLC fit p=0.41 nm"); the
  matplotlib figure rendered by `ForceExtensionPlot.render_curve`
  was thrown away because the v0.3 code path didn't have a
  Textual Console to drop the renderable into. v0.4 replaces the
  `Static` with a `Container` that holds the actual
  `ForceExtensionPlot` widget, mounted once in `compose()`.
  `_render_plot` now calls `render_curve` on the mounted widget —
  the half-block image actually shows up in the TUI. When the
  `[plot]` extra is not installed, the container holds a `Static`
  fallback with the install hint; either way the `#plot-panel` id
  is preserved so the toggle / render paths don't have to branch.

- **`.ibw` v5 read + write**. The v0.3 writer was v2-only (the
  upstream `igor==0.3` package's `save()` raises
  `NotImplementedError`, so we wrote a stdlib-only v2 emitter).
  v0.4 adds a v5 write path — `save_ibw(curve, path, *,
  version=5)` — and the v0.3 reader already accepted v5 via the
  upstream `igor.binarywave.load`. v5 is the modern Igor Pro
  6.00+ layout (`WAVE_HEADER5` = 320 B, `BIN_HEADER5` = 62 B, `=`
  byte order, `P` → `I` pointer substitution) and is required by
  Igor Pro 7+ for waves that reference text or extended dimension
  units. The v2 path stays the default for back-compat; the
  round-trip is identical between the two versions for the (ext,
  force) pairs afmkit emits. 7 new tests in `test_igor_ibw.py`
  cover the v5 path: round-trip preserves (ext, force) and
  k_cantilever, the 31-char v5 bname holds long source filenames,
  the v5 wave header carries the right `type` / `dataUnits` /
  `dimUnits[0]` / `sfA` / `sfB` / `fsValid` / `topFullScale` /
  `botFullScale` fields, the v2 and v5 files produce equivalent
  curves, and `save_ibw(..., version=3)` (or any other non-{2,5})
  raises `ValueError`.

- **Docs site is live at https://linjiema.github.io/afmkit/**.
  The one-time GitHub UI setup at repo Settings → Pages → Source
  = "GitHub Actions" is done; the `Docs` workflow's
  `actions/deploy-pages@v4` step now succeeds. The current
  deployment is from the v0.3.0 release commit and shows the
  Home, Quick start, Migration, Tutorials, API reference,
  Contributing, and Changelog pages. The v0.3 retrospective docs
  (`git-workflow.md` and `release-checklist.md`) are on `develop`
  and will land on the site at the next push to `main` (this
  v0.4 release).

- **`pre-commit` enforced in CI**. The `.pre-commit-config.yaml`
  hooks (general pre-commit-hooks + ruff lint/format + mypy
  `--strict`) now run on every push and PR to `main` and
  `develop` via a new `pre-commit` job in
  `.github/workflows/ci.yml` using `pre-commit/action@v3.0.1`.
  The build job now depends on the pre-commit job too, so a
  hook failure blocks a release build. The pre-commit config
  was tightened at the same time: the `blacken-docs` hook was
  removed (it conflicts with `ruff format`), the `mypy` hook
  is bumped from v1.10.1 to v2.3.0 (the same version CI uses
  — 1.x flagged the untyped `pluggy.HookimplMarker` decorator
  on every `@register_loader` / `@register_model` helper in
  `src/afmkit/plugins.py` as a `[misc] Untyped decorator`
  false positive, while 2.3.0 tolerates it), `numpy` is pinned
  to `>=1.26,<2.4` in the pre-commit `additional_dependencies`
  (mirrors the project `dependencies` pin; numpy 2.5+ stubs use
  PEP 696 TypeVar-default and PEP 695 `type` statement syntax
  that the pre-commit mypy cannot parse, which surfaces as
  `Type statement is only supported in Python 3.12 and greater`
  even on 3.12+ interpreters), and `pluggy`, `typer`, `rich`,
  and `pyarrow` are added to `additional_dependencies` because
  the pre-commit env does not auto-install the project `[dev]`
  extra. `ruff`'s `--fix` flag was removed so a dirty push
  fails loud in CI instead of getting silently rewritten. A
  handful of pre-existing formatting drifts in `docs/migration.md`,
  `docs/v0.3-roadmap.md`, `README.md`,
  `src/afmkit/io/jpk_txt.py`, and
  `.github/ISSUE_TEMPLATE/bug_report.md` are swept up here so
  the first CI run starts green.

### Fixed

- **Missing runtime dep: `pluggy>=1.5` is now in
  `[project] dependencies`**. `src/afmkit/plugins.py` does
  `import pluggy` at module load (the `pluggy.HookspecMarker`
  / `HookimplMarker` decorators power the
  `@register_loader` / `@register_model` /
  `@register_baseline` / `@register_fitter` entry points).
  Before this fix, `pluggy` was only present as a transitive
  dep of `pytest`, so a bare `pip install afmkit` (no
  `[dev]` / `[test]` extras) could resolve to an environment
  where `import afmkit` would fail at the `import pluggy`
  line the first time the plugin manager was initialised. CI
  was unaffected because it installs `[dev]` (which
  transitively pulls `pluggy` via `pytest`), but the missing
  declaration was a footgun for end users. The pre-commit
  hook still lists `pluggy` in `additional_dependencies`
  because pre-commit does not auto-install the project's
  runtime dependencies.

### Infrastructure

- **v0.3 retrospective** — the workflow doc, the release
  checklist, and the README sync. Two process regressions
  bit the v0.1 → v0.2 → v0.3 cycle: (1) every change was
  pushed directly to `main` instead of through a branch
  model, and (2) the README was never synced on release (it
  still pinned to v0.1.0 after v0.3.0 shipped). Both are
  fixed and structurally impossible from v0.4 onward.
  - `docs/git-workflow.md` — branch model: `main` (releases
    only, tagged) / `develop` (daily integration) /
    `feature/*` / `fix/*` / `chore/*` (short-lived work
    branches). Conventional Commits with a top-level
    subpackage as scope, one commit per logical change,
    squash-merge for develop PRs, non-squash for release
    and hotfix PRs against `main`.
  - `docs/release-checklist.md` — the pre-flight /
    paperwork / local-gates / CI-gates / tag-and-release /
    post-release flow. The "skipping the README sync"
    anti-pattern is called out as the most-burned-us
    mistake.
  - README sync to v0.3.0 happened in PR #1 (retroactive
    paperwork pass) and is now an explicit checklist item
    in the release flow.
- 414 unit + 12 doctest tests pass on every cell of the CI
  matrix (3 OS × 3 Python), up from 380 + 12 doctest in
  v0.3.0. The 34-test delta is split: 11 from the CSV /
  Markdown plumbing (v0.4 #1), 4 from the real-matplotlib
  TUI plot panel (v0.4 #2), 7 from the `.ibw` v5 path
  (v0.4 #3), and the rest from the docs / workflow /
  pre-commit config plumbing.

### Known limitations (v0.5+ roadmap)

- **`.ibw` v5 reader improvements** — the v0.4 reader still
  delegates to the upstream `igor.binarywave.load` for v5.
  A stdlib-only v5 reader (matching the v2 / v5 writer
  pattern) would let us drop the `igor` runtime dep for the
  read path too, and would let us test the v5 round-trip
  byte-by-byte instead of via the upstream black box.
- **Matplotlib TUI plot panel native image** — the v0.4
  plot panel renders the matplotlib figure as a half-block
  text image via `rich.Console`. A native image render via
  the Textual image protocol (Sixel / Kitty / iTerm
  graphics) would be sharper and faster on terminals that
  support it. The half-block path is the v0.4 default
  because it works everywhere.
- **PeakReviewer export plumbing for `.mat` / Parquet** —
  the v0.4 CSV / Markdown export picks up the per-peak
  reviewer state. `to_mat` and `to_parquet` still emit a
  one-row-per-fit shape; plumbing the reviewer through those
  is the symmetric v0.5 task.
- **FJC reading from the `.ibw` writer's 2-col layout** —
  the v0.4 `.ibw` writer emits a 2-column wave (ext, force)
  with an `afmkit=2col` note. The reader does not yet
  reconstruct the `ForceCurve` directly from the note;
  callers still need to know to pass `k_cantilever`
  explicitly. A round-trip helper that reads the note and
  re-hydrates the `ForceCurve` is the v0.5 piece.

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

[Unreleased]: https://github.com/linjiema/afmkit/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/linjiema/afmkit/releases/tag/v0.5.0
[0.4.0]: https://github.com/linjiema/afmkit/releases/tag/v0.4.0
[0.3.0]: https://github.com/linjiema/afmkit/releases/tag/v0.3.0
[0.2.0]: https://github.com/linjiema/afmkit/releases/tag/v0.2.0
[0.1.0]: https://github.com/linjiema/afmkit/releases/tag/v0.1.0
