# afmkit

> **Modern Python toolkit for single-molecule force spectroscopy data analysis.**

afmkit is a clean, extensible Python reimplementation of the workflow that
single-molecule biophysics labs have been running in Igor Pro for two
decades. It reads AFM force-extension curves (JPK Nanowizzard, ForceRobot,
plus legacy Igor Binary Wave), fits the standard polymer models (WLC / eWLC /
FJC), picks sawtooth unfolding peaks, and exports the results in formats
that drop straight into **Origin**, **Matlab**, and the **Python** data
stack.

The goal is a tool that a grad student can use from a notebook **and** a
lab can build a reproducible pipeline on top of.

## Install

```bash
pip install "afmkit @ git+https://github.com/linjiema/afmkit.git@v0.2.0"
```

afmkit is not on PyPI — install from GitHub and pin a tag for
reproducibility. Use `@main` for the bleeding edge.

## What's in here

<div class="grid cards" markdown>

-   :materialrocket-launch:{ .lg .middle } **Quick start**

    ---

    Install afmkit, load a folder of JPK curves, and fit your first
    WLC model in under five minutes.

    [:octicons-arrow-right-24: Get started](quickstart.md)

-   :material-swap-horizontal:{ .lg .middle } **Migration from Igor Pro**

    ---

    Already running `FX_Analysis` in Igor? Here's the one-to-one
    mapping of every original function (`FXImport`, `WLCurves`,
    `FitToCursor`, …) to its afmkit equivalent, plus a units &
    sign-conventions cheat sheet.

    [:octicons-arrow-right-24: Migrate](migration.md)

-   :material-book-open-variant:{ .lg .middle } **API reference**

    ---

    Full public surface, auto-generated from docstrings:
    `ForceCurve`, `CurveBatch`, `LmfitEngine`, `WLCModel`, exporters,
    and the `MODEL_REGISTRY` plugin hook.

    [:octicons-arrow-right-24: Browse the API](api/index.md)

-   :material-notebook:{ .lg .middle } **Tutorials**

    ---

    Worked examples for batch processing, WLC vs eWLC comparison,
    the Igor migration, and writing your first plugin.

    [:octicons-arrow-right-24: See tutorials](tutorials/index.md)

</div>

## Project

- :fontawesome-brands-github: [Source on GitHub](https://github.com/linjiema/afmkit)
- :material-tag: [Latest release (v0.2.0)](https://github.com/linjiema/afmkit/releases/tag/v0.2.0)
- :material-history: [Changelog](https://github.com/linjiema/afmkit/blob/main/CHANGELOG.md)
- :material-account-group: [Team & agent workflow](team.md)
- :material-roadmap: [Roadmap](v0.3-roadmap.md)
- :material-handshake: [Contributing / plugin authoring](contributing.md)

## Citation

If afmkit helped your research, please cite it. A `CITATION.cff` is
shipped at the repo root for GitHub's built-in citation parser.

```bibtex
@software{afmkit,
  title  = {afmkit: a Python toolkit for single-molecule force spectroscopy},
  author = {Ma, Linjie and contributors},
  url    = {https://github.com/linjiema/afmkit},
  year   = {2025},
}
```

## License

[MIT](https://github.com/linjiema/afmkit/blob/main/LICENSE) — do what
you want, just keep the copyright notice.
