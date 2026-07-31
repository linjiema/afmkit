# Tutorials

Jupyter notebook tutorials live in the [`examples/`](https://github.com/linjiema/afmkit/tree/main/examples)
directory of the repository. They are rendered into this section by
mkdocs.

| # | Topic | Status |
|---|---|---|
| 01 | Quickstart: load, fit, export | **shipped in v0.5** (uses WLC) |
| 02 | Batch processing a folder of curves | **shipped in v0.5** (one folder = one `CurveBatch`) |
| 03 | Comparing WLC vs eWLC on the same data | draft — see [Migration guide](../migration.md) and the `afmkit.models.ewlc` API |
| 04 | Migrating from Igor Pro | draft — see [Migration guide](../migration.md) |
| 05 | Writing your first plugin | draft — see [Contributing guide](../contributing.md) |
| 06 | Peak review in the TUI | planned for v0.6 — see the `PeakReviewer` + `presentation.gui` API |
| 07 | Igor `.ibw` round-trip with `roundtrip_ibw` | planned for v0.6 — see [`docs/quickstart.md`](../quickstart.md) for a working snippet |

To run the notebooks locally:

```bash
pip install "afmkit[all,docs] @ git+https://github.com/linjiema/afmkit.git@v0.5.0"
jupyter lab examples/
```

A notebook is "shipped" once it lives in `examples/` and runs
end-to-end against the current release. The "draft" notebooks
are documented in the guides above; the "planned" entries are
spelled out in the [CHANGELOG](https://github.com/linjiema/afmkit/blob/main/CHANGELOG.md)'s
`## [Unreleased]` section as v0.5+ / v0.6 candidates.
