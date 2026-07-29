# Tutorials

Jupyter notebook tutorials live in the [`examples/`](https://github.com/linjiema/afmkit/tree/main/examples)
directory of the repository. They are rendered into this section by
mkdocs.

| # | Topic | Status |
|---|---|---|
| 01 | Quickstart: load, fit, export | coming with v0.1 |
| 02 | Batch processing a folder of curves | coming with v0.1 |
| 03 | Comparing WLC vs eWLC on the same data | coming with v0.2 |
| 04 | Migrating from Igor Pro | draft — see [Migration guide](../migration.md) |
| 05 | Writing your first plugin | draft — see [Contributing guide](../contributing.md) |

To run the notebooks locally:

```bash
pip install afmkit[all,docs]
jupyter lab examples/
```
