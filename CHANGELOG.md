# Changelog

All notable changes to **afmkit** are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
and the format is inspired by [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Project scaffold: layered package layout, pluggy-based plugin system, full
  dev toolchain (ruff, mypy --strict, pytest, hypothesis, pre-commit, GitHub
  Actions CI, mkdocs-material).
- Core data model: `ForceCurve` (xarray-based) and `CurveBatch`.

## [0.1.0] — TBD

Initial public release. Scope: core library + CLI, no GUI yet.

### Planned
- `JPKTxtLoader` (4-column JPK Nanowizzard / ForceRobot `.txt`).
- `IgorIBWLoader` for legacy data.
- `WLCModel` (Marko-Siggia) — 1:1 with the original Igor implementation.
- `LmfitEngine` for non-linear least-squares fitting.
- HDF5 native storage; CSV / Matlab `.mat` / Parquet / Markdown exporters.
- `afmkit` CLI: `import`, `fit`, `info`, `export` subcommands.

[Unreleased]: https://github.com/linjiema/afmkit/compare/main...HEAD
