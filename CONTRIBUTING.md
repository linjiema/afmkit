# Contributing to afmkit

Thanks for your interest! afmkit is meant to be a community tool — every
contribution, from typo fixes to new polymer models, helps the SMFS
ecosystem.

## Quick links

- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Issue tracker](https://github.com/linjiema/afmkit/issues)
- [Discussions](https://github.com/linjiema/afmkit/discussions)

## Development setup

```bash
git clone https://github.com/linjiema/afmkit
cd afmkit
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[all,dev]"
pre-commit install
```

Verify your install:

```bash
pytest                   # should pass
ruff check src tests     # no output = clean
mypy src/afmkit          # no output = clean
```

## Workflow

1. **Open an issue first** for anything beyond a small fix. This avoids
   duplicate work and surfaces design questions early.
2. Fork & create a feature branch: `git checkout -b feature/short-name`
3. Write code + tests together. PRs without tests are unlikely to be merged
   for non-trivial changes.
4. Run the local checks before pushing:
   ```bash
   ruff check src tests
   ruff format src tests
   mypy src/afmkit
   pytest
   ```
5. Push and open a PR. The CI will run the same checks on Linux, macOS,
   and Windows across Python 3.11/3.12/3.13.
6. Update the "Unreleased" section of [CHANGELOG.md](CHANGELOG.md).

## Code conventions

- **Type hints everywhere** — the project runs `mypy --strict`.
  ```python
  def smooth(curve: np.ndarray, window: int = 5) -> np.ndarray: ...
  ```
- **Public functions take explicit parameters** — no module-level globals,
  no implicit config from environment. (Use :mod:`afmkit.utils.config` for
  user-tunable defaults.)
- **Docstrings** in NumPy style for all public symbols.
- **Purity** — processing/fitting functions must be pure (no side effects,
  no I/O). I/O and UI live in :mod:`afmkit.io` and :mod:`afmkit.presentation`.
- **Imports** — `from __future__ import annotations` at the top of every
  module, ordered with `ruff` defaults.

## Architecture rules

The layer cake below is enforced by code review, not by a test. The
upward arrows mean "may import from"; downward arrows are forbidden.

```
        presentation        ← may import from anything below
        analysis            ← may import processing, fitting, models, io, core
        processing, fitting ← may import models, core
        models              ← may import core
        io                  ← may import core
        core                ← no internal dependencies
```

If you find yourself wanting to import upward, that's a sign the design
needs another look — open an issue.

## Writing a plugin

afmkit uses [pluggy](https://pluggy.readthedocs.io) (the same hook system
that powers pytest). To add a new file format, polymer model, baseline
correction, or fitting engine:

1. Create a new package: `afmkit-<your-plugin>` on PyPI.
2. Implement the relevant Protocol from
   - :mod:`afmkit.io.base` for loaders
   - :mod:`afmkit.models.base` for models
   - :mod:`afmkit.processing.baseline` for baseline correctors
   - :mod:`afmkit.fitting.engine` for fitters
3. Register via entry points in your `pyproject.toml`:
   ```toml
   [project.entry-points."afmkit.models"]
   fjc = "afmkit_fjc:FJCModel"
   ```
4. Publish to PyPI. `afmkit` will discover the plugin on next import.

Example plugin repository: see [`docs/contributing.md`](docs/contributing.md).

## Testing

- Tests live under `tests/` and mirror the source layout.
- Use `pytest` for unit and integration tests.
- Use `hypothesis` for property-based tests of the physics layer.
- Add a golden-master test under `tests/golden/` for any new file format
  loader. The golden file should be a *tiny* synthetic example, not real
  data — keep the repo light.

## Documentation

- Tutorials live in `docs/tutorials/` as Jupyter notebooks.
- API reference is auto-generated from docstrings by mkdocstrings.
- Build locally with `mkdocs serve`.

## Release process

(For maintainers.)

1. Bump version in `src/afmkit/_version.py` and `pyproject.toml`.
2. Move the "Unreleased" section in `CHANGELOG.md` to a dated version.
3. Tag and push: `git tag vX.Y.Z && git push --tags`.
4. GitHub Actions builds the sdist + wheel and (optionally) publishes
   to PyPI via trusted publishing.
