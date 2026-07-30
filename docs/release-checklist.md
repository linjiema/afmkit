# Release checklist for afmkit

> Run this checklist top-to-bottom before every `vX.Y.Z` tag.
> The Lead is responsible for the cut; the user reviews the
> release PR before merge.

## Pre-flight (week-of, not day-of)

- [ ] All in-flight `feature/*` and `fix/*` branches are merged
      to `develop` (or explicitly deferred to the next release).
- [ ] `develop` is at HEAD; CI on `develop` is green.
- [ ] The version in `pyproject.toml` is the next release
      candidate (`0.4.0.dev0` for v0.4 work, bumped to
      `0.4.0` on the release commit).
- [ ] A `release/vX.Y.Z` branch has been cut off `develop` for
      the final paperwork pass.

## Paperwork pass (on the release branch)

- [ ] **CHANGELOG.md** — write the vX.Y.Z section above
      `Unreleased`. Use the existing v0.1.0 / v0.2.0 / v0.3.0
      sections as templates: an install block, an "Added" list
      sourced from the actual commits, an "Infrastructure"
      list of dependency / config changes, a "Known
      limitations" list that points at the next milestone.
- [ ] **README.md** — this is the part that has been skipped
      before. Check:
  - [ ] The install command in the Quick start block pins to
        the new tag (`pip install "afmkit @ git+...@vX.Y.Z"`).
  - [ ] The Features list matches the actual vX.Y.Z scope
        (drop vX.(Y-1) features that didn't change, add the
        new ones).
  - [ ] The "What's new" section at the bottom of the README
        links to the new GitHub release.
  - [ ] The "Verified on" table reflects the current CI matrix
        (drop cells that no longer apply if Python versions
        change).
  - [ ] The "Try it in 30 seconds" example uses the new
        public API surface (peak review if v0.3, etc.).
- [ ] **pyproject.toml** — version bumped from `.dev0` to the
      release number. No accidental dev-only extras in
      `dependencies` (the `[dev]` extras stay where they are).
- [ ] **src/afmkit/_version.py** — `__version__` matches.
- [ ] **docs/vX.Y-roadmap.md** — either archived to `docs/`
      (status: shipped) or removed; the next milestone is in a
      new file.

## Local gates (before pushing the release branch)

- [ ] `source .venv/bin/activate`
- [ ] `ruff check src tests` — clean
- [ ] `ruff format --check src tests` — clean
- [ ] `mypy src/afmkit` — clean (no new errors; pre-existing
      overrides are fine)
- [ ] `pytest tests/ -q --doctest-modules src/afmkit` — all
      green; total test count ≥ previous release + new tests
- [ ] End-to-end smoke (for v0.3+): a Python REPL run that
      exercises the headline new feature in 5 lines

## CI gates (before tagging)

- [ ] `git push origin release/vX.Y.Z`
- [ ] Open a PR `release/vX.Y.Z` → `main` with:
  - [ ] Title: `chore(release): cut vX.Y.Z — <one-line summary>`
  - [ ] Body: link the CHANGELOG section, link the README
        diff, list the new features, paste the test count
        delta, paste the `pytest` summary
  - [ ] The PR template's "How to verify" checklist above
        is fully checked
- [ ] CI on the release branch is green: lint, typecheck,
      every test matrix cell, the optional extras smoke, the
      sdist build. **The Docs workflow's deploy step is allowed
      to fail** until the one-time GitHub UI Pages setup is done
      (a separate human step, not gated by CI).
- [ ] User has reviewed and approved the PR

## Tag and release

- [ ] Merge the PR with a non-squash merge commit
      (so the release commit is identifiable in history)
- [ ] `git checkout main && git pull`
- [ ] `git tag -a vX.Y.Z -m "..."` (use the standard tag message
      format from the v0.3.0 example)
- [ ] `git push origin vX.Y.Z`
- [ ] `gh release create vX.Y.Z --title "..." --notes-file
      release-notes.md` where `release-notes.md` is the polished
      release-notes file (see the v0.3.0 template at
      `/tmp/afmkit_v0.3_release_notes.md` as the reference)
- [ ] The GitHub release page links to the vX.Y.Z tag commit
      and renders the install command with the tag pinned

## Post-release

- [ ] Verify the release page renders correctly on GitHub
- [ ] (Optional) Open a `docs/vX.Y+1-roadmap.md` for the
      next milestone
- [ ] Bump the version in `pyproject.toml` and `_version.py`
      to `X.Y+1.0.dev0` on `develop` so future work is on the
      next release cycle
- [ ] Update the Coder agent task template (if you have one)
      so the next round uses the vX.Y.Z tag for install hints

## Common mistakes to watch for

- **Skipping the README sync** — this is what burned us on
  v0.1 → v0.2 → v0.3. The Quick start block still pinned
  v0.1.0 even after v0.3.0 shipped. Read the README line-by-
  line, not just the version number.
- **Tagging from a dirty worktree** — always `git status` and
  `git log -1` before `git tag`. A release tag should point
  at a clean commit on `main`, not at the last PR merge's
  intermediate state.
- **Forgetting to fast-forward `develop`** — after the
  `release/vX.Y.Z` → `main` merge, run
  `git checkout develop && git merge --ff-only main` so the
  next cycle starts from the released state.
- **Re-shipping the same bug** — the Known limitations
  section in CHANGELOG is the contract for "what we're
  punting to next time". If a punted item got fixed in this
  release, remove it. If a new one surfaced, add it.
