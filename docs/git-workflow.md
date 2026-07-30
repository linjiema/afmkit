# Git workflow for afmkit

> Status: in effect from v0.4 onward. The v0.1, v0.2, v0.3
> history was committed directly to `main` and is left as-is —
> rewriting history would cause more pain than the broken process
> caused. **Future work uses this workflow.**

## Branches

We use a simplified Git Flow with three long-lived branches and
short-lived feature / fix / chore branches.

| Branch | Lifetime | Purpose | Who can push |
|---|---|---|---|
| `main` | permanent | Released versions only. Every commit is either a `chore(release): cut vX.Y.Z` or a `fix:` hotfix. Tagged on every release. | Lead, with a green CI on the most recent commit. |
| `develop` | permanent | Daily integration. Feature / fix / chore branches land here via PR. Tagged on every release. CI on every commit. | Lead, after at least one review approval. |
| `feature/*`, `fix/*`, `chore/*` | short-lived (deleted after merge) | One branch per logical change. Named `feature/peak-review`, `fix/igor-pin`, `chore/update-changelog`, etc. | Coder agents + Lead, on their own work. |

The Coder agents and the Lead push feature / fix / chore branches
directly to `origin` so the orchestrator cron can see them. The
Lead opens a PR from the feature branch to `develop` once the
local gates (ruff + mypy + pytest) are green and the
post-commit CI is green. The Lead merges the PR (squash-merge by
default) once `develop`'s CI is green; fast-forwarding `main` to
`develop` and tagging is the next step.

## Commit messages

We follow [Conventional Commits](https://www.conventionalcommits.org/).

```
<type>(<scope>): <short summary>          ← 50 chars, no period
<blank line>
<body: what changed and why, wrapped at 72 chars>            ← optional
<blank line>
<footer: refs, breaking changes, Co-authored-by>             ← optional
```

Types: `feat`, `fix`, `chore`, `refactor`, `test`, `docs`, `build`, `ci`, `style`, `perf`, `revert`.

Scopes: a top-level subpackage or area. Examples:
`io`, `models`, `fitting`, `analysis`, `presentation`, `cli`, `core`, `docs`, `ci`, `deps`, `release`.

One commit per logical change. If a feature needs both a
`feat(...)` and a `test(...)` commit, the convention is the
**feature commit first** (so `git bisect` lands on the failing
state, not the test that revealed it). Two-commit splits are
fine; ten-commit micro-splits are not.

The git author for every commit is `Michael <Michael@local>` —
set with `git -c user.name="Michael" -c user.email="Michael@local" ...`
each time. (This is the user's name and email as configured in
the original repo settings; not a placeholder.)

## Pull requests

- **Title**: matches the lead commit's subject line (Conventional
  Commit format). e.g. `feat(io): add Igor .ibw read + write`.
- **Body** (one short paragraph + the standard checklist):

  ```
  ## What

  One sentence on what this PR does and why.

  ## How to verify

  - [ ] `ruff check src tests` clean
  - [ ] `ruff format --check src tests` clean
  - [ ] `mypy src/afmkit` clean
  - [ ] `pytest tests/ -q --doctest-modules src/afmkit` clean
  - [ ] relevant CI jobs green
  ```

- **Review**: at least one approval before merge. For Coder-agent
  work, the **Lead** is the reviewer. For Lead work on the docs
  or release, the user reviews directly.

- **Merge**: squash-merge by default. The squash commit inherits
  the PR title and body, so `git log main` reads as a clean
  release-history timeline.

## Hotfixes

A `fix/*` branch cut directly off `main` (not `develop`) is a
hotfix. Open the PR against `main`, merge with a non-squash
merge commit (so the fix is identifiable in the history), then
**back-merge** to `develop` so the fix lands in the next release
too. The back-merge is `git checkout develop && git merge
--no-ff main` (or cherry-pick the fix commit).

## Releases

Releases follow `docs/release-checklist.md`. The flow is:

1. Develop on `develop` (multiple feature / fix / chore PRs).
2. Cut a `release/vX.Y.Z` branch off `develop` for the final
   paperwork (CHANGELOG, version bump, README sync).
3. Open a PR `release/vX.Y.Z` → `main`. Merge with a non-squash
   merge commit (so the release commit is identifiable).
4. Tag the merge commit `vX.Y.Z` (`git tag -a vX.Y.Z -m "..."`).
5. `git push origin vX.Y.Z`.
6. `gh release create vX.Y.Z --title "..." --notes-file ...`.

`main` and `develop` are then fast-forwarded to the same commit.
`main` gets the tag; `develop` keeps the same code and continues
toward the next release.

## Anti-patterns

- **Direct push to `main`** — never. Always go through a PR
  unless it's a tag push (`git push origin vX.Y.Z`).
- **Direct push to `develop`** without a PR — fine for tiny
  chore commits (typo, doc nit, version bump) but for any
  non-trivial change open a PR so the work is reviewable.
- **Squash-merging a hotfix** — the merge commit is the audit
  trail; squashing loses it.
- **Force-push to any shared branch** — never. Rebase locally
  before pushing if you need a clean history; the shared branch
  history is sacred.
- **Mixed commit types in one commit** — `feat:` and `fix:`
  should never appear in the same commit. Split them.

## Tools

- `gh` for PR creation, status checks, and release notes.
- `pre-commit` is set up but not yet enforced in CI; v0.4 plan is
  to add a `pre-commit` job to the CI matrix so the local hooks
  run on every push.
- Cron-based CI watch (orchestrator pattern, set up at the
  start of each release) keeps the Lead honest about not
  ignoring a red CI.
