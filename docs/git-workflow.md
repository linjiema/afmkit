# Git workflow for afmkit

> Status: in effect from v0.4 onward. The v0.1, v0.2, v0.3
> history was committed directly to `main` and is left as-is —
> rewriting history would cause more pain than the broken process
> caused. **Future work uses this workflow.**

## When you need a PR

The PR is the **review gate**, not a status symbol. Use it where
review actually adds value, skip it where the gates already do the
job.

| Move | PR required? | Why |
|---|---|---|
| `feature/*` / `fix/*` / `chore/*` → `develop` | **No** (default). Fast-forward or merge-commit directly. | Local gates (ruff + mypy + pytest) + ci.yml (after PR #3) already gate the merge. Opening a PR for self-review is ceremony. |
| Big / cross-module / multi-commit change → `develop` | **Yes** (recommended). | When a change touches many files or many commits, the PR's diff view and "How to verify" checklist earn their keep. |
| `release/vX.Y.Z` → `main` | **Yes** (always). | The release paperwork — CHANGELOG, version bump, README sync — is exactly the work a PR review should catch. |
| `main` hotfix (cut off `main`, not `develop`) | **Yes** (always, non-squash). | Hotfixes are an audit-trail event. The non-squash merge commit on `main` is the trail. |

If in doubt, open the PR. The cost of a 30-second PR is much
smaller than the cost of a regression that landed direct.

## Branches

We use a simplified Git Flow with three long-lived branches and
short-lived feature / fix / chore branches.

| Branch | Lifetime | Purpose | Who can push |
|---|---|---|---|
| `main` | permanent | Released versions only. Every commit is either a `chore(release): cut vX.Y.Z` or a `fix:` hotfix. Tagged on every release. | Lead, with a green CI on the most recent commit. |
| `develop` | permanent | Daily integration. Feature / fix / chore branches land here (direct fast-forward by default, PR for non-trivial changes). Tagged on every release. CI on every commit. | Lead, after local gates + CI are green. |
| `feature/*`, `fix/*`, `chore/*` | short-lived (deleted after merge) | One branch per logical change. Named `feature/peak-review`, `fix/igor-pin`, `chore/update-changelog`, etc. | Coder agents + Lead, on their own work. |

### feature / fix / chore → develop (default)

1. Branch off `develop`: `git checkout -b feature/my-change develop`.
2. Implement + commit (Conventional Commits, one logical change
   per commit).
3. Push the branch: `git push -u origin feature/my-change`.
4. Wait for the post-commit CI on the branch to turn green
   (pushes to non-`main` branches trigger ci.yml after PR #3).
5. Fast-forward `develop`: `git checkout develop && git merge
   --ff-only feature/my-change && git push origin develop`.

The merge is a clean fast-forward and `git log` on `develop`
reads as a linear timeline of the feature's individual commits.

### feature / fix / chore → develop (when the change is big)

If the change touches many files / many commits / public API
shape, open a PR. The PR is the place where:

  - the diff is reviewable side-by-side
  - the "How to verify" checklist gets ticked
  - the change history on `develop` is one squash-merge commit,
    not N individual feature commits

This is the same flow we used for v0.4 #1 (CSV / Markdown
plumbing) and v0.4 #2 (matplotlib plot panel) — those PRs
exercised the PR-path side of this workflow.

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

When you do open a PR (release / hotfix / big change), the format
is:

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

- **Merge**:
  - `feature/*|fix/*|chore/*` → `develop`: **squash-merge** by
    default. The squash commit inherits the PR title and body,
    so `git log develop` reads as a clean release-history
    timeline.
  - `release/vX.Y.Z` → `main` and any `fix/*` hotfix → `main`:
    **non-squash merge commit**. The merge commit is the audit
    trail; squashing loses it.

## Hotfixes

A `fix/*` branch cut directly off `main` (not `develop`) is a
hotfix. Open the PR against `main`, merge with a non-squash
merge commit (so the fix is identifiable in the history), then
**back-merge** to `develop` so the fix lands in the next release
too. The back-merge is `git checkout develop && git merge
--no-ff main` (or cherry-pick the fix commit).

## Releases

Releases follow `docs/release-checklist.md`. The flow is:

1. Develop on `develop` (multiple feature / fix / chore
   fast-forwards, with PRs for the non-trivial ones).
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
  unless it's a tag push (`git push origin vX.Y.Z`). The only
  exception is the `chore(release): cut vX.Y.Z` commit itself,
  which is the release PR's merge commit (not a direct push).
- **Direct push to `develop`** without a PR — fine for any
  change where local gates + CI are green, including
  multi-commit features (the merge is a clean fast-forward
  on `develop`'s linear history). Reserve PRs for non-trivial
  changes, releases, and hotfixes.
- **Squash-merging a hotfix or a release PR** — the merge
  commit is the audit trail; squashing loses it. Use a
  non-squash merge commit for these.
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
