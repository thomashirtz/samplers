# Merge requests

How we open, review, and land changes on `master`.

## Principles

- **One MR = one theme** — PGDM, a bug-fix batch, an operator, etc.
- **Squash merge** to `master` — one commit per MR on the default branch.
- **Small, reviewable diffs** — library code, tests, and scripts tied to the change.

## Workflow

1. Branch from `master`:
   ```bash
   git checkout master && git pull
   git checkout -b my-feature
   ```
2. Make changes; run tests locally.
3. Push the branch:
   ```bash
   git push -u origin my-feature
   ```
4. Open a pull request against `master` on GitHub.
5. **Squash and merge** when approved.

After merge, the squash commit title becomes the public history entry, e.g. `Fix networks (#13)`.

## MR title format

Use a short imperative summary. GitHub appends the PR number on squash:

```
Add PGDM and fix core sampler bugs (#14)
Fix networks (#13)
Add batch-agnostic support to DPS (#6)
```

## What belongs in an MR

| Include | Examples |
|---------|----------|
| Library code | `samplers/operators/`, `samplers/samplers/`, `samplers/networks/` |
| Tests | `tests/operators/`, `tests/samplers/` |
| Scripts | `scripts/run_dps.py` when behaviour changes |
| Process docs | This file, when updating workflow |

## What stays local (do not push)

| Exclude | Why |
|---------|-----|
| `.scratch/` | Personal session notes and WIP briefs |
| `TODO.md`, personal notes | Not part of the public repo |
| Generated images | `scripts/*.jpg`, `scripts/*.png` outputs |
| IDE / workspace files | `.idea/`, `*.code-workspace` |
| `__pycache__/`, `*.pyc` | Build artifacts |
| Chat logs, one-off scripts | `chat-*.txt`, throwaway experiments |

## README policy

README updates can land in a **separate commit** on `master` (direct push or dedicated MR), not bundled into unrelated feature MRs.

## Pre-push checklist

Before `git push`:

- [ ] `pytest tests/` passes (or the subset relevant to your change)
- [ ] `git diff --cached --name-only` lists only intended files
- [ ] README unchanged unless this MR is explicitly a README update
- [ ] No `.scratch/` files staged
- [ ] No local artifacts staged (images, chat logs, workspace files)

Quick audit:

```bash
git diff --cached --name-only
git status --short
```

## PR description template

Copy into the GitHub PR body:

```markdown
## Summary
- {what changed and why}

## Changes
| Area | Files |
|------|-------|
| {area} | `{paths}` |

## Test plan
- [ ] pytest tests/operators/
- [ ] pytest tests/samplers/
- [ ] python scripts/run_dps.py
```
