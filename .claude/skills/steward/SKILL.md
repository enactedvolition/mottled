---
name: steward
description: How to drive a Mottled pull request from open to merged — when to merge, when to hold, what to run before a push, and what a PR body must show. Read before acting on CI, review or merge events on any PR in this repo.
---

# Stewarding a Mottled PR

The maintainer works through Claude sessions and does not review line by
line. A PR you open or are asked to drive is yours to get **merged**, not
just green: nobody else will. Be proactive, including on releases, which
are gated on evidence rather than on asking (see "Outward actions").

## Working style (the maintainer's standing preferences)

- State assumptions; never guess silently. A decision you made goes in the
  PR body with its reason.
- Minimum code, nothing speculative. Surgical changes: don't refactor code
  next to the change.
- Define success before starting, and loop until it is verified by a
  command you ran. Numbers beat argument.

## Done, and then merge

A PR is done when all of these hold on its current head:

- `test (python 3.11)` (ci.yml) is green.
- `Evidence` (evidence.yml) is green if it ran. It runs when a PR touches
  `experiments/` or the workflow itself. Red there is red CI, not a
  side note.
- There is no merge conflict, and no review thread is waiting on you.
- The PR body's evidence is current (see below).

Then **squash-merge it yourself**, without waiting to be asked. The
`enable_pr_auto_merge` tool refuses this repo's PRs in both directions
(CLAUDE.md, "Workflow"), so merge directly. A merge to `main` also deploys
GitHub Pages (`pages.yml`). For a PR that touches `viewer/` or `site/`, the
viewer tests must pass and a sample scene must load in the pre-installed
Chromium before you merge.

Hold only for one of these, and say which in one line on the PR:

- CI is red and you cannot explain why, or a test is flaky.
- There is a merge conflict where both sides changed the same logic.
- A human review was explicitly requested and is still pending.
- It is the release PR and its checklist is not all PASSED (next section).
- A release is between its merge and its tag. `main` stays frozen until
  the tag is pushed.

## Outward actions: yours to take, behind their gates

The maintainer has handed these over too: don't ask first. Each one waits
on evidence instead of on a yes.

- **Dispatching workflows.** Dispatch `evidence.yml`, re-run CI, and
  dispatch `publish.yml` as a dry run on the release candidate's branch
  whenever that branch changes. Off a tag, the publish job is skipped:
  `if: startsWith(github.ref, 'refs/tags/')`.
- **Merging the release PR.** Squash-merge it only when its evidence
  checklist has a PASSED row, with a run URL or saved output, for every
  gate RELEASING.md requires. Prove the gates covered the merged tree:
  `git diff <gated commit> <squash commit>` must touch nothing but
  `CHANGELOG.md`. Merges to `main` wait until the tag is pushed.
- **Pushing the tag.** Tag the release PR's squash commit, never a later
  HEAD, and push it in the same sitting. A `v*` tag publishes to PyPI, and
  a version number can never be uploaded again. So a failed or unrun gate
  means no tag, not a retry under the same number. If `publish.yml` fails
  after the tag, bump the patch version for the next attempt.
- **The GitHub release.** Create it from the tag. The notes are the
  changelog block plus the parity table, as RELEASING.md asks. Then run
  its after-tagging checks and report what they showed.
- **Repository settings and secrets.** Change only what the task in hand
  needs. Record what changed and why in the PR, or in an issue comment if
  there is no PR.

A gate that needs a person still needs one: the WebGPU check
(`viewer/tests/parity.html` on a GPU) has no runner. Until the maintainer
reports it, the release stays open. Say so once, then move on.

## Before every push

- Run `pytest -m "not network" -q` and `node --test viewer/tests/`, or at
  least the changed modules' tests plus `tests/test_packaging.py`.
- If you touched `mtjschema.py` or `design_tokens.py`, run
  `python -m codegen`. Generated files are never edited by hand.
- If you changed one side of a Python↔JS pair in CLAUDE.md's table, change
  the other side and run its conformance test.
- A WGSL kernel change (`viewer/ops-webgpu.js`) cannot be verified by CI.
  Say so in the PR and ask the maintainer to run `viewer/tests/parity.html`
  on a GPU.
- Re-read the diff as a reviewer trying to reject it.

## What a PR body must show

- What changed and why, with any assumptions stated.
- For each check: the exact command, the commit it ran on, and the result
  line (`641 passed, 1 skipped, 15 deselected`), or the run URL. Keep
  results you **ran** apart from results you **quote** from another PR or
  a commit message. Quoted results are not evidence.
- What was not run, and why.
- Docs, UI strings and comments follow `docs/validity.md`'s vocabulary. A
  stronger claim is a correctness bug.

## Changelog

Every user-facing change adds an entry under `## Unreleased` in the PR that
makes it. A release PR that dates the heading conflicts with each such
merge. Resolve that by keeping the dated heading above everything that
ships in the release.

## Review findings

- Fix CI failures and blocking findings in the next push.
- A bot finding is a claim to verify. Trace a real path to the failure
  before you build a fix for it.
- Optional nits that are plainly correct ride along with the next push
  that already touches those files. Otherwise answer them in one line and
  resolve the thread.

## Cloud-session environment

- `huggingface.co`, `download.pytorch.org` and GitHub's artifact storage
  may be blocked by the session's network policy. Don't route around the
  proxy.
- Real-model checks run on a runner through `evidence.yml`. Read their
  results with `get_job_logs`: every result is printed there with its
  sha256, so a copy taken from the log can be checked against it.
- PyPI is reachable. Without the CPU wheel index, `torch` installs as its
  CUDA build, which runs on CPU. Record that difference next to any result
  from the session.
