# Simplicity pass implementation plan

**Goal:** Make this test platform easier to read and explain without changing its behavior.

**Approach:** Keep the existing modules and dependencies. Use ordinary functions,
remove duplicated logic, and preserve public signatures, report formats, validation,
and live-call safeguards. Implement in the current checkout without a new worktree.
The initial passes were left uncommitted; the user subsequently requested commits.

**Scope:** The user's requested readability pass, with README.md describing the
existing platform contract.

- [x] Establish the existing pytest baseline and add characterization coverage for
  fallback routes and report contents before refactoring.
- [x] In reporting.py, separate CSV writing and HTML rendering from write_report.
  Keep explicit HTML and CSV fields; do not introduce a template framework.
- [x] In routing.py, share fallback handling between failed and low-score decisions.
  In data.py, reuse audit_dataset for validation overlap checking. In metrics.py,
  build the truth/prediction pairs once per calculation.
- [x] Add a short code walkthrough to README.md. Review the diff and run pytest,
  Ruff, Pyright, and an offline report smoke check.

**Review focus:** HTML escaping, CSV columns and missing costs; threshold equality;
failed or unavailable fallbacks; cross-split overlap rejection. Preserve all of these.

## Updated scope: YAML experiment configuration

The user identified CLI options as overkill and suggested YAML. Add a flat validated
experiment file and make `run experiment.yaml [--live]` the primary interface.
Keep legacy flags compatible but out of the primary help. Reject mixed configuration
sources. Resolve dataset/output paths relative to the YAML file; reject unknown keys.
Use PyYAML safe_load (one small parsing dependency) and existing Pydantic validation.
Use the record limit as the request cap; keep total and per-request USD caps explicit.
Credentials stay in the environment. Test offline execution, dry runs, invalid config,
and live guard enforcement without making paid calls.

## Verification

Completed: 39 pytest tests passed; Ruff passed; Pyright reported zero errors.
The YAML offline smoke produced five predictions and all four report files.
Reviewed the diff; git diff --check passed. No paid API calls or commits were made.

## Continuation: configuration feedback (2026-10-16)

- [x] Turn missing files, malformed YAML, and invalid settings into concise CLI
  errors without echoing configuration values. Name YAML budget fields in live errors.
- [x] Print the report or preview location after a successful run.
- [x] Verify regressions with CLI tests, then pytest, Ruff, Pyright and diff review.

Continuation verification: 44 pytest tests passed; Ruff passed; Pyright reported
zero errors. New error and output-location tests reproduced the gaps before fixes.
Diff reviewed and git diff --check passed. No paid calls or commits were made.

## Commit handoff (2026-10-16)

The user authorized local commits after adopting the Superpowers commit policy.

- `197abeb`: simplify evaluation and reporting flow, with characterization tests.
- `3195892`: add YAML experiment configuration, readable CLI feedback, tests,
  dependency lock update, and README workflow.
- Repository guidance and this completed checklist are committed separately.

Fresh pre-commit verification: `uv run pytest` passed all 44 tests;
`uv run ruff check .` passed; `uv run pyright` reported zero errors;
`git diff --check` and staged diff checks passed. No paid calls or pushes.
