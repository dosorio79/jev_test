# Comparative reports implementation plan

**Goal:** Compare two or more saved runs using YAML and an offline Quarto report.
**Architecture:** A small comparison module validates saved run inputs and aligns
predictions by sample ID, then recalculates metrics using the existing evaluator.
The CLI writes a dated comparison snapshot and JSON, and renders one Quarto template.
**Tech Stack:** Existing Pydantic, Polars, Great Tables, Plotly, and Quarto.
**Spec:** The approved comparison proposal in this conversation.

## Constraints and decisions

- No provider calls, new dependencies, source-run changes, or automatic intersections.
- Require at least two distinct run folders and identical selected source records.
- Reject duplicate, missing, or foreign prediction IDs. Align reordered predictions.
- New runs record a selected-dataset SHA256 in report metadata. Reject a mismatch.
- Legacy runs remain usable with an explicit warning: historical data cannot be
  verified because the original runs did not record a fingerprint.
- Resolve comparison paths relative to YAML. Output defaults to `runs/` beside it.
- Use existing metrics, including unresolved predictions and partial pricing rules.
- Preserve generated reports; escape strings displayed through Great Tables.

## Review focus

Changed source content with stable IDs; reordered and duplicated predictions;
preview or incomplete runs; missing costs and unresolved labels; legacy runs,
empty disagreement tables, and report re-rendering after source runs move.

## Task 1: Saved-run validation and comparison data

Files: `src/cx_sentiment/comparison.py`, `src/cx_sentiment/cli.py`,
`tests/test_comparison.py`.

1. Write failing tests for matching runs, alignment, mismatched data/IDs,
   historical fingerprint mismatch, legacy warnings, and missing artifacts.
2. Run `uv run pytest tests/test_comparison.py` and confirm missing implementation.
3. Implement typed comparison settings and loading, dataset fingerprint helper,
   run validation, metrics recalculation, language summaries, and disagreements.
4. Save the selected dataset fingerprint in new run metadata from `_write_run`.
5. Rerun focused tests and Ruff/Pyright.

## Task 2: CLI, report, and example

Files: `src/cx_sentiment/cli.py`, `src/cx_sentiment/reporting.py`,
`src/cx_sentiment/reports/comparison.qmd`, `experiments/compare-smoke.yaml`,
`tests/test_comparison.py`, `README.md`, `docs/REPORTING.md`, `AGENTS.md`.

1. Test `compare FILE` with temporary saved runs and an actual HTML render;
   verify friendly invalid-config errors without credentials or network access.
2. Implement the command, dated comparison folders, replayable YAML snapshots,
   self-contained JSON, and editable `.qmd`/HTML with summary tables, cost/quality
   and latency/quality plots, language results, and disagreements.
3. Add the example referencing the two existing live smoke runs and document
   configuration, interpretation, legacy caveat, and local re-rendering.
4. Run the full test suite, Ruff, Pyright, and `git diff --check`.
5. Render the real smoke comparison, inspect its contents and source preservation,
   then commit the verified feature locally (no push).

## Progress

- Plan recorded before production edits.
- Task 1 complete: saved-run checks, prediction alignment, metrics recalculation,
  legacy warnings, and dataset fingerprints for new experiment reports.
- Task 2 complete: YAML CLI, dated snapshots, standalone Quarto report, smoke
  example, regression tests, and user/agent documentation.
- Verification: full suite 71 passed; comparison tests 14 passed; updated CLI
  fingerprint integration 1 passed; Ruff and Pyright clean; targeted CLI Mypy clean;
  `git diff --check` clean. No paid calls were made.
- Final real report:
  `experiments/runs/2026-10-16T20-40-06.939396Z-comparison/report.html`.
  Both smoke providers have 5/5 correct with no disagreements. Historical input
  verification remains explicitly unavailable for these two legacy runs.
- Feature, tests, and documentation form one coherent local commit; no push.
