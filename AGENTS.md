# Repository instructions

## Purpose and priorities

This is an offline-first Python test platform for comparing multilingual CX
sentiment providers and selective routing. Keep it **readable, explainable, and
simple**. The user explicitly identified excessive CLI options as overkill and
chose YAML experiment configuration.

- Choose the smallest solution that fully satisfies the request.
- Inspect the relevant code, README.md, and working-tree changes before editing.
- Prefer ordinary functions, explicit control flow, and existing dependencies.
- Preserve the current module boundaries and public behavior unless a change is
  requested. Avoid speculative abstractions, frameworks, and unrelated cleanup.
- Keep this file as coding-agent guidance. Usage belongs in README.md; research
  methodology belongs in docs/METHODS.md. Plans are checkpoints, not new authority
  to implement unrelated work.

## Workflow

- Make localized, low-risk changes directly. Do not create design documents,
  plans, worktrees, or subagents for a simple edit.
- For complex work, use the relevant skills and write a concise plan before
  substantial changes. Keep the process proportional to the task.
- Use subagents only for genuinely independent workstreams in a complex task.
- Proceed when scope is clear. Ask only when an answer would materially change
  the implementation; avoid repeated approval requests for authorized work.
- On continuation, read the relevant plan/checklist and inspect the actual diff.
  Resume unfinished work rather than repeating completed steps.
- Give short milestone updates and report concrete blockers and verification.

## Experiment interface

- Keep `uv run cx-sentiment run experiments/mock.yaml` as the primary run workflow.
  `--live` explicitly enables paid calls; experiment settings belong in YAML.
- Keep configuration flat and small. Do not grow the CLI with flags duplicating
  YAML settings. Preserve existing legacy commands, and reject mixing their run
  flags with a YAML configuration.
- Resolve input/output paths relative to the configuration file. Reject unknown
  settings, invalid providers, and nonpositive limits or cost caps.
- Treat YAML `output_dir` as a run parent. Create a dated child for every invocation,
  with provider/mode in its name and a resolved `experiment.yaml` beside results.
  Preserve old runs; keep credentials out of snapshots and `experiments/runs/` out of Git.
- Parse YAML with `safe_load` and validate it through the existing Pydantic model.
- Keep `cx-sentiment compare FILE` offline: use saved predictions, require matching
  selected source records and sample IDs, and preserve the original run folders.
  Record selected-dataset fingerprints on new runs; label legacy input verification
  as unavailable rather than inventing historical provenance.
- Give readable configuration errors without echoing configuration values or
  exposing tracebacks. Print the report or preview path after a successful run.
- Keep credentials in environment variables. Do not load `.env` implicitly.
  Follow the existing environment-based model configuration.

## Code map

- `src/cx_sentiment/cli.py`: command parsing and orchestration.
- `config.py`: YAML loading, validation, and relative path handling.
- `data.py`: Polars/Dataframely dataset validation, auditing, and sampling.
- `models.py` and `providers/`: shared prediction contract and provider adapters.
- `runner.py`, `cache.py`, `budget.py`, `pricing.py`: sequential execution,
  resumable results, spending guards, and token pricing.
- `metrics.py`, `routing.py`, `reporting.py`: evaluation, cached policy selection,
  and JSON/CSV output with Quarto rendering. Edit the readable report documents in
  `src/cx_sentiment/reports/`; keep data calculations in Python. See docs/REPORTING.md
  for setup and re-rendering. Use Great Tables and Plotly with Polars directly.
  Avoid pandas conversions, hand-built HTML/SVG, or a second rendering engine.
- `comparison.py`: comparison YAML, saved-run compatibility checks, dataset
  fingerprints, and recalculated cross-run metrics. Comparison templates read
  their saved JSON so re-rendering does not depend on the original source folders.

## Data, evaluation, and paid calls

- Preserve `negative=0`, `neutral=1`, `positive=2` and the six supported languages.
- The research corpus is local and gitignored. Never rewrite a locally supplied
  locked `data/cx_sentiment_test_600.csv`. Select prompts,
  calibration, and routing thresholds using validation data only; follow
  docs/METHODS.md before a final test evaluation.
- Keep sampling deterministic and preserve source indices, hashes, and overlap
  checks. Describe synthetic-data and near-duplicate limitations honestly.
- Never invent probabilities, confidence, usage, or prices. Preserve missing
  values, distinguish provider confidence from class probabilities, and keep
  failures in evaluation denominators. Mock accuracy demonstrates the pipeline.
- Keep mock runs offline and paid providers in preview mode unless `--live` is
  supplied. Actual paid calls also require user authorization for that run.
- Preserve request and cost guards. For YAML runs, `limit` also caps requests;
  live calls require `budget_usd` and `cost_per_request_usd`. Keep concurrency one.
- Preserve cache/resume behavior. Keep secrets and full raw provider responses
  out of normal reports, logs, examples, and Git. Escape HTML report values.
- Consult current official provider documentation when changing API contracts
  or pricing; retain dated pricing provenance.

## Jev probabilities and hybrid policy

- Validation failures are improper outputs, eligible for fresh collection; a valid
  but incorrect sentiment label is a scored prediction and must not be recollected
  to improve accuracy. Prefer bounded recollection over adding normalization.
- Recollection requires the existing live authorization and budget guards. Make
  at most one fresh attempt per rejected record unless further attempts are
  authorized. Bypass cached failures and use a distinct request identity so
  idempotency does not replay the rejected response. Preserve the original run,
  record all attempt costs, and keep remaining failures unresolved for fallback.
- Use the [official Choice docs](https://docs.typesafe.ai/primitives/choice) and
  [Python SDK response reference](https://docs.typesafe.ai/sdk/python/api/types/responses).
  As checked on 2026-10-16, they describe sums of 1 and approximately 1,
  respectively, without a numeric tolerance. Treat decimal rounding as a hypothesis.
- Keep the shared `Prediction` contract strict. Any future Jev normalization
  belongs in its adapter, with an explicit, justified bound and regression tests
  for valid, boundary, malformed, and nonfinite inputs. Never normalize arbitrary
  vectors or fill missing classes. No Jev rounding workaround is implemented yet.
- Preserve raw responses and provider-reported confidence. Keep normalized scores
  and any derived confidence distinguishable; their thresholds use different scales.
- Before freezing a hybrid policy, settle and record score interpretation,
  recollection limit, routing signal, and threshold. If interpretation changes,
  reprocess retained responses offline into a new derived run; preserve original
  runs and prevent reuse of incompatible cached results.
- Retain available usage and measured latency even when label/score validation
  fails. Preserve independently valid response references, credits, and cost estimates;
  never invent missing historical measurements.
  See [README research notes](README.md#jev-probability-precision-and-hybrid-policy)
  for evidence, routing implications, and current limitations.

## Verification and Git

Use Python 3.11 or 3.12 with uv. Start with the narrowest relevant test; add a
behavior-focused regression test for a bug when practical. For code changes,
run the applicable checks before claiming completion:

```bash
uv run pytest
uv run ruff check .
uv run pyright
git diff --check
```

- Exercise affected CLI commands and generated reports with mock data or a
  zero-network preview. Use temporary output folders to preserve existing runs.
- Report integration tests require the Quarto CLI on PATH. Verify actual renders
  when changing `.qmd` documents; Python lint/type checks do not cover their cells.
- Documentation-only changes need source/link checks, not a full test run.
- Report what actually ran and any checks that could not run. Avoid fixed test
  counts in instructions, and do not fix unrelated failures without a request.
- Treat uncommitted changes as user-owned. Preserve unrelated modifications.
- Avoid branches/worktrees for small changes.

## Commit policy (Superpowers)

- During authorized implementation work, make small, coherent local commits at
  completed task boundaries. Follow the plan's commit steps when a plan exists;
  do not wait for separate permission for each local commit.
- Verify before committing. Run the checks relevant to the change, read their
  results, and inspect the staged diff. Never commit known failing work as complete.
  Documentation-only commits need documentation checks rather than the code suite.
- Keep each commit focused on one reviewable change, including its relevant tests
  and documentation. Use a concise message describing the change, such as
  `feat: load YAML experiments` or `fix: clarify configuration errors`.
- Stage specific files or hunks. Never sweep unrelated pre-existing changes,
  secrets, caches, or generated experiment outputs into a commit.
- When using a plan/ledger, record commit hashes and verification results so a
  continuation can resume from the actual Git state. Report commits in the handoff.
- An explicit instruction to leave work uncommitted overrides this default.
  Pushes, PR creation, merges, history rewriting, and destructive Git operations
  require separate user authorization.
