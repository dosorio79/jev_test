# Latency-Quality-Cost Study Implementation Plan

> **Historical checkpoint.** The implementation described here was completed in
> commits `c673ba9` through `b43b243`. The validation-selected policy was then
> frozen in `f654b97`; a separately authorized final evaluation was completed on
> 2026-10-20. The unchecked boxes below preserve the original execution plan and
> are not the current project status. Current conclusions are in the
> [published experiment report](../../results/cx-sentiment-routing/index.html).

**Goal:** Add a reproducible offline routing analysis and a separately authorized live benchmark that can determine whether Jev-to-OpenAI routing improves typical latency and cost while preserving multilingual sentiment quality.

**Architecture:** Keep ordinary experiment runs unchanged. Add one YAML-driven `study` workflow with two explicit modes: offline analysis over complete saved validation runs, and controlled live collection through a dedicated benchmark runner. Store immutable attempt-level evidence first, derive statistics and Pareto frontiers from saved evidence, and render one Quarto study report without mixing historical, simulated, and controlled measurements.

**Tech Stack:** Python 3.11/3.12, Pydantic, Polars, NumPy, scikit-learn, Plotly, Great Tables, Quarto, pytest, uv.

**Method reference:** [`docs/PERFORMANCE_STUDY.md`](../../PERFORMANCE_STUDY.md)

## Global Constraints

- Preserve `uv run cx-sentiment run experiments/mock.yaml` and all existing run artifacts.
- Put study settings in YAML; add only `cx-sentiment study FILE` and the existing-style `--live` authorization switch.
- Make offline analysis and preview perform zero provider calls.
- Do not initiate paid inference without fresh, explicit user authorization for the bounded live run.
- Select thresholds only from `source_split=validation`; keep `data/cx_sentiment_test_600.csv` untouched until the policy is frozen.
- Preserve failures and retries in denominators and end-to-end timing; never fabricate missing timing, probability, usage, or cost fields.
- Keep historical provider latency distinguishable from controlled end-to-end benchmark latency.
- Keep concurrency at one for the first authorized live benchmark; concurrency 5 and 10 are a later gate, not an automatic continuation.
- Continue using Great Tables and Plotly with Polars directly; do not add pandas or another report engine.

## Decisions and Corrections to the Proposal

1. The completed 600-record runs use validation data. They are suitable for threshold selection and exploratory comparisons, but not for a final unbiased acceptance claim. The locked 600-record test set remains reserved.
2. The historical latency values are not a fair matched benchmark: they were collected at different times and include cached original timings. They remain visible but are labeled exploratory and excluded from controlled acceptance decisions.
3. A 100–150 record pilot may estimate P50/P95 sufficiently to validate instrumentation, but P99 is descriptive only at that size. A decision-grade P99 requires a larger, predeclared sample or repeated collection.
4. Jev calibration can be assessed because it has full class probabilities. The current OpenAI contract returns a label only, so full hybrid ECE/Brier calibration is unavailable and must remain `null` with an explanation.
5. The P95 target is a hypothesis, not an assumption. Sequential fallback naturally worsens deferred-request latency; the hybrid can still win overall if enough records terminate at Jev.
6. Bootstrap quality differences will be paired by `sample_id`. Clustered bootstrap by template/scenario is deferred because the dataset has no authoritative cluster identifier; the report must state this limitation.

## File Structure

- Create `src/cx_sentiment/study_config.py`: strict YAML schema and path resolution for offline, preview, and live study modes.
- Create `src/cx_sentiment/study.py`: saved-run alignment, threshold sweep, conditional error summaries, latency statistics, bootstrap intervals, and Pareto selection.
- Create `src/cx_sentiment/benchmark.py`: deterministic paired/interleaved live scheduling and attempt-level collection.
- Create `src/cx_sentiment/reports/performance_study.qmd`: report presentation only; calculations remain in Python.
- Create `experiments/routing-study.yaml`: safe offline example using saved validation runs.
- Create `docs/PERFORMANCE_STUDY.md`: measurement specification, protocol, and live authorization checklist.
- Keep `src/cx_sentiment/models.py` unchanged: study evidence uses separate models rather than expanding `Prediction`.
- Modify `src/cx_sentiment/cli.py`: add the small YAML-first `study` command.
- Modify `src/cx_sentiment/reporting.py`: atomically write study artifacts and render the bundled study template.
- Modify `src/cx_sentiment/provenance.py`: identify study format/version and separate exploratory, simulated, and controlled scopes.
- Modify `README.md` and `docs/METHODS.md`: document the workflow and validation/test boundary.
- Create `tests/test_study_config.py`, `tests/test_study.py`, `tests/test_benchmark.py`, and `tests/test_performance_reporting.py`.
- Create `tests/test_study_cli.py`; modify `tests/test_package.py` for package-resource coverage.

## Review Focus

- A test-set input or mixed split must never be accepted for threshold selection.
- Missing, duplicated, reordered, or mismatched saved predictions must fail before analysis.
- Cache hits and historical timings must never be labeled controlled live measurements.
- Budget exhaustion, interruption, and provider failure must leave a complete attempt ledger with unresolved work marked explicitly.
- Unknown costs or missing stage timings must remain unknown and must not enter a Pareto comparison as zero.

### Task 1: Freeze the Measurement Contract

**Files:**
- Create: `docs/PERFORMANCE_STUDY.md`
- Modify: `docs/METHODS.md`
- Test: `tests/test_documentation.py`

**Interfaces:**
- Produces: definitions used by all later artifacts: `provider_latency_ms`, `stage_wall_ms`, `operation_wall_ms`, `scheduler_wait_ms`, `cache_status`, `attempt_number`, `measurement_scope`, and `condition`.

- [ ] **Step 1: Write the failing documentation test** — assert that the performance document defines provider-stage versus operation latency, retries, failures, cache treatment, cold/warm conditions, interleaving, and the validation/test boundary.
- [ ] **Step 2: Run it, verify it fails** — `UV_CACHE_DIR=/tmp/jev_study_uv_cache uv run pytest tests/test_documentation.py -q`; expect the new-document assertion to fail.
- [ ] **Step 3: Write the measurement specification** — document that existing `Prediction.latency_ms` measures one adapter call from before SDK/HTTP invocation through local parsing, automatic retries are disabled, runner persistence/reporting time is excluded, and historical cached timings are not current-invocation latency. Define controlled `operation_wall_ms` around the entire classification operation and stage timings around Jev and fallback calls.
- [ ] **Step 4: Record the audit corrections** — list which existing runs are comparable for quality/cost simulation and why none is sufficient for matched live latency acceptance.
- [ ] **Step 5: Run the documentation test** — repeat Step 2 and expect success; run `git diff --check`.
- [ ] **Step 6: Commit** — `git add docs/PERFORMANCE_STUDY.md docs/METHODS.md tests/test_documentation.py && git commit -m "docs: define performance study measurements"`.

### Task 2: Add a Strict YAML Study Contract

**Files:**
- Create: `src/cx_sentiment/study_config.py`
- Create: `tests/test_study_config.py`
- Create: `experiments/routing-study.yaml`

**Interfaces:**
- Produces: `StudyConfig`, `OfflineInputs`, `LiveBenchmarkConfig`, and `load_study_config(path: Path) -> StudyConfig`.
- Consumes later: resolved saved-run paths, output parent, threshold grid, quality constraints, seed, sample size, repetitions, concurrency, conditions, request cap, and USD caps.

- [ ] **Step 1: Write failing schema tests** — cover relative-path resolution, defaults for thresholds `0.40..0.99`, unknown keys, non-validation inputs, invalid quantiles, nonpositive limits, concurrency outside `{1,5,10}`, missing live budgets, and credentials excluded from snapshots.
- [ ] **Step 2: Run them, verify failure** — `UV_CACHE_DIR=/tmp/jev_study_uv_cache uv run pytest tests/test_study_config.py -q`; expect import failure.
- [ ] **Step 3: Implement the models** — use frozen Pydantic models with `extra="forbid"`; represent thresholds as integer percentage points internally to avoid floating-step drift. Require `sample_size`, `repetitions`, `max_requests`, `budget_usd`, and `cost_per_request_usd` for live mode. Default the first live configuration to concurrency one and warm condition only; cold collection must be explicitly selected.
- [ ] **Step 4: Add the offline example** — point it at the complete Jev and OpenAI validation run directories and write outputs under `experiments/runs`; include no credentials or live section.
- [ ] **Step 5: Run focused tests and lint** — run the Step 2 command and `UV_CACHE_DIR=/tmp/jev_study_uv_cache uv run ruff check src/cx_sentiment/study_config.py tests/test_study_config.py`.
- [ ] **Step 6: Commit** — `git add src/cx_sentiment/study_config.py tests/test_study_config.py experiments/routing-study.yaml && git commit -m "feat: define YAML performance studies"`.

### Task 3: Implement Offline Threshold and Risk-Coverage Analysis

**Files:**
- Create: `src/cx_sentiment/study.py`
- Create: `tests/test_study.py`
- Modify: `src/cx_sentiment/routing.py`

**Interfaces:**
- Produces: `ThresholdPoint`, `LanguageThresholdPoint`, `ConditionalErrors`, `OfflineStudyResult`, and `analyze_saved_validation(config: StudyConfig) -> OfflineStudyResult`.
- Consumes: existing `Prediction`, `hybrid_prediction`, `evaluate_predictions`, dataset fingerprints, and complete-run metadata.

- [ ] **Step 1: Write failing alignment tests** — reject test/mixed splits, incomplete runs, unequal fingerprints, missing or duplicate IDs, incompatible prompt/model metadata, missing Jev probability vectors, and absent fallback predictions.
- [ ] **Step 2: Write failing metric tests** — for synthetic paired predictions, assert threshold equality is accepted, thresholds cover every integer point from 40 through 99, and each point reports accuracy, macro F1, paired delta from OpenAI, fallback fraction, Jev coverage, accepted/deferred error rates, expected cost per 1,000, plus per-language F1 and coverage.
- [ ] **Step 3: Run them, verify failure** — `UV_CACHE_DIR=/tmp/jev_study_uv_cache uv run pytest tests/test_study.py -q`; expect import or assertion failures.
- [ ] **Step 4: Implement saved-run loading and alignment** — reuse comparison/provenance readers rather than constructing providers; keep the original runs immutable. Treat any invalid Jev output as deferred and any missing fallback as unresolved.
- [ ] **Step 5: Implement the sweep** — calculate all metrics from aligned saved validation predictions. Use recorded per-record costs when fully priced; otherwise require explicit dated assumptions and preserve unknown totals.
- [ ] **Step 6: Add paired uncertainty** — bootstrap records with a fixed seed to report 95% intervals for hybrid-minus-OpenAI macro F1. Do not claim clustered uncertainty.
- [ ] **Step 7: Select candidates** — shortlist non-dominated thresholds that meet the configured quality and language constraints; do not select from historical latency. Return no recommendation when required values are unknown.
- [ ] **Step 8: Run focused verification** — run the Step 3 command, Ruff on touched files, and Pyright on `src/cx_sentiment/study.py`.
- [ ] **Step 9: Commit** — `git add src/cx_sentiment/study.py src/cx_sentiment/routing.py tests/test_study.py && git commit -m "feat: analyze offline routing thresholds"`.

### Task 4: Add Reproducible Attempt-Level Benchmark Instrumentation

**Files:**
- Create: `src/cx_sentiment/benchmark.py`
- Create: `tests/test_benchmark.py`
- Modify: `src/cx_sentiment/provenance.py`

**Interfaces:**
- Produces: `BenchmarkAttempt`, `BenchmarkOperation`, `BenchmarkRun`, `build_schedule(...)`, and `run_live_benchmark(...)`.
- Consumes: existing provider factory, strict predictions, `LiveSafety`, and Task 2 configuration.

- [ ] **Step 1: Write failing schedule tests** — assert deterministic stratified sampling by language/class, identical sample IDs across strategies, reproducible seeded order, balanced provider-order rotation across repetitions, and no provider construction during preview.
- [ ] **Step 2: Write failing instrumentation tests** — use fake providers and a fake monotonic clock to prove each attempt records strategy, stage, repetition, condition, attempt number, start/order index, stage wall time, operation wall time, retry/backoff, outcome, usage, cost, and cache status. Assert hybrid operation time includes Jev, fallback when taken, and local overhead.
- [ ] **Step 3: Write failing safety tests** — verify cache bypass for controlled live mode, cache-hit labeling for explicit cache mode, no automatic retries, bounded manual retry accounting when later enabled, budget-stop checkpointing, interruption checkpointing, and no calls without `--live`.
- [ ] **Step 4: Run them, verify failure** — `UV_CACHE_DIR=/tmp/jev_study_uv_cache uv run pytest tests/test_benchmark.py -q`; expect import failure.
- [ ] **Step 5: Implement evidence models and scheduler** — keep them separate from `Prediction`; write JSONL attempts and operations incrementally, then write completion metadata last. Use explicit strategy order blocks that are shuffled within each repetition while retaining paired records.
- [ ] **Step 6: Implement the first live protocol only** — support Jev-only, OpenAI-only, and sequential hybrid at concurrency one. Model cold/warm as explicit tagged conditions; never infer coldness from the first arbitrary request.
- [ ] **Step 7: Reserve concurrency plumbing without executing it** — configuration may validate 5/10, but execution must require an explicit second-stage authorization marker after the concurrency-one pilot. Do not implement unlimited concurrency.
- [ ] **Step 8: Run focused verification** — run Step 4, Ruff, Pyright, and `git diff --check`.
- [ ] **Step 9: Commit** — `git add src/cx_sentiment/benchmark.py src/cx_sentiment/provenance.py tests/test_benchmark.py && git commit -m "feat: record controlled benchmark attempts"`.

### Task 5: Add Latency, Throughput, and Pareto Statistics

**Files:**
- Modify: `src/cx_sentiment/study.py`
- Modify: `tests/test_study.py`

**Interfaces:**
- Produces: `LatencySummary`, `ThroughputSummary`, `ParetoPoint`, `summarize_benchmark(...)`, and `pareto_frontier(...)`.
- Consumes: completed `BenchmarkRun` artifacts from Task 4 and offline quality results from Task 3.

- [ ] **Step 1: Write failing summary tests** — assert P50/P90/P95/P99, operations per second, failure/retry/rate-limit counts, stage and total distributions, and bootstrap intervals are computed from operation-level wall times with failed/retried operations retained.
- [ ] **Step 2: Write failing comparability tests** — reject incomplete runs, mixed datasets, different samples/repetitions/conditions, cached live measurements, missing stage evidence, and incompatible provider/prompt identities.
- [ ] **Step 3: Write failing Pareto tests** — assert domination across quality, P95, and cost; exclude unknown dimensions from categorical domination; support matching within the configured one-point macro-F1 tolerance and per-language three-point constraint.
- [ ] **Step 4: Run them, verify failure** — `UV_CACHE_DIR=/tmp/jev_study_uv_cache uv run pytest tests/test_study.py -q`; expect new assertions to fail.
- [ ] **Step 5: Implement deterministic summaries** — compute quantiles with one documented NumPy method and bootstrap paired record/repetition units. Label P99 exploratory when effective sample size is below the configured minimum.
- [ ] **Step 6: Implement decision analysis** — combine only controlled latency with offline quality when dataset/config compatibility is proven. Produce candidate policies, not a forced winner.
- [ ] **Step 7: Run focused verification** — repeat Step 4, then Ruff and Pyright on touched files.
- [ ] **Step 8: Commit** — `git add src/cx_sentiment/study.py tests/test_study.py && git commit -m "feat: summarize benchmark tradeoffs"`.

### Task 6: Add YAML-First Orchestration and Dry Run

**Files:**
- Modify: `src/cx_sentiment/cli.py`
- Create: `tests/test_study_cli.py`

**Interfaces:**
- Produces: `uv run cx-sentiment study experiments/routing-study.yaml` for offline analysis/preview and the same command plus `--live` for an explicitly authorized live configuration.
- Consumes: Tasks 2–5.

- [ ] **Step 1: Write failing CLI tests** — verify offline study works without credentials/provider construction; live YAML without `--live` writes a zero-call preview; `--live` requires budgets; legacy run flags cannot be mixed; errors omit configuration values and tracebacks; successful commands print the output path.
- [ ] **Step 2: Run them, verify failure** — `UV_CACHE_DIR=/tmp/jev_study_uv_cache uv run pytest tests/test_study_cli.py -q`; expect parser failure.
- [ ] **Step 3: Add the command** — add only positional YAML plus `--live`; route offline mode to Task 3, live preview to a request/cost plan, and authorized live mode to Task 4. Snapshot resolved non-secret settings in the dated output child.
- [ ] **Step 4: Exercise zero-network paths** — run the focused tests and the example offline command into a temporary directory; inspect generated JSON and verify no execution ledger records network calls.
- [ ] **Step 5: Commit** — `git add src/cx_sentiment/cli.py tests/test_study_cli.py && git commit -m "feat: run performance studies from YAML"`.

### Task 7: Build the Quarto Performance Study Report

**Files:**
- Create: `src/cx_sentiment/reports/performance_study.qmd`
- Modify: `src/cx_sentiment/reporting.py`
- Create: `tests/test_performance_reporting.py`
- Modify: `tests/test_package.py`

**Interfaces:**
- Produces: `study.json`, normalized CSV tables, `report.qmd`, and `report.html` from saved evidence only.
- Consumes: offline and controlled-live results from Tasks 3 and 5.

- [ ] **Step 1: Write failing artifact tests** — assert atomic JSON/CSV output, bundled template availability in built wheels, HTML escaping, and successful re-render without original source folders.
- [ ] **Step 2: Write failing content tests** — require separate sections for measurement audit, historical exploratory values, offline threshold simulation, controlled live measurements, language slices, limitations, and recommendation status.
- [ ] **Step 3: Run them, verify failure** — `UV_CACHE_DIR=/tmp/jev_study_uv_cache uv run pytest tests/test_performance_reporting.py tests/test_package.py -q`; expect missing-template failures.
- [ ] **Step 4: Implement Python-owned report data** — write tables for quality/coverage, conditional risk, fallback/cost, language metrics, latency distributions, throughput, and Pareto points. The QMD must only load and present those artifacts.
- [ ] **Step 5: Implement visuals** — Plotly risk-coverage, F1-fallback, cost-quality, threshold-fallback, latency ECDF/quantiles, throughput-latency, and quality-P95-cost charts. Use Great Tables for candidate policies and measurement caveats.
- [ ] **Step 6: Render and inspect** — render a synthetic fixture with Quarto, inspect the HTML for headings, table values, chart payloads, scope labels, and absence of raw customer text/secrets.
- [ ] **Step 7: Commit** — `git add src/cx_sentiment/reports/performance_study.qmd src/cx_sentiment/reporting.py tests/test_performance_reporting.py tests/test_package.py && git commit -m "feat: report routing performance tradeoffs"`.

### Task 8: Documentation, Integration Verification, and Authorization Gate

**Files:**
- Modify: `README.md`
- Modify: `docs/PERFORMANCE_STUDY.md`
- Modify: `docs/METHODS.md`

**Interfaces:**
- Produces: a reproducible offline command, a zero-call live preview command, and a written checklist that must be satisfied before paid collection.

- [ ] **Step 1: Document the workflow** — show offline analysis first, then live preview; state that actual live execution requires explicit user authorization after reviewing sample size, repetitions, request maximum, and USD maximum.
- [ ] **Step 2: Run the offline acceptance flow** — `UV_CACHE_DIR=/tmp/jev_study_uv_cache uv run cx-sentiment study experiments/routing-study.yaml`; verify the report path, matched validation fingerprints, 60 thresholds, and zero network calls.
- [ ] **Step 3: Run the preview acceptance flow** — use a temporary live YAML with 100–150 stratified records, seed, one repetition, concurrency one, and explicit budgets, but omit `--live`; verify the request plan and zero provider calls.
- [ ] **Step 4: Run the repository gate** — `UV_CACHE_DIR=/tmp/jev_study_uv_cache uv run pytest`, `UV_CACHE_DIR=/tmp/jev_study_uv_cache uv run ruff check .`, `UV_CACHE_DIR=/tmp/jev_study_uv_cache uv run pyright`, and `git diff --check`.
- [ ] **Step 5: Inspect the staged diff** — ensure no locked dataset, saved historical run, cache, raw response, secret, generated live output, or unrelated file is staged.
- [ ] **Step 6: Commit** — `git add README.md docs/PERFORMANCE_STUDY.md docs/METHODS.md && git commit -m "docs: explain the routing performance study"`.
- [ ] **Step 7: Stop before paid work** — report the offline findings, candidate thresholds, exact proposed live request/cost envelope, and remaining uncertainty. Wait for explicit authorization; do not start the live benchmark.

## Separately Authorized Evidence Collection

After Tasks 1–8 are reviewed and the user explicitly authorizes a stated budget:

1. Run the concurrency-one pilot on the predeclared 100–150 stratified validation records, with paired/interleaved strategy order and the frozen candidate thresholds.
2. Inspect completeness, failures, rate limits, request accounting, costs, and stability before expanding the sample or repetitions.
3. Treat P99 as exploratory unless the expanded design provides sufficient tail observations.
4. Request separate authorization before concurrency 5/10 or before using the locked test set.
5. Freeze one operating policy from validation evidence, then run the locked test once and report uncertainty without retuning.

## Completion Criteria

- Offline study reproduces from saved runs with zero provider calls.
- Every threshold result is traceable to aligned validation records and immutable saved predictions.
- Controlled runs preserve attempt- and operation-level evidence, including failures and retries.
- Historical, simulated, cache-hit, cold, warm, and controlled-live measurements are never conflated.
- Recommendation logic applies the quality, language, latency, and cost constraints without converting uncertainty into unsupported pass/fail certainty.
- No paid call, concurrency test, or final-test evaluation occurs without its own explicit authorization.
