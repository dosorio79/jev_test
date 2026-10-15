# Multilingual CX Sentiment Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible offline-first benchmark with guarded OpenAI and Jev live adapters.

**Architecture:** A typed provider boundary feeds a cache-aware runner, metrics, threshold selection, and a self-contained HTML report. Network clients are injected and live execution is gated by explicit flags, credentials, request caps, and a dollar budget.

**Tech Stack:** Python 3.11+, uv, Pydantic, HTTPX, OpenAI SDK, scikit-learn, pytest, Ruff.

**Spec:** `docs/CODEX_PROMPT.md`

## Global Constraints

- Preserve the locked 600-row test sample byte-for-byte and never tune on it.
- Default to offline execution with zero network calls.
- Use only `negative`, `neutral`, and `positive`, mapped to `0`, `1`, and `2`.
- Require `--live`, a request limit, and a USD budget before any live request.
- Keep raw responses and generated outputs in gitignored paths.

## Review Focus

- Missing source archive must produce an actionable validation-sampling error.
- Test data must be rejected for threshold selection.
- A score exactly equal to the threshold must stay on the decision-provider route.
- Missing probability vectors must never be synthesized.
- Cached requests must not consume budget or network calls.

---

### Task 1: Data contracts, audit, and validation sampler

**Files:** Create `src/cx_sentiment/models.py`, `src/cx_sentiment/data.py`, `tests/test_data.py`; modify `make_sample.py`.

**Interfaces:** Produces typed records, canonical labels, dataset audit results, and deterministic archive sampling.

- [ ] Write failing tests for locked-test balance, mappings, archive sampling, overlap detection, and test-byte preservation.
- [ ] Implement the typed data boundary, audit, and CLI-configurable sampler.
- [ ] Run `uv run pytest tests/test_data.py`.

### Task 2: Providers, cache, budget, and runner

**Files:** Create provider modules, `cache.py`, `budget.py`, `runner.py`, and focused tests.

**Interfaces:** Produces `predict(...) -> Prediction`, content-addressed caching, and resumable guarded execution.

- [ ] Write failing tests for mock determinism, label parsing, Jev signal separation, OpenAI structured parsing, caching, budget cutoff, and provider failures.
- [ ] Implement mock, OpenAI, and Jev adapters with injected clients/transports.
- [ ] Implement cache-aware execution with explicit live safety gates.
- [ ] Run focused provider and runner tests.

### Task 3: Routing, metrics, selection, reports, and CLI

**Files:** Create `routing.py`, `metrics.py`, `reporting.py`, `cli.py`, and focused tests.

**Interfaces:** Consumes cached predictions and emits CSV/JSON/HTML benchmark artifacts.

- [ ] Write failing tests for threshold boundaries, failure fallback, metric values, probability-only calibration, validation-only selection, and offline end-to-end output.
- [ ] Implement routing simulation, metric calculation, bootstrap intervals, policy selection, and report generation.
- [ ] Add CLI commands for audit, sample-validation, offline demo, run, select-threshold, and report.
- [ ] Run focused tests and the offline demo.

### Task 4: Documentation and final verification

**Files:** Create or modify `README.md`, `docs/METHODS.md`, `.env.example`, `.gitignore`, `pyproject.toml`, and `uv.lock`.

**Interfaces:** Documents installation, scientific protocol, provider provenance, and guarded five-record live commands.

- [ ] Document exact offline and live commands, missing archive limitation, leakage risks, and pricing requirements.
- [ ] Run `uv run ruff format --check .`, `uv run ruff check .`, `uv run pyright`, and `uv run pytest`.
- [ ] Recheck the locked-test SHA256 and inspect generated offline artifacts.
