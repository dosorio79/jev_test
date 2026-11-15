# Routing performance study

This study separates three kinds of evidence: historical exploratory results,
offline routing simulations, and controlled live measurements. Values from one
scope must not be relabeled as another.

## Measurement contract

- `provider_latency_ms` is the duration reported by the existing provider
  adapter. It starts immediately before the SDK or HTTP invocation and ends
  after the response has been parsed into a `Prediction`. It describes one
  adapter call, not a complete hybrid classification.
- `stage_wall_ms` is the benchmark's monotonic wall time around one Jev or
  OpenAI stage, including local adapter and parsing overhead.
- `operation_wall_ms` is monotonic wall time around the complete classification
  operation. For a sequential hybrid it includes the Jev stage, any OpenAI
  fallback stage, and local routing overhead.
- `scheduler_wait_ms` is time after an operation becomes eligible but before its
  first stage starts. It is reported separately and is not added to
  `operation_wall_ms`.
- `attempt_number` starts at one for each stage. Automatic provider retries are
  disabled. Any future manual retry is a separate attempt and its backoff and
  outcome remain in the evidence ledger.
- `cache_status` distinguishes bypassed, hit, and miss states. Controlled live
  inference bypasses prediction caches. Cache-hit performance is useful
  operational evidence, but it is reported separately from live inference.
- `measurement_scope` is one of historical, simulated, or controlled. Existing
  `Prediction.latency_ms` values are historical even when loaded from a cache;
  a cache hit does not make the original request latency current-invocation
  latency.
- `condition` is explicitly declared as `cold` or `warm`. The first arbitrary
  request is not assumed to be cold. Conditions are collected and summarized
  separately.

Every provider failure, rejected response, retry, rate-limit response, and
interruption remains in the denominator and attempt ledger. Missing timings,
usage, probabilities, or costs remain unknown rather than becoming zero.
Runner persistence and report rendering occur outside `operation_wall_ms`.

## Controlled collection protocol

The same sample IDs are used for Jev-only, OpenAI-only, and hybrid strategies.
Provider and strategy order is deterministically interleaved by repetition to
reduce temporal bias. The first live pilot uses concurrency one. Higher
concurrency is a separately authorized throughput experiment.

Live collection requires `--live`, explicit request and USD caps, and fresh user
authorization for the stated envelope. Preview and offline analysis construct no
provider clients and make no network calls.

## Reproducible workflow

Run the supplied offline validation study first:

```bash
uv run cx-sentiment study experiments/routing-study.yaml
```

For a proposed live YAML, omit `--live` to write a zero-call preview. Before any
paid collection, review and explicitly authorize all of the following:

- selected validation sample size and random seed;
- repetitions and warm/cold condition;
- frozen routing threshold;
- concurrency, initially one;
- maximum provider requests;
- total USD budget and per-request USD reservation.

Authorization for the concurrency-one pilot does not authorize concurrency 5/10,
additional repetitions, a larger sample, or use of the locked test set.

## Data boundary

Threshold selection and calibration use only data marked `validation`. The
locked test set is not read for threshold selection and is evaluated once only
after a policy is frozen and separately authorized. Mixed validation/test inputs
are invalid.

Saved Jev and OpenAI validation predictions can support matched quality and cost
simulation when IDs, fingerprints, prompts, and model metadata align. The
historical runs were collected at different times and retain original timings
when cached, so they do not support a matched live latency acceptance decision.
They remain visible as exploratory evidence.

## Audit corrections

The existing adapter timer measures one SDK/HTTP call plus local parsing. It does
not measure scheduling, runner persistence, reporting, or the whole hybrid
operation. Standard provider clients have automatic retries disabled. Existing
saved predictions may be reused for quality, coverage, calibration where full
probabilities exist, and cost simulation where costs are complete. New attempt-
and operation-level evidence is required for comparable P50, P90, P95, P99,
throughput, retry, failure, and rate-limit measurements.
