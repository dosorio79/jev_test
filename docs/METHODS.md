# Methods

## Research questions and hypotheses

The benchmark measures zero-shot three-way sentiment classification and asks whether a typed decision provider can reduce LLM calls while keeping validation macro F1 within a predefined absolute degradation, initially 0.02, of OpenAI-only. The routing hypothesis is that low maximum class probability identifies cases worth escalating. Jev provider confidence is evaluated as a separate signal because it is not the same quantity as the returned class probability.

## Data protocol

The locked test contains 600 synthetic comments: 100 per language and 200 per class. It is reserved for one final evaluation. Prompt choice, confidence calibration, and thresholds use only a separately seeded 600-row sample of the original `validation.csv`. The sampling manifest records source indices and archive SHA256. Exact normalized duplicates are rejected within the sample and across source splits.

The source corpus was generated from 108 seed sentences with templated variations. Exact deduplication cannot remove semantic or template near-duplicates, so results do not establish generalization to independent real-world customer feedback. A trainable model would require grouping templates before any training split; this POC performs no fine-tuning.

## Prediction contract

Each result records provider, actual model ID, prompt version, canonical label, genuine class probabilities when available, provider confidence when available, its definition, latency, tokens, credits, USD estimate when a dated rate card exists, failure category, a response hash, and sample ID. CLI runs select rate cards from `pricing.py` by model ID. Missing probabilities remain missing. Customer text is marked as untrusted data and is never given the gold label or source metadata.

Provider failures remain in the denominator. Routing falls back on decision-provider failure or when the chosen signal is below the threshold. If fallback also fails, the record is unresolved. Sequential hybrid latency is the sum of the decision and fallback calls for routed records; simulations must be labeled as simulated rather than measured wall-clock runtime.

## Threshold selection

Sweep thresholds from 0.50 through 0.99 on cached validation predictions. A score equal to the threshold is accepted by the decision provider. Select the lowest expected-cost policy whose macro F1 is no more than 0.02 below the cached OpenAI-only baseline. If none qualifies, record no selected threshold. Freeze the selected policy and configuration before final test evaluation.

## Metrics

Report accuracy, macro and per-class precision/recall/F1, support, confusion matrices, language breakdowns, and operational coverage. Use deterministic 95% bootstrap intervals over sample IDs. Report expected calibration error and multiclass Brier score only when every record has a valid three-class probability vector. Report latency P50/P95, fallback fraction, token usage, separately reported credits when present, and USD per 1,000 records from dated official or user-supplied rate cards. Point estimates alone are not evidence of statistical significance.

## Spend and privacy

The default CLI is a dry run. Live calls require `--live`, an explicit request cap, a total USD budget, and a per-request upper bound. The first smoke test is capped at five records and concurrency one. Provided comments are synthetic, telemetry is disabled, secrets stay in environment variables, full provider bodies are excluded from outputs, and raw caches are gitignored.
