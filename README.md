# Multilingual CX sentiment benchmark

An offline-first research POC for comparing multilingual financial customer-experience sentiment providers and selective routing. The fixed taxonomy is `negative=0`, `neutral=1`, and `positive=2` across Portuguese, English, Spanish, French, German, and Italian.

The dataset boundary uses Polars and Dataframely, so schema, nullability, uniqueness, categorical values, and the label/sentiment relationship are validated as dataframe operations. Pydantic is reserved for provider predictions and configuration.

## Install and verify

Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run pytest
uv run ruff check .
uv run pyright
```

## Run an experiment

Edit [`experiment.yaml`](experiment.yaml), then run:

```bash
uv run cx-sentiment run experiment.yaml
```

The supplied config uses `mock`, five records, and `outputs/experiment`. It makes
no network calls and writes `predictions.jsonl`, `metrics.json`, `metrics.csv`, and
`report.html`. Mock accuracy is a pipeline test, not a model result.

```yaml
provider: mock  # mock, openai, or jev
input: data/cx_sentiment_test_600.csv
output_dir: outputs/experiment
limit: 5
```

Paths are relative to the YAML file. Unknown settings and nonpositive limits are
rejected. `limit` also caps provider requests. Use a separate output folder for
each experiment you want to keep; rerunning replaces its reports.
The command prints the report location when finished. Configuration errors name
the file and setting to fix.

For a paid provider, change `provider`, set credentials as described below, and add
explicit cost caps chosen for your model and input:

```yaml
budget_usd: 0.05
cost_per_request_usd: 0.01
```

Preview with the same command; it writes `dry_run.json` without requiring credentials.
To execute paid calls explicitly:

```bash
uv run cx-sentiment run experiment.yaml --live
```

The CLI does not read credentials from YAML. OpenAI's model remains configured with
`OPENAI_MODEL`. Legacy flag-based commands still work, but cannot be mixed with YAML.

## How the code fits together

`cli.py` reads the experiment settings from `config.py` and loads the dataset through
`data.py`. `runner.py` checks the cache and budget, then calls the selected provider.
Each provider returns the same `Prediction` model. `metrics.py` scores those results;
`reporting.py` writes JSON, CSV, and HTML. `routing.py` compares cached decision and
fallback predictions to select a validation threshold without extra API calls.

## Validation sample

The source archive is available and its SHA256 matches the starter manifest. The generated validation sample is balanced at 600 records with no normalized exact cross-split overlap. Regenerate it with:

```bash
uv run cx-sentiment sample-validation \
  --archive data/raw/financial_sentiment_hf_v2.zip \
  --output-dir data
```

This creates a deterministic balanced `data/validation_600.csv` with seed `20261016` and a manifest containing the archive SHA256 and all selected source indices. It reads only `validation.csv` for the sample and checks normalized exact overlap against the archive's train and test splits. The locked test CSV is never rewritten.

## Provider configuration

Copy `.env.example` to `.env` and export it in your shell. The CLI does not load `.env` implicitly, which avoids surprising secret reads:

```bash
set -a; . ./.env; set +a
```

OpenAI uses the official Python SDK and Responses API structured output with one enum label. Scores and confidence remain null because the adapter does not invent probabilities. Set `OPENAI_MODEL` explicitly. If all three dated rate-card variables are supplied, actual token usage is converted to a USD estimate.

The built-in standard short-context rate table, checked on 2026-10-15, covers `gpt-6-astra` ($10 input/$50 output), `gpt-6.1-sol` ($2/$10), and `gpt-6-luna` ($0.10/$0.50), all per million tokens. Environment rate variables override the table for another model or processing tier. The source URL is stored with every rate card.

Jev uses TypeSafe's official `https://api.typesafe.ai/v1/systemone` endpoint and a typed `choice` question. The default model is pinned to `jev-1.13.0` so calibrated thresholds do not move with an alias. Its probability vector remains separate from provider confidence. TypeSafe lists Jev 1.13 and 1.14 at $0.042 per million input tokens with output tokens free; the dated table in `pricing.py` converts actual usage to USD. Override the URL or model only when the issuer of your key documents a different contract.

Live mode requires credentials, a positive record limit, a total USD budget, and a
per-request upper bound. Concurrency is one. Responses are resumable through the
content-addressed `.cache/cx_sentiment/` cache. Keys and full raw response bodies are
never logged or written to normal outputs.

## Scientific boundary

Threshold selection belongs exclusively on the generated validation set. Cache both providers' validation predictions once, then simulate thresholds from cached results. Freeze the prompt version, model IDs, routing signal, and chosen threshold before the one-time test evaluation. See [METHODS.md](docs/METHODS.md) for the protocol and limitations.

After live validation prediction files exist, select and persist the policy without making new API calls:

```bash
uv run cx-sentiment select-threshold \
  --input data/validation_600.csv \
  --decision-predictions outputs/jev-validation/predictions.jsonl \
  --fallback-predictions outputs/openai-validation/predictions.jsonl \
  --output outputs/validation-selection.json --max-f1-degradation 0.02
```

Regenerate a report from any cached prediction file:

```bash
uv run cx-sentiment report --input data/validation_600.csv \
  --predictions outputs/jev-validation/predictions.jsonl \
  --output-dir outputs/jev-validation-report
```
