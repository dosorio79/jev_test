# Contributing

Read [AGENTS.md](AGENTS.md) for repository conventions and [README.md](README.md)
for setup. Keep changes focused and preserve YAML experiments, sequential
execution, strict prediction validation, and honest missing measurements.

## Local checks

Use Python 3.11 or 3.12, uv, and the Quarto CLI. Install locked dependencies and
run the relevant checks before submitting changes:

```bash
uv sync --frozen
uv run pytest
uv run ruff check .
uv run pyright
git diff --check
```

Report integration tests render real Quarto documents. Exercise changed workflows
with mock data or paid-provider previews, writing to a temporary output directory.
Tests and CI must never require provider credentials or make paid calls. Document
checks that could not run. Documentation-only changes need source/link checks.

## Branch and review workflow

Create feature/fix branches from `dev` and open PRs into `dev`. Release PRs target
`main` and must come from this repository's `dev`. Squash feature PRs into `dev`;
use merge commits for `dev` to `main` release promotion and subsequent `main` to
`dev` synchronization. Keep both long-lived branches after merging. `main` is
the public default branch.

The intended required checks are `Quality`, `Security`, and `Branch policy`.
Repository protection is activated only after workflows and branches exist and
checks have succeeded; consult the infrastructure bootstrap instructions for
actual settings. No second-person approval is required for solo maintenance.
Do not bypass checks or force-push protected branches.

## Data and research boundaries

Never edit the locked test sample or use it to tune prompts, models, or thresholds.
Follow [METHODS.md](docs/METHODS.md). Mock scores check pipeline behavior. They
are not evidence of provider quality. Preserve original runs and record model,
settings, input provenance, pricing date, and any missing measurements.

Do not submit keys, ignored `.env`, caches, raw provider bodies, private text,
or generated experiment runs. Review staged changes before committing. The
project code is MIT licensed, but this does not grant rights to bundled datasets:
read [DATA_LICENSE.md](docs/DATA_LICENSE.md) before adding or redistributing data.
See [SECURITY.md](SECURITY.md) for private vulnerability reporting.
