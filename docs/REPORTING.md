# Learning Quarto with this benchmark

Run an experiment to get a report you can open, edit, and re-render locally:

```bash
uv run cx-sentiment run experiments/mock.yaml
```

The supplied mock experiment makes no provider calls. Open
the printed `report.html` path in your browser. Each YAML invocation creates a
new dated folder under `experiments/runs/`, containing its resolved `experiment.yaml` and results. Its styles and plots are embedded,
so this HTML file can be shared on its own and viewed offline.

Local model runs can report quality, latency, and token usage without a USD price.
Unpriced runs display cost as unavailable. Optional electricity estimates appear
separately in euros, using an explicit tariff and distinguishing sampled GPU
energy from assumed whole-computer power. These estimates exclude hardware
amortization and do not replace provider pricing or imply that local inference is free.

The all-run comparison is configured in
[`experiments/compare-validation-all.yaml`](../experiments/compare-validation-all.yaml).
It includes the local Qwen and live hybrid snapshots. A `cost_estimates` entry can
supply a USD electricity ballpark for an otherwise entirely unpriced run, with its
assumptions in `basis`. Charts label it as an electricity estimate; original saved
predictions and provider-cost metrics remain unchanged. Latency charts include
every run on the full measured range.

## One-time setup

Install the [Quarto CLI](https://quarto.org/docs/get-started/) for your operating
system and ensure `quarto` is on PATH. In WSL, install the Linux version in WSL;
a Windows installation is separate. Quarto is a separate application, not the
Python package named `quarto`.

Then install the project's Python dependencies and check the Python engine:

```bash
uv sync
quarto --version
uv run quarto check jupyter
```

The project includes ipykernel/nbclient for Python execution, Great Tables for Polars
tables, and Plotly for interactive charts. It does not need a Jupyter notebook server or a PDF/TeX
installation. The application selects its own Python interpreter when rendering.

## What is a .qmd file?

A Quarto document combines three things:

1. YAML between `---` lines configures the title and output format.
2. Markdown supplies headings and explanatory text.
3. Executable Python blocks read saved results and display tables or figures.

Here is a small example of the pattern used in the benchmark document:

````markdown
---
title: "My benchmark report"
format:
  html:
    embed-resources: true
jupyter: python3
---

## Language results

```{python}
#| echo: false
import json
from html import escape
from pathlib import Path
import polars as pl
from great_tables import GT

metrics = json.loads(Path("metrics.json").read_text())
languages = pl.DataFrame([
    {"Language": language, **values}
    for language, values in metrics["by_language"].items()
])
GT(languages).fmt(escape, columns="Language").fmt_number(
    columns=["accuracy", "macro_f1"], decimals=3
)
```
````

Save this example beside a generated `metrics.json`, then render it with
`uv run quarto render your-example.qmd`. The `{python}` fence executes code;
an ordinary `python` fence only shows code. `#|` lines configure a block, and
`echo: false` hides its code while retaining its output.
`GT` displays the Polars frame using Great Tables. Explicit escaping keeps text
values as text, since Great Tables otherwise permits HTML in string cells.

## Edit and re-render a report

Each dated run folder saves these files:

| File | Purpose |
| --- | --- |
| `experiment.yaml` | Resolved experiment settings, without credentials |
| `predictions.jsonl` | Provider predictions, one record per line |
| `metrics.json` | Full calculated evaluation results |
| `metrics.csv` | Overall and per-language summary |
| `report_metadata.json` | Run title and supplied metadata |
| `report.qmd` | Editable report document |
| `report.html` | Rendered report |

Change a heading or the chart colors in your run folder's `report.qmd`. Open a
terminal inside that dated run folder, then render the edited document (uv finds
the project environment in the parent repository):

```bash
uv run quarto render report.qmd --to html
```

For an updating preview while you edit:

```bash
uv run quarto preview report.qmd
```

These commands execute only the report's local Python blocks. The supplied
documents read saved JSON and make no provider calls. Re-rendering also works
without API credentials. Quarto may create intermediate notebook and `_files`
artifacts; keep those under the ignored `experiments/runs/` or `outputs/` folders.

For changes that should apply to future experiments, edit the packaged sources:

- [benchmark.qmd](../src/cx_sentiment/reports/benchmark.qmd) contains performance,
  language/class tables, confusion plot, latency, usage, costs, calibration, and metadata.
- [selection.qmd](../src/cx_sentiment/reports/selection.qmd) contains the selected
  routing policy, a cost/quality plot, and the complete threshold table.

Each YAML invocation copies the packaged document into a new dated run folder.
Existing runs and their local `.qmd` edits are preserved. The explicit
`cx-sentiment report` command writes to its requested output directory and replaces
that directory's `.qmd`; copy an edited document elsewhere before using that command
on the same folder.

The Plotly charts support hover details, zooming, and image download. Their
JavaScript is embedded in the HTML, so interaction does not need a Python server
or network access. Plotly Express accepts Polars directly; no pandas conversion
is used. Standalone HTML files are larger because they contain the plotting library.

Threshold selection saves a `.qmd` and `.html` beside its JSON output. For example:

```bash
uv run quarto render outputs/validation-selection.qmd --to html
```

The selection document finds the JSON with its own filename stem. Keep both files
together when moving or renaming it.

## Compare saved runs

Put the source run paths in a separate comparison YAML. For example,
`experiments/compare-smoke.yaml` contains:

```yaml
title: First provider comparison
runs:
  - runs/2026-10-16T20-29-40.101899Z-openai-live
  - runs/2026-10-16T20-30-08.698065Z-jev-live
output_dir: runs
```

Paths are relative to this YAML. `title` and `output_dir` are optional; the
default output parent is `runs/` beside the config. List at least two distinct
completed run folders. Preview folders do not contain predictions and cannot
be compared. Example source runs are generated locally and are not in Git.

```bash
uv run cx-sentiment compare experiments/compare-smoke.yaml
```

No credentials or provider calls are needed. The command prints the new dated
`*-comparison/report.html` location. Open it in your browser for:

- Side-by-side quality, coverage, latency, costs, and usage.
- Macro F1 versus USD per 1,000 records and median request latency.
- Per-language results and disagreements with the sample ID, original feedback
  verbatim, reference label, and each run's predicted label. Saved comparison
  JSON and HTML include that feedback text; tables display it as escaped text.
- Run paths, model IDs, prompt versions, and input verification status.

The reader validates the original source dataset using each run's
`experiment.yaml`, checks identical selected records, and aligns predictions by
sample ID. It rejects duplicates, missing/foreign IDs, mixed model identities in
a run, changed fingerprints, and incompatible data. Metrics are recalculated from
saved predictions, including errors in denominators. Failed and unknown-cost
predictions are not silently dropped. Incompletely priced runs have null total
costs. The cost table shows known USD, complete totals when available, and the
priced fraction. When `recollection_audit.json` exists, the report also shows
the number of fresh attempts and their additional request USD separately; those
costs are already included in the derived run total. Partially priced runs remain visible as lower-bound bars and
points; fully unpriced runs are explicitly listed without a zero-cost bar.

The report displays source titles, cost/usage scope, collection policy, and
latency scope from run metadata. Recollected results can include both original
rejected attempts and fresh requests. Original rejected-attempt latency may be
unavailable. Compare these scopes before ranking runs; do not add totals across
runs that share cached predictions or ancestry to estimate additional spend.
Separate raw-usage audits are not silently incorporated into original runs.

New experiment reports record a SHA256 of selected source records, including
text, labels, and provenance. The first smoke runs predate this field: they remain
usable, but show **historical input unverified**. Matching source data now does
not prove those files were identical when the old runs executed. Do not add a
fingerprint retroactively to imply verification.

Each comparison saves its resolved `comparison.yaml`, self-contained
`comparison.json`, editable `report.qmd`, and standalone `report.html`. The source
runs remain untouched. Replaying `comparison.yaml` creates another dated folder
and revalidates its sources. Re-rendering only needs the saved comparison JSON:
open a terminal in the comparison folder and run:

```bash
uv run quarto render report.qmd --to html
```

Future comparison styling belongs in
[comparison.qmd](../src/cx_sentiment/reports/comparison.qmd). Small smoke samples
do not establish a model ranking or statistical significance. Recorded costs
estimate the original requests, including cached predictions; they are not new
spend incurred by generating this report.

## When rendering fails

- **Quarto not found:** install its CLI and check PATH. Experiment execution checks
  for Quarto before calling providers. Paid-provider previews still work without it.
- **Python import or kernel error:** run `uv sync`, then `uv run quarto check jupyter`.
  Use `uv run quarto ...` for manual rendering so it sees the project environment.
- **Rendering error or timeout:** JSON/CSV results remain saved. Run the displayed
  `quarto render` command through `uv run` to see the error and retry without paying
  for predictions again. The application limits each render to two minutes.

## Further reading

- [Quarto's Python guide](https://quarto.org/docs/computations/python.html)
- [Great Tables with Polars](https://posit-dev.github.io/great-tables/blog/bring-your-own-df/)
- [Plotly Express dataframe support](https://plotly.com/python/px-arguments/)
- [HTML output and embedded resources](https://quarto.org/docs/output-formats/html-basics.html#self-contained)
- [Quarto for VS Code](https://quarto.org/docs/tools/vscode.html)
