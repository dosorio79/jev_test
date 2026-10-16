# Learning Quarto with this benchmark

Run an experiment to get a report you can open, edit, and re-render locally:

```bash
uv run cx-sentiment run experiment.yaml
```

The supplied mock experiment makes no provider calls. Open
`outputs/experiment/report.html` in your browser. Its styles and plots are embedded,
so this HTML file can be shared on its own and viewed offline.

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

Each experiment saves these files:

| File | Purpose |
| --- | --- |
| `predictions.jsonl` | Provider predictions, one record per line |
| `metrics.json` | Full calculated evaluation results |
| `metrics.csv` | Overall and per-language summary |
| `report_metadata.json` | Run title and supplied metadata |
| `report.qmd` | Editable report document |
| `report.html` | Rendered report |

Change a heading or the chart colors in `outputs/experiment/report.qmd`, then run:

```bash
uv run quarto render outputs/experiment/report.qmd --to html
```

For an updating preview while you edit:

```bash
uv run quarto preview outputs/experiment/report.qmd
```

These commands execute only the report's local Python blocks. The supplied
documents read saved JSON and make no provider calls. Re-rendering also works
without API credentials. Quarto may create intermediate notebook and `_files`
artifacts; keep those under the ignored `outputs/` folder.

For changes that should apply to future experiments, edit the packaged sources:

- [benchmark.qmd](../src/cx_sentiment/reports/benchmark.qmd) contains performance,
  language/class tables, confusion plot, latency, usage, costs, calibration, and metadata.
- [selection.qmd](../src/cx_sentiment/reports/selection.qmd) contains the selected
  routing policy, a cost/quality plot, and the complete threshold table.

Running the benchmark or regenerating a report through `cx-sentiment report`
copies the packaged document again, replacing local `.qmd` edits. Copy your edited
document to another name if you want to keep it separately.

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
