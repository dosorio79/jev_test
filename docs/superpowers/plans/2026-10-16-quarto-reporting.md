# Quarto reporting implementation plan

**Goal:** Replace hand-built HTML/SVG with readable Quarto reports, following the
user's approval of Quarto and interest in learning it.

**Design:** Python retains prediction, metric, and selection calculations. Packaged
`.qmd` documents read saved JSON, use Polars/Great Tables tables and Plotly plots, and render
through the external Quarto CLI to standalone HTML. Keep existing commands and HTML
filenames. Copy report sources beside their data so users can edit and re-render
without provider calls. Credentials are never needed by a report.

**Dependencies:** Quarto CLI is separately installed. Add ipykernel and nbclient for
Quarto's Python engine, Great Tables for Polars tables, and Plotly for plots. These replace
custom table/SVG construction with ordinary library usage. Do not add a dashboard,
web service, custom template engine, PDF tooling, or extra CLI options.

- [x] Add regression tests for saved Quarto source/data, safe rendering of text,
  missing costs, selection frontier and empty selection, and renderer failures.
- [x] Replace rendering in reporting.py with two packaged Quarto documents and a
  small subprocess wrapper. Keep JSON/CSV contracts; check Quarto before paid work.
- [x] Install a local Quarto binary for verification, render both report types,
  inspect their output, and verify packaged templates are included in a wheel.
- [x] Document installation, the ordinary run, `.qmd` basics, and re-render/preview
  commands. Update AGENTS.md's reporting guidance.
- [x] Run pytest, Ruff, Pyright and diff checks. Commit coherent verified slices
  and record their hashes here.

**Acceptance:** Existing YAML commands produce report.html; missing Quarto fails
with an actionable error before provider calls; results remain saved if rendering
fails; documents work from paths with spaces and retain all previous report tables,
cost honesty and metadata. HTML escapes dynamic strings. Reports run offline.

## Progress

Checkout began clean at 8076db1. Quarto was absent from PATH.

User steering: use the existing Polars stack. Replaced the initial pandas table
implementation with Great Tables operating directly on Polars frames. Pandas is
absent from the environment and lockfile. Great Tables accepts raw HTML in string
cells, so the documents explicitly escape external text through its formatter.
The existing HTML-escaping regression covers this boundary.

User steering: use Plotly for interactive charts. Replaced the initial Matplotlib
plots with Plotly Express, passing Polars directly for the frontier. Bundle the
plotting JavaScript in each standalone HTML report; verify charts in a browser.


## Verification and handoff

- Quarto 1.10.19 was installed at ~/.local/share/quarto with a launcher in
  ~/.local/bin after verifying the official release SHA256 checksum.
- The final suite passed: 51 tests, including real benchmark and selection renders
  (selected policy, no qualifying policy, and empty frontier), text escaping,
  renderer failures/timeouts, and missing-tool preflight before predictions.
- Ruff and Pyright passed; git diff --check passed. The Python cells in both
  documents compiled. The built wheel contains the exact final .qmd sources.
- The 600-record mock demo is in outputs/quarto-demo/report.html with its editable
  report.qmd and saved results. This is pipeline evidence, not a model benchmark.
- Browser inspection verified six Great Tables and an interactive Plotly heatmap
  with no remote resources. The selection fixture's chart displayed threshold and
  fallback details on hover; both reports fit the viewport without page overflow.
- Integration runs needed sandbox escalation for Quarto's user-local runtime and
  Jupyter transport. No paid provider calls or pushes were made.
- Implementation commit: aec46d2 (Quarto reports, Polars tables, Plotly charts,
  dependencies, CLI error handling, and tests). Documentation and this checklist
  are committed as the following documentation slice.
