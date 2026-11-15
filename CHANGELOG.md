# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-11-15

### Added

- YAML-driven routing performance studies with offline threshold sweeps,
  risk-coverage analysis, language slices, and Pareto comparisons.
- Controlled benchmark instrumentation with attempt- and operation-level wall
  times, explicit cold/warm conditions, and cache-status labeling.
- Validation-selected frozen routing policy: Jev 1.13.0 first, GPT-6 Luna
  fallback below maximum class probability 0.72.
- One-time held-out evaluation and a shareable public report covering quality,
  latency, cost, and per-language results.
- Auditable run provenance, provider request accounting, and estimated-cost
  tracking with dated rate cards.
- Hybrid and local-baseline comparison reporting, including P50 and P95 latency
  views and model-labeled plots.

### Changed

- Paired routing bootstrap is faster while preserving the documented sampling
  behavior.
- Benchmark timing excludes runner persistence and report rendering.
- CI documentation now separates quality gates from security scans.

### Fixed

- Updated `tornado` from 6.5.10 to 6.6.1, addressing four known vulnerabilities
  (CVE-2026-116577, CVE-2026-115927, CVE-2026-116578, CVE-2026-126347).
- Comparison plots and reports label model identities consistently.
- Incomplete and legacy runs remain explicitly labeled instead of being treated
  as complete.

### Notes

- The research corpus remains excluded from Git because redistribution rights
  are unresolved. See [docs/DATA_LICENSE.md](docs/DATA_LICENSE.md).
- `v0.1.0` was cut from the earlier `dev` line and is not an ancestor of `main`.
  `v0.2.0` is the first release containing the performance-study workflow.
- The frozen 0.72 policy should be evaluated on privacy-reviewed, human-labeled
  real customer feedback without retuning before any production claim is made.

[0.2.0]: https://github.com/dosorio79/jev_test/releases/tag/v0.2.0
