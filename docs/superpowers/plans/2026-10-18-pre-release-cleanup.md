# Pre-release cleanup, BYOK usability, and repository automation

**Recorded:** 2026-10-18.
**Status:** Implementation in progress on `feat/pre-release-cleanup`.
**Authority:** User authorized implementation with Superpowers multiagent on 2026-10-18. Remote pushes and bootstrap activation require separate authorization.

## 1. Release goal and boundaries

Release a GitHub toolkit for technical experimenters using their own OpenAI or
Jev keys. Preserve YAML experiments, the existing sentiment contract, sequential
execution, and Quarto reports.

Keep the implementation small: targeted refactoring, existing module boundaries,
no hosted service, database, generic provider framework, or package-index release.

Add an MIT license for project code; document dataset attribution and verify
redistribution rights separately.

## 2. Auditable experiments

- Extend run metadata with a format version, code/package version, requested and
  returned model identities, effective non-secret settings, dataset fingerprint,
  and dated pricing provenance.
- Preserve environment-based credentials/model selection. Document how to
  reproduce the recorded settings; YAML alone does not restore the environment.
- Add a per-record execution ledger distinguishing cache hits, fresh attempts,
  failures, and records skipped when execution stops.
- Separate this invocation's request count, known estimated spend, unknown-cost
  attempts, and budget reservations from historical prediction costs.
- Save completed results incrementally. Mark interrupted or budget-stopped runs
  incomplete and reject them from normal comparisons with a readable explanation.
- Disable automatic retries in standard runs so request accounting matches actual
  attempts. Keep recollection outside this release.
- Preserve measured failure latency and independently valid usage, credits,
  response references, and costs. Never invent missing historical measurements.
- Keep strict probability validation and existing evaluation denominators.
  Version changed cache semantics and include effective endpoint identity;
  preserve old artifacts.
- Refactor configuration resolution, provider construction, execution, and
  reporting into small ordinary functions where needed.
- Provide concise, sanitized errors for expected configuration, credential,
  pricing, dataset, and budget failures.

## 3. Branch workflow and Terraform

**Workflow:** feature/fix branches -> PR into `dev` -> release PR from `dev` into
`main`. Keep `main` as the public default branch.

- Protect both `dev` and `main`: require PRs, passing checks, resolved
  conversations, and an up-to-date branch; block force pushes and deletion.
- Require no second-person approval, preserving solo maintenance. Configure no
  routine bypass.
- Require release PRs into `main` to originate from this repository's `dev`,
  enforced by a required workflow check.
- Squash feature PRs into `dev`; use merge commits for release promotion and
  subsequent `main` -> `dev` synchronization.
- Preserve long-lived branches after merges; target Dependabot PRs at `dev`.

Create a small Terraform root under `infra/github/` using the official
[`integrations/github` provider](https://registry.terraform.io/providers/integrations/github/latest/docs/resources/repository_ruleset):

- Import the existing repository and existing managed settings before applying;
  never recreate the repository or change its visibility.
- Manage repository merge settings, the `dev` branch, branch rulesets, and
  supported native security settings.
- Pin provider constraints and commit the Terraform dependency lockfile.
- Protect the repository resource against destruction.
- Use local state initially, excluded from Git, with documented secure backup and
  recovery. Exclude plans, state backups, and credential files.
- Supply the administration token through the environment. Never place it in
  Terraform configuration, committed variables, or ordinary CI.
- CI runs formatting and validation without GitHub administration credentials.
  Run plans/applies locally against inspected changes.

Bootstrap in order: land workflows, create `dev`, obtain successful named checks,
then activate required-check rules. Inspect existing remote rules before creating
overlapping protection.

## 4. CI, free security checks, and badges

### CI

Run checks on branch pushes and PRs targeting `dev` or `main`; cancel superseded
runs. GitHub tests each pushed branch tip, while developers run focused checks
before local commits.

- Install from `uv.lock` using frozen dependency resolution.
- Test Python 3.11 and 3.12.
- Run pytest with coverage artifacts, Ruff, Pyright, and whitespace checks.
- Render actual Quarto reports with mock data.
- Build the package and smoke-test its entry point and bundled report templates.
- Validate Terraform and workflow syntax.
- Use stable required checks named `Quality`, `Security`, and `Branch policy`;
  aggregation must fail when a dependency fails or is unexpectedly skipped.
- Pin Actions to full commit SHAs, minimize token permissions, set timeouts, and
  avoid privileged execution of untrusted PR code.
- No provider keys or paid calls in CI.

### Security and secrets

GitHub provides automatic free secret scanning for public repositories. Push
protection is distinct, so verify its repository-level configuration rather than
assuming it is enabled. See
[GitHub secret scanning](https://docs.github.com/en/code-security/concepts/secret-security/secret-scanning)
and [push protection](https://docs.github.com/en/code-security/concepts/secret-security/push-protection).
Recheck availability and settings during implementation.

Use three free tools:

| Check | Implementation |
| --- | --- |
| Secrets | Pinned Gitleaks CLI, full fetched Git history, redacted output; required on pushes and PRs. |
| Dependencies | `pip-audit` against dependencies exported from the lockfile; fail on known vulnerabilities. |
| Python security | Bandit against application code; fail on medium/high-severity findings. |

Tool references: [Gitleaks](https://github.com/gitleaks/gitleaks),
[pip-audit](https://github.com/pypa/pip-audit), and
[Bandit](https://bandit.readthedocs.io/en/latest/).

- Run security checks weekly and on demand as well.
- Enable available native secret scanning, push protection, dependency alerts,
  and grouped weekly dependency-update PRs without purchasing security products.
- Keep scanner tooling out of production dependencies.
- Allow only narrow, documented false-positive exceptions; no blanket baselines
  hiding findings.
- Scan tracked history without reading the developer's ignored `.env`. Never
  upload unredacted findings or raw experiment caches.
- Add `SECURITY.md` explaining reporting and credential rotation. Actual leaked
  credentials require rotation; history rewriting remains separately authorized.

### Badges

Add README badges for main-branch CI, security workflow status, Python 3.11/3.12,
and MIT licensing. Link each to its workflow or supporting file. Avoid claims
that a passing scan guarantees security; omit hosted coverage services initially.

## 5. Verification and delivery

Deliver coherent, verified commits in this order:

1. Provider boundaries and request/cost accounting.
2. Run provenance, incremental persistence, and reports.
3. BYOK quickstart, examples, licensing, and contributor guidance.
4. CI, security scans, and badges.
5. Terraform, branch workflow, and bootstrap documentation.

Acceptance checks:

- Offline/preview paths make no provider calls; cached results incur no new API
  spend.
- Failures preserve available measurements; budget stops preserve completed work.
- Incompatible model/settings/endpoints cannot share cached results.
- Legacy runs remain readable; incomplete runs are clearly identified.
- Mock reports render and locked dataset hashes remain unchanged.
- A fresh checkout follows the documented workflow successfully.
- Deliberately failing tests and synthetic secret fixtures fail the intended CI
  gates without exposing credentials.
- Terraform validates; the inspected plan contains no repository replacement or
  unrelated settings changes.
- After applying, GitHub reports the intended rules and security settings, and a
  failing PR cannot merge.

Remote GitHub settings could not be verified because API access failed during
planning. Confirm visibility, permissions, and existing protection during
bootstrap; do not change visibility or purchase a plan automatically. No product,
CI, or infrastructure changes or security scans were executed during planning.
