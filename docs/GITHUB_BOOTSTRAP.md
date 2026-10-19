# GitHub workflow and safe bootstrap

Work on feature/fix branches, open PRs into `dev`, and squash ordinary feature
PRs. Promote a release with a PR from this repository's `dev` into `main`, using
a merge commit. Synchronize `main` back into `dev` with a merge PR after a release
or a bootstrap change. Keep both long-lived branches. `main` remains the default.
Dependabot targets `dev` with grouped weekly Python, Action, and Terraform updates.

The required checks are exactly **Quality**, **Security**, and **Branch policy**.
Quality aggregates tests on Python 3.11/3.12 (including real Quarto renders), lint,
types, whitespace, an installed-wheel smoke test, workflow validation, and
Terraform validation. Security aggregates full fetched Git-history Gitleaks,
`pip-audit` of frozen hashed runtime/development dependencies, and Bandit at
medium/high severity. Both aggregates fail on failed, cancelled, skipped, or
unexpected dependency results. Branch policy rejects release PRs whose head is
not this repository's `dev`, including a fork's `dev` branch.

All Actions use full SHA pins. CI uses read-only contents permission, no
persisted checkout credentials, timeouts, and cancellation of superseded runs.
PR code runs on `pull_request`, never `pull_request_target`; no provider keys,
paid calls, administration token, deployment, or Terraform apply belong in CI.
Only coverage/JUnit diagnostics are uploaded, retained for seven days. Raw
experiment outputs and secret findings are not artifacts.

## Observed remote state and limits

Read-only inspection on **2026-10-18** found `dosorio79/jev_test` **private**, with
`main` as default, only `main` present, and `main` unprotected. The authenticated
maintainer has administrator access. Merge commits and squash/rebase merges are
currently allowed; merged branches are retained. Dependency alerts are disabled.
The repository response has no `security_and_analysis` configuration; the secret-scanning API explicitly reported that scanning is disabled.

The rulesets API returned HTTP 403: “Upgrade to GitHub Pro or make this repository
public to enable this feature.” The legacy branch-protection endpoint returned the same 403. Existing rulesets therefore could not be listed.
This is a confirmed current feature-availability blocker, not evidence that no
rules exist. No visibility change, plan purchase, push, branch creation, security
setting update, or apply was performed. The Terraform defaults leave rulesets and
native secret protection off. Required-check enforcement remains pending.

[GitHub secret scanning](https://docs.github.com/en/code-security/concepts/secret-security/secret-scanning)
automatically scans public repositories for free; private availability depends
on repository/account licensing. Repository-level
[push protection](https://docs.github.com/en/code-security/concepts/secret-security/push-protection)
is a distinct setting and must be verified explicitly. This free-only Terraform
root permits native scanning plus push protection only when the repository is
already public, and never configures Advanced Security or changes visibility.
Gitleaks, pip-audit, Bandit, and Dependabot alerts need no security-product
purchase. Private Actions execution still consumes the account's available
Actions allowance; no runner or paid usage settings are changed here.

## Bootstrap in stages

Do not apply until the maintainer separately authorizes remote changes. Complete
all local validation and inspect a concrete plan first.

1. Land the workflow files on `main`, then wait for successful CI and Security
   runs. Review Actions availability/allowlists and repository budget separately.
2. Install Terraform 1.7 or later (CI pins 1.15.8). Run locally from a trusted
   checkout. Supply an administration token only through `GITHUB_TOKEN`; use an
   existing authenticated shell or password manager. Do not put a token in HCL,
   `.tfvars`, command-line `-var` values, logs, or Actions. The token needs the
   repository administration/contents access required to manage its settings,
   branches, and enabled security features. Validation itself needs no token.
3. Inspect the repository, branches, and existing rules read-only before changing
   managed scope:

   ```bash
   gh api repos/dosorio79/jev_test --jq '{visibility,default_branch,security_and_analysis}'
   gh api repos/dosorio79/jev_test/branches --jq 'map({name,protected})'
   gh api repos/dosorio79/jev_test/rulesets
   gh api repos/dosorio79/jev_test/branches/main/protection
   gh api repos/dosorio79/jev_test/vulnerability-alerts --silent
   ```

   A 403/404 is not automatically “disabled”: check its message and account
   permissions. Stop activation if protection availability is unresolved. Never
   create overlapping rulesets to work around unknown existing protection.
4. Initialize and validate the root without administration credentials in CI:

   ```bash
   terraform -chdir=infra/github fmt -check -diff
   terraform -chdir=infra/github init -backend=false -input=false -lockfile=readonly
   terraform -chdir=infra/github validate
   ```

   The official `integrations/github` provider is exactly pinned, with its
   generated `.terraform.lock.hcl` committed. Update its pin and lockfile together,
   inspect the changelog, and re-run validation. No provider upgrades occur in CI.
5. Inspect a local bootstrap plan with the default feature flags:

   ```bash
   umask 077
   terraform -chdir=infra/github plan -input=false -out=bootstrap.tfplan
   terraform -chdir=infra/github show bootstrap.tfplan
   ```

   Import blocks load the **existing repository** and existing dependency-alert
   setting before planning changes. A data lookup requires the repository to
   exist; repository destruction is blocked by `prevent_destroy`. Visibility and
   settings outside the documented scope are preserved. A default branch other
   than `main` stops planning for inspection.

   If `dev` already exists, import it before planning:

   ```bash
   terraform -chdir=infra/github import github_branch.dev jev_test:dev
   ```

   The inspected 2026-10-18 default plan contained **2 imports, 1 addition, 2
   in-place changes, 0 destroys**: create `dev` from `main`; disable rebase merge;
   suggest updating PR branches; enable dependency alerts. It preserved private
   visibility and branch retention. The legacy provider's computed/default
   `ignore_vulnerability_alerts_during_read=false` also appeared in the imported
   repository diff; no other settings change was proposed. Refresh and review
   your own plan rather than reusing these counts or an old saved plan.

   Reject replacement, visibility/default-branch changes, deletion, or unrelated
   settings drift. Applying the reviewed plan is a separate authorized step:

   ```bash
   terraform -chdir=infra/github apply bootstrap.tfplan
   ```

6. After the authorized bootstrap, push a normal feature PR into `dev` and obtain
   successful **Quality**, **Security**, and **Branch policy** runs. Inspect the
   actual check contexts and GitHub App source IDs, for example:

   ```bash
   gh api repos/dosorio79/jev_test/commits/dev/check-runs --jq '.check_runs[] | {name,conclusion,app_id:.app.id}'
   ```

   The rules expect GitHub Actions app ID `15368`; confirm this against the
   successful runs. Resolve any checks that are missing before enforcing them.
7. Once GitHub makes rulesets available and existing rules are fully inspected,
   enable the flag in an ignored local variables file or use
   `-var=enable_rulesets=true`. If an intended ruleset already exists, import its
   ID before applying (repeat for `main`/`dev` as applicable):

   ```bash
   terraform -chdir=infra/github import -var=enable_rulesets=true 'github_repository_ruleset.branches["main"]' jev_test:RULESET_ID
   terraform -chdir=infra/github plan -var=enable_rulesets=true -out=protection.tfplan
   terraform -chdir=infra/github show protection.tfplan
   ```

   Keep the flag enabled in every subsequent plan/apply. Turning it off proposes
   ruleset deletion, so reject such a plan during normal operation. Both branch
   rulesets require PRs, the three successful checks, up-to-date branches, resolved
   conversations, and zero second-person approvals. They block force-push and
   deletion, with no bypass actors. `main` permits only merge commits; `dev`
   permits squash for feature PRs and merge for `main` synchronization. The
   maintainer must choose squash for ordinary feature PRs into `dev`.
8. If the repository later becomes public under **separate authorization**,
   verify native feature availability, then plan with
   `-var=enable_native_secret_protection=true` and inspect scanning **and** push
   protection status. This root never buys a product or changes visibility.
9. After each authorized apply, re-read repository settings, dependency alerts,
   rulesets, and both branch-protection endpoints. Demonstrate with a disposable
   PR that a failing test/secret scan blocks merging, a non-`dev` release PR is
   blocked, and a normal solo PR can merge once checks pass. Do not call protection
   complete until GitHub reports it and these remote checks succeed.

## State, backups, and recovery

This root initially uses Terraform's local backend. `.terraform/`, state and
state backups, saved plans, `.tfvars`, override files, and credential files are
ignored. The provider lockfile is intentionally tracked. State and plans can
contain private repository details; use owner-only permissions and encrypted,
access-controlled backups outside the checkout. Do not upload them as CI artifacts.

Before and after each apply, back up the state with restricted permissions to
that secure destination. Keep the matching Git commit and provider lockfile with
the backup record. Restore the latest verified state backup before a refresh-only
plan, and inspect drift before an ordinary plan. If all state is lost, re-import
the repository, dependency alerts, existing `dev`, and each managed ruleset by
its remote ID; validate the resulting plan before any apply. Never delete a
repository or branch to repair missing local state. Do not commit, print, or
upload token-bearing Terraform CLI configuration.

## Local checks and scanner maintenance

Use the same pinned CI tool installer (Linux x86-64) and guard tests:

```bash
bash .github/scripts/install_ci_tool.sh actionlint /tmp/jev-ci-tools
/tmp/jev-ci-tools/actionlint
python -m unittest discover -s .github/scripts -p 'test_*.py'
bash .github/scripts/install_ci_tool.sh gitleaks /tmp/jev-ci-tools
/tmp/jev-ci-tools/gitleaks git --log-opts='--all' --redact=100 --no-banner .
uv export --frozen --no-emit-project --output-file /tmp/jev-audit-requirements.txt > /dev/null
uvx --from pip-audit==2.10.1 pip-audit --require-hashes --disable-pip -r /tmp/jev-audit-requirements.txt
uvx --from bandit==1.9.4 bandit -r src/cx_sentiment -ll
bash .github/scripts/smoke_package.sh
```

Gitleaks scans committed history rather than the working directory, so ignored
`.env` and experiment caches are not read. CI fetches complete history and scans
all fetched refs. Shallow local clones cover only their fetched history; fetch
full history before interpreting a local result as comparable to CI. Use
`--redact=100` for every secret scan. Never suppress all history with a baseline.
A false positive needs a narrow documented exception; a real credential needs
rotation as described in [SECURITY.md](../SECURITY.md). History rewriting needs
separate authorization.

Scanner binaries use fixed versions plus hardcoded SHA-256 digests verified
against official release checksums. Python scanner versions and tool versions
are pinned in the workflow. Review and update those pins deliberately; grouped
Dependabot Action/Terraform updates do not update embedded scanner CLI versions.
A passing scan only reflects the selected tools and current advisory database.

Official references: [GitHub provider](https://registry.terraform.io/providers/integrations/github/6.13.0/docs),
[rulesets](https://registry.terraform.io/providers/integrations/github/6.13.0/docs/resources/repository_ruleset),
[required check behavior](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets),
[Gitleaks](https://github.com/gitleaks/gitleaks), [pip-audit](https://github.com/pypa/pip-audit),
[Bandit](https://bandit.readthedocs.io/en/latest/), and
[uv lockfile export](https://docs.astral.sh/uv/concepts/projects/export/).
