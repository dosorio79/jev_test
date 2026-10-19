# Security policy

This is an experimental local toolkit. Use the current `main` revision when
reporting an issue; there is no supported historical-release maintenance line.

## Report privately

Use [GitHub private vulnerability reporting](https://github.com/dosorio79/jev_test/security/advisories/new)
if available. Do not put credentials, customer text, raw provider responses, or
unredacted scanner findings in public issues. If private reporting is unavailable,
open an issue asking the maintainer to establish a private reporting channel,
without including vulnerability details.

Include the affected revision, a minimal reproduction with synthetic inputs,
the impact, and sanitized environment/tool versions. Do not run paid requests
to demonstrate an issue unless independently authorized.

## Credentials and experiment artifacts

Keep provider keys in environment variables or a local secret manager. `.env` is
ignored and is not loaded by the application; explicitly sourcing it executes
shell code. Never commit credentials, Terraform state, local caches, or raw
experiment artifacts. Review reports before sharing: metadata, paths, response
references, sample text, and comparison disagreements may identify private data.

If a credential is exposed, revoke or rotate it with the issuer immediately,
review usage and billing, remove it from active configuration, and update any
consumers. Deleting the file or rewriting Git history does not revoke a key.
History rewriting requires separate coordination and authorization.

The security workflow scans Git history for secrets with redacted output,
audits locked Python dependencies, and scans application code. CI uses no
provider credentials or paid calls. Native GitHub secret scanning and push
protection are separate repository settings; confirm their actual activation
when bootstrapping the repository. A green workflow is evidence of those checks,
not a guarantee that code or artifacts contain no vulnerabilities or secrets.
