# The existing repository must be read and imported before any settings change.
# This root must never create a repository or change its visibility.
data "github_repository" "existing" {
  full_name = "${var.owner}/${var.repository}"
}

import {
  to = github_repository.this
  id = var.repository
}

resource "github_repository" "this" {
  name                   = var.repository
  visibility             = data.github_repository.existing.visibility
  allow_merge_commit     = true
  allow_squash_merge     = true
  allow_rebase_merge     = false
  allow_update_branch    = true
  delete_branch_on_merge = false

  dynamic "security_and_analysis" {
    for_each = var.enable_native_secret_protection ? [true] : []
    content {
      secret_scanning {
        status = "enabled"
      }
      secret_scanning_push_protection {
        status = "enabled"
      }
    }
  }

  lifecycle {
    prevent_destroy = true
    precondition {
      condition     = data.github_repository.existing.default_branch == "main"
      error_message = "Inspect default-branch drift before planning; this root requires main."
    }
    precondition {
      condition     = !var.enable_native_secret_protection || data.github_repository.existing.visibility == "public"
      error_message = "This free-only bootstrap enables native secret protection only on public repositories; visibility is never changed here."
    }
    # Preserve settings outside this root's intentionally small scope.
    ignore_changes = [
      visibility, description, homepage_url, topics, archived, is_template,
      has_issues, has_projects, has_wiki, has_discussions, has_downloads,
      allow_auto_merge, allow_forking, web_commit_signoff_required,
      merge_commit_title, merge_commit_message,
      squash_merge_commit_title, squash_merge_commit_message,
    ]
  }
}

resource "github_branch" "dev" {
  repository    = github_repository.this.name
  branch        = "dev"
  source_branch = "main"

  lifecycle {
    prevent_destroy = true
  }
}

import {
  to = github_repository_vulnerability_alerts.this
  id = var.repository
}

resource "github_repository_vulnerability_alerts" "this" {
  repository = github_repository.this.name
  enabled    = true
}

resource "github_repository_ruleset" "branches" {
  for_each = var.enable_rulesets ? toset(["main", "dev"]) : toset([])

  name        = "${each.key}-pull-requests"
  repository  = github_repository.this.name
  target      = "branch"
  enforcement = "active"

  conditions {
    ref_name {
      include = ["refs/heads/${each.key}"]
      exclude = []
    }
  }

  # No bypass actors: the same requirements apply to the solo administrator.
  rules {
    deletion         = true
    non_fast_forward = true

    pull_request {
      # Preserve merge commits for main -> dev synchronization.
      allowed_merge_methods             = each.key == "main" ? ["merge"] : ["squash", "merge"]
      required_approving_review_count   = 0
      required_review_thread_resolution = true
    }

    required_status_checks {
      strict_required_status_checks_policy = true
      do_not_enforce_on_create             = false
      dynamic "required_check" {
        for_each = toset(["Quality", "Security", "Branch policy"])
        content {
          context        = required_check.value
          integration_id = 15368 # GitHub Actions; confirm with successful check-run app IDs.
        }
      }
    }
  }

  depends_on = [github_branch.dev]
}
