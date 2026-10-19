output "repository" {
  value = github_repository.this.full_name
}

output "rulesets_enabled" {
  value = var.enable_rulesets
}
