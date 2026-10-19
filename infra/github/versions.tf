terraform {
  required_version = ">= 1.7, < 2.0"

  required_providers {
    github = {
      source  = "integrations/github"
      version = "= 6.13.0"
    }
  }
}

# Administration credentials are read only from GITHUB_TOKEN by the provider.
provider "github" {
  owner = var.owner
}
