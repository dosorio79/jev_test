variable "owner" {
  type        = string
  description = "Existing repository owner."
  default     = "dosorio79"
}

variable "repository" {
  type        = string
  description = "Existing repository name; import is mandatory."
  default     = "jev_test"
}

variable "enable_rulesets" {
  type        = bool
  description = "Enable only after checking plan availability, existing rules, and successful named checks."
  default     = false
}

variable "enable_native_secret_protection" {
  type        = bool
  description = "Enable native scanning AND push protection only after verifying free feature availability. Never enables a paid product."
  default     = false
}
