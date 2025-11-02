// infra/terraform/variables.tf
variable "project_name" {
  description = "Base name/prefix for all resources"
  type        = string
  default     = "telemetryguard"
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "eastus"
}

variable "k8s_node_count" {
  description = "Default AKS node count"
  type        = number
  default     = 2
}

variable "k8s_vm_size" {
  description = "VM size for AKS agent nodes"
  type        = string
  default     = "Standard_DS2_v2"
}

variable "log_analytics_retention_days" {
  description = "Retention for Log Analytics workspace"
  type        = number
  default     = 30
}
