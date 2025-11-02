
# Generates a random 4-character suffix for globally unique names
resource "random_string" "rand_suffix" {
  length  = 4
  upper   = false
  special = false
}


#resource group - logic boundary for this project's resources

resource "azurerm_resource_group" "rg" {
    name = "${var.project_name}-rg"
    location = var.location
}

#Log Analytics workspace
#AKS and Prometheus/Grafana/insights feed (latency, uptime, drift metrics, etc)

resource "azurerm_log_analytics_workspace" "law" {
    name = "${var.project_name}-log${random_string.rand_suffix.result}"
    location = azurerm_resource_group.rg.location
    resource_group_name = azurerm_resource_group.rg.name
    sku = "PerGB2018"
    retention_in_days = var.log_analytics_retention_days
}

#Container Registry (ACR) for Docker image
#Where GitHub Actions pushes newly built images

resource "azurerm_container_registry" "acr" {
    name = "${var.project_name}-acr${random_string.rand_suffix.result}"
    resource_group_name = azurerm_resource_group.rg.name
    location = azurerm_resource_group.rg.location
    sku = "Basic"
    admin_enabled = false

}

#Storage account - for DVC remote and MLFlow artifacts

resource "azurerm_storage_account" "sa" {
    name = "${var.project_name}store${random_string.rand_suffix.result}"
    resource_group_name = azurerm_resource_group.rg.name
    location = azurerm_resource_group.rg.location
    account_tier = "Standard"
    account_replication_type = "LRS"
}

resource "azurerm_storage_container" "mlflow" {
    name = "mlflow-artifacts"
    storage_account_name = azurerm_storage_account.sa.name
    container_access_type = "private"
}

resource "azurerm_storage_container" "dvc" {
    name = "dvc-artifacts"
    storage_account_name = azurerm_storage_account.sa.name
    container_access_type = "private"
}

#Key Vault for secrets (DB strings, API keys, rollout thresholds, etc)
#for secrets like MLFLOW_TRACKING_URI, canary thresholds, rollback SLOs, etc.

resource "azurerm_key_vault" "kv" {
    name = "${var.project_name}-kv${random_string.rand_suffix.result}"
    location = azurerm_resource_group.rg.location
    resource_group_name = azurerm_resource_group.rg.name
    sku_name = "standard"
    tenant_id = data.azurerm_client_config.current.tenant_id
    purge_protection_enabled = false
    soft_delete_retention_days = 7

    access_policy {
        tenant_id = data.azurerm_client_config.current.tenant_id
        object_id = data.azurerm_client_config.current.object_id

        secret_permissions = ["Get", "List", "Set", "Delete"]
    }
}

data "azurerm_client_config" "current" {}

#AKS Cluster - will run containerized model APIs, batch jobs, and monitoring tools

resource "azurerm_kubernetes_cluster" "aks" {
    name = "${var.project_name}-aks"
    location = azurerm_resource_group.rg.location
    resource_group_name = azurerm_resource_group.rg.name
    dns_prefix = "${var.project_name}-dns"

    #system-assigned managed identity
    #AKS authenticates with ACR, Key Vault, etc using this identity
    identity {
        type = "SystemAssigned"
    }
    #default node pool - defines compute for pods (model servers, training jobs, etc)
    #compute substrate for serving models
    default_node_pool {
        name = "nodepool"
        node_count = var.k8s_node_count
        vm_size = var.k8s_vm_size
        os_disk_size_gb = 30
        orchestrator_version = var.k8s_version
    }

    #Enable metrics/insights: connects AKS to Log Analytics workspace
    #Ties into Log Analytics defined earlier, can view AKS metrics in Azure Monitor or Grafana
    oms_agent {
        log_analytics_workspace_id = azurerm_log_analytics_workspace.law.id
    }

    #lock down public API later
    #enforces RBAC and disables local accounts (no unauthenticated kubectl access)
    role_based_access_control_enabled =  true
    local_account_disabled = true

    lifecycle {
      ignore_changes = [ default_node_pool.0.node_count ]
    }


}

#Grants AKS access to ACR to pull images - pods pull the container images
resource "azurerm_role_assignment" "aks_to_acr_pull" {
    principal_id = azurerm_kubernetes_cluster.aks.kubelet_identity[0].object_id
    role_definition_name = "AcrPull"
    scope = azurerm_container_registry.acr.id
}