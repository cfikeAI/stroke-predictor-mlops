#provides values needed for 
#pushing Docker images to ACR
#Connecting kubectl to AKS
#Configuring MLFlow and DVC remotes

output "resource_group" {
    value = azurerm_resource_group.rg.name
    description = "The name of the Azure Resource Group"
}

output "location" {
    value = azurerm_resource_group.rg.location
    description = "The location of the Azure Resource Group"
}

output "acr_name" {
    value = azurerm_container_registry.acr.name
    description = "The name of the Azure Container Registry"
}

output "acr_login_server" {
    value = azurerm_container_registry.acr.login_server
    description = "The login server URL of the Azure Container Registry"
}

output "storage_account_name" {
    value = azurerm_storage_account.sa.name
    description = "The name of the Storage Account"
}

output "storage_account_dvc_container" {
    value = azurerm_storage_container.dvc.name
    description = "The name of the DVC artifacts container"
}   

output "storage_account_mlflow_container" {
    value = azurerm_storage_container.mlflow.name
    description = "The name of the MLFlow artifacts container"
}

output "aks_name" {
    value = azurerm_kubernetes_cluster.aks.name
    description = "The name of the AKS cluster"
}

output "aks_kubelet_identity" {
    value = azurerm_kubernetes_cluster.aks.kubelet_identity_object_id
    description = "The Object ID of the AKS cluster's kubelet identity"
}

output "log_analytics_workspace_id" {
    value = azurerm_log_analytics_workspace.law.workspace_id
    description = "The Log Analytics Workspace ID"
}

output "key_vault_name" {
    value = azurerm_key_vault.kv.name
    description = "The name of the Key Vault"
}