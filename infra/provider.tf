// infra/terraform/provider.tf
terraform {
  required_version = ">= 1.7.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }

  backend "local" {
    path = "terraform.tfstate"
  }
}

provider "azurerm" {
  features {}
  subscription_id = "59cd9d27-cc15-49a5-90ea-af107e69286b"
  tenant_id       = "4a65bba3-11fa-4c82-9209-8e276396f49a"
}
