# =====================================================================
# QuantNucleo — Terraform: Infraestrutura Azure
# =====================================================================
# Provisiona todos os recursos Azure necessários:
#  - AKS (Kubernetes)
#  - Blob Storage (Data Lake)
#  - Key Vault (Segredos)
#  - Synapse Analytics (Processamento)
#  - Application Insights (Monitoramento)

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.90.0"
    }
  }

  backend "azurerm" {
    resource_group_name  = "rg-quantnucleo-tfstate"
    storage_account_name = "stquantnucleotf"
    container_name       = "tfstate"
    key                  = "quantnucleo.tfstate"
  }
}

provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy = true
    }
  }
}

# =====================================================================
# Variables
# =====================================================================

variable "location" {
  default = "eastus2"
}

variable "project" {
  default = "quantnucleo"
}

variable "environment" {
  default = "prod"
}

# =====================================================================
# Resource Group
# =====================================================================

resource "azurerm_resource_group" "main" {
  name     = "rg-${var.project}"
  location = var.location

  tags = {
    project     = var.project
    environment = var.environment
    managed_by  = "terraform"
  }
}

# =====================================================================
# Azure Kubernetes Service (AKS)
# =====================================================================

resource "azurerm_kubernetes_cluster" "aks" {
  name                = "aks-${var.project}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = var.project

  default_node_pool {
    name       = "default"
    node_count = 2
    vm_size    = "Standard_B2s"

    # Auto-scaling
    enable_auto_scaling = true
    min_count           = 1
    max_count           = 5
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin = "azure"
    network_policy = "calico"
  }

  tags = azurerm_resource_group.main.tags
}

# =====================================================================
# Storage Account (Data Lake)
# =====================================================================

resource "azurerm_storage_account" "datalake" {
  name                     = "st${var.project}"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  is_hns_enabled           = true  # Data Lake Gen2

  tags = azurerm_resource_group.main.tags
}

resource "azurerm_storage_container" "raw" {
  name                  = "raw"
  storage_account_name  = azurerm_storage_account.datalake.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "processed" {
  name                  = "processed"
  storage_account_name  = azurerm_storage_account.datalake.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "features" {
  name                  = "features"
  storage_account_name  = azurerm_storage_account.datalake.name
  container_access_type = "private"
}

# =====================================================================
# Key Vault (Segredos)
# =====================================================================

data "azurerm_client_config" "current" {}

resource "azurerm_key_vault" "main" {
  name                = "kv-${var.project}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"

  purge_protection_enabled   = true
  soft_delete_retention_days = 90

  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id

    secret_permissions = [
      "Get", "List", "Set", "Delete", "Purge"
    ]
  }

  # AKS managed identity access
  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = azurerm_kubernetes_cluster.aks.kubelet_identity[0].object_id

    secret_permissions = ["Get", "List"]
  }

  tags = azurerm_resource_group.main.tags
}

# =====================================================================
# Application Insights
# =====================================================================

resource "azurerm_log_analytics_workspace" "main" {
  name                = "log-${var.project}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 90

  tags = azurerm_resource_group.main.tags
}

resource "azurerm_application_insights" "main" {
  name                = "ai-${var.project}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  workspace_id        = azurerm_log_analytics_workspace.main.id
  application_type    = "web"

  tags = azurerm_resource_group.main.tags
}

# =====================================================================
# Outputs
# =====================================================================

output "aks_cluster_name" {
  value = azurerm_kubernetes_cluster.aks.name
}

output "storage_account_name" {
  value = azurerm_storage_account.datalake.name
}

output "keyvault_uri" {
  value = azurerm_key_vault.main.vault_uri
}

output "app_insights_key" {
  value     = azurerm_application_insights.main.instrumentation_key
  sensitive = true
}
