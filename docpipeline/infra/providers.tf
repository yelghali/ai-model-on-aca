terraform {
  required_version = ">= 1.6.0"

  required_providers {
    azurerm = {
      source = "hashicorp/azurerm"
      # Pinned to the 5.x schema. Notably 5.0 moved private DNS links and
      # records onto `private_dns_zone_id`, so a 4.x provider will not plan
      # this configuration.
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = ">= 3.6.0"
    }
  }
}

provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }

  # REQUIRED for this subscription. Policy enforces
  # `shared_access_key_enabled = false` on storage, so the provider cannot use
  # account keys for data-plane calls (creating containers, for example). This
  # makes it authenticate as the signed-in principal / CI identity instead.
  storage_use_azuread = true
}

provider "random" {}

data "azurerm_client_config" "current" {}
