/*
  Network: one VNet, three subnets, private endpoints for storage.

  Storage is reachable ONLY over private endpoints. That is not a preference -
  the subscription policy forces `publicNetworkAccess = Disabled`, so without
  these endpoints nothing can read or write a blob.

  Subnet sizing is not arbitrary:
    snet-func  /24  delegated to Microsoft.App/environments (Flex VNet integration)
    snet-pe    /24  private endpoints; network policies must be disabled
    snet-aca   /23  Container Apps requires at least /23 for a workload-profile env
*/

resource "azurerm_resource_group" "main" {
  name     = var.resource_group_name
  location = var.location
  tags     = var.tags
}

resource "random_string" "suffix" {
  length  = 8
  lower   = true
  upper   = false
  numeric = true
  special = false
}

locals {
  suffix = random_string.suffix.result
  # Storage account names: 3-24 chars, lowercase alphanumeric only.
  storage_name         = substr("st${var.name_prefix}${local.suffix}", 0, 24)
  api_storage_name     = substr("st${var.name_prefix}api${local.suffix}", 0, 24)
  api_function_name    = "func-${var.name_prefix}-${local.suffix}"
  worker_function_name = "func-${var.name_prefix}-worker-${local.suffix}"
  aca_env_name         = "cae-${var.name_prefix}-${local.suffix}"
}

resource "azurerm_virtual_network" "main" {
  name                = "vnet-${var.name_prefix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  address_space       = ["10.20.0.0/16"]
  tags                = var.tags
}

resource "azurerm_subnet" "func" {
  name                            = "snet-func"
  resource_group_name             = azurerm_resource_group.main.name
  virtual_network_name            = azurerm_virtual_network.main.name
  address_prefixes                = ["10.20.1.0/24"]
  default_outbound_access_enabled = false

  # Flex Consumption VNet integration requires this exact delegation.
  delegation {
    name = "flex"
    service_delegation {
      name    = "Microsoft.App/environments"
      actions = ["Microsoft.Network/virtualNetworks/subnets/join/action"]
    }
  }
}

resource "azurerm_subnet" "pe" {
  name                            = "snet-pe"
  resource_group_name             = azurerm_resource_group.main.name
  virtual_network_name            = azurerm_virtual_network.main.name
  address_prefixes                = ["10.20.2.0/24"]
  default_outbound_access_enabled = false

  # Private endpoint NICs cannot live in a subnet that enforces network policies.
  private_endpoint_network_policies = "Disabled"
}

resource "azurerm_subnet" "aca" {
  name                            = "snet-aca"
  resource_group_name             = azurerm_resource_group.main.name
  virtual_network_name            = azurerm_virtual_network.main.name
  address_prefixes                = ["10.20.4.0/23"] # /23 minimum for Container Apps
  default_outbound_access_enabled = false

  # A workload-profiles environment requires the same delegation the Flex
  # subnet uses. Consumption-only environments do not, which is why this is
  # easy to miss: it fails at create time with
  # ManagedEnvironmentSubnetDelegationError, not at plan time.
  delegation {
    name = "aca"
    service_delegation {
      name    = "Microsoft.App/environments"
      actions = ["Microsoft.Network/virtualNetworks/subnets/join/action"]
    }
  }
}

# --- private DNS: blob, queue, table ----------------------------------------
# Durable Functions uses ALL THREE (blobs for leases and history payloads,
# queues for work items, tables for instance state). Missing one produces a
# runtime that starts and then silently fails to schedule anything.

locals {
  storage_services = ["blob", "queue", "table"]
}

resource "azurerm_private_dns_zone" "storage" {
  for_each            = toset(local.storage_services)
  name                = "privatelink.${each.key}.core.windows.net"
  resource_group_name = azurerm_resource_group.main.name
  tags                = var.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "storage" {
  for_each             = toset(local.storage_services)
  name                 = "link-${each.key}"
  private_dns_zone_id  = azurerm_private_dns_zone.storage[each.key].id
  virtual_network_id   = azurerm_virtual_network.main.id
  registration_enabled = false
  tags                 = var.tags
}

resource "azurerm_private_endpoint" "storage" {
  for_each            = toset(local.storage_services)
  name                = "pe-${local.storage_name}-${each.key}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  subnet_id           = azurerm_subnet.pe.id
  tags                = var.tags

  private_service_connection {
    name                           = "psc-${each.key}"
    private_connection_resource_id = azurerm_storage_account.main.id
    subresource_names              = [each.key]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "default"
    private_dns_zone_ids = [azurerm_private_dns_zone.storage[each.key].id]
  }
}

# The API Function App has its own host/deployment storage. It still reaches the
# worker storage account above as a Durable client and to write uploaded PDFs.
resource "azurerm_private_endpoint" "api_storage" {
  for_each            = toset(local.storage_services)
  name                = "pe-${local.api_storage_name}-${each.key}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  subnet_id           = azurerm_subnet.pe.id
  tags                = var.tags

  private_service_connection {
    name                           = "psc-api-${each.key}"
    private_connection_resource_id = azurerm_storage_account.api.id
    subresource_names              = [each.key]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "default"
    private_dns_zone_ids = [azurerm_private_dns_zone.storage[each.key].id]
  }
}
