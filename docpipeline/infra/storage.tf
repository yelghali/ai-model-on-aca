/*
  Storage: private-only, identity-only.

  These settings match what tenant policy enforces anyway - declaring them
  explicitly means Terraform's plan reflects reality instead of drifting on
  every apply:

    public_network_access_enabled = false   no data-plane access from internet
    shared_access_key_enabled     = false   connection strings CANNOT work
    default_action                = Deny    private endpoints only

  Consequence for the app: `BlobServiceClient.from_connection_string(...)` is
  impossible. shared/storage.py uses DefaultAzureCredential, and the Functions
  host uses the identity-based `AzureWebJobsStorage__*` settings.
*/

resource "azurerm_storage_account" "main" {
  name                = local.storage_name
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  account_tier             = "Standard"
  account_replication_type = "LRS"
  account_kind             = "StorageV2"
  min_tls_version          = "TLS1_2"

  https_traffic_only_enabled      = true
  public_network_access_enabled   = false
  shared_access_key_enabled       = false
  allow_nested_items_to_be_public = false

  network_rules {
    default_action = "Deny"
    bypass         = ["AzureServices"]
  }

  tags = var.tags
}

/*
  Containers.

  The provider must be configured with `storage_use_azuread = true` (see
  providers.tf) or these fail: with shared keys disabled there is no key for
  the data-plane call that creates a container.

  They also depend on the blob private endpoint - without it the control machine
  running Terraform has no route to the data plane at all.
*/

locals {
  containers = ["uploads", "work", "results", "deploy"]
}

resource "azurerm_storage_container" "main" {
  for_each              = toset(local.containers)
  name                  = each.key
  storage_account_id    = azurerm_storage_account.main.id
  container_access_type = "private"

  depends_on = [azurerm_private_endpoint.storage]
}
