/*
  RBAC.

  With `shared_access_key_enabled = false` these role assignments are the ONLY
  way anything reaches storage. Miss one and the failure is obscure: the host
  starts, the HTTP endpoints answer, and orchestrations simply never progress.

  Durable Functions needs all three data planes:
    Blob   - lease blobs, large history payloads, the deployment package
    Queue  - work-item and control queues
    Table  - instance and history tables

  Data-plane roles only. Nothing here grants control-plane rights, and no role
  grants key access (there are no keys).
*/

locals {
  function_principal_id = azurerm_function_app_flex_consumption.main.identity[0].principal_id

  storage_data_roles = [
    "Storage Blob Data Contributor",
    "Storage Queue Data Contributor",
    "Storage Table Data Contributor",
  ]
}

resource "azurerm_role_assignment" "function_storage" {
  for_each             = toset(local.storage_data_roles)
  scope                = azurerm_storage_account.main.id
  role_definition_name = each.key
  principal_id         = local.function_principal_id
}

/*
  Let the container app pull its image with its own identity when no
  user-assigned identity was supplied. Scoped to the registry, not the
  subscription.
*/
data "azurerm_container_registry" "acr" {
  count               = var.container_registry_identity_id == "" ? 1 : 0
  name                = split(".", var.container_registry_server)[0]
  resource_group_name = var.acr_resource_group != "" ? var.acr_resource_group : azurerm_resource_group.main.name
}

resource "azurerm_role_assignment" "yolo_acr_pull" {
  count                = var.container_registry_identity_id == "" ? 1 : 0
  scope                = data.azurerm_container_registry.acr[0].id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_container_app.yolo.identity[0].principal_id
}
