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
  Image pull.

  A system-assigned identity CANNOT be used here. It does not exist until the
  container app is created, but the app cannot be created until something with
  AcrPull can fetch the image - the first apply fails with
  "UNAUTHORIZED: authentication required".

  A user-assigned identity breaks the cycle: it is created first, granted
  AcrPull, and only then referenced by the app's registry block.
*/
data "azurerm_container_registry" "acr" {
  count               = var.container_registry_identity_id == "" ? 1 : 0
  name                = split(".", var.container_registry_server)[0]
  resource_group_name = var.acr_resource_group != "" ? var.acr_resource_group : azurerm_resource_group.main.name
}

resource "azurerm_user_assigned_identity" "acr_pull" {
  count               = var.container_registry_identity_id == "" ? 1 : 0
  name                = "id-acrpull-${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  tags                = var.tags
}

resource "azurerm_role_assignment" "yolo_acr_pull" {
  count                = var.container_registry_identity_id == "" ? 1 : 0
  scope                = data.azurerm_container_registry.acr[0].id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.acr_pull[0].principal_id
}

# Role assignments are eventually consistent. Without this pause the container
# app is often created before the grant is visible to the pull, which fails in
# exactly the same way as having no grant at all.
resource "time_sleep" "acr_rbac_propagation" {
  count           = var.container_registry_identity_id == "" ? 1 : 0
  depends_on      = [azurerm_role_assignment.yolo_acr_pull]
  create_duration = "60s"
}

locals {
  # Whichever identity ends up doing the pull.
  acr_identity_id = (
    var.container_registry_identity_id != ""
    ? var.container_registry_identity_id
    : azurerm_user_assigned_identity.acr_pull[0].id
  )
}
