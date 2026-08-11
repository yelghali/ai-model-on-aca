/*
  Durable Functions on Flex Consumption.

  Two things make this different from a normal function app:

  1. Deployment package storage is identity-based.
     Flex stores the zip in a blob container. With shared keys disabled that
     container must be reached with the app's managed identity, hence
     storage_authentication_type = "SystemAssignedIdentity".

  2. AzureWebJobsStorage is identity-based too.
     The `__accountName` form replaces the connection string. The runtime then
     uses the managed identity for the blobs, queues and tables Durable needs -
     which is why rbac.tf grants all three data roles, not just blob.
*/

resource "azurerm_service_plan" "flex" {
  name                = "asp-${var.name_prefix}-${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  os_type             = "Linux"
  sku_name            = "FC1" # Flex Consumption
  tags                = var.tags
}

resource "azurerm_application_insights" "main" {
  name                = "appi-${var.name_prefix}-${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  application_type    = "web"
  tags                = var.tags
}

resource "azurerm_function_app_flex_consumption" "main" {
  name                = local.function_name
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  service_plan_id     = azurerm_service_plan.flex.id

  storage_container_type      = "blobContainer"
  storage_container_endpoint  = "${azurerm_storage_account.main.primary_blob_endpoint}${azurerm_storage_container.main["deploy"].name}"
  storage_authentication_type = "SystemAssignedIdentity"

  runtime_name    = "python"
  runtime_version = "3.11"

  instance_memory_in_mb  = var.function_instance_memory_mb
  maximum_instance_count = var.function_max_instances

  # Outbound integration so the app can reach the storage private endpoints.
  virtual_network_subnet_id = azurerm_subnet.func.id

  identity {
    type = "SystemAssigned"
  }

  site_config {
    application_insights_connection_string = azurerm_application_insights.main.connection_string

    # The UI and upload API are meant to be publicly reachable; storage is not.
    cors {
      allowed_origins = ["*"]
    }
  }

  app_settings = {
    # Identity-based host storage. NOT a connection string - shared keys are off.
    #
    # This exact pair is what works. Two variants were tried and both broke it:
    #
    #   __blobServiceUri/__queueServiceUri/__tableServiceUri instead of
    #   __accountName - accepted by the host, but see the note in the README
    #   about the blob trigger; it does not buy anything here.
    #
    #   __serviceUri - DO NOT ADD. A single generic URI cannot serve blob,
    #   queue and table at once, so Durable loses its queues and the host stops
    #   starting entirely: the site returns 503 "Site Unavailable" while ARM
    #   still reports state=Running, availabilityState=Normal.
    "AzureWebJobsStorage__accountName" = azurerm_storage_account.main.name
    "AzureWebJobsStorage__credential"  = "managedidentity"

    # The application code reads this directly (shared/storage.py).
    "STORAGE_ACCOUNT_NAME" = azurerm_storage_account.main.name

    # The provider ALWAYS emits a connection-string AzureWebJobsStorage derived
    # from the storage account. With shared keys disabled there is no key to
    # put in it, so it is emitted as "...;AccountKey=;..." - a syntactically
    # valid, functionally dead credential. The host prefers it over the
    # identity settings above, fails to authenticate, and never starts: every
    # request returns 503 with no telemetry, because the host is not up to
    # report anything.
    #
    # Overriding both to empty makes the host fall through to the
    # identity-based settings. Do not remove these two lines.
    "AzureWebJobsStorage"                  = ""
    "DEPLOYMENT_STORAGE_CONNECTION_STRING" = ""

    "UPLOAD_CONTAINER" = "uploads"
    "WORK_CONTAINER"   = "work"
    "RESULT_CONTAINER" = "results"

    "PAGE_DPI"     = "150"
    "YOLO_CHUNK"   = var.yolo_chunk
    "TEXT_SERVICE" = var.text_service

    # Internal ingress: resolvable only from inside the VNet. http, not https -
    # see the ingress block in aca.tf for why.
    "YOLO_URL"     = "http://${azurerm_container_app.yolo.ingress[0].fqdn}"
    "YOLO_TIMEOUT" = "120"

    "MISTRAL_API_KEY" = var.mistral_api_key

    # Injected faults so the retry policy is demonstrable, not just asserted.
    "SIM_429_RATE"   = var.sim_429_rate
    "SIM_503_RATE"   = "0.05"
    "SIM_400_RATE"   = "0.01"
    "SIM_LATENCY_MS" = "250"

    "NOTIFY_URL"    = var.notify_url
    "NOTIFY_SECRET" = var.notify_secret
  }

  tags = var.tags

  depends_on = [
    azurerm_private_endpoint.storage,
    azurerm_storage_container.main,
  ]
}
