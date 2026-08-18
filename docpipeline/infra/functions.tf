/*
  Two Azure Functions apps on Flex Consumption.

  API app:
    - public UI/upload/start/status HTTP functions only
    - its own FC1 plan, host storage, and Application Insights
    - a client-only identity connection to the worker's task hub

  Worker app:
    - orchestrator and activity functions only
    - sole consumer/owner of the Durable task hub
    - private storage and ACA access through VNet integration

  Flex allows one app per FC1 plan, so the split requires two plans. Both apps
  can share the delegated /24 integration subnet.
*/

# Preserve the existing app, plan, and telemetry resource as the API tier when
# upgrading an already-applied single-app deployment.
moved {
  from = azurerm_service_plan.flex
  to   = azurerm_service_plan.api
}

moved {
  from = azurerm_application_insights.main
  to   = azurerm_application_insights.api
}

moved {
  from = azurerm_function_app_flex_consumption.main
  to   = azurerm_function_app_flex_consumption.api
}

resource "azurerm_service_plan" "api" {
  name                = "asp-${var.name_prefix}-${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  os_type             = "Linux"
  sku_name            = "FC1"
  tags                = merge(var.tags, { tier = "api" })
}

resource "azurerm_service_plan" "worker" {
  name                = "asp-${var.name_prefix}-worker-${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  os_type             = "Linux"
  sku_name            = "FC1"
  tags                = merge(var.tags, { tier = "worker" })
}

resource "azurerm_application_insights" "api" {
  name                = "appi-${var.name_prefix}-${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  application_type    = "web"
  tags                = merge(var.tags, { tier = "api" })
}

resource "azurerm_application_insights" "worker" {
  name                = "appi-${var.name_prefix}-worker-${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  application_type    = "web"
  tags                = merge(var.tags, { tier = "worker" })
}

resource "azurerm_function_app_flex_consumption" "api" {
  name                = local.api_function_name
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  service_plan_id     = azurerm_service_plan.api.id

  webdeploy_publish_basic_authentication_enabled = false

  storage_container_type      = "blobContainer"
  storage_container_endpoint  = "${azurerm_storage_account.api.primary_blob_endpoint}${azurerm_storage_container.api_deploy.name}"
  storage_authentication_type = "SystemAssignedIdentity"

  runtime_name    = "python"
  runtime_version = "3.11"

  instance_memory_in_mb  = var.api_function_instance_memory_mb
  maximum_instance_count = var.api_function_max_instances

  # Outbound integration so the app can reach the storage private endpoints.
  virtual_network_subnet_id = azurerm_subnet.func.id

  identity {
    type = "SystemAssigned"
  }

  site_config {
    application_insights_connection_string = azurerm_application_insights.api.connection_string

    cors {
      allowed_origins = ["*"]
    }
  }

  app_settings = {
    # This app's own host/deployment storage.
    "AzureWebJobsStorage__accountName"     = azurerm_storage_account.api.name
    "AzureWebJobsStorage__credential"      = "managedidentity"
    "AzureWebJobsStorage"                  = ""
    "DEPLOYMENT_STORAGE_CONNECTION_STRING" = ""

    # Cross-app Durable client connection. This host can start/query the task
    # hub but indexes no orchestrator or activity triggers.
    "WorkerStorage__accountName"                              = azurerm_storage_account.main.name
    "WorkerStorage__credential"                               = "managedidentity"
    "DURABLE_CLIENT_CONNECTION"                               = "WorkerStorage"
    "DURABLE_TASK_HUB"                                        = var.durable_task_hub_name
    "AzureFunctionsJobHost__extensions__durableTask__hubName" = var.durable_task_hub_name
    "FUNCTION_ROLE"                                           = "api"

    # Application blobs remain in worker/data storage.
    "STORAGE_ACCOUNT_NAME" = azurerm_storage_account.main.name
    "UPLOAD_CONTAINER"     = "uploads"
    "WORK_CONTAINER"       = "work"
    "RESULT_CONTAINER"     = "results"
  }

  tags = merge(var.tags, { tier = "api" })

  depends_on = [
    azurerm_private_endpoint.api_storage,
    azurerm_private_endpoint.storage,
    azurerm_storage_container.api_deploy,
    azurerm_storage_container.main,
  ]

  lifecycle {
    ignore_changes = [
      app_settings["AzureWebJobsStorage"],
      app_settings["DEPLOYMENT_STORAGE_CONNECTION_STRING"],
    ]
  }
}

resource "azurerm_function_app_flex_consumption" "worker" {
  name                = local.worker_function_name
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  service_plan_id     = azurerm_service_plan.worker.id

  webdeploy_publish_basic_authentication_enabled = false

  storage_container_type      = "blobContainer"
  storage_container_endpoint  = "${azurerm_storage_account.main.primary_blob_endpoint}${azurerm_storage_container.main["deploy"].name}"
  storage_authentication_type = "SystemAssignedIdentity"

  runtime_name    = "python"
  runtime_version = "3.11"

  instance_memory_in_mb     = var.function_instance_memory_mb
  maximum_instance_count    = var.function_max_instances
  virtual_network_subnet_id = azurerm_subnet.func.id

  always_ready {
    name           = "durable"
    instance_count = 1
  }

  identity {
    type = "SystemAssigned"
  }

  site_config {
    application_insights_connection_string = azurerm_application_insights.worker.connection_string
  }

  app_settings = {
    "AzureWebJobsStorage__accountName"     = azurerm_storage_account.main.name
    "AzureWebJobsStorage__credential"      = "managedidentity"
    "AzureWebJobsStorage"                  = ""
    "DEPLOYMENT_STORAGE_CONNECTION_STRING" = ""

    "WorkerStorage__accountName"                              = azurerm_storage_account.main.name
    "WorkerStorage__credential"                               = "managedidentity"
    "DURABLE_CLIENT_CONNECTION"                               = "WorkerStorage"
    "DURABLE_TASK_HUB"                                        = var.durable_task_hub_name
    "AzureFunctionsJobHost__extensions__durableTask__hubName" = var.durable_task_hub_name
    "FUNCTION_ROLE"                                           = "worker"

    "STORAGE_ACCOUNT_NAME" = azurerm_storage_account.main.name
    "UPLOAD_CONTAINER"     = "uploads"
    "WORK_CONTAINER"       = "work"
    "RESULT_CONTAINER"     = "results"

    "PAGE_DPI"     = "150"
    "YOLO_CHUNK"   = var.yolo_chunk
    "TEXT_SERVICE" = var.text_service

    # App-level external ingress on an internal-only ACA environment remains
    # VNet-private while accepting traffic from the Function subnet.
    "YOLO_URL"     = "http://${azurerm_container_app.yolo.ingress[0].fqdn}"
    "YOLO_TIMEOUT" = "120"

    "MISTRAL_API_KEY" = var.mistral_api_key

    "SIM_429_RATE"   = var.sim_429_rate
    "SIM_503_RATE"   = "0.05"
    "SIM_400_RATE"   = "0.01"
    "SIM_LATENCY_MS" = "250"

    "NOTIFY_URL"    = var.notify_url
    "NOTIFY_SECRET" = var.notify_secret
  }

  tags = merge(var.tags, { tier = "worker" })

  depends_on = [
    azurerm_private_endpoint.storage,
    azurerm_storage_container.main,
  ]

  lifecycle {
    ignore_changes = [
      app_settings["AzureWebJobsStorage"],
      app_settings["DEPLOYMENT_STORAGE_CONNECTION_STRING"],
    ]
  }
}
