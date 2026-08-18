output "resource_group_name" {
  value = azurerm_resource_group.main.name
}

output "api_function_app_name" {
  description = "Public UI/upload/start/status Function App."
  value       = azurerm_function_app_flex_consumption.api.name
}

output "worker_function_app_name" {
  description = "Durable orchestrator/activity Function App; sole task-hub worker."
  value       = azurerm_function_app_flex_consumption.worker.name
}

# Compatibility for scripts that previously expected the single public app.
output "function_app_name" {
  description = "Deprecated alias for api_function_app_name."
  value       = azurerm_function_app_flex_consumption.api.name
}

output "ui_url" {
  description = "Public demo UI."
  value       = "https://${azurerm_function_app_flex_consumption.api.default_hostname}/api/ui"
}

output "upload_api" {
  value = "https://${azurerm_function_app_flex_consumption.api.default_hostname}/api/upload"
}

output "api_host_storage_account_name" {
  value = azurerm_storage_account.api.name
}

output "worker_storage_account_name" {
  value = azurerm_storage_account.main.name
}

output "storage_account_name" {
  description = "Compatibility alias for the worker/data storage account."
  value       = azurerm_storage_account.main.name
}

output "durable_task_hub_name" {
  value = var.durable_task_hub_name
}

output "yolo_internal_fqdn" {
  description = "PRIVATE: resolves only inside the VNet."
  value       = azurerm_container_app.yolo.ingress[0].fqdn
}

output "aca_environment_static_ip" {
  value = azurerm_container_app_environment.main.static_ip_address
}

output "notes" {
  value = <<-EOT
    The public API app and Durable worker app use separate FC1 plans, host
    storage accounts, identities, and Application Insights resources.

    The API targets task hub '${var.durable_task_hub_name}' through the
    identity-based WorkerStorage connection. Only the worker app indexes and
    consumes orchestrator/activity triggers from that hub.

    Deploy the same role-gated package independently to both apps:
      cd ..
      .\deploy.ps1 -Target both

    During a one-time migration from the old combined app, deploy API first and
    worker second. Durable messages safely wait in storage during the gap; this
    order prevents two worker hosts from consuming the same task hub.
  EOT
}
