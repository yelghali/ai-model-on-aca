output "function_app_name" {
  description = "Deploy the code with: az functionapp deployment source config-zip -g <rg> -n <this> --src app.zip"
  value       = azurerm_function_app_flex_consumption.main.name
}

output "ui_url" {
  description = "Public demo UI. The upload API is the only public entry point."
  value       = "https://${azurerm_function_app_flex_consumption.main.default_hostname}/api/ui"
}

output "upload_api" {
  value = "https://${azurerm_function_app_flex_consumption.main.default_hostname}/api/upload"
}

output "storage_account_name" {
  value = azurerm_storage_account.main.name
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
    Storage is private-only and key-less by policy:
      public_network_access_enabled = false
      shared_access_key_enabled     = false

    So the browser has NO path to blob and cannot be given a SAS. Uploads go
    through the function's /api/upload, which is VNet-integrated and writes with
    its managed identity. That is also why YOLO can stay private: only the
    function needs to reach it.

    After `terraform apply`, deploy the application code:
      cd .. ; Compress-Archive -Path * -DestinationPath app.zip -Force
      az functionapp deployment source config-zip -g <rg> -n <function_app_name> --src app.zip
  EOT
}
