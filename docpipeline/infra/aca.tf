/*
  YOLO on Container Apps - PRIVATE.

  internal_load_balancer_enabled = true gives the environment an internal load
  balancer, so the ingress FQDN resolves to a VNet address and the API is not
  reachable from the internet. Only the Function App (VNet-integrated) can call
  it, which is the point: the public surface is the upload API, not the model.

  Because the FQDN is private, a private DNS zone for the environment's default
  domain is REQUIRED. Without it the Function App resolves the name against
  public DNS, gets nothing, and every YOLO call fails with a DNS error that
  looks nothing like a networking problem.

  KEDA HTTP scaling still applies. `min_replicas = 0` scales to zero; the
  measured cold start is ~2.1s (PyTorch) or ~0.2s (ONNX), and 503/507 during
  scale-out is handled by the SCALE_SENSITIVE retry policy in the pipeline.
*/

resource "azurerm_log_analytics_workspace" "main" {
  name                = "log-${var.name_prefix}-${local.suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "PerGB2018"
  retention_in_days   = 30
  tags                = var.tags
}

resource "azurerm_container_app_environment" "main" {
  name                = local.aca_env_name
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  # azurerm 5.x requires logs_destination to be stated before it will accept a
  # workspace id; omitting it is an error, not a default.
  logs_destination           = "log-analytics"
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id

  infrastructure_subnet_id       = azurerm_subnet.aca.id
  internal_load_balancer_enabled = true # <- private

  workload_profile {
    name                  = "Consumption"
    workload_profile_type = "Consumption"
  }

  tags = var.tags
}

# Resolve the environment's private domain from inside the VNet.
resource "azurerm_private_dns_zone" "aca" {
  name                = azurerm_container_app_environment.main.default_domain
  resource_group_name = azurerm_resource_group.main.name
  tags                = var.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "aca" {
  name                 = "link-aca"
  private_dns_zone_id  = azurerm_private_dns_zone.aca.id
  virtual_network_id   = azurerm_virtual_network.main.id
  registration_enabled = false
  tags                 = var.tags
}

# Wildcard A record -> the environment's internal load balancer IP.
resource "azurerm_private_dns_a_record" "aca" {
  name                = "*"
  private_dns_zone_id = azurerm_private_dns_zone.aca.id
  ttl                 = 300
  records             = [azurerm_container_app_environment.main.static_ip_address]
}

resource "azurerm_container_app" "yolo" {
  name                         = "yolo-api"
  resource_group_name          = azurerm_resource_group.main.name
  container_app_environment_id = azurerm_container_app_environment.main.id
  revision_mode                = "Single"
  workload_profile_name        = "Consumption"
  tags                         = var.tags

  identity {
    type         = "UserAssigned"
    identity_ids = [local.acr_identity_id]
  }

  # Always present: without a registry block ACA attempts an ANONYMOUS pull and
  # fails with "UNAUTHORIZED: authentication required", which reads like a
  # credential problem rather than a missing configuration block.
  registry {
    server   = var.container_registry_server
    identity = local.acr_identity_id
  }

  template {
    min_replicas = var.yolo_min_replicas
    max_replicas = var.yolo_max_replicas

    container {
      name   = "yolo"
      image  = var.yolo_image
      cpu    = var.yolo_cpu
      memory = var.yolo_memory

      # Parallel inferences INSIDE one replica. os.cpu_count() in a container
      # reports HOST cores, not the ACA limit, so this is always explicit.
      env {
        name  = "INFERENCE_WORKERS"
        value = var.yolo_inference_workers
      }

      # workers x threads must stay <= yolo_cpu or the workers fight for cores
      # and every request gets slower.
      env {
        name  = "TORCH_THREADS_PER_WORKER"
        value = "1"
      }

      liveness_probe {
        transport = "HTTP"
        port      = 8000
        path      = "/healthz"
      }

      readiness_probe {
        transport = "HTTP"
        port      = 8000
        path      = "/healthz"
      }
    }

    # Match the scale trigger to the pool size: a replica only receives more
    # requests than it can run in parallel once another replica is coming up.
    http_scale_rule {
      name                = "http-concurrency"
      concurrent_requests = var.yolo_inference_workers
    }
  }

  ingress {
    # "External" is relative to the ACA environment. This must be true for a
    # caller in another VNet subnet (the Function App) to enter through the
    # environment load balancer. The environment itself has an INTERNAL load
    # balancer, so this remains private and has no internet-facing endpoint.
    external_enabled = true
    target_port      = 8000
    transport        = "auto"

    # The internal ingress does not complete a TLS handshake on the default
    # domain - callers get SSLEOFError("EOF occurred in violation of protocol").
    # Serving port 80 directly (instead of redirecting to 443) makes the hop
    # work.
    #
    # TRADE-OFF, stated plainly: this one hop is unencrypted. It is acceptable
    # here only because it never leaves the VNet - the app has
    # external ingress inside an internal-only ACA environment, so it has no
    # internet-facing listener. Put a custom domain and certificate on the
    # environment if the hop must be encrypted.
    allow_insecure_connections = true

    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }

  # The grant must be visible before the platform attempts the pull.
  depends_on = [time_sleep.acr_rbac_propagation]
}
