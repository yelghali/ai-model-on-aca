variable "resource_group_name" {
  type        = string
  description = "Resource group to create or reuse."
  default     = "rg-docpipeline"
}

variable "location" {
  type        = string
  description = "Azure region. Must support BOTH Flex Consumption and Container Apps."
  default     = "swedencentral"
}

variable "name_prefix" {
  type        = string
  description = "Prefix for resource names. Keep it short: the storage account name is capped at 24 chars."
  default     = "docpipe"

  validation {
    condition     = can(regex("^[a-z0-9]{3,12}$", var.name_prefix))
    error_message = "name_prefix must be 3-12 lowercase alphanumeric characters."
  }
}

# --- sizing -----------------------------------------------------------------

variable "function_instance_memory_mb" {
  type        = number
  description = "Flex Consumption per-instance memory."
  default     = 2048

  validation {
    condition     = contains([512, 2048, 4096], var.function_instance_memory_mb)
    error_message = "Flex Consumption supports 512, 2048 or 4096 MB."
  }
}

variable "function_max_instances" {
  type        = number
  description = <<-EOT
    Flex Consumption scale ceiling.

    NOTE this multiplies with maxConcurrentActivityFunctions in host.json:
    40 instances x 8 concurrent activities = 320 simultaneous calls into Azure
    OpenAI. host.json bounds concurrency PER INSTANCE, not globally, so size
    this against the OpenAI quota rather than against YOLO.
  EOT
  default     = 40
}

variable "yolo_image" {
  type        = string
  description = "Fully qualified YOLO container image (e.g. myacr.azurecr.io/yolo-demo:v1)."
}

variable "yolo_cpu" {
  type    = number
  default = 2.0
}

variable "yolo_memory" {
  type    = string
  default = "4Gi"
}

variable "yolo_min_replicas" {
  type        = number
  description = "0 = scale to zero. Cold start is ~2.1s for PyTorch YOLO, ~0.2s for the ONNX build."
  default     = 0
}

variable "yolo_max_replicas" {
  type    = number
  default = 10
}

variable "yolo_inference_workers" {
  type        = string
  description = "Parallel inferences per replica. workers x threads must be <= yolo_cpu."
  default     = "2"
}

# --- registry ---------------------------------------------------------------

variable "container_registry_server" {
  type        = string
  description = "ACR login server hosting the YOLO image."
}

variable "container_registry_identity_id" {
  type        = string
  description = <<-EOT
    Optional user-assigned identity resource id with AcrPull on the registry.
    Leave empty to use the container app's system-assigned identity (the module
    grants it AcrPull).
  EOT
  default     = ""
}

variable "acr_resource_group" {
  type        = string
  description = "Resource group of the ACR, when it lives outside this deployment."
  default     = ""
}

# --- application settings ---------------------------------------------------

variable "text_service" {
  type        = string
  description = "content-understanding | mistral-docai"
  default     = "content-understanding"

  validation {
    condition     = contains(["content-understanding", "mistral-docai"], var.text_service)
    error_message = "text_service must be content-understanding or mistral-docai."
  }
}

variable "yolo_chunk" {
  type        = string
  description = "Figures per YOLO call. 8 matched the measured sweet spot."
  default     = "8"
}

variable "notify_url" {
  type        = string
  description = "Optional webhook called when a batch completes."
  default     = ""
}

variable "notify_secret" {
  type        = string
  description = "HMAC-SHA256 signing secret for callbacks. Empty disables signing."
  default     = ""
  sensitive   = true
}

variable "mistral_api_key" {
  type        = string
  description = "Mistral Document AI key. Empty = the module simulates the service."
  default     = ""
  sensitive   = true
}

variable "sim_429_rate" {
  type        = string
  description = "Injected throttling rate, so the retry policy is exercised in demos."
  default     = "0.15"
}

variable "tags" {
  type = map(string)
  default = {
    workload = "docpipeline"
    demo     = "true"
  }
}
