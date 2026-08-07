"""Blob helpers - managed identity, not connection strings.

Why identity is not optional here
----------------------------------
Storage in this subscription is governed by policy:

    publicNetworkAccess : Disabled
    allowSharedKeyAccess: false      <-- connection strings CANNOT authenticate

so `from_connection_string(...)` fails outright. Auth goes through
`DefaultAzureCredential`, which resolves to the Function App's system-assigned
managed identity in Azure and to `az login` locally.

Reaching the account is a SEPARATE problem from authenticating to it: with public
access disabled the data plane is only reachable through a private endpoint,
which is why the Function App is VNet-integrated - and why the BROWSER never
talks to storage directly. It uploads through the function, which is the only
component with a network path.

THE RULE, unchanged: orchestration history stores REFERENCES, never bytes.
Durable serialises every activity input/output and replays it on each
checkpoint, so page images move by blob path, not by value.

Layout:
  uploads/<file>.pdf                          <- app uploads here (via the API)
  work/<run_id>/<file>/pages/page-0001.png    <- rasterised pages
  work/<run_id>/<file>/figures/p0007-f00.png  <- figures extracted from pages
  results/<file>.json                         <- one merged result per file
"""
from __future__ import annotations

import os

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings

UPLOAD_CONTAINER = os.getenv("UPLOAD_CONTAINER", "uploads")
WORK_CONTAINER = os.getenv("WORK_CONTAINER", "work")
RESULT_CONTAINER = os.getenv("RESULT_CONTAINER", "results")

# The Functions host uses the identity-based form of AzureWebJobsStorage, which
# supplies the ACCOUNT NAME rather than a connection string.
ACCOUNT = (
    os.getenv("STORAGE_ACCOUNT_NAME")
    or os.getenv("AzureWebJobsStorage__accountName")
    or ""
)

_client: BlobServiceClient | None = None


def _service() -> BlobServiceClient:
    global _client
    if _client is None:
        if not ACCOUNT:
            raise RuntimeError(
                "STORAGE_ACCOUNT_NAME / AzureWebJobsStorage__accountName is not set")
        # Build the credential once and cache it with the client:
        # DefaultAzureCredential probes several sources on construction, which
        # would be wasted work on every activity invocation.
        _client = BlobServiceClient(
            f"https://{ACCOUNT}.blob.core.windows.net",
            credential=DefaultAzureCredential(
                exclude_interactive_browser_credential=True),
        )
    return _client


def ensure_containers() -> None:
    for name in (UPLOAD_CONTAINER, WORK_CONTAINER, RESULT_CONTAINER):
        try:
            _service().create_container(name)
        except Exception:
            pass  # already exists


def read_blob(container: str, path: str) -> bytes:
    return _service().get_blob_client(container, path).download_blob().readall()


def write_blob(container: str, path: str, data: bytes,
               content_type: str | None = None,
               metadata: dict | None = None) -> str:
    settings = ContentSettings(content_type=content_type) if content_type else None
    _service().get_blob_client(container, path).upload_blob(
        data, overwrite=True, content_settings=settings, metadata=metadata or None
    )
    return path


def blob_exists(container: str, path: str) -> bool:
    return _service().get_blob_client(container, path).exists()


def blob_properties(container: str, path: str) -> dict:
    """Size + metadata, used to resolve a run from a file name.

    Metadata keys come back lower-cased from the service, which is why the
    caller looks for `correlationid` rather than `correlationId`.
    """
    props = _service().get_blob_client(container, path).get_blob_properties()
    return {"size": props.size, "metadata": dict(props.metadata or {})}


def list_results(prefix: str = "") -> list[dict]:
    """Recent result documents, newest first - used by the demo UI."""
    client = _service().get_container_client(RESULT_CONTAINER)
    items = [
        {"name": b.name, "size": b.size,
         "modified": b.last_modified.isoformat() if b.last_modified else None}
        for b in client.list_blobs(name_starts_with=prefix)
    ]
    return sorted(items, key=lambda x: x["modified"] or "", reverse=True)
